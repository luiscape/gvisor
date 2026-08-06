#!/usr/bin/env bash
# --------------------------------------------------------------------------
#  bench_7_nccl.sh — minimal NCCL multi-GPU snapshot reproduction.
#
#  Boots apps/nccl_repro.py (CUDA-free parent + one NCCL rank per GPU),
#  snapshots it with gVisor's native cuda-checkpoint integration, restores,
#  and verifies (a) GPU memory integrity of a tensor allreduced before the
#  snapshot and (b) that a fresh deterministic allreduce still completes
#  and produces the same value. Unlike the other benches, checkpoint /
#  restore / verify failures are RESULTS here, not harness errors: the
#  script always reaches the summary and exits 0=PASS, 2=REPRO, 1=harness
#  error.
#
#  Variables that select the failure mode under test:
#    --nccl-env "K=V;K=V"  e.g. NCCL_P2P_DISABLE=1 (force SHM transport),
#                          NCCL_SHM_DISABLE=1 (force P2P/CUDA-IPC),
#                          NCCL_CUMEM_ENABLE=0/1 (legacy IPC vs cuMem FDs)
#    --active              keep allreduces in flight during the snapshot
#    --mode p2p1proc       single process, all GPUs, no NCCL/IPC (control)
#    --mode ncclraw        drive libnccl directly via ctypes (raw ncclComm_t)
#    --suspend             ncclCommSuspend(NCCL_SUSPEND_MEM) before the
#                          snapshot and ncclCommResume after restore
#                          (implies --mode ncclraw; needs NCCL >= 2.30)
#    --lifecycle           ncclCommDestroy before the snapshot and full
#                          communicator re-init after restore (implies
#                          --mode ncclraw). Lets cuMem/symm-mem stay
#                          enabled: no NCCL allocations exist at toggle
#                          time.
#    --native              no gVisor: docker/runc + direct
#                          `cuda-checkpoint --toggle` suspend/resume cycle
#                          (bisects driver-side vs nvproxy-side failures)
#
#  Examples:
#    sudo bash cr-bench/bench_7_nccl.sh                        # default NCCL
#    sudo bash cr-bench/bench_7_nccl.sh --active
#    sudo bash cr-bench/bench_7_nccl.sh --nccl-env "NCCL_SHM_DISABLE=1"
#    sudo bash cr-bench/bench_7_nccl.sh --native --active
# --------------------------------------------------------------------------
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

BENCH_NAME="cr-bench-nccl"
IMAGE="${IMAGE:-cr-bench-gpu}"
DOCKERFILE="images/Dockerfile.gpu"
PORT="${PORT:-8199}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
TENSOR_MB="${TENSOR_MB:-64}"
MODE="${MODE:-nccl}"
ACTIVE="${ACTIVE:-0}"
SUSPEND="${SUSPEND:-0}"
LIFECYCLE="${LIFECYCLE:-0}"
NCCL_ENV="${NCCL_ENV:-}"          # semicolon-separated KEY=VALUE
NATIVE="${NATIVE:-0}"
VERIFY_TIMEOUT="${VERIFY_TIMEOUT:-30}"
CB_GPU=1
APP_LOG="/applog/app.log"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)         GPU_DEVICES="$2"; shift 2 ;;
        --restore-gpus) RESTORE_GPU_DEVICES="$2"; shift 2 ;;
        --tensor-mb)    TENSOR_MB="$2"; shift 2 ;;
        --mode)         MODE="$2"; shift 2 ;;
        --active)       ACTIVE=1; shift ;;
        --suspend)      SUSPEND=1; MODE=ncclraw; shift ;;
        --lifecycle)    LIFECYCLE=1; MODE=ncclraw; shift ;;
        --nccl-env)     NCCL_ENV="$2"; shift 2 ;;
        --native)       NATIVE=1; shift ;;
        --image)        IMAGE="$2"; shift 2 ;;
        --rebuild-rootfs) REBUILD_ROOTFS=1; shift ;;
        --help|-h)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

NUM_GPUS=$(echo "$GPU_DEVICES" | tr ',' '\n' | grep -c .)
WORLD_SIZE=$NUM_GPUS
[[ "$MODE" == "p2p1proc" ]] && WORLD_SIZE=$NUM_GPUS  # informational only

# Result tracking: every stage records into these instead of exiting.
REPRO_STAGE=""
REPRO_DETAIL=""
mark_repro() {  # stage detail...
    if [[ -z "$REPRO_STAGE" ]]; then
        REPRO_STAGE="$1"; shift
        REPRO_DETAIL="$*"
    fi
    fail "REPRO at stage '$REPRO_STAGE': $REPRO_DETAIL"
}

# ── native (no gVisor) control: docker + direct cuda-checkpoint toggle ────
run_native() {
    banner "╔══════════════════════════════════════════════════════════════════╗"
    banner "║   Bench 7 NCCL repro — NATIVE control (docker, no gVisor)       ║"
    banner "╚══════════════════════════════════════════════════════════════════╝"
    local name="cr-bench-nccl-native-$$"
    local envflags=(-e WORLD_SIZE="$NUM_GPUS" -e MODE="$MODE" -e ACTIVE="$ACTIVE"
                    -e TENSOR_MB="$TENSOR_MB" -e PORT="$PORT"
                    -e VERIFY_TIMEOUT="$VERIFY_TIMEOUT" -e NCCL_DEBUG=INFO)
    local IFS=';'
    for kv in $NCCL_ENV; do [[ -n "$kv" ]] && envflags+=(-e "$kv"); done
    unset IFS

    docker rm -f "$name" >/dev/null 2>&1 || true
    docker run -d --name "$name" --gpus "\"device=${GPU_DEVICES}\"" \
        --shm-size="${CB_SHM_MB:-4096}m" \
        --entrypoint python3 "${envflags[@]}" "$IMAGE" /app/nccl_repro.py >/dev/null
    trap 'docker logs --tail 40 "$name" 2>&1 | sed "s/^/    | /"; docker rm -f "$name" >/dev/null 2>&1' EXIT

    info "waiting for /health"
    local i
    for i in $(seq 1 120); do
        docker exec "$name" curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
        sleep 1
        if [[ "$(docker inspect -f '{{.State.Running}}' "$name")" != "true" ]]; then
            fail "container died during startup"; exit 1
        fi
    done
    docker exec "$name" curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { fail "health timeout"; exit 1; }
    ok "healthy"

    local ref info pids
    info=$(docker exec "$name" curl -sf "http://127.0.0.1:$PORT/info")
    pids=$(echo "$info" | python3 -c 'import json,sys; print(" ".join(str(p) for p in json.load(sys.stdin)["worker_pids"].values()))')
    ok "worker pids (container ns): $pids"
    ref=$(docker exec "$name" curl -sf --max-time 60 "http://127.0.0.1:$PORT/verify")
    echo "    ref: $ref"

    if [[ "$SUSPEND" = "1" ]]; then
        info "ncclCommSuspend(NCCL_SUSPEND_MEM) on all ranks"
        local sus
        sus=$(docker exec "$name" curl -sf --max-time 60 "http://127.0.0.1:$PORT/suspend" || echo "")
        echo "    suspend: $sus"
        if [[ "$(echo "$sus" | python3 -c 'import json,sys; print(json.load(sys.stdin)["complete"])' 2>/dev/null)" != "True" ]]; then
            mark_repro "native-nccl-suspend-api" "ncclCommSuspend failed: $sus"
        else
            ok "NCCL suspended; GPU mem now: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$GPU_DEVICES" | tr '\n' ' ')"
        fi
    fi
    if [[ "$LIFECYCLE" = "1" ]]; then
        info "ncclCommDestroy on all ranks (pre-toggle)"
        local td
        td=$(docker exec "$name" curl -sf --max-time 60 "http://127.0.0.1:$PORT/teardown" || echo "")
        echo "    teardown: $td"
        if [[ "$(echo "$td" | python3 -c 'import json,sys; print(json.load(sys.stdin)["complete"])' 2>/dev/null)" != "True" ]]; then
            mark_repro "native-nccl-teardown" "ncclCommDestroy failed: $td"
        else
            ok "NCCL comms destroyed; GPU mem now: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$GPU_DEVICES" | tr '\n' ' ')"
        fi
    fi

    # Toggle all ranks IN PARALLEL: ranks coupled through NCCL (SHM/IPC
    # transports, peers blocked in a collective) cannot be suspended one at
    # a time — a sequential toggle deadlocks until the driver times out.
    # This mirrors what the gVisor sentry does.
    toggle_all() {  # stage-name
        local stage="$1" p rc pjobs=() pj
        local t0=$(ts_ms)
        for p in $pids; do
            docker exec "$name" cuda-checkpoint --toggle --pid "$p" \
                > "/tmp/nccl-toggle-$p.log" 2>&1 &
            pjobs+=("$!:$p")
        done
        for pj in "${pjobs[@]}"; do
            rc=0; wait "${pj%%:*}" || rc=$?
            sed 's/^/    | /' "/tmp/nccl-toggle-${pj##*:}.log"
            if [[ "$rc" -ne 0 ]]; then
                mark_repro "$stage" "cuda-checkpoint --toggle --pid ${pj##*:} failed"
            fi
        done
        ok "$stage done ($(( $(ts_ms) - t0 )) ms); GPU mem now: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$GPU_DEVICES" | tr '\n' ' ')"
    }

    info "cuda-checkpoint --toggle (suspend, parallel) on ranks: $pids"
    toggle_all "native-suspend"

    info "cuda-checkpoint --toggle (resume, parallel) on ranks: $pids"
    toggle_all "native-resume"

    if [[ "$SUSPEND" = "1" ]]; then
        info "ncclCommResume on all ranks"
        local res
        res=$(docker exec "$name" curl -sf --max-time 60 "http://127.0.0.1:$PORT/resume" || echo "")
        echo "    resume: $res"
        if [[ "$(echo "$res" | python3 -c 'import json,sys; print(json.load(sys.stdin)["complete"])' 2>/dev/null)" != "True" ]]; then
            mark_repro "native-nccl-resume-api" "ncclCommResume failed: $res"
        fi
    fi
    if [[ "$LIFECYCLE" = "1" ]]; then
        info "NCCL comm re-init on all ranks (post-toggle)"
        local ri
        ri=$(docker exec "$name" curl -sf --max-time 120 "http://127.0.0.1:$PORT/reinit" || echo "")
        echo "    reinit: $ri"
        if [[ "$(echo "$ri" | python3 -c 'import json,sys; print(json.load(sys.stdin)["complete"])' 2>/dev/null)" != "True" ]]; then
            mark_repro "native-nccl-reinit" "NCCL re-init failed: $ri"
        fi
    fi

    local post
    post=$(docker exec "$name" curl -sf --max-time 60 "http://127.0.0.1:$PORT/verify" || echo "")
    echo "    post: $post"
    if [[ -z "$post" ]]; then
        mark_repro "native-verify" "verify endpoint dead after toggle cycle"
    elif [[ "$(echo "$post" | python3 -c 'import json,sys; print(json.load(sys.stdin)["complete"])')" != "True" ]]; then
        mark_repro "native-verify-hang" "ranks hung: $(echo "$post" | python3 -c 'import json,sys; print(json.load(sys.stdin)["hung_ranks"])')"
    else
        local ref_v post_v
        ref_v=$(echo "$ref"  | python3 -c 'import json,sys; d=json.load(sys.stdin)["ranks"]; print({k:(v["persisted"],v["fresh"]) for k,v in d.items()})')
        post_v=$(echo "$post" | python3 -c 'import json,sys; d=json.load(sys.stdin)["ranks"]; print({k:(v["persisted"],v["fresh"]) for k,v in d.items()})')
        if [[ "$ref_v" == "$post_v" ]]; then
            ok "persisted + fresh allreduce values match after toggle cycle"
        else
            mark_repro "native-data" "value mismatch: ref=$ref_v post=$post_v"
        fi
    fi

    echo ""
    if [[ -z "$REPRO_STAGE" ]]; then banner "RESULT: PASS ✓ (native toggle cycle)"; exit 0
    else banner "RESULT: REPRO ✗ (native, stage: $REPRO_STAGE)"; exit 2; fi
}

if [[ "$NATIVE" = "1" ]]; then
    if [[ "$(id -u)" -ne 0 ]]; then fail "Must run as root"; exit 1; fi
    cb_detect_gpu
    run_native
fi

# ── gVisor path ───────────────────────────────────────────────────────────
cb_init
cb_detect_gpu
cb_runsc_flags

banner ""
banner "╔══════════════════════════════════════════════════════════════════╗"
banner "║   Benchmark 7: NCCL multi-GPU snapshot repro (gVisor)           ║"
banner "╚══════════════════════════════════════════════════════════════════╝"
echo ""
info "GPU:        $GPU_NAME, driver $HOST_DRIVER_VER"
info "GPUs:       $GPU_DEVICES ($NUM_GPUS) | mode=$MODE active=$ACTIVE"
info "NCCL env:   ${NCCL_ENV:-<default>}"
echo ""

cb_prepare_rootfs

CB_CMD="exec python3 /app/nccl_repro.py >$APP_LOG 2>&1"
CB_ENV="WORLD_SIZE=$NUM_GPUS
MODE=$MODE
ACTIVE=$ACTIVE
TENSOR_MB=$TENSOR_MB
PORT=$PORT
VERIFY_TIMEOUT=$VERIFY_TIMEOUT
NCCL_DEBUG=INFO"
if [[ -n "$NCCL_ENV" ]]; then
    IFS=';'
    for kv in $NCCL_ENV; do [[ -n "$kv" ]] && CB_ENV+=$'\n'"$kv"; done
    unset IFS
fi
cb_write_bundle

echo ""
cb_run_and_wait_health

# transport actually chosen (from NCCL_DEBUG=INFO in the app log)
TRANSPORTS=$(grep -oE 'via (P2P/[A-Za-z ]*|SHM|NET/[A-Za-z]*)' "$APPLOG_DIR/app.log" 2>/dev/null | sort | uniq -c | xargs || true)
info "NCCL transports: ${TRANSPORTS:-<none logged>}"

echo ""
info "Reference /verify (pre-checkpoint)"
REF=$(cb_curl "$CONTAINER_ID" --max-time 60 "http://127.0.0.1:$PORT/verify" || echo "")
echo "    $REF"
if [[ -z "$REF" ]] || [[ "$(echo "$REF" | python3 -c 'import json,sys; print(json.load(sys.stdin)["complete"])' 2>/dev/null)" != "True" ]]; then
    fail "reference verify incomplete — harness error"; exit 1
fi
REF_V=$(echo "$REF" | python3 -c 'import json,sys; d=json.load(sys.stdin)["ranks"]; print({k:(v["persisted"],v["fresh"]) for k,v in d.items()})')
ok "reference values: $REF_V"

if [[ "$SUSPEND" = "1" ]]; then
    echo ""
    info "ncclCommSuspend(NCCL_SUSPEND_MEM) on all ranks (pre-checkpoint)"
    SUS=$(cb_curl "$CONTAINER_ID" --max-time 60 "http://127.0.0.1:$PORT/suspend" || echo "")
    echo "    $SUS"
    if [[ "$(echo "$SUS" | python3 -c 'import json,sys; print(json.load(sys.stdin)["complete"])' 2>/dev/null)" != "True" ]]; then
        mark_repro "nccl-suspend-api" "ncclCommSuspend failed: $SUS"
    else
        ok "NCCL suspended; GPU mem in use: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$GPU_DEVICES" | tr '\n' ' ')"
    fi
fi

if [[ "$LIFECYCLE" = "1" ]]; then
    echo ""
    info "ncclCommDestroy on all ranks (pre-checkpoint)"
    TD=$(cb_curl "$CONTAINER_ID" --max-time 60 "http://127.0.0.1:$PORT/teardown" || echo "")
    echo "    $TD"
    if [[ "$(echo "$TD" | python3 -c 'import json,sys; print(json.load(sys.stdin)["complete"])' 2>/dev/null)" != "True" ]]; then
        mark_repro "nccl-teardown" "ncclCommDestroy failed: $TD"
    else
        ok "NCCL comms destroyed pre-checkpoint"
    fi
fi

# ── checkpoint (non-fatal) ──────────────────────────────────────────────────
echo ""
info "runsc checkpoint (cuda-checkpoint: $CUDA_CHECKPOINT_PATH)"
CKPT_FLAGS=(--image-path="$CKPT_DIR" --compression="$COMPRESSION"
            --cuda-checkpoint-path="$CUDA_CHECKPOINT_PATH")
[[ "$EXCLUDE_ZERO" = "1" ]] && CKPT_FLAGS+=(--exclude-committed-zero-pages)
T0=$(ts_ms); CKPT_RC=0
"$RUNSC" "${RUNSC_FLAGS[@]}" checkpoint "${CKPT_FLAGS[@]}" "$CONTAINER_ID" \
    >"$LOG_DIR/runsc-checkpoint.log" 2>&1 || CKPT_RC=$?
T_CHECKPOINT=$(( $(ts_ms) - T0 ))
if [[ "$CKPT_RC" -ne 0 ]]; then
    mark_repro "checkpoint" "runsc checkpoint exited $CKPT_RC after ${T_CHECKPOINT}ms"
    tail -15 "$LOG_DIR/runsc-checkpoint.log" | sed 's/^/    | /'
    grep -hE 'cuda-checkpoint|CUDA|Nv[A-Za-z]*(Status|Error)|toggle' "$LOG_DIR"/runsc.log.*boot.txt 2>/dev/null | tail -15 | sed 's/^/    boot| /' || true
else
    ok "checkpoint OK: ${T_CHECKPOINT} ms ($(du -sh "$CKPT_DIR" | cut -f1))"
fi

# ── restore (non-fatal) ───────────────────────────────────────────────────
POST=""
if [[ -z "$REPRO_STAGE" ]]; then
    echo ""
    info "runsc restore"
    "$RUNSC" --root "$RUNSC_ROOT" delete --force "$CONTAINER_ID" 2>/dev/null || true
    "$RUNSC" --root "$RUNSC_ROOT" delete --force "$RESTORE_ID"  2>/dev/null || true
    T0=$(ts_ms); RST_RC=0
    "$RUNSC" "${RUNSC_FLAGS[@]}" restore --detach \
        --image-path="$CKPT_DIR" --bundle="$RESTORE_BUNDLE_DIR" \
        --pid-file="$LOG_DIR/restore.pid" "$RESTORE_ID" \
        >"$LOG_DIR/runsc-restore.log" 2>&1 || RST_RC=$?
    T_RESTORE=$(( $(ts_ms) - T0 ))
    if [[ "$RST_RC" -ne 0 || "$(cb_state "$RESTORE_ID")" != "running" ]]; then
        mark_repro "restore" "runsc restore rc=$RST_RC state=$(cb_state "$RESTORE_ID") after ${T_RESTORE}ms"
        tail -15 "$LOG_DIR/runsc-restore.log" | sed 's/^/    | /'
        grep -hE 'cuda-checkpoint|NvStatus|NV_ERR|toggle|panic' "$LOG_DIR"/runsc.log.*boot.txt 2>/dev/null | tail -15 | sed 's/^/    boot| /' || true
    else
        ok "restore OK: ${T_RESTORE} ms"
        if [[ "$SUSPEND" = "1" ]]; then
            echo ""
            info "ncclCommResume on all ranks (post-restore)"
            RES=$(cb_curl "$RESTORE_ID" --max-time 90 "http://127.0.0.1:$PORT/resume" || echo "")
            echo "    $RES"
            if [[ "$(echo "$RES" | python3 -c 'import json,sys; print(json.load(sys.stdin)["complete"])' 2>/dev/null)" != "True" ]]; then
                mark_repro "nccl-resume-api" "ncclCommResume failed after restore: $RES"
            fi
        fi
        if [[ "$LIFECYCLE" = "1" ]]; then
            echo ""
            info "NCCL comm re-init on all ranks (post-restore)"
            RI=$(cb_curl "$RESTORE_ID" --max-time 120 "http://127.0.0.1:$PORT/reinit" || echo "")
            echo "    $RI"
            if [[ "$(echo "$RI" | python3 -c 'import json,sys; print(json.load(sys.stdin)["complete"])' 2>/dev/null)" != "True" ]]; then
                mark_repro "nccl-reinit" "NCCL re-init failed after restore: $RI"
            fi
        fi
        # postRestore toggles CUDA procs asynchronously; give /verify one try
        # with the full timeout — hangs are exactly what we're probing for.
        echo ""
        info "Post-restore /verify"
        POST=$(cb_curl "$RESTORE_ID" --max-time 90 "http://127.0.0.1:$PORT/verify" || echo "")
        echo "    $POST"
        if [[ -z "$POST" ]]; then
            mark_repro "verify" "verify endpoint unreachable after restore"
            tail -15 "$APPLOG_DIR/app.log" 2>/dev/null | sed 's/^/    app| /' || true
        elif [[ "$(echo "$POST" | python3 -c 'import json,sys; print(json.load(sys.stdin)["complete"])' 2>/dev/null)" != "True" ]]; then
            mark_repro "verify-hang" "ranks hung: $(echo "$POST" | python3 -c 'import json,sys; print(json.load(sys.stdin)["hung_ranks"])' 2>/dev/null)"
            grep -hE 'NCCL|cuda' "$APPLOG_DIR/app.log" 2>/dev/null | tail -8 | sed 's/^/    app| /' || true
        else
            POST_V=$(echo "$POST" | python3 -c 'import json,sys; d=json.load(sys.stdin)["ranks"]; print({k:(v["persisted"],v["fresh"]) for k,v in d.items()})')
            if [[ "$POST_V" == "$REF_V" ]]; then
                ok "persisted + fresh allreduce values match EXACTLY"
            else
                mark_repro "data" "mismatch: ref=$REF_V post=$POST_V"
            fi
        fi
    fi
fi

# ── summary ───────────────────────────────────────────────────────────────
echo ""
banner "── Bench 7 summary ──────────────────────────────────────────────"
row "Mode / GPUs / active:"   "$MODE / $GPU_DEVICES / $ACTIVE (suspend=$SUSPEND)"
row "NCCL env:"               "${NCCL_ENV:-<default>}"
row "Transports:"             "${TRANSPORTS:-?}"
row "Checkpoint:"             "${T_CHECKPOINT:-–} ms (rc=${CKPT_RC:-–})"
row "Restore:"                "${T_RESTORE:-–} ms (rc=${RST_RC:-–})"
echo ""
if [[ -z "$REPRO_STAGE" ]]; then
    banner "RESULT: PASS ✓ (no NCCL snapshot failure reproduced)"
    exit 0
else
    banner "RESULT: REPRO ✗ stage=$REPRO_STAGE — $REPRO_DETAIL"
    exit 2
fi
