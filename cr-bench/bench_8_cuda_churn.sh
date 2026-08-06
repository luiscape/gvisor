#!/usr/bin/env bash
# --------------------------------------------------------------------------
#  bench_8_cuda_churn.sh — snapshot under continuous CUDA process churn.
#
#  Runs apps/cuda_churn.py (long-lived CUDA verifier processes + an
#  endless stream of short-lived multi-threaded CUDA processes) and
#  checkpoints WHILE processes are being born (mid-cuInit) and dying.
#  This stresses the sentry's CUDA process enumeration
#  (state_cuda.go: re-enumeration passes, exactly-once undo, toggle of
#  processes that may exit mid-pass).
#
#  Verification:
#    - checkpoint + restore succeed (or fail cleanly — reported as REPRO)
#    - long-lived workers: persisted checksums + fresh matmul match
#    - churn RESUMES after restore (births counter increases) — new
#      processes can still init CUDA post-restore
#
#  Exit: 0 PASS, 2 REPRO, 1 harness error.
#
#  Usage: sudo bash cr-bench/bench_8_cuda_churn.sh [--gpus 0,1]
#         [--churn-slots N] [--threads N] [--long-workers N] [--repeat N]
# --------------------------------------------------------------------------
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

BENCH_NAME="cr-bench-churn"
IMAGE="${IMAGE:-cr-bench-gpu}"
DOCKERFILE="images/Dockerfile.gpu"
PORT="${PORT:-8199}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
LONG_WORKERS="${LONG_WORKERS:-2}"
CHURN_SLOTS="${CHURN_SLOTS:-4}"
NVML_WORKERS="${NVML_WORKERS:-1}"
NVML_CHURN_SLOTS="${NVML_CHURN_SLOTS:-2}"
RAWFD_WORKERS="${RAWFD_WORKERS:-1}"
RAWFD_CHURN_SLOTS="${RAWFD_CHURN_SLOTS:-2}"
THREADS="${THREADS:-4}"
TENSOR_MB="${TENSOR_MB:-32}"
JITTER_MS="${JITTER_MS:-500}"
MIN_LIFE_MS="${MIN_LIFE_MS:-500}"
MAX_LIFE_MS="${MAX_LIFE_MS:-3000}"
REPEAT="${REPEAT:-1}"          # checkpoint/restore cycles (churn re-rolls the dice)
CUDA_CKPT="${CUDA_CKPT:-1}"    # 0: omit --cuda-checkpoint-path (no GPU toggle;
                               # reproduces plain-save behavior on old runsc)
MODAL_FLOW="${MODAL_FLOW:-0}"  # 1: mimic modal-client gpu_memory_snapshot.py:
                               # toggle CUDA pids via in-container cuda-checkpoint
                               # BEFORE a plain checkpoint (set CUDA_CKPT=0), and
                               # untoggle after restore. The sentry is not involved
                               # in GPU state handling at all.
CB_GPU=1
APP_LOG="/applog/app.log"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)         GPU_DEVICES="$2"; shift 2 ;;
        --long-workers) LONG_WORKERS="$2"; shift 2 ;;
        --churn-slots)  CHURN_SLOTS="$2"; shift 2 ;;
        --threads)      THREADS="$2"; shift 2 ;;
        --repeat)       REPEAT="$2"; shift 2 ;;
        --rebuild-rootfs) REBUILD_ROOTFS=1; shift ;;
        --help|-h)      sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

NUM_GPUS=$(echo "$GPU_DEVICES" | tr ',' '\n' | grep -c .)

REPRO_STAGE=""
REPRO_DETAIL=""
mark_repro() {
    if [[ -z "$REPRO_STAGE" ]]; then REPRO_STAGE="$1"; shift; REPRO_DETAIL="$*"; fi
    fail "REPRO at stage '$1'"
}

cb_init
cb_detect_gpu
cb_runsc_flags

banner ""
banner "╔══════════════════════════════════════════════════════════════════╗"
banner "║   Benchmark 8: snapshot under continuous CUDA process churn     ║"
banner "╚══════════════════════════════════════════════════════════════════╝"
info "GPUs: $GPU_DEVICES | cuda long=$LONG_WORKERS/churn=$CHURN_SLOTS nvml long=$NVML_WORKERS/churn=$NVML_CHURN_SLOTS rawfd long=$RAWFD_WORKERS/churn=$RAWFD_CHURN_SLOTS | threads=$THREADS life ${MIN_LIFE_MS}-${MAX_LIFE_MS}ms jitter ${JITTER_MS}ms | cycles=$REPEAT"
echo ""

cb_prepare_rootfs

CB_CMD="exec python3 /app/cuda_churn.py >$APP_LOG 2>&1"
CB_ENV="LONG_WORKERS=$LONG_WORKERS
CHURN_SLOTS=$CHURN_SLOTS
NVML_WORKERS=$NVML_WORKERS
NVML_CHURN_SLOTS=$NVML_CHURN_SLOTS
RAWFD_WORKERS=$RAWFD_WORKERS
RAWFD_CHURN_SLOTS=$RAWFD_CHURN_SLOTS
THREADS=$THREADS
TENSOR_MB=$TENSOR_MB
JITTER_MS=$JITTER_MS
MIN_LIFE_MS=$MIN_LIFE_MS
MAX_LIFE_MS=$MAX_LIFE_MS
PORT=$PORT"
cb_write_bundle

echo ""
cb_run_and_wait_health

_get() { cb_curl "$1" --max-time 60 "http://127.0.0.1:$PORT/$2" || echo ""; }
_field() { echo "$1" | python3 -c "import json,sys; print(json.load(sys.stdin)['$2'])" 2>/dev/null || echo ""; }

# Let churn build up steam, grab reference state.
sleep 3
REF_STATS=$(_get "$CONTAINER_ID" stats); echo "    pre stats: $REF_STATS"
REF=$(_get "$CONTAINER_ID" verify)
if [[ "$(_field "$REF" complete)" != "True" ]]; then
    fail "reference verify incomplete — harness error"; echo "    $REF"; exit 1
fi
REF_V=$(echo "$REF" | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["ranks"], sort_keys=True))')
ok "reference values: $REF_V"
if [[ "$(_field "$REF_STATS" deaths_fail)" != "0" ]]; then
    fail "churn workers failing before any checkpoint — harness error"; exit 1
fi

# In-container parallel toggle of the long-lived CUDA workers' pids,
# mimicking modal-client's CudaCheckpointSession (it can only see
# toggleable CUDA sessions; NVML/bare-FD holders are invisible to it).
modal_toggle() {  # container-id
    local cid="$1" pids p rc=0
    pids=$(_get "$cid" stats | python3 -c 'import json,sys; d=json.load(sys.stdin)["long_worker_pids"]; print(" ".join(str(v) for k,v in d.items() if k.startswith("cuda:")))')
    [[ -z "$pids" ]] && { echo "        (no CUDA pids)"; return 0; }
    echo "        in-container cuda-checkpoint --toggle on pids: $pids"
    for p in $pids; do
        "$RUNSC" --root "$RUNSC_ROOT" exec "$cid" /usr/local/bin/cuda-checkpoint --toggle --pid "$p" &
    done
    local j; for j in $(jobs -p); do wait "$j" || rc=1; done
    return $rc
}

SRC_ID="$CONTAINER_ID"
CYCLE_TIMES=()
for cycle in $(seq 1 "$REPEAT"); do
    echo ""
    info "── cycle $cycle/$REPEAT: checkpoint DURING churn ──"
    CKPT_FLAGS=(--image-path="$CKPT_DIR" --compression="$COMPRESSION")
    [[ "$CUDA_CKPT" = "1" ]] && CKPT_FLAGS+=(--cuda-checkpoint-path="$CUDA_CHECKPOINT_PATH")
    [[ "$EXCLUDE_ZERO" = "1" ]] && CKPT_FLAGS+=(--exclude-committed-zero-pages)
    if [[ "$MODAL_FLOW" = "1" ]]; then
        if modal_toggle "$SRC_ID"; then
            ok "modal-flow: CUDA pids toggled in-container (GPU mem: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$GPU_DEVICES" | tr '\n' ' '))"
        else
            mark_repro "modal-toggle-c$cycle" "in-container cuda-checkpoint toggle failed"
            break
        fi
    fi
    rm -rf "$CKPT_DIR"; mkdir -p "$CKPT_DIR"
    T0=$(ts_ms); CKPT_RC=0
    "$RUNSC" "${RUNSC_FLAGS[@]}" checkpoint "${CKPT_FLAGS[@]}" "$SRC_ID" \
        >"$LOG_DIR/runsc-checkpoint-$cycle.log" 2>&1 || CKPT_RC=$?
    T_CKPT=$(( $(ts_ms) - T0 ))
    if [[ "$CKPT_RC" -ne 0 ]]; then
        # Classify: driver-side (cuda-checkpoint exec failed) vs gVisor-side
        # (sentry save error, e.g. serializing untoggled nvproxy state).
        if grep -q 'cuda-checkpoint' "$LOG_DIR/runsc-checkpoint-$cycle.log"; then
            mark_repro "checkpoint-driver-c$cycle" "cuda-checkpoint toggle failed (rc=$CKPT_RC, ${T_CKPT}ms)"
        else
            mark_repro "checkpoint-GVISOR-c$cycle" "sentry save failed without cuda-checkpoint error (rc=$CKPT_RC, ${T_CKPT}ms)"
        fi
        tail -10 "$LOG_DIR/runsc-checkpoint-$cycle.log" | sed 's/^/    | /'
        grep -hE 'panic|Fatal|save failed|Unimplemented|unsupported|unsavable' "$LOG_DIR"/runsc.log.*boot.txt 2>/dev/null | tail -6 | sed 's/^/    boot| /' || true
        break
    fi
    ok "checkpoint OK: ${T_CKPT} ms"

    RESTORE_TARGET="$RESTORE_ID-c$cycle"
    "$RUNSC" --root "$RUNSC_ROOT" delete --force "$SRC_ID" 2>/dev/null || true
    T0=$(ts_ms); RST_RC=0
    "$RUNSC" "${RUNSC_FLAGS[@]}" restore --detach \
        --image-path="$CKPT_DIR" --bundle="$RESTORE_BUNDLE_DIR" \
        --pid-file="$LOG_DIR/restore-$cycle.pid" "$RESTORE_TARGET" \
        >"$LOG_DIR/runsc-restore-$cycle.log" 2>&1 || RST_RC=$?
    T_RST=$(( $(ts_ms) - T0 ))
    if [[ "$RST_RC" -ne 0 || "$(cb_state "$RESTORE_TARGET")" != "running" ]]; then
        mark_repro "restore-c$cycle" "rc=$RST_RC state=$(cb_state "$RESTORE_TARGET") after ${T_RST}ms"
        tail -10 "$LOG_DIR/runsc-restore-$cycle.log" | sed 's/^/    | /'
        grep -hE 'cuda-checkpoint|NvStatus|post restore' "$LOG_DIR"/runsc.log.*boot.txt 2>/dev/null | tail -8 | sed 's/^/    boot| /' || true
        break
    fi
    ok "restore OK: ${T_RST} ms"
    CYCLE_TIMES+=("c$cycle:${T_CKPT}/${T_RST}ms")
    SRC_ID="$RESTORE_TARGET"

    if [[ "$MODAL_FLOW" = "1" ]]; then
        if modal_toggle "$SRC_ID"; then
            ok "modal-flow: CUDA pids untoggled after restore"
        else
            mark_repro "modal-untoggle-c$cycle" "in-container cuda-checkpoint restore toggle failed"
            break
        fi
    fi

    # Long-lived integrity + churn resumption.
    POST=$(_get "$SRC_ID" verify); 
    if [[ "$(_field "$POST" complete)" != "True" ]]; then
        mark_repro "verify-c$cycle" "long workers hung/errored: $POST"
        break
    fi
    POST_V=$(echo "$POST" | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["ranks"], sort_keys=True))')
    if [[ "$POST_V" != "$REF_V" ]]; then
        mark_repro "data-c$cycle" "mismatch ref=$REF_V post=$POST_V"
        break
    fi
    TOTAL_CHURN=$(( CHURN_SLOTS + NVML_CHURN_SLOTS + RAWFD_CHURN_SLOTS ))
    if [[ "$TOTAL_CHURN" -gt 0 ]]; then
        S1=$(_get "$SRC_ID" stats); B1=$(_field "$S1" births); F1=$(_field "$S1" deaths_fail)
        sleep 4
        S2=$(_get "$SRC_ID" stats); B2=$(_field "$S2" births); F2=$(_field "$S2" deaths_fail)
        if [[ -z "$B1" || -z "$B2" || "$B2" -le "$B1" ]]; then
            mark_repro "churn-stalled-c$cycle" "births $B1 -> $B2 (churn did not resume)"
            break
        fi
        if [[ "$F2" != "$F1" ]]; then
            mark_repro "churn-failures-c$cycle" "worker failures grew: $F1 -> $F2"
            break
        fi
        ok "cycle $cycle verified: checksums exact, churn resumed (births $B1 -> $B2, failures $F2)"
    else
        ok "cycle $cycle verified: checksums exact (no churn configured)"
    fi
done

# Enumeration evidence from sentry logs.
echo ""
info "sentry enumeration evidence:"
grep -hE 'cuda-checkpoint on [0-9]+ processes|additional CUDA process|get-state|pass' \
    "$LOG_DIR"/runsc.log.*boot.txt 2>/dev/null | grep -iE 'cuda' | tail -8 | sed 's/^/    | /' || true

echo ""
banner "── Bench 8 summary ──────────────────────────────────────────────"
row "Cycles:"       "${CYCLE_TIMES[*]:-none completed}"
row "Final stats:"  "$(_get "$SRC_ID" stats 2>/dev/null | head -c 200)"
echo ""
if [[ -z "$REPRO_STAGE" ]]; then
    banner "RESULT: PASS ✓ (snapshot survived continuous CUDA churn x$REPEAT)"
    exit 0
else
    banner "RESULT: REPRO ✗ stage=$REPRO_STAGE — $REPRO_DETAIL"
    exit 2
fi
