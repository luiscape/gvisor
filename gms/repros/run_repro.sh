#!/usr/bin/env bash
# --------------------------------------------------------------------------
#  run_repro.sh — run one local gVisor GPU-snapshot repro (classes A/B1/B2/C1/C2).
#
#  Boots the chosen repro app under runsc, snapshots it, restores it, and
#  classifies the outcome:
#    PASS          checkpoint + restore + /verify all succeed (class A, and
#                  B*/C* once their fix is in the runsc under test)
#    CHECKPOINT    checkpoint blocked  (B-class: frontendFD / live nvproxy
#                  clients not saveable)
#    RESTORE       restore blocked     (C-class: cuda-checkpoint toggle-to-
#                  running "operation not supported")
#
#  Snapshot flow (--mode):
#    native (default)  runsc checkpoint --cuda-checkpoint-path=...   (the
#                      sentry enumerates + toggles CUDA procs itself)
#    modal             mimic modal-client: toggle CUDA pids in-container via
#                      cuda-checkpoint, THEN a plain runsc checkpoint
#
#  Usage:
#    sudo bash gms/repros/run_repro.sh <a|b1|b2|c1|c2> [--mode native|modal]
#                                      [--gpus 0] [--rebuild]
#    sudo bash gms/repros/run_repro.sh all        # run every class
#
#  Requires: a runsc with nvproxy + cuda-checkpoint support at $RUNSC
#  (default /usr/local/bin/runsc-crbench).
# --------------------------------------------------------------------------
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Reuse the cr-bench harness for rootfs/bundle/boot/GPU plumbing.
source "$SCRIPT_DIR/../../cr-bench/common.sh"

REPRO_IMAGE="gms-repro"
MODE="native"
GPU_DEVICES="${GPU_DEVICES:-0}"
REBUILD="${REBUILD:-0}"

usage() { sed -n '2,30p' "$0"; exit "${1:-0}"; }

[[ $# -lt 1 ]] && usage 1
ID="$1"; shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)    MODE="$2"; shift 2 ;;
        --gpus)    GPU_DEVICES="$2"; shift 2 ;;
        --rebuild) REBUILD=1; shift ;;
        --help|-h) usage 0 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ── per-class configuration ───────────────────────────────────────────────
declare -A APP=(
    [a]=repro_a_cuda_context.py
    [b1]=repro_b1_nvml_fd.py
    [b2]=repro_b2_inherited_fork_fd.py
    [c1]=repro_c1_cuda_ipc_handle.py
    [c2]=repro_c2_cumem_vmm.py
    [inductor]=repro_inductor.py
)
declare -A APP_ENV=(
    [c2]="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
)
# Optional: pin Inductor's compile-worker pool size (COMPILE_THREADS=N).
[[ -n "${COMPILE_THREADS:-}" ]] && APP_ENV[inductor]="TORCHINDUCTOR_COMPILE_THREADS=$COMPILE_THREADS"
# Classes whose /verify result must match pre vs post (GPU buffer survived).
declare -A CHECK_VERIFY=( [a]=1 [c2]=1 [inductor]=1 )

run_one() {
    local id="$1"
    local app="${APP[$id]}"
    if [[ -z "$app" ]]; then echo "unknown repro id: $id"; return 2; fi

    BENCH_NAME="gms-repro-$id"
    IMAGE="$REPRO_IMAGE"
    CB_GPU=1
    PORT="${PORT:-8199}"
    APP_LOG="/applog/app.log"
    COMPRESSION=none

    # Build the repro image once (cb_prepare_rootfs reuses it if present).
    if [[ "$REBUILD" = "1" ]] || ! docker image inspect "$REPRO_IMAGE" >/dev/null 2>&1; then
        info "Building $REPRO_IMAGE"
        docker build -q -t "$REPRO_IMAGE" -f "$SCRIPT_DIR/images/Dockerfile" "$SCRIPT_DIR" >/dev/null
        REBUILD_ROOTFS=1
    fi

    cb_init
    cb_detect_gpu
    cb_runsc_flags

    banner ""
    banner "== Repro $id ($app) | mode=$MODE | GPUs=$GPU_DEVICES =="

    cb_prepare_rootfs
    CB_CMD="exec python3 /app/$app >$APP_LOG 2>&1"
    CB_ENV="PORT=$PORT"
    [[ -n "${APP_ENV[$id]:-}" ]] && CB_ENV+=$'\n'"${APP_ENV[$id]}"
    cb_write_bundle
    cb_run_and_wait_health

    local ref=""
    if [[ -n "${CHECK_VERIFY[$id]:-}" ]]; then
        ref=$(cb_curl "$CONTAINER_ID" --max-time 60 "http://127.0.0.1:$PORT/verify" || echo "")
        ok "reference /verify: $ref"
    fi

    # ── checkpoint (non-fatal, classified) ────────────────────────────────
    local stage="" detail=""
    if [[ "$MODE" = "modal" ]]; then
        info "modal-flow: in-container cuda-checkpoint toggle of CUDA pids"
        _modal_toggle "$CONTAINER_ID" || { stage=RESTORE; detail="in-container toggle (checkpoint) failed"; }
    fi

    local ckpt_flags=(--image-path="$CKPT_DIR" --compression=none --exclude-committed-zero-pages)
    [[ "$MODE" = "native" ]] && ckpt_flags+=(--cuda-checkpoint-path="$CUDA_CHECKPOINT_PATH")
    [[ "${CUDA_CKPT_SEQUENTIAL:-0}" = "1" ]] && ckpt_flags+=(--cuda-checkpoint-sequential)
    local t0 rc=0
    t0=$(ts_ms)
    if [[ -z "$stage" ]]; then
        "$RUNSC" "${RUNSC_FLAGS[@]}" checkpoint "${ckpt_flags[@]}" "$CONTAINER_ID" \
            >"$LOG_DIR/runsc-checkpoint.log" 2>&1 || rc=$?
        local t_ckpt=$(( $(ts_ms) - t0 ))
        if [[ "$rc" -ne 0 ]]; then
            if grep -qE 'frontendFD is not saveable|live nvproxy clients|uvmFD is not saveable' \
                "$LOG_DIR/runsc-checkpoint.log" "$LOG_DIR"/runsc.log.*boot.txt 2>/dev/null; then
                stage=CHECKPOINT; detail="nvproxy FD not saveable (B-class), rc=$rc ${t_ckpt}ms"
            else
                stage=CHECKPOINT; detail="checkpoint failed rc=$rc ${t_ckpt}ms (see log)"
            fi
        else
            ok "checkpoint OK: ${t_ckpt} ms"
        fi
    fi

    # ── restore (non-fatal, classified) ───────────────────────────────────
    local post=""
    if [[ -z "$stage" ]]; then
        "$RUNSC" --root "$RUNSC_ROOT" delete --force "$CONTAINER_ID" 2>/dev/null || true
        "$RUNSC" --root "$RUNSC_ROOT" delete --force "$RESTORE_ID" 2>/dev/null || true
        t0=$(ts_ms); rc=0
        "$RUNSC" "${RUNSC_FLAGS[@]}" restore --detach \
            --image-path="$CKPT_DIR" --bundle="$RESTORE_BUNDLE_DIR" \
            --pid-file="$LOG_DIR/restore.pid" "$RESTORE_ID" \
            >"$LOG_DIR/runsc-restore.log" 2>&1 || rc=$?
        local t_rst=$(( $(ts_ms) - t0 ))
        if [[ "$rc" -ne 0 || "$(cb_state "$RESTORE_ID")" != "running" ]]; then
            if grep -qE 'operation not supported|OS call failed|invalid argument|toggle' \
                "$LOG_DIR/runsc-restore.log" "$LOG_DIR"/runsc.log.*boot.txt 2>/dev/null; then
                stage=RESTORE; detail="cuda-checkpoint toggle-to-running failed (C-class), ${t_rst}ms"
            else
                stage=RESTORE; detail="restore failed rc=$rc state=$(cb_state "$RESTORE_ID") ${t_rst}ms"
            fi
        else
            ok "restore OK: ${t_rst} ms"
            if [[ "$MODE" = "modal" ]]; then
                _modal_toggle "$RESTORE_ID" || { stage=RESTORE; detail="in-container untoggle (restore) failed"; }
            fi
        fi
    fi

    # ── verify (data integrity for A / C2) ────────────────────────────────
    if [[ -z "$stage" && -n "${CHECK_VERIFY[$id]:-}" ]]; then
        post=$(cb_curl "$RESTORE_ID" --max-time 60 "http://127.0.0.1:$PORT/verify" || echo "")
        ok "post /verify: $post"
        if [[ -z "$post" ]]; then
            stage=RESTORE; detail="/verify unreachable after restore"
        elif [[ -n "$ref" && "$post" != "$ref" ]]; then
            stage=RESTORE; detail="/verify mismatch: ref=$ref post=$post"
        fi
    fi

    # ── FD-holder dump + versions from container logs ──────────────────────
    echo ""; info "pre-snapshot FD dump (from app):"
    grep -hE 'nvidia FD holders|PID [0-9]+ \(' "$APPLOG_DIR/app.log" 2>/dev/null | sed 's/^/    | /' | head -20 || true
    DRIVER_LINE=$(grep -hm1 'NVRM version\|driver' "$LOG_DIR"/runsc.log.*boot.txt 2>/dev/null | head -1 || true)

    echo ""
    if [[ -z "$stage" ]]; then
        banner "REPRO $id: RESULT = PASS ✓"
        RESULT="PASS"
    else
        banner "REPRO $id: RESULT = $stage-BLOCKED ✗ — $detail"
        RESULT="$stage"
    fi
    LAST_DETAIL="$detail"
    cb_cleanup 2>/dev/null || true
    trap - EXIT
}

# In-container toggle of CUDA pids, faithful to modal-client
# gpu_memory_snapshot.py: enumerate /proc, keep only pids whose
# `cuda-checkpoint --get-state` is "running" (real CUDA sessions), and
# toggle just those in parallel. NVML-only / inherited-FD holders are
# invisible to it (get-state fails) and are deliberately NOT toggled — which
# is exactly why their frontend FDs survive to the checkpoint.
_modal_toggle() {
    local cid="$1" pids p rc=0
    pids=$("$RUNSC" --root "$RUNSC_ROOT" exec "$cid" /bin/sh -c '
        for d in /proc/[0-9]*; do
            p=${d#/proc/}
            /usr/local/bin/cuda-checkpoint --get-state --pid "$p" 2>/dev/null | grep -q running && echo "$p"
        done' 2>/dev/null | sort -un | tr '\n' ' ')
    if [[ -z "$pids" ]]; then
        echo "        (no checkpointable CUDA sessions found)"
        return 0
    fi
    echo "        toggling CUDA sessions: $pids"
    for p in $pids; do
        "$RUNSC" --root "$RUNSC_ROOT" exec "$cid" /usr/local/bin/cuda-checkpoint --toggle --pid "$p" \
            >/dev/null 2>&1 &
    done
    local j; for j in $(jobs -p); do wait "$j" || rc=1; done
    return $rc
}

if [[ "$ID" = "all" ]]; then
    for id in a b1 b2 c1 c2 inductor; do
        run_one "$id" || true
    done
else
    run_one "$ID"
fi
