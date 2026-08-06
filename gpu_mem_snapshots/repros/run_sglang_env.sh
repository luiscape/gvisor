#!/usr/bin/env bash
# --------------------------------------------------------------------------
#  run_sglang_env.sh — local adaptation of gcr/qwen_repro.py.
#
#  Boots SGLang (small model) under runsc, warms up, releases GPU memory,
#  dumps every /dev/nvidia* FD holder's environment (dump_gpu_process_env),
#  then snapshots + restores + wakes + verifies. The OUTCOME is decided by
#  the env-var --profile, exactly as in qwen_repro.py:
#
#    good          NCCL_CUMEM_ENABLE=0 NCCL_CUMEM_HOST_ENABLE=0
#                  TORCHINDUCTOR_COMPILE_THREADS=1  IPC transport+cache off
#                  -> snapshot-safe
#    bad-nccl      NCCL_CUMEM_ENABLE=1 NCCL_CUMEM_HOST_ENABLE=1 (rest good)
#                  -> at TP>=2, restore fails: NCCL host-cuMem FD handles
#                     ("operation not supported"). No effect at TP=1 (no comm).
#    bad-inductor  --enable-torch-compile with TORCHINDUCTOR_COMPILE_THREADS
#                  unset -> Inductor SubprocPool compile-workers hold
#                  inherited /dev/nvidia* FDs -> checkpoint blocked on a
#                  stub runsc ("nvproxy.frontendFD is not saveable").
#
#  Usage:
#    sudo bash gms/repros/run_sglang_env.sh [--profile good|bad-nccl|bad-inductor]
#         [--tp N] [--gpus 0,1] [--model ID]
#    RUNSC=/usr/local/bin/runsc  ...   # use the stub (prod-equivalent) build
# --------------------------------------------------------------------------
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../cr-bench/common.sh"

PROFILE="good"
TP="${TP:-1}"
GPU_DEVICES="${GPU_DEVICES:-0}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --tp)      TP="$2"; shift 2 ;;
        --gpus)    GPU_DEVICES="$2"; shift 2 ;;
        --model)   MODEL="$2"; shift 2 ;;
        --help|-h) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

BENCH_NAME="gms-sglang-env-$PROFILE"
IMAGE="cr-bench-sglang"
CB_GPU=1
COMPRESSION=none
PORT="${PORT:-8199}"          # wrapper control port (always responsive)
SGLANG_PORT=30000
APP_LOG="/applog/app.log"
CUDA_CKPT="${CUDA_CKPT:-1}"   # 0 to omit --cuda-checkpoint-path (stub repro)

# torch_memory_saver (--enable-memory-saver) LD_PRELOADs a hook linked
# against libcudart.so.13, shipped in torch's pip nvidia/cu13 package.
LD_EXTRA="/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib"

# ── env-var profile (the whole point of this repro) ───────────────────────
TORCH_COMPILE=0
case "$PROFILE" in
    good)
        PROFILE_ENV="NCCL_CUMEM_ENABLE=0
NCCL_CUMEM_HOST_ENABLE=0
TORCHINDUCTOR_COMPILE_THREADS=1
SGLANG_USE_CUDA_IPC_TRANSPORT=0
SGLANG_USE_IPC_POOL_HANDLE_CACHE=0" ;;
    bad-nccl)
        PROFILE_ENV="NCCL_CUMEM_ENABLE=1
NCCL_CUMEM_HOST_ENABLE=1
TORCHINDUCTOR_COMPILE_THREADS=1
SGLANG_USE_CUDA_IPC_TRANSPORT=0
SGLANG_USE_IPC_POOL_HANDLE_CACHE=0" ;;
    bad-inductor)
        # Enable torch.compile with TORCHINDUCTOR_COMPILE_THREADS>1 so
        # Inductor's SubprocPool spawns persistent compile-worker
        # subprocesses that inherit the model runner's /dev/nvidia* FDs
        # (clientless holders). This is the case (1) must checkpoint.
        TORCH_COMPILE=1
        PROFILE_ENV="NCCL_CUMEM_ENABLE=0
NCCL_CUMEM_HOST_ENABLE=0
TORCHINDUCTOR_COMPILE_THREADS=${COMPILE_THREADS:-4}
SGLANG_USE_CUDA_IPC_TRANSPORT=0
SGLANG_USE_IPC_POOL_HANDLE_CACHE=0" ;;
    *) echo "unknown profile: $PROFILE"; exit 1 ;;
esac

cb_init
cb_detect_gpu
cb_runsc_flags

banner ""
banner "== SGLang env repro | profile=$PROFILE | TP=$TP | GPUs=$GPU_DEVICES =="
info "model: $MODEL | runsc: $($RUNSC --version | head -1)"
info "profile env:"; echo "$PROFILE_ENV" | sed 's/^/      /'

cb_prepare_rootfs
# Drop the repro app into the (already-built) SGLang rootfs overlay.
cp "$SCRIPT_DIR/repro_sglang_env.py" "$ROOTFS_MERGED/app/repro_sglang_env.py"

CB_CMD="exec python3 /app/repro_sglang_env.py >$APP_LOG 2>&1"
CB_ENV="HF_HOME=/app/hf_cache
HF_HUB_OFFLINE=1
MODEL=$MODEL
TP=$TP
SGLANG_PORT=$SGLANG_PORT
PORT=$PORT
TORCH_COMPILE=$TORCH_COMPILE
LD_LIBRARY_PATH=$LD_EXTRA
$PROFILE_ENV"
cb_write_bundle
cb_run_and_wait_health
# The reference answer is captured inside the app while awake (before sleep);
# a pre-snapshot /verify would fail because the server is asleep.
REF=$(grep -m1 'reference answer' "$APPLOG_DIR/app.log" 2>/dev/null || echo "")
info "$REF"

STAGE=""; DETAIL=""
mark() { [[ -z "$STAGE" ]] && { STAGE="$1"; DETAIL="$2"; }; }

# ── checkpoint (classified) ───────────────────────────────────────────────
echo ""
info "runsc checkpoint (cuda-checkpoint=$CUDA_CKPT)"
CKPT_FLAGS=(--image-path="$CKPT_DIR" --compression=none --exclude-committed-zero-pages)
[[ "$CUDA_CKPT" = "1" ]] && CKPT_FLAGS+=(--cuda-checkpoint-path="$CUDA_CHECKPOINT_PATH")
t0=$(ts_ms); rc=0
"$RUNSC" "${RUNSC_FLAGS[@]}" checkpoint "${CKPT_FLAGS[@]}" "$CONTAINER_ID" \
    >"$LOG_DIR/runsc-checkpoint.log" 2>&1 || rc=$?
T_CKPT=$(( $(ts_ms) - t0 ))
if [[ "$rc" -ne 0 ]]; then
    if grep -qE 'frontendFD is not saveable|live nvproxy clients|uvmFD is not saveable' \
        "$LOG_DIR/runsc-checkpoint.log" "$LOG_DIR"/runsc.log.*boot.txt 2>/dev/null; then
        mark CHECKPOINT "nvproxy FD not saveable (inherited-fork FDs) rc=$rc ${T_CKPT}ms"
    else
        mark CHECKPOINT "checkpoint failed rc=$rc ${T_CKPT}ms"
    fi
    tail -8 "$LOG_DIR/runsc-checkpoint.log" | sed 's/^/    | /'
else
    ok "checkpoint OK: ${T_CKPT} ms"
fi

# ── restore + wake + verify (classified) ──────────────────────────────────
if [[ -z "$STAGE" ]]; then
    "$RUNSC" --root "$RUNSC_ROOT" delete --force "$CONTAINER_ID" 2>/dev/null || true
    "$RUNSC" --root "$RUNSC_ROOT" delete --force "$RESTORE_ID" 2>/dev/null || true
    t0=$(ts_ms); rc=0
    "$RUNSC" "${RUNSC_FLAGS[@]}" restore --detach \
        --image-path="$CKPT_DIR" --bundle="$RESTORE_BUNDLE_DIR" \
        --pid-file="$LOG_DIR/restore.pid" "$RESTORE_ID" \
        >"$LOG_DIR/runsc-restore.log" 2>&1 || rc=$?
    T_RST=$(( $(ts_ms) - t0 ))
    if [[ "$rc" -ne 0 || "$(cb_state "$RESTORE_ID")" != "running" ]]; then
        if grep -qE 'Error toggling CUDA in process ID|post restore work failed' \
            "$LOG_DIR"/runsc.log.*boot.txt 2>/dev/null; then
            mark RESTORE "cuda-checkpoint toggle-to-running failed (NCCL host-cuMem) ${T_RST}ms"
        else
            mark RESTORE "restore failed rc=$rc state=$(cb_state "$RESTORE_ID") ${T_RST}ms"
        fi
        grep -hE 'operation not supported|post restore|Killing' "$LOG_DIR"/runsc.log.*boot.txt 2>/dev/null | tail -4 | sed 's/^/    boot| /' || true
    else
        ok "restore OK: ${T_RST} ms"
        WAKE=$(cb_curl "$RESTORE_ID" --max-time 360 "http://127.0.0.1:$PORT/wake" || echo "")
        info "wake: $WAKE"
        POST=$(cb_curl "$RESTORE_ID" --max-time 120 "http://127.0.0.1:$PORT/verify" || echo "")
        info "post /verify: $POST"
        # The sentry may have killed the sandbox during postRestore (async
        # cuda-checkpoint toggle) — that surfaces here as an unreachable
        # server. Check the boot log for the real signature first.
        if grep -qE 'Error toggling CUDA in process ID|post restore work failed' "$LOG_DIR"/runsc.log.*boot.txt 2>/dev/null; then
            mark RESTORE "cuda-checkpoint toggle-to-running failed (NCCL host-cuMem): 'operation not supported'"
            grep -hE 'Error toggling CUDA|Killing the sandbox' "$LOG_DIR"/runsc.log.*boot.txt 2>/dev/null | tail -3 | sed 's/^/    boot| /'
        elif [[ -z "$POST" ]]; then
            mark RESTORE "/verify unreachable after wake"
        elif echo "$POST" | grep -q '"match": *false'; then
            mark RESTORE "answer mismatch pre/post: $POST"
        fi
    fi
fi

# ── FD-holder env dump (from app log) ──────────────────────────────────────
echo ""; info "dump_gpu_process_env (pre-snapshot):"
sed -n '/GPU FD-holder environment (pre-snapshot)/,/end GPU FD-holder environment/p' \
    "$APPLOG_DIR/app.log" 2>/dev/null | sed 's/^/    | /' | head -40 || true

echo ""
if [[ -z "$STAGE" ]]; then
    banner "SGLang env repro ($PROFILE): RESULT = PASS ✓"
else
    banner "SGLang env repro ($PROFILE): RESULT = $STAGE-BLOCKED ✗ — $DETAIL"
fi
