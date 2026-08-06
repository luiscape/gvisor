#!/usr/bin/env bash
# --------------------------------------------------------------------------
#  bench_2_gpu.sh — GPU memory checkpoint/restore benchmark using gVisor's
#  native cuda-checkpoint support.
#
#  Boots a PyTorch container that allocates tensors with known contents on
#  one or more GPUs, then:
#
#    runsc checkpoint --cuda-checkpoint-path=/usr/local/bin/cuda-checkpoint
#      → the sentry finds every CUDA process, execs
#        `cuda-checkpoint --toggle --pid N` on each (in parallel), which
#        drains the GPU, copies device memory to host, and releases the
#        GPU — then gVisor serializes the sandbox (GPU state travels in
#        the process's host memory inside pages.img).
#
#    runsc restore
#      → after deserialization the sentry automatically re-execs
#        `cuda-checkpoint --toggle` on the recorded processes, restoring
#        GPU memory and contexts.  No LD_PRELOAD, no helper, no daemon.
#
#  Verifies per-GPU tensor checksums match pre/post and that live compute
#  (matmul on every GPU) works after restore.
#
#  Usage:
#    sudo bash cr-bench/bench_2_gpu.sh                    # 1 GPU
#    sudo bash cr-bench/bench_2_gpu.sh --gpus 0,1,2,3     # multi-GPU (no NCCL)
#    sudo bash cr-bench/bench_2_gpu.sh --gpu-mem-mb 4096
#
#  Cross-GPU restore (device remapping): checkpoint on one GPU set and
#  restore on a different one.  The sentry records the saved device set in
#  checkpoint metadata; on restore, nvproxy remaps device FDs onto the
#  restore bundle's GPUs (positionally, sorted by minor):
#    sudo bash cr-bench/bench_2_gpu.sh --gpus 0 --restore-gpus 1
#    sudo bash cr-bench/bench_2_gpu.sh --gpus 0,1 --restore-gpus 2,3
#
#  Prerequisites: runsc with cuda-checkpoint support, NVIDIA driver >= R550
#  (>= R570 recommended), docker, nvidia-container-cli.
# --------------------------------------------------------------------------
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# ── Configuration ─────────────────────────────────────────────────────────
BENCH_NAME="cr-bench-gpu"
IMAGE="${IMAGE:-cr-bench-gpu}"
DOCKERFILE="images/Dockerfile.gpu"
PORT="${PORT:-8199}"
GPU_DEVICES="${GPU_DEVICES:-0}"     # comma-separated host GPU indices
GPU_MEM_MB="${GPU_MEM_MB:-512}"     # per GPU
NUM_TENSORS="${NUM_TENSORS:-4}"
CB_GPU=1
APP_LOG="/applog/app.log"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)            GPU_DEVICES="$2"; shift 2 ;;
        --restore-gpus)    RESTORE_GPU_DEVICES="$2"; shift 2 ;;
        --gpu-mem-mb)      GPU_MEM_MB="$2"; shift 2 ;;
        --num-tensors)     NUM_TENSORS="$2"; shift 2 ;;
        --port)            PORT="$2"; shift 2 ;;
        --compression)     COMPRESSION="$2"; shift 2 ;;
        --no-exclude-zero) EXCLUDE_ZERO=0; shift ;;
        --sequential)      CUDA_CKPT_SEQUENTIAL=1; shift ;;
        --image)           IMAGE="$2"; shift 2 ;;
        --rebuild-rootfs)  REBUILD_ROOTFS=1; shift ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "  --gpus LIST            comma-separated host GPU indices (default: 0)"
            echo "  --restore-gpus LIST    restore on a DIFFERENT GPU set (device remapping)"
            echo "  --gpu-mem-mb MB        GPU memory to allocate per GPU (default: 512)"
            echo "  --num-tensors N        tensors per GPU (default: 4)"
            echo "  --port PORT            HTTP port (default: 8199)"
            echo "  --compression MODE     none | flate-best-speed (default: none)"
            echo "  --no-exclude-zero      keep committed zero pages"
            echo "  --sequential           run cuda-checkpoint sequentially (debugging)"
            echo "  --image IMAGE          Docker image name (default: cr-bench-gpu)"
            echo "  --rebuild-rootfs       force re-extract rootfs from image"
            exit 0 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

NUM_GPUS=$(echo "$GPU_DEVICES" | tr ',' '\n' | grep -c .)

cb_init
cb_detect_gpu
cb_runsc_flags

banner ""
banner "╔══════════════════════════════════════════════════════════════════╗"
banner "║   Benchmark 2: GPU memory C/R via native cuda-checkpoint        ║"
banner "╚══════════════════════════════════════════════════════════════════╝"
echo ""
info "runsc:           $($RUNSC --version 2>&1 | head -1 || echo '?')"
info "GPU:             $GPU_NAME ($GPU_MEM_TOTAL MiB), driver $HOST_DRIVER_VER"
info "GPUs used:       $GPU_DEVICES ($NUM_GPUS)"
[[ -n "${RESTORE_GPU_DEVICES:-}" ]] && \
info "Restore GPUs:    $RESTORE_GPU_DEVICES (cross-GPU device remapping)"
info "GPU memory:      ${GPU_MEM_MB} MiB x ${NUM_TENSORS} tensors per GPU"
info "cuda-checkpoint: $CUDA_CHECKPOINT_PATH (in container), $([ "$CUDA_CKPT_SEQUENTIAL" = 1 ] && echo sequential || echo parallel)"
echo ""

# ── Phase 0/1: rootfs + bundle ────────────────────────────────────────────
cb_prepare_rootfs

CB_CMD="exec python3 /app/gpu_mem_server.py >$APP_LOG 2>&1"
CB_ENV="GPU_MEM_MB=$GPU_MEM_MB
NUM_TENSORS=$NUM_TENSORS
PORT=$PORT"
cb_write_bundle

# ── Phase 2: cold boot ────────────────────────────────────────────────────
echo ""
cb_run_and_wait_health

# ── Phase 3: reference state ──────────────────────────────────────────────
echo ""
info "Reference GPU state (pre-checkpoint)"
GPU_INFO=$(cb_curl "$CONTAINER_ID" "http://127.0.0.1:${PORT}/info" || echo "")
echo "$GPU_INFO" | python3 -m json.tool 2>/dev/null | sed 's/^/    /' | head -25 || true

REF_SUMS=$(cb_curl "$CONTAINER_ID" "http://127.0.0.1:${PORT}/checksums" || echo "")
if [[ -z "$REF_SUMS" ]]; then
    fail "Failed to fetch reference checksums"; exit 1
fi
SEEN_GPUS=$(echo "$REF_SUMS" | python3 -c 'import json,sys; print(json.load(sys.stdin)["num_gpus"])' || echo 0)
if [[ "$SEEN_GPUS" != "$NUM_GPUS" ]]; then
    fail "App sees $SEEN_GPUS GPUs, expected $NUM_GPUS"; exit 1
fi
ok "Tensor checksums captured for $SEEN_GPUS GPU(s)"

REF_MATMUL=$(cb_curl "$CONTAINER_ID" "http://127.0.0.1:${PORT}/matmul" || echo "")
if [[ -z "$REF_MATMUL" ]]; then
    fail "Failed to run reference matmul"; exit 1
fi
ok "Reference matmul: $REF_MATMUL"

# ── Phase 4: checkpoint (native cuda-checkpoint) ──────────────────────────
echo ""
cb_checkpoint

# GPU memory should be released while checkpointed.
FREE_AFTER_CKPT=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_DEVICES}" 2>/dev/null | tr '\n' ' ' || echo "?")
info "  GPU memory in use after checkpoint (MiB per GPU): $FREE_AFTER_CKPT"

# ── Phase 5: restore ──────────────────────────────────────────────────────
echo ""
cb_restore_and_wait_health

# ── Phase 6: verification ─────────────────────────────────────────────────
echo ""
info "Verification"
VERIFY_OK=0
MATMUL_OK=0

POST_SUMS=$(cb_curl "$RESTORE_ID" "http://127.0.0.1:${PORT}/checksums" || echo "")
if [[ -z "$POST_SUMS" ]]; then
    fail "Failed to fetch post-restore checksums"
elif [[ "$(echo "$REF_SUMS" | python3 -c 'import json,sys; print(json.load(sys.stdin)["checksums"])')" \
     == "$(echo "$POST_SUMS" | python3 -c 'import json,sys; print(json.load(sys.stdin)["checksums"])')" ]]; then
    ok "GPU tensor checksums match EXACTLY on all $NUM_GPUS GPU(s)"
    VERIFY_OK=1
else
    fail "GPU tensor checksum MISMATCH"
    echo "    Pre:  $REF_SUMS"
    echo "    Post: $POST_SUMS"
fi

cb_verify_gpu_placement 200

T_MM0=$(ts_ms)
POST_MATMUL=$(cb_curl "$RESTORE_ID" "http://127.0.0.1:${PORT}/matmul" || echo "")
T_MATMUL=$(( $(ts_ms) - T_MM0 ))
if [[ -n "$POST_MATMUL" ]]; then
    if [[ "$POST_MATMUL" == "$REF_MATMUL" ]]; then
        ok "Live GPU compute works on all GPUs (${T_MATMUL} ms) — deterministic matmul matches"
    else
        ok "Live GPU compute works on all GPUs (${T_MATMUL} ms)"
        warn "matmul result differs from reference: $POST_MATMUL vs $REF_MATMUL"
    fi
    MATMUL_OK=1
else
    fail "Post-restore matmul failed"
fi

# ── cuda-checkpoint timings from sentry logs ──────────────────────────────
echo ""
cb_cuda_ckpt_log_summary

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
banner "╔══════════════════════════════════════════════════════════════════╗"
banner "║   Benchmark 2 (GPU memory C/R, native cuda-checkpoint)          ║"
banner "╚══════════════════════════════════════════════════════════════════╝"
echo ""
row "GPU:"                            "$GPU_NAME, driver $HOST_DRIVER_VER"
row "GPUs used:"                      "$GPU_DEVICES ($NUM_GPUS)"
[[ -n "${RESTORE_GPU_DEVICES:-}" ]] && \
row "Restored on GPUs:"               "$RESTORE_GPU_DEVICES (placement verified: $([ "${PLACEMENT_OK:-1}" = 1 ] && echo 'YES ✓' || echo 'NO ✗'))"
row "GPU memory per GPU:"             "${GPU_MEM_MB} MiB"
row "Cold boot (run → health):"       "${T_COLD_BOOT} ms"
row "runsc checkpoint (incl. GPU):"   "${T_CHECKPOINT} ms"
row "runsc restore returned:"         "${T_RESTORE_RETURNED} ms"
row "Health after restore:"           "${T_HEALTH_MS} ms"
row "First matmul after restore:"     "${T_MATMUL} ms"
row "Checkpoint size (total):"        "$TOTAL_SIZE (pages: $PAGES_SIZE)"
row "GPU checksums match:"            "$([ "$VERIFY_OK" = 1 ] && echo 'YES ✓' || echo 'NO ✗')"
row "GPU compute after restore:"      "$([ "$MATMUL_OK" = 1 ] && echo 'YES ✓' || echo 'NO ✗')"
echo ""

if [[ "$VERIFY_OK" = "1" && "$MATMUL_OK" = "1" && "${PLACEMENT_OK:-1}" = "1" ]]; then
    banner "RESULT: PASS ✓"
    exit 0
else
    banner "RESULT: FAIL ✗"
    exit 1
fi
