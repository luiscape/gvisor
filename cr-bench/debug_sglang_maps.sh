#!/usr/bin/env bash
# Throwaway debug harness: boot SGLang under runsc, list nvidia device
# mappings per process, toggle all CUDA procs with cuda-checkpoint (like
# the sentry would), and list which nvidia mappings SURVIVE the toggle.
# Those survivors are what make `runsc checkpoint` panic with
# "Can't save pma with non-MemoryFile of type *nvproxy.frontendFDMemmapFile".
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

BENCH_NAME="cr-bench-sglang-dbg"
IMAGE="cr-bench-sglang"
DOCKERFILE="images/Dockerfile.sglang"
PORT=8199
CB_GPU=1
GPU_DEVICES="${GPU_DEVICES:-0}"
TP="${TP:-1}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
APP_LOG="/applog/sglang.log"

cb_init
cb_detect_gpu
cb_runsc_flags
cb_prepare_rootfs

CB_CMD="exec python3 -m sglang.launch_server \
--model-path $MODEL --host 0.0.0.0 --port $PORT --tp-size $TP \
--mem-fraction-static 0.7 --context-length 2048 --dtype float16 \
--attention-backend triton --sampling-backend pytorch --disable-cuda-graph \
>$APP_LOG 2>&1"
CB_ENV="HF_HOME=/app/hf_cache
HF_HUB_OFFLINE=1
NCCL_CUMEM_ENABLE=0"
cb_write_bundle

cb_run_and_wait_health

# Only DEVICE mappings (/dev/nvidia*) matter: file mappings of nvidia .so
# libraries are gofer-backed and savable.  Device mappings are
# frontendFDMemmapFile/UVM-backed and panic the save if they survive.
list_nvidia_maps() {
    _rexec "$CONTAINER_ID" /bin/sh -c '
for m in /proc/[0-9]*/maps; do
  p=${m%/maps}
  if grep -q "/dev/nvidia" "$m" 2>/dev/null; then
    echo "== PID ${p#/proc/} ($(cat $p/comm 2>/dev/null))"
    grep "/dev/nvidia" "$m" | awk "{print \$2, \$6}" | sort | uniq -c
  fi
done
true'
}

info "=== inference (like the benchmark does before checkpoint) ==="
_rexec "$CONTAINER_ID" /usr/bin/curl -s --max-time 60 \
    -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":5}" || true
echo ""
sleep 1

info "=== nvidia maps BEFORE toggle ==="
list_nvidia_maps

info "=== identify processes with /dev/nvidia mappings ==="
_rexec "$CONTAINER_ID" /bin/sh -c '
for m in /proc/[0-9]*/maps; do
  p=${m%/maps}; pid=${p#/proc/}
  if grep -q "/dev/nvidia" "$m" 2>/dev/null; then
    ppid=$(awk "{print \$4}" $p/stat 2>/dev/null)
    nvfds=$(ls -l $p/fd 2>/dev/null | grep -c nvidia)
    echo "PID $pid ppid=$ppid nvidia_fds=$nvfds cmd=$(tr \"\\0\" \" \" < $p/cmdline | cut -c1-120)"
  fi
done
true'

info "=== pause_generation ==="
_rexec "$CONTAINER_ID" /usr/bin/curl -s --max-time 30 -X POST \
    -H "Content-Type: application/json" -d '{}' \
    "http://127.0.0.1:${PORT}/pause_generation" || true
echo ""
sleep 1

info "=== toggling all CUDA procs (parallel, like the sentry) ==="
CUDA_PIDS=$(_rexec "$CONTAINER_ID" /bin/sh -c '
for m in /proc/[0-9]*/maps; do
  p=${m%/maps}; pid=${p#/proc/}
  grep -q "/dev/nvidia" "$m" 2>/dev/null && echo "$pid"
done
true' || true)
echo "candidate PIDs: $CUDA_PIDS"
for pid in $CUDA_PIDS; do
    state=$(_rexec "$CONTAINER_ID" /usr/local/bin/cuda-checkpoint --get-state --pid "$pid" 2>&1 || true)
    echo "PID $pid state: $state"
done
TOGGLE_PIDS=""
for pid in $CUDA_PIDS; do
    state=$(_rexec "$CONTAINER_ID" /usr/local/bin/cuda-checkpoint --get-state --pid "$pid" 2>&1 || true)
    [[ "$state" == "running" ]] && TOGGLE_PIDS="$TOGGLE_PIDS $pid"
done
echo "toggling: $TOGGLE_PIDS"
for pid in $TOGGLE_PIDS; do
    _rexec "$CONTAINER_ID" /usr/local/bin/cuda-checkpoint --toggle --pid "$pid" &
done
wait
ok "toggle done"

info "=== nvidia maps AFTER toggle (survivors cause the save panic) ==="
list_nvidia_maps

info "=== waiting 15s (background threads may create NEW device mappings) ==="
sleep 15
info "=== nvidia maps 15s AFTER toggle ==="
list_nvidia_maps

info "=== runsc checkpoint WITHOUT --cuda-checkpoint-path (procs already toggled) ==="
rc=0
"$RUNSC" "${RUNSC_FLAGS[@]}" checkpoint --image-path="$CKPT_DIR" --compression=none \
    --exclude-committed-zero-pages --leave-running "$CONTAINER_ID" \
    >"$LOG_DIR/runsc-checkpoint.log" 2>&1 || rc=$?
if [[ "$rc" -eq 0 ]]; then
    ok "manual-toggle checkpoint SUCCEEDED — sentry exec of cuda-checkpoint is the suspect"
else
    fail "manual-toggle checkpoint FAILED too (exit $rc) — mapping exists regardless of who toggles"
    grep -m1 "Can't save pma\|panic" "$LOG_DIR/runsc-checkpoint.log" || true
fi

info "=== toggling back ==="
for pid in $TOGGLE_PIDS; do
    _rexec "$CONTAINER_ID" /usr/local/bin/cuda-checkpoint --toggle --pid "$pid" &
done
wait
ok "restored"
