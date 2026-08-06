#!/usr/bin/env bash
# native_ab.sh — native runc + cuda-checkpoint A/B for the fabric-free vLLM TP
# restore failure.
#
# Runs the SAME vLLM image (cr-bench-vllm) NATIVELY (docker --runtime=nvidia,
# NO gVisor), wrapped in `cuda-checkpoint --launch-job`, with the exact
# fabric-free config the failing gVisor runs used. Then drives cuda-checkpoint
# lock -> checkpoint -> restore -> unlock on every CUDA pid, mirroring gVisor's
# pkg/sentry/control/state_cuda.go phased flow -- but with NO CRIU serialize /
# nvproxy replay in between (the process stays alive the whole time).
#
# Decision:
#   restore FAILS here too  -> blocker is cuda-checkpoint (NOT gVisor-fixable)
#   restore SUCCEEDS here    -> gVisor serialize/replay introduces the failure
#
# Usage: sudo bash gpu_mem_snapshots/native_ab.sh
set -uo pipefail

NAME=vllm-native-ab
IMAGE=cr-bench-vllm
PORT=8000
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
GPUS="0,1"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"
# EAGER=1 adds --enforce-eager (no torch.compile / no piecewise CUDA graphs),
# to bisect whether the compiled/graph state is what cuda-checkpoint cannot
# restore.
EAGER="${EAGER:-0}"
EXTRA_ARGS="--enable-sleep-mode --disable-custom-all-reduce"
[[ "$EAGER" = "1" ]] && EXTRA_ARGS+=" --enforce-eager"
log(){ echo "[native-ab $(date +%H:%M:%S)] $*"; }
log "config: EAGER=$EAGER  extra_args='$EXTRA_ARGS'"

cleanup(){ docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker rm -f "$NAME" >/dev/null 2>&1 || true

log "launching native vLLM TP=2 (fabric-free, cuda-checkpoint --launch-job wrap)"
docker run -d --name "$NAME" \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES="$GPUS" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  --shm-size=8g \
  -e HF_HOME=/app/hf_cache -e HF_HUB_OFFLINE=1 \
  -e VLLM_USAGE_SOURCE=production \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_NVLS_ENABLE=0 \
  -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib \
  --entrypoint /usr/local/bin/cuda-checkpoint \
  "$IMAGE" \
  --launch-job python3 /app/vllm_sleep_server.py \
    --model "$MODEL" --host 0.0.0.0 --port "$PORT" \
    --gpu-memory-utilization 0.7 --max-model-len 2048 \
    --tensor-parallel-size 2 --distributed-executor-backend mp \
    --dtype float16 $EXTRA_ARGS \
  || { log "docker run failed"; exit 1; }

log "waiting for /health (timeout ${HEALTH_TIMEOUT}s)"
healthy=0
for ((i=0; i<HEALTH_TIMEOUT; i+=5)); do
  if ! docker ps -q -f name="$NAME" | grep -q .; then
    log "CONTAINER EXITED during boot; last logs:"; docker logs --tail 60 "$NAME"; exit 1
  fi
  if docker exec "$NAME" curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    healthy=1; log "healthy after ~${i}s"; break
  fi
  sleep 5
done
[[ "$healthy" = 1 ]] || { log "NOT healthy in ${HEALTH_TIMEOUT}s; last logs:"; docker logs --tail 60 "$NAME"; exit 1; }

infer(){
  docker exec "$NAME" curl -s --max-time 60 -X POST \
    "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer with just the city name.\"}],\"temperature\":0.0,\"max_tokens\":16,\"seed\":42}" 2>/dev/null
}

log "reference inference:"; REF=$(infer); echo "  REF = $REF"

log "POST /sleep?level=1 (quiesce engine, offload weights, drop KV cache)"
docker exec "$NAME" curl -s --max-time 120 -X POST "http://127.0.0.1:$PORT/sleep?level=1" 2>/dev/null; echo

log "driving cuda-checkpoint lock->checkpoint->restore->unlock IN-CONTAINER (per pid, parallel)"
docker exec "$NAME" sh -c '
CC=/usr/local/bin/cuda-checkpoint
PIDS=$(for d in /proc/[0-9]*; do p=${d#/proc/}; $CC --get-state --pid "$p" 2>/dev/null | grep -q running && echo "$p"; done | sort -un)
echo "  CUDA pids (running): $PIDS"
[ -z "$PIDS" ] && { echo "  NO CUDA pids found"; exit 3; }
phase(){
  action="$1"; shift; extra="$*"
  echo "== phase: $action $extra =="
  for p in $PIDS; do ( $CC --action "$action" $extra --pid "$p" >/tmp/o.$p 2>&1; echo $? >/tmp/r.$p ) & done
  wait
  fail=0
  for p in $PIDS; do
    rc=$(cat /tmp/r.$p 2>/dev/null); out=$(cat /tmp/o.$p 2>/dev/null)
    echo "   pid $p: rc=$rc out=[$out]"
    [ "$rc" != 0 ] && fail=1
  done
  return $fail
}
phase lock --timeout 30000; echo "  >>> lock_rc=$?"
phase checkpoint;           echo "  >>> checkpoint_rc=$?"
phase restore;              echo "  >>> restore_rc=$?   (KEY STEP)"
phase unlock;               echo "  >>> unlock_rc=$?"
'
CYCLE_RC=$?
log "cuda-checkpoint cycle exec rc=$CYCLE_RC"

log "POST /wake_up (resume engine)"
docker exec "$NAME" curl -s --max-time 120 -X POST "http://127.0.0.1:$PORT/wake_up" 2>/dev/null; echo

log "post-restore inference:"; POST=$(infer); echo "  POST = $POST"

echo ""
log "==== SUMMARY ===="
echo "  REF  = $REF"
echo "  POST = $POST"
