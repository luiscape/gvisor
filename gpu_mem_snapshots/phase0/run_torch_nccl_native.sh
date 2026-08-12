#!/usr/bin/env bash
# run_torch_nccl_native.sh — native (no gVisor) A/B of the NCCL NVLS
# suspend/resume patch driven by NCCL's own checkpoint control thread, with a
# PyTorch workload that never calls the NCCL API itself.
#
# This is the fast CONTROL/MAIN pair that must pass before the gVisor tier is
# worth running:
#
#   CONTROL (NCCL_CKPT_CTRL_DIR unset): live NVLS multicast, no suspend
#            -> `cuda-checkpoint --action checkpoint` HANGS
#   MAIN    (control thread enabled): marker -> every rank's NCCL suspends
#            -> checkpoint/restore succeed -> marker removed -> resume
#            -> the workload keeps verifying, having never paused itself
#
# torch lives in the benchmark image, so this runs inside that image with
# docker (native runtime), the patched libnccl preloaded over the bundled one.
#
# Usage:
#   sudo [WORLD=4] [CONTROL=1] [NCCL_LIB=/opt/phase0/nccl-patched/libnccl.so.2] \
#        bash run_torch_nccl_native.sh
set -uo pipefail
cd "$(dirname "$0")"
PHASE0=$(pwd)

WORLD="${WORLD:-4}"
CONTROL="${CONTROL:-1}"
IMAGE="${IMAGE:-cr-bench-vllm}"
NCCL_LIB="${NCCL_LIB:-/opt/phase0/nccl-patched/libnccl.so.2}"
CUDA_CHECKPOINT="${CUDA_CHECKPOINT:-/usr/local/bin/cuda-checkpoint}"
NAME=torch-nccl-native
DIR=/tmp/torchnccl
log(){ echo "[torch-native $(date +%H:%M:%S)] $*"; }

[[ -f "$NCCL_LIB" ]] || { log "patched NCCL not found at $NCCL_LIB"; exit 1; }

cleanup(){ sudo docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

GPUS=$(seq -s, 0 $((WORLD-1)))

# CTRL_DIR empty => control thread disabled => CONTROL leg.
CTRL_ENV=()
[[ "${LEG:-main}" == "main" ]] && CTRL_ENV=(-e NCCL_CKPT_CTRL_DIR="$DIR")

log "launching WORLD=$WORLD torch ranks (leg=${LEG:-main}) in $IMAGE"
sudo docker run -d --name "$NAME" \
  --runtime=nvidia --shm-size=8g --ipc=host \
  -e NVIDIA_VISIBLE_DEVICES="$GPUS" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e NCCL_NVLS_ENABLE=1 \
  -e NCCL_DEBUG=WARN \
  -e LD_PRELOAD=/opt/nccl-patched/libnccl.so.2 \
  "${CTRL_ENV[@]}" \
  -v "$PHASE0":/phase0:ro \
  -v "$(dirname "$NCCL_LIB")":/opt/nccl-patched:ro \
  --entrypoint python3 \
  "$IMAGE" /phase0/torch_nccl_launcher.py --dir "$DIR" --world "$WORLD" \
  || { log "docker run failed"; exit 1; }

log "waiting for ALL-READY"
for i in $(seq 300); do
  sudo docker logs "$NAME" 2>&1 | grep -q ALL-READY && break
  sudo docker ps -q -f name="$NAME" | grep -q . || { log "container exited:"; sudo docker logs --tail 30 "$NAME"; exit 1; }
  sleep 2
done
sudo docker logs "$NAME" 2>&1 | grep -q ALL-READY || { log "timed out"; sudo docker logs --tail 30 "$NAME"; exit 1; }
log "ranks ready"
sudo docker exec "$NAME" sh -c "cat $DIR/status.0" 2>/dev/null

# Confirm NVLS actually engaged; otherwise the test proves nothing.
if sudo docker exec "$NAME" sh -c "grep -l . $DIR/status.* >/dev/null 2>&1"; then :; fi

pids(){ sudo docker exec "$NAME" sh -c 'for d in /proc/[0-9]*; do p=${d#/proc/}; '"$CUDA_CHECKPOINT"' --get-state --pid $p 2>/dev/null | grep -q running && echo $p; done | sort -un'; }
CUDA_PIDS=$(pids)
log "CUDA pids: $(echo $CUDA_PIDS | tr '\n' ' ')"

# Quiesce the application first. ncclCommSuspend copies UC contents to a CPU
# backup and cannot run under a live collective; NCCL's gate stops new API
# calls but not replays of a captured CUDA graph, so the app stops itself.
log "pausing the workload"
sudo docker exec "$NAME" touch "$DIR/pause"
for i in $(seq 60); do
  n=$(sudo docker exec "$NAME" sh -c "grep -l PAUSED $DIR/status.* 2>/dev/null | wc -l")
  [[ "$n" -ge "$WORLD" ]] && break
  sleep 1
done
log "paused: $(sudo docker exec "$NAME" sh -c "grep -l PAUSED $DIR/status.* 2>/dev/null | wc -l")/$WORLD ranks"

if [[ "${LEG:-main}" == "main" ]]; then
  log "requesting suspend (marker) -- NCCL control thread does the work"
  sudo docker exec "$NAME" touch "$DIR/suspend"
  for i in $(seq 120); do
    n=$(sudo docker exec "$NAME" sh -c "ls $DIR/suspended.* 2>/dev/null | wc -l")
    [[ "$n" -ge "$WORLD" ]] && break
    sleep 1
  done
  log "suspended acks: $(sudo docker exec "$NAME" sh -c "ls $DIR/suspended.* 2>/dev/null | wc -l")/$WORLD"
fi

phase(){
  local action="$1" extra="${2:-}" rc=0
  log "cuda-checkpoint --action $action"
  for p in $CUDA_PIDS; do
    out=$(sudo docker exec "$NAME" timeout 60 "$CUDA_CHECKPOINT" --action "$action" $extra --pid "$p" 2>&1); r=$?
    [[ $r -ne 0 ]] && { log "  pid $p rc=$r out=[$out]"; rc=1; }
  done
  return $rc
}

phase lock "--timeout 30000"; log "  lock rc=$?"
phase checkpoint;             CK=$?; log "  checkpoint rc=$CK"
if [[ $CK -ne 0 ]]; then
  log "CHECKPOINT FAILED/HUNG (expected for the CONTROL leg)"
  [[ "${LEG:-main}" == "control" ]] && { log "==== RESULT: CONTROL confirmed (live NVLS blocks checkpoint) ===="; exit 0; }
  exit 1
fi
phase restore; log "  restore rc=$?"
phase unlock;  log "  unlock rc=$?"

if [[ "${LEG:-main}" == "main" ]]; then
  log "requesting resume (remove marker)"
  sudo docker exec "$NAME" rm -f "$DIR/suspend"
  for i in $(seq 120); do
    n=$(sudo docker exec "$NAME" sh -c "ls $DIR/resumed.* 2>/dev/null | wc -l")
    [[ "$n" -ge "$WORLD" ]] && break
    sleep 1
  done
  log "resumed acks: $(sudo docker exec "$NAME" sh -c "ls $DIR/resumed.* 2>/dev/null | wc -l")/$WORLD"
fi

log "unpausing the workload"
sudo docker exec "$NAME" rm -f "$DIR/pause"
sleep 8
log "final per-rank status:"
FAIL=0
for r in $(seq 0 $((WORLD-1))); do
  s=$(sudo docker exec "$NAME" sh -c "cat $DIR/status.$r" 2>/dev/null)
  echo "  rank$r: $s"
  grep -q 'failures=0' <<<"$s" || FAIL=1
  grep -q 'pass' <<<"$s" || FAIL=1
done
log "==== RESULT: $([[ $FAIL -eq 0 ]] && echo PASS || echo FAIL) ===="
exit $FAIL
