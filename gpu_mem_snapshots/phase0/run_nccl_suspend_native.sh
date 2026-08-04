#!/usr/bin/env bash
# run_nccl_suspend_native.sh — NATIVE e2e of the NCCL suspend/resume
# checkpoint model, using upstream NCCL's ncclCommSuspend/ncclCommResume
# (>= 2.29.7, extended here to cover NVLS multicast) and cuda-checkpoint:
#
#   (a) 4-GPU NCCL clique (NVLS engaged), verified allreduce + CUDA GRAPH
#   (b) ncclCommSuspend(comm, NCCL_SUSPEND_MEM) on all comms   <- suspend
#   (c) cuda-checkpoint lock -> checkpoint -> restore -> unlock <- checkpoint
#       ncclCommResume on all comms                             <- resume
#       verified allreduce + graph replay must keep passing
#
# CONTROL=1 first runs the no-suspend control: with live NVLS, the checkpoint
# action must HANG (bounded by timeout) -- proving suspend is the enabler.
#
# Usage:
#   sudo [CUDA_CHECKPOINT=/usr/local/bin/cuda-checkpoint] [NGPUS=4] \
#        [CONTROL=1] bash run_nccl_suspend_native.sh
set -uo pipefail
cd "$(dirname "$0")"

CUDA_CHECKPOINT="${CUDA_CHECKPOINT:-/usr/local/bin/cuda-checkpoint}"
NGPUS="${NGPUS:-4}"
CONTROL="${CONTROL:-1}"
DIR=/tmp/nccl-suspend-native
LOG=$DIR/workload.log
log(){ echo "[nccl-native $(date +%H:%M:%S)] $*"; }

[[ -x "$CUDA_CHECKPOINT" ]] || { log "cuda-checkpoint not found"; exit 1; }
rm -rf "$DIR"; mkdir -p "$DIR"

start_workload(){
  rm -f "$DIR"/{status,suspend,resume,restored}
  NCCL_NVLS_ENABLE=1 nohup python3 nccl_suspend_workload.py \
    --ngpus "$NGPUS" --graph --dir "$DIR" --interval 0.5 > "$LOG" 2>&1 &
  WPID=$!
  for i in $(seq 120); do
    grep -q READY "$LOG" 2>/dev/null && return 0
    kill -0 $WPID 2>/dev/null || { log "workload died:"; tail -5 "$LOG"; return 1; }
    sleep 1
  done
  log "workload never became READY"; return 1
}

wait_status(){
  local pattern="$1" timeout="$2" i
  for ((i=0; i<timeout*2; i++)); do
    grep -q "$pattern" "$DIR/status" 2>/dev/null && { log "status: $(cat "$DIR/status")"; return 0; }
    sleep 0.5
  done
  log "TIMEOUT waiting for $pattern; last: $(cat "$DIR/status" 2>/dev/null)"
  return 1
}

cc_action(){
  local act="$1" tmo="$2"; shift 2
  local t0=$SECONDS out rc
  out=$(timeout "$tmo" "$CUDA_CHECKPOINT" --action "$act" --pid "$WPID" "$@" 2>&1)
  rc=$?
  log "cuda-checkpoint $act: rc=$rc ($((SECONDS-t0))s) out=[$out]"
  return $rc
}

FAIL=0

# ---------------- CONTROL: no suspend => checkpoint must hang ----------------
if [[ "$CONTROL" == "1" ]]; then
  log "=== CONTROL LEG: live NVLS, NO suspend -- checkpoint should hang ==="
  start_workload || exit 1
  wait_status "pass" 30 || exit 1
  cc_action lock 60 --timeout 30000 || { log "control lock failed"; }
  if cc_action checkpoint 60; then
    log "UNEXPECTED: checkpoint succeeded with live NVLS (no suspend)!"
    cc_action restore 60; cc_action unlock 30
  else
    log "control confirmed: checkpoint hangs/fails with live NVLS multicast"
    cc_action unlock 30 || true
  fi
  kill -9 $WPID 2>/dev/null; wait 2>/dev/null
  sleep 2
fi

# ---------------- MAIN LEG: suspend -> checkpoint -> restore -> resume -------
log "=== MAIN LEG: ncclCommSuspend -> cuda-checkpoint -> ncclCommResume ==="
start_workload || exit 1
wait_status "pass" 30 || exit 1

log "(b) ncclCommSuspend(NCCL_SUSPEND_MEM) on all comms"
touch "$DIR/suspend"
wait_status "SUSPENDED\|suspended (idle)" 30 || { log "FAIL: suspend"; exit 1; }
grep -m1 SUSPENDED "$LOG"

log "(c) cuda-checkpoint lock -> checkpoint -> restore -> unlock"
cc_action lock 60 --timeout 30000       || { log "FAIL: lock"; FAIL=1; }
cc_action checkpoint 120                || { log "FAIL: checkpoint (with suspend!)"; FAIL=1; }
if [[ $FAIL -eq 0 ]]; then
  cc_action restore 120                 || { log "FAIL: restore"; FAIL=1; }
  cc_action unlock 30                   || { log "FAIL: unlock"; FAIL=1; }
fi

if [[ $FAIL -eq 0 ]]; then
  log "ncclCommResume on all comms"
  touch "$DIR/restored" "$DIR/resume"
  # RESUMED is a transient single-line marker in the status file; grep the
  # append-only workload log for it instead.
  for i in $(seq 60); do grep -q RESUMED "$LOG" && break; sleep 1; done
  grep -m1 RESUMED "$LOG" || { log "FAIL: resume"; FAIL=1; }
  wait_status "post-restore pass" 60 || { log "FAIL: post-restore verification"; FAIL=1; }
  # several more verified iterations (eager + graph replay)
  sleep 5
  FINAL=$(cat "$DIR/status")
  log "final: $FINAL"
  grep -q "pass" <<<"$FINAL" && grep -q "failures=0" <<<"$FINAL" || FAIL=1
fi

kill -9 $WPID 2>/dev/null; wait 2>/dev/null

echo ""
if [[ $FAIL -eq 0 ]]; then
  log "==== RESULT: PASS ===="
  log "Native e2e: NVLS multicast workload (with captured CUDA graphs)"
  log "checkpoint/restores via ncclCommSuspend -> cuda-checkpoint -> ncclCommResume."
else
  log "==== RESULT: FAIL ===="
  tail -5 "$LOG"
fi
exit $FAIL
