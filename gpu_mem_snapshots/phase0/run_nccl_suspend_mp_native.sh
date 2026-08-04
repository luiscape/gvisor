#!/usr/bin/env bash
# run_nccl_suspend_mp_native.sh — native e2e of NCCL suspend/resume with
# MULTI-PROCESS ranks (one process per GPU, like vLLM/SGLang tensor-parallel),
# under a single cuda-checkpoint --launch-job.
#
#   (a) launcher forks WORLD ranks (nccl_suspend_mp.py); NVLS clique + verified
#       allreduce + captured CUDA graph
#   (b) suspend marker -> every rank calls ncclCommSuspend(NCCL_SUSPEND_MEM)
#   (c) cuda-checkpoint lock -> checkpoint -> restore -> unlock on EVERY rank pid
#       resume marker -> every rank calls ncclCommResume; verification continues
#
# CONTROL=1 first proves live NVLS hangs the checkpoint (no suspend).
#
# Usage:
#   sudo NCCL_LIB=/opt/phase0/nccl/nvidia/nccl/lib/libnccl.so.2 \
#        [WORLD=4] [CONTROL=1] bash run_nccl_suspend_mp_native.sh
set -uo pipefail
cd "$(dirname "$0")"

CUDA_CHECKPOINT="${CUDA_CHECKPOINT:-/usr/local/bin/cuda-checkpoint}"
WORLD="${WORLD:-4}"
CONTROL="${CONTROL:-1}"
export NCCL_LIB="${NCCL_LIB:-/opt/phase0/nccl/nvidia/nccl/lib/libnccl.so.2}"
export NCCL_NVLS_ENABLE=1
# Multi-process NCCL bootstraps rank<->rank over a socket interface (unlike
# the single-process ncclCommInitAll clique). All ranks are on this node, so
# loopback works.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
DIR=/tmp/nccl-mp-native
LOG=$DIR/launcher.log
log(){ echo "[mp-native $(date +%H:%M:%S)] $*"; }

[[ -x "$CUDA_CHECKPOINT" ]] || { log "cuda-checkpoint not found"; exit 1; }

rank_pids(){ pgrep -f "nccl_suspend_mp.py .*--world $WORLD" | sort -n; }

start(){
  rm -rf "$DIR"; mkdir -p "$DIR"
  NCCL_NVLS_ENABLE=1 nohup python3 nccl_mp_launcher.py --dir "$DIR" \
    --world "$WORLD" --graph --interval 0.5 > "$LOG" 2>&1 &
  LPID=$!
  for i in $(seq 180); do
    grep -q ALL-READY "$LOG" 2>/dev/null && return 0
    kill -0 $LPID 2>/dev/null || { log "launcher died:"; tail -8 "$LOG"; return 1; }
    sleep 1
  done
  log "ranks never all READY"; tail -8 "$LOG"; return 1
}

all_status(){ cat "$DIR"/status.* 2>/dev/null; }
wait_all(){  # pattern present in ALL rank status files
  local pattern="$1" timeout="$2" i r ok
  for ((i=0; i<timeout*2; i++)); do
    ok=1
    for ((r=0; r<WORLD; r++)); do
      grep -q "$pattern" "$DIR/status.$r" 2>/dev/null || { ok=0; break; }
    done
    [[ $ok -eq 1 ]] && return 0
    sleep 0.5
  done
  log "TIMEOUT waiting all ranks for '$pattern'"; all_status | tail -$WORLD; return 1
}

cc_all(){  # action on every rank pid; parallel; returns 1 if any timed out/failed
  local act="$1" tmo="$2"; shift 2
  local pids; pids=$(rank_pids); local rc=0
  log "cuda-checkpoint $act on pids: $(echo $pids | tr '\n' ' ')"
  local declpids=(); for p in $pids; do
    timeout "$tmo" "$CUDA_CHECKPOINT" --action "$act" --pid "$p" "$@" \
      > "$DIR/cc.$act.$p" 2>&1 &
    declpids+=($!)
  done
  local j; for j in "${declpids[@]}"; do wait "$j" || rc=1; done
  for p in $pids; do log "  pid $p: rc-file=$(cat "$DIR/cc.$act.$p" 2>/dev/null | head -1)"; done
  return $rc
}

FAIL=0

if [[ "$CONTROL" == "1" ]]; then
  log "=== CONTROL: live NVLS, NO suspend -- checkpoint should hang ==="
  start || exit 1
  wait_all "pass" 60 || exit 1
  cc_all lock 60 --timeout 30000 || true
  if cc_all checkpoint 60; then
    log "UNEXPECTED: checkpoint succeeded with live NVLS"
    cc_all restore 60 || true; cc_all unlock 30 || true
  else
    log "control confirmed: checkpoint hangs with live NVLS multicast"
    cc_all unlock 30 || true
  fi
  kill $LPID 2>/dev/null; pkill -9 -f "nccl_suspend_mp.py .*--world $WORLD" 2>/dev/null; sleep 3
fi

log "=== MAIN: multi-process ncclCommSuspend -> cuda-checkpoint -> resume ==="
start || exit 1
wait_all "pass" 60 || exit 1

log "(b) suspend all ranks (ncclCommSuspend)"
touch "$DIR/suspend"
wait_all "suspended (idle)" 60 || { log "FAIL: suspend"; FAIL=1; }

log "(c) cuda-checkpoint lock/checkpoint/restore/unlock on all rank pids"
cc_all lock 60 --timeout 30000 || { log "FAIL: lock"; FAIL=1; }
cc_all checkpoint 120         || { log "FAIL: checkpoint (with suspend!)"; FAIL=1; }
if [[ $FAIL -eq 0 ]]; then
  cc_all restore 120 || { log "FAIL: restore"; FAIL=1; }
  cc_all unlock 30   || { log "FAIL: unlock"; FAIL=1; }
fi

if [[ $FAIL -eq 0 ]]; then
  log "resume all ranks (ncclCommResume)"
  touch "$DIR/restored" "$DIR/resume"
  for i in $(seq 60); do grep -q RESUMED "$LOG" "$DIR"/status.* 2>/dev/null && break; sleep 1; done
  wait_all "post-restore pass" 90 || { log "FAIL: post-restore verification"; FAIL=1; }
  sleep 4
  log "final per-rank status:"; all_status | grep -oE '\[rank[0-9]+\] iter=[0-9]+ [a-z-]+ pass failures=[0-9]+' | sort -u
  for ((r=0; r<WORLD; r++)); do grep -q "failures=0" "$DIR/status.$r" 2>/dev/null || FAIL=1; done
fi

kill $LPID 2>/dev/null; pkill -9 -f "nccl_suspend_mp.py .*--world $WORLD" 2>/dev/null
echo ""
log "==== RESULT: $([[ $FAIL -eq 0 ]] && echo PASS || echo FAIL) ===="
exit $FAIL
