#!/usr/bin/env bash
# run_nccl_mcshim_native.sh -- native e2e: STOCK (unpatched) NCCL NVLS,
# multi-process ranks, checkpointed via the mcshim LD_PRELOAD interposer.
#
# This is the decisive stock-NCCL validation for Idea D: the ranks run a real
# NCCL NVLS allreduce + captured CUDA graph and NEVER call ncclCommSuspend --
# the shim transparently tears down / rebuilds NCCL's multicast layer around
# cuda-checkpoint. NCCL is the stock build from `git archive HEAD` of nccl/
# (no NVLS-suspend patch).
#
#   CONTROL (CONTROL=1, default): stock NCCL, live NVLS, no shim
#       -> cuda-checkpoint checkpoint HANGS.
#   MAIN: pause (ranks idle) -> shim suspend (all ranks ack) -> per-pid
#       lock/checkpoint/restore/unlock -> shim resume -> unpause -> verified
#       eager allreduce + CUDA-graph replay on every rank.
#
# Usage:
#   sudo [WORLD=4] [CONTROL=1] bash run_nccl_mcshim_native.sh
set -uo pipefail
cd "$(dirname "$0")"
PHASE0_DIR=$(pwd)

CUDA_CHECKPOINT="${CUDA_CHECKPOINT:-/usr/local/bin/cuda-checkpoint}"
WORLD="${WORLD:-4}"
CONTROL="${CONTROL:-1}"
export NCCL_LIB="${NCCL_LIB:-/opt/phase0/nccl-stock/libnccl.so.2}"
export NCCL_NVLS_ENABLE=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
DIR=/tmp/nccl-mcshim-native
LOG=$DIR/launcher.log
SHIM=$PHASE0_DIR/mcshim/mcshim.so
log(){ echo "[nccl-mcshim $(date +%H:%M:%S)] $*"; }

[[ -x "$CUDA_CHECKPOINT" ]] || { log "cuda-checkpoint not found"; exit 1; }
[[ -f "$NCCL_LIB" ]] || { log "stock NCCL not found at $NCCL_LIB"; exit 1; }
[[ -f "$SHIM" ]] || { log "mcshim.so not built"; exit 1; }

rank_pids(){ pgrep -f "nccl_suspend_mp.py .*--world $WORLD" | sort -n; }

start(){  # $1 = "preload" | "plain"
  rm -rf "$DIR"; mkdir -p "$DIR"
  local env=(NCCL_NVLS_ENABLE=1 MCSHIM_DIR="$DIR" MCSHIM_LOG="$DIR/mcshim.log")
  [[ "$1" == "preload" ]] && env+=(LD_PRELOAD="$SHIM")
  # `--launch-job` (R610): the launcher + all rank children share one
  # cuda-checkpoint job, so cross-rank CUDA IPC state (the UC buffers NCCL
  # ranks import from each other) is checkpointed/restored coherently.
  # Without it, restore fails with "invalid argument" on every rank even
  # after the multicast layer is suspended. This matches production: gVisor
  # always wraps GPU containers this way (runsc/boot/loader.go).
  env "${env[@]}" nohup "$CUDA_CHECKPOINT" --launch-job \
    python3 nccl_mp_launcher.py --dir "$DIR" \
    --world "$WORLD" --graph --pause-only --interval 0.5 > "$LOG" 2>&1 &
  LPID=$!
  for i in $(seq 180); do
    grep -q ALL-READY "$LOG" 2>/dev/null && return 0
    kill -0 $LPID 2>/dev/null || { log "launcher died:"; tail -8 "$LOG"; return 1; }
    sleep 1
  done
  log "ranks never all READY"; tail -8 "$LOG"; return 1
}

stop(){
  kill $LPID 2>/dev/null
  local p; for p in $(rank_pids); do kill -9 "$p" 2>/dev/null; done
  sleep 3
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
wait_acks(){  # $1 prefix, $2 timeout: WORLD ack files from the shims
  local prefix="$1" timeout="$2" i n
  for ((i=0; i<timeout*2; i++)); do
    n=$(ls "$DIR/$prefix".* 2>/dev/null | wc -l)
    [[ "$n" -ge "$WORLD" ]] && return 0
    sleep 0.5
  done
  log "TIMEOUT: $n/$WORLD $prefix acks"; cat "$DIR"/error.* 2>/dev/null
  tail -20 "$DIR/mcshim.log" 2>/dev/null; return 1
}

cc_all(){  # action on every rank pid; SEQUENTIAL (job members require it)
  local act="$1" tmo="$2"; shift 2
  local pids; pids=$(rank_pids); local rc=0 p
  log "cuda-checkpoint $act (sequential) on pids: $(echo $pids | tr '\n' ' ')"
  for p in $pids; do
    timeout "$tmo" "$CUDA_CHECKPOINT" --action "$act" --pid "$p" "$@" \
      > "$DIR/cc.$act.$p" 2>&1 || rc=1
  done
  return $rc
}

FAIL=0

if [[ "$CONTROL" == "1" ]]; then
  log "=== CONTROL: stock NCCL, live NVLS, NO shim -- checkpoint should hang ==="
  start plain || exit 1
  wait_all "pass" 60 || exit 1
  cc_all lock 60 --timeout 30000 || true
  if cc_all checkpoint 60; then
    log "UNEXPECTED: checkpoint succeeded with live NVLS"
    cc_all restore 60 || true; cc_all unlock 30 || true
  else
    log "control confirmed: stock-NCCL live NVLS hangs the checkpoint"
    cc_all unlock 30 || true
  fi
  stop
fi

log "=== MAIN: stock NCCL + mcshim transparent suspend/resume ==="
start preload || exit 1
wait_all "pass" 60 || exit 1
log "NVLS engaged; sample: $(grep -m1 -oE 'WARMUP-OK.*' "$DIR"/status.0 2>/dev/null || head -c120 "$DIR"/status.0)"

log "(a) pause all ranks"
touch "$DIR/pause"
wait_all "PAUSED" 60 || { log "FAIL: pause"; stop; exit 1; }

log "(b) shim suspend on all ranks (stock NCCL never knows)"
touch "$DIR/suspend"
wait_acks suspended 120 || { log "FAIL: shim suspend"; stop; exit 1; }
log "all $WORLD shims SUSPENDED: $(grep -c 'SUSPEND done' "$DIR/mcshim.log") teardowns"

log "(c) cuda-checkpoint lock/checkpoint/restore/unlock on all rank pids"
cc_all lock 60 --timeout 30000 || { log "FAIL: lock"; FAIL=1; }
cc_all checkpoint 180         || { log "FAIL: checkpoint (with shim suspend!)"; FAIL=1; }
if [[ $FAIL -eq 0 ]]; then
  cc_all restore 180 || { log "FAIL: restore"; FAIL=1; }
  cc_all unlock 30   || { log "FAIL: unlock"; FAIL=1; }
fi

if [[ $FAIL -eq 0 ]]; then
  log "shim resume on all ranks"
  rm -f "$DIR/suspend"
  wait_acks resumed 180 || { log "FAIL: shim resume"; FAIL=1; }
fi

if [[ $FAIL -eq 0 ]]; then
  log "unpause + post-restore verification (eager + CUDA graph, every rank)"
  rm -f "$DIR/pause"
  wait_all "post-restore pass" 90 || { log "FAIL: post-restore verify"; FAIL=1; }
  sleep 4
  log "final per-rank status:"
  all_status | grep -oE '\[rank[0-9]+\] iter=[0-9]+ [a-z-]+ pass failures=[0-9]+' | sort -u
  for ((r=0; r<WORLD; r++)); do grep -q "failures=0" "$DIR/status.$r" 2>/dev/null || FAIL=1; done
fi

echo ""
log "==== SHIM LOG (multicast rendezvous) ===="
grep -hE 'SUSPEND done|RESUME done|re-established|serving|IDENTICAL' "$DIR/mcshim.log" 2>/dev/null | tail -20

stop
echo ""
log "==== RESULT: $([[ $FAIL -eq 0 ]] && echo PASS || echo FAIL) ===="
exit $FAIL
