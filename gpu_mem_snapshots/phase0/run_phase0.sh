#!/usr/bin/env bash
# run_phase0.sh — Phase 0 measurements for multicast suspend/restore in nvproxy.
#
# Runs NATIVELY on the host (no gVisor, no containers): the questions are about
# CUDA driver / cuda-checkpoint semantics, so gVisor must be out of the loop.
#
#   test 2 (attach_blocking): does cuMulticastAddDevice / cuMulticastBindMem
#           block until all participants join?     -> decides work item 4
#   test 1 (ipc_taint):       is a POSIX-FD-exported cuMemCreate allocation
#           checkpointable again after full release? -> go/no-go for the design
#           Runs a `hold` sensitivity control first, then the real `taint` leg.
#
# Measurement 3 (object-graph census) is gVisor instrumentation, exercised by
# any gVisor checkpoint with --cuda-checkpoint-path; see phase0/README.md.
#
# Usage:
#   sudo [CUDA_CHECKPOINT=/path/to/cuda-checkpoint] \
#        [GPUS="0 1"] [DELAY=8] [ONLY=taint|hold|attach] \
#        bash gpu_mem_snapshots/phase0/run_phase0.sh
set -uo pipefail
cd "$(dirname "$0")"

GPUS=(${GPUS:-0 1})
DELAY="${DELAY:-8}"
ONLY="${ONLY:-}"
log(){ echo "[phase0 $(date +%H:%M:%S)] $*"; }

# --- locate cuda-checkpoint (needed by test 1 only) -------------------------
find_cuda_checkpoint(){
  if [[ -n "${CUDA_CHECKPOINT:-}" ]]; then echo "$CUDA_CHECKPOINT"; return; fi
  local c
  c=$(command -v cuda-checkpoint 2>/dev/null) && { echo "$c"; return; }
  for c in /usr/local/bin/cuda-checkpoint /usr/local/cuda/bin/cuda-checkpoint \
           "$HOME/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint"; do
    [[ -x "$c" ]] && { echo "$c"; return; }
  done
  echo ""
}
CC_BIN=$(find_cuda_checkpoint)

declare -A RC

run_leg(){
  local name="$1"; shift
  log "=============================================================="
  log "RUN: $name  ->  $*"
  log "=============================================================="
  "$@" 2>&1 | sed "s/^/  /"
  RC[$name]=${PIPESTATUS[0]}
  log "$name: rc=${RC[$name]}"
}

# --- test 2: attach blocking (no cuda-checkpoint needed) ---------------------
if [[ -z "$ONLY" || "$ONLY" == attach ]]; then
  run_leg attach_blocking python3 attach_blocking.py \
    --gpu "${GPUS[0]}" --peer-gpu "${GPUS[1]}" --delay "$DELAY"
fi

# --- test 1: IPC taint (control leg, then real leg) --------------------------
if [[ -z "$ONLY" || "$ONLY" == hold || "$ONLY" == taint ]]; then
  if [[ -z "$CC_BIN" ]]; then
    log "SKIP ipc_taint: cuda-checkpoint not found."
    log "  Get it from https://github.com/NVIDIA/cuda-checkpoint (bin/x86_64_Linux/)"
    log "  then re-run with CUDA_CHECKPOINT=/path/to/cuda-checkpoint"
    RC[ipc_taint_hold]=127; RC[ipc_taint]=127
  else
    log "using cuda-checkpoint: $CC_BIN"
    if [[ -z "$ONLY" || "$ONLY" == hold ]]; then
      run_leg ipc_taint_hold python3 ipc_taint.py --mode hold \
        --gpu "${GPUS[0]}" --peer-gpu "${GPUS[1]}" --cuda-checkpoint "$CC_BIN"
    fi
    if [[ -z "$ONLY" || "$ONLY" == taint ]]; then
      run_leg ipc_taint python3 ipc_taint.py --mode taint \
        --gpu "${GPUS[0]}" --peer-gpu "${GPUS[1]}" --cuda-checkpoint "$CC_BIN"
    fi
  fi
fi

# --- summary -----------------------------------------------------------------
echo ""
log "==== PHASE 0 SUMMARY ===="
for k in attach_blocking ipc_taint_hold ipc_taint; do
  [[ -v RC[$k] ]] || continue
  case "${RC[$k]}" in
    0)   v="OK (see VERDICT above)";;
    3)   v="WATCHDOG HANG";;
    127) v="SKIPPED (missing cuda-checkpoint)";;
    *)   v="rc=${RC[$k]}";;
  esac
  printf '  %-18s %s\n' "$k" "$v"
done
log "Record verdicts in gpu_mem_snapshots/phase0/README.md 'Results'."
