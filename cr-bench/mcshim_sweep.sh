#!/usr/bin/env bash
# mcshim_sweep.sh — exploratory sweep of the vLLM parameter cells the
# multicast interposer has never been run against.
#
# The interposer is the only mechanism that covers every multicast owner, so
# the question for it is coverage, not capability: vLLM 0.27 can create
# multicast from four places -- NCCL NVLS, custom all-reduce, torch symmetric
# memory, and the FlashInfer all-reduce backend -- and only the first two are
# on in the benchmark's default configuration.
#
# One run per cell, TP=2 for speed. This is a search for breakage, not a
# measurement: any cell that fails is then worth repeating under
# vllm_trials.sh to separate a real interposer gap from cuda-checkpoint's
# intermittent restore-toggle bug.
#
# Usage: sudo [TP=2] [GPUS=0,1] bash mcshim_sweep.sh [cell ...]
set -uo pipefail
cd "$(dirname "$0")"

TP="${TP:-2}"
GPUS="${GPUS:-0,1}"
export RUNSC="${RUNSC:-/usr/local/bin/runsc-phase0}"
export CUDA_CKPT_JOB_FILE=1 CUDA_CKPT_SEQUENTIAL=1 CUDA_MULTICAST_SHIM=1

# Each cell is a name plus the environment that distinguishes it from the
# benchmark defaults (custom all-reduce on, cuMem off, sleep level 1).
declare -A CELLS=(
  [baseline]=""
  [symmmem]="VLLM_ALLREDUCE_USE_SYMM_MEM=1"
  [flashinfer]="VLLM_ALLREDUCE_USE_FLASHINFER=1"
  [cumem]="NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=1"
  [sleep2]="SLEEP_LEVEL=2"
  [allon]="VLLM_ALLREDUCE_USE_SYMM_MEM=1 NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=1"
)
ORDER=(baseline symmmem flashinfer cumem sleep2 allon)
[[ $# -gt 0 ]] && ORDER=("$@")

reap() {
    local pid root cid
    for pid in $(ps -eo pid,cmd | awk '/[r]unsc-sandbox/{print $1}'); do
        root=$(tr '\0' '\n' < /proc/$pid/cmdline 2>/dev/null | grep -m1 -oE '^--root=.*' | cut -d= -f2)
        [[ -n "$root" ]] || continue
        cid=$("$RUNSC" --root "$root" list 2>/dev/null | awk 'NR>1{print $1}' | head -1)
        [[ -n "$cid" ]] && "$RUNSC" --root "$root" delete -force "$cid" >/dev/null 2>&1
    done
    sleep 5
}

printf '%-12s %-8s %-10s %-10s %s\n' CELL RESULT CKPT RESTORE NOTE
for cell in "${ORDER[@]}"; do
    env_str="${CELLS[$cell]-}"
    log=$(mktemp /tmp/mcsweep.$cell.XXXXXX.log)
    # shellcheck disable=SC2086
    env $env_str timeout 1800 bash ./bench_4_vllm_multi.sh --gpus "$GPUS" --tp "$TP" \
        >"$log" 2>&1
    art=$(grep -ohE '==> Artifacts at .*' "$log" | tail -1 | awk '{print $NF}')

    ck=$(grep -ohE 'runsc checkpoint \(incl. GPU\): +[0-9]+ ms' "$log" | grep -oE '[0-9]+ ms' | head -1)
    rs=$(grep -ohE 'runsc restore returned: +[0-9]+ ms' "$log" | grep -oE '[0-9]+ ms' | head -1)

    note=""
    if grep -q "RESULT: PASS" "$log"; then
        res=PASS
        # How many multicast owners did the interposer actually handle? If a
        # cell adds an owner, this is where it shows up.
        note=$(grep -ohE 'SUSPEND done: groups=[0-9]+ imports=[0-9]+' "$log" | head -1)
        [[ -z "$note" && -n "$art" ]] && note=$(grep -ohE 'SUSPEND done: groups=[0-9]+ imports=[0-9]+' "$art"/applog/* 2>/dev/null | head -1)
    elif grep -qE 'cannot proceed: .* multicast' "$log"; then
        res=FAIL
        note="BLOCKER: $(grep -ohE 'still live after [^:]*: .*' "$log" | tail -1 | cut -c1-60)"
    elif [[ -n "$art" ]] && grep -qlE 'restore toggle failed|Error toggling CUDA' "$art"/logs/*boot* 2>/dev/null; then
        res=FAIL
        note="cuda-checkpoint restore toggle (pre-existing, upstream)"
    else
        res=FAIL
        note="other: $(grep -ohE 'FAIL:.*|Container exited.*|wake_up response: .*' "$log" | tail -1 | cut -c1-60)"
    fi
    printf '%-12s %-8s %-10s %-10s %s\n' "$cell" "$res" "${ck:-–}" "${rs:-–}" "$note"
    rm -f "$log"
    reap
done
