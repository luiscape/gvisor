#!/usr/bin/env bash
# vllm_trials.sh -- repeat a vLLM checkpoint/restore configuration and report a
# pass rate, classifying each failure.
#
# The dominant remaining failure mode is cuda-checkpoint's own restore toggle
# returning "unknown error" on a tensor-parallel worker, which happens before
# gVisor's multicast rebuild runs. That is PROGRESS.md finding #3/#4, reproduced
# natively under runc with no gVisor in the loop. Classifying it separately keeps
# it from being confused with an interposer problem.
#
# MECH selects which mechanism releases the multicast layer. Both are driven by
# the same gVisor orchestration, so this is a like-for-like comparison:
#   mcshim (default) -- generic libcuda interposer; covers every multicast
#                       owner (NCCL NVLS, vLLM custom all-reduce, torch
#                       symmetric memory)
#   nccl             -- patched libnccl whose ncclCommSuspend/Resume also
#                       release the NVLS layer, plus NCCL's control thread.
#                       Covers only multicast NCCL owns, so the other owners
#                       must be disabled (done automatically below).
#
# Usage: sudo [TRIALS=3] [TP=2] [GPUS=0,1] [EAGER=0] [MECH=mcshim|nccl] \
#             bash vllm_trials.sh
set -uo pipefail
cd "$(dirname "$0")"

TRIALS="${TRIALS:-3}"
TP="${TP:-2}"
GPUS="${GPUS:-0,1}"
MECH="${MECH:-mcshim}"
export EAGER="${EAGER:-0}"
export RUNSC="${RUNSC:-/usr/local/bin/runsc-phase0}"
export CUDA_CKPT_JOB_FILE=1
case "$MECH" in
  mcshim) export CUDA_MULTICAST_SHIM=1 NCCL_CKPT_PATCH=0 ;;
  nccl)
    export CUDA_MULTICAST_SHIM=0 NCCL_CKPT_PATCH=1
    # NCCL can only release multicast NCCL owns; leave any other owner live
    # and the blocker gate refuses the checkpoint (by design).
    export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
    export DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
    export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
    ;;
  *) echo "unknown MECH=$MECH (want mcshim|nccl)" >&2; exit 2 ;;
esac
# Job members have historically had to be toggled one at a time so that an
# importer restores after its exporter. Overridable, because once the interposer
# has released the multicast objects and peer imports before the checkpoint,
# that dependency may no longer exist.
export CUDA_CKPT_SEQUENTIAL="${CUDA_CKPT_SEQUENTIAL:-1}"

# Reap a sandbox left behind by a failed run; otherwise it holds tens of GB and
# starves the next trial.
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

pass=0 toggle=0 other=0
for ((i = 1; i <= TRIALS; i++)); do
    log=$(mktemp /tmp/vllmtrial.XXXXXX.log)
    printf 'trial %d/%d (mech=%s TP=%s eager=%s seq=%s): ' "$i" "$TRIALS" "$MECH" "$TP" "$EAGER" "$CUDA_CKPT_SEQUENTIAL"
    timeout 1500 bash ./bench_4_vllm_multi.sh --gpus "$GPUS" --tp "$TP" >"$log" 2>&1
    # cuda-checkpoint reports a failed restore toggle in the sentry log, not on
    # the benchmark's stdout, so classify against the run's artifacts too.
    art=$(grep -ohE '==> Artifacts at .*' "$log" | tail -1 | awk '{print $NF}')
    toggle_err=""
    [[ -n "$art" ]] && toggle_err=$(grep -lE 'restore toggle failed|Error toggling CUDA' "$art"/logs/*boot* 2>/dev/null | head -1)
    if grep -q "RESULT: PASS" "$log"; then
        pass=$((pass + 1))
        echo "PASS"
    elif [[ -n "$toggle_err" ]] || grep -qE 'restore toggle failed|Error toggling CUDA' "$log"; then
        toggle=$((toggle + 1))
        echo "FAIL (cuda-checkpoint restore toggle -- pre-existing, not multicast)"
    else
        other=$((other + 1))
        echo "FAIL (other): $(grep -ohE 'wake_up response: .*|FAIL: .*|Container exited.*' "$log" | tail -1 | cut -c1-120)"
    fi
    rm -f "$log"
    # Each trial leaves a checkpoint image of tens of GB. Left to accumulate
    # they fill the disk, and a full disk shows up as a save failure that lands
    # in the "other" bucket -- i.e. it looks like our bug. Classification is
    # already done by this point, so drop the artifacts.
    # reap first: the sandbox still holds mounts under the artifact dir.
    reap
    [[ "${KEEP_ARTIFACTS:-0}" = "1" || -z "$art" ]] || rm -rf "$art" 2>/dev/null
done

echo
echo "mech=$MECH TP=$TP: pass $pass/$TRIALS, cuda-checkpoint toggle failures $toggle, other $other"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '
echo
[[ $other -eq 0 ]]
