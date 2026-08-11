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
# Usage: sudo [TRIALS=3] [TP=2] [GPUS=0,1] [EAGER=0] bash vllm_trials.sh
set -uo pipefail
cd "$(dirname "$0")"

TRIALS="${TRIALS:-3}"
TP="${TP:-2}"
GPUS="${GPUS:-0,1}"
export EAGER="${EAGER:-0}"
export RUNSC="${RUNSC:-/usr/local/bin/runsc-phase0}"
export CUDA_CKPT_JOB_FILE=1 CUDA_MULTICAST_SHIM=1
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
    printf 'trial %d/%d (TP=%s eager=%s seq=%s): ' "$i" "$TRIALS" "$TP" "$EAGER" "$CUDA_CKPT_SEQUENTIAL"
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
    reap
done

echo
echo "pass $pass/$TRIALS, cuda-checkpoint toggle failures $toggle, other $other"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '
echo
[[ $other -eq 0 ]]
