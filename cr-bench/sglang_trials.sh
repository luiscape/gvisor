#!/usr/bin/env bash
# sglang_trials.sh -- repeat the SGLang checkpoint/restore benchmark and report
# a pass rate, mirroring vllm_trials.sh. SGLang's cold boot is ~10 minutes, so
# budget ~12 minutes per trial.
#
# Usage: sudo [TRIALS=3] [TP=2] [GPUS=0,1] bash sglang_trials.sh
set -uo pipefail
cd "$(dirname "$0")"

TRIALS="${TRIALS:-3}"
TP="${TP:-2}"
GPUS="${GPUS:-0,1}"

export CUDA_MULTICAST_SHIM=1
export CUDA_CKPT_JOB_FILE="${CUDA_CKPT_JOB_FILE:-1}"
export CUDA_CKPT_SEQUENTIAL="${CUDA_CKPT_SEQUENTIAL:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
export MCSHIM_IPC_SUSPEND="${MCSHIM_IPC_SUSPEND:-1}"
export RUNSC="${RUNSC:-/usr/local/bin/runsc-phase0-mapfix}"

reap() {
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
        kill -9 "$p" 2>/dev/null
    done
    sleep 5
}

pass=0 toggle=0 other=0
for ((i = 1; i <= TRIALS; i++)); do
    log=$(mktemp /tmp/sglangtrial.XXXXXX.log)
    printf 'trial %d/%d (TP=%s cumem=%s ipc_suspend=%s): ' \
        "$i" "$TRIALS" "$TP" "$NCCL_CUMEM_ENABLE" "$MCSHIM_IPC_SUSPEND"
    timeout 1500 bash ./bench_6_sglang_multi.sh --gpus "$GPUS" --tp "$TP" >"$log" 2>&1
    art=$(grep -ohE '==> Artifacts at .*' "$log" | tail -1 | awk '{print $NF}')
    toggle_err=""
    [[ -n "$art" ]] && toggle_err=$(grep -lE 'restore toggle failed|Error toggling CUDA' "$art"/logs/*boot* 2>/dev/null | head -1)
    if grep -q "RESULT: PASS" "$log"; then
        pass=$((pass + 1)); echo "PASS"
    elif [[ -n "$toggle_err" ]] || grep -qE 'restore toggle failed|Error toggling CUDA' "$log"; then
        toggle=$((toggle + 1)); echo "FAIL (cuda-checkpoint restore toggle)"
    else
        other=$((other + 1))
        echo "FAIL (other): $(grep -ohE 'FAIL: .*|Container exited.*|never responded.*' "$log" | tail -1 | cut -c1-110)"
    fi
    rm -f "$log"
    reap
    [[ "${KEEP_ARTIFACTS:-0}" = "1" || -z "$art" ]] || rm -rf "$art" 2>/dev/null
done
echo
echo "sglang TP=$TP: pass $pass/$TRIALS, toggle failures $toggle, other $other"
