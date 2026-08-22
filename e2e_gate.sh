#!/usr/bin/env bash
# e2e regression gate for the final PR binary (single-pass enforcement).
# Runs the highest-value validated configs on runsc-r580 and records verdicts.
# Driver log: $OUT/driver.log; per-trial logs: $OUT/<trial>.log
set -u
cd /home/ubuntu/gvisor

TS="$(date +%Y%m%d_%H%M%S)"
OUT="/data/e2e_gate_$TS"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"
touch "$SUMMARY"

RUNSC=/usr/local/bin/runsc-r580
SHIM=/home/ubuntu/gvisor/tools/mcshim/mcshim.so

note() { echo "[gate $(date +%H:%M:%S)] $*" | tee -a "$SUMMARY"; }

# Kill any stray CUDA apps and wait until GPUs read clean.
gpu_settle() {
    for _ in $(seq 1 36); do
        local pids
        pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' || true)"
        if [ -z "$pids" ]; then return 0; fi
        for p in $pids; do sudo kill -9 "$p" 2>/dev/null || true; done
        sleep 5
    done
    note "WARNING: GPUs not clean after settle window"
    return 1
}

# Lazy-unmount any mounts a failed run left behind.
umount_strays() {
    for m in $(mount | grep /data/cr-bench/cr-bench- | awk '{print $3}'); do
        sudo umount -l "$m" 2>/dev/null || true
    done
}

newest_rundir() { # $1: sglang|vllm
    ls -dt /data/cr-bench/cr-bench-$1-multi-* 2>/dev/null | head -1
}

# run_trial <name> <engine:sglang|vllm> <extra-env...> -- <bench args...>
run_trial() {
    local name="$1" engine="$2"; shift 2
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    local bench="cr-bench/bench_4_vllm_multi.sh"
    [ "$engine" = sglang ] && bench="cr-bench/bench_6_sglang_multi.sh"

    local attempt verdict rc
    for attempt in 1 2; do
        note "TRIAL $name attempt=$attempt starting: env=${envs[*]:-none} args=$*"
        timeout 2400 sudo env \
            RUNSC="$RUNSC" \
            CUDA_MULTICAST_SHIM=1 \
            CUDA_MULTICAST_SHIM_SRC="$SHIM" \
            CUDA_CKPT_JOB_FILE=0 \
            CUDA_CKPT_SEQUENTIAL=1 \
            NCCL_CUMEM_ENABLE=1 \
            MCSHIM_IPC_SUSPEND=1 \
            "${envs[@]}" \
            bash "$bench" "$@" >"$OUT/$name.attempt$attempt.log" 2>&1
        rc=$?
        verdict=FAIL
        grep -q "RESULT: PASS" "$OUT/$name.attempt$attempt.log" && verdict=PASS
        note "TRIAL $name attempt=$attempt rc=$rc verdict=$verdict rundir=$(newest_rundir "$engine")"
        umount_strays
        gpu_settle
        if [ "$verdict" = PASS ]; then break; fi
        # Retry only the known pre-existing pma save flake. The bench log
        # tail sometimes shows only the stack trace, so also check the run
        # dir's checkpoint log for the signature.
        flake=0
        grep -q "non-MemoryFile of type" "$OUT/$name.attempt$attempt.log" && flake=1
        rd="$(newest_rundir "$engine")"
        [ -n "$rd" ] && sudo grep -q "non-MemoryFile of type" "$rd/logs/runsc-checkpoint.log" 2>/dev/null && flake=1
        if [ "$flake" = 0 ]; then break; fi
        note "TRIAL $name hit the known pma save flake; retrying once"
    done
    echo "$name $verdict" >>"$OUT/verdicts.txt"

    # Engagement evidence (best-effort).
    local rundir
    rundir="$(newest_rundir "$engine")"
    if [ -n "$rundir" ]; then
        local fla skip
        fla="$(grep -h "host-freed.*FLA registrations" "$rundir"/logs/* 2>/dev/null | tail -1)"
        [ -n "$fla" ] && note "TRIAL $name FLA-suspend: $fla"
        skip="$(grep -h "Skipping allreduce fusion" "$rundir"/applog/*.log 2>/dev/null | tail -1)"
        [ -n "$skip" ] && note "TRIAL $name fusion-SKIPPED marker present (vacuous fusion)"
    fi
}

note "gate start: binary=$($RUNSC --version | head -1); out=$OUT"
gpu_settle
umount_strays

# 1. vLLM TP=4 same-GPU, stock config (custom-AR default).
run_trial vllm_tp4 vllm -- --gpus 0,1,2,3 --tp 4

# 2. vLLM TP=2 cross-GPU restore 0,1 -> 4,5 (device remapping).
run_trial vllm_tp2_xgpu vllm -- --gpus 0,1 --tp 2 --restore-gpus 4,5

# 3. SGLang TP=4 forced NVLS (fabric objects transient, shim-released).
run_trial sglang_tp4_nvls sglang SGLANG_EXTRA_ARGS=--enable-nccl-nvls -- --gpus 0,1,2,3 --tp 4

# 4. SGLang TP=4 FlashInfer fusion ENGAGED (persistent FLA/rank ->
#    exercises nvproxy FLA suspend + post-restore lazy re-register).
run_trial sglang_tp4_fusion sglang SGLANG_EXTRA_ARGS=--flashinfer-allreduce-fusion-backend\ trtllm -- --gpus 0,1,2,3 --tp 4 --no-torch-compile

note "==== GATE VERDICTS ===="
cat "$OUT/verdicts.txt" | tee -a "$SUMMARY"
pass_n="$(grep -c PASS "$OUT/verdicts.txt" || true)"
total_n="$(wc -l <"$OUT/verdicts.txt")"
note "GATE RESULT: $pass_n/$total_n PASS"
