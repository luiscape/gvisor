#!/usr/bin/env bash
# Gate trial 5: SGLang TP=4 torch symm-mem two-shot (bf16, no compile,
# MCSHIM_HIDE_MULTICAST=1) -- the config that holds FLA registrations LIVE
# at checkpoint, exercising nvproxy suspend n>0 + true-restore lazy path.
set -u
cd /home/ubuntu/gvisor
OUT=/data/e2e_gate_20260820_034759
timeout 2400 sudo env \
    RUNSC=/usr/local/bin/runsc-r580 \
    CUDA_MULTICAST_SHIM=1 \
    CUDA_MULTICAST_SHIM_SRC=/home/ubuntu/gvisor/tools/mcshim/mcshim.so \
    CUDA_CKPT_JOB_FILE=0 \
    CUDA_CKPT_SEQUENTIAL=1 \
    NCCL_CUMEM_ENABLE=1 \
    MCSHIM_IPC_SUSPEND=1 \
    MCSHIM_HIDE_MULTICAST=1 \
    SGLANG_EXTRA_ARGS="--dtype bfloat16 --enable-torch-symm-mem" \
    bash cr-bench/bench_6_sglang_multi.sh --gpus 0,1,2,3 --tp 4 --no-torch-compile \
    >"$OUT/sglang_tp4_symm.attempt1.log" 2>&1
rc=$?
v=FAIL
grep -q "RESULT: PASS" "$OUT/sglang_tp4_symm.attempt1.log" && v=PASS
echo "sglang_tp4_symm $v" >>"$OUT/verdicts.txt"
echo "[gate-t5] rc=$rc verdict=$v" >>"$OUT/summary.txt"
