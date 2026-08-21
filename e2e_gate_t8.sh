#!/usr/bin/env bash
# Gate trial 8: SGLang TP=4 forced NVLS, CROSS-GPU restore 0-3 -> 4-7.
# Closes the one cell the final regate did not re-verify.
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
    SGLANG_EXTRA_ARGS=--enable-nccl-nvls \
    bash cr-bench/bench_6_sglang_multi.sh --gpus 0,1,2,3 --tp 4 --restore-gpus 4,5,6,7 \
    >"$OUT/sglang_tp4_nvls_xgpu.attempt1.log" 2>&1
rc=$?
v=FAIL
grep -q "RESULT: PASS" "$OUT/sglang_tp4_nvls_xgpu.attempt1.log" && v=PASS
echo "sglang_tp4_nvls_xgpu $v" >>"$OUT/verdicts.txt"
echo "[gate-t8] rc=$rc verdict=$v" >>"$OUT/summary.txt"
