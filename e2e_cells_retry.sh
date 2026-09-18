#!/usr/bin/env bash
# Retry: multimem TP=4 cross-GPU (attempt 1 hit the pre-existing pma flake).
set -u
cd /home/ubuntu/gvisor
OUT=$(ls -dt /data/close_cells_2026* | head -1)
timeout 2400 sudo env RUNSC=/usr/local/bin/runsc-r580 \
    CUDA_MULTICAST_SHIM=1 CUDA_MULTICAST_SHIM_SRC=/home/ubuntu/gvisor/tools/mcshim/mcshim.so \
    CUDA_CKPT_JOB_FILE=0 CUDA_CKPT_SEQUENTIAL=1 NCCL_CUMEM_ENABLE=1 \
    MCSHIM_IPC_SUSPEND=1 MCSHIM_FREE_UC_EXPORTS=1 \
    SGLANG_EXTRA_ARGS="--dtype bfloat16 --enable-torch-symm-mem" \
    bash cr-bench/bench_6_sglang_multi.sh --gpus 0,1,2,3 --tp 4 --no-torch-compile \
    --restore-gpus 4,5,6,7 >"$OUT/multimem_tp4_xgpu.attempt3.log" 2>&1
rc=$?
v=FAIL
grep -q "RESULT: PASS" "$OUT/multimem_tp4_xgpu.attempt3.log" && v=PASS
echo "[cells-retry] multimem_tp4_xgpu attempt3 rc=$rc verdict=$v" >>"$OUT/summary.txt"
