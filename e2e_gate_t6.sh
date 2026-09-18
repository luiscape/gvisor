#!/usr/bin/env bash
# Gate trial 6: SGLang TP=4 FlashInfer fusion trtllm with QUIESCE=pause --
# GPU memory (and the fusion-workspace FLA registrations) stays RESIDENT at
# checkpoint, so nvproxy's SuspendFLARegistrations must fire with n>0 and
# the true restore must come back clean (lazy re-register, guard silent).
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
    SGLANG_EXTRA_ARGS="--flashinfer-allreduce-fusion-backend trtllm" \
    bash cr-bench/bench_6_sglang_multi.sh --gpus 0,1,2,3 --tp 4 \
    --no-torch-compile --quiesce pause \
    >"$OUT/sglang_tp4_fusion_pause.attempt-pmafix.log" 2>&1
rc=$?
v=FAIL
grep -q "RESULT: PASS" "$OUT/sglang_tp4_fusion_pause.attempt-pmafix.log" && v=PASS
echo "sglang_tp4_fusion_pause $v" >>"$OUT/verdicts.txt"
echo "[gate-t6] rc=$rc verdict=$v" >>"$OUT/summary.txt"
