#!/usr/bin/env bash
# Multimem VALIDATION: same config as the failing probe, plus
# MCSHIM_FREE_UC_EXPORTS=1. Expected: full checkpoint/restore PASS.
set -u
cd /home/ubuntu/gvisor
OUT=/data/probe_multimem2_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
timeout 2400 sudo env \
    RUNSC=/usr/local/bin/runsc-r580 \
    CUDA_MULTICAST_SHIM=1 \
    CUDA_MULTICAST_SHIM_SRC=/home/ubuntu/gvisor/tools/mcshim/mcshim.so \
    CUDA_CKPT_JOB_FILE=0 \
    CUDA_CKPT_SEQUENTIAL=1 \
    NCCL_CUMEM_ENABLE=1 \
    MCSHIM_IPC_SUSPEND=1 \
    MCSHIM_FREE_UC_EXPORTS=1 \
    SGLANG_EXTRA_ARGS="--dtype bfloat16 --enable-torch-symm-mem" \
    bash cr-bench/bench_6_sglang_multi.sh --gpus 0,1,2,3 --tp 4 --no-torch-compile \
    >"$OUT/bench.log" 2>&1
rc=$?
v=FAIL
grep -q "RESULT: PASS" "$OUT/bench.log" && v=PASS
echo "multimem_free_uc rc=$rc verdict=$v" >>"$OUT/notes.txt"
ls -dt /data/cr-bench/cr-bench-sglang-multi-* | head -1 >>"$OUT/notes.txt"
