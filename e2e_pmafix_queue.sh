#!/usr/bin/env bash
# Validation queue for the frontendFD/uvmFD InvalidateUnsavable fix:
# full gate + 2x the flakiest config (multimem TP=4 cross-GPU, which hit
# the pma save race 2 of 3 runs before the fix).
set -u
cd /home/ubuntu/gvisor
bash e2e_gate.sh
OUT=/data/pmafix_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
for i in 1 2; do
    timeout 2400 sudo env RUNSC=/usr/local/bin/runsc-r580 \
        CUDA_MULTICAST_SHIM=1 CUDA_MULTICAST_SHIM_SRC=/home/ubuntu/gvisor/tools/mcshim/mcshim.so \
        CUDA_CKPT_JOB_FILE=0 CUDA_CKPT_SEQUENTIAL=1 NCCL_CUMEM_ENABLE=1 \
        MCSHIM_IPC_SUSPEND=1 MCSHIM_FREE_UC_EXPORTS=1 \
        SGLANG_EXTRA_ARGS="--dtype bfloat16 --enable-torch-symm-mem" \
        bash cr-bench/bench_6_sglang_multi.sh --gpus 0,1,2,3 --tp 4 --no-torch-compile \
        --restore-gpus 4,5,6,7 >"$OUT/multimem_xgpu.run$i.log" 2>&1
    v=FAIL; grep -q "RESULT: PASS" "$OUT/multimem_xgpu.run$i.log" && v=PASS
    echo "multimem_xgpu run$i $v" >>"$OUT/summary.txt"
    for m in $(mount | grep /data/cr-bench/cr-bench- | awk '{print $3}'); do sudo umount -l "$m" 2>/dev/null || true; done
    sleep 10
done
echo done >>"$OUT/summary.txt"
