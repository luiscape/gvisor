#!/usr/bin/env bash
# Close the remaining validation-matrix cells:
#   1. torch symm-mem MULTIMEM TP=4, CROSS-GPU restore 0-3 -> 4-7
#   2. torch symm-mem MULTIMEM TP=8 (Qwen2.5-3B, custom-AR off)
#   3. vLLM symm-mem (VLLM_ALLREDUCE_USE_SYMM_MEM=1) TP=4, CROSS-GPU
set -u
cd /home/ubuntu/gvisor
OUT=/data/close_cells_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"
note() { echo "[cells $(date +%H:%M:%S)] $*" | tee -a "$SUMMARY"; }

gpu_settle() {
    for _ in $(seq 1 36); do
        local pids
        pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' || true)"
        [ -z "$pids" ] && return 0
        for p in $pids; do sudo kill -9 "$p" 2>/dev/null || true; done
        sleep 5
    done
    note "WARNING: GPUs not clean"
}
umount_strays() {
    for m in $(mount | grep /data/cr-bench/cr-bench- | awk '{print $3}'); do
        sudo umount -l "$m" 2>/dev/null || true
    done
}
verdict_of() { # $1 log
    if grep -q "RESULT: PASS" "$1"; then echo PASS; else echo FAIL; fi
}

note "start; disk: $(df -h / | tail -1 | awk '{print $4}') free"

# 1. multimem TP=4 cross-GPU
note "TRIAL multimem_tp4_xgpu starting"
timeout 2400 sudo env RUNSC=/usr/local/bin/runsc-r580 \
    CUDA_MULTICAST_SHIM=1 CUDA_MULTICAST_SHIM_SRC=/home/ubuntu/gvisor/tools/mcshim/mcshim.so \
    CUDA_CKPT_JOB_FILE=0 CUDA_CKPT_SEQUENTIAL=1 NCCL_CUMEM_ENABLE=1 \
    MCSHIM_IPC_SUSPEND=1 MCSHIM_FREE_UC_EXPORTS=1 \
    SGLANG_EXTRA_ARGS="--dtype bfloat16 --enable-torch-symm-mem" \
    bash cr-bench/bench_6_sglang_multi.sh --gpus 0,1,2,3 --tp 4 --no-torch-compile \
    --restore-gpus 4,5,6,7 >"$OUT/multimem_tp4_xgpu.log" 2>&1
note "TRIAL multimem_tp4_xgpu rc=$? verdict=$(verdict_of "$OUT/multimem_tp4_xgpu.log")"
umount_strays; gpu_settle

# 2. multimem TP=8 (3B model, custom-AR off, generous health timeout)
note "TRIAL multimem_tp8 starting"
timeout 3600 sudo env RUNSC=/usr/local/bin/runsc-r580 \
    CUDA_MULTICAST_SHIM=1 CUDA_MULTICAST_SHIM_SRC=/home/ubuntu/gvisor/tools/mcshim/mcshim.so \
    CUDA_CKPT_JOB_FILE=0 CUDA_CKPT_SEQUENTIAL=1 NCCL_CUMEM_ENABLE=1 \
    MCSHIM_IPC_SUSPEND=1 MCSHIM_FREE_UC_EXPORTS=1 \
    MODEL=Qwen/Qwen2.5-3B-Instruct HEALTH_TIMEOUT=1500 \
    SGLANG_EXTRA_ARGS="--dtype bfloat16 --enable-torch-symm-mem --disable-custom-all-reduce" \
    bash cr-bench/bench_6_sglang_multi.sh --gpus 0,1,2,3,4,5,6,7 --tp 8 --no-torch-compile \
    >"$OUT/multimem_tp8.log" 2>&1
note "TRIAL multimem_tp8 rc=$? verdict=$(verdict_of "$OUT/multimem_tp8.log")"
umount_strays; gpu_settle

# 3. vLLM symm-mem TP=4 cross-GPU
note "TRIAL vllm_symm_tp4_xgpu starting"
timeout 2400 sudo env RUNSC=/usr/local/bin/runsc-r580 \
    CUDA_MULTICAST_SHIM=1 CUDA_MULTICAST_SHIM_SRC=/home/ubuntu/gvisor/tools/mcshim/mcshim.so \
    CUDA_CKPT_JOB_FILE=0 CUDA_CKPT_SEQUENTIAL=1 NCCL_CUMEM_ENABLE=1 \
    MCSHIM_IPC_SUSPEND=1 VLLM_ALLREDUCE_USE_SYMM_MEM=1 \
    bash cr-bench/bench_4_vllm_multi.sh --gpus 0,1,2,3 --tp 4 --restore-gpus 4,5,6,7 \
    >"$OUT/vllm_symm_tp4_xgpu.log" 2>&1
note "TRIAL vllm_symm_tp4_xgpu rc=$? verdict=$(verdict_of "$OUT/vllm_symm_tp4_xgpu.log")"
umount_strays; gpu_settle
note "ALL DONE"
