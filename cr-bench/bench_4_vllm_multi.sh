#!/usr/bin/env bash
# --------------------------------------------------------------------------
#  bench_4_vllm_multi.sh — vLLM MULTI-GPU checkpoint/restore benchmark
#  using gVisor's native cuda-checkpoint support.
#
#  This is the hard case: vLLM with --tensor-parallel-size N spawns one
#  engine-core process plus N tensor-parallel worker processes that share
#  GPU memory via CUDA IPC and communicate through NCCL.  A correct
#  snapshot requires suspending ALL CUDA processes together:
#
#    runsc checkpoint --cuda-checkpoint-path=…
#      → the sentry discovers every CUDA process in the sandbox (by its
#        open nvproxy device FDs), filters with cuda-checkpoint
#        --get-state, and runs `cuda-checkpoint --toggle` on all of them
#        IN PARALLEL (required: IPC-connected processes cannot be
#        suspended one at a time), then serializes the sandbox.
#      → on restore the sentry re-toggles all of them in parallel.
#
#  Lifecycle management around the snapshot:
#    POST /sleep?level=0 before checkpoint  (engine quiesced, NCCL idle)
#    POST /wake_up       after restore
#
#  Notes:
#    - Driver >= R570 is strongly recommended (CUDA IPC support in
#      cuda-checkpoint).
#    - NCCL_CUMEM_ENABLE=0 is set by default so NCCL uses classic
#      allocations (override with NCCL_CUMEM_ENABLE=1 to test VMM).
#    - The default model (Qwen2.5-1.5B-Instruct) works with TP=2 and
#      TP=4.  Qwen2.5-0.5B-Instruct (14 heads) only supports TP=2.
#
#  Usage:
#    sudo bash cr-bench/bench_4_vllm_multi.sh                      # TP=2, GPUs 0,1
#    sudo bash cr-bench/bench_4_vllm_multi.sh --gpus 0,1,2,3 --tp 4
#    sudo bash cr-bench/bench_4_vllm_multi.sh --sequential         # debug toggling
#
#  Prerequisites: runsc with cuda-checkpoint support, >= 2 GPUs, NVIDIA
#  driver >= R570 recommended, docker, nvidia-container-cli.
# --------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BENCH_NAME="cr-bench-vllm-multi"
BENCH_BANNER="║   Benchmark 4: vLLM MULTI-GPU C/R (cuda-checkpoint + sleep)     ║"

# Multi-GPU defaults: 2-way tensor parallel on GPUs 0,1.
GPU_DEVICES="${GPU_DEVICES:-0,1}"
TP="${TP:-2}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

source "$SCRIPT_DIR/_bench_vllm_impl.sh"

vllm_parse_flags "$@"

# Multi-GPU sanity: warn early about IPC support on older drivers.
_drv_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d. -f1)"
if [[ -n "$_drv_major" ]] && (( _drv_major < 570 )); then
    warn "Driver R${_drv_major} < R570: cuda-checkpoint may not support the CUDA IPC"
    warn "memory shared between tensor-parallel workers — expect failures."
fi

vllm_run
