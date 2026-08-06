#!/usr/bin/env bash
# --------------------------------------------------------------------------
#  bench_6_sglang_multi.sh — SGLang MULTI-GPU checkpoint/restore benchmark
#  using gVisor's native cuda-checkpoint support.
#
#  SGLang counterpart of bench_4_vllm_multi.sh, same model
#  (Qwen2.5-1.5B-Instruct).  SGLang with --tp-size N runs one HTTP/
#  tokenizer process plus N TP scheduler processes that share GPU memory
#  via CUDA IPC and communicate through NCCL — the sentry must toggle all
#  CUDA processes in parallel, exactly as with vLLM.
#
#  Lifecycle uses SGLang's NATIVE endpoints (no wrapper):
#    POST /pause_generation    before checkpoint
#    POST /continue_generation after restore
#
#  Usage:
#    sudo bash cr-bench/bench_6_sglang_multi.sh                      # TP=2, GPUs 0,1
#    sudo bash cr-bench/bench_6_sglang_multi.sh --gpus 0,1,2,3 --tp 4
#    sudo bash cr-bench/bench_6_sglang_multi.sh --gpus 0,1 --restore-gpus 2,3
#
#  Prerequisites: runsc with cuda-checkpoint support (see README), >= 2
#  GPUs, NVIDIA driver >= R570 recommended, docker, nvidia-container-cli.
# --------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BENCH_NAME="cr-bench-sglang-multi"
BENCH_BANNER="║   Benchmark 6: SGLang MULTI-GPU C/R (cuda-checkpoint + pause)   ║"

# Multi-GPU defaults: 2-way tensor parallel on GPUs 0,1 — same model as
# the vLLM multi-GPU benchmark (works with TP=2 and TP=4).
GPU_DEVICES="${GPU_DEVICES:-0,1}"
TP="${TP:-2}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

source "$SCRIPT_DIR/_bench_sglang_impl.sh"

sglang_parse_flags "$@"

# Multi-GPU sanity: warn early about IPC support on older drivers.
_drv_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d. -f1)"
if [[ -n "$_drv_major" ]] && (( _drv_major < 570 )); then
    warn "Driver R${_drv_major} < R570: cuda-checkpoint may not support the CUDA IPC"
    warn "memory shared between tensor-parallel workers — expect failures."
fi

sglang_run
