#!/usr/bin/env bash
# --------------------------------------------------------------------------
#  bench_3_vllm_single.sh — vLLM single-GPU checkpoint/restore benchmark
#  using gVisor's native cuda-checkpoint support and the /sleep + /wake_up
#  lifecycle endpoints.
#
#  Flow:
#    boot vLLM → reference inference → POST /sleep?level=0 (quiesce)
#    → runsc checkpoint --cuda-checkpoint-path=… (sentry toggles the
#      CUDA processes + serializes) → runsc restore (sentry re-toggles)
#    → POST /wake_up → first inference (timed) → verification
#
#  Usage:
#    sudo bash cr-bench/bench_3_vllm_single.sh
#    sudo bash cr-bench/bench_3_vllm_single.sh --gpus 2            # pick a GPU
#    sudo bash cr-bench/bench_3_vllm_single.sh --sleep-level 1     # offload weights
#
#  Prerequisites: runsc with cuda-checkpoint support, NVIDIA driver >= R550
#  (>= R555 recommended), docker, nvidia-container-cli.
# --------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BENCH_NAME="cr-bench-vllm-1gpu"
BENCH_BANNER="║   Benchmark 3: vLLM single-GPU C/R (cuda-checkpoint + sleep)    ║"

# Single GPU defaults.
GPU_DEVICES="${GPU_DEVICES:-0}"
TP="${TP:-1}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

source "$SCRIPT_DIR/_bench_vllm_impl.sh"

vllm_parse_flags "$@"
vllm_run
