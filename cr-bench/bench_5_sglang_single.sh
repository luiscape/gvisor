#!/usr/bin/env bash
# --------------------------------------------------------------------------
#  bench_5_sglang_single.sh — SGLang single-GPU checkpoint/restore benchmark
#  using gVisor's native cuda-checkpoint support.
#
#  SGLang counterpart of bench_3_vllm_single.sh, same model
#  (Qwen2.5-0.5B-Instruct).  Uses SGLang's NATIVE lifecycle endpoints —
#  no wrapper needed:
#    POST /pause_generation    before checkpoint
#    POST /continue_generation after restore
#
#  Usage:
#    sudo bash cr-bench/bench_5_sglang_single.sh
#    sudo bash cr-bench/bench_5_sglang_single.sh --gpus 2              # pick a GPU
#    sudo bash cr-bench/bench_5_sglang_single.sh --gpus 0 --restore-gpus 1
#
#  Prerequisites: runsc with cuda-checkpoint support (see README), NVIDIA
#  driver >= R550, docker, nvidia-container-cli.
# --------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BENCH_NAME="cr-bench-sglang-1gpu"
BENCH_BANNER="║   Benchmark 5: SGLang single-GPU C/R (cuda-checkpoint + pause)  ║"

# Single GPU defaults — same model as the vLLM single-GPU benchmark.
GPU_DEVICES="${GPU_DEVICES:-0}"
TP="${TP:-1}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

source "$SCRIPT_DIR/_bench_sglang_impl.sh"

sglang_parse_flags "$@"
sglang_run
