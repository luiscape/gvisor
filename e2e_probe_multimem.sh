#!/usr/bin/env bash
# PROBE: torch symm-mem MULTIMEM (no MCSHIM_HIDE_MULTICAST) across C/R.
# Expected to FAIL at restore -- the point is the diagnostics:
#   - checkpoint-side boot log: every 00f8 alloc (handle -> covered hVidMem),
#     object classes, and the host-freed FLA registration set;
#   - restore-side boot log: the failing EXPORT_OBJECT(S)_TO_FD with the
#     stale handle (new nvproxy failure-path logging);
#   - app log: mcshim VERBOSE resume progress (which phase died) + torch error.
set -u
cd /home/ubuntu/gvisor
OUT=/data/probe_multimem_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
timeout 2400 sudo env \
    RUNSC=/usr/local/bin/runsc-r580-probe \
    CUDA_MULTICAST_SHIM=1 \
    CUDA_MULTICAST_SHIM_SRC=/home/ubuntu/gvisor/tools/mcshim/mcshim.so \
    CUDA_CKPT_JOB_FILE=0 \
    CUDA_CKPT_SEQUENTIAL=1 \
    NCCL_CUMEM_ENABLE=1 \
    MCSHIM_IPC_SUSPEND=1 \
    MCSHIM_VERBOSE=1 \
    SGLANG_EXTRA_ARGS="--dtype bfloat16 --enable-torch-symm-mem" \
    bash cr-bench/bench_6_sglang_multi.sh --gpus 0,1,2,3 --tp 4 --no-torch-compile \
    >"$OUT/bench.log" 2>&1
rc=$?
echo "bench rc=$rc" >>"$OUT/notes.txt"
# Preserve the run dir path for analysis.
ls -dt /data/cr-bench/cr-bench-sglang-multi-* | head -1 >>"$OUT/notes.txt"
