#!/usr/bin/env bash
# --------------------------------------------------------------------------
#  _bench_vllm_impl.sh — shared implementation for the vLLM checkpoint /
#  restore benchmarks (bench_3_vllm_single.sh and bench_4_vllm_multi.sh).
#
#  Lifecycle (the /sleep + /wake_up endpoints come from
#  apps/vllm_sleep_server.py, baked into the image):
#
#    1. runsc run           → wait for /health          (cold boot, timed)
#    2. reference inference (deterministic, temperature=0, seed)
#    3. POST /sleep?level=L → quiesce the engine so no inference is
#                             in-flight and the GPU is idle
#    4. runsc checkpoint --cuda-checkpoint-path=…
#                           → sentry runs cuda-checkpoint --toggle on
#                             EVERY CUDA process (API server / engine
#                             core / TP workers) in parallel, then
#                             serializes the sandbox
#    5. runsc restore       → sentry re-toggles all CUDA processes
#    6. wait /health
#    7. POST /wake_up       → resume the engine
#    8. first inference (timed) + verification queries
#
#  The caller (wrapper script) must set: BENCH_NAME, TP, GPU_DEVICES,
#  MODEL, and may override any knob below.  Do not run this directly.
# --------------------------------------------------------------------------
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# ── Configuration (defaults; wrappers/CLI may override) ───────────────────
IMAGE="${IMAGE:-cr-bench-vllm}"
DOCKERFILE="images/Dockerfile.vllm"
PORT="${PORT:-8199}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"
# Sleep level before checkpoint (default 1: offload weights to CPU and
# discard the idle KV cache, so cuda-checkpoint stages minimal GPU memory;
# /wake_up after restore moves weights back and re-allocates the KV cache).
# Level 1 requires launching vLLM with --enable-sleep-mode.
SLEEP_LEVEL="${SLEEP_LEVEL:-1}"
# CUDA graph capture + torch.compile (vLLM's default; disable with --eager):
# skipping this expensive warmup on restore is the main C/R use-case.
EAGER="${EAGER:-0}"
CB_SHM_MB="${CB_SHM_MB:-8192}"
CB_GPU=1
APP_LOG="/applog/vllm.log"
# NCCL knobs (multi-GPU). NCCL_CUMEM_ENABLE picks which API NCCL uses for its
# intra-node P2P buffers, and with the interposer in play that choice decides
# whether the workload can be restored at all:
#
#   0  classic allocator -> NCCL shares P2P buffers over LEGACY CUDA IPC.
#      Measured at TP=2: ~48 live legacy imports per worker. The interposer
#      can close them but cannot get them back at their original addresses
#      (cuIpcOpenMemHandle takes no address hint), so the resume fails.
#   1  VMM allocator -> the same buffers go through cuMemCreate/cuMemMap,
#      which the interposer restores at IDENTICAL addresses via retained
#      address reservations. Measured: 0 legacy imports, 49 VMM imports, and
#      the full sleep/checkpoint/restore/wake_up cycle PASSES with compile
#      and CUDA graphs on.
#
# So 1 is the correct default here, not the cautious-looking 0. The old
# comment ("cuda-checkpoint's VMM coverage is driver-dependent, default to the
# safe path") predates the interposer: cuda-checkpoint is no longer the thing
# restoring these mappings.
NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
# NCCL_NVLS_ENABLE=0 disables NVLS (NVLink SHARP) multicast, which allocates
# fabric memory (cuMemExportToShareableHandle) that cuda-checkpoint cannot
# checkpoint. Empty (default) leaves NCCL's own default in effect.
NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-}"
# DISABLE_CUSTOM_ALL_REDUCE=1 passes --disable-custom-all-reduce to vLLM.
# vLLM's custom allreduce registers its (graph-captured) peer buffers via
# fabric IPC (cuMemExportToShareableHandle), which cuda-checkpoint cannot
# checkpoint; disabling it routes allreduce through NCCL instead.
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
# VLLM_ALLREDUCE_USE_SYMM_MEM=0 disables vLLM's torch symmetric-memory
# allreduce backend, which allocates fabric + multicast memory via cuMem
# (cuMemExportToShareableHandle) even when NCCL NVLS/cuMem are disabled.
# Empty (default) leaves vLLM's default in effect.
VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-}"
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"

vllm_parse_flags() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --gpus)            GPU_DEVICES="$2"; shift 2 ;;
            --restore-gpus)    RESTORE_GPU_DEVICES="$2"; shift 2 ;;
            --tp)              TP="$2"; shift 2 ;;
            --model)           MODEL="$2"; shift 2 ;;
            --max-model-len)   MAX_MODEL_LEN="$2"; shift 2 ;;
            --gpu-mem-util)    GPU_MEM_UTIL="$2"; shift 2 ;;
            --sleep-level)     SLEEP_LEVEL="$2"; shift 2 ;;
            --eager)           EAGER=1; shift ;;
            --port)            PORT="$2"; shift 2 ;;
            --compression)     COMPRESSION="$2"; shift 2 ;;
            --no-exclude-zero) EXCLUDE_ZERO=0; shift ;;
            --sequential)      CUDA_CKPT_SEQUENTIAL=1; shift ;;
            --image)           IMAGE="$2"; shift 2 ;;
            --rebuild-rootfs)  REBUILD_ROOTFS=1; shift ;;
            --help|-h)
                echo "Usage: $0 [options]"
                echo ""
                echo "  --gpus LIST            comma-separated host GPU indices"
                echo "  --restore-gpus LIST    restore on a DIFFERENT GPU set (device remapping)"
                echo "  --tp N                 tensor parallel size"
                echo "  --model MODEL          HuggingFace model ID"
                echo "  --max-model-len N      max context length (default: 2048)"
                echo "  --gpu-mem-util F       GPU memory utilization (default: 0.7)"
                echo "  --sleep-level L        1: offload weights to CPU + drop KV cache"
                echo "                         (default), 0: pause scheduler only"
                echo "  --eager                --enforce-eager (no CUDA graphs / torch.compile)"
                echo "  --port PORT            vLLM listen port (default: 8199)"
                echo "  --compression MODE     none | flate-best-speed (default: none)"
                echo "  --no-exclude-zero      keep committed zero pages"
                echo "  --sequential           run cuda-checkpoint sequentially (debugging)"
                echo "  --image IMAGE          Docker image name (default: cr-bench-vllm)"
                echo "  --rebuild-rootfs       force re-extract rootfs from image"
                exit 0 ;;
            *) echo "Unknown flag: $1"; exit 1 ;;
        esac
    done
}

# ── Inference helpers ─────────────────────────────────────────────────────
_send_chat() {
    local cid="$1" prompt="$2" max_tokens="${3:-16}" timeout="${4:-60}"
    _rexec "$cid" /usr/bin/curl -s --max-time "$timeout" \
        -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"temperature\":0.0,\"max_tokens\":$max_tokens,\"seed\":42}"
}

extract_content() {
    python3 -c "
import json, sys
r = json.load(sys.stdin)
print(r['choices'][0]['message']['content'])
" 2>/dev/null
}

vllm_run() {
    NUM_GPUS=$(echo "$GPU_DEVICES" | tr ',' '\n' | grep -c .)
    if (( TP > NUM_GPUS )); then
        fail "--tp $TP needs at least $TP GPUs but --gpus lists $NUM_GPUS"; exit 1
    fi

    cb_init
    cb_detect_gpu
    cb_runsc_flags

    banner ""
    banner "╔══════════════════════════════════════════════════════════════════╗"
    banner "$BENCH_BANNER"
    banner "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    info "runsc:           $($RUNSC --version 2>&1 | head -1 || echo '?')"
    info "GPU:             $GPU_NAME ($GPU_MEM_TOTAL MiB), driver $HOST_DRIVER_VER"
    info "GPUs used:       $GPU_DEVICES ($NUM_GPUS), tensor-parallel: $TP"
    [[ -n "${RESTORE_GPU_DEVICES:-}" ]] && \
    info "Restore GPUs:    $RESTORE_GPU_DEVICES (cross-GPU device remapping)"
    info "Model:           $MODEL (max-len $MAX_MODEL_LEN, gpu-mem-util $GPU_MEM_UTIL)"
    info "CUDA graphs + torch.compile: $([ "$EAGER" = 1 ] && echo disabled || echo enabled)"
    info "Sleep level:     $SLEEP_LEVEL $([ "$SLEEP_LEVEL" -ge 1 ] && echo '(weights → CPU, KV cache discarded)')"
    info "cuda-checkpoint: $CUDA_CHECKPOINT_PATH ($([ "$CUDA_CKPT_SEQUENTIAL" = 1 ] && echo sequential || echo parallel))"
    echo ""

    # ── rootfs + bundle ───────────────────────────────────────────────────
    cb_prepare_rootfs

    # Sleep level >= 1 requires vLLM's sleep mode (CuMemAllocator), which
    # backs weights/KV with CUDA VMM allocations that can be released to the
    # driver on sleep.
    local extra_args=""
    if (( SLEEP_LEVEL >= 1 )); then
        extra_args+=" --enable-sleep-mode"
    fi
    # --enforce-eager disables CUDA graph capture and torch.compile; the
    # default (off) exercises the expensive warmup that restore skips.
    if [[ "$EAGER" = "1" ]]; then
        extra_args+=" --enforce-eager"
    fi
    if [[ "$DISABLE_CUSTOM_ALL_REDUCE" = "1" ]]; then
        extra_args+=" --disable-custom-all-reduce"
    fi

    CB_CMD="exec python3 /app/vllm_sleep_server.py \
--model $MODEL \
--host 0.0.0.0 \
--port $PORT \
--gpu-memory-utilization $GPU_MEM_UTIL \
--max-model-len $MAX_MODEL_LEN \
--tensor-parallel-size $TP \
--distributed-executor-backend mp \
--dtype float16 \
$extra_args \
>$APP_LOG 2>&1"
    # VLLM_USE_FLASHINFER_SAMPLER=0: recent vLLM defaults to the FlashInfer
    # sampler which JIT-compiles kernels at warmup (needs ninja + libnvrtc,
    # not in the image, and JIT'd state complicates C/R anyway).
    # LD_LIBRARY_PATH addition: vllm.cumem_allocator (required for sleep
    # mode) links libnvrtc.so.13, which ships in torch's pip 'nvidia'
    # packages but is not on the default loader path.
    CB_ENV="HF_HOME=/app/hf_cache
HF_HUB_OFFLINE=1
VLLM_USAGE_SOURCE=production
VLLM_WORKER_MULTIPROC_METHOD=spawn
VLLM_USE_FLASHINFER_SAMPLER=0
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=$TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC
NCCL_CUMEM_ENABLE=$NCCL_CUMEM_ENABLE
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib"
    [[ -n "$NCCL_NVLS_ENABLE" ]] && CB_ENV+=$'\n'"NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE"
    [[ -n "$VLLM_ALLREDUCE_USE_SYMM_MEM" ]] && CB_ENV+=$'\n'"VLLM_ALLREDUCE_USE_SYMM_MEM=$VLLM_ALLREDUCE_USE_SYMM_MEM"
    # vLLM 0.27 has a fourth potential multicast owner besides NCCL NVLS,
    # custom all-reduce and symmetric memory: the FlashInfer all-reduce
    # backend. Passed through so the interposer can be tested against it.
    #
    # FlashInfer JIT-compiles its kernels at startup and looks for nvcc under
    # CUDA_HOME. The image has no /usr/local/cuda, but it does ship a complete
    # toolchain in torch's pip tree (nvidia/cu13: bin/nvcc, include, nvvm), so
    # point CUDA_HOME there rather than adding the CUDA toolkit to the image.
    if [[ -n "${VLLM_ALLREDUCE_USE_FLASHINFER:-}" ]]; then
        CB_ENV+=$'\n'"VLLM_ALLREDUCE_USE_FLASHINFER=$VLLM_ALLREDUCE_USE_FLASHINFER"
        local cu_home=/usr/local/lib/python3.12/dist-packages/nvidia/cu13
        CB_ENV+=$'\n'"CUDA_HOME=$cu_home"
        CB_ENV+=$'\n'"CUDA_PATH=$cu_home"
        CB_ENV+=$'\n'"PATH=$cu_home/bin:/usr/local/bin:/usr/bin:/bin"
    fi
    # Interposer tuning passthrough. MCSHIM_IPC_EARLY moves the legacy CUDA
    # IPC reopen to before the VMM remap; which position preserves the
    # original addresses is workload-dependent, so it has to be measurable
    # from here.
    [[ -n "${MCSHIM_IPC_EARLY:-}" ]] && CB_ENV+=$'\n'"MCSHIM_IPC_EARLY=$MCSHIM_IPC_EARLY"
    [[ -n "${MCSHIM_IPC_SUSPEND:-}" ]] && CB_ENV+=$'\n'"MCSHIM_IPC_SUSPEND=$MCSHIM_IPC_SUSPEND"
    [[ -n "${MCSHIM_IPC_LOWBAND:-}" ]] && CB_ENV+=$'\n'"MCSHIM_IPC_LOWBAND=$MCSHIM_IPC_LOWBAND"
    [[ -n "${MCSHIM_IPC_SUSPEND_MIN:-}" ]] && CB_ENV+=$'\n'"MCSHIM_IPC_SUSPEND_MIN=$MCSHIM_IPC_SUSPEND_MIN"
    [[ -n "${MCSHIM_IPC_REPLAY_FLOOR:-}" ]] && CB_ENV+=$'\n'"MCSHIM_IPC_REPLAY_FLOOR=$MCSHIM_IPC_REPLAY_FLOOR"
    [[ -n "${NCCL_WIN_ENABLE:-}" ]] && CB_ENV+=$'\n'"NCCL_WIN_ENABLE=$NCCL_WIN_ENABLE"
    [[ -n "${MCSHIM_ALLOC_PAD_MIN:-}" ]] && CB_ENV+=$'\n'"MCSHIM_ALLOC_PAD_MIN=$MCSHIM_ALLOC_PAD_MIN"
    # Diagnostics passthrough: NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NVLS is
    # how to confirm which algorithm NCCL actually selected.
    [[ -n "${NCCL_DEBUG:-}" ]] && CB_ENV+=$'\n'"NCCL_DEBUG=$NCCL_DEBUG"
    [[ -n "${NCCL_DEBUG_SUBSYS:-}" ]] && CB_ENV+=$'\n'"NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS"
    # NCCL-native multicast path (NCCL_CKPT_PATCH=1): LD_PRELOAD the patched
    # libnccl over the torch-bundled one and turn on its checkpoint control
    # thread. The engine is not modified; /sleep already provides the quiesce
    # that ncclCommSuspend requires.
    #
    # GVISOR_CUDA_MULTICAST_SHIM_DIR is what tells the sentry it owns a
    # multicast transition in this container. NCCL's control thread speaks the
    # same marker protocol as the interposer, so pointing that variable at the
    # same directory makes gVisor's existing orchestration drive NCCL --
    # gate/lock/suspend before the checkpoint, rebuild after the restore
    # toggle -- with no change to gVisor itself.
    if [[ "${NCCL_CKPT_PATCH:-0}" = "1" ]]; then
        CB_ENV+=$'\n'"LD_PRELOAD=$NCCL_CKPT_PATCH_PATH"
        # Point vLLM's pynccl at the patched library explicitly, so its
        # ctypes.CDLL loads the one that has the NVLS suspend extension
        # rather than the torch-bundled 2.29.7.
        CB_ENV+=$'\n'"VLLM_NCCL_SO_PATH=$NCCL_CKPT_PATCH_PATH"
        if [[ "${NCCL_ENGINE_HOOK:-0}" = "1" ]]; then
            # Engine-driven: vLLM's sleep/wake_up call the API themselves.
            # Deliberately no NCCL_CKPT_CTRL_DIR (NCCL's control thread stays
            # inert) and no GVISOR_CUDA_MULTICAST_SHIM_DIR (the sentry drives
            # nothing and waits for no acks), so this measures the engine hook
            # alone. gVisor's blocker gate still runs and will name anything
            # left live.
            CB_ENV+=$'\n'"VLLM_NCCL_SUSPEND_HOOK=1"
        else
            CB_ENV+=$'\n'"NCCL_CKPT_CTRL_DIR=$NCCL_CKPT_CTRL_DIR"
            CB_ENV+=$'\n'"GVISOR_CUDA_MULTICAST_SHIM_DIR=$NCCL_CKPT_CTRL_DIR"
        fi
    fi
    cb_write_bundle

    # ── cold boot ─────────────────────────────────────────────────────────
    echo ""
    cb_run_and_wait_health

    # ── reference inference ───────────────────────────────────────────────
    echo ""
    info "Reference inference (pre-checkpoint)"
    _send_chat "$CONTAINER_ID" "Hi" 3 60 >/dev/null || true   # warm up
    sleep 1

    local t0
    t0=$(ts_ms)
    REF_RAW=$(_send_chat "$CONTAINER_ID" "Capital of France? One word only." 5 60 || echo "")
    T_REF_INFER=$(( $(ts_ms) - t0 ))
    REF_ANSWER=$(echo "$REF_RAW" | extract_content)
    if [[ -z "$REF_ANSWER" || "$REF_ANSWER" == "null" ]]; then
        fail "Failed to get reference response"
        echo "    Raw: $REF_RAW"
        tail -30 "$APPLOG_DIR/vllm.log" 2>/dev/null || true
        exit 1
    fi
    ok "\"Capital of France?\" → \"$REF_ANSWER\" (${T_REF_INFER} ms)"

    # Show the CUDA processes the sentry will need to checkpoint.
    info "  CUDA processes on host GPUs:"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
        | sed 's/^/    /' || true

    # ── sleep: quiesce the engine before checkpoint ───────────────────────
    echo ""
    info "Lifecycle: POST /sleep?level=$SLEEP_LEVEL (quiesce engine before checkpoint)"
    SLEEP_RESP=$(_rexec "$CONTAINER_ID" /usr/bin/curl -s --max-time 120 -X POST \
        "http://127.0.0.1:${PORT}/sleep?level=${SLEEP_LEVEL}" 2>/dev/null || echo "FAILED")
    if echo "$SLEEP_RESP" | grep -qi '"sleeping":\s*true\|"status":\s*"ok"'; then
        ok "vLLM sleep($SLEEP_LEVEL) → $SLEEP_RESP"
    else
        warn "vLLM sleep($SLEEP_LEVEL) response: $SLEEP_RESP (continuing anyway)"
    fi
    IS_SLEEPING=$(_rexec "$CONTAINER_ID" /usr/bin/curl -s --max-time 5 \
        "http://127.0.0.1:${PORT}/is_sleeping" 2>/dev/null || echo "UNKNOWN")
    info "  is_sleeping: $IS_SLEEPING"
    sleep 1

    # ── checkpoint (native cuda-checkpoint) ───────────────────────────────
    echo ""
    cb_checkpoint

    GPU_USED_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        -i "$GPU_DEVICES" 2>/dev/null | tr '\n' ' ' || echo "?")
    info "  GPU memory in use after checkpoint (MiB per GPU): $GPU_USED_AFTER"

    # ── restore ───────────────────────────────────────────────────────────
    echo ""
    cb_restore_and_wait_health

    # ── verify the restore landed on the GPUs it was told to ──────────────
    # Done after wake_up would be too late to be meaningful for a cross-GPU
    # run, and before it the engine still holds its devices, so check here.
    echo ""
    info "Verifying GPU placement after restore"
    if ! cb_assert_gpu_placement "${RESTORE_GPU_DEVICES:-$GPU_DEVICES}"; then
        PLACEMENT_OK=0
    else
        PLACEMENT_OK=1
    fi

    # ── wake up ───────────────────────────────────────────────────────────
    echo ""
    WAKE_OK=0
    if [[ "$HEALTH_OK" = "1" ]]; then
        info "Lifecycle: POST /wake_up (resume engine after restore)"
        WAKE_RESP=$(_rexec "$RESTORE_ID" /usr/bin/curl -s --max-time 120 -X POST \
            "http://127.0.0.1:${PORT}/wake_up" 2>/dev/null || echo "FAILED")
        if echo "$WAKE_RESP" | grep -qi '"sleeping":\s*false\|"status":\s*"ok"'; then
            ok "vLLM wake_up → $WAKE_RESP"
            WAKE_OK=1
        else
            warn "vLLM wake_up response: $WAKE_RESP (continuing anyway)"
        fi
        sleep 1
    fi

    # ── first inference after restore ─────────────────────────────────────
    INFER_OK=0
    T_INFER_MS=0
    POST_ANSWER=""
    if [[ "$HEALTH_OK" = "1" ]]; then
        info "Waiting for first inference after restore …"
        local i resp cs
        for i in $(seq 1 240); do
            resp=$(_send_chat "$RESTORE_ID" "Capital of France? One word only." 5 10 || echo "")
            if echo "$resp" | python3 -c "import json,sys; json.load(sys.stdin)['choices']" >/dev/null 2>&1; then
                T_INFER_MS=$(( $(ts_ms) - T_RESTORE_START ))
                POST_ANSWER=$(echo "$resp" | extract_content)
                ok "First inference: ${T_INFER_MS} ms after restore → \"$POST_ANSWER\""
                INFER_OK=1
                break
            fi
            cs=$(cb_state "$RESTORE_ID")
            if [[ "$cs" != "running" ]]; then
                fail "Container exited ($cs) while waiting for inference"
                tail -40 "$APPLOG_DIR/vllm.log" 2>/dev/null || true
                _cb_dump_boot_log
                break
            fi
            (( i % 20 == 0 )) && echo "    $(( i / 2 ))s … waiting (container=$cs)"
            sleep 0.5
        done
        if [[ "$INFER_OK" = "0" ]]; then
            fail "Inference not available after restore"
            tail -40 "$APPLOG_DIR/vllm.log" 2>/dev/null || true
        fi
    fi

    # ── GPU placement check (only does anything for cross-GPU restores) ────
    echo ""
    cb_verify_gpu_placement 500

    # ── verification queries ──────────────────────────────────────────────
    echo ""
    ANSWER_MATCH="N/A"
    VERIFY_OK=0
    if [[ "$INFER_OK" = "1" ]]; then
        info "Verification"
        ANSWER_MATCH="MISMATCH"
        [[ "$REF_ANSWER" == "$POST_ANSWER" ]] && ANSWER_MATCH="EXACT MATCH"
        echo "    Pre:  \"$REF_ANSWER\""
        echo "    Post: \"$POST_ANSWER\"  → $ANSWER_MATCH"

        echo ""
        local all=1 q raw ans
        for q in \
            "What is 7 times 8? Just the number." \
            "What color is the sky? One word." \
            "Name the largest planet in our solar system. One word."
        do
            raw=$(_send_chat "$RESTORE_ID" "$q" 10 30 || echo "")
            ans=$(echo "$raw" | extract_content 2>/dev/null || echo "")
            if [[ -z "$ans" || "$ans" == "null" ]]; then
                fail "\"$q\" → FAILED"
                all=0
            else
                ok "\"$q\" → \"$ans\""
            fi
        done
        if [[ "$all" = "1" ]]; then
            VERIFY_OK=1
            ok "All queries passed"
        fi
    fi

    # ── cuda-checkpoint timings from sentry logs ──────────────────────────
    echo ""
    cb_cuda_ckpt_log_summary

    # ── summary ───────────────────────────────────────────────────────────
    echo ""
    banner "╔══════════════════════════════════════════════════════════════════╗"
    banner "$BENCH_BANNER"
    banner "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    row "GPU:"                            "$GPU_NAME, driver $HOST_DRIVER_VER"
    row "GPUs / TP:"                      "$GPU_DEVICES / $TP"
    [[ -n "${RESTORE_GPU_DEVICES:-}" ]] && \
    row "Restored on GPUs:"               "$RESTORE_GPU_DEVICES (placement verified: $([ "${PLACEMENT_OK:-1}" = 1 ] && echo 'YES ✓' || echo 'NO ✗'))"
    row "Model:"                          "$MODEL"
    row "CUDA graphs + torch.compile:"    "$([ "$EAGER" = 1 ] && echo no || echo yes)"
    row "Sleep level:"                    "$SLEEP_LEVEL"
    echo ""
    row "Cold boot (run → health):"       "${T_COLD_BOOT} ms"
    row "Pre-checkpoint inference:"       "${T_REF_INFER} ms"
    row "runsc checkpoint (incl. GPU):"   "${T_CHECKPOINT} ms"
    row "runsc restore returned:"         "${T_RESTORE_RETURNED} ms"
    row "Health after restore:"           "${T_HEALTH_MS} ms"
    [[ "$INFER_OK" = "1" ]] && \
    row "First inference after restore:"  "${T_INFER_MS} ms"
    row "Checkpoint size (total):"        "$TOTAL_SIZE (pages: $PAGES_SIZE)"
    echo ""
    row "Sleep before checkpoint:"        "level $SLEEP_LEVEL"
    row "Wake_up after restore:"          "$([ "$WAKE_OK" = 1 ] && echo 'YES ✓' || echo 'NO ✗')"
    row "Answer match (pre vs post):"     "$ANSWER_MATCH"
    row "Server functional:"              "$([ "$VERIFY_OK" = 1 ] && echo 'YES ✓' || echo 'NO ✗')"
    if [[ "$INFER_OK" = "1" && "$T_INFER_MS" -gt 0 ]]; then
        row "Speedup vs cold boot:"       "$(speedup "$T_COLD_BOOT" "$T_INFER_MS") (to first inference)"
    fi
    echo ""

    if [[ "$INFER_OK" = "1" && "$VERIFY_OK" = "1" && "${PLACEMENT_OK:-1}" = "1" ]]; then
        banner "RESULT: PASS ✓"
        return 0
    else
        banner "RESULT: FAIL ✗"
        return 1
    fi
}
