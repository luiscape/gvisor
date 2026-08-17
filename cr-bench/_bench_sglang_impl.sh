#!/usr/bin/env bash
# --------------------------------------------------------------------------
#  _bench_sglang_impl.sh — shared implementation for the SGLang checkpoint /
#  restore benchmarks (bench_5_sglang_single.sh and bench_6_sglang_multi.sh).
#
#  SGLang counterpart of _bench_vllm_impl.sh, using the same OpenAI-
#  compatible chat API for verification.  Unlike vLLM, SGLang ships native
#  lifecycle endpoints, so no wrapper script is needed:
#
#    POST /pause_generation     quiesce the scheduler before checkpoint
#                               (in-flight generation drained/queued; GPUs idle)
#    POST /continue_generation  resume after restore
#
#  Lifecycle:
#    1. runsc run           → wait for /health          (cold boot, timed)
#    2. reference inference (deterministic, temperature=0)
#    3. POST /pause_generation, then (default --quiesce release)
#       POST /release_memory_occupation: weights → CPU (cpu-backup),
#       idle KV cache discarded, minimizing GPU state to stage
#    4. runsc checkpoint --cuda-checkpoint-path=…
#                           → sentry runs cuda-checkpoint --toggle on
#                             EVERY CUDA process (HTTP server + TP
#                             scheduler workers) in parallel, then
#                             serializes the sandbox
#    5. runsc restore       → sentry re-toggles all CUDA processes
#                             (onto different GPUs if --restore-gpus)
#    6. POST /resume_memory_occupation (weights → GPU, re-alloc KV cache),
#       then POST /continue_generation (both BEFORE waiting for health:
#       SGLang's /health reflects detokenizer heartbeats, which stay
#       stopped while paused)
#    7. wait /health
#    8. first inference (timed) + verification queries
#
#  CUDA graph capture and torch.compile are ON by default: skipping that
#  warmup on restore is the main use-case for C/R.
#
#  The caller (wrapper script) must set: BENCH_NAME, TP, GPU_DEVICES,
#  MODEL, and may override any knob below.  Do not run this directly.
# --------------------------------------------------------------------------
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# ── Configuration (defaults; wrappers/CLI may override) ───────────────────
IMAGE="${IMAGE:-cr-bench-sglang}"
DOCKERFILE="images/Dockerfile.sglang"
PORT="${PORT:-8199}"
CONTEXT_LEN="${CONTEXT_LEN:-2048}"
MEM_FRAC="${MEM_FRAC:-0.7}"
CB_SHM_MB="${CB_SHM_MB:-8192}"
CB_GPU=1
APP_LOG="/applog/sglang.log"
# Quiesce mode before checkpoint (default "release"): pause generation AND
# POST /release_memory_occupation — weights are backed up to CPU
# (--enable-weights-cpu-backup) and the idle KV cache is discarded, so
# cuda-checkpoint stages minimal GPU memory. After restore,
# /resume_memory_occupation moves weights back to the GPU and re-allocates
# the KV cache. "pause" only pauses the scheduler (GPU memory retained).
QUIESCE="${QUIESCE:-release}"
# CUDA graph capture (default on) and torch.compile (default on): skipping
# this expensive warmup on restore is the main C/R use-case.
TORCH_COMPILE="${TORCH_COMPILE:-1}"
EAGER="${EAGER:-0}"
# NCCL knobs (multi-GPU) — same rationale as the vLLM benchmarks, including
# the same corrected default: 1 puts NCCL's P2P buffers on the VMM API, which
# the interposer restores at identical addresses; 0 puts them on legacy CUDA
# IPC, which rides the driver's intermittent live-import path. See
# SLEEP_CHECKPOINT_WORKFLOW.md.
# (Old rationale, kept for the record: 0 forces classic allocations,
# of VMM allocations is driver-dependent).
NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"

sglang_parse_flags() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --gpus)            GPU_DEVICES="$2"; shift 2 ;;
            --restore-gpus)    RESTORE_GPU_DEVICES="$2"; shift 2 ;;
            --tp)              TP="$2"; shift 2 ;;
            --model)           MODEL="$2"; shift 2 ;;
            --context-len)     CONTEXT_LEN="$2"; shift 2 ;;
            --mem-frac)        MEM_FRAC="$2"; shift 2 ;;
            --port)            PORT="$2"; shift 2 ;;
            --quiesce)         QUIESCE="$2"; shift 2 ;;
            --no-torch-compile) TORCH_COMPILE=0; shift ;;
            --eager)           EAGER=1; shift ;;
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
                echo "  --context-len N        max context length (default: 2048)"
                echo "  --mem-frac F           static memory fraction (default: 0.7)"
                echo "  --port PORT            SGLang listen port (default: 8199)"
                echo "  --quiesce MODE         release: offload weights to CPU + drop KV cache"
                echo "                         (default), pause: pause scheduler only"
                echo "  --no-torch-compile     disable torch.compile"
                echo "  --eager                disable CUDA graph capture"
                echo "  --compression MODE     none | flate-best-speed (default: none)"
                echo "  --no-exclude-zero      keep committed zero pages"
                echo "  --sequential           run cuda-checkpoint sequentially (debugging)"
                echo "  --image IMAGE          Docker image name (default: cr-bench-sglang)"
                echo "  --rebuild-rootfs       force re-extract rootfs from image"
                exit 0 ;;
            *) echo "Unknown flag: $1"; exit 1 ;;
        esac
    done
}

# ── Inference helpers (same OpenAI-compatible API as vLLM) ────────────────
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

sglang_run() {
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
    info "Model:           $MODEL (context $CONTEXT_LEN, mem-frac $MEM_FRAC)"
    info "CUDA graphs:     $([ "$EAGER" = 1 ] && echo disabled || echo enabled), torch.compile: $([ "$TORCH_COMPILE" = 1 ] && echo enabled || echo disabled)"
    info "Quiesce:         $QUIESCE $([ "$QUIESCE" = release ] && echo '(weights → CPU, KV cache discarded)')"
    info "Lifecycle:       /pause_generation + /continue_generation (SGLang native)"
    info "cuda-checkpoint: $CUDA_CHECKPOINT_PATH ($([ "$CUDA_CKPT_SEQUENTIAL" = 1 ] && echo sequential || echo parallel))"
    echo ""

    # ── rootfs + bundle ───────────────────────────────────────────────────
    cb_prepare_rootfs

    # SGLang's --enable-memory-saver OVERWRITES LD_PRELOAD when spawning its
    # TP workers (it needs its own hook library there), which silently drops
    # the interposer from exactly the processes that hold the shared-memory
    # state. gVisor's --cuda-multicast-shim-path only touches the initial
    # process's env, so it cannot help here. /etc/ld.so.preload is immune to
    # env overwrites, so inject the interposer that way, into this run's
    # PRIVATE overlay (never the shared rootfs cache).
    # SKIP_LDSO_INJECT=1 leaves this to the sentry, which (new) writes
    # /etc/ld.so.preload itself through the container's VFS -- set it to A/B
    # the sentry-side injection against this harness-side one.
    if [[ "${CUDA_MULTICAST_SHIM:-0}" = "1" && "${SKIP_LDSO_INJECT:-0}" != "1" ]]; then
        echo "$CUDA_MULTICAST_SHIM_PATH" | sudo tee "$ROOTFS_MERGED/etc/ld.so.preload" >/dev/null
        ok "Injected interposer via /etc/ld.so.preload (memory-saver overwrites LD_PRELOAD)"
    fi

    # triton attention + pytorch sampling: avoid FlashInfer JIT (needs
    # nvcc/nvrtc at runtime; same reasoning as vLLM benchmarks).
    local extra_args=""
    if [[ "$EAGER" = "1" ]]; then
        extra_args+=" --disable-cuda-graph"
    fi
    if [[ "$TORCH_COMPILE" = "1" ]]; then
        extra_args+=" --enable-torch-compile"
    fi
    if [[ "$QUIESCE" = "release" ]]; then
        extra_args+=" --enable-memory-saver --enable-weights-cpu-backup"
    fi
    CB_CMD="exec python3 -m sglang.launch_server \
--model-path $MODEL \
--host 0.0.0.0 \
--port $PORT \
--tp-size $TP \
--mem-fraction-static $MEM_FRAC \
--context-length $CONTEXT_LEN \
--dtype float16 \
--attention-backend triton \
--sampling-backend pytorch \
$extra_args \
>$APP_LOG 2>&1"
    # LD_LIBRARY_PATH addition: --enable-memory-saver LD_PRELOADs the
    # torch_memory_saver hook library, which links libcudart.so.13; that
    # ships in torch's pip 'nvidia' packages but is not on the default
    # loader path (common.sh appends this to the base LD_LIBRARY_PATH).
    CB_ENV="HF_HOME=/app/hf_cache
HF_HUB_OFFLINE=1
NCCL_CUMEM_ENABLE=$NCCL_CUMEM_ENABLE
LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib"
    # Interposer knobs, same as the vLLM impl. Without the passthrough these
    # env-gated features silently never fire inside the sandbox.
    [[ -n "${MCSHIM_IPC_SUSPEND:-}" ]] && CB_ENV+=$'\n'"MCSHIM_IPC_SUSPEND=$MCSHIM_IPC_SUSPEND"
    [[ -n "${MCSHIM_IPC_REPLAY_FLOOR:-}" ]] && CB_ENV+=$'\n'"MCSHIM_IPC_REPLAY_FLOOR=$MCSHIM_IPC_REPLAY_FLOOR"
    [[ -n "${MCSHIM_VERBOSE:-}" ]] && CB_ENV+=$'\n'"MCSHIM_VERBOSE=$MCSHIM_VERBOSE"
    [[ -n "${NCCL_DEBUG:-}" ]] && CB_ENV+=$'\n'"NCCL_DEBUG=$NCCL_DEBUG"
    [[ -n "${NCCL_DEBUG_SUBSYS:-}" ]] && CB_ENV+=$'\n'"NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS"
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
        tail -30 "$APPLOG_DIR/sglang.log" 2>/dev/null || true
        exit 1
    fi
    ok "\"Capital of France?\" → \"$REF_ANSWER\" (${T_REF_INFER} ms)"

    # Show the CUDA processes the sentry will need to checkpoint.
    info "  CUDA processes on host GPUs:"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
        | sed 's/^/    /' || true

    # ── pause: quiesce the engine before checkpoint ───────────────────────
    echo ""
    info "Lifecycle: POST /pause_generation (quiesce engine before checkpoint)"
    # Note: recent SGLang requires a JSON body on these endpoints (400
    # "Field required" otherwise).
    PAUSE_RESP=$(_rexec "$CONTAINER_ID" /usr/bin/curl -s --max-time 120 -X POST \
        -H "Content-Type: application/json" -d '{}' \
        "http://127.0.0.1:${PORT}/pause_generation" 2>/dev/null || echo "FAILED")
    if echo "$PAUSE_RESP" | grep -qi '"status":\s*"ok"\|paused'; then
        ok "SGLang pause_generation → $PAUSE_RESP"
    else
        warn "SGLang pause_generation response: $PAUSE_RESP (continuing anyway)"
    fi

    if [[ "$QUIESCE" = "release" ]]; then
        # Back up weights to CPU and discard the idle KV cache so
        # cuda-checkpoint stages minimal GPU memory. tags=null releases both
        # 'weights' (CPU-backed via --enable-weights-cpu-backup) and
        # 'kv_cache' (discarded; it is idle after pause_generation).
        info "Lifecycle: POST /release_memory_occupation (weights → CPU, discard KV cache)"
        local t_rel0
        t_rel0=$(ts_ms)
        RELEASE_RESP=$(_rexec "$CONTAINER_ID" /usr/bin/curl -s --max-time 300 -X POST \
            -H "Content-Type: application/json" -d '{}' \
            "http://127.0.0.1:${PORT}/release_memory_occupation" 2>/dev/null || echo "FAILED")
        if echo "$RELEASE_RESP" | grep -qiv "FAILED\|error"; then
            ok "SGLang release_memory_occupation → ${RELEASE_RESP:-200 OK} ($(( $(ts_ms) - t_rel0 )) ms)"
        else
            warn "SGLang release_memory_occupation response: $RELEASE_RESP (continuing anyway)"
        fi
        GPU_USED_RELEASED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            -i "$GPU_DEVICES" 2>/dev/null | tr '\n' ' ' || echo "?")
        info "  GPU memory in use after release (MiB per GPU): $GPU_USED_RELEASED"
    fi
    sleep 1

    # ── checkpoint (native cuda-checkpoint) ───────────────────────────────
    echo ""
    cb_checkpoint

    GPU_USED_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        -i "$GPU_DEVICES" 2>/dev/null | tr '\n' ' ' || echo "?")
    info "  GPU memory in use after checkpoint (MiB per GPU): $GPU_USED_AFTER"

    # ── restore ─────────────────────────────────────────────────────────────────
    echo ""
    cb_restore

    # ── resume memory + continue BEFORE waiting for health ───────────────
    # SGLang's /health reflects detokenizer heartbeats, which stay stopped
    # while generation is paused — health cannot recover until we resume.
    # The HTTP server answers soon after restore even while the scheduler is
    # still mid-GPU-restore (~15s), so retry until the requests land.
    echo ""
    RESUME_OK=0
    local rt0 relapsed
    rt0=$(ts_ms)
    T_MEMRESUME_MS=0
    if [[ "$QUIESCE" = "release" ]]; then
        # Move weights back to the GPU and re-allocate the KV cache.
        info "Lifecycle: POST /resume_memory_occupation (weights → GPU, re-alloc KV cache)"
        while :; do
            MEMRESUME_RESP=$(_rexec "$RESTORE_ID" /usr/bin/curl -s --max-time 300 -X POST \
                -H "Content-Type: application/json" -d '{}' \
                "http://127.0.0.1:${PORT}/resume_memory_occupation" 2>/dev/null || echo "FAILED")
            if echo "$MEMRESUME_RESP" | grep -qiv "FAILED\|error"; then
                T_MEMRESUME_MS=$(( $(ts_ms) - rt0 ))
                ok "SGLang resume_memory_occupation → ${MEMRESUME_RESP:-200 OK} (${T_MEMRESUME_MS} ms after restore returned)"
                break
            fi
            relapsed=$(( $(ts_ms) - rt0 ))
            if (( relapsed > 120000 )); then
                warn "resume_memory_occupation not accepted after 120s: $MEMRESUME_RESP"
                break
            fi
            sleep 1
        done
    fi
    info "Lifecycle: POST /continue_generation (resume engine after restore)"
    while :; do
        RESUME_RESP=$(_rexec "$RESTORE_ID" /usr/bin/curl -s --max-time 10 -X POST \
            -H "Content-Type: application/json" -d '{}' \
            "http://127.0.0.1:${PORT}/continue_generation" 2>/dev/null || echo "FAILED")
        if echo "$RESUME_RESP" | grep -qi '"status":\s*"ok"\|resumed\|continued'; then
            ok "SGLang continue_generation → $RESUME_RESP ($(( $(ts_ms) - rt0 )) ms after restore returned)"
            RESUME_OK=1
            break
        fi
        relapsed=$(( $(ts_ms) - rt0 ))
        if (( relapsed > 60000 )); then
            warn "continue_generation not accepted after 60s: $RESUME_RESP"
            break
        fi
        sleep 1
    done

    # ── wait for health ─────────────────────────────────────────────────
    echo ""
    cb_wait_health_restored

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
                tail -40 "$APPLOG_DIR/sglang.log" 2>/dev/null || true
                _cb_dump_boot_log
                break
            fi
            (( i % 20 == 0 )) && echo "    $(( i / 2 ))s … waiting (container=$cs)"
            sleep 0.5
        done
        if [[ "$INFER_OK" = "0" ]]; then
            fail "Inference not available after restore"
            tail -40 "$APPLOG_DIR/sglang.log" 2>/dev/null || true
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
    row "CUDA graphs / torch.compile:"    "$([ "$EAGER" = 1 ] && echo no || echo yes) / $([ "$TORCH_COMPILE" = 1 ] && echo yes || echo no)"
    row "Quiesce mode:"                   "$QUIESCE"
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
    row "Pause before checkpoint:"        "/pause_generation$([ "$QUIESCE" = release ] && echo ' + /release_memory_occupation')"
    if [[ "$QUIESCE" = "release" && "$T_MEMRESUME_MS" -gt 0 ]]; then
        row "Memory resume after restore:"   "${T_MEMRESUME_MS} ms"
    fi
    row "Continue after restore:"         "$([ "$RESUME_OK" = 1 ] && echo 'YES ✓' || echo 'NO ✗')"
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
