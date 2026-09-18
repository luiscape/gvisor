#!/usr/bin/env bash
# Gate trial 7: FAILED-SAVE RECOVERY (the cudaSaveFailedKey marker path).
#
# Boot SGLang TP=4 with FlashInfer fusion engaged, pause-quiesce (fusion
# workspace FLA registrations stay LIVE), then checkpoint with
# --leave-running. The save deterministically fails in encoding (the
# pre-existing frontendFDMemmapFile pma limitation with resident device
# mappings). The point of the trial is what happens NEXT: postResumeCuda
# must take the failure-recovery branch (marker present) -- replay the
# host-freed FLA registrations, resume the interposer -- and the ORIGINAL
# container must serve correct inference afterwards.
set -uo pipefail
cd /home/ubuntu/gvisor/cr-bench

source /home/ubuntu/gvisor/cr-bench/common.sh

BENCH_NAME="cr-bench-sglang-rec"
IMAGE="cr-bench-sglang"
DOCKERFILE="images/Dockerfile.sglang"
PORT=8199
CB_GPU=1
GPU_DEVICES="0,1,2,3"
TP=4
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
APP_LOG="/applog/sglang.log"
OUT=/data/e2e_gate_20260820_034759

cb_init
cb_detect_gpu
cb_runsc_flags
cb_prepare_rootfs

CB_CMD="exec python3 -m sglang.launch_server \
--model-path $MODEL --host 0.0.0.0 --port $PORT --tp-size $TP \
--mem-fraction-static 0.7 --context-length 2048 --dtype float16 \
--attention-backend triton --sampling-backend pytorch \
--flashinfer-allreduce-fusion-backend trtllm \
>$APP_LOG 2>&1"
CB_ENV="HF_HOME=/app/hf_cache
HF_HUB_OFFLINE=1
NCCL_CUMEM_ENABLE=1
LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib"
cb_write_bundle

cb_run_and_wait_health

verdict=FAIL
finish() {
    echo "sglang_tp4_recovery $verdict" >>"$OUT/verdicts.txt"
    echo "[gate-t7] verdict=$verdict rundir=$BASE_DIR" >>"$OUT/summary.txt"
    cb_cleanup 2>/dev/null || true
}
trap finish EXIT

ask() { # $1 request tag
    _rexec "$CONTAINER_ID" /usr/bin/curl -s --max-time 120 \
        -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Capital of France? One word.\"}],\"max_tokens\":8}"
}

info "=== pre-checkpoint inference ==="
R1="$(ask pre)"
echo "$R1" | grep -qi paris || { fail "pre-checkpoint inference wrong: $R1"; exit 1; }
ok "pre-checkpoint inference OK"

info "=== fusion engagement check ==="
if ! grep -q "AllReduce Fusion enabled" "$APPLOG_DIR/sglang.log"; then
    fail "fusion did not engage; trial invalid"; exit 1
fi
ok "fusion engaged"

info "=== pause_generation (keep FLAs live) ==="
_rexec "$CONTAINER_ID" /usr/bin/curl -s --max-time 60 -X POST \
    -H "Content-Type: application/json" -d '{}' \
    "http://127.0.0.1:${PORT}/pause_generation" || true

info "=== checkpoint --leave-running (expected to FAIL in encoding) ==="
rc=0
"$RUNSC" "${RUNSC_FLAGS[@]}" checkpoint \
    --image-path="$CKPT_DIR" --compression=none --exclude-committed-zero-pages \
    --leave-running \
    --cuda-checkpoint-path="$CUDA_CHECKPOINT_PATH" --cuda-checkpoint-sequential \
    "$CONTAINER_ID" >"$LOG_DIR/runsc-checkpoint.log" 2>&1 || rc=$?
if [[ "$rc" -eq 0 ]]; then
    warn "checkpoint unexpectedly SUCCEEDED; recovery path not exercised"
    exit 1
fi
ok "checkpoint failed as expected (rc=$rc)"
grep -q "frontendFDMemmapFile" "$LOG_DIR/runsc-checkpoint.log" \
    || warn "failure signature differs from expected pma limitation"

info "=== recovery evidence in sentry log ==="
BOOTLOG="$(ls -t "$LOG_DIR"/runsc.log.*boot.txt 2>/dev/null | head -1)"
grep -q "host-freed FLA registration" "$BOOTLOG" \
    && ok "FLA suspend fired (n>0)" || { fail "no FLA suspend"; exit 1; }
if grep -q "replayed [0-9]* FLA registrations after failed save" "$BOOTLOG" \
    || grep -q "dropped FLA registration.*client torn down" "$BOOTLOG"; then
    ok "failed-save recovery branch fired (replay/drop)"
else
    fail "no failed-save recovery evidence in log"; exit 1
fi
grep -q "Failed to resume CUDA processes after failed save" "$BOOTLOG" \
    && { fail "recovery errored in sentry log"; exit 1; } \
    || ok "recovery clean in sentry log"
grep -q "pending after resume" "$BOOTLOG" \
    && { fail "scope guard misfired on recovery path"; exit 1; } \
    || ok "scope guard silent (correct)"

info "=== continue_generation + post-recovery inference on ORIGINAL container ==="
_rexec "$CONTAINER_ID" /usr/bin/curl -s --max-time 60 -X POST \
    -H "Content-Type: application/json" -d '{}' \
    "http://127.0.0.1:${PORT}/continue_generation" || true
sleep 3
R2="$(ask post)"
echo "$R2" | grep -qi paris || { fail "post-recovery inference wrong: $R2"; exit 1; }
ok "post-recovery inference OK -- container fully recovered from failed save"

verdict=PASS
