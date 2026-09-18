#!/usr/bin/env bash
# FAILED-SAVE RECOVERY trial (the cudaSaveFailedKey path).
#
# Boot SGLang TP=4 (stock: custom AR on, torch.compile off), pause-quiesce so
# the FLA registrations stay LIVE (n>0 host-freed in the suspend window),
# then `checkpoint --leave-running` into a 2 GiB tmpfs so the save fails
# deterministically in encoding (ENOSPC), AFTER preSaveCuda has run. What
# matters is what happens next: postResumeCuda must take the failure-recovery
# branch (replay the host-freed FLA registrations, resume the interposer,
# reopen the admission gate) and the ORIGINAL container must serve correct
# inference afterwards.
#
#   sudo RUNSC=/usr/local/bin/runsc-r580 CUDA_MULTICAST_SHIM=1 bash e2e_recovery.sh
set -uo pipefail
cd /home/ubuntu/gvisor/cr-bench
export GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"
source /home/ubuntu/gvisor/cr-bench/common.sh

BENCH_NAME="cr-bench-sglang-rec"
IMAGE="cr-bench-sglang"
TP=4
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
APP_LOG="/applog/sglang.log"

cb_init
cb_detect_gpu
cb_runsc_flags
cb_prepare_rootfs

CB_CMD="exec python3 -m sglang.launch_server \
--model-path $MODEL --host 0.0.0.0 --port $PORT --tp-size $TP \
--mem-fraction-static 0.7 --context-length 2048 --dtype float16 \
--attention-backend triton --sampling-backend pytorch \
>$APP_LOG 2>&1"
CB_ENV="$(cb_common_env)"
cb_write_bundle
cb_run_and_wait_health

verdict=FAIL
finish() {
    echo "[recovery] verdict=$verdict rundir=$BASE_DIR"
    sudo umount "$CKPT_DIR" 2>/dev/null || true
    cb_cleanup 2>/dev/null || true
    [ "$verdict" = PASS ] && banner "RESULT: PASS ✓" || banner "RESULT: FAIL ✗"
}
trap finish EXIT

info "=== pre-checkpoint inference ==="
cb_infer "$CONTAINER_ID" pre || exit 1

info "=== pause_generation (weights + KV resident; FLAs live) ==="
_post "$CONTAINER_ID" /pause_generation >/dev/null 2>&1 || true
sleep 2

info "=== checkpoint --leave-running into a 2 GiB tmpfs (expected: ENOSPC in encoding) ==="
sudo mount -t tmpfs -o size=2g tmpfs "$CKPT_DIR" || { fail "tmpfs mount"; exit 1; }
rc=0
"$RUNSC" "${RUNSC_FLAGS[@]}" checkpoint \
    --image-path="$CKPT_DIR" --compression=none --exclude-committed-zero-pages \
    --leave-running \
    --cuda-checkpoint-path="$CUDA_CHECKPOINT_PATH" --cuda-checkpoint-sequential \
    "$CONTAINER_ID" >"$LOG_DIR/runsc-checkpoint.log" 2>&1 || rc=$?
if [ "$rc" -eq 0 ]; then
    warn "checkpoint unexpectedly SUCCEEDED; recovery path not exercised"; exit 1
fi
ok "checkpoint failed as expected (rc=$rc): $(grep -m1 -o 'checkpoint failed.*' "$LOG_DIR/runsc-checkpoint.log" | cut -c1-160)"

info "=== recovery evidence in sentry log ==="
BOOTLOG="$(ls -t "$LOG_DIR"/runsc.log.*boot.txt 2>/dev/null | head -1)"
sudo grep -q "host-freed FLA registration" "$BOOTLOG" \
    && ok "FLA suspend fired (n>0)" || { fail "no FLA suspend"; exit 1; }
if sudo grep -q "replayed [0-9]* FLA registrations after failed save" "$BOOTLOG" \
    || sudo grep -q "dropped FLA registration.*client torn down" "$BOOTLOG"; then
    ok "failed-save recovery branch fired (replay/drop)"
else
    fail "no failed-save recovery evidence in log"; exit 1
fi
sudo grep -q "Failed to resume CUDA processes after failed save" "$BOOTLOG" \
    && { fail "recovery errored in sentry log"; exit 1; } || ok "recovery clean in sentry log"
sudo grep -q "pending after resume" "$BOOTLOG" \
    && { fail "scope guard misfired on recovery path"; exit 1; } || ok "scope guard silent (correct)"
sudo grep -q "RESUME done" "$APPLOG_DIR/sglang.log" \
    && ok "interposer resumed" || { fail "interposer did not resume"; exit 1; }

info "=== a process must be admitted after the failed save (gate reopened) ==="
# cuInit in a fresh process inside the container: blocks forever if the
# admission gate was left closed.
# (`timeout` cannot run a shell function, hence the explicit runsc exec.)
if timeout 60 "$RUNSC" --root "$RUNSC_ROOT" exec "$CONTAINER_ID" python3 -c "import ctypes; c=ctypes.CDLL('libcuda.so.1'); print('cuInit rc', c.cuInit(0))" | grep -q 'cuInit rc 0'; then
    ok "fresh process initialized CUDA (admission gate open)"
else
    fail "fresh process could not initialize CUDA after failed save (admission gate stuck?)"; exit 1
fi

info "=== continue_generation + post-recovery inference on ORIGINAL container ==="
_post "$CONTAINER_ID" /continue_generation >/dev/null 2>&1 || true
sleep 3
cb_infer "$CONTAINER_ID" post-recovery || { fail "post-recovery inference wrong"; exit 1; }
ok "container fully recovered from failed save"
verdict=PASS
