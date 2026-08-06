#!/usr/bin/env bash
# --------------------------------------------------------------------------
#  bench_1_cpu.sh — CPU memory checkpoint/restore benchmark (simple case)
#
#  Boots a container that fills MEM_MB of anonymous memory with
#  incompressible random data, checkpoints it with runsc, restores it,
#  and verifies the memory contents survived bit-exactly (sha256 of every
#  buffer, compared pre vs post).
#
#  No GPU, no nvproxy, no cuda-checkpoint — this is the baseline that the
#  GPU benchmarks build on.
#
#  Usage:
#    sudo bash cr-bench/bench_1_cpu.sh
#    sudo bash cr-bench/bench_1_cpu.sh --mem-mb 4096
#
#  Prerequisites: runsc at $RUNSC (default /usr/local/bin/runsc), docker.
# --------------------------------------------------------------------------
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# ── Configuration ─────────────────────────────────────────────────────────
BENCH_NAME="cr-bench-cpu"
IMAGE="${IMAGE:-cr-bench-cpu}"
DOCKERFILE="images/Dockerfile.cpu"
PORT="${PORT:-8199}"
MEM_MB="${MEM_MB:-1024}"
NUM_BUFFERS="${NUM_BUFFERS:-8}"
CB_GPU=0
APP_LOG="/applog/app.log"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mem-mb)          MEM_MB="$2"; shift 2 ;;
        --num-buffers)     NUM_BUFFERS="$2"; shift 2 ;;
        --port)            PORT="$2"; shift 2 ;;
        --compression)     COMPRESSION="$2"; shift 2 ;;
        --no-exclude-zero) EXCLUDE_ZERO=0; shift ;;
        --image)           IMAGE="$2"; shift 2 ;;
        --rebuild-rootfs)  REBUILD_ROOTFS=1; shift ;;
        --help|-h)
            echo "Usage: $0 [--mem-mb MB] [--num-buffers N] [--port P]"
            echo "          [--compression none|flate-best-speed] [--no-exclude-zero]"
            echo "          [--image NAME] [--rebuild-rootfs]"
            exit 0 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

cb_init
cb_runsc_flags

banner ""
banner "╔══════════════════════════════════════════════════════════════════╗"
banner "║   Benchmark 1: CPU memory checkpoint / restore (pure runsc)     ║"
banner "╚══════════════════════════════════════════════════════════════════╝"
echo ""
info "runsc:        $($RUNSC --version 2>&1 | head -1 || echo '?')"
info "Memory:       ${MEM_MB} MiB in ${NUM_BUFFERS} buffers"
info "Compression:  $COMPRESSION"
echo ""

# ── Phase 0/1: rootfs + bundle ────────────────────────────────────────────
cb_prepare_rootfs

CB_CMD="exec python3 /app/cpu_mem_server.py >$APP_LOG 2>&1"
CB_ENV="MEM_MB=$MEM_MB
NUM_BUFFERS=$NUM_BUFFERS
PORT=$PORT"
GPU_DEVICES=""
cb_write_bundle

# ── Phase 2: cold boot ────────────────────────────────────────────────────
echo ""
cb_run_and_wait_health

# ── Phase 3: reference checksums ──────────────────────────────────────────
echo ""
info "Reference checksums (pre-checkpoint)"
REF_SUMS=$(cb_curl "$CONTAINER_ID" "http://127.0.0.1:${PORT}/checksums" || echo "")
if [[ -z "$REF_SUMS" ]]; then
    fail "Failed to fetch reference checksums"; exit 1
fi
ok "Got $(echo "$REF_SUMS" | python3 -c 'import json,sys; print(json.load(sys.stdin)["num_buffers"])') buffer checksums"

# ── Phase 4: checkpoint ───────────────────────────────────────────────────
echo ""
cb_checkpoint

# ── Phase 5: restore ──────────────────────────────────────────────────────
echo ""
cb_restore_and_wait_health

# ── Phase 6: verification ─────────────────────────────────────────────────
echo ""
info "Verification"
VERIFY_OK=0
POST_SUMS=$(cb_curl "$RESTORE_ID" "http://127.0.0.1:${PORT}/checksums" || echo "")
if [[ -z "$POST_SUMS" ]]; then
    fail "Failed to fetch post-restore checksums"
elif [[ "$(echo "$REF_SUMS" | python3 -c 'import json,sys; print(json.load(sys.stdin)["checksums"])')" \
     == "$(echo "$POST_SUMS" | python3 -c 'import json,sys; print(json.load(sys.stdin)["checksums"])')" ]]; then
    ok "All ${NUM_BUFFERS} buffer checksums match EXACTLY"
    VERIFY_OK=1
else
    fail "Checksum MISMATCH"
    echo "    Pre:  $REF_SUMS"
    echo "    Post: $POST_SUMS"
fi

TOUCH_OK=0
TOUCH_RESP=$(cb_curl "$RESTORE_ID" "http://127.0.0.1:${PORT}/touch" || echo "")
if echo "$TOUCH_RESP" | grep -q '"ok": true'; then
    ok "Memory writable + readable after restore"
    TOUCH_OK=1
else
    fail "Post-restore memory touch failed: $TOUCH_RESP"
fi

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
banner "╔══════════════════════════════════════════════════════════════════╗"
banner "║   Benchmark 1 (CPU memory C/R) — Results                        ║"
banner "╚══════════════════════════════════════════════════════════════════╝"
echo ""
row "Memory:"                        "${MEM_MB} MiB / ${NUM_BUFFERS} buffers"
row "Cold boot (run → health):"      "${T_COLD_BOOT} ms"
row "runsc checkpoint:"              "${T_CHECKPOINT} ms"
row "runsc restore returned:"        "${T_RESTORE_RETURNED} ms"
row "Health after restore:"          "${T_HEALTH_MS} ms"
row "Checkpoint size (total):"       "$TOTAL_SIZE (pages: $PAGES_SIZE)"
row "Checksums match:"               "$([ "$VERIFY_OK" = 1 ] && echo 'YES ✓' || echo 'NO ✗')"
row "Memory live after restore:"     "$([ "$TOUCH_OK" = 1 ] && echo 'YES ✓' || echo 'NO ✗')"
echo ""

if [[ "$VERIFY_OK" = "1" && "$TOUCH_OK" = "1" ]]; then
    banner "RESULT: PASS ✓"
    exit 0
else
    banner "RESULT: FAIL ✗"
    exit 1
fi
