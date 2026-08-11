#!/usr/bin/env bash
# run_matrix.sh -- acceptance matrix for the gVisor-driven multicast interposer.
#
# Each row is one full checkpoint/restore of stock NCCL under runsc. Knobs:
#   NO_PAUSE=1   workload never stops (the production shape; the interposer's
#                suspend gate is what keeps it off torn-down multicast VAs)
#   INTERVAL=0   collectives back-to-back with no idle gap (stresses the gate
#                and the lock/gate quiesce interplay)
#   NCCL_NVLS_ENABLE=0  bisect: P2P IPC imports only, no multicast objects
#
# Usage: sudo bash run_matrix.sh
set -uo pipefail
cd "$(dirname "$0")"

FAILED=0
row() {
  local name="$1"; shift
  local log
  log=$(mktemp /tmp/mcmatrix.XXXXXX.log)
  printf '%-46s ' "$name"
  if env "$@" bash ./run_nccl_shim_gvisor_driven.sh >"$log" 2>&1; then
    local faults
    faults=$(grep -oE 'context faults: [0-9]+' "$log" | tail -1)
    printf 'PASS  (%s)\n' "${faults:-no fault probe}"
  else
    FAILED=1
    printf 'FAIL  -- %s\n' "$(grep -ohE 'FAIL: .*|RESULT: FAIL' "$log" | tail -1)"
    printf '        %s\n' "$(grep -ohE 'Checkpoint attempt failed with error: .*' /tmp/nccl-shim-driven/logs/*boot* 2>/dev/null | tail -1 | cut -c1-160)"
  fi
  rm -f "$log"
  sleep 3
}

echo "=== gVisor-driven multicast interposer: acceptance matrix ==="
row "TP=4 NVLS, paused"                  WORLD=4
row "TP=4 NVLS, running"                 WORLD=4 NO_PAUSE=1
row "TP=4 NVLS, running, no idle gap"    WORLD=4 NO_PAUSE=1 INTERVAL=0
row "TP=4 no NVLS, running, no idle gap" WORLD=4 NO_PAUSE=1 INTERVAL=0 NCCL_NVLS_ENABLE=0
row "TP=8 NVLS, paused"                  WORLD=8
row "TP=8 NVLS, running, no idle gap"    WORLD=8 NO_PAUSE=1 INTERVAL=0
echo
if [[ $FAILED -eq 0 ]]; then
  echo "=== MATRIX: ALL PASS ==="
else
  echo "=== MATRIX: FAILURES PRESENT ==="
fi
exit $FAILED
