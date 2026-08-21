#!/bin/bash
# Copyright 2026 The gVisor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Benchmark harness for access-order-traced ("Spice-style") checkpoint
# restore. See README.md in this directory.
#
# Must be run as root (drops page caches between trials). Example:
#   sudo tools/mem-restore-bench/run.sh
#
# Environment variables:
#   RUNSC     - path to runsc binary (default: <repo>/bin/runsc)
#   WORK      - scratch directory (default: /data/mem-restore-bench if /data
#               exists, else /tmp/mem-restore-bench)
#   TOTAL_MB  - total memory allocated & checkpointed (default: 4096)
#   HOT_MB    - working set touched per pass (default: 1024)
#   TRIALS    - measured restores per variant (default: 5)
#   PLATFORM  - runsc platform (default: systrap)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUNSC="${RUNSC:-${REPO_DIR}/bin/runsc}"
if [[ -d /data ]]; then
  WORK="${WORK:-/data/mem-restore-bench}"
else
  WORK="${WORK:-/tmp/mem-restore-bench}"
fi
TOTAL_MB="${TOTAL_MB:-4096}"
HOT_MB="${HOT_MB:-1024}"
REGION_KB="${REGION_KB:-256}"
TRIALS="${TRIALS:-5}"
PLATFORM="${PLATFORM:-systrap}"

if [[ "$(id -u)" != 0 ]]; then
  echo "must run as root (needed to run runsc and drop page caches)" >&2
  exit 1
fi
if [[ ! -x "${RUNSC}" ]]; then
  echo "runsc not found at ${RUNSC}; build it with:" >&2
  echo "  make copy TARGETS=runsc DESTINATION=bin/" >&2
  exit 1
fi

BUNDLE="${WORK}/bundle"
OUT_DIR="${WORK}/out"
OUT_LOG="${OUT_DIR}/bench.log"
LOG_DIR="${WORK}/logs"
RESULTS="${WORK}/results.csv"

# Clean up leftovers from previous (possibly interrupted) runs.
if [[ -d "${WORK}/runsc-root" ]]; then
  for c in $("${RUNSC}" --root="${WORK}/runsc-root" list 2>/dev/null | awk 'NR>1 {print $1}'); do
    "${RUNSC}" --root="${WORK}/runsc-root" kill "${c}" KILL >/dev/null 2>&1 || true
    "${RUNSC}" --root="${WORK}/runsc-root" delete -force "${c}" >/dev/null 2>&1 || true
  done
  umount "${WORK}/runsc-root/null-netns" >/dev/null 2>&1 || true
fi
rm -rf "${WORK}"
mkdir -p "${BUNDLE}/rootfs" "${OUT_DIR}" "${LOG_DIR}" \
  "${WORK}/img-orig" "${WORK}/img-base" "${WORK}/img-reordered" "${WORK}/runsc-root"

runsc() {
  "${RUNSC}" \
    --root="${WORK}/runsc-root" \
    --platform="${PLATFORM}" \
    --network=none \
    --file-access-mounts=shared \
    --host-settings=ignore \
    --debug --debug-log="${LOG_DIR}/" \
    "$@"
}

GO="${GO:-$(command -v go || true)}"
if [[ -z "${GO}" && -x /usr/local/go/bin/go ]]; then
  GO=/usr/local/go/bin/go
fi
if [[ -z "${GO}" ]]; then
  echo "go toolchain not found; set GO=/path/to/go" >&2
  exit 1
fi

echo "=== Building workload"
(cd "${SCRIPT_DIR}/workload" && CGO_ENABLED=0 GOCACHE="${WORK}/gocache" "${GO}" build -o "${BUNDLE}/rootfs/workload" .)

echo "=== Creating OCI bundle"
(cd "${BUNDLE}" && "${RUNSC}" spec -- /workload)
python3 - "${BUNDLE}/config.json" "${OUT_DIR}" "${TOTAL_MB}" "${HOT_MB}" "${REGION_KB}" <<'EOF'
import json, sys
path, out_dir, total_mb, hot_mb, region_kb = sys.argv[1:6]
with open(path) as f:
    spec = json.load(f)
spec["process"]["terminal"] = False
spec["process"]["env"] = [
    "PATH=/",
    f"TOTAL_MB={total_mb}",
    f"HOT_MB={hot_mb}",
    f"REGION_KB={region_kb}",
    "SLEEP_MS=1000",
    "OUT=/out/bench.log",
    "TRIGGER=/out/trigger",
]
spec.setdefault("mounts", []).append({
    "destination": "/out",
    "type": "bind",
    "source": out_dir,
    "options": ["rbind", "rw"],
})
with open(path, "w") as f:
    json.dump(spec, f, indent=2)
EOF

# wait_for_line <byte-offset> <regex> [timeout-sec]: waits until a line
# matching regex appears in ${OUT_LOG} past byte-offset, then prints it.
wait_for_line() {
  local off="$1" regex="$2" timeout="${3:-180}"
  local deadline=$((SECONDS + timeout))
  while true; do
    if [[ -f "${OUT_LOG}" ]]; then
      local line
      line="$(tail -c "+$((off + 1))" "${OUT_LOG}" | grep -m 1 -E "${regex}" || true)"
      if [[ -n "${line}" ]]; then
        echo "${line}"
        return 0
      fi
    fi
    if (( SECONDS >= deadline )); then
      echo "timed out waiting for ${regex}" >&2
      return 1
    fi
    sleep 0.05
  done
}

out_size() {
  stat -c %s "${OUT_LOG}" 2>/dev/null || echo 0
}

cleanup_container() {
  runsc kill "$1" KILL >/dev/null 2>&1 || true
  sleep 0.2
  runsc delete -force "$1" >/dev/null 2>&1 || true
}

cleanup_all_containers() {
  local ids
  ids="$(runsc list 2>/dev/null | awk 'NR>1 {print $1}')" || true
  for id in ${ids}; do
    cleanup_container "${id}"
  done
}
trap cleanup_all_containers EXIT

now_ms() {
  date +%s%3N
}

echo "=== Creating initial checkpoint (img-orig)"
cleanup_container c-init
off=$(out_size)
runsc run -detach --bundle="${BUNDLE}" c-init
wait_for_line "${off}" "PASS 1 " >/dev/null
sleep 0.3  # let the pass finish emitting and enter its sleep loop
runsc checkpoint --image-path="${WORK}/img-orig" --compression=none c-init
cleanup_container c-init

# write_trigger <token>: asks the workload to run a pass as soon as it
# observes the token (including immediately after being restored).
write_trigger() {
  echo "$1" > "${OUT_DIR}/trigger.tmp"
  mv "${OUT_DIR}/trigger.tmp" "${OUT_DIR}/trigger"
}

# make_image <src-image> <dst-image> <container> [extra restore flags...]:
# restores src-image, waits for one post-restore pass (so the restored
# sandbox demonstrably ran), and checkpoints it to dst-image.
make_image() {
  local src="$1" dst="$2" name="$3"
  shift 3
  cleanup_container "${name}"
  local off
  off=$(out_size)
  write_trigger "mkimage-${name}-$(now_ms)"
  runsc restore -detach --direct --bundle="${BUNDLE}" --image-path="${src}" "$@" "${name}"
  wait_for_line "${off}" "^PASS " >/dev/null
  sleep 0.5
  runsc checkpoint --image-path="${dst}" --compression=none "${name}"
  cleanup_container "${name}"
}

echo "=== Creating baseline image (restore without tracing, checkpoint again)"
make_image "${WORK}/img-orig" "${WORK}/img-base" c-mkbase

echo "=== Creating reordered image (restore with --pages-trace, checkpoint again)"
make_image "${WORK}/img-orig" "${WORK}/img-reordered" c-mktrace --pages-trace

echo
echo "=== Image sizes"
du -sh "${WORK}"/img-*/ | sed 's/^/    /'
echo

echo "variant,trial,restore_cmd_ms,time_to_pass_ms,pass_dur_ms" > "${RESULTS}"

run_trials() {
  local variant="$1" image="$2"
  echo "=== Measuring: ${variant} (${TRIALS} trials)"
  for t in $(seq 1 "${TRIALS}"); do
    local name="c-${variant}-${t}"
    cleanup_container "${name}"
    sync
    echo 3 > /proc/sys/vm/drop_caches
    sleep 0.5
    local off t0 t1 pass_line t2
    off=$(out_size)
    write_trigger "${variant}-${t}-$(now_ms)"
    t0=$(now_ms)
    runsc restore -detach --direct --background --bundle="${BUNDLE}" --image-path="${image}" "${name}"
    t1=$(now_ms)
    pass_line="$(wait_for_line "${off}" "^PASS ")"
    t2=$(now_ms)
    local pass_dur
    pass_dur="$(sed -E 's/.*dur_ms=([0-9]+).*/\1/' <<< "${pass_line}")"
    echo "    trial ${t}: restore_cmd=$((t1 - t0))ms time_to_pass=$((t2 - t0))ms pass_dur=${pass_dur}ms"
    echo "${variant},${t},$((t1 - t0)),$((t2 - t0)),${pass_dur}" >> "${RESULTS}"
    # Let background loading finish before killing, so that the async page
    # loader stats are logged.
    sleep "${POST_PASS_SLEEP:-6}"
    cleanup_container "${name}"
  done
}

run_trials baseline "${WORK}/img-base"
run_trials reordered "${WORK}/img-reordered"

echo
echo "=== Results (${RESULTS})"
python3 - "${RESULTS}" <<'EOF'
import csv, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
cols = ["restore_cmd_ms", "time_to_pass_ms", "pass_dur_ms"]
stats = defaultdict(lambda: defaultdict(list))
for r in rows:
    for c in cols:
        stats[r["variant"]][c].append(int(r[c]))
print(f"{'metric':<22}", end="")
variants = list(stats.keys())
for v in variants:
    print(f"{v + ' (mean/min)':>26}", end="")
print(f"{'speedup (mean)':>18}")
for c in cols:
    print(f"{c:<22}", end="")
    means = {}
    for v in variants:
        vals = stats[v][c]
        means[v] = sum(vals) / len(vals)
        print(f"{means[v]:>17.1f}/{min(vals):<8}", end="")
    if len(variants) == 2 and means[variants[1]] > 0:
        print(f"{means[variants[0]] / means[variants[1]]:>15.2f}x", end="")
    print()
EOF

echo
echo "=== Async page loader stats from sandbox logs (last per boot)"
grep -h "Async page loading completed" "${LOG_DIR}"/*boot* 2>/dev/null | tail -20 | sed 's/^/    /' || true
echo
echo "Raw results: ${RESULTS}"
echo "Workload log: ${OUT_LOG}"
echo "Sandbox logs: ${LOG_DIR}"
