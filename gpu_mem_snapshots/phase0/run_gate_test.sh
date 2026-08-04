#!/usr/bin/env bash
# run_gate_test.sh — E2E acceptance for slices 1+2 (blocker inventory + gate,
# multicast control recording) on a 2-GPU host, no docker needed.
#
# Sequence:
#   1. Run multicast_workload.py (real 00FD object: 2x ATTACH_GPU, 2x
#      ATTACH_MEM, exported fd, mapped MC VA) under runsc.
#   2. `runsc checkpoint` while the multicast object is LIVE
#        -> must FAIL FAST with a per-client blocker message naming the
#           multicast object and the exported fd (slice 1 acceptance).
#   3. Tell the workload to tear the multicast layer down (app-side release).
#   4. `runsc checkpoint` again -> must SUCCEED (blockers gone).
#   5. `runsc restore` -> workload must keep passing (unicast patterns intact).
#   6. Print the nvproxy census (with handle values) for slice 3 design data.
#
# Usage:
#   sudo nvidia-smi -pm 1   # persistence mode required (see phase0/README.md)
#   sudo [RUNSC=...] [CUDA_CHECKPOINT=...] [GPUS="0,1"] \
#        bash gpu_mem_snapshots/phase0/run_gate_test.sh
set -uo pipefail
cd "$(dirname "$0")"
PHASE0_DIR=$(pwd)

RUNSC="${RUNSC:-/usr/local/bin/runsc-phase0}"
CUDA_CHECKPOINT="${CUDA_CHECKPOINT:-/usr/local/bin/cuda-checkpoint}"
GPUS="${GPUS:-0,1}"
GPU_A="${GPUS%%,*}"; GPU_B="${GPUS##*,}"
WORK=/tmp/gate-test
STAGE=/opt/phase0
CID=gate
CID_R=gate-r
log(){ echo "[gate $(date +%H:%M:%S)] $*"; }

[[ -x "$RUNSC" ]] || { log "runsc not found at $RUNSC"; exit 1; }
[[ -x "$CUDA_CHECKPOINT" ]] || { log "cuda-checkpoint not found at $CUDA_CHECKPOINT"; exit 1; }
UVM_MAJOR=$(awk '$2=="nvidia-uvm"{print $1}' /proc/devices)

runsc(){ "$RUNSC" --root "$WORK/root" --debug --debug-log="$WORK/logs/" "$@"; }
cleanup(){
  runsc delete -force "$CID"   >/dev/null 2>&1 || true
  runsc delete -force "$CID_R" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cleanup
rm -rf "$WORK" 2>/dev/null
mkdir -p "$WORK"/{root,logs,img,bundle}

mkdir -p "$STAGE"
cp "$PHASE0_DIR/multicast_workload.py" "$PHASE0_DIR/_cuda.py" "$STAGE/"
for m in nvidia nvidia_uvm; do
  mkdir -p "$STAGE/sys_module_$m"; echo live > "$STAGE/sys_module_$m/initstate"
done
chmod -R a+rX "$STAGE"

cat > "$WORK/bundle/config.json" <<EOF
{
  "ociVersion": "1.1.0",
  "process": {
    "terminal": false,
    "user": {"uid": 0, "gid": 0},
    "args": [
      "/usr/bin/python3", "$STAGE/multicast_workload.py",
      "--gpus", "0,1", "--dir", "/tmp"
    ],
    "env": ["PATH=/usr/local/bin:/usr/bin:/bin"],
    "cwd": "/"
  },
  "root": {"path": "/", "readonly": true},
  "hostname": "gate",
  "mounts": [
    {"destination": "/proc", "type": "proc"},
    {"destination": "/tmp", "type": "tmpfs"},
    {"destination": "/sys/module/nvidia", "type": "bind",
     "source": "$STAGE/sys_module_nvidia", "options": ["rbind", "ro"]},
    {"destination": "/sys/module/nvidia_uvm", "type": "bind",
     "source": "$STAGE/sys_module_nvidia_uvm", "options": ["rbind", "ro"]}
  ],
  "linux": {
    "namespaces": [
      {"type": "pid"}, {"type": "ipc"}, {"type": "uts"}, {"type": "mount"}
    ],
    "devices": [
      {"path": "/dev/nvidiactl", "type": "c", "major": 195, "minor": 255, "fileMode": 438},
      {"path": "/dev/nvidia$GPU_A", "type": "c", "major": 195, "minor": $GPU_A, "fileMode": 438},
      {"path": "/dev/nvidia$GPU_B", "type": "c", "major": 195, "minor": $GPU_B, "fileMode": 438},
      {"path": "/dev/nvidia-uvm", "type": "c", "major": $UVM_MAJOR, "minor": 0, "fileMode": 438}
    ]
  }
}
EOF

status_of(){ runsc exec "$1" /bin/cat /tmp/status 2>/dev/null; }
wait_status(){
  local cid="$1" pattern="$2" timeout="$3" i s
  for ((i=0; i<timeout; i++)); do
    s=$(status_of "$cid")
    grep -q "$pattern" <<<"$s" && { log "status: $s"; return 0; }
    sleep 1
  done
  log "TIMEOUT waiting for $pattern; last: $(status_of "$cid")"
  return 1
}

FAIL=0

# --- 1. run -------------------------------------------------------------------
log "starting 2-GPU multicast workload under runsc"
runsc run -detach -bundle "$WORK/bundle" -pid-file "$WORK/pid" "$CID" \
  || { log "runsc run failed"; tail -30 "$WORK"/logs/*boot* 2>/dev/null; exit 1; }
wait_status "$CID" "READY\|pass" 180 || { tail -40 "$WORK"/logs/*boot*; exit 1; }
wait_status "$CID" "pass" 30 || exit 1

# --- 2. checkpoint with live multicast: MUST FAIL with blocker message ---------
log "checkpoint attempt #1 (multicast LIVE) -- expecting gate failure"
t0=$SECONDS
GATE_OUT=$(runsc checkpoint -image-path "$WORK/img" \
  -cuda-checkpoint-path "$CUDA_CHECKPOINT" -cuda-checkpoint-sequential \
  -cuda-blocker-timeout 5s "$CID" 2>&1)
GATE_RC=$?
log "checkpoint #1 rc=$GATE_RC ($((SECONDS-t0))s); output:"
echo "$GATE_OUT" | sed 's/^/    /' | head -6
if [[ $GATE_RC -eq 0 ]]; then
  log "FAIL: checkpoint #1 unexpectedly SUCCEEDED with live multicast"; FAIL=1
elif grep -q 'multicast' <<<"$GATE_OUT" && grep -q 'client 0x' <<<"$GATE_OUT"; then
  log "PASS: gate refused with per-client blocker attribution"
else
  log "FAIL: checkpoint #1 failed but without the expected blocker message"; FAIL=1
fi
wait_status "$CID" "pass" 30 || { log "FAIL: workload not healthy after refused checkpoint"; FAIL=1; }

# --- 3. app-side teardown -------------------------------------------------------
log "requesting app-side multicast teardown"
runsc exec "$CID" /bin/touch /tmp/teardown
wait_status "$CID" "torndown pass" 60 || { log "FAIL: teardown did not complete"; FAIL=1; }

# --- 4. checkpoint again: MUST SUCCEED ------------------------------------------
log "checkpoint attempt #2 (multicast torn down) -- expecting success"
rm -rf "$WORK/img"; mkdir -p "$WORK/img"  # clear partial image from the refused attempt
t0=$SECONDS
runsc checkpoint -image-path "$WORK/img" \
  -cuda-checkpoint-path "$CUDA_CHECKPOINT" -cuda-checkpoint-sequential "$CID"
CKPT_RC=$?
log "checkpoint #2 rc=$CKPT_RC ($((SECONDS-t0))s)"
[[ $CKPT_RC -eq 0 ]] || { log "FAIL: checkpoint #2 failed"; FAIL=1; }
runsc delete -force "$CID" >/dev/null 2>&1 || true

# --- 5. restore ------------------------------------------------------------------
if [[ $CKPT_RC -eq 0 ]]; then
  log "restoring"
  runsc restore -detach -image-path "$WORK/img" -bundle "$WORK/bundle" \
    -pid-file "$WORK/pid-r" "$CID_R"
  REST_RC=$?
  log "restore rc=$REST_RC"
  if [[ $REST_RC -eq 0 ]]; then
    runsc exec "$CID_R" /bin/touch /tmp/restored 2>/dev/null || true
    wait_status "$CID_R" "post-restore+torndown pass" 120 \
      || { log "FAIL: post-restore verification"; FAIL=1; }
  else
    FAIL=1
  fi
  runsc kill "$CID_R" KILL >/dev/null 2>&1 || true
fi

# --- 6. extract logs ---------------------------------------------------------------
echo ""
log "==== NVPROXY CENSUS + BLOCKER LOG LINES ===="
grep -h 'nvproxy object census\|nvproxy census diff\|checkpoint blockers\|CUDA checkpoint blockers' \
  "$WORK"/logs/* 2>/dev/null | sed 's/^.*\] //' | awk '!seen[$0]++'
echo ""
log "==== RESULT: $([[ $FAIL -eq 0 ]] && echo PASS || echo FAIL) ===="
exit $FAIL
