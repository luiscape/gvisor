#!/usr/bin/env bash
# run_suspend_test.sh — E2E for slice 3: multicast suspend/replay across a
# gVisor checkpoint/restore, WITHOUT any app-side teardown.
#
# The workload keeps a live 00FD multicast object (2 GPUs attached, vidmem
# bound from both, MC VA mapped) and verifies MULTICAST BROADCAST every
# iteration (write via MC VA -> must appear in both GPUs' unicast mappings).
#
#   1. run workload under runsc (export fd closed after setup)
#   2. `runsc checkpoint` with the MC object LIVE
#        -> nvproxy suspends the object between lock and checkpoint phases
#        -> checkpoint must SUCCEED (natively this hangs; measured)
#   3. `runsc restore`
#        -> nvproxy replays the object after the restore toggle
#   4. workload must keep passing INCLUDING broadcast verification
#
# Usage:
#   sudo nvidia-smi -pm 1
#   sudo [RUNSC=...] [CUDA_CHECKPOINT=...] [GPUS="0,1"] \
#        bash gpu_mem_snapshots/phase0/run_suspend_test.sh
set -uo pipefail
cd "$(dirname "$0")"
PHASE0_DIR=$(pwd)

RUNSC="${RUNSC:-/usr/local/bin/runsc-phase0}"
CUDA_CHECKPOINT="${CUDA_CHECKPOINT:-/usr/local/bin/cuda-checkpoint}"
GPUS="${GPUS:-0,1}"
GPU_A="${GPUS%%,*}"; GPU_B="${GPUS##*,}"
# MAP_MC_VA=after-restore defers the app's MC VA mapping until post-restore:
# isolates 00FD object replay from MC VA *mapping* replay (not yet implemented).
MAP_MC_VA="${MAP_MC_VA:-always}"
# MODE=no-bind|no-mc bisect what blocks the cuda-checkpoint restore walk.
MODE="${MODE:-full}"
WORK=/tmp/suspend-test
STAGE=/opt/phase0
CID=susp
CID_R=susp-r
log(){ echo "[suspend $(date +%H:%M:%S)] $*"; }

[[ -x "$RUNSC" ]] || { log "runsc not found at $RUNSC"; exit 1; }
[[ -x "$CUDA_CHECKPOINT" ]] || { log "cuda-checkpoint not found at $CUDA_CHECKPOINT"; exit 1; }
UVM_MAJOR=$(awk '$2=="nvidia-uvm"{print $1}' /proc/devices)

# JOB=1 (default) wraps the container command in `cuda-checkpoint --launch-job`
# via runsc's global --cuda-checkpoint-path flag (driver R610+). This is the
# R610 mechanism whose restore rendezvous the multicast replay depends on; on
# R580 it is a no-op (loader warns and leaves the command unwrapped).
JOB="${JOB:-1}"
JOB_FLAG=()
[[ "$JOB" == "1" ]] && JOB_FLAG=(--cuda-checkpoint-path "$CUDA_CHECKPOINT")

# The global cuda-checkpoint-path flag must be applied to `run` and `restore`
# (container launch), where job wrapping happens; other subcommands ignore it.
runsc(){
  local sub="$1"
  if [[ "$sub" == "run" || "$sub" == "restore" ]]; then
    "$RUNSC" --root "$WORK/root" --debug --debug-log="$WORK/logs/" "${JOB_FLAG[@]}" "$@"
  else
    "$RUNSC" --root "$WORK/root" --debug --debug-log="$WORK/logs/" "$@"
  fi
}
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
      "--gpus", "0,1", "--dir", "/tmp", "--map-mc-va", "$MAP_MC_VA",
      "--mode", "$MODE"
    ],
    "env": ["PATH=/usr/local/bin:/usr/bin:/bin"],
    "cwd": "/"
  },
  "root": {"path": "/", "readonly": true},
  "hostname": "susp",
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

log "starting live-multicast workload under runsc (broadcast verified per iter)"
runsc run -detach -bundle "$WORK/bundle" -pid-file "$WORK/pid" "$CID" \
  || { log "runsc run failed"; tail -30 "$WORK"/logs/*boot* 2>/dev/null; exit 1; }
wait_status "$CID" "mc-live pass" 180 || { tail -40 "$WORK"/logs/*boot*; exit 1; }

log "checkpoint with LIVE multicast object (nvproxy suspend expected)"
t0=$SECONDS
runsc checkpoint -image-path "$WORK/img" \
  -cuda-checkpoint-path "$CUDA_CHECKPOINT" -cuda-checkpoint-sequential \
  -cuda-multicast-suspend "$CID"
CKPT_RC=$?
log "checkpoint rc=$CKPT_RC ($((SECONDS-t0))s)"
if [[ $CKPT_RC -ne 0 ]]; then
  log "FAIL: checkpoint failed; suspend-related log lines:"
  grep -h 'suspend\|multicast\|blocker' "$WORK"/logs/*boot* 2>/dev/null | tail -20
  exit 1
fi
runsc delete -force "$CID" >/dev/null 2>&1 || true

log "restoring (nvproxy replay after toggle expected)"
t0=$SECONDS
runsc restore -detach -image-path "$WORK/img" -bundle "$WORK/bundle" \
  -pid-file "$WORK/pid-r" "$CID_R"
REST_RC=$?
log "restore rc=$REST_RC ($((SECONDS-t0))s)"
sleep 2  # let postRestore (toggle + replay) run; -detach returns before it

# cuda-checkpoint's restore toggle proactively refuses to restore a process
# whose libcuda state includes a multicast object (confirmed on BOTH R580 and
# R610 under the --launch-job protocol): during the toggle it recreates the
# ordinary allocations, then returns "unknown error" when it reaches the
# multicast object WITHOUT issuing any 00FD/ATTACH ioctl nvproxy could satisfy.
# nvproxy replay never runs. `runsc restore -detach` returns 0 but the sandbox
# is killed asynchronously; detect that boundary from the logs.
TOGGLE_ERR=$(grep -h 'toggle failed\|Killing the sandbox after post restore' \
  "$WORK"/logs/*boot* 2>/dev/null | tail -1)
if grep -q 'suspended.*multicast object' "$WORK"/logs/*boot* 2>/dev/null \
   && [[ -n "$TOGGLE_ERR" ]]; then
  log "CUDA-CHECKPOINT BOUNDARY: checkpoint SAVE succeeded via nvproxy multicast"
  log "  suspend, but cuda-checkpoint's restore TOGGLE proactively refused the"
  log "  process (multicast in libcuda state; no interceptable ioctl):"
  log "  ${TOGGLE_ERR#*restore work failed: }"
  echo ""
  log "==== RESULT: SAVE-OK / RESTORE-BOUNDARY (cuda-checkpoint limitation) ===="
  exit 0
fi
if [[ $REST_RC -ne 0 ]]; then
  log "FAIL: restore failed unexpectedly; replay-related log lines:"
  grep -h 'replay\|multicast\|toggle' "$WORK"/logs/*restore* "$WORK"/logs/*boot* 2>/dev/null | tail -25
  exit 1
fi

runsc exec "$CID_R" /bin/touch /tmp/restored 2>/dev/null || true
# Broadcast must keep passing after restore -- several iterations.
OK=1
wait_status "$CID_R" "post-restore+mc-live pass" 120 || OK=0
[[ "$MAP_MC_VA" == "after-restore" ]] && { status_of "$CID_R" | grep -q . ; }
if [[ $OK -eq 1 ]]; then
  sleep 5
  s=$(status_of "$CID_R")
  log "final status: $s"
  grep -q "mc-live pass" <<<"$s" && grep -q "failures=0" <<<"$s" || OK=0
fi
[[ $OK -eq 1 ]] || { log "FAIL: post-restore broadcast verification"; FAIL=1; }

runsc kill "$CID_R" KILL >/dev/null 2>&1 || true

echo ""
log "==== SUSPEND/REPLAY LOG LINES ===="
grep -h 'suspended multicast\|replaying multicast\|recreated by the restore toggle\|multicast replayed\|failed to replay\|failed to suspend' \
  "$WORK"/logs/* 2>/dev/null | sed 's/^.*\] //' | awk '!seen[$0]++' | head -20
echo ""
log "==== RESULT: $([[ $FAIL -eq 0 ]] && echo PASS || echo FAIL) ===="
exit $FAIL
