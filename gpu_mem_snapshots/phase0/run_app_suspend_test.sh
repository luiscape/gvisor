#!/usr/bin/env bash
# run_app_suspend_test.sh — validates the in-process multicast suspend/resume
# MODEL under gVisor on R610: the app itself suspends its multicast layer
# through libcuda (the same in-process teardown this exploration adds to NCCL),
# then gVisor checkpoints with cuda-checkpoint, and after restore the app
# resumes (recreates multicast at the IDENTICAL VA).
#
# Flow (a)->(b)->(c):
#   (a) workload is idle between iterations       (pause)
#   (b) `runsc exec touch /tmp/suspend`           (app-level suspend)
#       -> app unmaps MC VA (keeps reservation), unbinds, releases MC handle,
#          ALL through libcuda => libcuda bookkeeping stays consistent
#       -> nvproxy's blocker gate now sees ZERO multicast blockers
#   (c) `runsc checkpoint` (job protocol, NO --cuda-multicast-suspend)
#       -> gate verifies clean; cuda-checkpoint has nothing multicast to save
#   restore -> `runsc exec touch /tmp/resume` (app-level resume)
#       -> app recreates MC (new handle OK) + rebinds + re-maps at same VA
#   PASS = post-restore iterations pass with VA inventory identical.
#
# Usage:
#   sudo nvidia-smi -pm 1
#   sudo [RUNSC=...] [CUDA_CHECKPOINT=...] bash run_app_suspend_test.sh
set -uo pipefail
cd "$(dirname "$0")"
PHASE0_DIR=$(pwd)

RUNSC="${RUNSC:-/usr/local/bin/runsc-phase0}"
CUDA_CHECKPOINT="${CUDA_CHECKPOINT:-/usr/local/bin/cuda-checkpoint}"
GPUS="${GPUS:-0,1}"
GPU_A="${GPUS%%,*}"; GPU_B="${GPUS##*,}"
WORK=/tmp/app-suspend-test
STAGE=/opt/phase0
CID=appsusp
CID_R=appsusp-r
log(){ echo "[app-suspend $(date +%H:%M:%S)] $*"; }

[[ -x "$RUNSC" ]] || { log "runsc not found at $RUNSC"; exit 1; }
[[ -x "$CUDA_CHECKPOINT" ]] || { log "cuda-checkpoint not found"; exit 1; }
UVM_MAJOR=$(awk '$2=="nvidia-uvm"{print $1}' /proc/devices)

JOB_FLAG=(--cuda-checkpoint-path "$CUDA_CHECKPOINT")
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
      "--gpus", "0,1", "--dir", "/tmp", "--mode", "full",
      "--map-mc-va", "always"
    ],
    "env": ["PATH=/usr/local/bin:/usr/bin:/bin"],
    "cwd": "/"
  },
  "root": {"path": "/", "readonly": true},
  "hostname": "appsusp",
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

log "(a) starting live-multicast workload under runsc (job-wrapped)"
runsc run -detach -bundle "$WORK/bundle" -pid-file "$WORK/pid" "$CID" \
  || { log "runsc run failed"; tail -30 "$WORK"/logs/*boot* 2>/dev/null; exit 1; }
wait_status "$CID" "mc-live pass" 180 || { tail -40 "$WORK"/logs/*boot*; exit 1; }

log "(b) app-level multicast SUSPEND (in-process teardown through libcuda)"
runsc exec "$CID" /bin/touch /tmp/suspend
wait_status "$CID" "SUSPENDED\|pass" 30
wait_status "$CID" "pass" 30 || { log "FAIL: not healthy after suspend"; exit 1; }

log "(c) checkpoint (blocker gate must pass: zero multicast blockers)"
t0=$SECONDS
runsc checkpoint -image-path "$WORK/img" \
  -cuda-checkpoint-path "$CUDA_CHECKPOINT" -cuda-checkpoint-sequential "$CID"
CKPT_RC=$?
log "checkpoint rc=$CKPT_RC ($((SECONDS-t0))s)"
[[ $CKPT_RC -eq 0 ]] || { log "FAIL: checkpoint refused/failed"; grep -h 'blocker' "$WORK"/logs/*boot* | tail -3; exit 1; }
runsc delete -force "$CID" >/dev/null 2>&1 || true

log "restore"
runsc restore -detach -image-path "$WORK/img" -bundle "$WORK/bundle" \
  -pid-file "$WORK/pid-r" "$CID_R"
REST_RC=$?
log "restore rc=$REST_RC"
sleep 2
TOGGLE_ERR=$(grep -h 'Killing the sandbox after post restore' "$WORK"/logs/*boot* 2>/dev/null | tail -1)
if [[ -n "$TOGGLE_ERR" ]]; then
  log "FAIL: restore toggle failed even after app-level suspend:"
  log "  ${TOGGLE_ERR#*restore work failed: }"
  exit 1
fi
[[ $REST_RC -eq 0 ]] || { log "FAIL: restore failed"; exit 1; }

runsc exec "$CID_R" /bin/touch /tmp/restored 2>/dev/null || true
wait_status "$CID_R" "post-restore" 60 || FAIL=1

log "RESUME (ncclCommResume analog): recreate multicast at the identical VA"
runsc exec "$CID_R" /bin/touch /tmp/resume
wait_status "$CID_R" "RESUMED" 60 || { log "FAIL: resume did not complete"; FAIL=1; }
wait_status "$CID_R" "post-restore+mc-live pass" 60 || { log "FAIL: post-resume verification"; FAIL=1; }
# A few more clean iterations (VA identity is checked every iteration now
# that the workload is fully resumed).
sleep 4
FINAL=$(status_of "$CID_R")
log "final status: $FINAL"
grep -q "mc-live pass" <<<"$FINAL" || FAIL=1

runsc kill "$CID_R" KILL >/dev/null 2>&1 || true

echo ""
log "==== KEY LOG LINES ===="
grep -hE 'SUSPENDED|RESUMED|blocker' "$WORK"/logs/*boot* 2>/dev/null | sed 's/^.*\] //' | awk '!seen[$0]++' | head -8
echo ""
if [[ $FAIL -eq 0 ]]; then
  log "==== RESULT: PASS ===="
  log "The ncclCommSuspend/Resume MODEL round-trips under gVisor+cuda-checkpoint:"
  log "app-level (in-process, libcuda-consistent) suspend -> gate-verified"
  log "checkpoint -> restore -> resume recreates multicast at IDENTICAL VAs."
else
  log "==== RESULT: FAIL ===="
fi
exit $FAIL
