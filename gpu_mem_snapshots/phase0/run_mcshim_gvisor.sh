#!/usr/bin/env bash
# run_mcshim_gvisor.sh -- gVisor e2e for the mcshim multicast interposer
# (Idea D, single process).
#
# Same model as run_app_suspend_test.sh, but the multicast suspend/resume is
# performed TRANSPARENTLY by the LD_PRELOAD shim (mcshim.so) instead of by the
# application: the workload (mcshim_workload.py) contains zero suspend logic.
# This is the injection model gVisor itself would use (env injection next to
# the cuda-checkpoint --launch-job wrapping in runsc/boot/loader.go).
#
# Flow:
#   (a) `runsc exec touch pause`      -> workload quiesces (idle engine)
#   (b) `runsc exec touch suspend`    -> SHIM unmaps MC VA (keeps reservation),
#                                        unbinds, releases the 00FD handle
#   (c) `runsc checkpoint`            -> zero multicast blockers; job protocol
#   restore -> `touch resume`         -> SHIM recreates MC at IDENTICAL VA
#           -> `touch unpause`        -> workload verifies post-restore
#
# Usage:
#   sudo nvidia-smi -pm 1
#   sudo [RUNSC=...] [CUDA_CHECKPOINT=...] [GPUS=0,1] bash run_mcshim_gvisor.sh
set -uo pipefail
cd "$(dirname "$0")"
PHASE0_DIR=$(pwd)

RUNSC="${RUNSC:-/usr/local/bin/runsc-phase0}"
CUDA_CHECKPOINT="${CUDA_CHECKPOINT:-/usr/local/bin/cuda-checkpoint}"
GPUS="${GPUS:-0,1}"
GPU_A="${GPUS%%,*}"; GPU_B="${GPUS##*,}"
WORK=/tmp/mcshim-gvisor-test
STAGE=/opt/phase0
MCDIR=/tmp/mcshim   # inside the sandbox (container /tmp tmpfs)
CID=mcshim
CID_R=mcshim-r
log(){ echo "[mcshim-gvisor $(date +%H:%M:%S)] $*"; }

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
cp "$PHASE0_DIR/mcshim_workload.py" "$PHASE0_DIR/_cuda.py" \
   "$PHASE0_DIR/mcshim/mcshim.so" "$STAGE/"
for m in nvidia nvidia_uvm; do
  mkdir -p "$STAGE/sys_module_$m"; echo live > "$STAGE/sys_module_$m/initstate"
done
chmod -R a+rX "$STAGE"

# The shim is injected purely via env (LD_PRELOAD + MCSHIM_*): the workload
# argv has no shim awareness at all.
cat > "$WORK/bundle/config.json" <<EOF
{
  "ociVersion": "1.1.0",
  "process": {
    "terminal": false,
    "user": {"uid": 0, "gid": 0},
    "args": [
      "/usr/bin/python3", "$STAGE/mcshim_workload.py",
      "--gpus", "0,1", "--dir", "$MCDIR"
    ],
    "env": [
      "PATH=/usr/local/bin:/usr/bin:/bin",
      "LD_PRELOAD=$STAGE/mcshim.so",
      "MCSHIM_DIR=$MCDIR",
      "MCSHIM_LOG=$MCDIR/mcshim.log"
    ],
    "cwd": "/"
  },
  "root": {"path": "/", "readonly": true},
  "hostname": "mcshim",
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

status_of(){ runsc exec "$1" /bin/cat "$MCDIR/wl.status" 2>/dev/null; }
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
wait_acks(){
  local cid="$1" prefix="$2" count="$3" timeout="$4" i n
  for ((i=0; i<timeout; i++)); do
    n=$(runsc exec "$cid" /bin/sh -c "ls $MCDIR/$prefix.* 2>/dev/null | wc -l")
    [[ "${n:-0}" -ge "$count" ]] && return 0
    sleep 1
  done
  log "TIMEOUT waiting for $count $prefix.* acks (got ${n:-0})"
  runsc exec "$cid" /bin/sh -c "cat $MCDIR/error.* 2>/dev/null" || true
  return 1
}
# `runsc exec touch` must not inherit shim env: markers are plain files, and
# exec'd helpers don't get the container process env anyway (runsc exec uses
# its own env), so no MCSHIM vars leak. The lazy (cuInit-gated) control
# thread guards the cuda-checkpoint invocations the sentry runs with spec env.
touch_marker(){ runsc exec "$1" /bin/touch "$MCDIR/$2"; }
rm_marker(){ runsc exec "$1" /bin/rm -f "$MCDIR/$2"; }

FAIL=0

log "starting live-multicast workload under runsc (shim injected via env)"
runsc run -detach -bundle "$WORK/bundle" -pid-file "$WORK/pid" "$CID" \
  || { log "runsc run failed"; tail -30 "$WORK"/logs/*boot* 2>/dev/null; exit 1; }
wait_status "$CID" "mc-live pass" 180 || { tail -40 "$WORK"/logs/*boot*; exit 1; }

log "(a) pause workload (quiesce)"
touch_marker "$CID" pause
wait_status "$CID" "PAUSED" 30 || { log "FAIL: workload did not pause"; exit 1; }

log "(b) SHIM multicast suspend (transparent, in-process through libcuda)"
touch_marker "$CID" suspend
wait_acks "$CID" suspended 1 30 || { log "FAIL: shim did not suspend"; \
  runsc exec "$CID" /bin/cat "$MCDIR/mcshim.log" 2>/dev/null | tail -20; exit 1; }
log "shim SUSPENDED"

log "(c) checkpoint (blocker gate must see zero multicast blockers)"
t0=$SECONDS
runsc checkpoint -image-path "$WORK/img" \
  -cuda-checkpoint-path "$CUDA_CHECKPOINT" -cuda-checkpoint-sequential "$CID"
CKPT_RC=$?
log "checkpoint rc=$CKPT_RC ($((SECONDS-t0))s)"
[[ $CKPT_RC -eq 0 ]] || { log "FAIL: checkpoint refused/failed"; \
  grep -h 'blocker' "$WORK"/logs/*boot* 2>/dev/null | tail -3; exit 1; }
runsc delete -force "$CID" >/dev/null 2>&1 || true

log "restore"
runsc restore -detach -image-path "$WORK/img" -bundle "$WORK/bundle" \
  -pid-file "$WORK/pid-r" "$CID_R"
REST_RC=$?
log "restore rc=$REST_RC"
sleep 2
TOGGLE_ERR=$(grep -h 'Killing the sandbox after post restore' "$WORK"/logs/*boot* 2>/dev/null | tail -1)
if [[ -n "$TOGGLE_ERR" ]]; then
  log "FAIL: restore toggle failed even after shim suspend:"
  log "  ${TOGGLE_ERR#*restore work failed: }"
  exit 1
fi
[[ $REST_RC -eq 0 ]] || { log "FAIL: restore failed"; exit 1; }

log "SHIM multicast resume (recreate at IDENTICAL VAs): remove marker"
rm_marker "$CID_R" suspend
wait_acks "$CID_R" resumed 1 60 || { log "FAIL: shim did not resume"; \
  runsc exec "$CID_R" /bin/cat "$MCDIR/mcshim.log" 2>/dev/null | tail -20; FAIL=1; }
log "shim RESUMED"

log "unpause + post-restore verification"
rm_marker "$CID_R" pause
wait_status "$CID_R" "post-restore+mc-live pass" 60 || FAIL=1
sleep 4
FINAL=$(status_of "$CID_R")
log "final status: $FINAL"
grep -q "post-restore+mc-live pass" <<<"$FINAL" || FAIL=1
grep -q "failures=0" <<<"$FINAL" || FAIL=1

echo ""
log "==== SHIM LOG (suspend/resume) ===="
runsc exec "$CID_R" /bin/cat "$MCDIR/mcshim.log" 2>/dev/null \
  | grep -E 'SUSPEND|RESUME|track' | tail -12

runsc kill "$CID_R" KILL >/dev/null 2>&1 || true

echo ""
if [[ $FAIL -eq 0 ]]; then
  log "==== RESULT: PASS ===="
  log "The generic libcuda interposer (no NCCL fork, no app/engine hooks)"
  log "round-trips a live-multicast process through gVisor checkpoint/restore"
  log "with the multicast VA re-mapped at the IDENTICAL address."
else
  log "==== RESULT: FAIL ===="
fi
exit $FAIL
