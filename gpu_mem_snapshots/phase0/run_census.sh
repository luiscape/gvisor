#!/usr/bin/env bash
# run_census.sh — Phase 0 measurement #3: nvproxy object-graph census across a
# gVisor GPU checkpoint/restore.
#
# Runs census_workload.py (single-GPU, fabric-free, checkpointable) under a
# runsc built from this branch, drives `runsc checkpoint` (with
# --cuda-checkpoint-path) and `runsc restore`, then extracts the
# "nvproxy object census" lines that state_cuda.go logs around the
# cuda-checkpoint phases.
#
# No docker needed: a bare OCI bundle with the host rootfs (read-only) and
# GKE-style GPU injection (nvidia devices listed in spec.Linux.Devices).
# The workload is staged in /opt/phase0 (world-readable, visible through the
# root gofer); its status file lives in the sandbox's private /tmp and is
# polled via `runsc exec`.
#
# Usage:
#   sudo [RUNSC=/usr/local/bin/runsc-phase0] [GPU=0] \
#        [CUDA_CHECKPOINT=/usr/local/bin/cuda-checkpoint] \
#        bash gpu_mem_snapshots/phase0/run_census.sh
set -uo pipefail
cd "$(dirname "$0")"
PHASE0_DIR=$(pwd)

RUNSC="${RUNSC:-/usr/local/bin/runsc-phase0}"
CUDA_CHECKPOINT="${CUDA_CHECKPOINT:-/usr/local/bin/cuda-checkpoint}"
GPU="${GPU:-0}"
WORK=/tmp/census
STAGE=/opt/phase0
CID=census
CID_R=census-r
log(){ echo "[census $(date +%H:%M:%S)] $*"; }

[[ -x "$RUNSC" ]] || { log "runsc not found at $RUNSC"; exit 1; }
[[ -x "$CUDA_CHECKPOINT" ]] || { log "cuda-checkpoint not found at $CUDA_CHECKPOINT"; exit 1; }
UVM_MAJOR=$(awk '$2=="nvidia-uvm"{print $1}' /proc/devices)
[[ -n "$UVM_MAJOR" ]] || { log "nvidia-uvm not in /proc/devices"; exit 1; }

runsc(){ "$RUNSC" --root "$WORK/root" --debug --debug-log="$WORK/logs/" "$@"; }

cleanup(){
  runsc delete -force "$CID"   >/dev/null 2>&1 || true
  runsc delete -force "$CID_R" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cleanup
rm -rf "$WORK" 2>/dev/null
mkdir -p "$WORK"/{root,logs,img,bundle}

# Stage the workload where the (deprivileged) gofer can read it via the host
# rootfs: /opt/phase0, world-readable.
mkdir -p "$STAGE"
cp "$PHASE0_DIR/census_workload.py" "$PHASE0_DIR/_cuda.py" "$STAGE/"
chmod a+rX "$STAGE" "$STAGE"/*.py

# libcuda on a host rootfs (nvidia-modprobe present + setuid) verifies
# /sys/module/{nvidia,nvidia_uvm}/initstate before opening the devices;
# gVisor's sysfs doesn't synthesize those, so cuInit fails NO_DEVICE without
# these shims. (Docker GPU images lack nvidia-modprobe, skipping this check,
# which is why docker+runsc GPU containers don't need them.)
for m in nvidia nvidia_uvm; do
  mkdir -p "$STAGE/sys_module_$m"
  echo live > "$STAGE/sys_module_$m/initstate"
done
chmod -R a+rX "$STAGE"

# --- OCI bundle: host rootfs (ro), GKE-style nvidia devices ------------------
cat > "$WORK/bundle/config.json" <<EOF
{
  "ociVersion": "1.1.0",
  "process": {
    "terminal": false,
    "user": {"uid": 0, "gid": 0},
    "args": [
      "/usr/bin/python3", "$STAGE/census_workload.py",
      "--gpu", "0", "--status-file", "/tmp/status"
    ],
    "env": [
      "PATH=/usr/local/bin:/usr/bin:/bin",
      "CUDA_VISIBLE_DEVICES=0"
    ],
    "cwd": "/"
  },
  "root": {"path": "/", "readonly": true},
  "hostname": "census",
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
      {"path": "/dev/nvidiactl", "type": "c", "major": 195, "minor": 255,
       "fileMode": 438},
      {"path": "/dev/nvidia$GPU", "type": "c", "major": 195, "minor": $GPU,
       "fileMode": 438},
      {"path": "/dev/nvidia-uvm", "type": "c", "major": $UVM_MAJOR, "minor": 0,
       "fileMode": 438}
    ]
  }
}
EOF

status_of(){ runsc exec "$1" /bin/cat /tmp/status 2>/dev/null; }

wait_status(){
  local cid="$1" pattern="$2" timeout="$3" i s
  for ((i=0; i<timeout; i++)); do
    s=$(status_of "$cid")
    if grep -q "$pattern" <<<"$s"; then
      log "status: $s"
      return 0
    fi
    sleep 1
  done
  log "TIMEOUT waiting for status $pattern; last: $(status_of "$cid")"
  return 1
}

# --- run ---------------------------------------------------------------------
log "starting workload under runsc (bundle=$WORK/bundle)"
runsc run -detach -bundle "$WORK/bundle" -pid-file "$WORK/pid" "$CID" || {
  log "runsc run failed; boot log tail:"
  tail -30 "$WORK"/logs/*boot* 2>/dev/null; exit 1;
}
wait_status "$CID" "READY\|pass" 120 || { tail -40 "$WORK"/logs/*boot* 2>/dev/null; exit 1; }
sleep 3  # a few verified iterations
wait_status "$CID" "pass" 30 || exit 1

# --- checkpoint ----------------------------------------------------------------
log "checkpointing (cuda-checkpoint-path=$CUDA_CHECKPOINT)"
t0=$SECONDS
runsc checkpoint -image-path "$WORK/img" \
  -cuda-checkpoint-path "$CUDA_CHECKPOINT" -cuda-checkpoint-sequential \
  "$CID"
CKPT_RC=$?
log "checkpoint rc=$CKPT_RC ($((SECONDS-t0))s)"
[[ $CKPT_RC -eq 0 ]] || {
  grep -h 'nvproxy object census\|nvproxy census diff' "$WORK"/logs/* 2>/dev/null | tail -40
  tail -20 "$WORK"/logs/*boot* 2>/dev/null
  exit 1
}
runsc delete -force "$CID" >/dev/null 2>&1 || true

# --- restore -------------------------------------------------------------------
log "restoring"
t0=$SECONDS
runsc restore -detach -image-path "$WORK/img" -bundle "$WORK/bundle" \
  -pid-file "$WORK/pid-r" "$CID_R"
REST_RC=$?
log "restore rc=$REST_RC ($((SECONDS-t0))s)"
[[ $REST_RC -eq 0 ]] || { tail -30 "$WORK"/logs/*restore* 2>/dev/null; exit 1; }

# gVisor restores the sandbox clocks, so the workload can't detect the restore
# by a wall-clock jump; mark it explicitly.
runsc exec "$CID_R" /bin/touch /tmp/restored 2>/dev/null \
  || log "WARN: failed to create restore marker"

wait_status "$CID_R" "post-restore pass" 120
POST_RC=$?
log "final status: $(status_of "$CID_R")"

runsc kill "$CID_R" KILL >/dev/null 2>&1 || true

# --- extract the census ---------------------------------------------------------
echo ""
log "==== NVPROXY OBJECT CENSUS (the Phase 0 measurement) ===="
grep -h 'nvproxy object census\|nvproxy census diff' "$WORK"/logs/* 2>/dev/null \
  | sed 's/^.*] //' | awk '!seen[$0]++'
echo ""
log "full logs in $WORK/logs/; workload result above (post-restore pass = C/R worked)"
[[ $CKPT_RC -eq 0 && $REST_RC -eq 0 && $POST_RC -eq 0 ]]
