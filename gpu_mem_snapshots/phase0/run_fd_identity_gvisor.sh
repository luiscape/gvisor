#!/usr/bin/env bash
# Step-0 probe under gVisor: run fd_identity_probe.py in a runsc sandbox and
# report whether exported-object fds have distinct identities in-sandbox.
set -uo pipefail
cd "$(dirname "$0")"
PHASE0_DIR=$(pwd)

RUNSC="${RUNSC:-/usr/local/bin/runsc-phase0}"
GPUS="${GPUS:-0,1}"
GPU_A="${GPUS%%,*}"; GPU_B="${GPUS##*,}"
WORK=/tmp/fdid-gvisor-test
STAGE=/opt/phase0
CID=fdid
log(){ echo "[fdid-gvisor $(date +%H:%M:%S)] $*"; }

[[ -x "$RUNSC" ]] || { log "runsc not found at $RUNSC"; exit 1; }
UVM_MAJOR=$(awk '$2=="nvidia-uvm"{print $1}' /proc/devices)

runsc(){ "$RUNSC" --root "$WORK/root" --debug --debug-log="$WORK/logs/" "$@"; }
cleanup(){ runsc delete -force "$CID" >/dev/null 2>&1 || true; }
trap cleanup EXIT

cleanup
rm -rf "$WORK"; mkdir -p "$WORK"/{root,logs,bundle}
mkdir -p "$STAGE"
cp "$PHASE0_DIR/fd_identity_probe.py" "$PHASE0_DIR/_cuda.py" "$STAGE/"
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
    "args": ["/usr/bin/python3", "$STAGE/fd_identity_probe.py"],
    "env": ["PATH=/usr/local/bin:/usr/bin:/bin"],
    "cwd": "/"
  },
  "root": {"path": "/", "readonly": true},
  "hostname": "fdid",
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

log "running fd_identity_probe.py under runsc"
runsc run -bundle "$WORK/bundle" "$CID"
RC=$?
log "probe exit code: $RC (0 = distinct, 2 = colliding)"
exit $RC
