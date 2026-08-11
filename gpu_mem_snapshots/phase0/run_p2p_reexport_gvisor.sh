#!/usr/bin/env bash
# run_p2p_reexport_gvisor.sh -- p2p_reexport_probe under gVisor: does a
# re-imported VMM P2P allocation's NVLink peer access survive a
# runsc checkpoint/restore? (Native already answered YES; this isolates
# whether gVisor's C/R + nvproxy replay is what breaks the mcshim NCCL run.)
#
# The launcher's two ranks do the release/re-import in-process on markers; the
# host drives release -> runsc checkpoint -> runsc restore -> resume, then
# reads the post-restore peer-read result.
set -uo pipefail
cd "$(dirname "$0")"
PHASE0_DIR=$(pwd)

RUNSC="${RUNSC:-/usr/local/bin/runsc-phase0}"
CUDA_CHECKPOINT="${CUDA_CHECKPOINT:-/usr/local/bin/cuda-checkpoint}"
GPUS="${GPUS:-0,1}"
GPU_A="${GPUS%%,*}"; GPU_B="${GPUS##*,}"
WORK=/tmp/p2p-reexport-gvisor
STAGE=/opt/phase0
D=/tmp/p2p_reexport_probe   # in-sandbox marker dir (must match the probe)
CID=p2prx
CID_R=p2prx-r
log(){ echo "[p2prx-gvisor $(date +%H:%M:%S)] $*"; }

[[ -x "$RUNSC" ]] || { log "runsc not found"; exit 1; }
UVM_MAJOR=$(awk '$2=="nvidia-uvm"{print $1}' /proc/devices)

JOB_FLAG=(--cuda-checkpoint-path "$CUDA_CHECKPOINT" --network=none)
runsc(){
  local sub="$1"
  if [[ "$sub" == "run" || "$sub" == "restore" ]]; then
    "$RUNSC" --root "$WORK/root" --debug --debug-log="$WORK/logs/" "${JOB_FLAG[@]}" "$@"
  else
    "$RUNSC" --root "$WORK/root" --debug --debug-log="$WORK/logs/" "$@"
  fi
}
cleanup(){ runsc delete -force "$CID" >/dev/null 2>&1 || true; runsc delete -force "$CID_R" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup; rm -rf "$WORK" 2>/dev/null; mkdir -p "$WORK"/{root,logs,img,bundle}
mkdir -p "$STAGE"; cp "$PHASE0_DIR"/{p2p_reexport_probe.py,_cuda.py} "$STAGE/"
for m in nvidia nvidia_uvm; do mkdir -p "$STAGE/sys_module_$m"; echo live > "$STAGE/sys_module_$m/initstate"; done
chmod -R a+rX "$STAGE"

cat > "$WORK/bundle/config.json" <<EOF
{
  "ociVersion": "1.1.0",
  "process": {
    "terminal": false, "user": {"uid": 0, "gid": 0},
    "args": ["/usr/bin/python3", "$STAGE/p2p_reexport_probe.py", "--launcher", "--gpus", "0,1"],
    "env": ["PATH=/usr/local/bin:/usr/bin:/bin", "P2P_THREADED=${P2P_THREADED:-0}", "P2P_NBUF=${P2P_NBUF:-48}"], "cwd": "/"
  },
  "root": {"path": "/", "readonly": true},
  "hostname": "p2prx",
  "mounts": [
    {"destination": "/proc", "type": "proc"},
    {"destination": "/tmp", "type": "tmpfs"},
    {"destination": "/sys/module/nvidia", "type": "bind", "source": "$STAGE/sys_module_nvidia", "options": ["rbind", "ro"]},
    {"destination": "/sys/module/nvidia_uvm", "type": "bind", "source": "$STAGE/sys_module_nvidia_uvm", "options": ["rbind", "ro"]}
  ],
  "linux": {
    "namespaces": [{"type": "pid"}, {"type": "ipc"}, {"type": "uts"}, {"type": "mount"}, {"type": "network"}],
    "devices": [
      {"path": "/dev/nvidiactl", "type": "c", "major": 195, "minor": 255, "fileMode": 438},
      {"path": "/dev/nvidia$GPU_A", "type": "c", "major": 195, "minor": $GPU_A, "fileMode": 438},
      {"path": "/dev/nvidia$GPU_B", "type": "c", "major": 195, "minor": $GPU_B, "fileMode": 438},
      {"path": "/dev/nvidia-uvm", "type": "c", "major": $UVM_MAJOR, "minor": 0, "fileMode": 438}
    ]
  }
}
EOF

rex(){ runsc exec "$CID" /bin/sh -c "$1" 2>/dev/null; }
rexr(){ runsc exec "$CID_R" /bin/sh -c "$1" 2>/dev/null; }
wait_in(){ local cid="$1" f="$2" t="$3" i; for ((i=0;i<t;i++)); do runsc exec "$cid" /bin/test -f "$D/$f" 2>/dev/null && return 0; sleep 1; done; log "TIMEOUT $f"; return 1; }

log "run launcher (2 ranks, GPU $GPU_A exporter / GPU $GPU_B importer), job-wrapped"
runsc run -detach -bundle "$WORK/bundle" -pid-file "$WORK/pid" "$CID" \
  || { log "run failed"; tail -20 "$WORK"/logs/*boot* 2>/dev/null; exit 1; }
wait_in "$CID" ready.1 180 || { tail -30 "$WORK"/logs/*boot*; exit 1; }
BASE=$(rex "cat $D/ready.1"); log "baseline: $BASE"

log "release import (in-process)"
rex "touch $D/release" >/dev/null
wait_in "$CID" released.1 30 || exit 1

log "runsc checkpoint"
runsc checkpoint -image-path "$WORK/img" -cuda-checkpoint-path "$CUDA_CHECKPOINT" -cuda-checkpoint-sequential "$CID"
[[ $? -eq 0 ]] || { log "FAIL checkpoint"; exit 1; }
runsc delete -force "$CID" >/dev/null 2>&1 || true

log "runsc restore"
runsc restore -detach -image-path "$WORK/img" -bundle "$WORK/bundle" -pid-file "$WORK/pid-r" "$CID_R"
[[ $? -eq 0 ]] || { log "FAIL restore"; exit 1; }
sleep 2

log "resume (re-export + re-import + peer-read kernel)"
rexr "touch $D/resume" >/dev/null
wait_in "$CID_R" resumed.1 60 || exit 1
POST=$(rexr "cat $D/resumed.1")
for r in 0 1; do
  rexr "cat $D/maps.pre.$r"  > "$WORK/maps.pre.$r"  2>/dev/null || true
  rexr "cat $D/maps.post.$r" > "$WORK/maps.post.$r" 2>/dev/null || true
done
rexr "touch $D/stop" >/dev/null
runsc kill "$CID_R" KILL >/dev/null 2>&1 || true

echo ""
log "baseline: $BASE"
log "post:     $POST"
if [[ "$POST" == *"post=OK"* ]]; then
  log "==== RESULT: PASS (peer access survives gVisor C/R) -> mcshim 719 is elsewhere ===="
else
  log "==== RESULT: FAIL (gVisor C/R breaks re-imported peer access) ===="
fi
