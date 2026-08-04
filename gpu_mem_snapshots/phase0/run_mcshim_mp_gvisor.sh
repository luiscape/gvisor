#!/usr/bin/env bash
# run_mcshim_mp_gvisor.sh -- gVisor e2e for the mcshim multicast interposer,
# MULTI-PROCESS ranks (one process per GPU: the vLLM/SGLang TP topology).
#
# A launcher (the cuda-checkpoint job root) forks one rank per GPU. Rank 0
# creates+exports the multicast group; peers import (app-level socketpair
# handoff, standing in for NCCL bootstrap). The LD_PRELOAD shim -- injected
# purely via container env -- transparently suspends the multicast layer on
# every rank before `runsc checkpoint` and rebuilds it after restore, brokering
# the cross-rank fd rendezvous over its own unix socket in the shared /tmp.
#
# Usage:
#   sudo [RUNSC=...] [CUDA_CHECKPOINT=...] [WORLD=2] bash run_mcshim_mp_gvisor.sh
set -uo pipefail
cd "$(dirname "$0")"
PHASE0_DIR=$(pwd)

RUNSC="${RUNSC:-/usr/local/bin/runsc-phase0}"
CUDA_CHECKPOINT="${CUDA_CHECKPOINT:-/usr/local/bin/cuda-checkpoint}"
WORLD="${WORLD:-2}"
WORK=/tmp/mcshim-mp-gvisor-test
STAGE=/opt/phase0
MCDIR=/tmp/mcshim   # inside the sandbox (container /tmp tmpfs)
CID=mcshimmp
CID_R=mcshimmp-r
log(){ echo "[mcshim-mp-gvisor $(date +%H:%M:%S)] $*"; }

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
cp "$PHASE0_DIR/mcshim_mp.py" "$PHASE0_DIR/_cuda.py" \
   "$PHASE0_DIR/mcshim/mcshim.so" "$STAGE/"
for m in nvidia nvidia_uvm; do
  mkdir -p "$STAGE/sys_module_$m"; echo live > "$STAGE/sys_module_$m/initstate"
done
chmod -R a+rX "$STAGE"

# GPU device nodes 0..WORLD-1.
DEVICES=""
for ((g=0; g<WORLD; g++)); do
  DEVICES+=",{\"path\": \"/dev/nvidia$g\", \"type\": \"c\", \"major\": 195, \"minor\": $g, \"fileMode\": 438}"
done

cat > "$WORK/bundle/config.json" <<EOF
{
  "ociVersion": "1.1.0",
  "process": {
    "terminal": false,
    "user": {"uid": 0, "gid": 0},
    "args": [
      "/usr/bin/python3", "$STAGE/mcshim_mp.py",
      "--world", "$WORLD", "--dir", "$MCDIR"
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
  "hostname": "mcshimmp",
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
      {"path": "/dev/nvidia-uvm", "type": "c", "major": $UVM_MAJOR, "minor": 0, "fileMode": 438}$DEVICES
    ]
  }
}
EOF

rank_statuses(){
  local cid="$1"
  runsc exec "$cid" /bin/sh -c "cat $MCDIR/wl.status.rank* 2>/dev/null"
}
wait_all_status(){
  local cid="$1" pattern="$2" timeout="$3" i s n
  for ((i=0; i<timeout; i++)); do
    s=$(rank_statuses "$cid")
    n=$(grep -c "$pattern" <<<"$s")
    [[ "$n" -ge "$WORLD" ]] && { log "all $WORLD ranks: $pattern"; return 0; }
    sleep 1
  done
  log "TIMEOUT waiting for all-ranks $pattern; last:"; rank_statuses "$cid" | sed 's/^/    /'
  return 1
}
wait_acks(){
  local cid="$1" prefix="$2" timeout="$3" i n
  for ((i=0; i<timeout; i++)); do
    n=$(runsc exec "$cid" /bin/sh -c "ls $MCDIR/$prefix.* 2>/dev/null | wc -l")
    [[ "${n:-0}" -ge "$WORLD" ]] && return 0
    sleep 1
  done
  log "TIMEOUT waiting for $WORLD $prefix.* acks (got ${n:-0})"
  runsc exec "$cid" /bin/sh -c "cat $MCDIR/error.* 2>/dev/null" || true
  runsc exec "$cid" /bin/sh -c "tail -20 $MCDIR/mcshim.log 2>/dev/null" || true
  return 1
}
touch_marker(){ runsc exec "$1" /bin/touch "$MCDIR/$2"; }
rm_marker(){ runsc exec "$1" /bin/rm -f "$MCDIR/$2"; }

FAIL=0

log "starting $WORLD-rank live-multicast workload under runsc (shim via env)"
runsc run -detach -bundle "$WORK/bundle" -pid-file "$WORK/pid" "$CID" \
  || { log "runsc run failed"; tail -30 "$WORK"/logs/*boot* 2>/dev/null; exit 1; }
wait_all_status "$CID" "mc-live pass" 180 || { tail -40 "$WORK"/logs/*boot*; exit 1; }

log "(a) pause all ranks (quiesce)"
touch_marker "$CID" pause
wait_all_status "$CID" "PAUSED" 30 || { log "FAIL: ranks did not pause"; exit 1; }

log "(b) SHIM multicast suspend on every rank"
touch_marker "$CID" suspend
wait_acks "$CID" suspended 60 || { log "FAIL: shims did not suspend"; exit 1; }
log "all $WORLD shims SUSPENDED"

log "(c) checkpoint (job protocol; zero multicast blockers expected)"
t0=$SECONDS
runsc checkpoint -image-path "$WORK/img" \
  -cuda-checkpoint-path "$CUDA_CHECKPOINT" -cuda-checkpoint-sequential "$CID"
CKPT_RC=$?
log "checkpoint rc=$CKPT_RC ($((SECONDS-t0))s)"
[[ $CKPT_RC -eq 0 ]] || { log "FAIL: checkpoint refused/failed"; \
  grep -h 'blocker' "$WORK"/logs/*boot* 2>/dev/null | tail -3; exit 1; }
runsc delete -force "$CID" >/dev/null 2>&1 || true

log "restore"
t0=$SECONDS
runsc restore -detach -image-path "$WORK/img" -bundle "$WORK/bundle" \
  -pid-file "$WORK/pid-r" "$CID_R"
REST_RC=$?
log "restore rc=$REST_RC ($((SECONDS-t0))s)"
sleep 2
TOGGLE_ERR=$(grep -h 'Killing the sandbox after post restore' "$WORK"/logs/*boot* 2>/dev/null | tail -1)
if [[ -n "$TOGGLE_ERR" ]]; then
  log "FAIL: restore toggle failed even after shim suspend:"
  log "  ${TOGGLE_ERR#*restore work failed: }"
  exit 1
fi
[[ $REST_RC -eq 0 ]] || { log "FAIL: restore failed"; exit 1; }

log "SHIM multicast resume on every rank (rank0 serves fd; peers re-import)"
rm_marker "$CID_R" suspend
wait_acks "$CID_R" resumed 90 || { log "FAIL: shims did not all resume"; FAIL=1; }
log "all $WORLD shims RESUMED"

log "unpause + post-restore verification on every rank"
rm_marker "$CID_R" pause
wait_all_status "$CID_R" "post-restore+mc-live pass" 60 || FAIL=1
sleep 4
FINAL=$(rank_statuses "$CID_R")
log "final rank statuses:"; sed 's/^/    /' <<<"$FINAL"
n=$(grep -c "post-restore+mc-live pass" <<<"$FINAL"); [[ "$n" -ge "$WORLD" ]] || FAIL=1
n=$(grep -c "failures=0" <<<"$FINAL"); [[ "$n" -ge "$WORLD" ]] || FAIL=1

echo ""
log "==== SHIM LOG (suspend/resume/rendezvous) ===="
runsc exec "$CID_R" /bin/sh -c "cat $MCDIR/mcshim.log 2>/dev/null" \
  | grep -E 'SUSPEND done|RESUME|serving|IMPORT|EXPORT' | tail -14

runsc kill "$CID_R" KILL >/dev/null 2>&1 || true

echo ""
if [[ $FAIL -eq 0 ]]; then
  log "==== RESULT: PASS ===="
  log "$WORLD-process multicast (create/export/import, vLLM/SGLang topology)"
  log "round-trips gVisor checkpoint/restore transparently; the shims brokered"
  log "the cross-rank fd rendezvous inside the sandbox."
else
  log "==== RESULT: FAIL ===="
fi
exit $FAIL
