#!/usr/bin/env bash
# run_nccl_suspend_gvisor.sh — the NCCL suspend/resume checkpoint model under
# gVisor on 4 GPUs (NVLS engaged), with nvproxy as the instrument:
#
#   (a) 4-GPU NCCL clique + verified allreduce + captured CUDA graph
#   (b) ncclCommSuspend(NCCL_SUSPEND_MEM) on all comms  [upstream NCCL API]
#   (*) nvproxy blocker census: WHAT SURVIVES the NCCL suspend? (00FD/00F8...)
#   (c) runsc checkpoint:
#        MODE=gate    (default) -> default gate; expect refusal listing the
#                      survivors (pure diagnostic)
#        MODE=combo   -> --cuda-multicast-suspend: nvproxy suspends the
#                      surviving bare multicast objects on top of NCCL's
#                      suspend; then restore -> toggle -> nvproxy replay ->
#                      ncclCommResume -> verified allreduce + graph replay
#
# Usage:
#   sudo nvidia-smi -pm 1
#   sudo [MODE=gate|combo] [NGPUS=4] bash run_nccl_suspend_gvisor.sh
set -uo pipefail
cd "$(dirname "$0")"
PHASE0_DIR=$(pwd)

RUNSC="${RUNSC:-/usr/local/bin/runsc-phase0}"
CUDA_CHECKPOINT="${CUDA_CHECKPOINT:-/usr/local/bin/cuda-checkpoint}"
NGPUS="${NGPUS:-4}"
# MODE=plain  : patched NCCL suspended NVLS; plain runsc checkpoint (no nvproxy flag)
# MODE=combo  : additionally enable nvproxy --cuda-multicast-suspend
# MODE=gate   : diagnostic; enumerate what survives ncclCommSuspend, then stop
MODE="${MODE:-plain}"
WORK=/tmp/nccl-suspend-gvisor
STAGE=/opt/phase0
CID=ncclsusp
CID_R=ncclsusp-r
log(){ echo "[nccl-gvisor $(date +%H:%M:%S)] $*"; }

[[ -x "$RUNSC" ]] || { log "runsc not found"; exit 1; }
[[ -x "$CUDA_CHECKPOINT" ]] || { log "cuda-checkpoint not found"; exit 1; }
UVM_MAJOR=$(awk '$2=="nvidia-uvm"{print $1}' /proc/devices)

# --network=none provides a netstack loopback (NCCL bootstrap needs a socket
# interface; a bare OCI bundle otherwise has none). Loopback-only sockets live
# entirely in sentry state, so they checkpoint/restore cleanly.
JOB_FLAG=(--cuda-checkpoint-path "$CUDA_CHECKPOINT" --network=none)
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
cp "$PHASE0_DIR"/{nccl_suspend_workload.py,_nccl.py,_cuda.py} "$STAGE/"
for m in nvidia nvidia_uvm; do
  mkdir -p "$STAGE/sys_module_$m"; echo live > "$STAGE/sys_module_$m/initstate"
done
chmod -R a+rX "$STAGE"

# GPU device entries for the OCI spec.
DEVS=""
for ((g=0; g<NGPUS; g++)); do
  DEVS+="      {\"path\": \"/dev/nvidia$g\", \"type\": \"c\", \"major\": 195, \"minor\": $g, \"fileMode\": 438},"$'\n'
done

cat > "$WORK/bundle/config.json" <<EOF
{
  "ociVersion": "1.1.0",
  "process": {
    "terminal": false,
    "user": {"uid": 0, "gid": 0},
    "args": [
      "/usr/bin/python3", "$STAGE/nccl_suspend_workload.py",
      "--ngpus", "$NGPUS", "--graph", "--dir", "/tmp", "--interval", "1"
    ],
    "env": [
      "PATH=/usr/local/bin:/usr/bin:/bin",
      "NCCL_LIB=$STAGE/nccl/nvidia/nccl/lib/libnccl.so.2",
      "NCCL_NVLS_ENABLE=1",
      "NCCL_SOCKET_IFNAME=lo"
    ],
    "cwd": "/"
  },
  "root": {"path": "/", "readonly": true},
  "hostname": "ncclsusp",
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
      {"type": "pid"}, {"type": "ipc"}, {"type": "uts"}, {"type": "mount"},
      {"type": "network"}
    ],
    "devices": [
$DEVS      {"path": "/dev/nvidiactl", "type": "c", "major": 195, "minor": 255, "fileMode": 438},
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

log "(a) starting $NGPUS-GPU NCCL+NVLS workload under runsc (job-wrapped, graph)"
runsc run -detach -bundle "$WORK/bundle" -pid-file "$WORK/pid" "$CID" \
  || { log "runsc run failed"; tail -30 "$WORK"/logs/*boot* 2>/dev/null; exit 1; }
wait_status "$CID" "pre-checkpoint pass" 300 || { tail -40 "$WORK"/logs/*boot*; exit 1; }

log "(b) ncclCommSuspend(NCCL_SUSPEND_MEM) on all comms"
runsc exec "$CID" /bin/touch /tmp/suspend
wait_status "$CID" "suspended (idle)" 60 || { log "FAIL: suspend"; exit 1; }

if [[ "$MODE" == "gate" ]]; then
  log "(*) DIAGNOSTIC: default gate enumerates what survives ncclCommSuspend"
  runsc checkpoint -image-path "$WORK/img" \
    -cuda-checkpoint-path "$CUDA_CHECKPOINT" -cuda-checkpoint-sequential \
    -cuda-blocker-timeout 5s "$CID" 2>&1 | sed 's/^/    /' | head -5
  log "(see the blocker list above: kind/count of RM objects NCCL suspend left live)"
  exit 0
fi

# ---------------- MODE=combo (default now) ----------------
# With the patched NCCL, ncclCommSuspend already released the NVLS multicast,
# so a plain checkpoint should succeed (no --cuda-multicast-suspend needed).
# combo additionally turns on nvproxy multicast suspend to catch any stray
# multicast objects NCCL suspend did not cover.
EXTRA=()
[[ "$MODE" == "combo" ]] && EXTRA=(-cuda-multicast-suspend)
log "(c) runsc checkpoint (NCCL already suspended NVLS)${EXTRA:+ + nvproxy multicast-suspend}"
t0=$SECONDS
runsc checkpoint -image-path "$WORK/img" \
  -cuda-checkpoint-path "$CUDA_CHECKPOINT" -cuda-checkpoint-sequential \
  "${EXTRA[@]}" "$CID"
CKPT_RC=$?
log "checkpoint rc=$CKPT_RC ($((SECONDS-t0))s)"
[[ $CKPT_RC -eq 0 ]] || { log "FAIL: checkpoint"; grep -h 'blocker\|suspend' "$WORK"/logs/*boot* | tail -5; exit 1; }
runsc delete -force "$CID" >/dev/null 2>&1 || true

log "restore"
runsc restore -detach -image-path "$WORK/img" -bundle "$WORK/bundle" \
  -pid-file "$WORK/pid-r" "$CID_R"
REST_RC=$?
log "restore rc=$REST_RC"
sleep 2
TOGGLE_ERR=$(grep -h 'Killing the sandbox after post restore' "$WORK"/logs/*boot* 2>/dev/null | tail -1)
if [[ -n "$TOGGLE_ERR" ]]; then
  log "RESTORE-BOUNDARY: toggle failed even after NCCL+nvproxy combined suspend:"
  log "  ${TOGGLE_ERR#*restore work failed: }"
  exit 1
fi
[[ $REST_RC -eq 0 ]] || { log "FAIL: restore"; exit 1; }

runsc exec "$CID_R" /bin/touch /tmp/restored 2>/dev/null || true
log "ncclCommResume on all comms"
runsc exec "$CID_R" /bin/touch /tmp/resume
# RESUMED is a transient single-line status marker; check the sandbox log too.
RESUMED_OK=0
for i in $(seq 60); do
  s=$(status_of "$CID_R")
  grep -q 'RESUMED\|post-restore pass' <<<"$s" && { RESUMED_OK=1; break; }
  sleep 1
done
[[ $RESUMED_OK -eq 1 ]] || { log "FAIL: resume did not complete"; FAIL=1; }
wait_status "$CID_R" "post-restore pass" 120 || { log "FAIL: post-restore verification"; FAIL=1; }
sleep 5
FINAL=$(status_of "$CID_R")
log "final: $FINAL"
grep -q "pass" <<<"$FINAL" && grep -q "failures=0" <<<"$FINAL" || FAIL=1

runsc kill "$CID_R" KILL >/dev/null 2>&1 || true

echo ""
log "==== KEY LOG LINES ===="
grep -hE 'SUSPENDED|RESUMED|suspended multicast|replaying multicast|recreated by' \
  "$WORK"/logs/*boot* /dev/null 2>/dev/null | sed 's/^.*\] //' | awk '!seen[$0]++' | head
log "==== RESULT: $([[ $FAIL -eq 0 ]] && echo PASS || echo FAIL) ===="
exit $FAIL
