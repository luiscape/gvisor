#!/usr/bin/env bash
# run_nccl_suspend_mp_gvisor.sh — MULTI-PROCESS NCCL suspend/resume under
# gVisor: one process per GPU (vLLM/SGLang tensor-parallel topology), all
# wrapped in a single cuda-checkpoint --launch-job, checkpointed/restored by
# runsc.
#
# This is the faithful reduction of the repo's vLLM/SGLang multi-GPU case
# (cf. gpu_mem_snapshots/repros/repro_tp_nccl.py "graph" mode, which mimics a
# captured-graph + coupled-NCCL TP worker), with the addition of the NCCL
# ncclCommSuspend/Resume calls that make NVLS checkpointable.
#
#   (a) container entrypoint = nccl_mp_launcher.py; the loader wraps it in
#       `cuda-checkpoint --launch-job`, so the launcher AND its forked rank
#       children share one checkpoint job.
#   (b) `runsc exec touch /shared/suspend` -> every rank ncclCommSuspend
#   (c) `runsc checkpoint` (job protocol) -> `runsc restore`
#       `runsc exec touch /shared/resume` -> every rank ncclCommResume
#   PASS = all ranks post-restore pass failures=0 (eager + CUDA graph).
#
# Usage:
#   sudo nvidia-smi -pm 1
#   sudo [WORLD=4] bash run_nccl_suspend_mp_gvisor.sh
set -uo pipefail
cd "$(dirname "$0")"
PHASE0_DIR=$(pwd)

RUNSC="${RUNSC:-/usr/local/bin/runsc-phase0}"
CUDA_CHECKPOINT="${CUDA_CHECKPOINT:-/usr/local/bin/cuda-checkpoint}"
WORLD="${WORLD:-4}"
WORK=/tmp/nccl-mp-gvisor
STAGE=/opt/phase0
CID=ncclmp
CID_R=ncclmp-r
log(){ echo "[mp-gvisor $(date +%H:%M:%S)] $*"; }

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
cleanup(){
  runsc delete -force "$CID"   >/dev/null 2>&1 || true
  runsc delete -force "$CID_R" >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup
rm -rf "$WORK" 2>/dev/null
mkdir -p "$WORK"/{root,logs,img,bundle}

mkdir -p "$STAGE"
cp "$PHASE0_DIR"/{nccl_mp_launcher.py,nccl_suspend_mp.py,_nccl.py,_cuda.py} "$STAGE/"
for m in nvidia nvidia_uvm; do
  mkdir -p "$STAGE/sys_module_$m"; echo live > "$STAGE/sys_module_$m/initstate"
done
chmod -R a+rX "$STAGE"

DEVS=""
for ((g=0; g<WORLD; g++)); do
  DEVS+="      {\"path\": \"/dev/nvidia$g\", \"type\": \"c\", \"major\": 195, \"minor\": $g, \"fileMode\": 438},"$'\n'
done

cat > "$WORK/bundle/config.json" <<EOF
{
  "ociVersion": "1.1.0",
  "process": {
    "terminal": false,
    "user": {"uid": 0, "gid": 0},
    "args": [
      "/usr/bin/python3", "$STAGE/nccl_mp_launcher.py",
      "--dir", "/tmp", "--world", "$WORLD", "--graph", "--interval", "1"
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
  "hostname": "ncclmp",
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

# rank status files live in the container's tmpfs /tmp; read them via exec.
all_status(){ runsc exec "$1" /bin/sh -c 'cat /tmp/status.* 2>/dev/null'; }
all_have(){  # pattern present in ALL rank status files
  local cid="$1" pattern="$2" s r
  s=$(all_status "$cid")
  for ((r=0; r<WORLD; r++)); do
    grep -q "\[rank$r\].*$pattern" <<<"$s" || return 1
  done
  return 0
}
wait_all(){
  local cid="$1" pattern="$2" timeout="$3" i
  for ((i=0; i<timeout; i++)); do
    all_have "$cid" "$pattern" && { log "all ranks: $pattern"; return 0; }
    sleep 1
  done
  log "TIMEOUT waiting all ranks '$pattern'; last:"; all_status "$cid" | tail -$WORLD
  return 1
}

FAIL=0

log "(a) launching $WORLD-rank NCCL+NVLS TP group under runsc (job-wrapped)"
runsc run -detach -bundle "$WORK/bundle" -pid-file "$WORK/pid" "$CID" \
  || { log "runsc run failed"; tail -30 "$WORK"/logs/*boot* 2>/dev/null; exit 1; }
wait_all "$CID" "pre-checkpoint pass" 300 || { tail -40 "$WORK"/logs/*boot*; exit 1; }

log "(b) suspend all ranks (ncclCommSuspend via /tmp/suspend)"
runsc exec "$CID" /bin/touch /tmp/suspend
wait_all "$CID" "suspended (idle)" 60 || { log "FAIL: suspend"; FAIL=1; }

log "(c) runsc checkpoint (job over launcher + all ranks)"
t0=$SECONDS
runsc checkpoint -image-path "$WORK/img" \
  -cuda-checkpoint-path "$CUDA_CHECKPOINT" -cuda-checkpoint-sequential "$CID"
CKPT_RC=$?
log "checkpoint rc=$CKPT_RC ($((SECONDS-t0))s)"
[[ $CKPT_RC -eq 0 ]] || { log "FAIL: checkpoint"; grep -h 'blocker\|suspend' "$WORK"/logs/*boot* | tail -5; exit 1; }
runsc delete -force "$CID" >/dev/null 2>&1 || true

log "restore"
t0=$SECONDS
runsc restore -detach -image-path "$WORK/img" -bundle "$WORK/bundle" \
  -pid-file "$WORK/pid-r" "$CID_R"
REST_RC=$?
log "restore rc=$REST_RC ($((SECONDS-t0))s)"
sleep 2
TOGGLE_ERR=$(grep -h 'Killing the sandbox after post restore' "$WORK"/logs/*boot* 2>/dev/null | tail -1)
[[ -z "$TOGGLE_ERR" && $REST_RC -eq 0 ]] || { log "FAIL: restore: ${TOGGLE_ERR#*restore work failed: }"; exit 1; }

runsc exec "$CID_R" /bin/touch /tmp/restored 2>/dev/null || true
log "resume all ranks (ncclCommResume via /tmp/resume)"
runsc exec "$CID_R" /bin/touch /tmp/resume
wait_all "$CID_R" "post-restore pass" 120 || { log "FAIL: post-restore verification"; FAIL=1; }
sleep 4

echo ""
log "==== per-rank final status ===="
FINAL=$(all_status "$CID_R")
grep -oE '\[rank[0-9]+\] iter=[0-9]+ [a-z-]+ pass failures=[0-9]+' <<<"$FINAL" | sort -u
for ((r=0; r<WORLD; r++)); do grep -q "\[rank$r\].*failures=0" <<<"$FINAL" || FAIL=1; done
log "==== SUSPEND/RESUME (NVLS) log lines ===="
grep -h 'NVLS Suspend\|NVLS Resume' "$WORK"/logs/*boot* 2>/dev/null | sed 's/^.*\] //' | awk '!seen[$0]++' | head -6

runsc kill "$CID_R" KILL >/dev/null 2>&1 || true
echo ""
log "==== RESULT: $([[ $FAIL -eq 0 ]] && echo PASS || echo FAIL) ===="
exit $FAIL
