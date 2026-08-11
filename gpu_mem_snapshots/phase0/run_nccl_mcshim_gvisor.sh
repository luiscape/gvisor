#!/usr/bin/env bash
# run_nccl_mcshim_gvisor.sh -- ACCEPTANCE: STOCK (unpatched) NCCL NVLS,
# multi-process ranks (one per GPU, vLLM/SGLang TP topology), checkpointed and
# restored under gVisor via the mcshim LD_PRELOAD interposer -- NO NCCL fork,
# NO engine hooks.
#
# The ranks run a real NCCL NVLS allreduce + captured CUDA graph and never call
# ncclCommSuspend (`--pause-only`). The shim transparently:
#   - suspends the multicast layer AND the ~48/rank live P2P UC imports,
#   - rebuilds them on resume, using nvproxy's fdinfo identity oracle
#     (client:object) to match each importer to its exporter across restore.
# The launcher + all ranks share one cuda-checkpoint --launch-job.
#
# Usage:  sudo [WORLD=4] bash run_nccl_mcshim_gvisor.sh
set -uo pipefail
cd "$(dirname "$0")"
PHASE0_DIR=$(pwd)

RUNSC="${RUNSC:-/usr/local/bin/runsc-phase0}"
CUDA_CHECKPOINT="${CUDA_CHECKPOINT:-/usr/local/bin/cuda-checkpoint}"
NCCL_STOCK="${NCCL_STOCK:-/opt/phase0/nccl-stock/libnccl.so.2}"
NVLS="${NCCL_NVLS_ENABLE:-1}"   # set 0 to bisect: P2P UC imports only, no multicast
GRAPH="${GRAPH:-1}"             # set 0 to bisect: eager collective only, no CUDA graph
WORLD="${WORLD:-4}"
# MCSHIM_UC_REBUILD: 0 = carry unicast allocations through cuda-checkpoint
# (default), 1 = re-create IPC-exported ones fresh at resume, 2 = re-create all.
# Modes >=1 mirror NCCL's ncclCommMemSuspend policy of never asking
# cuda-checkpoint to carry IPC-exported device memory.
UC_REBUILD="${MCSHIM_UC_REBUILD:-0}"
# TOGGLE_ONLY=1 replaces `runsc checkpoint`+`runsc restore` with the bare
# cuda-checkpoint lock/checkpoint/restore/unlock sequence that state_cuda.go
# drives, run in place via `runsc exec`. Same GPU-side operations, but no
# gVisor memory save/restore in between -- so a context fault under
# TOGGLE_ONLY=1 is attributable to cuda-checkpoint alone.
TOGGLE_ONLY="${TOGGLE_ONLY:-0}"
WORK=/tmp/nccl-mcshim-gvisor
STAGE=/opt/phase0
MCDIR=/tmp/mcshim
CID=ncclmcshim
CID_R=ncclmcshim-r
log(){ echo "[nccl-mcshim-gvisor $(date +%H:%M:%S)] $*"; }

[[ -x "$RUNSC" ]] || { log "runsc not found at $RUNSC"; exit 1; }
[[ -f "$NCCL_STOCK" ]] || { log "stock NCCL not found at $NCCL_STOCK"; exit 1; }
[[ -f "$PHASE0_DIR/mcshim/mcshim.so" ]] || { log "mcshim.so not built"; exit 1; }
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
cp "$PHASE0_DIR/mcshim/mcshim.so" "$STAGE/"
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
      "--dir", "$MCDIR", "--world", "$WORLD", $([ "$GRAPH" = 1 ] && echo '"--graph",') "--pause-only",
      "--interval", "1"
    ],
    "env": [
      "PATH=/usr/local/bin:/usr/bin:/bin",
      "LD_PRELOAD=$STAGE/mcshim.so",
      "MCSHIM_DIR=$MCDIR",
      "MCSHIM_LOG=$MCDIR/mcshim.log",
      "MCSHIM_UC_REBUILD=$UC_REBUILD",
      "NCCL_LIB=$NCCL_STOCK",
      "NCCL_NVLS_ENABLE=$NVLS",
      "NCCL_SOCKET_IFNAME=lo",
      "NCCL_DEBUG=WARN",
      "NCCL_DEBUG_FILE=$MCDIR/nccl.%h.%p.log"
    ],
    "cwd": "/"
  },
  "root": {"path": "/", "readonly": true},
  "hostname": "ncclmcshim",
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

all_status(){ runsc exec "$1" /bin/sh -c "cat $MCDIR/status.* 2>/dev/null"; }
all_have(){
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
wait_acks(){  # $2 prefix, $3 timeout: WORLD shim ack files
  local cid="$1" prefix="$2" timeout="$3" i n
  for ((i=0; i<timeout; i++)); do
    n=$(runsc exec "$cid" /bin/sh -c "ls $MCDIR/$prefix.* 2>/dev/null | wc -l")
    [[ "${n:-0}" -ge "$WORLD" ]] && { log "$WORLD $prefix acks"; return 0; }
    sleep 1
  done
  log "TIMEOUT: ${n:-0}/$WORLD $prefix acks"
  runsc exec "$cid" /bin/sh -c "cat $MCDIR/error.* 2>/dev/null" || true
  runsc exec "$cid" /bin/sh -c "tail -20 $MCDIR/mcshim.log 2>/dev/null" || true
  return 1
}
touch_m(){ runsc exec "$1" /bin/touch "$MCDIR/$2"; }
rm_m(){ runsc exec "$1" /bin/rm -f "$MCDIR/$2"; }

FAIL=0

log "(a) launching $WORLD-rank STOCK-NCCL NVLS TP group under runsc (job-wrapped)"
runsc run -detach -bundle "$WORK/bundle" -pid-file "$WORK/pid" "$CID" \
  || { log "runsc run failed"; tail -30 "$WORK"/logs/*boot* 2>/dev/null; exit 1; }
wait_all "$CID" "pre-checkpoint pass" 300 || { tail -40 "$WORK"/logs/*boot*; exit 1; }
NVER=$(runsc exec "$CID" /bin/sh -c "grep -m1 -oE 'NCCL [0-9]+' $MCDIR/status.0 2>/dev/null" || true)
log "NCCL banner: ${NVER:-?} (stock, no suspend patch)"

log "(a) pause all ranks (quiesce; stock NCCL is idle, no suspend call)"
touch_m "$CID" pause
wait_all "$CID" "PAUSED" 60 || { log "FAIL: pause"; exit 1; }

log "(b) SHIM suspend: multicast layer + live P2P UC imports on every rank"
touch_m "$CID" suspend
wait_acks "$CID" suspended 120 || { log "FAIL: shim suspend"; exit 1; }
log "shim SUSPENDED: $(runsc exec "$CID" /bin/sh -c "grep -c 'SUSPEND done' $MCDIR/mcshim.log")/$WORLD"
runsc exec "$CID" /bin/sh -c "grep -m1 'SUSPEND done' $MCDIR/mcshim.log" || true

if [[ "$TOGGLE_ONLY" = 1 ]]; then
  # Rank pids are the suffixes of the shim's ack files.
  PIDS=$(runsc exec "$CID" /bin/sh -c "ls $MCDIR/suspended.* | sed 's|.*suspended\.||'" | tr -d '\r')
  log "(c) TOGGLE-ONLY: cuda-checkpoint over pids: $(echo $PIDS | tr '\n' ' ')"
  for action in lock checkpoint restore unlock; do
    for p in $PIDS; do
      out=$(runsc exec "$CID" "$CUDA_CHECKPOINT" --action "$action" --pid "$p" 2>&1)
      rc=$?
      [[ $rc -eq 0 ]] || log "  cuda-checkpoint --action $action --pid $p rc=$rc: $out"
    done
    log "  $action done on all pids"
  done
  CID_R="$CID"
else

log "(c) runsc checkpoint (job over launcher + all ranks)"
t0=$SECONDS
runsc checkpoint -image-path "$WORK/img" \
  -cuda-checkpoint-path "$CUDA_CHECKPOINT" -cuda-checkpoint-sequential "$CID"
CKPT_RC=$?
log "checkpoint rc=$CKPT_RC ($((SECONDS-t0))s)"
[[ $CKPT_RC -eq 0 ]] || { log "FAIL: checkpoint"; grep -h 'blocker' "$WORK"/logs/*boot* | tail -5; exit 1; }
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

# `runsc restore -detach` returns as soon as tasks are runnable, but
# postRestoreCuda's cuda-checkpoint --toggle keeps rebuilding GPU state for
# several more seconds -- and the app's non-CUDA threads (including the shim's
# control thread) run concurrently with it. Resuming the shim during that
# window rebuilds multicast on a context whose device state is not restored
# yet, which faults it (sticky 719) on exactly the one rank that happened not
# to be frozen by the toggle. Wait for the toggle to finish on every rank
# before signalling resume.
log "waiting for postRestoreCuda cuda-checkpoint --toggle to finish on all $WORLD ranks"
for ((i=0; i<180; i++)); do
  n=$(grep -h 'cuda-checkpoint --toggle for PID .* succeeded' "$WORK"/logs/*boot* 2>/dev/null | wc -l)
  [[ "${n:-0}" -ge "$WORLD" ]] && { log "toggle complete on $n/$WORLD ranks"; break; }
  sleep 1
done
[[ "${n:-0}" -ge "$WORLD" ]] || { log "FAIL: cuda-checkpoint toggle did not complete (${n:-0}/$WORLD)"; exit 1; }
fi

log "SHIM resume: rebuild multicast + re-import P2P buffers (remove marker)"
rm_m "$CID_R" suspend
wait_acks "$CID_R" resumed 120 || { log "FAIL: shim resume"; FAIL=1; }
# Snapshot the shim log now, while the container is certainly alive.
runsc exec "$CID_R" /bin/sh -c "cat $MCDIR/mcshim.log" > "$WORK/mcshim.log" 2>/dev/null || true

log "unpause + post-restore verification (eager allreduce + CUDA graph)"
rm_m "$CID_R" pause
wait_all "$CID_R" "post-restore pass" 45 || { log "FAIL: post-restore verify"; FAIL=1; }
sleep 4
runsc exec "$CID_R" /bin/sh -c "cat $MCDIR/nccl.*.log" > "$WORK/nccl.log" 2>/dev/null || true
for r in $(seq 0 $((WORLD-1))); do
  for tag in phase0_preinit phase1_comminit phase2_warmup pre post; do
    runsc exec "$CID_R" /bin/sh -c "cat $MCDIR/maps.$tag.$r" > "$WORK/maps.$tag.$r" 2>/dev/null || true
  done
done
log "maps snapshots saved to $WORK/maps.{pre,post}.<rank>"

echo ""
log "==== per-rank final status ===="
FINAL=$(all_status "$CID_R")
grep -oE '\[rank[0-9]+\] iter=[0-9]+ [a-z-]+ pass failures=[0-9]+' <<<"$FINAL" | sort -u
for ((r=0; r<WORLD; r++)); do grep -q "\[rank$r\].*failures=0" <<<"$FINAL" || FAIL=1; done
log "==== SHIM suspend/resume summary ===="
grep -hE 'SUSPEND done|RESUME done' "$WORK/mcshim.log" 2>/dev/null | tail -10
log "(full shim log saved to $WORK/mcshim.log)"

runsc kill "$CID_R" KILL >/dev/null 2>&1 || true
echo ""
if [[ $FAIL -eq 0 ]]; then
  log "==== RESULT: PASS ===="
  log "STOCK NCCL NVLS ($WORLD-way TP) checkpointed/restored under gVisor with"
  log "NO fork and NO engine hooks: the shim suspended multicast + P2P imports"
  log "and rebuilt them using nvproxy's fdinfo identity oracle."
else
  log "==== RESULT: FAIL ===="
fi
exit $FAIL
