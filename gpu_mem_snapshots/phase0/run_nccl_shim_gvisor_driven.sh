#!/usr/bin/env bash
# run_nccl_shim_gvisor_driven.sh -- ACCEPTANCE for the gVisor-DRIVEN multicast
# interposer path.
#
# Difference from run_nccl_mcshim_gvisor.sh: nothing outside gVisor touches the
# interposer. The harness never creates or removes the suspend marker and never
# waits for acks. Instead:
#
#   * `--cuda-multicast-shim-path` makes the sentry LD_PRELOAD mcshim.so into
#     the container and export MCSHIM_DIR (runsc/boot/loader.go).
#   * `checkpointCudaProcs` suspends the interposer after cuda-checkpoint has
#     locked every rank and before it checkpoints any of them.
#   * `postResumeCuda` resumes it strictly after the restore toggle has
#     finished on every rank, while tasks are still frozen.
#
# That ordering is the whole point: `runsc restore` makes tasks runnable while
# the toggle is still rebuilding GPU state, so any external resume signal races
# it and permanently faults whichever rank is not frozen at that instant
# (sticky 719, surfacing later as a one-rank collective failure).
#
# The workload is STOCK (unpatched) NCCL doing an NVLS allreduce inside a
# captured CUDA graph, one process per GPU -- no NCCL fork, no engine hooks.
#
# Usage:  sudo [WORLD=4] [NCCL_NVLS_ENABLE=1] [GRAPH=1] bash run_nccl_shim_gvisor_driven.sh
set -uo pipefail
cd "$(dirname "$0")"
PHASE0_DIR=$(pwd)

RUNSC="${RUNSC:-/usr/local/bin/runsc-phase0}"
CUDA_CHECKPOINT="${CUDA_CHECKPOINT:-/usr/local/bin/cuda-checkpoint}"
NCCL_STOCK="${NCCL_STOCK:-/opt/phase0/nccl-stock/libnccl.so.2}"
NVLS="${NCCL_NVLS_ENABLE:-1}"
GRAPH="${GRAPH:-1}"
WORLD="${WORLD:-4}"
WORK=/tmp/nccl-shim-driven
STAGE=/opt/phase0
MCDIR=/tmp/mcshim
CID=nccldriven
CID_R=nccldriven-r
log(){ echo "[shim-driven $(date +%H:%M:%S)] $*"; }

[[ -x "$RUNSC" ]] || { log "runsc not found at $RUNSC"; exit 1; }
[[ -f "$NCCL_STOCK" ]] || { log "stock NCCL not found at $NCCL_STOCK"; exit 1; }
[[ -f "$PHASE0_DIR/mcshim/mcshim.so" ]] || { log "mcshim.so not built"; exit 1; }
UVM_MAJOR=$(awk '$2=="nvidia-uvm"{print $1}' /proc/devices)

# The sentry needs both the job wrapper and the interposer path.
JOB_FLAG=(--cuda-checkpoint-path "$CUDA_CHECKPOINT"
          --cuda-multicast-shim-path "$STAGE/mcshim.so"
          --network=none)
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

# NOTE: no LD_PRELOAD and no MCSHIM_DIR here -- the sentry injects both.
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
      "MCSHIM_LOG=$MCDIR/mcshim.log",
      $([ "${SPEC_PRELOAD:-0}" = 1 ] && echo "\"LD_PRELOAD=$STAGE/mcshim.so\", \"MCSHIM_DIR=$MCDIR\",")
      "NCCL_LIB=$NCCL_STOCK",
      "NCCL_NVLS_ENABLE=$NVLS",
      "NCCL_SOCKET_IFNAME=lo",
      "NCCL_DEBUG=WARN",
      "NCCL_DEBUG_FILE=$MCDIR/nccl.%h.%p.log"
    ],
    "cwd": "/"
  },
  "root": {"path": "/", "readonly": true},
  "hostname": "nccldriven",
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
wait_all(){
  local cid="$1" pattern="$2" timeout="$3" i r s
  for ((i=0; i<timeout; i++)); do
    s=$(all_status "$cid")
    local ok=1
    for ((r=0; r<WORLD; r++)); do
      grep -q "\[rank$r\].*$pattern" <<<"$s" || { ok=0; break; }
    done
    [[ $ok -eq 1 ]] && { log "all ranks: $pattern"; return 0; }
    sleep 1
  done
  log "TIMEOUT waiting all ranks '$pattern'; last:"; all_status "$cid" | tail -$WORLD
  return 1
}
touch_m(){ runsc exec "$1" /bin/touch "$MCDIR/$2"; }
rm_m(){ runsc exec "$1" /bin/rm -f "$MCDIR/$2"; }
wait_acks(){  # $1 cid, $2 prefix, $3 timeout
  local cid="$1" prefix="$2" timeout="$3" i n
  for ((i=0; i<timeout; i++)); do
    n=$(runsc exec "$cid" /bin/sh -c "ls $MCDIR/$prefix.* 2>/dev/null | wc -l")
    [[ "${n:-0}" -ge "$WORLD" ]] && { log "$WORLD $prefix acks"; return 0; }
    sleep 1
  done
  log "TIMEOUT: ${n:-0}/$WORLD $prefix acks"; return 1
}

FAIL=0

log "(a) launching $WORLD-rank STOCK-NCCL NVLS group (sentry preloads the interposer)"
runsc run -detach -bundle "$WORK/bundle" -pid-file "$WORK/pid" "$CID" \
  || { log "runsc run failed"; tail -30 "$WORK"/logs/*boot* 2>/dev/null; exit 1; }
wait_all "$CID" "pre-checkpoint pass" 300 || { tail -40 "$WORK"/logs/*boot*; exit 1; }
grep -h "Preloaded multicast interposer" "$WORK"/logs/*boot* | tail -1

log "(a) pause the workload (application-level quiesce only)"
touch_m "$CID" pause
wait_all "$CID" "PAUSED" 60 || { log "FAIL: pause"; exit 1; }

log "(b) runsc checkpoint -- gVisor suspends the interposer itself"
t0=$SECONDS
runsc checkpoint -image-path "$WORK/img" \
  -cuda-checkpoint-path "$CUDA_CHECKPOINT" -cuda-checkpoint-sequential "$CID"
CKPT_RC=$?
log "checkpoint rc=$CKPT_RC ($((SECONDS-t0))s)"
if [[ $CKPT_RC -ne 0 ]]; then
  log "FAIL: checkpoint"
  grep -hE 'blocker|interposer' "$WORK"/logs/*boot* | tail -8
  log "--- interposer log (how far did suspend get?) ---"
  runsc exec "$CID" /bin/sh -c "tail -25 $MCDIR/mcshim.log" 2>/dev/null || log "(container gone)"
  exit 1
fi
grep -h "Multicast interposer suspended" "$WORK"/logs/*boot* | tail -1
runsc delete -force "$CID" >/dev/null 2>&1 || true

log "(c) runsc restore -- gVisor resumes the interposer after the toggle"
t0=$SECONDS
runsc restore -detach -image-path "$WORK/img" -bundle "$WORK/bundle" \
  -pid-file "$WORK/pid-r" "$CID_R"
REST_RC=$?
log "restore rc=$REST_RC ($((SECONDS-t0))s)"
sleep 2
TOGGLE_ERR=$(grep -h 'Killing the sandbox after post restore' "$WORK"/logs/*boot* 2>/dev/null | tail -1)
[[ -z "$TOGGLE_ERR" && $REST_RC -eq 0 ]] || { log "FAIL: restore: ${TOGGLE_ERR#*restore work failed: }"; exit 1; }
grep -h "Multicast interposer resumed" "$WORK"/logs/*boot* | tail -1

# The workload must not run again until gVisor has finished rebuilding the
# multicast layer. gVisor does that from postResumeCuda, which completes after
# `runsc restore` has already returned, so wait for the interposer's own
# per-rank "resumed" acknowledgements before unpausing. Unpausing early lets
# the ranks touch multicast VAs that are still unmapped, which faults every
# rank's context with CUDA_ERROR_ILLEGAL_ADDRESS (700).
log "(d) waiting for gVisor's interposer rebuild to finish on all $WORLD ranks"
wait_acks "$CID_R" resumed 180 || { log "FAIL: interposer rebuild did not complete"; FAIL=1; }
grep -h "Multicast interposer resumed" "$WORK"/logs/*boot* | tail -1

log "(e) unpause + post-restore verification (eager allreduce + CUDA graph)"
rm_m "$CID_R" pause
wait_all "$CID_R" "post-restore pass" 60 || { log "FAIL: post-restore verify"; FAIL=1; }
sleep 3
runsc exec "$CID_R" /bin/sh -c "cat $MCDIR/mcshim.log" > "$WORK/mcshim.log" 2>/dev/null || true

echo ""
log "==== per-rank final status ===="
FINAL=$(all_status "$CID_R")
grep -oE '\[rank[0-9]+\] iter=[0-9]+ [a-z-]+ pass failures=[0-9]+' <<<"$FINAL" | sort -u
for ((r=0; r<WORLD; r++)); do grep -q "\[rank$r\].*failures=0" <<<"$FINAL" || FAIL=1; done

log "==== interposer context health (must be 0 faults) ===="
NFAULT=$(grep -c "FAULTED" "$WORK/mcshim.log" 2>/dev/null); NFAULT=${NFAULT:-0}
grep -hE 'CTXPROBE' "$WORK/mcshim.log" 2>/dev/null | tail -8
log "context faults: $NFAULT"
[[ "${NFAULT:-0}" -eq 0 ]] || FAIL=1
grep -hE 'SUSPEND done|RESUME done' "$WORK/mcshim.log" 2>/dev/null | tail -8

runsc kill "$CID_R" KILL >/dev/null 2>&1 || true
echo ""
if [[ $FAIL -eq 0 ]]; then
  log "==== RESULT: PASS ===="
  log "STOCK NCCL NVLS ($WORLD-way) checkpointed/restored with gVisor owning the"
  log "entire multicast suspend/resume sequence: no NCCL fork, no engine hooks,"
  log "and nothing outside gVisor driving the transitions."
else
  log "==== RESULT: FAIL ===="
fi
exit $FAIL
