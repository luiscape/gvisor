#!/usr/bin/env bash
# run_torch_nccl_gvisor.sh — PyTorch tier end-to-end under gVisor.
#
# torch.distributed NCCL with NVLS multicast and a captured CUDA graph,
# WORLD ranks (one process per GPU), through a real `runsc checkpoint` /
# `runsc restore`. The application never calls the NCCL suspend API: NCCL's
# own checkpoint control thread (NCCL_CKPT_CTRL_DIR) does it, so this is what
# a stock PyTorch stack would experience.
#
#   (a) launch WORLD ranks under runsc, job-wrapped by --cuda-checkpoint-path
#   (b) pause the app, then set the `suspend` marker; every rank's NCCL
#       releases its NVLS multicast layer and acks
#   (c) runsc checkpoint   (multicast is gone, so it does not hang)
#   (d) runsc restore, then remove the marker; every rank's NCCL rebuilds the
#       multicast layer at identical VAs and acks
#   (e) unpause; the captured CUDA graph must still replay correct results
#
# The rootfs is the benchmark image's (it carries PyTorch); the patched
# libnccl is bind-mounted over it via LD_PRELOAD.
#
# Usage:
#   sudo nvidia-smi -pm 1
#   sudo [WORLD=4] [RUNSC=/usr/local/bin/runsc-phase0] \
#        [NCCL_LIB=/opt/phase0/nccl-patched/libnccl.so.2] \
#        [NO_GRAPH=1] [SYMM_MEM=1] bash run_torch_nccl_gvisor.sh
set -uo pipefail
cd "$(dirname "$0")"
PHASE0_DIR=$(pwd)

# MECH selects what releases the multicast layer:
#   nccl   (default) patched libnccl + NCCL's control thread. Covers multicast
#                    NCCL owns; a torch symmetric-memory team is invisible to
#                    it (SYMM_MEM=1 then fails, by design).
#   mcshim           the libcuda-level interposer, driven by gVisor. Sits below
#                    every multicast owner, so it should cover BOTH NCCL's NVLS
#                    and torch symmetric memory.
MECH="${MECH:-nccl}"
RUNSC="${RUNSC:-/usr/local/bin/runsc-phase0}"
CUDA_CHECKPOINT="${CUDA_CHECKPOINT:-/usr/local/bin/cuda-checkpoint}"
NCCL_LIB="${NCCL_LIB:-/opt/phase0/nccl-patched/libnccl.so.2}"
MCSHIM_SO="${MCSHIM_SO:-$PHASE0_DIR/mcshim/mcshim.so}"
MCSHIM_IN_CTR=/opt/phase0/torchtier/mcshim.so
ROOTFS="${ROOTFS:-/data/cr-bench/rootfs-cr-bench-vllm}"
WORLD="${WORLD:-4}"
WORK=/tmp/torch-nccl-gvisor
STAGE=/opt/phase0/torchtier
CID=torchnccl
CID_R=torchnccl-r
CTRL_DIR=/tmp/torchnccl
log(){ echo "[torch-gvisor $(date +%H:%M:%S)] $*"; }

[[ -x "$RUNSC" ]] || { log "runsc not found at $RUNSC"; exit 1; }
[[ -f "$NCCL_LIB" ]] || { log "patched NCCL not found at $NCCL_LIB"; exit 1; }
[[ -d "$ROOTFS" ]] || { log "rootfs not found at $ROOTFS (extract the benchmark image first)"; exit 1; }
UVM_MAJOR=$(awk '$2=="nvidia-uvm"{print $1}' /proc/devices)

SHIM_FLAG=()
if [[ "$MECH" == "mcshim" ]]; then
  [[ -f "$MCSHIM_SO" ]] || { log "mcshim not built at $MCSHIM_SO (run mcshim/build.sh)"; exit 1; }
  SHIM_FLAG=(--cuda-multicast-shim-path "$MCSHIM_IN_CTR")
fi
runsc(){ "$RUNSC" --root "$WORK/root" --debug --debug-log="$WORK/logs/" \
         --cuda-checkpoint-path "$CUDA_CHECKPOINT" --network=none "${SHIM_FLAG[@]}" "$@"; }
cleanup(){ runsc delete -force "$CID" >/dev/null 2>&1 || true
           runsc delete -force "$CID_R" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup
rm -rf "$WORK" 2>/dev/null; mkdir -p "$WORK"/{root,logs,img,bundle}

# Stage the workload + the patched NCCL where the container can read them.
mkdir -p "$STAGE"
cp "$PHASE0_DIR"/{torch_nccl_launcher.py,torch_nccl_ckpt.py} "$STAGE/"
mkdir -p "$STAGE/nccl" && cp "$NCCL_LIB" "$STAGE/nccl/libnccl.so.2"
[[ "$MECH" == "mcshim" ]] && cp "$MCSHIM_SO" "$STAGE/mcshim.so"
for m in nvidia nvidia_uvm; do
  mkdir -p "$STAGE/sys_module_$m"; echo live > "$STAGE/sys_module_$m/initstate"
done
chmod -R a+rX "$STAGE"

# The NCCL control thread is enabled only for MECH=nccl. Under MECH=mcshim the
# interposer is the sole mechanism, so NCCL must not also be tearing its own
# multicast down underneath it.
if [[ "$MECH" == "nccl" ]]; then
  CTRL_ENV_LINE="      \"NCCL_CKPT_CTRL_DIR=$CTRL_DIR\","
else
  CTRL_ENV_LINE=""
fi

WORKLOAD_ARGS='"--dir", "'"$CTRL_DIR"'"'
[[ "${NO_GRAPH:-0}" = "1" ]] && WORKLOAD_ARGS+=', "--no-graph"'
[[ "${SYMM_MEM:-0}" = "1" ]] && WORKLOAD_ARGS+=', "--symm-mem"'

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
      "/usr/bin/python3", "$STAGE/torch_nccl_launcher.py",
      "--world", "$WORLD", $WORKLOAD_ARGS
    ],
    "env": [
      "PATH=/usr/local/bin:/usr/bin:/bin",
      "LD_PRELOAD=$STAGE/nccl/libnccl.so.2",
$CTRL_ENV_LINE
      "NCCL_NVLS_ENABLE=1",
      "NCCL_SOCKET_IFNAME=lo",
      "NCCL_DEBUG=WARN",
      "HOME=/root"
    ],
    "cwd": "/"
  },
  "root": {"path": "$ROOTFS", "readonly": false},
  "hostname": "torchnccl",
  "mounts": [
    {"destination": "/proc", "type": "proc"},
    {"destination": "/tmp", "type": "tmpfs"},
    {"destination": "/dev/shm", "type": "tmpfs", "options": ["size=8g"]},
    {"destination": "$STAGE", "type": "bind", "source": "$STAGE", "options": ["rbind", "ro"]},
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

status_all(){ runsc exec "$1" /bin/sh -c "cat $CTRL_DIR/status.* 2>/dev/null"; }
count_matching(){ runsc exec "$1" /bin/sh -c "grep -l '$2' $CTRL_DIR/status.* 2>/dev/null | wc -l" 2>/dev/null | tr -d '[:space:]'; }
count_files(){ runsc exec "$1" /bin/sh -c "ls $CTRL_DIR/$2 2>/dev/null | wc -l" 2>/dev/null | tr -d '[:space:]'; }
wait_count(){ # cid, kind(files|status), pattern, timeout
  local cid="$1" kind="$2" pat="$3" tmo="$4" i n
  for ((i=0; i<tmo; i++)); do
    if [[ "$kind" == files ]]; then n=$(count_files "$cid" "$pat"); else n=$(count_matching "$cid" "$pat"); fi
    [[ "${n:-0}" -ge "$WORLD" ]] && { log "  $pat: $n/$WORLD"; return 0; }
    sleep 1
  done
  log "  TIMEOUT on '$pat' (got ${n:-0}/$WORLD)"; return 1
}

FAIL=0

log "(a) launching $WORLD torch ranks under runsc (NVLS + CUDA graph)"
runsc run -detach -bundle "$WORK/bundle" -pid-file "$WORK/pid" "$CID" \
  || { log "runsc run failed"; tail -40 "$WORK"/logs/*boot* 2>/dev/null; exit 1; }
wait_count "$CID" status "iter=" 900 || { log "ranks never started"; status_all "$CID"; tail -40 "$WORK"/logs/*boot*; exit 1; }
log "  ranks verifying: $(status_all "$CID" | head -1)"

# The app pauses in both mechanisms: ncclCommSuspend cannot run under a live
# collective, and a captured CUDA graph replay bypasses NCCL's own gate.
log "(b) pause the app"
runsc exec "$CID" /bin/touch "$CTRL_DIR/pause"
wait_count "$CID" status PAUSED 120 || FAIL=1

if [[ "$MECH" == "nccl" ]]; then
  log "    ask NCCL to release its NVLS multicast layer"
  runsc exec "$CID" /bin/touch "$CTRL_DIR/suspend"
  wait_count "$CID" files 'suspended.*' 300 || { log "suspend not acknowledged"; FAIL=1; }
else
  log "    (mcshim: gVisor drives the teardown inside the checkpoint)"
fi

log "(c) runsc checkpoint"
t0=$SECONDS
# -cuda-checkpoint-path must be given to the subcommand as well: the global
# flag only tells the loader to job-wrap the container command, while this is
# what makes preSaveCuda actually run the cuda-checkpoint phases.
runsc checkpoint -image-path "$WORK/img" \
  -cuda-checkpoint-path "$CUDA_CHECKPOINT" -cuda-checkpoint-sequential "$CID"
CK=$?
log "  checkpoint rc=$CK ($((SECONDS-t0))s)"

# With a torch symmetric-memory tensor there is a multicast owner NCCL does
# not know about, so suspending NCCL is not enough. The correct outcome is a
# refused checkpoint naming the rank, not a hang and not a broken snapshot.
if [[ "${SYMM_MEM:-0}" = "1" && "$MECH" == "nccl" ]]; then
  BLOCKED=$(grep -h 'cannot proceed.*multicast' "$WORK"/logs/*boot* 2>/dev/null | tail -1)
  if [[ $CK -ne 0 && -n "$BLOCKED" ]]; then
    log "EXPECTED: NCCL released its own NVLS, but torch symmetric memory is a"
    log "  separate multicast owner, so the blocker gate refused the checkpoint:"
    log "  ${BLOCKED#*] }"
    log "  => covering this workload needs mcshim for the non-NCCL owner."
    log "==== RESULT: PASS (expected refusal) ===="
    exit 0
  fi
  log "FAIL: expected a multicast blocker refusal, got rc=$CK"
  exit 1
fi

[[ $CK -eq 0 ]] || { log "FAIL: checkpoint"; grep -h 'multicast\|blocker\|cuda-checkpoint' "$WORK"/logs/*boot* | tail -15; exit 1; }
runsc delete -force "$CID" >/dev/null 2>&1 || true

log "(d) runsc restore, then ask NCCL to rebuild"
t0=$SECONDS
runsc restore -detach -image-path "$WORK/img" -bundle "$WORK/bundle" -pid-file "$WORK/pid-r" "$CID_R"
RC=$?
log "  restore rc=$RC ($((SECONDS-t0))s)"
[[ $RC -eq 0 ]] || { log "FAIL: restore"; tail -30 "$WORK"/logs/*restore* 2>/dev/null; exit 1; }
runsc exec "$CID_R" /bin/touch "$CTRL_DIR/restored" 2>/dev/null || true
if [[ "$MECH" == "nccl" ]]; then
  runsc exec "$CID_R" /bin/rm -f "$CTRL_DIR/suspend"
  wait_count "$CID_R" files 'resumed.*' 300 || { log "resume not acknowledged"; FAIL=1; }
else
  log "    (mcshim: gVisor rebuilt after the restore toggle)"
fi

log "(e) unpause; the captured graph must keep producing correct results"
runsc exec "$CID_R" /bin/rm -f "$CTRL_DIR/pause"
sleep 10
log "final per-rank status:"
S=$(status_all "$CID_R")
echo "$S" | sed 's/^/    /'
for ((r=0; r<WORLD; r++)); do :; done
grep -q 'FAIL' <<<"$S" && FAIL=1
grep -q 'post-restore pass' <<<"$S" || { log "no rank reported a post-restore pass"; FAIL=1; }
awk '/failures=[1-9]/{bad=1} END{exit bad?1:0}' <<<"$S" || { log "a rank reported failures"; FAIL=1; }

runsc kill "$CID_R" KILL >/dev/null 2>&1 || true
echo ""
log "==== RESULT: $([[ $FAIL -eq 0 ]] && echo PASS || echo FAIL) ===="
exit $FAIL
