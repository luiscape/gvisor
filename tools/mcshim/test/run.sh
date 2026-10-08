#!/bin/bash
# Copyright 2026 The gVisor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Runs the interposer tests on a host with two or more NVLS-capable GPUs.
#
#   abi gate mc refcount refuse mapwait  native (mcshim_test.c)
#   silent                       native: the preloaded shim prints nothing
#   ipc                          under runsc (-r), over the host's libraries:
#                                needs nvproxy's exported-object identity
#   orphan                       under runsc (-r): `runsc checkpoint` refuses an
#                                import whose exporter freed the object, and
#                                the application keeps running
#   deadline reason optout       under runsc (-r), with a stub cuda-checkpoint:
#                                a hung invocation is killed and the checkpoint
#                                fails, with the application still running; a
#                                refusal's reason reaches the sentry's error;
#                                without the interposer, no job and one
#                                --toggle per process, as before it
#   torch-kernel torch-symm      under runsc (-r) in a rootfs with PyTorch and
#                                the host driver's userspace (-i, -e: its env)
#
# Usage: run.sh [-r RUNSC] [-i ROOTFS] [-e ENVFILE] [test...]   (default: all)
# MCSHIM_TEST_GPUS picks the two GPUs (default 0,1).
set -euo pipefail
cd "$(dirname "$0")"
RUNSC="" ROOTFS="" ENVFILE=""
while getopts r:i:e: o; do
  case $o in
    r) RUNSC=$(realpath "$OPTARG") ;;
    i) ROOTFS=$(realpath "$OPTARG") ;;
    e) ENVFILE=$(realpath "$OPTARG") ;;
    *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))
if [ $# -eq 0 ]; then
  set -- abi gate mc refcount refuse mapwait silent ipc orphan deadline reason \
    optout torch-kernel torch-symm
fi
GPUS=${MCSHIM_TEST_GPUS:-0,1}

W=$(mktemp -d)
chmod 777 "$W"
cleanup() {
  if [ -n "$RUNSC" ] && [ -d "$W/root" ]; then
    for id in $(sudo "$RUNSC" --root "$W/root" list -q 2>/dev/null); do
      sudo "$RUNSC" --root "$W/root" delete -force "$id" || true
    done
    sudo umount "$W/root/null-netns" 2>/dev/null || true
  fi
  sudo rm -rf "$W"
}
trap cleanup EXIT
gcc -O2 -g -Wall -Wextra -fPIC -shared -o "$W/mcshim.so" ../mcshim.c -ldl -lpthread
gcc -O2 -g -Wall -Wextra -rdynamic -o "$W/mcshim_test" mcshim_test.c -ldl \
  -lpthread
gcc -O2 -g -Wall -Wextra -o "$W/ckpt_stub" ckpt_stub.c
cp torch_gate_test.py "$W/"

native() {
  CUDA_VISIBLE_DEVICES=$GPUS MCSHIM_LOG="$W/$1.log" LD_PRELOAD="$W/mcshim.so" \
    "$W/mcshim_test" "$1"
}

# sandboxed NAME ROOT SHIM ARGS...: runs ARGS under runsc with SHIM preloaded,
# in container mcshim-test-$$-NAME (detached if DETACH=1).
# ROOT "" means the host's libraries and /etc only: with the host's
# nvidia-modprobe on its path, libcuda tries to load the module and finds no
# device.
sandboxed() {
  local name=$1 root=$2 shim=$3
  shift 3
  if [ -z "$RUNSC" ]; then
    echo "SKIP $name: needs -r RUNSC"
    return 0
  fi
  if [ -z "$root" ]; then
    root=$W/rootfs
    mkdir -p "$root"/{usr/lib,usr/lib64,lib,lib64,etc,proc,dev,sys,tmp,mnt}
  fi
  NAME=$name ROOT=$root SHIM=$shim W=$W GPUS=$GPUS ENVFILE=$ENVFILE \
    NOLOG=${NOLOG:-} python3 - "$@" > "$W/config.json" <<'EOF'
import json, os, sys
e = os.environ
gpus = e["GPUS"].split(",")
def dev(path):
    st = os.stat(path)
    return {"path": path, "type": "c", "major": os.major(st.st_rdev),
            "minor": os.minor(st.st_rdev), "fileMode": 0o666, "uid": 0,
            "gid": 0}
env = ["PATH=/usr/local/bin:/usr/bin:/bin"]
if e["ENVFILE"]:
    env = [l for l in open(e["ENVFILE"]).read().splitlines() if "=" in l]
env += ["HOME=/root", "NVIDIA_VISIBLE_DEVICES=" + ",".join(gpus)]
if e["SHIM"]:
    env += ["LD_PRELOAD=/mnt/" + e["SHIM"]]
if not e.get("NOLOG"):
    env += ["MCSHIM_LOG=/mnt/%s.log" % e["NAME"]]
mounts = []
if e["ROOT"] == e["W"] + "/rootfs":
    mounts = [{"destination": d, "type": "bind", "source": d,
               "options": ["rbind", "ro"]}
              for d in ["/usr/lib", "/usr/lib64", "/lib", "/lib64", "/etc"]]
mounts += [
    {"destination": "/proc", "type": "proc", "source": "proc"},
    {"destination": "/dev", "type": "tmpfs", "source": "tmpfs",
     "options": ["nosuid", "mode=755"]},
    {"destination": "/dev/shm", "type": "tmpfs", "source": "shm",
     "options": ["nosuid", "nodev", "mode=1777", "size=4294967296"]},
    {"destination": "/sys", "type": "sysfs", "source": "sysfs",
     "options": ["ro"]},
    {"destination": "/tmp", "type": "tmpfs", "source": "tmpfs"},
    {"destination": "/mnt", "type": "bind", "source": e["W"],
     "options": ["rbind", "rw"]},
]
print(json.dumps({
    "ociVersion": "1.0.0",
    "process": {"user": {"uid": 0, "gid": 0}, "args": sys.argv[1:],
                "env": env, "cwd": "/"},
    "root": {"path": e["ROOT"], "readonly": True},
    "mounts": mounts,
    "linux": {
        "namespaces": [{"type": t} for t in
                       ("pid", "mount", "ipc", "uts", "network")],
        "devices": [dev("/dev/nvidia" + g) for g in gpus] +
                   [dev(p) for p in ["/dev/nvidiactl", "/dev/nvidia-uvm",
                                     "/dev/nvidia-uvm-tools"]],
    },
}))
EOF
  # runsc takes over the stdio it inherits, so give it a file of its own.
  local rc=0
  (cd "$W" && runsc run ${DETACH:+-detach} --bundle "$W" "mcshim-test-$$-$name") \
    > "$W/$name.out" 2>&1 || rc=$?
  sudo cat "$W/$name.out"
  return $rc
}

runsc() {
  sudo "$RUNSC" --root "$W/root" --nvproxy \
    --nvproxy-allowed-driver-capabilities=all --network=none --ignore-cgroups \
    ${RUNSC_FLAGS:-} "$@"
}

# silent: every process in a container loads the shim, so it must not print
# unless MCSHIM_LOG=stderr; by default it logs to /tmp/mcshim/mcshim.log.
silent() {
  local out log=/tmp/mcshim/mcshim.log
  out=$(env -u MCSHIM_LOG LD_PRELOAD="$W/mcshim.so" sh -c true 2>&1)
  if [ -n "$out" ]; then
    echo "sh -c true printed: $out"
    return 1
  fi
  rm -f "$log"
  out=$(env -u MCSHIM_LOG CUDA_VISIBLE_DEVICES="$GPUS" \
    LD_PRELOAD="$W/mcshim.so" "$W/mcshim_test" abi 2>&1)
  if echo "$out" | grep -q '\[mcshim'; then
    echo "a CUDA process logged to stderr: $out"
    return 1
  fi
  if ! grep -q "control thread started" "$log"; then
    echo "nothing logged to $log"
    return 1
  fi
  out=$(MCSHIM_LOG=stderr CUDA_VISIBLE_DEVICES="$GPUS" \
    LD_PRELOAD="$W/mcshim.so" "$W/mcshim_test" abi 2>&1)
  if ! echo "$out" | grep -q '\[mcshim'; then
    echo "MCSHIM_LOG=stderr logged nothing to stderr"
    return 1
  fi
}

# count FILE: the number in FILE, or 0.
count() {
  local n
  n=$(cat "$1" 2>/dev/null) || true
  echo "${n:-0}"
}

# stub_start NAME ARGS...: runs `mcshim_test ARGS...` detached under runsc, with
# RUNSC_FLAGS, and waits for its heartbeat. The stub cuda-checkpoint logs its
# invocations to $W/ckpt_stub.log.
stub_start() {
  local name=$1
  shift
  rm -f "$W/beat" "$W/ckpt_stub.log" "$W/ckpt_stub.hang" "$W/ckpt_stub.fail"
  DETACH=1 sandboxed "$name" "" "" /mnt/mcshim_test "$@" || return 1
  for _ in $(seq 600); do
    [ "$(count "$W/beat")" -gt 0 ] && return 0
    sleep 0.1
  done
  echo "$name: no heartbeat"
  return 1
}

# stub_ckpt NAME: `runsc checkpoint` of NAME's container; its output is in
# $W/NAME.ckpt.
stub_ckpt() {
  local rc=0
  runsc checkpoint --image-path "$W/ckpt-$1" "mcshim-test-$$-$1" \
    > "$W/$1.ckpt" 2>&1 || rc=$?
  sudo cat "$W/$1.ckpt"
  return $rc
}

# beating: the application is still launching kernels.
beating() {
  local n
  n=$(count "$W/beat")
  sleep 2
  [ "$(count "$W/beat")" -gt "$n" ]
}

# hung NAME HANG: a checkpoint whose cuda-checkpoint invocation matching HANG
# never returns must fail within the deadline, kill it, and leave the
# application running.
hung() {
  local name=$1 start rc=0
  stub_start "$name" beat || return 1
  echo "$2" > "$W/ckpt_stub.hang"
  start=$(date +%s)
  stub_ckpt "$name" || rc=$?
  echo "$name: checkpoint rc=$rc after $(($(date +%s) - start))s; cuda-checkpoint calls:"
  sed 's/^/  /' "$W/ckpt_stub.log"
  if [ $rc -eq 0 ] || ! grep -q "did not finish in time and was killed" "$W/$name.ckpt"; then
    echo "$name: the checkpoint did not fail on the deadline"
    return 1
  fi
  if [ $(($(date +%s) - start)) -gt 60 ]; then
    echo "$name: the checkpoint took too long"
    return 1
  fi
  if ! beating; then
    echo "$name: the application stopped"
    return 1
  fi
  runsc kill "mcshim-test-$$-$name" KILL || true
}

# deadline: with the interposer, a lock and a checkpoint that hang are killed.
deadline() {
  local flags="${RUNSC_FLAGS:-} --cuda-checkpoint-path=/mnt/ckpt_stub --cuda-checkpoint-timeout=5s --cuda-multicast-shim-path=/mnt/mcshim.so"
  RUNSC_FLAGS="$flags" hung deadline-lock "--action lock" || return 1
  RUNSC_FLAGS="$flags" hung deadline-ckpt "--action checkpoint" || return 1
  if ! grep -q -- "--launch-job /mnt/mcshim_test beat" "$W/ckpt_stub.log"; then
    echo "deadline: the container did not run in a cuda-checkpoint job"
    return 1
  fi
}

# optout: with the runtime --cuda-checkpoint-path but no interposer, the
# container is not wrapped in a job and is checkpointed with one --toggle per
# process, exactly as before the interposer. A failed toggle fails the
# checkpoint and leaves the application running.
optout() {
  local rc=0 want
  export RUNSC_FLAGS="${RUNSC_FLAGS:-} --cuda-checkpoint-path=/mnt/ckpt_stub"
  stub_start optout beat || return 1
  echo "--toggle" > "$W/ckpt_stub.fail"
  stub_ckpt optout || rc=$?
  want=$(printf -- '--get-state --pid 1\n--toggle --pid 1')
  echo "optout: checkpoint rc=$rc; cuda-checkpoint calls:"
  sed 's/^/  /' "$W/ckpt_stub.log"
  if [ $rc -eq 0 ] || [ "$(cat "$W/ckpt_stub.log")" != "$want" ]; then
    echo "optout: want only: $want"
    return 1
  fi
  if ! beating; then
    echo "optout: the application stopped"
    return 1
  fi
  runsc kill "mcshim-test-$$-optout" KILL || true
}

# reason: a shim refusal names its cause in the sentry's error, including the
# shim's default log.
reason() {
  local rc=0
  export RUNSC_FLAGS="${RUNSC_FLAGS:-} --cuda-checkpoint-path=/mnt/ckpt_stub --cuda-multicast-shim-path=/mnt/mcshim.so"
  NOLOG=1 stub_start reason beat managed || return 1
  stub_ckpt reason || rc=$?
  if [ $rc -eq 0 ] || ! grep -q '"managed memory"; last log lines: .*GATE: refusing: managed memory' "$W/reason.ckpt"; then
    echo "reason: the refusal's reason is not in the error"
    return 1
  fi
  if ! beating; then
    echo "reason: the application stopped"
    return 1
  fi
  runsc kill "mcshim-test-$$-reason" KILL || true
}

orphan() {
  local id=mcshim-test-$$-orphan
  cp "$(command -v cuda-checkpoint)" "$W/" || return 77
  export RUNSC_FLAGS="${RUNSC_FLAGS:-} --cuda-checkpoint-path=/mnt/cuda-checkpoint --cuda-multicast-shim-path=/mnt/mcshim.so"
  DETACH=1 sandboxed orphan "" mcshim.so /mnt/mcshim_test orphan || return 1
  for _ in $(seq 600); do
    [ "$(count "$W/orphan.0")" -gt 0 ] && [ "$(count "$W/orphan.1")" -gt 0 ] &&
      break
    sleep 0.1
  done
  local rc=0
  runsc checkpoint --image-path "$W/ckpt" "$id" > "$W/orphan.ckpt" 2>&1 || rc=$?
  sudo cat "$W/orphan.ckpt"
  if [ $rc -eq 0 ] || ! grep -q "could not be rebuilt" "$W/orphan.ckpt"; then
    echo "checkpoint not refused for the orphaned import"
    return 1
  fi
  local n0 n1
  n0=$(count "$W/orphan.0") n1=$(count "$W/orphan.1")
  sleep 2
  if [ "$(count "$W/orphan.0")" -le "$n0" ] ||
    [ "$(count "$W/orphan.1")" -le "$n1" ]; then
    echo "the application stopped after the refused checkpoint"
    return 1
  fi
  runsc kill "$id" KILL || true
}

# The image's glibc may be older than the host's: build a portable copy.
portable_shim() {
  [ -e "$W/mcshim-portable.so" ] || ../build.sh "$W/mcshim-portable.so" >&2
  echo mcshim-portable.so
}

failed=0
for t in "$@"; do
  echo "=== $t"
  rc=0
  case $t in
    silent) silent || rc=$? ;;
    deadline | reason | optout)
      if [ -z "$RUNSC" ]; then
        echo "SKIP $t: needs -r RUNSC"
        continue
      fi
      ("$t") || rc=$?
      ;;
    ipc) sandboxed "$t" "" mcshim.so /mnt/mcshim_test ipc || rc=$? ;;
    orphan)
      if [ -z "$RUNSC" ]; then
        echo "SKIP $t: needs -r RUNSC"
        continue
      fi
      (orphan) || rc=$?
      ;;
    torch-*)
      if [ -z "$ROOTFS" ]; then
        echo "SKIP $t: needs -i ROOTFS"
        continue
      fi
      sandboxed "$t" "$ROOTFS" "$(portable_shim)" python3 \
        /mnt/torch_gate_test.py "${t#torch-}" || rc=$?
      ;;
    *) native "$t" || rc=$? ;;
  esac
  case $rc in
    0) echo "PASS $t" ;;
    77) echo "SKIP $t" ;;
    *)
      echo "FAIL $t (exit $rc); interposer log:"
      sudo tail -n 20 "$W/$t.log" 2>/dev/null || true
      failed=1
      ;;
  esac
done
exit $failed
