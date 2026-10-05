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
#   ipc                          under runsc (-r), over the host's libraries:
#                                needs nvproxy's exported-object identity
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
  set -- abi gate mc refcount refuse mapwait ipc torch-kernel torch-symm
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
gcc -O2 -g -Wall -Wextra -o "$W/mcshim_test" mcshim_test.c -ldl -lpthread
cp torch_gate_test.py "$W/"

native() {
  CUDA_VISIBLE_DEVICES=$GPUS MCSHIM_LOG="$W/$1.log" LD_PRELOAD="$W/mcshim.so" \
    "$W/mcshim_test" "$1"
}

# sandboxed NAME ROOT SHIM ARGS...: runs ARGS under runsc with SHIM preloaded.
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
    python3 - "$@" > "$W/config.json" <<'EOF'
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
env += ["HOME=/root", "LD_PRELOAD=/mnt/" + e["SHIM"],
        "MCSHIM_LOG=/mnt/%s.log" % e["NAME"],
        "NVIDIA_VISIBLE_DEVICES=" + ",".join(gpus)]
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
  (cd "$W" && sudo "$RUNSC" --root "$W/root" --nvproxy \
    --nvproxy-allowed-driver-capabilities=all --network=none --ignore-cgroups \
    ${RUNSC_FLAGS:-} run --bundle "$W" "mcshim-test-$$-$name") \
    > "$W/$name.out" 2>&1 || rc=$?
  sudo cat "$W/$name.out"
  return $rc
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
    ipc) sandboxed "$t" "" mcshim.so /mnt/mcshim_test ipc || rc=$? ;;
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
