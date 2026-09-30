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
# "ipc" needs nvproxy's exported-object identity, so it runs under runsc over
# the host's libraries; the others run natively.
#
# Usage: run.sh [-r /path/to/runsc] [test...]   (default: all)
set -euo pipefail
cd "$(dirname "$0")"
RUNSC=""
while getopts r: o; do
  case $o in
    r) RUNSC=$(realpath "$OPTARG") ;;
    *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))
if [ $# -eq 0 ]; then set -- abi gate mc refcount refuse ipc; fi

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

native() {
  MCSHIM_LOG="$W/$1.log" LD_PRELOAD="$W/mcshim.so" "$W/mcshim_test" "$1"
}

# The sandbox sees only the host's libraries and /etc: with the host's
# nvidia-modprobe on its path, libcuda tries to load the module and finds no
# device.
sandboxed() {
  if [ -z "$RUNSC" ]; then
    echo "SKIP $1: needs -r RUNSC"
    return 0
  fi
  mkdir -p "$W"/rootfs/{usr/lib,usr/lib64,lib,lib64,etc,proc,dev,sys,tmp,mnt}
  python3 - "$1" "$W" > "$W/config.json" <<'EOF'
import json, os, sys
test, w = sys.argv[1], sys.argv[2]
def dev(path):
    st = os.stat(path)
    return {"path": path, "type": "c", "major": os.major(st.st_rdev),
            "minor": os.minor(st.st_rdev), "fileMode": 0o666, "uid": 0,
            "gid": 0}
libs = ["/usr/lib", "/usr/lib64", "/lib", "/lib64", "/etc"]
print(json.dumps({
    "ociVersion": "1.0.0",
    "process": {
        "user": {"uid": 0, "gid": 0},
        "args": ["/mnt/mcshim_test", test],
        "env": ["LD_PRELOAD=/mnt/mcshim.so", "MCSHIM_LOG=/mnt/%s.log" % test,
                "NVIDIA_VISIBLE_DEVICES=0,1"],
        "cwd": "/",
    },
    "root": {"path": w + "/rootfs", "readonly": True},
    "mounts": [{"destination": d, "type": "bind", "source": d,
                "options": ["rbind", "ro"]} for d in libs] + [
        {"destination": "/proc", "type": "proc", "source": "proc"},
        {"destination": "/dev", "type": "tmpfs", "source": "tmpfs",
         "options": ["nosuid", "mode=755"]},
        {"destination": "/sys", "type": "sysfs", "source": "sysfs",
         "options": ["ro"]},
        {"destination": "/tmp", "type": "tmpfs", "source": "tmpfs"},
        {"destination": "/mnt", "type": "bind", "source": w,
         "options": ["rbind", "rw"]},
    ],
    "linux": {
        "namespaces": [{"type": t} for t in
                       ("pid", "mount", "ipc", "uts", "network")],
        "devices": [dev(p) for p in ["/dev/nvidia0", "/dev/nvidia1",
                                     "/dev/nvidiactl", "/dev/nvidia-uvm",
                                     "/dev/nvidia-uvm-tools"]],
    },
}))
EOF
  (cd "$W" && sudo "$RUNSC" --root "$W/root" --nvproxy \
    --nvproxy-allowed-driver-capabilities=all --network=none --ignore-cgroups \
    ${RUNSC_FLAGS:-} run --bundle "$W" "mcshim-test-$$-$1")
}

failed=0
for t in "$@"; do
  echo "=== $t"
  rc=0
  case $t in
    ipc) sandboxed "$t" || rc=$? ;;
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
