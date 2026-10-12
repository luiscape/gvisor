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

# Checks that the shim's control directory cannot be used to make a root or
# setuid process to write to, or change the mode of, a file elsewhere. CPU only;
# needs docker. Each case runs in a fresh container, as root: it plants its
# attack, then puts the shim in /etc/ld.so.preload (every process after that,
# including the test's own commands, loads it) with the fake libcuda
# (fakecuda.c) on the library path, and checks that a 0600 victim file keeps
# its mode and contents.
#
# Usage: ctlsec.sh mcshim.so DIR   (DIR has libcuda.so.1 and ctlsec)
set -uo pipefail
if [ $# -ne 2 ]; then
  echo "usage: $0 mcshim.so DIR" >&2
  exit 2
fi
SHIM=$(realpath "$1")
DIR=$(realpath "$2")
IMAGE=${CTLSEC_IMAGE:-ubuntu:24.04}
docker="docker"
$docker info >/dev/null 2>&1 || docker="sudo docker"

# run NAME SETUP CHECK: in a fresh container, runs SETUP without the shim, then
# CHECK with it preloaded; prints PASS or FAIL.
fail=0
run() {
  local name=$1 out rc=0
  out=$($docker run --rm -v "$SHIM:/m.so:ro" -v "$DIR:/t:ro" \
    -e LD_LIBRARY_PATH=/t "$IMAGE" /bin/bash -c '
    set -u
    printf secret >/victim; chmod 0600 /victim
    # intact: the victim still has mode 600 and its contents.
    intact() {
      [ "$(stat -c %a /victim)" = 600 ] && [ "$(cat /victim)" = secret ] ||
        { echo "victim changed: $(stat -c %a /victim) $(head -c 200 /victim)"; exit 1; }
    }
    mkdir -m 1777 /tmp/mcshim
    '"$2"'
    echo /m.so >/etc/ld.so.preload
    '"$3" 2>&1) || rc=$?
  if [ $rc -eq 0 ]; then
    echo "PASS $name"
  else
    echo "FAIL $name: $out"
    fail=1
  fi
}

run "log symlinked to another file" '
  ln -s /victim /tmp/mcshim/mcshim.log' '
  /t/ctlsec; intact'
run "log hard-linked to another file" '
  ln /victim /tmp/mcshim/mcshim.log' '
  /t/ctlsec; intact'
run "marker symlinked to another file" '' '
  bash -c "ln -s /victim /tmp/mcshim/present.\$\$; exec /t/ctlsec"; intact'

run "control directory symlinked elsewhere" '
  rmdir /tmp/mcshim; mkdir /target; ln -s /target /tmp/mcshim' '
  /t/ctlsec; [ -z "$(ls -A /target)" ] || { echo "wrote into /target: $(ls /target)"; exit 1; }'
run "setuid process with MCSHIM_LOG" '
  cp /t/ctlsec /suid; chmod 4755 /suid' '
  setpriv --reuid=65534 --regid=65534 --clear-groups env MCSHIM_LOG=/victim /suid
  intact
  if ls /tmp/mcshim | grep -q .; then echo "the setuid process wrote: $(ls /tmp/mcshim)"; exit 1; fi'
run "shared log and acks across uids" '' '
  /t/ctlsec || { echo "root: no control thread"; exit 1; }
  [ "$(stat -c %a /tmp/mcshim/mcshim.log)" = 666 ] || { echo "log mode $(stat -c %a /tmp/mcshim/mcshim.log)"; exit 1; }
  n=$(wc -l </tmp/mcshim/mcshim.log)
  setpriv --reuid=65534 --regid=65534 --clear-groups /t/ctlsec ||
    { echo "uid 65534: no control thread"; exit 1; }
  [ "$(wc -l </tmp/mcshim/mcshim.log)" -gt "$n" ] || { echo "uid 65534 could not log"; exit 1; }'
exit $fail
