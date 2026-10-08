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

# Checks that a mcshim.so loads in container images with old glibc. The shim
# is preloaded into every process of a container, and a library that needs a
# symbol version the image's glibc lacks makes the dynamic linker exit each
# one with status 1, so nothing in the image can start. CPU only; needs
# docker. For each image, `sh -c true` and `/bin/true` must exit 0 and print
# nothing, with the shim preloaded through LD_PRELOAD and again through
# /etc/ld.so.preload.
#
# Usage: glibc_smoke.sh mcshim.so [image...]
#   (default images: ubuntu:20.04 debian:11 rockylinux:8 centos:7, glibc 2.31,
#   2.31, 2.28 and 2.17)
set -uo pipefail
if [ $# -lt 1 ]; then
  echo "usage: $0 mcshim.so [image...]" >&2
  exit 2
fi
SHIM=$(realpath "$1")
shift
if [ $# -eq 0 ]; then
  set -- ubuntu:20.04 debian:11 rockylinux:8 centos:7
fi
docker="docker"
$docker info >/dev/null 2>&1 || docker="sudo docker"

# in_image IMAGE CMD...: runs CMD in IMAGE with the shim at /m.so.
in_image() {
  local image=$1
  shift
  $docker run --rm -v "$SHIM:/m.so:ro" "$image" "$@"
}

fail=0
for image in "$@"; do
  glibc=$(in_image "$image" /bin/sh -c 'ldd --version 2>&1 | head -1' 2>/dev/null)
  rc=0
  out=$(in_image "$image" /bin/sh -c '
    set -e
    LD_PRELOAD=/m.so sh -c true
    LD_PRELOAD=/m.so /bin/true
    echo /m.so >/etc/ld.so.preload
    sh -c true
    /bin/true
  ' 2>&1) || rc=$?
  if [ $rc -eq 0 ] && [ -z "$out" ]; then
    echo "PASS $image ($glibc)"
  else
    echo "FAIL $image ($glibc): exit $rc, output: $out"
    fail=1
  fi
done
exit $fail
