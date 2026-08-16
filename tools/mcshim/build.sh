#!/usr/bin/env bash
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

# Build the multicast interposer (mcshim.so) and its create/attach proxy
# helper (mcshim-helper, used on pre-R610 drivers). Toolkit-free: no
# cuda.h / nvcc required (CUDA types are declared locally), so it builds on
# a bare driver install.
#
# By default the build runs inside ubuntu:22.04 so the result is loadable in
# container images with older glibc than the host. A host toolchain links
# against whatever glibc it has, and a newer one is not forward compatible:
# glibc 2.38+ redirects sscanf to __isoc23_sscanf, so a host-built library
# fails to load in a 22.04 image with
#   version `GLIBC_2.38' not found (required by mcshim.so)
# which surfaces as the container exiting immediately with an empty log.
#
# Usage:
#   build.sh [out.so]         # containerized build (portable, default)
#   MCSHIM_HOST_BUILD=1 build.sh [out.so]
#   MCSHIM_BUILD_IMAGE=ubuntu:20.04 build.sh   # target an even older glibc
#
# The helper is written next to the interposer as "mcshim-helper".
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-mcshim.so}"
HELPER_OUT="$(dirname "$OUT")/mcshim-helper"
CFLAGS="-O2 -g -Wall -Wextra -fPIC -shared"
HELPER_CFLAGS="-O2 -g -Wall -Wextra"
# Base image pinned by digest (ubuntu:22.04 as of 2026-08) so the glibc floor
# the result links against is stable; gcc and libc6-dev still come from the
# live apt archive, so this is a stable target, not a reproducible build.
# Bump the digest deliberately, not implicitly via the tag.
IMAGE="${MCSHIM_BUILD_IMAGE:-ubuntu:22.04@sha256:3b06811b2afd352be909dd088a004166d665dc76d38b13eada33522a9d915c6f}"

host_build() {
    gcc $CFLAGS -o "$OUT" mcshim.c -ldl -lpthread
    gcc $HELPER_CFLAGS -o "$HELPER_OUT" mcshim_helper.c -ldl
    echo "built $(realpath "$OUT") + $(realpath "$HELPER_OUT") (host toolchain: $(ldd --version | head -1))"
}

if [[ "${MCSHIM_HOST_BUILD:-0}" = "1" ]]; then
    host_build
    exit 0
fi

docker="docker"
$docker info >/dev/null 2>&1 || docker="sudo docker"
if ! $docker info >/dev/null 2>&1; then
    echo "warning: docker unavailable; falling back to a host build, which may" >&2
    echo "         not load in images with older glibc than this host" >&2
    host_build
    exit 0
fi

# The container compiles to temporary names inside the bind mount and the
# results are moved to their outputs afterwards, so absolute output paths
# work too. The output names, uid and gid travel as positional arguments
# (never interpolated into the shell script), and the container chowns its
# outputs -- docker runs the build as root -- back to the invoking user so no
# root-owned file is left in the tree.
TMP_OUT=".mcshim-build-$$.so"
TMP_HELPER=".mcshim-helper-build-$$"
trap 'rm -f "$TMP_OUT" "$TMP_HELPER"' EXIT
$docker run --rm -v "$PWD:/src" -w /src "$IMAGE" /bin/sh -c '
    set -e
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends gcc libc6-dev >/dev/null
    gcc '"$CFLAGS"' -o "$1" mcshim.c -ldl -lpthread
    gcc '"$HELPER_CFLAGS"' -o "$4" mcshim_helper.c -ldl
    chown "$2:$3" "$1" "$4"
' mcshim-build "$TMP_OUT" "$(id -u)" "$(id -g)" "$TMP_HELPER"
mv "$TMP_OUT" "$OUT"
mv "$TMP_HELPER" "$HELPER_OUT"
trap - EXIT
echo "built $(realpath "$OUT") + $(realpath "$HELPER_OUT") in $IMAGE"
