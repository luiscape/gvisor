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

# Build the multicast interposer (mcshim.so). Toolkit-free: no cuda.h / nvcc required (CUDA
# types are declared locally), so it builds on a bare driver install.
#
# The result is preloaded into every process of a container image, and a
# library that needs a symbol version the image's glibc lacks makes every
# process exit with status 1:
#   version `GLIBC_2.34' not found (required by mcshim.so)
# so it is linked against stubs of libdl.so.2 and libpthread.so.0 that pin the
# symbol versions glibc 2.34 changed (glibc_stubs.sh), and mcshim.c avoids
# the calls glibc 2.38 changed. Floor: GLIBC_2.17 (see README.md); this
# script checks the result. The build still runs inside ubuntu:22.04 by
# default so that the rest does not depend on the host's toolchain.
#
# Usage:
#   build.sh [out.so]         # containerized build (portable, default)
#   MCSHIM_HOST_BUILD=1 build.sh [out.so]
#   MCSHIM_BUILD_IMAGE=ubuntu:20.04 build.sh   # build against an older glibc
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-mcshim.so}"
CFLAGS="-O2 -g -Wall -Wextra -fPIC -shared"
# The stubs must stay NEEDED even where the build environment's glibc defines
# all their symbols in libc: hence --no-as-needed.
STUBS=".mcshim-stubs-$$"
LDLIBS="-Wl,--no-as-needed $STUBS/libdl.so.2 $STUBS/libpthread.so.0 -Wl,--as-needed"
GLIBC_FLOOR="2.17"

# check_floor FILE: fails if FILE needs a GLIBC_ symbol version above the floor.
check_floor() {
    local newest
    newest=$(objdump -T "$1" | grep -oE 'GLIBC_[0-9.]+' | sort -t_ -k2,2V | tail -1)
    if [ "$(printf '%s\nGLIBC_%s\n' "$newest" "$GLIBC_FLOOR" | sort -t_ -k2,2V | tail -1)" != "GLIBC_$GLIBC_FLOOR" ]; then
        echo "error: $1 needs $newest, above the GLIBC_$GLIBC_FLOOR floor" >&2
        objdump -T "$1" | grep "$newest" >&2
        return 1
    fi
    echo "newest glibc symbol version: $newest (floor GLIBC_$GLIBC_FLOOR)"
}
# Base image pinned by digest (ubuntu:22.04 as of 2026-08) so the glibc floor
# the result links against is stable; gcc and libc6-dev still come from the
# live apt archive, so this is a stable target, not a reproducible build.
# Bump the digest deliberately, not implicitly via the tag.
IMAGE="${MCSHIM_BUILD_IMAGE:-ubuntu:22.04@sha256:3b06811b2afd352be909dd088a004166d665dc76d38b13eada33522a9d915c6f}"

host_build() {
    ./glibc_stubs.sh "$STUBS"
    gcc $CFLAGS -o "$OUT" mcshim.c $LDLIBS
    check_floor "$OUT"
    echo "built $(realpath "$OUT") (host toolchain: $(ldd --version | head -1))"
}

trap 'rm -rf "$STUBS"' EXIT
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
trap 'rm -rf "$TMP_OUT" "$STUBS"' EXIT
$docker run --rm -v "$PWD:/src" -w /src "$IMAGE" /bin/sh -c '
    set -e
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends gcc libc6-dev >/dev/null
    ./glibc_stubs.sh "$4"
    gcc '"$CFLAGS"' -o "$1" mcshim.c '"$LDLIBS"'
    chown -R "$2:$3" "$1" "$4"
' mcshim-build "$TMP_OUT" "$(id -u)" "$(id -g)" "$STUBS"
check_floor "$TMP_OUT"
mv "$TMP_OUT" "$OUT"
echo "built $(realpath "$OUT") in $IMAGE"
