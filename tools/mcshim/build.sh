#!/usr/bin/env bash
# Build the multicast interposer. Toolkit-free: no cuda.h / nvcc required
# (CUDA types are declared locally in mcshim.c), so it builds on a bare driver
# install exactly like the _cuda.py harnesses.
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
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-mcshim.so}"
CFLAGS="-O2 -g -Wall -Wextra -fPIC -shared"
IMAGE="${MCSHIM_BUILD_IMAGE:-ubuntu:22.04}"

host_build() {
    gcc $CFLAGS -o "$OUT" mcshim.c -ldl -lpthread
    echo "built $(pwd)/$OUT (host toolchain: $(ldd --version | head -1))"
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

$docker run --rm -v "$PWD:/src" -w /src "$IMAGE" /bin/sh -c "
    set -e
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends gcc libc6-dev >/dev/null
    gcc $CFLAGS -o '$OUT' mcshim.c -ldl -lpthread
" 
echo "built $(pwd)/$OUT in $IMAGE"
