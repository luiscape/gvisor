#!/usr/bin/env bash
# Build the multicast interposer. Toolkit-free: no cuda.h / nvcc required
# (CUDA types are declared locally in mcshim.c), so it builds on a bare driver
# install exactly like the _cuda.py harnesses.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-mcshim.so}"
gcc -O2 -g -Wall -Wextra -fPIC -shared -o "$OUT" mcshim.c -ldl -lpthread
echo "built $(pwd)/$OUT"
