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

# Checks that the shim forwards the dlsym lookups it does not rewrite with a
# tail call, so that RTLD_NEXT resolves relative to the real caller: an
# interposer preloaded after the shim (or before it) still finds the next
# definition rather than itself. A build that loses the tail call (the shim
# compiled without optimization by a GCC without musttail) fails here.
#
# Usage: preload_test.sh mcshim.so libinterp.so interp_main
#   libinterp.so and interp_main are built from interp.c (interp_main with
#   -DMAIN). Bazel runs it on the built //tools/mcshim; run.sh preload on its
#   own build.
set -uo pipefail
if [ $# -ne 3 ]; then
  echo "usage: $0 mcshim.so libinterp.so interp_main" >&2
  exit 2
fi
shim=$(realpath "$1")
interp=$(realpath "$2")
main=$(realpath "$3")
for order in "$shim:$interp" "$interp:$shim"; do
  rc=0
  out=$(LD_PRELOAD="$order" "$main" 2>&1) || rc=$?
  if [ $rc -ne 0 ] || [ "$out" != ok ]; then
    echo "LD_PRELOAD=$order: exit $rc (99: RTLD_NEXT found the interposer itself), output: $out"
    exit 1
  fi
done
echo "ok: RTLD_NEXT resolves relative to the caller in both preload orders"
