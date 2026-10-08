#!/bin/sh
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

# Builds link-time stubs of libdl.so.2 and libpthread.so.0 for mcshim.so's
# glibc floor (see mcshim.c). They define the entry points glibc 2.34 moved
# into libc, at the versions every glibc has exported them at, and nothing
# else; they are never loaded.
#
# Linking mcshim.so against them, rather than against the build environment's
# libc, records in its version needs that dlopen@GLIBC_2.2.5 comes from
# libdl.so.2. The dynamic linker requires the object named there to define
# that version (every libdl.so.2 does, including the empty one glibc >= 2.34
# ships), then looks the symbol up by name and version in the whole search
# scope: it binds to libdl.so.2 where that still defines dlopen (glibc <
# 2.34) and to libc.so.6's compatibility symbol where it no longer does.
# Linked against a glibc >= 2.34 libc instead, the version need names
# libc.so.6, and a glibc < 2.34 loader fails every process at startup with
# "symbol dlopen, version GLIBC_2.2.5 not defined in file libc.so.6".
#
# Usage: glibc_stubs.sh OUTDIR        (CC selects the compiler; default gcc)
set -eu
out=$1
cc=${CC:-gcc}
case $($cc -dumpmachine) in
  x86_64-*) base=GLIBC_2.2.5 ;;
  aarch64-*) base=GLIBC_2.17 ;;
  *)
    echo "glibc_stubs.sh: unsupported target $($cc -dumpmachine)" >&2
    exit 1
    ;;
esac
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$out"

# stub SONAME SYMBOL...: writes $out/SONAME defining each SYMBOL at $base.
stub() {
  soname=$1
  shift
  : >"$tmp/stub.c"
  printf '%s {\n  global:' "$base" >"$tmp/stub.map"
  for sym in "$@"; do
    printf 'void %s(void) {}\n' "$sym" >>"$tmp/stub.c"
    printf ' %s;' "$sym" >>"$tmp/stub.map"
  done
  printf '\n  local: *;\n};\n' >>"$tmp/stub.map"
  $cc -shared -nostdlib -fPIC -Wl,-soname,"$soname" \
    -Wl,--version-script,"$tmp/stub.map" -o "$out/$soname" "$tmp/stub.c"
}

stub libdl.so.2 dlopen dlclose dlerror dladdr dlvsym
stub libpthread.so.0 pthread_create pthread_detach
