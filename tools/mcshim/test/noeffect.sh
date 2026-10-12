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

# Checks that the preloaded shim does not change processes that never
# checkpoint: each program below must print the same, and exit the same, with
# the shim preloaded as without it. CPU only: the CUDA libraries are fakes that
# answer a handful of entry points.
#
# Usage: noeffect.sh mcshim.so
set -uo pipefail
if [ $# -ne 1 ]; then
  echo "usage: $0 mcshim.so" >&2
  exit 2
fi
SHIM=$(realpath "$1")
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
cd "$W" || exit 1
mkdir -p cuda rt

# A partial libcuda: a few entry points, like an older driver. Nothing links
# it unless a case says so; cases that want it installed but not loaded put
# its directory on LD_LIBRARY_PATH.
cat >cuda/cuda.c <<'EOF'
int cuInit(unsigned f) { (void)f; return 0; }
int cuCtxCreate_v2(void** c, unsigned f, int d) { (void)c; (void)f; (void)d; return 0; }
int cuCtxGetDevice(int* d) { *d = 0; return 0; }
int cuCtxGetCurrent(void** c) { *c = 0; return 0; }
int cuCtxSetCurrent(void* c) { (void)c; return 0; }
int cuCtxSynchronize(void) { return 0; }
int cuDevicePrimaryCtxRetain(void** c, int d) { (void)c; (void)d; return 0; }
int cuDevicePrimaryCtxRelease_v2(int d) { (void)d; return 0; }
int cuDevicePrimaryCtxGetState(int d, unsigned* f, int* a) { (void)d; *f = 0; *a = 0; return 0; }
int cuPointerGetAttribute(void* p, int a, unsigned long long v) { (void)p; (void)a; (void)v; return 0; }
int cuMemAllocManaged(void) { return 0; }
int cuMemMap(unsigned long long a, unsigned long s, unsigned long o, unsigned long long h, unsigned long long f) { (void)a; (void)s; (void)o; (void)h; (void)f; return 1; }
EOF
gcc -O2 -fPIC -shared -Wl,-soname,libcuda.so.1 -o cuda/libcuda.so.1 cuda/cuda.c || exit 1

# probe: looks up driver entry points without linking libcuda, and calls them
# only if found; then reports whether libcuda got loaded.
cat >probe.c <<'EOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
int main(void) {
  const char* names[] = {"cuInit", "cuMemCreate", "cuMulticastCreate", "cudaGetDriverEntryPoint"};
  for (int i = 0; i < 4; i++) {
    void* p = dlsym(RTLD_DEFAULT, names[i]);
    printf("%s %s\n", names[i], p ? "found" : "absent");
  }
  int (*ci)(unsigned) = (int (*)(unsigned))dlsym(RTLD_DEFAULT, "cuInit");
  if (ci) printf("cuInit() = %d\n", ci(0));
  printf("libcuda %s\n", dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD) ? "loaded" : "not loaded");
  return 0;
}
EOF
gcc -O2 -o probe probe.c -ldl || exit 1

# weak: a weak reference to cuInit, which the shim's definition satisfies
# when preloaded. Calling it without a driver must fail, not crash or succeed.
cat >weak.c <<'EOF'
#include <stdio.h>
int cuInit(unsigned) __attribute__((weak));
int cuMemCreate(unsigned long long*, unsigned long, const void*, unsigned long long) __attribute__((weak));
int main(void) {
  unsigned long long h;
  int a = cuInit ? cuInit(0) : 3, b = cuMemCreate ? cuMemCreate(&h, 1, 0, 0) : 3;
  printf("cuInit %s, cuMemCreate %s\n", a ? "failed" : "succeeded", b ? "failed" : "succeeded");
  return 0;
}
EOF
gcc -O2 -o weak weak.c || exit 1

# local: a library loaded RTLD_LOCAL, linked against libcuda, looks up a
# driver entry point with RTLD_DEFAULT, which also searches its own
# dependencies; Python loads extension modules this way.
cat >local_lib.c <<'EOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
int cuInit(unsigned);
void probe(void) {
  printf("cuCtxCreate_v2 %s\n", dlsym(RTLD_DEFAULT, "cuCtxCreate_v2") ? "found" : "absent");
  printf("cuMemAllocManaged %s\n", dlsym(RTLD_DEFAULT, "cuMemAllocManaged") ? "found" : "absent");
}
void* keep(void) { return (void*)cuInit; }
EOF
cat >local.c <<'EOF'
#include <dlfcn.h>
#include <stdio.h>
int main(int argc, char** argv) {
  (void)argc;
  void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!h) { puts(dlerror()); return 1; }
  ((void (*)(void))dlsym(h, "probe"))();
  return 0;
}
EOF
gcc -O2 -fPIC -shared -o liblocal.so local_lib.c -Lcuda -l:libcuda.so.1 -Wl,-rpath,"$W/cuda" -ldl || exit 1
gcc -O2 -o local local.c -ldl || exit 1

# dlerror: the POSIX idiom for telling a NULL symbol from a failed lookup,
# after the process has used the driver.
cat >dlerr.c <<'EOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
int cuMemMap(unsigned long long, unsigned long, unsigned long, unsigned long long, unsigned long long);
int cuMemAllocManaged(void);
int main(void) {
  cuMemMap(0, 0, 0, 0, 0);
  cuMemAllocManaged();
  void* h = dlopen("libcuda.so.1", RTLD_NOW);
  dlerror();
  void* p = dlsym(h, "cuInit");
  const char* e = dlerror();
  printf("cuInit %s, dlerror %s\n", p ? "found" : "absent", e ? e : "(none)");
  return 0;
}
EOF
gcc -O2 -o dlerr dlerr.c -Lcuda -l:libcuda.so.1 -Wl,-rpath,"$W/cuda" -ldl || exit 1

# sigwait: a CUDA process that blocks SIGTERM, receives it, and collects it
# with sigwait, after cuInit (which starts the shim's control thread).
cat >sigw.c <<'EOF'
#define _GNU_SOURCE
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
int cuInit(unsigned);
int main(void) {
  cuInit(0);
  usleep(100000);
  sigset_t s;
  sigemptyset(&s);
  sigaddset(&s, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &s, 0);
  /* Sent while this thread blocks it but is not yet waiting for it: the
   * kernel delivers it to any thread that does not block it. */
  kill(getpid(), SIGTERM);
  usleep(200000);
  int sig;
  sigwait(&s, &sig);
  printf("sigwait got signal %d\n", sig);
  return 0;
}
EOF
gcc -O2 -o sigw sigw.c -Lcuda -l:libcuda.so.1 -Wl,-rpath,"$W/cuda" -lpthread || exit 1

# resolver: the main program, linked against libcudart.so.12, calls
# cudaGetDriverEntryPoint while a library loaded RTLD_LOCAL brings in
# libcudart.so.13; the main program's own runtime must answer.
cat >rt/rt.c <<'EOF'
int cudaGetDriverEntryPoint(const char* s, void** p, unsigned long long f, int* st) {
  (void)s; (void)f; *p = (void*)(long)VER; if (st) *st = 0; return 0;
}
EOF
cat >rt/other.c <<'EOF'
int cudaGetDriverEntryPoint(const char*, void**, unsigned long long, int*);
void* other(void) { return (void*)cudaGetDriverEntryPoint; }
EOF
cat >resolver.c <<'EOF'
#include <dlfcn.h>
#include <stdio.h>
int cudaGetDriverEntryPoint(const char*, void**, unsigned long long, int*);
int main(int argc, char** argv) {
  (void)argc;
  dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  void* p = 0;
  int rc = cudaGetDriverEntryPoint("probe", &p, 0, 0);
  printf("answered by runtime %ld, rc %d\n", rc ? 0L : (long)p, rc);
  return 0;
}
EOF
gcc -DVER=12 -fPIC -shared -Wl,-soname,libcudart.so.12 -o rt/libcudart.so.12 rt/rt.c || exit 1
gcc -DVER=13 -fPIC -shared -Wl,-soname,libcudart.so.13 -o rt/libcudart.so.13 rt/rt.c || exit 1
gcc -fPIC -shared -o rt/libother.so rt/other.c -Lrt -l:libcudart.so.13 -Wl,-rpath,"$W/rt" || exit 1
gcc -o resolver resolver.c -Lrt -l:libcudart.so.12 -Wl,-rpath,"$W/rt" -ldl || exit 1

# same NAME CMD...: CMD prints and exits the same with and without the shim.
fail=0
same() {
  local name=$1 want got rw rg
  shift
  want=$("$@" 2>&1)
  rw=$?
  got=$(MCSHIM_LOG=/dev/null LD_PRELOAD="$SHIM" "$@" 2>&1)
  rg=$?
  if [ "$want" = "$got" ] && [ $rw -eq $rg ]; then
    echo "PASS $name"
  else
    echo "FAIL $name"
    echo "  without the shim (exit $rw): $(echo "$want" | tr '\n' ';')"
    echo "  with the shim    (exit $rg): $(echo "$got" | tr '\n' ';')"
    fail=1
  fi
}

same "lookup without a driver" ./probe
same "lookup with a driver installed but not loaded" env LD_LIBRARY_PATH="$W/cuda" ./probe
same "RTLD_DEFAULT from an RTLD_LOCAL library" ./local "$W/liblocal.so"
same "dlerror after a successful lookup" ./dlerr
same "sigwait in a CUDA process" timeout 10 ./sigw
same "resolver called from the main program" ./resolver "$W/rt/libother.so"

# weak references bind to the shim, so the output differs by design; calling
# them without a driver must fail cleanly.
out=$(MCSHIM_LOG=/dev/null LD_PRELOAD="$SHIM" ./weak 2>&1)
rc=$?
if [ $rc -eq 0 ] && [ "$out" = "cuInit failed, cuMemCreate failed" ]; then
  echo "PASS weak references without a driver"
else
  echo "FAIL weak references without a driver (exit $rc): $out"
  fail=1
fi
exit $fail
