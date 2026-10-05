/*
 * Copyright 2026 The gVisor Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Interposer tests (see run.sh). mcshim must be LD_PRELOADed. Each test plays
 * the sentry's side of the marker protocol, with a suspend and resume in place
 * of a checkpoint and restore. Multicast tests need two GPUs with switch
 * multicast (NVLS); "ipc" needs nvproxy's fdinfo identity, so runsc.
 *
 *   abi       every lookup gets the wrapper of the exact ABI it returned
 *   gate      the gate stops submissions and drains calls in flight
 *   mc        multicast rebuild: v1 and v2 binds, merged access, retained
 *             handles, stale and colliding handle values
 *   refcount  objects whose application handles were released while mapped
 *   refuse    state that cannot be carried refuses at the gate
 *   ipc       two processes: exported groups imported in opposite order, and
 *             a peer buffer
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef void* CUmemoryPool;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long H;

typedef struct {
  int type, id;
} Loc;
typedef struct {
  int type, handleTypes;
  Loc loc;
  void* win32;
  unsigned char flags[8];
} AllocProp;
typedef struct {
  Loc loc;
  int flags;
} Access;
typedef struct {
  unsigned numDevices;
  size_t size;
  unsigned long long handleTypes, flags;
} McProp;
typedef struct {
  int allocType, handleTypes;
  Loc loc;
  void* win32;
  size_t maxSize;
  unsigned short usage;
  unsigned char rdma, reserved[53];
} PoolProp;
typedef struct {
  int srcAccessOrder;
  Loc src, dst;
  unsigned flags;
} CopyAttr;

#define POSIX_FD 1
#define DEVICE 1
#define RW 3
#define ATTR_CLOCK_RATE 13
#define ATTR_MULTICAST 132

static void* lib;
static void* (*rdlsym)(void*, const char*);

#define FNS(X)                                                                 \
  X(cuInit, (unsigned))                                                        \
  X(cuDeviceGetCount, (int*))                                                  \
  X(cuDeviceGetAttribute, (int*, int, CUdevice))                               \
  X(cuDevicePrimaryCtxRetain, (CUcontext*, CUdevice))                          \
  X(cuCtxSetCurrent, (CUcontext))                                              \
  X(cuCtxSynchronize, (void))                                                  \
  X(cuModuleLoadData, (CUmodule*, const void*))                                \
  X(cuModuleGetFunction, (CUfunction*, CUmodule, const char*))                 \
  X(cuLaunchKernel, (CUfunction, unsigned, unsigned, unsigned, unsigned,       \
                     unsigned, unsigned, unsigned, CUstream, void**, void**))  \
  X(cuStreamCreate, (CUstream*, unsigned))                                     \
  X(cuStreamSynchronize, (CUstream))                                           \
  X(cuMemAlloc_v2, (CUdeviceptr*, size_t))                                     \
  X(cuMemGetInfo_v2, (size_t*, size_t*))                                       \
  X(cuMemcpyDtoH_v2, (void*, CUdeviceptr, size_t))                             \
  X(cuMemcpyHtoD_v2, (CUdeviceptr, const void*, size_t))                       \
  X(cuMemcpyDtoD_v2, (CUdeviceptr, CUdeviceptr, size_t))                       \
  X(cuMemsetD32_v2, (CUdeviceptr, unsigned, size_t))                           \
  X(cuMemsetD8Async, (CUdeviceptr, unsigned char, size_t, CUstream))           \
  X(cuMemCreate, (H*, size_t, const AllocProp*, unsigned long long))           \
  X(cuMemRelease, (H))                                                         \
  X(cuMemMap, (CUdeviceptr, size_t, size_t, H, unsigned long long))            \
  X(cuMemUnmap, (CUdeviceptr, size_t))                                         \
  X(cuMemSetAccess, (CUdeviceptr, size_t, const Access*, size_t))              \
  X(cuMemAddressReserve,                                                       \
    (CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long))           \
  X(cuMemGetAllocationGranularity, (size_t*, const AllocProp*, int))           \
  X(cuMemRetainAllocationHandle, (H*, void*))                                  \
  X(cuMemGetAllocationPropertiesFromHandle, (AllocProp*, H))                   \
  X(cuMemExportToShareableHandle, (void*, H, int, unsigned long long))         \
  X(cuMemImportFromShareableHandle, (H*, void*, int))                          \
  X(cuMulticastCreate, (H*, const McProp*))                                    \
  X(cuMulticastAddDevice, (H, CUdevice))                                       \
  X(cuMulticastBindMem, (H, size_t, H, size_t, size_t, unsigned long long))    \
  X(cuMulticastBindAddr, (H, size_t, CUdeviceptr, size_t, unsigned long long)) \
  X(cuMemAllocManaged, (CUdeviceptr*, size_t, unsigned))                       \
  X(cuMulticastUnbind, (H, CUdevice, size_t, size_t))                          \
  X(cuMulticastGetGranularity, (size_t*, const McProp*, int))                  \
  X(cuMemPoolCreate, (CUmemoryPool*, const PoolProp*))                         \
  X(cuMemPoolExportToShareableHandle,                                          \
    (void*, CUmemoryPool, int, unsigned long long))                            \
  X(cuGetProcAddress_v2, (const char*, void**, int, unsigned long long, int*))

#define DECL(name, proto) static CUresult(*name) proto;
FNS(DECL)

static int g_failed;
#define CK(x)                                                                \
  do {                                                                       \
    CUresult rc_ = (x);                                                      \
    if (rc_ != 0) {                                                          \
      fprintf(stderr, "FAIL %s:%d: %s = %d\n", __FILE__, __LINE__, #x, rc_); \
      exit(1);                                                               \
    }                                                                        \
  } while (0)
#define EXPECT(c, ...)                                             \
  do {                                                             \
    if (!(c)) {                                                    \
      fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #c); \
      fprintf(stderr, __VA_ARGS__);                                \
      fputc('\n', stderr);                                         \
      g_failed = 1;                                                \
    }                                                              \
  } while (0)

static void resolve(void) {
  *(void**)&rdlsym = dlvsym(RTLD_DEFAULT, "dlsym", "GLIBC_2.34");
  if (!rdlsym) *(void**)&rdlsym = dlvsym(RTLD_DEFAULT, "dlsym", "GLIBC_2.2.5");
  lib = dlopen("libcuda.so.1", RTLD_NOW);
  if (!lib || !rdlsym) {
    fprintf(stderr, "cannot load libcuda\n");
    exit(2);
  }
  /* Through the interposed dlsym, as ctypes and torch do. */
#define RES(name, proto) *(void**)&name = dlsym(lib, #name);
  FNS(RES)
}

/* A driver entry point as resolved by torch and NCCL. */
static void* gpa(const char* name, int ver, int flags) {
  void* p = NULL;
  int st;
  CK(cuGetProcAddress_v2(name, &p, ver, flags, &st));
  return p;
}

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void msleep(int ms) {
  struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
  nanosleep(&ts, NULL);
}

/* The sentry's side of the marker protocol. */

static void mpath(char* p, const char* name) {
  snprintf(p, 256, "/tmp/mcshim/%s", name);
}

static void mk(const char* name) {
  char p[256];
  mpath(p, name);
  close(open(p, O_CREAT | O_WRONLY, 0666));
}

static void rm(const char* name) {
  char p[256];
  mpath(p, name);
  unlink(p);
}

static int exists(const char* name) {
  char p[256];
  mpath(p, name);
  return access(p, F_OK) == 0;
}

/* Waits for <prefix>.<pid> from every pid: 0, or -1 on error.<pid>. */
static int wait_ack(const char* prefix, const pid_t* pids, int n) {
  for (double end = now() + 120; now() < end; msleep(10)) {
    int done = 0;
    for (int i = 0; i < n; i++) {
      char a[64];
      snprintf(a, sizeof(a), "error.%d", (int)pids[i]);
      if (exists(a)) return -1;
      snprintf(a, sizeof(a), "%s.%d", prefix, (int)pids[i]);
      done += exists(a);
    }
    if (done == n) return 0;
  }
  fprintf(stderr, "timed out waiting for %s\n", prefix);
  return -1;
}

static void clear_markers(void) {
  rm("gate");
  rm("suspend");
}

/* The sentry's preamble to each request: stale acks must not satisfy it. */
static void clear_acks(const pid_t* pids, int n) {
  static const char* const kinds[] = {"gated", "suspended", "resumed", "error"};
  for (int i = 0; i < n; i++)
    for (int k = 0; k < 4; k++) {
      char a[64];
      snprintf(a, sizeof(a), "%s.%d", kinds[k], (int)pids[i]);
      rm(a);
    }
}

static int gate_up(const pid_t* pids, int n) {
  mk("gate");
  return wait_ack("gated", pids, n);
}

static void gate_down(void) {
  rm("gate");
  msleep(50);
}

/* Gate, suspend, resume, ungate: a checkpoint and restore without the images.
 */
static int cycle(const pid_t* pids, int n) {
  if (gate_up(pids, n) != 0) return -1;
  mk("suspend");
  if (wait_ack("suspended", pids, n) != 0) return -1;
  rm("suspend");
  if (wait_ack("resumed", pids, n) != 0) return -1;
  gate_down();
  return 0;
}

static CUcontext g_ctx[2];
static CUfunction g_spin, g_bump, g_bcast;

/* spin(p, cycles): busy-wait, then add 1 to *p. bump(p): add 1 to *p.
 * bcast(p, v): multimem store of v through the multicast VA p. */
static const char kPTX[] =
    ".version 8.1\n.target sm_90\n.address_size 64\n"
    ".visible .entry spin(.param .u64 p, .param .u64 c) {\n"
    " .reg .u64 %a<6>; .reg .pred %q;\n"
    " ld.param.u64 %a1, [p]; ld.param.u64 %a2, [c];\n"
    " mov.u64 %a3, %clock64;\n"
    "L: mov.u64 %a4, %clock64; sub.u64 %a5, %a4, %a3;\n"
    " setp.lt.u64 %q, %a5, %a2; @%q bra L;\n"
    " red.global.add.u32 [%a1], 1; ret; }\n"
    ".visible .entry bump(.param .u64 p) {\n"
    " .reg .u64 %a1; ld.param.u64 %a1, [p];\n"
    " red.global.add.u32 [%a1], 1; ret; }\n"
    ".visible .entry bcast(.param .u64 p, .param .u32 v) {\n"
    " .reg .u64 %a1; .reg .b32 %b1;\n"
    " ld.param.u64 %a1, [p]; ld.param.u32 %b1, [v];\n"
    " multimem.st.relaxed.sys.global.b32 [%a1], %b1; ret; }\n";

static void use(int d) { CK(cuCtxSetCurrent(g_ctx[d])); }

static int setup(int ndev) {
  resolve();
  CK(cuInit(0));
  int n;
  CK(cuDeviceGetCount(&n));
  if (n < ndev) {
    fprintf(stderr, "SKIP: need %d GPUs\n", ndev);
    exit(77);
  }
  for (int d = 0; d < ndev; d++) {
    CK(cuDevicePrimaryCtxRetain(&g_ctx[d], d));
    int mc = 0;
    CK(cuDeviceGetAttribute(&mc, ATTR_MULTICAST, d));
    if (ndev > 1 && !mc) {
      fprintf(stderr, "SKIP: no multicast support\n");
      exit(77);
    }
  }
  use(0);
  CUmodule m;
  CK(cuModuleLoadData(&m, kPTX));
  CK(cuModuleGetFunction(&g_spin, m, "spin"));
  CK(cuModuleGetFunction(&g_bump, m, "bump"));
  CK(cuModuleGetFunction(&g_bcast, m, "bcast"));
  /* present.<pid> appears once cuInit starts the control thread. */
  char p[64];
  snprintf(p, sizeof(p), "present.%d", (int)getpid());
  for (int i = 0; i < 500 && !exists(p); i++) msleep(10);
  return 0;
}

static CUresult launch1(CUfunction f, void** args, CUstream st) {
  return cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, st, args, NULL);
}

static unsigned read32(int d, CUdeviceptr p) {
  unsigned v = 0;
  use(d);
  CK(cuMemcpyDtoH_v2(&v, p, 4));
  return v;
}

/* abi */

static const char* const kBases[] = {"cuInit",
                                     "cuDeviceGetAttribute",
                                     "cuGetProcAddress",
                                     "cuMemCreate",
                                     "cuMemRelease",
                                     "cuMemMap",
                                     "cuMemUnmap",
                                     "cuMemSetAccess",
                                     "cuMemRetainAllocationHandle",
                                     "cuMemGetAllocationPropertiesFromHandle",
                                     "cuMemExportToShareableHandle",
                                     "cuMemImportFromShareableHandle",
                                     "cuMemMapArrayAsync",
                                     "cuMulticastCreate",
                                     "cuMulticastAddDevice",
                                     "cuMulticastBindMem",
                                     "cuMulticastBindAddr",
                                     "cuMulticastUnbind",
                                     "cuLaunchKernel",
                                     "cuLaunchKernelEx",
                                     "cuLaunchCooperativeKernel",
                                     "cuLaunchHostFunc",
                                     "cuGraphLaunch",
                                     "cuMemsetD8",
                                     "cuMemsetD16",
                                     "cuMemsetD32",
                                     "cuMemsetD8Async",
                                     "cuMemsetD16Async",
                                     "cuMemsetD32Async",
                                     "cuMemsetD2D8",
                                     "cuMemsetD2D16",
                                     "cuMemsetD2D32",
                                     "cuMemsetD2D8Async",
                                     "cuMemsetD2D16Async",
                                     "cuMemsetD2D32Async",
                                     "cuMemcpy",
                                     "cuMemcpyAsync",
                                     "cuMemcpyPeer",
                                     "cuMemcpyPeerAsync",
                                     "cuMemcpyHtoD",
                                     "cuMemcpyDtoH",
                                     "cuMemcpyDtoD",
                                     "cuMemcpyHtoDAsync",
                                     "cuMemcpyDtoHAsync",
                                     "cuMemcpyDtoDAsync",
                                     "cuMemcpyAtoD",
                                     "cuMemcpyDtoA",
                                     "cuMemcpyAtoH",
                                     "cuMemcpyHtoA",
                                     "cuMemcpyAtoA",
                                     "cuMemcpyHtoAAsync",
                                     "cuMemcpyAtoHAsync",
                                     "cuMemcpy2D",
                                     "cuMemcpy2DUnaligned",
                                     "cuMemcpy2DAsync",
                                     "cuMemcpy3D",
                                     "cuMemcpy3DAsync",
                                     "cuMemcpy3DPeer",
                                     "cuMemcpy3DPeerAsync",
                                     "cuMemcpyBatchAsync",
                                     "cuMemcpy3DBatchAsync",
                                     "cuMemcpyWithAttributesAsync",
                                     "cuMemcpy3DWithAttributesAsync",
                                     "cuStreamWaitValue32",
                                     "cuStreamWaitValue64",
                                     "cuStreamWriteValue32",
                                     "cuStreamWriteValue64",
                                     "cuStreamBatchMemOp",
                                     "cuStreamSynchronize",
                                     "cuMemPoolExportToShareableHandle",
                                     "cuMemPoolImportFromShareableHandle",
                                     "cuMemPoolExportPointer",
                                     "cuMemPoolImportPointer",
                                     "cuLogicalEndpointCreate",
                                     "cuLogicalEndpointImport",
                                     "cuLogicalEndpointBindMem",
                                     "cuMemAllocManaged",
                                     NULL};

static int t_abi(void) {
  setup(1);
  Dl_info di;
  dladdr((void*)dlsym, &di);
  void* shim = dlopen(di.dli_fname, RTLD_NOW | RTLD_NOLOAD);
  EXPECT(shim && strstr(di.dli_fname, "mcshim"), "dlsym is %s", di.dli_fname);
  CUresult (*real_gpa)(const char*, void**, int, unsigned long long, int*);
  *(void**)&real_gpa = rdlsym(lib, "cuGetProcAddress_v2");
  EXPECT((void*)cuGetProcAddress_v2 == rdlsym(shim, "cuGetProcAddress_v2"),
         "dlsym(cuGetProcAddress_v2) not redirected");
  static const int vers[] = {11030, 12000, 12030, 12080,
                             13000, 13010, 13020, 13030};
  int checked = 0, wrapped = 0;
  for (int b = 0; kBases[b]; b++)
    for (size_t v = 0; v < sizeof(vers) / sizeof(vers[0]); v++)
      for (int fl = 1; fl <= 2; fl++) {
        void *pr = NULL, *ps = NULL;
        int st;
        CUresult rr = real_gpa(kBases[b], &pr, vers[v], fl, &st);
        CUresult rs = cuGetProcAddress_v2(kBases[b], &ps, vers[v], fl, &st);
        EXPECT(rr == rs, "%s@%d: rc %d vs %d", kBases[b], vers[v], rr, rs);
        if (rr != 0 || !pr) {
          EXPECT(!ps, "%s@%d: unexpected pointer", kBases[b], vers[v]);
          continue;
        }
        checked++;
        const char* sym =
            dladdr(pr, &di) && di.dli_sname ? di.dli_sname : "(none)";
        EXPECT(rdlsym(lib, sym) == pr, "%s@%d: not an exported symbol",
               kBases[b], vers[v]);
        void* w = rdlsym(shim, sym);
        /* Every ABI of these entry points must be wrapped. */
        EXPECT(w, "%s@%d/%d resolves to %s, which has no wrapper", kBases[b],
               vers[v], fl, sym);
        EXPECT(ps == (w ? w : pr), "%s@%d/%d (%s): wrong pointer", kBases[b],
               vers[v], fl, sym);
        wrapped += w && ps == w;
        /* The same symbol through dlsym. */
        EXPECT(dlsym(lib, sym) == (w ? w : pr), "dlsym(%s): wrong pointer",
               sym);
      }
  /* Legacy 32-bit ABI names stay the driver's. */
  EXPECT(dlsym(lib, "cuMemcpyHtoD") == rdlsym(lib, "cuMemcpyHtoD"),
         "dlsym(cuMemcpyHtoD) redirected");
  printf("abi: %d lookups, %d redirected\n", checked, wrapped);
  /* None of those lookups may have made the process uncheckpointable. */
  pid_t me = getpid();
  EXPECT(gate_up(&me, 1) == 0, "gate refused after lookups");
  gate_down();
  return g_failed;
}

/* gate */

static volatile int g_stop, g_count;

static void* launcher(void* arg) {
  CUdeviceptr p = *(CUdeviceptr*)arg;
  use(0);
  void* args[] = {&p};
  while (!g_stop) {
    CK(launch1(g_bump, args, NULL));
    __atomic_add_fetch(&g_count, 1, __ATOMIC_SEQ_CST);
  }
  return NULL;
}

static CUdeviceptr g_buf;
static double g_sync_done;

static void* long_sync(void* arg) {
  (void)arg;
  use(0);
  int khz = 0;
  CK(cuDeviceGetAttribute(&khz, ATTR_CLOCK_RATE, 0));
  unsigned long long cycles = (unsigned long long)khz * 1500; /* ~1.5 s */
  void* args[] = {&g_buf, &cycles};
  CK(launch1(g_spin, args, NULL));
  unsigned v;
  CK(cuMemcpyDtoH_v2(&v, g_buf, 4)); /* synchronous: waits for spin */
  g_sync_done = now();
  return NULL;
}

typedef struct {
  const char* name;
  CUresult (*fn)(void);
  volatile int done;
  CUresult rc;
} GatedCall;

static CUstream g_st;

static CUresult c_memset(void) { return cuMemsetD32_v2(g_buf, 7, 16); }
static CUresult c_memset_ptds(void) {
  CUresult (*f)(CUdeviceptr, unsigned, size_t) = gpa("cuMemsetD32", 13030, 2);
  return f(g_buf, 7, 16);
}
static CUresult c_memset_async(void) {
  return cuMemsetD8Async(g_buf, 1, 16, g_st);
}
static CUresult c_dtod(void) { return cuMemcpyDtoD_v2(g_buf + 64, g_buf, 64); }
static CUresult c_copy_async(void) {
  CUresult (*f)(CUdeviceptr, CUdeviceptr, size_t, CUstream) =
      gpa("cuMemcpyAsync", 13030, 1);
  return f(g_buf + 128, g_buf, 64, g_st);
}
static CUresult c_write_value(void) {
  CUresult (*f)(CUstream, CUdeviceptr, unsigned, unsigned) =
      gpa("cuStreamWriteValue32", 13030, 1);
  return f(g_st, g_buf + 256, 5, 0);
}
static CUresult c_wait_value(void) {
  CUresult (*f)(CUstream, CUdeviceptr, unsigned, unsigned) =
      gpa("cuStreamWaitValue32", 13030, 1);
  return f(g_st, g_buf + 256, 0, 0 /* GEQ */);
}
static CUresult c_batch(void) {
  CUresult (*f)(CUdeviceptr*, CUdeviceptr*, size_t*, size_t, CopyAttr*, size_t*,
                size_t, CUstream) = gpa("cuMemcpyBatchAsync", 13030, 1);
  CUdeviceptr dst = g_buf + 512, src = g_buf;
  size_t size = 64, idx = 0;
  CopyAttr attr = {1 /* STREAM */, {DEVICE, 0}, {DEVICE, 0}, 0};
  return f(&dst, &src, &size, 1, &attr, &idx, 1, g_st);
}
static void host_fn(void* p) { (void)p; }
static CUresult c_host_fn(void) {
  CUresult (*f)(CUstream, void (*)(void*), void*) =
      gpa("cuLaunchHostFunc", 12000, 1);
  return f(g_st, host_fn, NULL);
}
static CUresult c_launch_ptsz(void) {
  CUresult (*f)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                unsigned, unsigned, CUstream, void**, void**) =
      gpa("cuLaunchKernel", 13030, 2);
  void* args[] = {&g_buf};
  return f(g_bump, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL);
}
static CUresult c_stream_sync(void) { return cuStreamSynchronize(g_st); }

static void* run_call(void* arg) {
  GatedCall* c = arg;
  use(0);
  c->rc = c->fn();
  c->done = 1;
  return NULL;
}

static int t_gate(void) {
  setup(1);
  pid_t me = getpid();
  CK(cuMemAlloc_v2(&g_buf, 1 << 20));
  CK(cuMemsetD32_v2(g_buf, 0, 1 << 18));
  CK(cuStreamCreate(&g_st, 1));

  /* Submissions stop while gated. */
  pthread_t t;
  pthread_create(&t, NULL, launcher, &g_buf);
  msleep(100);
  EXPECT(gate_up(&me, 1) == 0, "gate refused");
  int c1 = g_count;
  msleep(300);
  int c2 = g_count;
  EXPECT(c2 - c1 <= 1, "%d launches while gated", c2 - c1);
  gate_down();
  msleep(100);
  EXPECT(g_count > c2, "launches did not resume");
  g_stop = 1;
  pthread_join(t, NULL);
  use(0);
  CK(cuCtxSynchronize());

  /* The gate waits for a synchronous call in flight. */
  pthread_create(&t, NULL, long_sync, NULL);
  msleep(300);
  double t0 = now();
  EXPECT(gate_up(&me, 1) == 0, "gate refused");
  double t1 = now();
  EXPECT(g_sync_done > 0 && g_sync_done <= t1,
         "gated before the call in flight returned");
  EXPECT(t1 - t0 > 0.5, "gated after %.2fs, before the call drained", t1 - t0);
  gate_down();
  pthread_join(t, NULL);

  /* Every variant blocks while gated. */
  GatedCall calls[] = {
      {"cuMemsetD32_v2", c_memset, 0, 0},
      {"cuMemsetD32_v2_ptds", c_memset_ptds, 0, 0},
      {"cuMemsetD8Async", c_memset_async, 0, 0},
      {"cuMemcpyDtoD_v2", c_dtod, 0, 0},
      {"cuMemcpyAsync", c_copy_async, 0, 0},
      {"cuStreamWriteValue32_v2", c_write_value, 0, 0},
      {"cuStreamWaitValue32_v2", c_wait_value, 0, 0},
      {"cuMemcpyBatchAsync_v2", c_batch, 0, 0},
      {"cuLaunchHostFunc", c_host_fn, 0, 0},
      {"cuLaunchKernel_ptsz", c_launch_ptsz, 0, 0},
      {"cuStreamSynchronize", c_stream_sync, 0, 0},
  };
  for (size_t i = 0; i < sizeof(calls) / sizeof(calls[0]); i++) {
    EXPECT(gate_up(&me, 1) == 0, "gate refused");
    pthread_create(&t, NULL, run_call, &calls[i]);
    msleep(150);
    EXPECT(!calls[i].done, "%s ran while gated", calls[i].name);
    gate_down();
    pthread_join(t, NULL);
    EXPECT(calls[i].done && calls[i].rc == 0, "%s rc=%d", calls[i].name,
           calls[i].rc);
  }
  /* After its resume, the application stays gated until the gate is removed,
   * since peers may still be binding. */
  EXPECT(gate_up(&me, 1) == 0, "gate refused");
  mk("suspend");
  EXPECT(wait_ack("suspended", &me, 1) == 0, "suspend failed");
  rm("suspend");
  EXPECT(wait_ack("resumed", &me, 1) == 0, "resume failed");
  GatedCall late = {"cuMemsetD32_v2", c_memset, 0, 0};
  pthread_create(&t, NULL, run_call, &late);
  msleep(150);
  EXPECT(!late.done, "released at its own resume, before the gate was removed");
  gate_down();
  pthread_join(t, NULL);
  EXPECT(late.done && late.rc == 0, "rc=%d after the gate was removed",
         late.rc);

  /* A suspend without the gate refuses, and leaves the application running. */
  clear_acks(&me, 1);
  mk("suspend");
  EXPECT(wait_ack("suspended", &me, 1) != 0, "suspended without the gate");
  CK(cuMemsetD32_v2(g_buf, 3, 16));
  rm("suspend");
  msleep(50);

  /* While torn down, the gate can go and come back (a sentry retry or
   * abort). Re-arming must neither release the application over the torn
   * state nor, if the preflight refuses, strand it after the resume. */
  clear_acks(&me, 1);
  EXPECT(gate_up(&me, 1) == 0, "gate refused");
  mk("suspend");
  EXPECT(wait_ack("suspended", &me, 1) == 0, "suspend failed");
  GatedCall torn = {"cuMemsetD32_v2", c_memset, 0, 0};
  pthread_create(&t, NULL, run_call, &torn);
  rm("gate");
  msleep(100);
  EXPECT(!torn.done, "released over torn-down state");
  clear_acks(&me, 1);
  EXPECT(gate_up(&me, 1) == 0, "re-armed gate refused");
  /* Now make the preflight refuse: a lookup of an ABI without a wrapper. */
  gpa("cuMemsetD8", 2000, 1);
  rm("gate");
  msleep(50);
  clear_acks(&me, 1);
  EXPECT(gate_up(&me, 1) != 0, "gate accepted after an unknown ABI lookup");
  msleep(100);
  EXPECT(!torn.done, "released over torn-down state after a refusal");
  clear_acks(&me, 1);
  rm("suspend");
  EXPECT(wait_ack("resumed", &me, 1) == 0, "resume failed");
  msleep(100);
  EXPECT(!torn.done, "released before the gate was removed");
  gate_down();
  pthread_join(t, NULL);
  EXPECT(torn.done && torn.rc == 0, "stranded after the resume: rc=%d",
         torn.rc);

  use(0);
  CK(cuCtxSynchronize());
  printf("gate: ok\n");
  return g_failed;
}

/* mc */

typedef struct {
  H mc, uc[2];
  CUdeviceptr vmc, vuc[2];
  size_t size;
} Group;

static AllocProp uc_prop(int d, int types) {
  AllocProp p;
  memset(&p, 0, sizeof(p));
  p.type = 1; /* PINNED */
  p.handleTypes = types;
  p.loc.type = DEVICE;
  p.loc.id = d;
  return p;
}

static CUdeviceptr map(H h, size_t size, int d, int naccess) {
  CUdeviceptr va;
  CK(cuMemAddressReserve(&va, size, 0, 0, 0));
  CK(cuMemMap(va, size, 0, h, 0));
  /* One grant per call, as torch does for peers. */
  for (int k = 0; k < naccess; k++) {
    Access a = {{DEVICE, (d + k) % 2}, RW};
    CK(cuMemSetAccess(va, size, &a, 1));
  }
  return va;
}

static size_t mc_size(void) {
  McProp p = {2, 0, 0, 0};
  size_t g;
  CK(cuMulticastGetGranularity(&g, &p, 0 /* MINIMUM */));
  return g;
}

/* A group with one unicast buffer per device: device 0's bound through
 * cuMulticastBindMem, device 1's through cuMulticastBindMem_v2. */
static void make_group(Group* g, int types) {
  g->size = mc_size();
  McProp p = {2, g->size, (unsigned long long)types, 0};
  use(0);
  CK(cuMulticastCreate(&g->mc, &p));
  CK(cuMulticastAddDevice(g->mc, 0));
  CK(cuMulticastAddDevice(g->mc, 1));
  CUresult (*bind_v2)(H, CUdevice, size_t, H, size_t, size_t,
                      unsigned long long) = gpa("cuMulticastBindMem", 13010, 1);
  for (int d = 0; d < 2; d++) {
    use(d);
    AllocProp up = uc_prop(d, types);
    CK(cuMemCreate(&g->uc[d], g->size, &up, 0));
    if (d == 0)
      CK(cuMulticastBindMem(g->mc, 0, g->uc[d], 0, g->size, 0));
    else
      CK(bind_v2(g->mc, d, 0, g->uc[d], 0, g->size, 0));
    /* Device 0's buffer is granted to both devices, one call each. */
    g->vuc[d] = map(g->uc[d], g->size, d, d == 0 ? 2 : 1);
  }
  use(0);
  g->vmc = map(g->mc, g->size, 0, 1);
}

static void bcast(const Group* g, unsigned v) {
  use(0);
  void* args[] = {(void*)&g->vmc, &v};
  CK(launch1(g_bcast, args, NULL));
  CK(cuCtxSynchronize());
}

static void check_group(const Group* g, unsigned v, const char* when) {
  bcast(g, v);
  for (int d = 0; d < 2; d++)
    EXPECT(read32(d, g->vuc[d]) == v, "%s: dev %d reads 0x%x, want 0x%x", when,
           d, read32(d, g->vuc[d]), v);
  /* Device 0's buffer through device 1: the second, separate grant. */
  EXPECT(read32(1, g->vuc[0]) == v, "%s: peer read", when);
}

static void destroy_group(Group* g) {
  use(0);
  CK(cuMemUnmap(g->vmc, g->size));
  for (int d = 0; d < 2; d++) {
    CK(cuMulticastUnbind(g->mc, d, 0, g->size));
    CK(cuMemUnmap(g->vuc[d], g->size));
    CK(cuMemRelease(g->uc[d]));
  }
  CK(cuMemRelease(g->mc));
}

/* Like make_group, but bound by address: device 0's buffer is the second half
 * of a larger mapping (cuMulticastBindAddr), device 1's a whole mapping
 * (cuMulticastBindAddr_v2). */
static void make_group_addr(Group* g) {
  g->size = mc_size();
  McProp p = {2, g->size, 0, 0};
  use(0);
  CK(cuMulticastCreate(&g->mc, &p));
  CK(cuMulticastAddDevice(g->mc, 0));
  CK(cuMulticastAddDevice(g->mc, 1));
  CUresult (*bind_addr_v2)(H, CUdevice, size_t, CUdeviceptr, size_t,
                           unsigned long long) =
      gpa("cuMulticastBindAddr", 13010, 1);
  for (int d = 0; d < 2; d++) {
    use(d);
    size_t sz = d == 0 ? 2 * g->size : g->size;
    AllocProp up = uc_prop(d, 0);
    CK(cuMemCreate(&g->uc[d], sz, &up, 0));
    CUdeviceptr va = map(g->uc[d], sz, d, d == 0 ? 2 : 1);
    g->vuc[d] = d == 0 ? va + g->size : va;
    if (d == 0)
      CK(cuMulticastBindAddr(g->mc, 0, g->vuc[0], g->size, 0));
    else
      CK(bind_addr_v2(g->mc, 1, 0, g->vuc[1], g->size, 0));
  }
  use(0);
  g->vmc = map(g->mc, g->size, 0, 1);
}

static void destroy_group_addr(Group* g) {
  use(0);
  CK(cuMemUnmap(g->vmc, g->size));
  for (int d = 0; d < 2; d++) CK(cuMulticastUnbind(g->mc, d, 0, g->size));
  CK(cuMemUnmap(g->vuc[0] - g->size, 2 * g->size));
  CK(cuMemUnmap(g->vuc[1], g->size));
  for (int d = 0; d < 2; d++) CK(cuMemRelease(g->uc[d]));
  CK(cuMemRelease(g->mc));
}

static int t_mc(void) {
  setup(2);
  pid_t me = getpid();
  /* Two groups, so that the rebuild can hand each the other's old handle. */
  Group a, b, c;
  make_group(&a, 0);
  make_group(&b, 0);
  make_group_addr(&c);
  check_group(&a, 0x1111, "before");
  check_group(&b, 0x2222, "before");
  check_group(&c, 0x6666, "before (by address)");

  /* A retained handle is the application's value; releasing it keeps the
   * object. */
  H r;
  CK(cuMemRetainAllocationHandle(&r, (void*)(uintptr_t)a.vuc[0]));
  EXPECT(r == a.uc[0], "retain returned 0x%llx, want 0x%llx", r, a.uc[0]);
  CK(cuMemRelease(r));

  CUresult (*real_retain)(H*, void*);
  *(void**)&real_retain = rdlsym(lib, "cuMemRetainAllocationHandle");
  H ra, rb;
  CK(real_retain(&ra, (void*)(uintptr_t)a.vmc));
  CK(real_retain(&rb, (void*)(uintptr_t)b.vmc));
  printf("mc: before: app a=0x%llx b=0x%llx real a=0x%llx b=0x%llx\n", a.mc,
         b.mc, ra, rb);
  CUresult (*real_release)(H);
  *(void**)&real_release = rdlsym(lib, "cuMemRelease");
  CK(real_release(ra));
  CK(real_release(rb));
  EXPECT(cycle(&me, 1) == 0, "suspend/resume failed");
  CK(real_retain(&ra, (void*)(uintptr_t)a.vmc));
  CK(real_retain(&rb, (void*)(uintptr_t)b.vmc));
  printf("mc: after: app a=0x%llx b=0x%llx real a=0x%llx b=0x%llx\n", a.mc,
         b.mc, ra, rb);
  CK(real_release(ra));
  CK(real_release(rb));
  check_group(&a, 0x3333, "after");
  check_group(&b, 0x4444, "after");
  check_group(&c, 0x7777, "after (by address)");
  destroy_group_addr(&c);

  /* After the rebuild the application still uses its original values. */
  CK(cuMemRetainAllocationHandle(&r, (void*)(uintptr_t)a.vmc));
  EXPECT(r == a.mc, "retain(mc) returned 0x%llx, want 0x%llx", r, a.mc);
  CK(cuMemRelease(r));
  AllocProp got;
  CK(cuMemGetAllocationPropertiesFromHandle(&got, b.uc[1]));
  EXPECT(got.loc.id == 1, "properties of b.uc[1]: device %d", got.loc.id);

  /* New objects never alias a live application value, even when the driver
   * reissues a value the rebuild freed. */
  H fresh[32];
  int collisions = 0;
  use(0);
  AllocProp up = uc_prop(0, 0);
  for (int i = 0; i < 32; i++) {
    CK(cuMemCreate(&fresh[i], a.size, &up, 0));
    collisions += (fresh[i] >> 56) == 0xdc;
    EXPECT(fresh[i] != a.mc && fresh[i] != b.mc,
           "fresh handle aliases a group");
    for (int j = 0; j < i; j++)
      EXPECT(fresh[i] != fresh[j], "fresh handles alias");
  }
  McProp mp = {2, a.size, 0, 0};
  H extra;
  CK(cuMulticastCreate(&extra, &mp));
  collisions += (extra >> 56) == 0xdc;
  EXPECT(extra != a.mc && extra != b.mc, "fresh group aliases a group");
  for (int i = 0; i < 32; i++) CK(cuMemRelease(fresh[i]));
  CK(cuMemRelease(extra));
  printf("mc: synthetic handles issued: %d\n", collisions);

  /* Stale values still route to their own objects. */
  destroy_group(&a);
  check_group(&b, 0x5555, "after destroying a");
  destroy_group(&b);

  /* And a second cycle with nothing left to rebuild. */
  EXPECT(cycle(&me, 1) == 0, "second suspend/resume failed");
  printf("mc: ok\n");
  return g_failed;
}

/* refcount */

static int t_refcount(void) {
  setup(2);
  pid_t me = getpid();
  size_t free0, total;
  use(0);
  CK(cuMemGetInfo_v2(&free0, &total));
  Group g, g2;
  make_group(&g, 0);
  /* g2's group holds two references. */
  make_group(&g2, 0);
  H r3;
  CK(cuMemRetainAllocationHandle(&r3, (void*)(uintptr_t)g2.vmc));
  EXPECT(r3 == g2.mc, "retain(g2.mc) 0x%llx, want 0x%llx", r3, g2.mc);
  /* Keep only mappings: the group's and device 1's buffer are held by no
   * handle at all; device 0's buffer by three references. */
  H r1, r2;
  CK(cuMemRetainAllocationHandle(&r1, (void*)(uintptr_t)g.vuc[0]));
  CK(cuMemRetainAllocationHandle(&r2, (void*)(uintptr_t)g.vuc[0]));
  CK(cuMemRelease(g.mc));
  CK(cuMemRelease(g.uc[1]));
  check_group(&g, 0xa1, "before");

  EXPECT(cycle(&me, 1) == 0, "suspend/resume failed");
  check_group(&g, 0xa2, "after");
  check_group(&g2, 0xa3, "after");
  CK(cuMemRelease(r3));
  destroy_group(&g2);

  /* Tear down through retained handles; every reference must balance. */
  H mc, u1;
  CK(cuMemRetainAllocationHandle(&mc, (void*)(uintptr_t)g.vmc));
  CK(cuMemRetainAllocationHandle(&u1, (void*)(uintptr_t)g.vuc[1]));
  EXPECT(mc == g.mc, "retain(mc) 0x%llx, want 0x%llx", mc, g.mc);
  use(0);
  CK(cuMemUnmap(g.vmc, g.size));
  for (int d = 0; d < 2; d++) {
    CK(cuMulticastUnbind(mc, d, 0, g.size));
    CK(cuMemUnmap(g.vuc[d], g.size));
  }
  CK(cuMemRelease(mc));
  CK(cuMemRelease(u1));
  CK(cuMemRelease(g.uc[0]));
  CK(cuMemRelease(r1));
  CK(cuMemRelease(r2));
  size_t free1;
  use(0);
  CK(cuMemGetInfo_v2(&free1, &total));
  EXPECT(free1 + (1 << 20) >= free0, "leaked %zu bytes on device 0",
         free0 - free1);
  printf("refcount: ok\n");
  return g_failed;
}

/* refuse */

/* Memory the shim never saw: created through the driver's own cuMemCreate. */
static H untracked_alloc(int d, size_t size) {
  CUresult (*real)(H*, size_t, const AllocProp*, unsigned long long);
  *(void**)&real = rdlsym(lib, "cuMemCreate");
  AllocProp up = uc_prop(d, 0);
  H h;
  CK(real(&h, size, &up, 0));
  return h;
}

static H plain_group(size_t size) {
  McProp p = {2, size, 0, 0};
  H mc;
  use(0);
  CK(cuMulticastCreate(&mc, &p));
  CK(cuMulticastAddDevice(mc, 0));
  CK(cuMulticastAddDevice(mc, 1));
  return mc;
}

/* One refusal case, in a fresh process. Returns 0 if the gate accepted
 * ("clean") or refused (the rest) as expected, and the application kept
 * running. */
static int refuse_case(const char* name) {
  int multicast =
      strcmp(name, "clean") && strcmp(name, "pool") && strcmp(name, "managed");
  setup(multicast ? 2 : 1);
  pid_t me = getpid();
  if (!strcmp(name, "pool")) {
    PoolProp pp;
    memset(&pp, 0, sizeof(pp));
    pp.allocType = 1;
    pp.handleTypes = POSIX_FD;
    pp.loc.type = DEVICE;
    CUmemoryPool pool;
    CK(cuMemPoolCreate(&pool, &pp));
    int fd = -1;
    CK(cuMemPoolExportToShareableHandle(&fd, pool, POSIX_FD, 0));
  } else if (!strcmp(name, "managed")) {
    CUdeviceptr m;
    CK(cuMemAllocManaged(&m, 1 << 20, 1 /* GLOBAL */));
  } else if (!strcmp(name, "untracked-bind")) {
    size_t size = mc_size();
    H mc = plain_group(size);
    CK(cuMulticastBindMem(mc, 0, untracked_alloc(0, size), 0, size, 0));
  } else if (!strcmp(name, "span")) {
    /* Two allocations mapped back to back, bound by address as one range. */
    size_t size = mc_size();
    H mc = plain_group(2 * size);
    AllocProp up = uc_prop(0, 0);
    H u[2];
    CUdeviceptr va;
    CK(cuMemAddressReserve(&va, 2 * size, 0, 0, 0));
    for (int k = 0; k < 2; k++) {
      CK(cuMemCreate(&u[k], size, &up, 0));
      CK(cuMemMap(va + k * size, size, 0, u[k], 0));
    }
    Access acc = {{DEVICE, 0}, RW};
    CK(cuMemSetAccess(va, 2 * size, &acc, 1));
    CK(cuMulticastBindAddr(mc, 0, va, 2 * size, 0));
  } else if (strcmp(name, "clean")) {
    fprintf(stderr, "unknown refusal case %s\n", name);
    return 2;
  }
  int refused = gate_up(&me, 1) != 0;
  gate_down();
  if (!strcmp(name, "clean"))
    EXPECT(!refused && cycle(&me, 1) == 0, "clean process refused");
  else
    EXPECT(refused, "gate accepted %s", name);
  /* A refusal leaves the application running. */
  CUdeviceptr p;
  use(0);
  CK(cuMemAlloc_v2(&p, 4096));
  CK(cuMemsetD32_v2(p, 1, 1024));
  CK(cuCtxSynchronize());
  return g_failed;
}

static int t_refuse(void) {
  static const char* const cases[] = {"clean", "pool", "managed",
                                      "untracked-bind", "span"};
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
    pid_t pid = fork();
    if (pid == 0) {
      execl("/proc/self/exe", "mcshim_test", "refuse1", cases[i], (char*)NULL);
      _exit(127);
    }
    int st;
    waitpid(pid, &st, 0);
    int rc = WIFEXITED(st) ? WEXITSTATUS(st) : 128;
    if (rc == 77) {
      printf("refuse %s: skipped\n", cases[i]);
      continue;
    }
    EXPECT(rc == 0, "case %s exited 0x%x", cases[i], st);
  }
  printf("refuse: %s\n", g_failed ? "FAILED" : "ok");
  return g_failed;
}

/* ipc */

static int send_fds(int s, const int* fds, int n) {
  char c = 0;
  struct iovec iov = {&c, 1};
  char buf[CMSG_SPACE(8 * sizeof(int))];
  struct msghdr m = {0};
  m.msg_iov = &iov;
  m.msg_iovlen = 1;
  m.msg_control = buf;
  m.msg_controllen = CMSG_SPACE(n * sizeof(int));
  struct cmsghdr* cm = CMSG_FIRSTHDR(&m);
  cm->cmsg_level = SOL_SOCKET;
  cm->cmsg_type = SCM_RIGHTS;
  cm->cmsg_len = CMSG_LEN(n * sizeof(int));
  memcpy(CMSG_DATA(cm), fds, n * sizeof(int));
  return sendmsg(s, &m, 0) == 1 ? 0 : -1;
}

static int recv_fds(int s, int* fds, int n) {
  char c;
  struct iovec iov = {&c, 1};
  char buf[CMSG_SPACE(8 * sizeof(int))];
  struct msghdr m = {0};
  m.msg_iov = &iov;
  m.msg_iovlen = 1;
  m.msg_control = buf;
  m.msg_controllen = sizeof(buf);
  if (recvmsg(s, &m, 0) != 1) return -1;
  struct cmsghdr* cm = CMSG_FIRSTHDR(&m);
  if (!cm) return -1;
  memcpy(fds, CMSG_DATA(cm), n * sizeof(int));
  return 0;
}

static void sync_byte(int s, char out) {
  char in;
  if (write(s, &out, 1) != 1 || read(s, &in, 1) != 1) exit(3);
}

/* Rank d of two. Rank 0 creates groups X and Y and a peer buffer P and
 * exports them; rank 1 imports Y before X, so the two ranks hold the groups in
 * opposite table order, and maps P. */
static int ipc_rank(int d, int peer, int ctl) {
  setup(2);
  size_t size = mc_size();
  use(d);
  H grp[2], uc[2], pbuf;
  CUdeviceptr vmc[2], vuc[2], vp;
  McProp mp = {2, size, POSIX_FD, 0};
  if (d == 0) {
    int fds[3];
    for (int k = 0; k < 2; k++) {
      CK(cuMulticastCreate(&grp[k], &mp));
      CK(cuMemExportToShareableHandle(&fds[k], grp[k], POSIX_FD, 0));
    }
    AllocProp up = uc_prop(0, POSIX_FD);
    CK(cuMemCreate(&pbuf, size, &up, 0));
    vp = map(pbuf, size, 0, 2);
    CK(cuMemsetD32_v2(vp, 0xbeef, size / 4));
    CK(cuMemExportToShareableHandle(&fds[2], pbuf, POSIX_FD, 0));
    if (send_fds(peer, fds, 3) != 0) exit(3);
    for (int k = 0; k < 3; k++) close(fds[k]);
  } else {
    int fds[3];
    if (recv_fds(peer, fds, 3) != 0) exit(3);
    CK(cuMemImportFromShareableHandle(&grp[1], (void*)(intptr_t)fds[1],
                                      POSIX_FD));
    CK(cuMemImportFromShareableHandle(&grp[0], (void*)(intptr_t)fds[0],
                                      POSIX_FD));
    CK(cuMemImportFromShareableHandle(&pbuf, (void*)(intptr_t)fds[2],
                                      POSIX_FD));
    for (int k = 0; k < 3; k++) close(fds[k]);
    /* Map the peer buffer, then drop the import handle: only the mapping
     * keeps it. */
    vp = map(pbuf, size, 1, 1);
    CK(cuMemRelease(pbuf));
  }
  /* Binds wait for every device, so add ours to both groups first. */
  int order[2] = {d, 1 - d};
  for (int i = 0; i < 2; i++) CK(cuMulticastAddDevice(grp[order[i]], d));
  for (int i = 0; i < 2; i++) {
    int k = order[i];
    AllocProp up = uc_prop(d, POSIX_FD);
    CK(cuMemCreate(&uc[k], size, &up, 0));
    CK(cuMulticastBindMem(grp[k], 0, uc[k], 0, size, 0));
    vuc[k] = map(uc[k], size, d, 1);
    vmc[k] = map(grp[k], size, d, 1);
  }
  for (int round = 0; round < 2; round++) {
    /* round 0 before the cycle, round 1 after it. */
    sync_byte(ctl, 'r');
    if (d == 0) {
      for (int k = 0; k < 2; k++) {
        unsigned v = 0x100 * (round + 1) + k;
        void* args[] = {&vmc[k], &v};
        CK(launch1(g_bcast, args, NULL));
      }
      CK(cuCtxSynchronize());
    }
    sync_byte(peer, 'b');
    for (int k = 0; k < 2; k++) {
      unsigned want = 0x100 * (round + 1) + k;
      EXPECT(read32(d, vuc[k]) == want, "rank %d round %d group %d: 0x%x", d,
             round, k, read32(d, vuc[k]));
    }
    EXPECT(read32(d, vp) == 0xbeef, "rank %d round %d: peer buffer 0x%x", d,
           round, read32(d, vp));
  }
  /* The exporter frees the peer buffer that rank 1 still maps: nobody would
   * republish it, so rank 1 must now refuse the gate. */
  if (d == 0) {
    CK(cuMemUnmap(vp, size));
    CK(cuMemRelease(pbuf));
  }
  sync_byte(ctl, 'f');
  sync_byte(ctl, 'd');
  return g_failed;
}

static int t_ipc(void) {
  int peer[2], ctl0[2], ctl1[2];
  socketpair(AF_UNIX, SOCK_STREAM, 0, peer);
  socketpair(AF_UNIX, SOCK_STREAM, 0, ctl0);
  socketpair(AF_UNIX, SOCK_STREAM, 0, ctl1);
  pid_t pids[2];
  for (int d = 0; d < 2; d++) {
    pids[d] = fork();
    if (pids[d] == 0) exit(ipc_rank(d, peer[d], d ? ctl1[1] : ctl0[1]));
  }
  int ctl[2] = {ctl0[0], ctl1[0]};
  char c;
  for (int round = 0; round < 2; round++) {
    for (int d = 0; d < 2; d++)
      if (read(ctl[d], &c, 1) != 1) return 1;
    if (round == 1) EXPECT(cycle(pids, 2) == 0, "suspend/resume failed");
    for (int d = 0; d < 2; d++)
      if (write(ctl[d], "g", 1) != 1) return 1;
  }
  for (int d = 0; d < 2; d++)
    if (read(ctl[d], &c, 1) != 1) return 1;
  char e1[64];
  snprintf(e1, sizeof(e1), "error.%d", (int)pids[1]);
  EXPECT(gate_up(pids, 2) != 0 && exists(e1),
         "rank 1 accepted the gate with an import nobody will republish");
  gate_down();
  for (int d = 0; d < 2; d++)
    if (write(ctl[d], "g", 1) != 1) return 1;
  for (int d = 0; d < 2; d++) {
    if (read(ctl[d], &c, 1) != 1 || write(ctl[d], "g", 1) != 1) return 1;
  }
  for (int d = 0; d < 2; d++) {
    int st;
    waitpid(pids[d], &st, 0);
    EXPECT(WIFEXITED(st) && WEXITSTATUS(st) == 0, "rank %d exited 0x%x", d, st);
  }
  printf("ipc: %s\n", g_failed ? "FAILED" : "ok");
  return g_failed;
}

int main(int argc, char** argv) {
  if (argc == 3 && !strcmp(argv[1], "refuse1")) {
    clear_markers();
    return refuse_case(argv[2]) ? 1 : 0;
  }
  if (argc != 2) {
    fprintf(stderr, "usage: %s abi|gate|mc|refcount|refuse|ipc\n", argv[0]);
    return 2;
  }
  clear_markers();
  static const struct {
    const char* name;
    int (*fn)(void);
  } tests[] = {{"abi", t_abi},           {"gate", t_gate},     {"mc", t_mc},
               {"refcount", t_refcount}, {"refuse", t_refuse}, {"ipc", t_ipc}};
  for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); i++)
    if (strcmp(argv[1], tests[i].name) == 0) {
      int rc = tests[i].fn();
      clear_markers();
      return rc ? 1 : 0;
    }
  fprintf(stderr, "unknown test %s\n", argv[1]);
  return 2;
}
