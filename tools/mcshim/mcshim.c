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

/* mcshim: libcuda-level multicast suspend/resume interposer (see README.md).
 *
 * An LD_PRELOAD shim that tracks multicast groups, VMM allocations, exports,
 * imports, binds and mappings, and on request tears them down and rebuilds them
 * at identical virtual addresses, for any owner (NCCL NVLS, torch
 * _symmetric_memory, raw cuMulticast). It works in-process because libcuda
 * keeps its own bookkeeping: freeing the RM objects from nvproxy leaves libcuda
 * inconsistent, and the restore fails.
 *
 * The sentry drives it through marker files in /tmp/mcshim (see
 * pkg/sentry/control/state_cuda_shim.go). After a rebuild, exporters publish
 * the re-exported fd under the original export's identity, and importers copy
 * it with pidfd_getfd(2) and re-import it. Unicast device memory stays
 * cuda-checkpoint's responsibility, and so does legacy CUDA IPC (cuIpc*),
 * which cuda-checkpoint carries when the processes share a job (runsc
 * --cuda-checkpoint-path).
 *
 * The application only ever sees the handle values it was given: a rebuild
 * gives objects new driver handles, and every handle-taking entry point
 * translates. Besides symbol interposition, the shim interposes dlsym,
 * cuGetProcAddress and cudart's cudaGetDriverEntryPoint* resolvers, and
 * redirects by the address they return, so each ABI version of an entry point
 * gets the wrapper with its own signature. */

#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434
#endif
#ifndef SYS_pidfd_getfd
#define SYS_pidfd_getfd 438
#endif

/* Minimal CUDA driver ABI (x86_64), mirrored from cuda.h. */

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUstream;
typedef void* CUfunction;
typedef void* CUgraphExec;
typedef void* CUhostFn;
typedef void* CUarray;
typedef void* CUmemoryPool;
typedef unsigned int cuuint32_t;
typedef unsigned long long cuuint64_t;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUmemGenericAllocationHandle;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_ERROR_NOT_INITIALIZED 3
/* Returned by an import when the process cannot address the exporting device.
 */
#define CUDA_ERROR_INVALID_DEVICE 101
#define CUDA_ERROR_NOT_FOUND 500
#define CUDA_ERROR_NOT_SUPPORTED 801

#define CU_MEM_HANDLE_TYPE_POSIX_FD 0x1
#define CU_MEM_HANDLE_TYPE_FABRIC 0x8
#define CU_MEM_LOCATION_TYPE_DEVICE 1
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE 3
#define CU_MEM_OPERATION_TYPE_MAP 1
#define CU_MEM_HANDLE_TYPE_GENERIC 0
#define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED 128

typedef struct {
  int type;
  int id;
} CUmemLocation;

typedef struct {
  unsigned char compressionType;
  unsigned char gpuDirectRDMACapable;
  unsigned short usage;
  unsigned char reserved[4];
} CUmemAllocFlags;

typedef struct {
  int type;
  int requestedHandleTypes;
  CUmemLocation location;
  void* win32HandleMetaData;
  CUmemAllocFlags allocFlags;
} CUmemAllocationProp;

typedef struct {
  CUmemLocation location;
  int flags;
} CUmemAccessDesc;

typedef struct {
  unsigned int numDevices;
  size_t size;
  unsigned long long handleTypes;
  unsigned long long flags;
} CUmulticastObjectProp;

typedef struct {
  int resourceType;
  union {
    void* mipmap;
    CUarray array;
  } resource;
  int subresourceType;
  union {
    struct {
      unsigned level, layer, offsetX, offsetY, offsetZ;
      unsigned extentWidth, extentHeight, extentDepth;
    } sparseLevel;
    struct {
      unsigned layer;
      unsigned long long offset, size;
    } miptail;
  } subresource;
  int memOperationType;
  int memHandleType;
  union {
    CUmemGenericAllocationHandle memHandle;
  } memHandle;
  unsigned long long offset;
  unsigned int deviceBitMask, flags, reserved[2];
} CUarrayMapInfo;
_Static_assert(sizeof(CUarrayMapInfo) == 96, "CUarrayMapInfo layout");

/* Logging. */

static FILE* g_log;
static pthread_mutex_t g_loglock = PTHREAD_MUTEX_INITIALIZER;

static void mclog(const char* fmt, ...) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  struct tm tm;
  localtime_r(&ts.tv_sec, &tm);
  char t[32], line[1024];
  strftime(t, sizeof(t), "%H:%M:%S", &tm);
  int n = snprintf(line, sizeof(line), "[mcshim %s.%03ld pid=%d] ", t,
                   ts.tv_nsec / 1000000, (int)getpid());
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(line + n, sizeof(line) - n, fmt, ap);
  va_end(ap);
  pthread_mutex_lock(&g_loglock);
  if (!g_log) {
    const char* p = getenv("MCSHIM_LOG");
    g_log = p && *p ? fopen(p, "a") : stderr;
    if (!g_log) g_log = stderr;
  }
  /* One write per line: ranks share the log file. */
  fprintf(g_log, "%s\n", line);
  fflush(g_log);
  pthread_mutex_unlock(&g_loglock);
}

static int allow_fabric(void) { return getenv("MCSHIM_ALLOW_FABRIC") != NULL; }

/* The real dlsym, resolved via dlvsym (not interposed) so that the dlsym
 * wrapper can delegate without recursing. */

static void* (*real_dlsym)(void*, const char*);

static void init_real_dlsym(void) {
  if (real_dlsym) return;
  /* glibc >= 2.34 exports dlsym as GLIBC_2.34, older glibc as GLIBC_2.2.5. */
  *(void**)(&real_dlsym) = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
  if (!real_dlsym)
    *(void**)(&real_dlsym) = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
  if (!real_dlsym) mclog("FATAL: could not resolve real dlsym via dlvsym");
}

/* Resolve reals against an explicit libcuda.so.1 handle, not RTLD_NEXT:
 * consumers often dlopen libcuda RTLD_LOCAL. Uses real_dlsym, since the
 * interposed dlsym would return our own wrappers. */

static void* libcuda_handle(void) {
  static void* h;
  if (!h) h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
  if (!h) h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!h) mclog("FATAL: dlopen(libcuda.so.1) failed: %s", dlerror());
  return h;
}

/* Whether libcuda is loaded, without loading it. */
static int libcuda_loaded(void) {
  static int loaded;
  if (__atomic_load_n(&loaded, __ATOMIC_ACQUIRE)) return 1;
  void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
  if (!h) return 0;
  dlclose(h);
  __atomic_store_n(&loaded, 1, __ATOMIC_RELEASE);
  return 1;
}

#define REAL(var, name)                                               \
  do {                                                                \
    if (!(var)) {                                                     \
      init_real_dlsym();                                              \
      void* h_ = libcuda_handle();                                    \
      if (real_dlsym && h_) *(void**)(&(var)) = real_dlsym(h_, name); \
    }                                                                 \
  } while (0)

/* Reals the shim calls but does not interpose. */
static CUresult (*r_cuCtxGetDevice)(CUdevice*);
static CUresult (*r_cuCtxGetCurrent)(CUcontext*);
static CUresult (*r_cuCtxSetCurrent)(CUcontext);
static CUresult (*r_cuCtxSynchronize)(void);
static CUresult (*r_cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
static CUresult (*r_cuDevicePrimaryCtxRelease)(CUdevice);
static CUresult (*r_cuDevicePrimaryCtxGetState)(CUdevice, unsigned int*, int*);

static void resolve_internal(void) {
  REAL(r_cuCtxGetDevice, "cuCtxGetDevice");
  REAL(r_cuCtxGetCurrent, "cuCtxGetCurrent");
  REAL(r_cuCtxSetCurrent, "cuCtxSetCurrent");
  REAL(r_cuCtxSynchronize, "cuCtxSynchronize");
  REAL(r_cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
  REAL(r_cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2");
  REAL(r_cuDevicePrimaryCtxGetState, "cuDevicePrimaryCtxGetState");
}

/* Suspend gate. While suspended, multicast groups and imports are released and
 * their VAs unmapped: an app thread touching the GPU then faults its context
 * (700), and through a shared group every rank. cuda-checkpoint --toggle
 * restores and unlocks the app before the rebuild, so the entry points that
 * submit GPU work or change tracked state block until it completes. The shim's
 * own work calls the reals.
 *
 * Calls that enter through gate_enter are counted until they return, and
 * gate_arm waits for the count to drain: once it returns, no app thread is
 * inside a tracked call, so none can finish one over the teardown. */

static pthread_mutex_t g_gate_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_gate_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_drain_cv = PTHREAD_COND_INITIALIZER;
static int g_suspended; /* atomic */
static int g_inflight;  /* atomic */

#define GATE_DRAIN_SECS 30

static void gate_wait(void) {
  /* fast path: one load on every GPU submission */
  if (!__atomic_load_n(&g_suspended, __ATOMIC_SEQ_CST)) return;
  static __thread int logged;
  pthread_mutex_lock(&g_gate_lock);
  if (g_suspended && !logged) {
    logged = 1;
    mclog("GATE: app thread blocked until resume");
  }
  while (g_suspended) pthread_cond_wait(&g_gate_cv, &g_gate_lock);
  pthread_mutex_unlock(&g_gate_lock);
}

static void gate_exit(void) {
  if (__atomic_sub_fetch(&g_inflight, 1, __ATOMIC_SEQ_CST) == 0 &&
      __atomic_load_n(&g_suspended, __ATOMIC_SEQ_CST)) {
    pthread_mutex_lock(&g_gate_lock);
    pthread_cond_broadcast(&g_drain_cv);
    pthread_mutex_unlock(&g_gate_lock);
  }
}

/* Paired with gate_arm (Dekker): either the arming thread sees the count, or
 * this thread sees the gate. */
static void gate_enter(void) {
  for (;;) {
    gate_wait();
    __atomic_add_fetch(&g_inflight, 1, __ATOMIC_SEQ_CST);
    if (!__atomic_load_n(&g_suspended, __ATOMIC_SEQ_CST)) return;
    gate_exit();
  }
}

/* Arm and drain. Returns -1 if calls are still in flight after
 * GATE_DRAIN_SECS; the gate stays armed. Must not hold g_lock: a draining
 * call may need it. */
static int gate_arm(void) {
  struct timespec dl;
  clock_gettime(CLOCK_REALTIME, &dl);
  dl.tv_sec += GATE_DRAIN_SECS;
  int rc = 0;
  pthread_mutex_lock(&g_gate_lock);
  __atomic_store_n(&g_suspended, 1, __ATOMIC_SEQ_CST);
  while (__atomic_load_n(&g_inflight, __ATOMIC_SEQ_CST) > 0) {
    if (pthread_cond_timedwait(&g_drain_cv, &g_gate_lock, &dl) == ETIMEDOUT &&
        __atomic_load_n(&g_inflight, __ATOMIC_SEQ_CST) > 0) {
      rc = -1;
      break;
    }
  }
  pthread_mutex_unlock(&g_gate_lock);
  return rc;
}

static void gate_disarm(void) {
  pthread_mutex_lock(&g_gate_lock);
  __atomic_store_n(&g_suspended, 0, __ATOMIC_SEQ_CST);
  pthread_cond_broadcast(&g_gate_cv);
  pthread_mutex_unlock(&g_gate_lock);
}

/* Entry points that submit GPU work: counted and gated. Each entry is an
 * exported symbol and its per-thread-stream twin, which share a prototype
 * (cudaTypedefs.h, CUDA 13.4); every exported name is its own ABI. */
#define GATED_LIST(X)                                                         \
  X(cuLaunchKernel, _ptsz,                                                    \
    (CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,        \
     unsigned by, unsigned bz, unsigned shmem, CUstream st, void** kp,        \
     void** extra),                                                           \
    (f, gx, gy, gz, bx, by, bz, shmem, st, kp, extra))                        \
  X(cuLaunchKernelEx, _ptsz,                                                  \
    (const void* cfg, CUfunction f, void** kp, void** extra),                 \
    (cfg, f, kp, extra))                                                      \
  X(cuLaunchCooperativeKernel, _ptsz,                                         \
    (CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,        \
     unsigned by, unsigned bz, unsigned shmem, CUstream st, void** kp),       \
    (f, gx, gy, gz, bx, by, bz, shmem, st, kp))                               \
  X(cuLaunchHostFunc, _ptsz, (CUstream st, CUhostFn fn, void* ud),            \
    (st, fn, ud))                                                             \
  X(cuLaunchHostFunc_v2, _ptsz,                                               \
    (CUstream st, CUhostFn fn, void* ud, unsigned mode), (st, fn, ud, mode))  \
  X(cuGraphLaunch, _ptsz, (CUgraphExec g, CUstream st), (g, st))              \
  X(cuMemsetD8_v2, _ptds, (CUdeviceptr d, unsigned char v, size_t n),         \
    (d, v, n))                                                                \
  X(cuMemsetD16_v2, _ptds, (CUdeviceptr d, unsigned short v, size_t n),       \
    (d, v, n))                                                                \
  X(cuMemsetD32_v2, _ptds, (CUdeviceptr d, unsigned v, size_t n), (d, v, n))  \
  X(cuMemsetD8Async, _ptsz,                                                   \
    (CUdeviceptr d, unsigned char v, size_t n, CUstream st), (d, v, n, st))   \
  X(cuMemsetD16Async, _ptsz,                                                  \
    (CUdeviceptr d, unsigned short v, size_t n, CUstream st), (d, v, n, st))  \
  X(cuMemsetD32Async, _ptsz,                                                  \
    (CUdeviceptr d, unsigned v, size_t n, CUstream st), (d, v, n, st))        \
  X(cuMemsetD2D8_v2, _ptds,                                                   \
    (CUdeviceptr d, size_t p, unsigned char v, size_t w, size_t h),           \
    (d, p, v, w, h))                                                          \
  X(cuMemsetD2D16_v2, _ptds,                                                  \
    (CUdeviceptr d, size_t p, unsigned short v, size_t w, size_t h),          \
    (d, p, v, w, h))                                                          \
  X(cuMemsetD2D32_v2, _ptds,                                                  \
    (CUdeviceptr d, size_t p, unsigned v, size_t w, size_t h),                \
    (d, p, v, w, h))                                                          \
  X(cuMemsetD2D8Async, _ptsz,                                                 \
    (CUdeviceptr d, size_t p, unsigned char v, size_t w, size_t h,            \
     CUstream st),                                                            \
    (d, p, v, w, h, st))                                                      \
  X(cuMemsetD2D16Async, _ptsz,                                                \
    (CUdeviceptr d, size_t p, unsigned short v, size_t w, size_t h,           \
     CUstream st),                                                            \
    (d, p, v, w, h, st))                                                      \
  X(cuMemsetD2D32Async, _ptsz,                                                \
    (CUdeviceptr d, size_t p, unsigned v, size_t w, size_t h, CUstream st),   \
    (d, p, v, w, h, st))                                                      \
  X(cuMemcpy, _ptds, (CUdeviceptr dst, CUdeviceptr src, size_t n),            \
    (dst, src, n))                                                            \
  X(cuMemcpyAsync, _ptsz,                                                     \
    (CUdeviceptr dst, CUdeviceptr src, size_t n, CUstream st),                \
    (dst, src, n, st))                                                        \
  X(cuMemcpyPeer, _ptds,                                                      \
    (CUdeviceptr dst, CUcontext dc, CUdeviceptr src, CUcontext sc, size_t n), \
    (dst, dc, src, sc, n))                                                    \
  X(cuMemcpyPeerAsync, _ptsz,                                                 \
    (CUdeviceptr dst, CUcontext dc, CUdeviceptr src, CUcontext sc, size_t n,  \
     CUstream st),                                                            \
    (dst, dc, src, sc, n, st))                                                \
  X(cuMemcpyHtoD_v2, _ptds, (CUdeviceptr dst, const void* src, size_t n),     \
    (dst, src, n))                                                            \
  X(cuMemcpyDtoH_v2, _ptds, (void* dst, CUdeviceptr src, size_t n),           \
    (dst, src, n))                                                            \
  X(cuMemcpyDtoD_v2, _ptds, (CUdeviceptr dst, CUdeviceptr src, size_t n),     \
    (dst, src, n))                                                            \
  X(cuMemcpyHtoDAsync_v2, _ptsz,                                              \
    (CUdeviceptr dst, const void* src, size_t n, CUstream st),                \
    (dst, src, n, st))                                                        \
  X(cuMemcpyDtoHAsync_v2, _ptsz,                                              \
    (void* dst, CUdeviceptr src, size_t n, CUstream st), (dst, src, n, st))   \
  X(cuMemcpyDtoDAsync_v2, _ptsz,                                              \
    (CUdeviceptr dst, CUdeviceptr src, size_t n, CUstream st),                \
    (dst, src, n, st))                                                        \
  X(cuMemcpyAtoD_v2, _ptds,                                                   \
    (CUdeviceptr dst, CUarray src, size_t off, size_t n), (dst, src, off, n)) \
  X(cuMemcpyDtoA_v2, _ptds,                                                   \
    (CUarray dst, size_t off, CUdeviceptr src, size_t n), (dst, off, src, n)) \
  X(cuMemcpyAtoH_v2, _ptds, (void* dst, CUarray src, size_t off, size_t n),   \
    (dst, src, off, n))                                                       \
  X(cuMemcpyHtoA_v2, _ptds,                                                   \
    (CUarray dst, size_t off, const void* src, size_t n), (dst, off, src, n)) \
  X(cuMemcpyAtoA_v2, _ptds,                                                   \
    (CUarray dst, size_t doff, CUarray src, size_t soff, size_t n),           \
    (dst, doff, src, soff, n))                                                \
  X(cuMemcpyHtoAAsync_v2, _ptsz,                                              \
    (CUarray dst, size_t off, const void* src, size_t n, CUstream st),        \
    (dst, off, src, n, st))                                                   \
  X(cuMemcpyAtoHAsync_v2, _ptsz,                                              \
    (void* dst, CUarray src, size_t off, size_t n, CUstream st),              \
    (dst, src, off, n, st))                                                   \
  X(cuMemcpy2D_v2, _ptds, (const void* p), (p))                               \
  X(cuMemcpy2DUnaligned_v2, _ptds, (const void* p), (p))                      \
  X(cuMemcpy2DAsync_v2, _ptsz, (const void* p, CUstream st), (p, st))         \
  X(cuMemcpy3D_v2, _ptds, (const void* p), (p))                               \
  X(cuMemcpy3DAsync_v2, _ptsz, (const void* p, CUstream st), (p, st))         \
  X(cuMemcpy3DPeer, _ptds, (const void* p), (p))                              \
  X(cuMemcpy3DPeerAsync, _ptsz, (const void* p, CUstream st), (p, st))        \
  X(cuMemcpyBatchAsync, _ptsz,                                                \
    (CUdeviceptr * dsts, CUdeviceptr * srcs, size_t* sizes, size_t count,     \
     void* attrs, size_t* attrIdxs, size_t numAttrs, size_t* failIdx,         \
     CUstream st),                                                            \
    (dsts, srcs, sizes, count, attrs, attrIdxs, numAttrs, failIdx, st))       \
  X(cuMemcpyBatchAsync_v2, _ptsz,                                             \
    (CUdeviceptr * dsts, CUdeviceptr * srcs, size_t* sizes, size_t count,     \
     void* attrs, size_t* attrIdxs, size_t numAttrs, CUstream st),            \
    (dsts, srcs, sizes, count, attrs, attrIdxs, numAttrs, st))                \
  X(cuMemcpy3DBatchAsync, _ptsz,                                              \
    (size_t n, void* ops, size_t* failIdx, unsigned long long fl,             \
     CUstream st),                                                            \
    (n, ops, failIdx, fl, st))                                                \
  X(cuMemcpy3DBatchAsync_v2, _ptsz,                                           \
    (size_t n, void* ops, unsigned long long fl, CUstream st),                \
    (n, ops, fl, st))                                                         \
  X(cuMemcpyWithAttributesAsync, _ptsz,                                       \
    (CUdeviceptr dst, CUdeviceptr src, size_t n, void* attr, CUstream st),    \
    (dst, src, n, attr, st))                                                  \
  X(cuMemcpy3DWithAttributesAsync, _ptsz,                                     \
    (void* op, unsigned long long fl, CUstream st), (op, fl, st))             \
  MEMOP(X, cuStreamWaitValue32, cuuint32_t)                                   \
  MEMOP(X, cuStreamWaitValue32_v2, cuuint32_t)                                \
  MEMOP(X, cuStreamWaitValue64, cuuint64_t)                                   \
  MEMOP(X, cuStreamWaitValue64_v2, cuuint64_t)                                \
  MEMOP(X, cuStreamWriteValue32, cuuint32_t)                                  \
  MEMOP(X, cuStreamWriteValue32_v2, cuuint32_t)                               \
  MEMOP(X, cuStreamWriteValue64, cuuint64_t)                                  \
  MEMOP(X, cuStreamWriteValue64_v2, cuuint64_t)                               \
  X(cuStreamBatchMemOp, _ptsz,                                                \
    (CUstream st, unsigned n, void* ops, unsigned fl), (st, n, ops, fl))      \
  X(cuStreamBatchMemOp_v2, _ptsz,                                             \
    (CUstream st, unsigned n, void* ops, unsigned fl), (st, n, ops, fl))

#define MEMOP(X, name, T) \
  X(name, _ptsz, (CUstream st, CUdeviceptr a, T v, unsigned fl), (st, a, v, fl))

/* Waits: blocked while suspended, but not counted, since a wait may depend on
 * a peer that is already gated. */
#define WAIT_LIST(X) X(cuStreamSynchronize, _ptsz, (CUstream st), (st))

#define GATED_ONE(name, proto, args)            \
  static CUresult(*r_##name) proto;             \
  CUresult name proto;                          \
  CUresult name proto {                         \
    REAL(r_##name, #name);                      \
    if (!r_##name) return CUDA_ERROR_NOT_FOUND; \
    gate_enter();                               \
    CUresult rc = r_##name args;                \
    gate_exit();                                \
    return rc;                                  \
  }

#define WAIT_ONE(name, proto, args)             \
  static CUresult(*r_##name) proto;             \
  CUresult name proto;                          \
  CUresult name proto {                         \
    REAL(r_##name, #name);                      \
    if (!r_##name) return CUDA_ERROR_NOT_FOUND; \
    gate_wait();                                \
    return r_##name args;                       \
  }

#define GATED_DEF(name, sfx, proto, args) \
  GATED_ONE(name, proto, args) GATED_ONE(name##sfx, proto, args)
#define WAIT_DEF(name, sfx, proto, args) \
  WAIT_ONE(name, proto, args) WAIT_ONE(name##sfx, proto, args)

GATED_LIST(GATED_DEF)
WAIT_LIST(WAIT_DEF)

/* Tracked state: the live object graph. */

/* Static tables (about 3 MB per process) keep hot paths allocation-free. An
 * SGLang TP=8 rank tracks more than 512 objects. */
#define MAXN 4096
#define MAX_DEV 16
#define MAX_ACCESS 16

/* KIND_IMP is an import; cuMulticastAddDevice or a bind on it proves it a
 * multicast group and makes it KIND_MC. */
enum { KIND_FREE = 0, KIND_UC = 1, KIND_MC = 2, KIND_IMP = 3 };

typedef struct {
  int kind;
  CUmemGenericAllocationHandle handle; /* driver handle; 0 while torn down */
  CUmemGenericAllocationHandle app;    /* the value the application holds */
  /* Application references: 1 for the create or import, +1 per retain, -1
   * per release. The object lives while referenced, mapped or bound. */
  int app_refs;
  int shim_ref; /* the shim holds one reference (resume, until phase 4) */
  CUdevice dev; /* device whose primary context issues calls on the object */
  size_t size;
  CUmemAllocationProp uprop;   /* KIND_UC */
  CUmulticastObjectProp mprop; /* KIND_MC */
  CUdevice devs[MAX_DEV];      /* KIND_MC: devices this process added */
  int ndev;
  int imported;
  /* Rendezvous identity of the export (see record_key). */
  int has_key;
  unsigned long key_client, key_object;
  int exp_marked; /* exp-<key>.<pid> exists (see mark_export) */
  /* After a resume: the re-exported fd, published until the gate is removed
   * (see publish_fd). */
  int pub_fd;
  /* Contents of a multicast-bound exporter freed across the checkpoint (see
   * do_suspend); NULL if none. */
  void* uc_content;
} Alloc;

typedef struct {
  int used;
  CUdeviceptr va;
  size_t size;
  size_t offset;
  int allocIdx;
  /* Access set applied to this mapping, merged by location. */
  CUmemAccessDesc access[MAX_ACCESS];
  int naccess;
  CUdevice dev;
} Mapping;

typedef struct {
  int used;
  int groupIdx;
  int v2;      /* bound through the _v2 entry point (explicit device) */
  int by_addr; /* cuMulticastBindAddr: replayed by VA */
  int memIdx;  /* BindMem: the bound allocation */
  CUdeviceptr va;
  size_t mcOffset;
  size_t memOffset;
  size_t size;
  CUdevice dev; /* device the binding applies to (unbind is per device) */
} Bind;

static Alloc g_alloc[MAXN];
static Mapping g_map[MAXN];
static Bind g_bind[MAXN];
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

/* Sticky: some state could not be tracked or cannot be carried, so arming the
 * gate refuses. Must hold g_lock. */
static int g_untracked;
static const char* g_untracked_why;

/* Sticky, lock-free (set from the dlsym interposer): a lookup of a tracked
 * entry point returned an ABI the shim has no wrapper for. */
static const char* g_bad_lookup;

/* Sticky: a suspend or resume failed partway, so this process stays gated.
 * Written by the control thread under g_lock; the control thread reads it
 * without. */
static int g_broken;

static void mark_untracked(const char* why) {
  if (!g_untracked) {
    g_untracked = 1;
    g_untracked_why = why;
    mclog("NOTE: checkpoint disabled for this process: %s", why);
  }
}

static CUdevice cur_dev(void) {
  CUdevice d = -1;
  if (!r_cuCtxGetDevice || r_cuCtxGetDevice(&d) != CUDA_SUCCESS) return -1;
  return d;
}

static int alloc_by_app(CUmemGenericAllocationHandle h) {
  for (int i = 0; i < MAXN; i++)
    if (g_alloc[i].kind != KIND_FREE && g_alloc[i].app == h) return i;
  return -1;
}

static int alloc_by_handle(CUmemGenericAllocationHandle h) {
  if (!h) return -1;
  for (int i = 0; i < MAXN; i++)
    if (g_alloc[i].kind != KIND_FREE && g_alloc[i].handle == h) return i;
  return -1;
}

/* Must hold g_lock. The driver handle for an application value. */
static CUmemGenericAllocationHandle xlate(CUmemGenericAllocationHandle h) {
  int i = alloc_by_app(h);
  return i >= 0 && g_alloc[i].handle ? g_alloc[i].handle : h;
}

/* Must hold g_lock. Track a new object with driver handle h and one app
 * reference. The application gets h, unless a live object's application value
 * already is h (the driver reuses values that a rebuild freed): then it gets a
 * synthetic value. Returns the slot, -1 if untracked (*app = h), or -2 if h
 * cannot be represented (the caller releases it and fails). */
static int track_new(int kind, CUmemGenericAllocationHandle h, CUdevice dev,
                     CUmemGenericAllocationHandle* app) {
  static unsigned long long synth;
  int i = 0;
  while (i < MAXN && g_alloc[i].kind != KIND_FREE) i++;
  if (i == MAXN) {
    mark_untracked("object table overflow");
    if (alloc_by_app(h) >= 0) return -2;
    *app = h;
    return -1;
  }
  CUmemGenericAllocationHandle v = h;
  while (alloc_by_app(v) >= 0) v = 0xdc00000000000000ULL | ++synth;
  Alloc* a = &g_alloc[i];
  memset(a, 0, sizeof(*a));
  a->kind = kind;
  a->handle = h;
  a->app = v;
  a->app_refs = 1;
  a->dev = dev;
  a->pub_fd = -1;
  *app = v;
  return i;
}

static int has_maps(int i) {
  for (int m = 0; m < MAXN; m++)
    if (g_map[m].used && g_map[m].allocIdx == i) return 1;
  return 0;
}

static int first_map(int i) {
  for (int m = 0; m < MAXN; m++)
    if (g_map[m].used && g_map[m].allocIdx == i) return m;
  return -1;
}

static int bound_as_mem(int i) {
  for (int b = 0; b < MAXN; b++)
    if (g_bind[b].used && !g_bind[b].by_addr && g_bind[b].memIdx == i) return 1;
  return 0;
}

/* Must hold g_lock. The mapping containing va, or -1. */
static int map_at(CUdeviceptr va) {
  for (int m = 0; m < MAXN; m++)
    if (g_map[m].used && va >= g_map[m].va && va < g_map[m].va + g_map[m].size)
      return m;
  return -1;
}

static void unpublish_fd(Alloc* a);
static void alloc_gc_all(void);
static void unmark_export(Alloc* a);

/* Must hold g_lock. Forget alloc i and the binds and maps that reference it. */
static void alloc_forget(int i) {
  int group = g_alloc[i].kind == KIND_MC || g_alloc[i].kind == KIND_IMP;
  for (int b = 0; b < MAXN; b++)
    if (g_bind[b].used && (g_bind[b].groupIdx == i ||
                           (!g_bind[b].by_addr && g_bind[b].memIdx == i)))
      g_bind[b].used = 0;
  for (int m = 0; m < MAXN; m++)
    if (g_map[m].used && g_map[m].allocIdx == i) g_map[m].used = 0;
  unpublish_fd(&g_alloc[i]);
  unmark_export(&g_alloc[i]);
  free(g_alloc[i].uc_content);
  memset(&g_alloc[i], 0, sizeof(g_alloc[i]));
  g_alloc[i].pub_fd = -1;
  /* Memory bound only through the group's binds is now dead too. */
  if (group) alloc_gc_all();
}

/* Must hold g_lock. Forget alloc i if nothing keeps it alive. */
static void alloc_gc(int i) {
  if (i < 0 || g_alloc[i].kind == KIND_FREE) return;
  if (g_alloc[i].app_refs > 0 || has_maps(i) || bound_as_mem(i)) return;
  alloc_forget(i);
}

static void alloc_gc_all(void) {
  for (int i = 0; i < MAXN; i++)
    if (g_alloc[i].kind != KIND_FREE && g_alloc[i].app_refs <= 0) alloc_gc(i);
}

/* nvproxy reports an exported RM object's identity in /proc/self/fdinfo/<fd>:
 *
 *   nvproxy_exported_object:\tclient=0x... object=0x... class=0x...
 *
 * The pair is unique, and the same for the exporter and every SCM_RIGHTS
 * recipient. Returns 0 on success. */
static int fdinfo_oracle(int fd, unsigned long* client, unsigned long* object) {
  char path[64], line[256];
  snprintf(path, sizeof(path), "/proc/self/fdinfo/%d", fd);
  FILE* f = fopen(path, "r");
  if (!f) return -1;
  int found = -1;
  while (fgets(line, sizeof(line), f)) {
    if (sscanf(line, "nvproxy_exported_object: client=%lx object=%lx", client,
               object) == 2) {
      found = 0;
      break;
    }
  }
  fclose(f);
  return found;
}

static void mark_export(Alloc* a);

/* Must hold g_lock. Record the rendezvous identity from nvproxy's fdinfo line;
 * without it the object cannot be rebuilt, so checkpoints are refused. The
 * identity is the exported object's, so every export of an object has the
 * same one. The first key wins: a rebuilt object's new identity only matters
 * to a later restore of a restored process, which is out of scope. */
static void record_key(int i, int fd) {
  Alloc* a = &g_alloc[i];
  if (a->has_key) return;
  if (fd < 0 || fdinfo_oracle(fd, &a->key_client, &a->key_object) != 0) {
    mark_untracked("export without an nvproxy identity");
    return;
  }
  a->has_key = 1;
  if (!a->imported) mark_export(a);
}

/* Interposed entry points. Mutators hold g_lock across the real call, so a
 * translation cannot go stale before it is used; binds, which block until
 * every device has joined, are the exception. */

static void ensure_control_thread(void);

static CUresult (*r_cuInit)(unsigned int);
static CUresult (*r_cuMemCreate)(CUmemGenericAllocationHandle*, size_t,
                                 const CUmemAllocationProp*,
                                 unsigned long long);
static CUresult (*r_cuMemRelease)(CUmemGenericAllocationHandle);
static CUresult (*r_cuMemMap)(CUdeviceptr, size_t, size_t,
                              CUmemGenericAllocationHandle, unsigned long long);
static CUresult (*r_cuMemUnmap)(CUdeviceptr, size_t);
static CUresult (*r_cuMemSetAccess)(CUdeviceptr, size_t, const CUmemAccessDesc*,
                                    size_t);
static CUresult (*r_cuMemRetainAllocationHandle)(CUmemGenericAllocationHandle*,
                                                 void*);
static CUresult (*r_cuMemGetAllocationPropertiesFromHandle)(
    CUmemAllocationProp*, CUmemGenericAllocationHandle);
static CUresult (*r_cuMulticastCreate)(CUmemGenericAllocationHandle*,
                                       const CUmulticastObjectProp*);
static CUresult (*r_cuMulticastAddDevice)(CUmemGenericAllocationHandle,
                                          CUdevice);
static CUresult (*r_cuMulticastBindMem)(CUmemGenericAllocationHandle, size_t,
                                        CUmemGenericAllocationHandle, size_t,
                                        size_t, unsigned long long);
static CUresult (*r_cuMulticastBindAddr)(CUmemGenericAllocationHandle, size_t,
                                         CUdeviceptr, size_t,
                                         unsigned long long);
static CUresult (*r_cuMulticastBindMem_v2)(CUmemGenericAllocationHandle,
                                           CUdevice, size_t,
                                           CUmemGenericAllocationHandle, size_t,
                                           size_t, unsigned long long);
static CUresult (*r_cuMulticastBindAddr_v2)(CUmemGenericAllocationHandle,
                                            CUdevice, size_t, CUdeviceptr,
                                            size_t, unsigned long long);
static CUresult (*r_cuMulticastUnbind)(CUmemGenericAllocationHandle, CUdevice,
                                       size_t, size_t);
static CUresult (*r_cuMemExportToShareableHandle)(void*,
                                                  CUmemGenericAllocationHandle,
                                                  int, unsigned long long);
static CUresult (*r_cuMemImportFromShareableHandle)(
    CUmemGenericAllocationHandle*, void*, int);
static CUresult (*r_cuDeviceGetAttribute)(int*, int, CUdevice);

CUresult cuInit(unsigned int flags) {
  REAL(r_cuInit, "cuInit");
  if (!r_cuInit) return CUDA_ERROR_NOT_INITIALIZED;
  /* Only processes that initialize CUDA take part in the protocol. */
  ensure_control_thread();
  return r_cuInit(flags);
}

/* Fabric handle types create an NV_MEMORY_FABRIC (00f8) object at allocation
 * time, which cuda-checkpoint cannot serialize. On a single node POSIX fds are
 * equivalent. Masking the device attribute is not enough, since statically
 * linked runtimes bypass it. */
static unsigned long long strip_fabric(unsigned long long types,
                                       const char* what) {
  if (!(types & CU_MEM_HANDLE_TYPE_FABRIC) || allow_fabric()) return types;
  unsigned long long fixed =
      types & ~(unsigned long long)CU_MEM_HANDLE_TYPE_FABRIC;
  if (!fixed) fixed = CU_MEM_HANDLE_TYPE_POSIX_FD;
  static int logged;
  if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
    mclog(
        "stripping CU_MEM_HANDLE_TYPE_FABRIC from %s (-> 0x%llx); set "
        "MCSHIM_ALLOW_FABRIC=1 to keep it",
        what, fixed);
  return fixed;
}

CUresult cuMemCreate(CUmemGenericAllocationHandle* h, size_t size,
                     const CUmemAllocationProp* prop,
                     unsigned long long flags) {
  REAL(r_cuMemCreate, "cuMemCreate");
  REAL(r_cuMemRelease, "cuMemRelease");
  resolve_internal();
  CUmemAllocationProp fixed;
  if (prop) {
    fixed = *prop;
    fixed.requestedHandleTypes =
        (int)strip_fabric((unsigned)prop->requestedHandleTypes, "cuMemCreate");
    prop = &fixed;
  }
  gate_enter();
  pthread_mutex_lock(&g_lock);
  CUresult rc = r_cuMemCreate(h, size, prop, flags);
  if (rc == CUDA_SUCCESS) {
    CUdevice dev = prop && prop->location.type == CU_MEM_LOCATION_TYPE_DEVICE
                       ? prop->location.id
                       : cur_dev();
    CUmemGenericAllocationHandle real = *h;
    int i = track_new(KIND_UC, real, dev, h);
    if (i >= 0) {
      g_alloc[i].size = size;
      if (prop) g_alloc[i].uprop = *prop;
    } else if (i == -2) {
      r_cuMemRelease(real);
      rc = CUDA_ERROR_OUT_OF_MEMORY;
    }
  }
  pthread_mutex_unlock(&g_lock);
  gate_exit();
  /* In case cuInit came through a path the shim does not see. */
  if (rc == CUDA_SUCCESS) ensure_control_thread();
  return rc;
}

CUresult cuMulticastCreate(CUmemGenericAllocationHandle* h,
                           const CUmulticastObjectProp* prop) {
  REAL(r_cuMulticastCreate, "cuMulticastCreate");
  REAL(r_cuMemRelease, "cuMemRelease");
  resolve_internal();
  CUmulticastObjectProp fixed;
  if (prop) {
    fixed = *prop;
    fixed.handleTypes = strip_fabric(prop->handleTypes, "cuMulticastCreate");
    prop = &fixed;
  }
  gate_enter();
  pthread_mutex_lock(&g_lock);
  CUresult rc = r_cuMulticastCreate(h, prop);
  if (rc == CUDA_SUCCESS) {
    CUmemGenericAllocationHandle real = *h;
    int i = track_new(KIND_MC, real, cur_dev(), h);
    if (i >= 0 && prop) {
      g_alloc[i].mprop = *prop;
      g_alloc[i].size = prop->size;
    } else if (i == -2) {
      r_cuMemRelease(real);
      rc = CUDA_ERROR_OUT_OF_MEMORY;
    }
  }
  pthread_mutex_unlock(&g_lock);
  gate_exit();
  /* In case cuInit came through a path the shim does not see. */
  if (rc == CUDA_SUCCESS) ensure_control_thread();
  return rc;
}

CUresult cuMemExportToShareableHandle(void* shHandle,
                                      CUmemGenericAllocationHandle h, int type,
                                      unsigned long long flags) {
  REAL(r_cuMemExportToShareableHandle, "cuMemExportToShareableHandle");
  /* Refuse fabric exports. The driver fabric-exports memory that never asked
   * for fabric handles, so torch's export-as-FABRIC probe would succeed and
   * create one 00f8 object per pool chunk. Failing it makes torch fall back to
   * POSIX fds. */
  if (type == CU_MEM_HANDLE_TYPE_FABRIC && !allow_fabric()) {
    static int logged;
    if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
      mclog(
          "refusing fabric-typed cuMemExportToShareableHandle; set "
          "MCSHIM_ALLOW_FABRIC=1 to permit it");
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  gate_enter();
  pthread_mutex_lock(&g_lock);
  int i = alloc_by_app(h);
  CUresult rc = r_cuMemExportToShareableHandle(shHandle, xlate(h), type, flags);
  if (rc == CUDA_SUCCESS && i >= 0) {
    if (type == CU_MEM_HANDLE_TYPE_POSIX_FD && shHandle)
      record_key(i, *(int*)shHandle);
    else
      mark_untracked("export of a non-POSIX-fd handle type");
  }
  pthread_mutex_unlock(&g_lock);
  gate_exit();
  return rc;
}

CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* h,
                                        void* osHandle, int type) {
  REAL(r_cuMemImportFromShareableHandle, "cuMemImportFromShareableHandle");
  REAL(r_cuMemRelease, "cuMemRelease");
  resolve_internal();
  gate_enter();
  pthread_mutex_lock(&g_lock);
  CUresult rc = r_cuMemImportFromShareableHandle(h, osHandle, type);
  if (rc == CUDA_SUCCESS && h) {
    int j = alloc_by_handle(*h);
    if (type != CU_MEM_HANDLE_TYPE_POSIX_FD) {
      mark_untracked("import of a non-POSIX-fd handle type");
    } else if (j >= 0) {
      /* The driver returned an object this process already holds. */
      g_alloc[j].app_refs++;
      *h = g_alloc[j].app;
    } else {
      CUmemGenericAllocationHandle real = *h;
      int i = track_new(KIND_IMP, real, cur_dev(), h);
      if (i >= 0) {
        g_alloc[i].imported = 1;
        /* For POSIX-FD imports osHandle is the fd. */
        record_key(i, (int)(intptr_t)osHandle);
      } else if (i == -2) {
        r_cuMemRelease(real);
        rc = CUDA_ERROR_OUT_OF_MEMORY;
      }
    }
  }
  pthread_mutex_unlock(&g_lock);
  gate_exit();
  /* In case cuInit came through a path the shim does not see. */
  if (rc == CUDA_SUCCESS) ensure_control_thread();
  return rc;
}

/* Report no fabric-handle support unless MCSHIM_ALLOW_FABRIC=1, so that
 * frameworks choose POSIX fds. */
CUresult cuDeviceGetAttribute(int* pi, int attrib, CUdevice dev) {
  REAL(r_cuDeviceGetAttribute, "cuDeviceGetAttribute");
  if (!r_cuDeviceGetAttribute) return CUDA_ERROR_NOT_INITIALIZED;
  CUresult rc = r_cuDeviceGetAttribute(pi, attrib, dev);
  if (rc == CUDA_SUCCESS && pi &&
      attrib == CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED && *pi != 0 &&
      !allow_fabric())
    *pi = 0;
  return rc;
}

CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* h,
                                     void* addr) {
  REAL(r_cuMemRetainAllocationHandle, "cuMemRetainAllocationHandle");
  gate_enter();
  pthread_mutex_lock(&g_lock);
  CUresult rc = r_cuMemRetainAllocationHandle(h, addr);
  if (rc == CUDA_SUCCESS && h) {
    int i = alloc_by_handle(*h);
    if (i >= 0) {
      g_alloc[i].app_refs++;
      *h = g_alloc[i].app;
    }
  }
  pthread_mutex_unlock(&g_lock);
  gate_exit();
  return rc;
}

CUresult cuMemGetAllocationPropertiesFromHandle(
    CUmemAllocationProp* prop, CUmemGenericAllocationHandle h) {
  REAL(r_cuMemGetAllocationPropertiesFromHandle,
       "cuMemGetAllocationPropertiesFromHandle");
  gate_enter();
  pthread_mutex_lock(&g_lock);
  CUresult rc = r_cuMemGetAllocationPropertiesFromHandle(prop, xlate(h));
  pthread_mutex_unlock(&g_lock);
  gate_exit();
  return rc;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle h) {
  REAL(r_cuMemRelease, "cuMemRelease");
  gate_enter();
  pthread_mutex_lock(&g_lock);
  int i = alloc_by_app(h);
  CUresult rc = r_cuMemRelease(xlate(h));
  if (rc == CUDA_SUCCESS && i >= 0) {
    g_alloc[i].app_refs--;
    alloc_gc(i);
  }
  pthread_mutex_unlock(&g_lock);
  gate_exit();
  return rc;
}

CUresult cuMulticastAddDevice(CUmemGenericAllocationHandle h, CUdevice dev) {
  REAL(r_cuMulticastAddDevice, "cuMulticastAddDevice");
  gate_enter();
  pthread_mutex_lock(&g_lock);
  int i = alloc_by_app(h);
  CUresult rc = r_cuMulticastAddDevice(xlate(h), dev);
  if (rc == CUDA_SUCCESS && i >= 0) {
    Alloc* a = &g_alloc[i];
    if (a->kind == KIND_IMP) a->kind = KIND_MC;
    if (a->kind == KIND_MC) {
      if (a->ndev < MAX_DEV)
        a->devs[a->ndev++] = dev;
      else
        mark_untracked("too many devices in a multicast group");
    }
  }
  pthread_mutex_unlock(&g_lock);
  gate_exit();
  return rc;
}

/* Must hold g_lock. Record a successful bind into group gi. */
static void bind_record(int gi, int v2, int by_addr, int mi, CUdeviceptr va,
                        size_t mcOffset, size_t memOffset, size_t size,
                        CUdevice dev) {
  if (gi < 0) {
    mark_untracked("bind to an untracked multicast object");
    return;
  }
  if (!by_addr && mi < 0) {
    mark_untracked("bind of untracked memory");
    return;
  }
  if (dev < 0 || dev >= MAX_DEV) {
    mark_untracked("bind on an unknown device");
    return;
  }
  if (g_alloc[gi].kind == KIND_IMP) g_alloc[gi].kind = KIND_MC;
  for (int b = 0; b < MAXN; b++) {
    if (g_bind[b].used) continue;
    g_bind[b] = (Bind){.used = 1,
                       .groupIdx = gi,
                       .v2 = v2,
                       .by_addr = by_addr,
                       .memIdx = mi,
                       .va = va,
                       .mcOffset = mcOffset,
                       .memOffset = memOffset,
                       .size = size,
                       .dev = dev};
    return;
  }
  mark_untracked("bind table overflow");
}

/* The device a v1 bind applies to: the one hosting the memory. */
static CUdevice mem_dev(int mi) {
  if (mi >= 0 && g_alloc[mi].kind == KIND_UC &&
      g_alloc[mi].uprop.location.type == CU_MEM_LOCATION_TYPE_DEVICE)
    return g_alloc[mi].uprop.location.id;
  return cur_dev();
}

/* Binds block until every device has joined the group, so the real call runs
 * without g_lock. A bind by address over tracked memory is recorded against
 * the allocation, so that its replay depends neither on the VA nor on the
 * order of remaps; only untracked memory, which stays resident, is replayed by
 * address. */
static CUresult do_bind(int v2, int by_addr, CUmemGenericAllocationHandle mc,
                        CUdevice dev, size_t mcOffset,
                        CUmemGenericAllocationHandle mem, CUdeviceptr va,
                        size_t memOffset, size_t size,
                        unsigned long long flags) {
  resolve_internal();
  gate_enter();
  pthread_mutex_lock(&g_lock);
  int gi = alloc_by_app(mc);
  int mi = by_addr ? -1 : alloc_by_app(mem);
  int spans = 0;
  if (by_addr) {
    int m = map_at(va);
    if (m >= 0 && va + size <= g_map[m].va + g_map[m].size) {
      mi = g_map[m].allocIdx;
      memOffset = va - g_map[m].va + g_map[m].offset;
    } else if (m >= 0) {
      spans = 1;
    }
  }
  int tracked_mem = mi >= 0;
  CUmemGenericAllocationHandle rmc = xlate(mc);
  CUmemGenericAllocationHandle rmem = by_addr ? 0 : xlate(mem);
  CUmemGenericAllocationHandle hmem = tracked_mem ? g_alloc[mi].handle : 0;
  pthread_mutex_unlock(&g_lock);
  CUresult rc;
  if (v2 && by_addr)
    rc = r_cuMulticastBindAddr_v2(rmc, dev, mcOffset, va, size, flags);
  else if (v2)
    rc = r_cuMulticastBindMem_v2(rmc, dev, mcOffset, rmem, memOffset, size,
                                 flags);
  else if (by_addr)
    rc = r_cuMulticastBindAddr(rmc, mcOffset, va, size, flags);
  else
    rc = r_cuMulticastBindMem(rmc, mcOffset, rmem, memOffset, size, flags);
  if (rc == CUDA_SUCCESS) {
    pthread_mutex_lock(&g_lock);
    /* An object freed and replaced meanwhile would be an app race. */
    if (gi >= 0 && g_alloc[gi].handle != rmc) gi = -1;
    if (tracked_mem && g_alloc[mi].handle != hmem) mi = -1;
    if (spans)
      mark_untracked("multicast bind across mappings");
    else if (tracked_mem && mi < 0)
      mark_untracked("bind of memory freed concurrently");
    else
      bind_record(gi, v2, by_addr && !tracked_mem, mi, va, mcOffset, memOffset,
                  size, v2 ? dev : mem_dev(mi));
    pthread_mutex_unlock(&g_lock);
  }
  gate_exit();
  return rc;
}

CUresult cuMulticastBindMem(CUmemGenericAllocationHandle mc, size_t mcOffset,
                            CUmemGenericAllocationHandle mem, size_t memOffset,
                            size_t size, unsigned long long flags) {
  REAL(r_cuMulticastBindMem, "cuMulticastBindMem");
  return do_bind(0, 0, mc, -1, mcOffset, mem, 0, memOffset, size, flags);
}

CUresult cuMulticastBindAddr(CUmemGenericAllocationHandle mc, size_t mcOffset,
                             CUdeviceptr memptr, size_t size,
                             unsigned long long flags) {
  REAL(r_cuMulticastBindAddr, "cuMulticastBindAddr");
  return do_bind(0, 1, mc, -1, mcOffset, 0, memptr, 0, size, flags);
}

CUresult cuMulticastBindMem_v2(CUmemGenericAllocationHandle mc, CUdevice dev,
                               size_t mcOffset,
                               CUmemGenericAllocationHandle mem,
                               size_t memOffset, size_t size,
                               unsigned long long flags) {
  REAL(r_cuMulticastBindMem_v2, "cuMulticastBindMem_v2");
  return do_bind(1, 0, mc, dev, mcOffset, mem, 0, memOffset, size, flags);
}

CUresult cuMulticastBindAddr_v2(CUmemGenericAllocationHandle mc, CUdevice dev,
                                size_t mcOffset, CUdeviceptr memptr,
                                size_t size, unsigned long long flags) {
  REAL(r_cuMulticastBindAddr_v2, "cuMulticastBindAddr_v2");
  return do_bind(1, 1, mc, dev, mcOffset, 0, memptr, 0, size, flags);
}

CUresult cuMulticastUnbind(CUmemGenericAllocationHandle mc, CUdevice dev,
                           size_t mcOffset, size_t size) {
  REAL(r_cuMulticastUnbind, "cuMulticastUnbind");
  gate_enter();
  pthread_mutex_lock(&g_lock);
  int gi = alloc_by_app(mc);
  CUresult rc = r_cuMulticastUnbind(xlate(mc), dev, mcOffset, size);
  if (rc == CUDA_SUCCESS && gi >= 0) {
    for (int b = 0; b < MAXN; b++)
      if (g_bind[b].used && g_bind[b].groupIdx == gi && g_bind[b].dev == dev &&
          g_bind[b].mcOffset >= mcOffset &&
          g_bind[b].mcOffset + g_bind[b].size <= mcOffset + size)
        g_bind[b].used = 0;
    alloc_gc_all();
  }
  pthread_mutex_unlock(&g_lock);
  gate_exit();
  return rc;
}

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle h, unsigned long long flags) {
  REAL(r_cuMemMap, "cuMemMap");
  resolve_internal();
  gate_enter();
  pthread_mutex_lock(&g_lock);
  int ai = alloc_by_app(h);
  CUresult rc = r_cuMemMap(ptr, size, offset, xlate(h), flags);
  if (rc == CUDA_SUCCESS && ai >= 0) {
    int m = 0;
    while (m < MAXN && g_map[m].used) m++;
    if (m < MAXN) {
      CUdevice d = cur_dev();
      /* Created without a current context: adopt the mapping's device. */
      if (g_alloc[ai].dev < 0) g_alloc[ai].dev = d;
      g_map[m] = (Mapping){.used = 1,
                           .va = ptr,
                           .size = size,
                           .offset = offset,
                           .allocIdx = ai,
                           .dev = d >= 0 ? d : g_alloc[ai].dev};
    } else {
      mark_untracked("mapping table overflow");
    }
  }
  pthread_mutex_unlock(&g_lock);
  gate_exit();
  return rc;
}

/* Must hold g_lock. Mark (in sel) the mappings inside [ptr, ptr+size);
 * returns 1 if a mapping straddles the range. */
static int map_within(const Mapping* mp, CUdeviceptr ptr, size_t size) {
  return mp->used && mp->va >= ptr && mp->va + mp->size <= ptr + size;
}

static int maps_in(CUdeviceptr ptr, size_t size, unsigned char* sel) {
  int partial = 0;
  for (int m = 0; m < MAXN; m++) {
    const Mapping* mp = &g_map[m];
    sel[m] = map_within(mp, ptr, size);
    if (mp->used && !sel[m] && mp->va < ptr + size && mp->va + mp->size > ptr)
      partial = 1;
  }
  return partial;
}

/* cuMemUnmap and cuMemSetAccess can wait for GPU work, which may wait for a
 * host thread that needs g_lock, so the real calls run without it: the
 * affected mappings are selected before and updated after. */

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
  REAL(r_cuMemUnmap, "cuMemUnmap");
  unsigned char sel[MAXN];
  gate_enter();
  pthread_mutex_lock(&g_lock);
  maps_in(ptr, size, sel);
  pthread_mutex_unlock(&g_lock);
  CUresult rc = r_cuMemUnmap(ptr, size);
  if (rc == CUDA_SUCCESS) {
    pthread_mutex_lock(&g_lock);
    /* The range may span several mappings. */
    for (int m = 0; m < MAXN; m++) {
      if (!sel[m] || !map_within(&g_map[m], ptr, size)) continue;
      g_map[m].used = 0;
      alloc_gc(g_map[m].allocIdx);
    }
    pthread_mutex_unlock(&g_lock);
  }
  gate_exit();
  return rc;
}

static int same_location(const CUmemLocation* a, const CUmemLocation* b) {
  return a->type == b->type &&
         (a->type != CU_MEM_LOCATION_TYPE_DEVICE || a->id == b->id);
}

CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size,
                        const CUmemAccessDesc* desc, size_t count) {
  REAL(r_cuMemSetAccess, "cuMemSetAccess");
  unsigned char sel[MAXN];
  gate_enter();
  pthread_mutex_lock(&g_lock);
  int partial = maps_in(ptr, size, sel);
  pthread_mutex_unlock(&g_lock);
  CUresult rc = r_cuMemSetAccess(ptr, size, desc, count);
  if (rc == CUDA_SUCCESS && desc) {
    pthread_mutex_lock(&g_lock);
    if (partial) mark_untracked("cuMemSetAccess over part of a mapping");
    /* The call updates only the listed locations, over every mapping in the
     * range: NCCL sets access once over a reservation holding several maps,
     * and torch grants peers one at a time. */
    for (int m = 0; m < MAXN; m++) {
      Mapping* mp = &g_map[m];
      if (!sel[m] || !map_within(mp, ptr, size)) continue;
      for (size_t k = 0; k < count; k++) {
        int j = 0;
        while (j < mp->naccess &&
               !same_location(&mp->access[j].location, &desc[k].location))
          j++;
        if (j == mp->naccess) {
          if (j == MAX_ACCESS) {
            mark_untracked("access table overflow");
            break;
          }
          mp->naccess++;
        }
        mp->access[j] = desc[k];
      }
    }
    pthread_mutex_unlock(&g_lock);
  }
  gate_exit();
  return rc;
}

/* Sparse array mappings are not replayed: translate, and refuse checkpoints. */
static CUresult map_array(CUresult (*real)(CUarrayMapInfo*, unsigned, CUstream),
                          CUarrayMapInfo* list, unsigned count, CUstream st) {
  if (!real) return CUDA_ERROR_NOT_FOUND;
  CUarrayMapInfo* copy = count ? malloc(count * sizeof(*list)) : NULL;
  if (count && !copy) return CUDA_ERROR_OUT_OF_MEMORY;
  gate_enter();
  pthread_mutex_lock(&g_lock);
  for (unsigned k = 0; k < count; k++) {
    copy[k] = list[k];
    if (copy[k].memOperationType == CU_MEM_OPERATION_TYPE_MAP &&
        copy[k].memHandleType == CU_MEM_HANDLE_TYPE_GENERIC)
      copy[k].memHandle.memHandle = xlate(copy[k].memHandle.memHandle);
  }
  CUresult rc = real(copy, count, st);
  if (rc == CUDA_SUCCESS) mark_untracked("sparse array mapping");
  pthread_mutex_unlock(&g_lock);
  gate_exit();
  free(copy);
  return rc;
}

static CUresult (*r_cuMemMapArrayAsync)(CUarrayMapInfo*, unsigned, CUstream);
static CUresult (*r_cuMemMapArrayAsync_ptsz)(CUarrayMapInfo*, unsigned,
                                             CUstream);

CUresult cuMemMapArrayAsync(CUarrayMapInfo* list, unsigned count, CUstream st) {
  REAL(r_cuMemMapArrayAsync, "cuMemMapArrayAsync");
  return map_array(r_cuMemMapArrayAsync, list, count, st);
}

CUresult cuMemMapArrayAsync_ptsz(CUarrayMapInfo* list, unsigned count,
                                 CUstream st) {
  REAL(r_cuMemMapArrayAsync_ptsz, "cuMemMapArrayAsync_ptsz");
  return map_array(r_cuMemMapArrayAsync_ptsz, list, count, st);
}

/* Shared state neither the shim nor cuda-checkpoint can carry: succeed, and
 * refuse checkpoints. */
#define REFUSED_LIST(X)                                                      \
  X(cuMemPoolExportToShareableHandle,                                        \
    (void* out, CUmemoryPool pool, int type, unsigned long long fl),         \
    (out, pool, type, fl), "memory pool export")                             \
  X(cuMemPoolImportFromShareableHandle,                                      \
    (CUmemoryPool * out, void* h, int type, unsigned long long fl),          \
    (out, h, type, fl), "memory pool import")                                \
  X(cuMemPoolExportPointer, (void* out, CUdeviceptr ptr), (out, ptr),        \
    "memory pool pointer export")                                            \
  X(cuMemPoolImportPointer,                                                  \
    (CUdeviceptr * out, CUmemoryPool pool, void* data), (out, pool, data),   \
    "memory pool pointer import")                                            \
  X(cuLogicalEndpointCreate, (cuuint32_t id, const void* prop), (id, prop),  \
    "logical endpoint")                                                      \
  X(cuLogicalEndpointImport, (cuuint32_t id, const void* h, int type),       \
    (id, h, type), "logical endpoint import")                                \
  X(cuMemAllocManaged, (CUdeviceptr * p, size_t n, unsigned fl), (p, n, fl), \
    "managed memory")

#define REFUSED_DEF(name, proto, args, why)     \
  static CUresult(*r_##name) proto;             \
  CUresult name proto;                          \
  CUresult name proto {                         \
    REAL(r_##name, #name);                      \
    if (!r_##name) return CUDA_ERROR_NOT_FOUND; \
    gate_enter();                               \
    CUresult rc = r_##name args;                \
    if (rc == CUDA_SUCCESS) {                   \
      pthread_mutex_lock(&g_lock);              \
      mark_untracked(why);                      \
      pthread_mutex_unlock(&g_lock);            \
    }                                           \
    gate_exit();                                \
    return rc;                                  \
  }

REFUSED_LIST(REFUSED_DEF)

/* Logical endpoints are refused above, but a bind takes an allocation handle,
 * which must still be translated. */
static CUresult (*r_cuLogicalEndpointBindMem)(cuuint32_t, CUdevice, cuuint64_t,
                                              CUmemGenericAllocationHandle,
                                              cuuint64_t, cuuint64_t,
                                              unsigned long long);

CUresult cuLogicalEndpointBindMem(cuuint32_t id, CUdevice dev, cuuint64_t off,
                                  CUmemGenericAllocationHandle h,
                                  cuuint64_t memOff, cuuint64_t size,
                                  unsigned long long fl) {
  REAL(r_cuLogicalEndpointBindMem, "cuLogicalEndpointBindMem");
  if (!r_cuLogicalEndpointBindMem) return CUDA_ERROR_NOT_FOUND;
  gate_enter();
  pthread_mutex_lock(&g_lock);
  CUmemGenericAllocationHandle rh = xlate(h);
  pthread_mutex_unlock(&g_lock);
  CUresult rc = r_cuLogicalEndpointBindMem(id, dev, off, rh, memOff, size, fl);
  if (rc == CUDA_SUCCESS) {
    pthread_mutex_lock(&g_lock);
    mark_untracked("logical endpoint bind");
    pthread_mutex_unlock(&g_lock);
  }
  gate_exit();
  return rc;
}

/* Cross-rank fd rendezvous. After a restore, each exporter re-exports its
 * object and publishes "<pid> <fd>" in /tmp/mcshim under the original export's
 * identity, and importers copy the fd with pidfd_getfd(2). That needs ptrace
 * access, which YAMA denies between sibling processes, so an exporter allows
 * any tracer (PR_SET_PTRACER_ANY) while it has fds published, which is until
 * the sentry removes the gate after every process has resumed. */

/* Fixed; see cudaShimDir in pkg/sentry/control/state_cuda_shim.go. */
static const char g_dir[] = "/tmp/mcshim";
static int g_ptracer_any; /* under g_lock */

static void pub_path(const Alloc* a, char* out, size_t n) {
  snprintf(out, n, "%s/fd-%lx-%lx", g_dir, a->key_client, a->key_object);
}

/* Must hold g_lock. */
static int publish_fd(Alloc* a, int fd) {
  char path[600], tmp[610];
  pub_path(a, path, sizeof(path));
  snprintf(tmp, sizeof(tmp), "%s.tmp", path);
  if (!g_ptracer_any) {
    prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
    g_ptracer_any = 1;
  }
  FILE* f = fopen(tmp, "w");
  if (!f) return -1;
  fprintf(f, "%d %d\n", (int)getpid(), fd);
  if (fclose(f) != 0 || rename(tmp, path) != 0) return -1;
  a->pub_fd = fd;
  return 0;
}

/* Must hold g_lock. */
static void unpublish_fd(Alloc* a) {
  if (a->pub_fd < 0) return;
  char path[600];
  pub_path(a, path, sizeof(path));
  unlink(path);
  close(a->pub_fd); /* a held export fd blocks the next checkpoint */
  a->pub_fd = -1;
}

/* Must hold g_lock. */
static void unpublish_all(void) {
  for (int i = 0; i < MAXN; i++)
    if (g_alloc[i].kind != KIND_FREE) unpublish_fd(&g_alloc[i]);
  if (g_ptracer_any) {
    prctl(PR_SET_PTRACER, 0, 0, 0, 0);
    g_ptracer_any = 0;
  }
}

/* Exporters announce each exported identity as exp-<client>-<object>.<pid>, so
 * that an importer can check at the gate, before anything is torn down, that
 * its exporter will republish (see can_carry). */
static void exp_path(const Alloc* a, int pid, char* out, size_t n) {
  snprintf(out, n, "%s/exp-%lx-%lx.%d", g_dir, a->key_client, a->key_object,
           pid);
}

/* Must hold g_lock. */
static void mark_export(Alloc* a) {
  char p[600];
  exp_path(a, (int)getpid(), p, sizeof(p));
  int fd = open(p, O_CREAT | O_WRONLY | O_CLOEXEC, 0666);
  if (fd >= 0) close(fd);
  a->exp_marked = fd >= 0;
}

/* Must hold g_lock. */
static void unmark_export(Alloc* a) {
  if (!a->exp_marked) return;
  char p[600];
  exp_path(a, (int)getpid(), p, sizeof(p));
  unlink(p);
  a->exp_marked = 0;
}

/* Whether a live process announces a's identity. */
static int export_announced(const Alloc* a) {
  char prefix[96];
  int n = snprintf(prefix, sizeof(prefix), "exp-%lx-%lx.", a->key_client,
                   a->key_object);
  DIR* d = opendir(g_dir);
  if (!d) return 0;
  int found = 0;
  for (struct dirent* e; !found && (e = readdir(d));) {
    if (strncmp(e->d_name, prefix, n) != 0) continue;
    int pid = atoi(e->d_name + n);
    found = pid > 0 && (kill(pid, 0) == 0 || errno == EPERM);
  }
  closedir(d);
  return found;
}

/* Remove announcements a dead process with this pid left behind. */
static void unmark_stale_exports(void) {
  char suffix[32];
  int n = snprintf(suffix, sizeof(suffix), ".%d", (int)getpid());
  DIR* d = opendir(g_dir);
  if (!d) return;
  for (struct dirent* e; (e = readdir(d));) {
    size_t len = strlen(e->d_name);
    if (strncmp(e->d_name, "exp-", 4) != 0 || len <= (size_t)n ||
        strcmp(e->d_name + len - n, suffix) != 0)
      continue;
    char p[600];
    snprintf(p, sizeof(p), "%s/%s", g_dir, e->d_name);
    unlink(p);
  }
  closedir(d);
}

/* Copy the fd published for a, waiting up to timeout_ms for it to appear. */
static int fetch_fd(const Alloc* a, int timeout_ms) {
  char path[600];
  pub_path(a, path, sizeof(path));
  for (int waited = 0; waited < timeout_ms; waited += 10) {
    int pid, fd;
    FILE* f = fopen(path, "r");
    if (f) {
      int n = fscanf(f, "%d %d", &pid, &fd);
      fclose(f);
      if (n == 2) {
        int pfd = (int)syscall(SYS_pidfd_open, pid, 0);
        int got = pfd < 0 ? -1 : (int)syscall(SYS_pidfd_getfd, pfd, fd, 0);
        if (got < 0)
          mclog("RESUME: copying fd %d from pid %d: %s", fd, pid,
                strerror(errno));
        if (pfd >= 0) close(pfd);
        return got;
      }
    }
    struct timespec ts = {0, 10 * 1000 * 1000};
    nanosleep(&ts, NULL);
  }
  mclog("RESUME: timed out waiting for %s", path);
  return -1;
}

/* Suspend/resume helpers. Calls are issued from each device's primary
 * context, retained for the duration of a transition; can_carry checks that
 * the application still holds it. */

static CUcontext g_pctx[MAX_DEV];

/* Whether the application holds device d's primary context. */
static int dev_active(CUdevice d) {
  unsigned flags = 0;
  int active = 0;
  return d >= 0 && d < MAX_DEV &&
         r_cuDevicePrimaryCtxGetState(d, &flags, &active) == CUDA_SUCCESS &&
         active;
}

/* Must hold g_lock. Never creates a context: retaining an inactive primary
 * context would, and can fail (exclusive-process compute mode). */
static int use_dev(CUdevice d) {
  if (d < 0 || d >= MAX_DEV) return -1;
  if (!g_pctx[d] && (!dev_active(d) || r_cuDevicePrimaryCtxRetain(
                                           &g_pctx[d], d) != CUDA_SUCCESS)) {
    g_pctx[d] = NULL;
    return -1;
  }
  return r_cuCtxSetCurrent(g_pctx[d]) == CUDA_SUCCESS ? 0 : -1;
}

static void release_devs(void) {
  for (int d = 0; d < MAX_DEV; d++)
    if (g_pctx[d]) {
      r_cuDevicePrimaryCtxRelease(d);
      g_pctx[d] = NULL;
    }
}

/* Must hold g_lock. Bitmask of the devices the tracked state lives on. */
static unsigned devs_in_use(void) {
  unsigned mask = 0;
  for (int i = 0; i < MAXN; i++) {
    if (g_alloc[i].kind == KIND_FREE) continue;
    if (g_alloc[i].dev >= 0 && g_alloc[i].dev < MAX_DEV)
      mask |= 1u << g_alloc[i].dev;
    for (int d = 0; d < g_alloc[i].ndev; d++)
      if (g_alloc[i].devs[d] >= 0 && g_alloc[i].devs[d] < MAX_DEV)
        mask |= 1u << g_alloc[i].devs[d];
  }
  for (int m = 0; m < MAXN; m++)
    if (g_map[m].used && g_map[m].dev >= 0 && g_map[m].dev < MAX_DEV)
      mask |= 1u << g_map[m].dev;
  for (int b = 0; b < MAXN; b++)
    if (g_bind[b].used) mask |= 1u << g_bind[b].dev;
  return mask;
}

/* Must hold g_lock. Synchronize every device in use. After a restore, the
 * first VMM call on a context can fail with CUDA_ERROR_UNKNOWN until it is. */
static int sync_devs(const char* what) {
  unsigned mask = devs_in_use();
  for (int d = 0; d < MAX_DEV; d++) {
    if (!(mask & (1u << d))) continue;
    CUresult rc = use_dev(d) ? -1 : r_cuCtxSynchronize();
    if (rc != CUDA_SUCCESS) {
      mclog("%s: synchronizing device %d rc=%d", what, d, rc);
      return -1;
    }
  }
  return 0;
}

/* Must hold g_lock. A device in mask with read-write access to mapping mp,
 * preferring the mapping's own, or -1. */
static CUdevice rw_dev(const Mapping* mp, unsigned mask) {
  CUdevice best = -1;
  for (int j = 0; j < mp->naccess; j++) {
    const CUmemAccessDesc* a = &mp->access[j];
    CUdevice d = a->location.id;
    if (a->location.type != CU_MEM_LOCATION_TYPE_DEVICE ||
        a->flags != CU_MEM_ACCESS_FLAGS_PROT_READWRITE || d < 0 ||
        d >= MAX_DEV || !(mask & (1u << d)))
      continue;
    if (d == mp->dev) return d;
    if (best < 0) best = d;
  }
  return best;
}

/* Must hold g_lock. Whether do_suspend releases alloc i. */
static int torn_down(int i) {
  const Alloc* a = &g_alloc[i];
  return a->kind == KIND_MC || a->kind == KIND_IMP ||
         (a->kind == KIND_UC && a->has_key && bound_as_mem(i));
}

/* Must hold g_lock. Why the tracked state cannot be carried through a
 * checkpoint, or NULL. Runs at the gate, before anything is torn down. */
static const char* can_carry(void) {
  if (g_untracked) return g_untracked_why;
  const char* bad = __atomic_load_n(&g_bad_lookup, __ATOMIC_ACQUIRE);
  if (bad) return bad;
  unsigned mask = devs_in_use();
  for (int d = 0; d < MAX_DEV; d++)
    if ((mask & (1u << d)) && !dev_active(d))
      return "a device's primary context is not active";
  for (int i = 0; i < MAXN; i++) {
    const Alloc* a = &g_alloc[i];
    if (a->kind == KIND_FREE) continue;
    int maps = has_maps(i), torn = torn_down(i);
    if ((a->dev < 0 || a->dev >= MAX_DEV) &&
        (torn || a->has_key || bound_as_mem(i)))
      return "an object on an unknown device";
    /* Restoring more than one reference needs a mapped VA to retain from. */
    if (torn && a->app_refs > 1 && !maps)
      return "several references to an unmapped object";
    /* A freed export's contents are saved through a mapping, and memory kept
     * alive only by a bind dies when suspend unbinds it. */
    if (a->kind == KIND_UC && !maps && (torn || a->app_refs <= 0))
      return "multicast-bound memory without a mapping";
    if (a->kind == KIND_UC && torn)
      for (int m = 0; m < MAXN; m++)
        if (g_map[m].used && g_map[m].allocIdx == i &&
            rw_dev(&g_map[m], mask) < 0)
          return "multicast-bound export without read-write access";
    /* An import is rebuilt from its exporter's republished fd. */
    if (a->imported && a->has_key && !export_announced(a))
      return "an import whose exporter will not republish it";
  }
  return NULL;
}

/* Must hold g_lock. A usable handle for alloc i: its own while referenced,
 * otherwise a reference retained from a mapping (*held = 1; release it). */
static int hold_handle(int i, CUmemGenericAllocationHandle* h, int* held) {
  Alloc* a = &g_alloc[i];
  *held = 0;
  if (a->app_refs > 0 || a->shim_ref) {
    *h = a->handle;
    return 0;
  }
  int m = first_map(i);
  if (m < 0 || use_dev(g_map[m].dev) != 0 ||
      r_cuMemRetainAllocationHandle(h, (void*)(uintptr_t)g_map[m].va) !=
          CUDA_SUCCESS)
    return -1;
  *held = 1;
  return 0;
}

/* Must hold g_lock. Unmap every VA that maps alloc gi, KEEPING the VA
 * reservations (cuMemUnmap only -- never cuMemAddressFree). */
static int unmap_alloc(int gi, const char* what, int* unmapped) {
  for (int m = 0; m < MAXN; m++) {
    if (!g_map[m].used || g_map[m].allocIdx != gi) continue;
    CUresult rc =
        use_dev(g_map[m].dev) ? -1 : r_cuMemUnmap(g_map[m].va, g_map[m].size);
    if (rc != CUDA_SUCCESS) {
      mclog("SUSPEND: cuMemUnmap(%s 0x%llx) rc=%d", what,
            (unsigned long long)g_map[m].va, rc);
      return -1;
    }
    (*unmapped)++;
  }
  return 0;
}

/* Must hold g_lock. Re-map every VA of alloc gi at the identical address and
 * replay its access set. */
static int remap_alloc(int gi, const char* what, int* remapped) {
  for (int m = 0; m < MAXN; m++) {
    Mapping* mp = &g_map[m];
    if (!mp->used || mp->allocIdx != gi) continue;
    CUresult rc = use_dev(mp->dev) ? -1
                                   : r_cuMemMap(mp->va, mp->size, mp->offset,
                                                g_alloc[gi].handle, 0);
    if (rc != CUDA_SUCCESS) {
      mclog("RESUME: %s re-map at 0x%llx rc=%d", what,
            (unsigned long long)mp->va, rc);
      return -1;
    }
    /* With no access set recorded, grant RW to the mapping device: an
     * inaccessible view faults the collective (719). */
    CUmemAccessDesc fallback = {{CU_MEM_LOCATION_TYPE_DEVICE, mp->dev},
                                CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
    rc = mp->naccess ? r_cuMemSetAccess(mp->va, mp->size, mp->access,
                                        (size_t)mp->naccess)
                     : r_cuMemSetAccess(mp->va, mp->size, &fallback, 1);
    if (rc != CUDA_SUCCESS) {
      mclog("RESUME: %s cuMemSetAccess(0x%llx) rc=%d", what,
            (unsigned long long)mp->va, rc);
      return -1;
    }
    (*remapped)++;
  }
  return 0;
}

/* Must hold g_lock. Drop the application's references to alloc i, and forget
 * its driver handle: the object is freed once unmapped. */
static int release_refs(int i, const char* what, int* released) {
  for (int k = 0; k < g_alloc[i].app_refs; k++) {
    CUresult rc =
        use_dev(g_alloc[i].dev) ? -1 : r_cuMemRelease(g_alloc[i].handle);
    if (rc != CUDA_SUCCESS) {
      mclog("SUSPEND: cuMemRelease(%s 0x%llx) rc=%d", what,
            (unsigned long long)g_alloc[i].handle, rc);
      return -1;
    }
    (*released)++;
  }
  g_alloc[i].handle = 0;
  return 0;
}

/* Must hold g_lock. Export alloc gi's handle h and publish it for importers.
 */
static int reexport(int gi, CUmemGenericAllocationHandle h) {
  /* A freshly restored allocation can transiently fail its export with
   * INVALID_VALUE; retry briefly (unretried, peers time out and fault with
   * 719). */
  int fd = -1;
  CUresult rc = 0;
  for (int attempt = 0; attempt < 100; attempt++) {
    rc = use_dev(g_alloc[gi].dev) ? -1
                                  : r_cuMemExportToShareableHandle(
                                        &fd, h, CU_MEM_HANDLE_TYPE_POSIX_FD, 0);
    if (rc == CUDA_SUCCESS && fd >= 0) break;
    if (attempt == 0)
      mclog("RESUME: re-export idx=%d kind=%d dev=%d rc=%d, retrying", gi,
            g_alloc[gi].kind, g_alloc[gi].dev, rc);
    struct timespec ts = {0, 100 * 1000 * 1000}; /* 100ms */
    nanosleep(&ts, NULL);
  }
  if (rc != CUDA_SUCCESS || fd < 0) {
    mclog("RESUME: re-export idx=%d gave up rc=%d fd=%d", gi, rc, fd);
    return -1;
  }
  /* Keep the export fd out of exec'd children, where it blocks the next
   * checkpoint. */
  fcntl(fd, F_SETFD, FD_CLOEXEC);
  if (publish_fd(&g_alloc[gi], fd) != 0) {
    close(fd);
    return -1;
  }
  return 0;
}

/* Must hold g_lock. Fetch gi's re-exported fd and re-import it. Concurrent
 * imports can transiently fail with 304, so retry, bounded. */
static int reimport(int gi) {
  Alloc* a = &g_alloc[gi];
  if (!a->has_key) {
    mclog("RESUME: imported idx=%d has no rendezvous key", gi);
    return -1;
  }
  for (int attempt = 0;; attempt++) {
    int fd = fetch_fd(a, 60 * 1000);
    if (fd < 0) return -1;
    CUmemGenericAllocationHandle h = 0;
    CUresult rc = use_dev(a->dev) ? -1
                                  : r_cuMemImportFromShareableHandle(
                                        &h, (void*)(intptr_t)fd,
                                        CU_MEM_HANDLE_TYPE_POSIX_FD);
    close(fd);
    if (rc == CUDA_SUCCESS) {
      a->handle = h;
      a->shim_ref = 1;
      if (attempt > 0)
        mclog("RESUME: re-import idx=%d key=%lx:%lx ok after %d retries", gi,
              a->key_client, a->key_object, attempt);
      return 0;
    }
    if (attempt == 0)
      mclog(
          "RESUME: re-import idx=%d key=%lx:%lx dev=%d rc=%d on first "
          "attempt",
          gi, a->key_client, a->key_object, a->dev, rc);
    /* INVALID_DEVICE is never transient: this process cannot address the
     * exporter's device. */
    if (rc == CUDA_ERROR_INVALID_DEVICE) {
      mclog("RESUME: re-import idx=%d: exporting device not addressable", gi);
      return -1;
    }
    if (attempt >= 100) {
      mclog("RESUME: re-import idx=%d rc=%d after %d attempts", gi, rc,
            attempt);
      return -1;
    }
    struct timespec ts = {0, 200 * 1000 * 1000};
    nanosleep(&ts, NULL);
  }
}

/* Suspend: runs with the gate armed and drained, and can_carry satisfied. */
static int suspend_locked(void) {
  int groups = 0, imports = 0, unmapped = 0, unbound = 0, released = 0;
  int uc_freed = 0;

  if (g_untracked) {
    mclog("SUSPEND: refusing: %s", g_untracked_why);
    return -1;
  }
  if (sync_devs("SUSPEND") != 0) return -1;
  /* Withdraw the previous resume's fds: a held export fd blocks the
   * checkpoint. */
  unpublish_all();

  /* Multicast groups: unmap, unbind each device, release. */
  for (int gi = 0; gi < MAXN; gi++) {
    if (g_alloc[gi].kind != KIND_MC) continue;
    groups++;
    /* With no application reference, the unmap would free the group before
     * the unbinds. */
    CUmemGenericAllocationHandle h;
    int held;
    if (hold_handle(gi, &h, &held) != 0) {
      mclog("SUSPEND: no handle for group idx=%d", gi);
      return -1;
    }
    if (unmap_alloc(gi, "MC", &unmapped) != 0) return -1;
    for (int b = 0; b < MAXN; b++) {
      Bind* bp = &g_bind[b];
      if (!bp->used || bp->groupIdx != gi) continue;
      CUresult rc = use_dev(bp->dev) ? -1
                                     : r_cuMulticastUnbind(
                                           h, bp->dev, bp->mcOffset, bp->size);
      if (rc != CUDA_SUCCESS) {
        mclog(
            "SUSPEND: cuMulticastUnbind(dev=%d, mcOff=0x%zx, size=0x%zx) "
            "rc=%d",
            bp->dev, bp->mcOffset, bp->size, rc);
        return -1;
      }
      unbound++;
    }
    if (held && r_cuMemRelease(h) != CUDA_SUCCESS) return -1;
    if (release_refs(gi, "MC", &released) != 0) return -1;
  }

  /* Multicast-bound exporters: save contents, unmap (keeping reservations) and
   * release. Left resident, the next export after restore fails with
   * OBJECT_NOT_FOUND (R610, vLLM TP=4 and torch symmetric memory). Runs after
   * the group teardown, so the memory is already unbound. */
  for (int gi = 0; gi < MAXN; gi++) {
    if (g_alloc[gi].kind != KIND_UC || !torn_down(gi)) continue;
    void* buf = malloc(g_alloc[gi].size);
    if (!buf) {
      mclog("SUSPEND: no memory for UC-export backup (0x%zx bytes)",
            g_alloc[gi].size);
      return -1;
    }
    g_alloc[gi].uc_content = buf;
    for (int m = 0; m < MAXN; m++) {
      Mapping* mp = &g_map[m];
      if (!mp->used || mp->allocIdx != gi) continue;
      CUresult rc =
          use_dev(rw_dev(mp, devs_in_use()))
              ? -1
              : r_cuMemcpyDtoH_v2((char*)buf + mp->offset, mp->va, mp->size);
      if (rc != CUDA_SUCCESS) {
        mclog("SUSPEND: UC-export backup copy (va=0x%llx size=0x%zx) rc=%d",
              (unsigned long long)mp->va, mp->size, rc);
        return -1;
      }
    }
    if (unmap_alloc(gi, "UC-export", &unmapped) != 0) return -1;
    if (release_refs(gi, "UC-export", &released) != 0) return -1;
    uc_freed++;
  }

  /* Imports: unmap and release. The memory is the exporter's and
   * cuda-checkpoint saves it; only the live import must go, since
   * cuda-checkpoint cannot restore it. */
  for (int ii = 0; ii < MAXN; ii++) {
    if (g_alloc[ii].kind != KIND_IMP) continue;
    imports++;
    if (unmap_alloc(ii, "import", &unmapped) != 0) return -1;
    if (release_refs(ii, "import", &released) != 0) return -1;
  }

  mclog(
      "SUSPEND done: groups=%d imports=%d uc_freed=%d unmapped=%d "
      "unbound=%d released=%d",
      groups, imports, uc_freed, unmapped, unbound, released);
  return 0;
}

/* Resume: rebuild everything suspend released. A rank both exports and
 * imports, so every exporter publishes (phase 1) before anyone fetches (phase
 * 2). Binds block until every device has joined their group, so all
 * AddDevices (3a) precede all binds (3b); remaps follow (3c). Phase 4 returns
 * the reference counts to the application's. */
static int resume_locked(void) {
  int groups = 0, imports = 0, remapped = 0, rebound = 0, published = 0;

  if (sync_devs("RESUME") != 0) return -1;

  /* Phase 1: exporters re-create their objects and publish. */
  for (int gi = 0; gi < MAXN; gi++) {
    Alloc* a = &g_alloc[gi];
    if (a->kind == KIND_MC && !a->imported) {
      CUmemGenericAllocationHandle h = 0;
      if (use_dev(a->dev) != 0 ||
          r_cuMulticastCreate(&h, &a->mprop) != CUDA_SUCCESS) {
        mclog("RESUME: cuMulticastCreate idx=%d failed", gi);
        return -1;
      }
      a->handle = h;
      a->shim_ref = 1;
      if (a->has_key && reexport(gi, h) != 0) return -1;
      groups++;
    } else if (a->kind == KIND_UC && a->uc_content) {
      /* Freed across the checkpoint: recreate, re-map at the identical VAs and
       * restore the contents, before the re-export and the binds. */
      CUmemGenericAllocationHandle h = 0;
      CUresult rc =
          use_dev(a->dev) ? -1 : r_cuMemCreate(&h, a->size, &a->uprop, 0);
      if (rc != CUDA_SUCCESS) {
        mclog("RESUME: recreate UC-export idx=%d (size=0x%zx) rc=%d", gi,
              a->size, rc);
        return -1;
      }
      a->handle = h;
      a->shim_ref = 1;
      if (remap_alloc(gi, "UC-export", &remapped) != 0) return -1;
      for (int m = 0; m < MAXN; m++) {
        Mapping* mp = &g_map[m];
        if (!mp->used || mp->allocIdx != gi) continue;
        rc = use_dev(rw_dev(mp, devs_in_use()))
                 ? -1
                 : r_cuMemcpyHtoD_v2(mp->va, (char*)a->uc_content + mp->offset,
                                     mp->size);
        if (rc != CUDA_SUCCESS) {
          mclog("RESUME: UC-export content restore (va=0x%llx) rc=%d",
                (unsigned long long)mp->va, rc);
          return -1;
        }
      }
      free(a->uc_content);
      a->uc_content = NULL;
      if (a->has_key) {
        if (reexport(gi, h) != 0) return -1;
        published++;
      }
    } else if (a->kind == KIND_UC && a->has_key) {
      /* Resident P2P exporter: publish the handle importers fetch. */
      CUmemGenericAllocationHandle h;
      int held;
      if (hold_handle(gi, &h, &held) != 0) {
        mclog("RESUME: no handle for resident export idx=%d", gi);
        return -1;
      }
      int rc = reexport(gi, h);
      if (held) r_cuMemRelease(h);
      if (rc != 0) return -1;
      published++;
    }
  }

  /* Phase 2: importers fetch and re-import. */
  for (int gi = 0; gi < MAXN; gi++) {
    Alloc* a = &g_alloc[gi];
    if ((a->kind == KIND_MC && a->imported) || a->kind == KIND_IMP) {
      if (reimport(gi) != 0) return -1;
      if (a->kind == KIND_MC)
        groups++;
      else
        imports++;
    }
  }

  /* Phase 3a: re-add devices. AddDevice does not block. */
  for (int gi = 0; gi < MAXN; gi++) {
    Alloc* a = &g_alloc[gi];
    if (a->kind != KIND_MC) continue;
    for (int d = 0; d < a->ndev; d++)
      if (use_dev(a->dev) != 0 ||
          r_cuMulticastAddDevice(a->handle, a->devs[d]) != CUDA_SUCCESS) {
        mclog("RESUME: AddDevice dev=%d failed", a->devs[d]);
        return -1;
      }
  }

  /* Phase 3b: re-bind. */
  for (int b = 0; b < MAXN; b++) {
    Bind* bp = &g_bind[b];
    if (!bp->used) continue;
    CUmemGenericAllocationHandle mc = g_alloc[bp->groupIdx].handle, mem = 0;
    int held = 0;
    if (!bp->by_addr && hold_handle(bp->memIdx, &mem, &held) != 0) {
      mclog("RESUME: no handle for bound memory idx=%d", bp->memIdx);
      return -1;
    }
    CUresult rc = use_dev(bp->dev);
    if (rc == 0) {
      if (bp->v2 && bp->by_addr)
        rc = r_cuMulticastBindAddr_v2(mc, bp->dev, bp->mcOffset, bp->va,
                                      bp->size, 0);
      else if (bp->v2)
        rc = r_cuMulticastBindMem_v2(mc, bp->dev, bp->mcOffset, mem,
                                     bp->memOffset, bp->size, 0);
      else if (bp->by_addr)
        rc = r_cuMulticastBindAddr(mc, bp->mcOffset, bp->va, bp->size, 0);
      else
        rc = r_cuMulticastBindMem(mc, bp->mcOffset, mem, bp->memOffset,
                                  bp->size, 0);
    }
    if (held) r_cuMemRelease(mem);
    if (rc != CUDA_SUCCESS) {
      mclog("RESUME: re-bind (%s%s, dev=%d) rc=%d",
            bp->by_addr ? "addr" : "mem", bp->v2 ? "_v2" : "", bp->dev, rc);
      return -1;
    }
    rebound++;
  }

  /* Phase 3c: re-map groups and imports at their identical addresses. */
  for (int gi = 0; gi < MAXN; gi++) {
    int k = g_alloc[gi].kind;
    if ((k == KIND_MC || k == KIND_IMP) &&
        remap_alloc(gi, k == KIND_MC ? "MC" : "import", &remapped) != 0)
      return -1;
  }

  /* Phase 4: hand the references back. The shim's one reference stands for
   * the first of the application's; more are retained from a mapping. */
  for (int gi = 0; gi < MAXN; gi++) {
    Alloc* a = &g_alloc[gi];
    if (a->kind == KIND_FREE || !a->shim_ref) continue;
    if (use_dev(a->dev) != 0) return -1;
    CUresult rc = CUDA_SUCCESS;
    if (a->app_refs == 0) rc = r_cuMemRelease(a->handle);
    int m = first_map(gi);
    for (int k = 1; k < a->app_refs && rc == CUDA_SUCCESS; k++) {
      CUmemGenericAllocationHandle h = 0;
      rc = m < 0 ? -1
                 : r_cuMemRetainAllocationHandle(&h,
                                                 (void*)(uintptr_t)g_map[m].va);
      if (rc == CUDA_SUCCESS && h != a->handle) rc = -1;
    }
    if (rc != CUDA_SUCCESS) {
      mclog("RESUME: restoring %d reference(s) to idx=%d rc=%d", a->app_refs,
            gi, rc);
      return -1;
    }
    a->shim_ref = 0;
  }

  mclog(
      "RESUME done: groups=%d imports=%d published=%d rebound=%d "
      "remapped=%d",
      groups, imports, published, rebound, remapped);
  return 0;
}

/* Must hold g_lock. Run a transition with the caller's context restored and
 * the primary contexts released afterwards. */
static int run_transition(int (*fn)(void)) {
  CUcontext saved = NULL;
  r_cuCtxGetCurrent(&saved);
  int rc = fn();
  r_cuCtxSetCurrent(saved);
  release_devs();
  return rc;
}

/* Lookup interposition: torch, NCCL and ctypes resolve driver entry points with
 * dlsym, cuGetProcAddress or cudart's resolvers, bypassing symbol
 * interposition. The address a lookup returns identifies the exact exported
 * symbol, and so its ABI version, so it is redirected to the wrapper of that
 * symbol. */

CUresult cuGetProcAddress(const char*, void**, int, unsigned long long);
CUresult cuGetProcAddress_v2(const char*, void**, int, unsigned long long,
                             int*);
static CUresult (*r_cuGetProcAddress)(const char*, void**, int,
                                      unsigned long long);
static CUresult (*r_cuGetProcAddress_v2)(const char*, void**, int,
                                         unsigned long long, int*);

typedef struct {
  const char* name;
  void* wrapper;
  void** real;
  int tracked; /* an unknown ABI of this entry point refuses checkpoints */
} Hook;

#define HOOK(name, tracked) {#name, (void*)name, (void**)&r_##name, tracked},
#define GATED_HOOK(name, sfx, proto, args) HOOK(name, 1) HOOK(name##sfx, 1)
#define REFUSED_HOOK(name, proto, args, why) HOOK(name, 1)

static const Hook g_hooks[] = {
    HOOK(cuInit, 0)                                 /**/
    HOOK(cuDeviceGetAttribute, 0)                   /**/
    HOOK(cuGetProcAddress, 0)                       /**/
    HOOK(cuGetProcAddress_v2, 0)                    /**/
    HOOK(cuMemCreate, 1)                            /**/
    HOOK(cuMemRelease, 1)                           /**/
    HOOK(cuMemMap, 1)                               /**/
    HOOK(cuMemUnmap, 1)                             /**/
    HOOK(cuMemSetAccess, 1)                         /**/
    HOOK(cuMemRetainAllocationHandle, 1)            /**/
    HOOK(cuMemGetAllocationPropertiesFromHandle, 1) /**/
    HOOK(cuMemExportToShareableHandle, 1)           /**/
    HOOK(cuMemImportFromShareableHandle, 1)         /**/
    HOOK(cuMemMapArrayAsync, 1)                     /**/
    HOOK(cuMemMapArrayAsync_ptsz, 1)                /**/
    HOOK(cuMulticastCreate, 1)                      /**/
    HOOK(cuMulticastAddDevice, 1)                   /**/
    HOOK(cuMulticastBindMem, 1)                     /**/
    HOOK(cuMulticastBindAddr, 1)                    /**/
    HOOK(cuMulticastBindMem_v2, 1)                  /**/
    HOOK(cuMulticastBindAddr_v2, 1)                 /**/
    HOOK(cuMulticastUnbind, 1)                      /**/
    HOOK(cuLogicalEndpointBindMem, 1)               /**/
    GATED_LIST(GATED_HOOK)                          /**/
    WAIT_LIST(GATED_HOOK)                           /**/
    REFUSED_LIST(REFUSED_HOOK)                      /**/
};
#define NHOOKS ((int)(sizeof(g_hooks) / sizeof(g_hooks[0])))

static void resolve_hooks(void) {
  static int done;
  if (__atomic_load_n(&done, __ATOMIC_ACQUIRE)) return;
  init_real_dlsym();
  void* h = libcuda_handle();
  if (!real_dlsym || !h) return;
  for (int i = 0; i < NHOOKS; i++)
    if (!*g_hooks[i].real) *g_hooks[i].real = real_dlsym(h, g_hooks[i].name);
  resolve_internal();
  __atomic_store_n(&done, 1, __ATOMIC_RELEASE);
}

/* Length of name without a _ptsz/_ptds suffix and then a _v<N> suffix. */
static size_t base_len(const char* name, int* suffixed) {
  size_t n = strlen(name);
  *suffixed = 0;
  if (n > 5 && (strcmp(name + n - 5, "_ptsz") == 0 ||
                strcmp(name + n - 5, "_ptds") == 0)) {
    n -= 5;
    *suffixed = 1;
  }
  size_t k = n;
  while (k > 0 && name[k - 1] >= '0' && name[k - 1] <= '9') k--;
  if (k < n && k >= 2 && name[k - 1] == 'v' && name[k - 2] == '_') {
    n = k - 2;
    *suffixed = 1;
  }
  return n;
}

/* Whether symbol is a variant of a tracked entry point. Unsuffixed dlsym names
 * are the pre-3.2 32-bit ABI where a newer one exists; nothing resolves those
 * through the driver resolvers. */
static int tracked_symbol(const char* symbol, int from_resolver) {
  int suffixed, hsuffixed;
  size_t n = base_len(symbol, &suffixed);
  if (!suffixed && !from_resolver) return 0;
  for (int i = 0; i < NHOOKS; i++) {
    if (!g_hooks[i].tracked) continue;
    size_t hn = base_len(g_hooks[i].name, &hsuffixed);
    if (hn == n && strncmp(g_hooks[i].name, symbol, n) == 0) return 1;
  }
  return 0;
}

/* The wrapper for the libcuda function at p, or p. */
static void* redirect(const char* symbol, void* p, int from_resolver) {
  if (!p) return p;
  for (int i = 0; i < NHOOKS; i++)
    if (g_hooks[i].wrapper == p) return p;
  if (!libcuda_loaded()) return p;
  resolve_hooks();
  for (int i = 0; i < NHOOKS; i++)
    if (*g_hooks[i].real == p) return g_hooks[i].wrapper;
  if (symbol && tracked_symbol(symbol, from_resolver)) {
    const char* prev = NULL;
    if (__atomic_compare_exchange_n(&g_bad_lookup, &prev,
                                    "lookup of an unknown ABI version of a "
                                    "tracked entry point",
                                    0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
      mclog(
          "NOTE: checkpoint disabled for this process: no wrapper for the "
          "ABI %s resolved to",
          symbol);
  }
  return p;
}

CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion,
                          unsigned long long flags) {
  REAL(r_cuGetProcAddress, "cuGetProcAddress");
  if (!r_cuGetProcAddress) return CUDA_ERROR_NOT_INITIALIZED;
  CUresult rc = r_cuGetProcAddress(symbol, pfn, cudaVersion, flags);
  if (rc == CUDA_SUCCESS && pfn) *pfn = redirect(symbol, *pfn, 1);
  return rc;
}

CUresult cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion,
                             unsigned long long flags, int* symbolStatus) {
  REAL(r_cuGetProcAddress_v2, "cuGetProcAddress_v2");
  if (!r_cuGetProcAddress_v2) return CUDA_ERROR_NOT_INITIALIZED;
  CUresult rc =
      r_cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus);
  if (rc == CUDA_SUCCESS && pfn) *pfn = redirect(symbol, *pfn, 1);
  return rc;
}

/* Interposed cudart resolvers. torch >= 2.11 resolves its driver API through
 * cudaGetDriverEntryPointByVersion, and cudart reaches libcuda by a path
 * neither hook sees, so interpose the resolver, let cudart look the symbol up,
 * and redirect the result. The reals live in libcudart: resolve via RTLD_NEXT,
 * then in an already-loaded libcudart (torch dlopens it RTLD_LOCAL). Never
 * force-load it. */

#define CUDA_ERROR_RT_SYMBOL_NOT_FOUND 500 /* cudaErrorSymbolNotFound */

static void* libcudart_sym(const char* name) {
  init_real_dlsym();
  if (!real_dlsym) return NULL;
  void* s = real_dlsym(RTLD_NEXT, name);
  if (s) return s;
  static void* h;
  if (!h) {
    static const char* const sonames[] = {"libcudart.so.13", "libcudart.so.12",
                                          "libcudart.so.11.0", "libcudart.so",
                                          NULL};
    for (int i = 0; !h && sonames[i]; i++)
      h = dlopen(sonames[i], RTLD_NOW | RTLD_NOLOAD);
  }
  return h ? real_dlsym(h, name) : NULL;
}

#define RTREAL(var, name)                                \
  do {                                                   \
    if (!(var)) *(void**)(&(var)) = libcudart_sym(name); \
  } while (0)

int cudaGetDriverEntryPoint(const char* symbol, void** pfn,
                            unsigned long long flags, int* driverStatus) {
  static int (*real)(const char*, void**, unsigned long long, int*);
  RTREAL(real, "cudaGetDriverEntryPoint");
  if (!real) return CUDA_ERROR_RT_SYMBOL_NOT_FOUND;
  int rc = real(symbol, pfn, flags, driverStatus);
  if (rc == 0 && pfn) *pfn = redirect(symbol, *pfn, 1);
  return rc;
}

int cudaGetDriverEntryPoint_ptsz(const char* symbol, void** pfn,
                                 unsigned long long flags, int* driverStatus) {
  static int (*real)(const char*, void**, unsigned long long, int*);
  RTREAL(real, "cudaGetDriverEntryPoint_ptsz");
  if (!real) return CUDA_ERROR_RT_SYMBOL_NOT_FOUND;
  int rc = real(symbol, pfn, flags, driverStatus);
  if (rc == 0 && pfn) *pfn = redirect(symbol, *pfn, 1);
  return rc;
}

int cudaGetDriverEntryPointByVersion(const char* symbol, void** pfn,
                                     unsigned int cudaVersion,
                                     unsigned long long flags,
                                     int* driverStatus) {
  static int (*real)(const char*, void**, unsigned int, unsigned long long,
                     int*);
  RTREAL(real, "cudaGetDriverEntryPointByVersion");
  if (!real) return CUDA_ERROR_RT_SYMBOL_NOT_FOUND;
  int rc = real(symbol, pfn, cudaVersion, flags, driverStatus);
  if (rc == 0 && pfn) *pfn = redirect(symbol, *pfn, 1);
  return rc;
}

int cudaGetDriverEntryPointByVersion_ptsz(const char* symbol, void** pfn,
                                          unsigned int cudaVersion,
                                          unsigned long long flags,
                                          int* driverStatus) {
  static int (*real)(const char*, void**, unsigned int, unsigned long long,
                     int*);
  RTREAL(real, "cudaGetDriverEntryPointByVersion_ptsz");
  if (!real) return CUDA_ERROR_RT_SYMBOL_NOT_FOUND;
  int rc = real(symbol, pfn, cudaVersion, flags, driverStatus);
  if (rc == 0 && pfn) *pfn = redirect(symbol, *pfn, 1);
  return rc;
}

/* Covers apps that dlsym the resolvers from a dlopen'd libcudart. */
static void* cudart_wrapper(const char* name) {
  static const struct {
    const char* name;
    void* fn;
  } t[] = {
      {"cudaGetDriverEntryPoint", (void*)cudaGetDriverEntryPoint},
      {"cudaGetDriverEntryPoint_ptsz", (void*)cudaGetDriverEntryPoint_ptsz},
      {"cudaGetDriverEntryPointByVersion",
       (void*)cudaGetDriverEntryPointByVersion},
      {"cudaGetDriverEntryPointByVersion_ptsz",
       (void*)cudaGetDriverEntryPointByVersion_ptsz},
  };
  for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++)
    if (strcmp(t[i].name, name) == 0) return t[i].fn;
  return NULL;
}

/* Interposed dlsym. Delegating through a dlvsym-resolved dlsym re-anchors
 * RTLD_NEXT at mcshim, which can confuse interposers stacked after it. */
void* dlsym(void* handle, const char* symbol) {
  init_real_dlsym();
  if (!real_dlsym) return NULL;
  void* r = real_dlsym(handle, symbol);
  if (!r || symbol[0] != 'c' || symbol[1] != 'u') return r;
  if (strncmp(symbol, "cudaGetDriverEntryPoint", 23) == 0) {
    void* w = cudart_wrapper(symbol);
    return w ? w : r;
  }
  return redirect(symbol, r, 0);
}

/* Control thread: polls /tmp/mcshim for markers. */

static void marker(const char* name, char* out, size_t n) {
  snprintf(out, n, "%s/%s", g_dir, name);
}

static int marker_exists(const char* name) {
  char p[600];
  marker(name, p, sizeof(p));
  return access(p, F_OK) == 0;
}

static void marker_rm(const char* name) {
  char p[600];
  marker(name, p, sizeof(p));
  unlink(p);
}

static void marker_write(const char* name, const char* body) {
  char p[600];
  marker(name, p, sizeof(p));
  FILE* f = fopen(p, "w");
  if (f) {
    fputs(body, f);
    fputc('\n', f);
    fclose(f);
  }
}

typedef struct {
  char suspended[64], resumed[64], error[64], gated[64];
} Acks;

/* Whether this process's state is torn down: from a successful suspend until
 * a successful resume. Control thread only; part of the checkpoint image. */
static int g_torn;

/* Arm the gate, drain it, and check that the state can be carried. Returns
 * why not, or NULL. A refusal disarms the gate, unless the process is torn
 * down or broken: then it must stay gated. */
static const char* preflight(void) {
  const char* why = NULL;
  if (g_broken) {
    why = "an earlier transition failed";
  } else if (gate_arm() != 0) {
    /* Without g_lock: the call that did not drain may hold it. */
    why = "application CUDA calls did not drain";
  } else {
    pthread_mutex_lock(&g_lock);
    resolve_hooks();
    why = can_carry();
    pthread_mutex_unlock(&g_lock);
  }
  if (why) {
    mclog("GATE: refusing: %s", why);
    if (!g_torn && !g_broken) gate_disarm();
  }
  return why;
}

/* The gate appeared. The refusal comes before any process tears anything
 * down, so the sentry can still unwind. The sentry arms the gate before it
 * locks the processes, and locks every one of them, which drains their GPU
 * work, before it requests the suspend. */
static void on_gate_up(const Acks* k) {
  /* A refusal from an earlier attempt must not fail this one. */
  marker_rm(k->error);
  const char* why = preflight();
  marker_write(why ? k->error : k->gated, why ? why : "ok");
}

/* Release the application: the gate is gone and nothing is torn down, so
 * every process has resumed. Withdraw the published fds. */
static void release(const Acks* k) {
  pthread_mutex_lock(&g_lock);
  unpublish_all();
  pthread_mutex_unlock(&g_lock);
  gate_disarm();
  marker_rm(k->gated);
}

static void on_suspend_edge(const Acks* k, int want) {
  /* Drop a stale error ack: the sentry fails fast on error.<pid>. */
  marker_rm(k->error);
  const char* why = NULL;
  pthread_mutex_lock(&g_lock);
  resolve_hooks();
  if (g_broken) {
    why = "an earlier transition failed";
  } else if (want && !__atomic_load_n(&g_suspended, __ATOMIC_SEQ_CST)) {
    /* The sentry always gates first. */
    why = "suspend without the gate";
  } else if (want || g_torn) {
    /* A failure leaves this process torn down partway, and its peers may have
     * released state it needs, so it stays gated for good: better blocked
     * than corrupt. The sentry does not unwind it. */
    if (run_transition(want ? suspend_locked : resume_locked) != 0) {
      g_broken = 1;
      why = want ? "suspend failed" : "resume failed";
    }
  }
  pthread_mutex_unlock(&g_lock);
  if (why) {
    mclog("%s: %s", want ? "SUSPEND" : "RESUME", why);
    marker_write(k->error, why);
    return;
  }
  g_torn = want;
  marker_rm(want ? k->resumed : k->suspended);
  marker_write(want ? k->suspended : k->resumed, "ok");
}

/* Edge-triggered on marker existence: "gate" appearing gates and acks
 * gated.<pid>; "suspend" appearing suspends and acks suspended.<pid>,
 * disappearing resumes and acks resumed.<pid>; failures ack error.<pid>. The
 * markers are in the checkpoint image, so after a restore the shim stays
 * suspended until the sentry removes them.
 *
 * The release is level-triggered on state: the application runs again once
 * the gate is gone and nothing is torn down. The sentry removes the gate after
 * every process has resumed: a bind waits for every device to be added, not
 * for every rank's memory to be bound, so a rank released at its own resume
 * could reach a group that a peer is still binding. */
static void* control_thread(void* arg) {
  (void)arg;
  Acks k;
  int pid = (int)getpid();
  snprintf(k.suspended, sizeof(k.suspended), "suspended.%d", pid);
  snprintf(k.resumed, sizeof(k.resumed), "resumed.%d", pid);
  snprintf(k.error, sizeof(k.error), "error.%d", pid);
  snprintf(k.gated, sizeof(k.gated), "gated.%d", pid);
  /* present.<pid> tells the sentry that this process will ack. The sentry
   * selects CUDA processes by their open NVIDIA fds, a broader set. */
  char present[64];
  snprintf(present, sizeof(present), "present.%d", pid);
  /* Clear acks left by a dead process with the same pid. */
  marker_rm(k.suspended);
  marker_rm(k.resumed);
  marker_rm(k.error);
  marker_rm(k.gated);
  marker_write(present, "ok");
  mclog("control thread started (dir=%s)", g_dir);
  int prev_suspend = 0, prev_gate = 0, logged = 0;
  for (;;) {
    /* Gate first, so that a process that starts while both markers exist is
     * gated before it suspends. */
    int gate = marker_exists("gate");
    if (gate && !prev_gate) on_gate_up(&k);
    prev_gate = gate;
    int want = marker_exists("suspend");
    if (want != prev_suspend) {
      prev_suspend = want;
      on_suspend_edge(&k, want);
    }
    if (!gate && !g_torn && __atomic_load_n(&g_suspended, __ATOMIC_SEQ_CST)) {
      if (!g_broken)
        release(&k);
      else if (!logged++)
        mclog(
            "FATAL: refusing to release the gate after a failed transition; "
            "the application would run over unmapped GPU state");
    }
    /* Poll every 5 ms: the spread in when ranks see the gate bounds how often a
     * collective straddles it, which fails the sentry's lock. */
    struct timespec ts = {0, 5 * 1000 * 1000}; /* 5ms */
    nanosleep(&ts, NULL);
  }
  return NULL;
}

static int g_disabled;

/* Start the control thread only in processes that initialize CUDA, so that
 * helpers inheriting LD_PRELOAD (shells, runsc exec, cuda-checkpoint) never
 * consume or ack markers. */
static void control_thread_start(void) {
  if (g_disabled) return;
  pthread_t t;
  int err = pthread_create(&t, NULL, control_thread, NULL);
  if (err == 0)
    pthread_detach(t);
  else
    /* Otherwise this surfaces as an unexplained sentry ack timeout. */
    mclog(
        "FATAL: control thread creation failed: %s -- this process will "
        "never acknowledge suspend/resume markers",
        strerror(err));
}

/* Not pthread_once, which cannot be reset in a fork child. Guarded by g_lock.
 */
static int g_control_started;

static void ensure_control_thread(void) {
  pthread_mutex_lock(&g_lock);
  int need = !g_control_started;
  g_control_started = 1;
  pthread_mutex_unlock(&g_lock);
  if (need) control_thread_start();
}

/* Fork: a lock held by another parent thread would deadlock the child, and a
 * child of a process that initialized CUDA cannot use it, so the shim stays
 * inactive there. Inherited published fds would block the parent's next
 * checkpoint. */
static void mcshim_atfork_child(void) {
  g_loglock = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
  g_lock = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
  g_gate_lock = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
  g_gate_cv = (pthread_cond_t)PTHREAD_COND_INITIALIZER;
  g_drain_cv = (pthread_cond_t)PTHREAD_COND_INITIALIZER;
  __atomic_store_n(&g_suspended, 0, __ATOMIC_SEQ_CST);
  __atomic_store_n(&g_inflight, 0, __ATOMIC_SEQ_CST);
  memset(g_pctx, 0, sizeof(g_pctx));
  if (g_control_started) g_disabled = 1;
  for (int i = 0; i < MAXN; i++) {
    if (g_alloc[i].kind != KIND_FREE && g_alloc[i].pub_fd >= 0)
      close(g_alloc[i].pub_fd);
    g_alloc[i].pub_fd = -1;
  }
}

__attribute__((constructor)) static void mcshim_init(void) {
  pthread_atfork(NULL, NULL, mcshim_atfork_child);
  if (getenv("MCSHIM_DISABLE")) {
    /* Silent: the sentry parses the output of the cuda-checkpoint processes it
     * runs with MCSHIM_DISABLE. */
    g_disabled = 1;
    return;
  }
  /* Create the control dir, which may also hold MCSHIM_LOG, before the first
   * mclog. */
  mkdir(g_dir, 0777);
  unmark_stale_exports();
  /* Markers belong to the sentry; the control thread starts from cuInit. */
  mclog("loaded; control dir=%s", g_dir);
}
