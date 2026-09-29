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
 * The sentry drives it through marker files in $MCSHIM_DIR (see
 * pkg/sentry/control/state_cuda_shim.go). After a rebuild, exporters publish
 * the re-exported fd under the original export's identity, and importers copy
 * it with pidfd_getfd(2) and re-import it. Unicast device memory stays
 * cuda-checkpoint's responsibility, and so does legacy CUDA IPC (cuIpc*),
 * which cuda-checkpoint carries when the processes share a job (runsc
 * --cuda-checkpoint-path).
 *
 * Besides symbol interposition, the shim interposes dlsym, cuGetProcAddress and
 * cudart's cudaGetDriverEntryPoint* resolvers, through which torch, NCCL and
 * ctypes resolve driver entry points. */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
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
typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUmemGenericAllocationHandle;

#define CUDA_SUCCESS 0
/* Returned by an import when the process cannot address the exporting device.
 */
#define CUDA_ERROR_INVALID_DEVICE 101

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

/* Logging. */

static FILE* g_log;
static pthread_mutex_t g_loglock = PTHREAD_MUTEX_INITIALIZER;

/* Started lazily from cuInit, so only CUDA users poll for markers. */
static void ensure_control_thread(void);
static void gate_wait(void);

static void mclog(const char* fmt, ...) {
  pthread_mutex_lock(&g_loglock);
  if (!g_log) {
    const char* p = getenv("MCSHIM_LOG");
    g_log = p && *p ? fopen(p, "a") : stderr;
    if (!g_log) g_log = stderr;
  }
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  struct tm tm;
  localtime_r(&ts.tv_sec, &tm);
  char t[32];
  strftime(t, sizeof(t), "%H:%M:%S", &tm);
  fprintf(g_log, "[mcshim %s.%03ld pid=%d] ", t, ts.tv_nsec / 1000000,
          (int)getpid());
  va_list ap;
  va_start(ap, fmt);
  vfprintf(g_log, fmt, ap);
  va_end(ap);
  fputc('\n', g_log);
  fflush(g_log);
  pthread_mutex_unlock(&g_loglock);
}

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

#define REAL(var, name)                                               \
  do {                                                                \
    if (!(var)) {                                                     \
      init_real_dlsym();                                              \
      void* h_ = libcuda_handle();                                    \
      if (real_dlsym && h_) *(void**)(&(var)) = real_dlsym(h_, name); \
    }                                                                 \
  } while (0)

static CUresult (*r_cuMemCreate)(CUmemGenericAllocationHandle*, size_t,
                                 const CUmemAllocationProp*,
                                 unsigned long long);
static CUresult (*r_cuMemRelease)(CUmemGenericAllocationHandle);
static CUresult (*r_cuMemMap)(CUdeviceptr, size_t, size_t,
                              CUmemGenericAllocationHandle, unsigned long long);
static CUresult (*r_cuMemUnmap)(CUdeviceptr, size_t);
static CUresult (*r_cuMemSetAccess)(CUdeviceptr, size_t, const CUmemAccessDesc*,
                                    size_t);
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
static CUresult (*r_cuMulticastUnbind)(CUmemGenericAllocationHandle, CUdevice,
                                       size_t, size_t);
static CUresult (*r_cuCtxGetDevice)(CUdevice*);
static CUresult (*r_cuMemExportToShareableHandle)(void*,
                                                  CUmemGenericAllocationHandle,
                                                  int, unsigned long long);
static CUresult (*r_cuMemImportFromShareableHandle)(
    CUmemGenericAllocationHandle*, void*, int);
static CUresult (*r_cuCtxGetCurrent)(CUcontext*);
static CUresult (*r_cuCtxSetCurrent)(CUcontext);
static CUresult (*r_cuCtxSynchronize)(void);
static CUresult (*r_cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
static CUresult (*r_cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
static CUresult (*r_cuDeviceGetAttribute)(int*, int, CUdevice);

#define CU_MEM_HANDLE_TYPE_POSIX_FD 0x1
#define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED 128

static void resolve_reals(void) {
  REAL(r_cuMemCreate, "cuMemCreate");
  REAL(r_cuMemRelease, "cuMemRelease");
  REAL(r_cuMemMap, "cuMemMap");
  REAL(r_cuMemUnmap, "cuMemUnmap");
  REAL(r_cuMemSetAccess, "cuMemSetAccess");
  REAL(r_cuMulticastCreate, "cuMulticastCreate");
  REAL(r_cuMulticastAddDevice, "cuMulticastAddDevice");
  REAL(r_cuMulticastBindMem, "cuMulticastBindMem");
  REAL(r_cuMulticastBindAddr, "cuMulticastBindAddr");
  REAL(r_cuMulticastUnbind, "cuMulticastUnbind");
  REAL(r_cuCtxGetDevice, "cuCtxGetDevice");
  REAL(r_cuMemExportToShareableHandle, "cuMemExportToShareableHandle");
  REAL(r_cuMemImportFromShareableHandle, "cuMemImportFromShareableHandle");
  REAL(r_cuCtxGetCurrent, "cuCtxGetCurrent");
  REAL(r_cuCtxSetCurrent, "cuCtxSetCurrent");
  REAL(r_cuCtxSynchronize, "cuCtxSynchronize");
  REAL(r_cuDeviceGetAttribute, "cuDeviceGetAttribute");
  REAL(r_cuMemcpyDtoH, "cuMemcpyDtoH_v2");
  REAL(r_cuMemcpyHtoD, "cuMemcpyHtoD_v2");
}

/* Tracked state: the live object graph. Frees remove entries, so freed objects
 * drop out of the replay set. */

/* Static tables (about 3 MB per process) keep hot paths allocation-free. An
 * SGLang TP=8 rank tracks more than 512 objects. */
#define MAXN 4096
#define MAX_DEV 16

/* KIND_IMP is an import; cuMulticastAddDevice on it proves it a multicast group
 * and makes it KIND_MC. */
enum { KIND_FREE = 0, KIND_UC = 1, KIND_MC = 2, KIND_IMP = 3 };

typedef struct {
  int kind;
  CUmemGenericAllocationHandle handle; /* current handle */
  CUmemGenericAllocationHandle orig;   /* the handle the application holds */
  size_t size;
  CUcontext ctx;
  CUmemAllocationProp uprop;   /* KIND_UC */
  CUmulticastObjectProp mprop; /* KIND_MC */
  int devs[MAX_DEV];           /* KIND_MC: added devices */
  int ndev;
  /* Rendezvous identity of the export (see record_key). */
  int imported; /* 1 = handle came from an import */
  int has_key;
  unsigned long key_client, key_object;
  /* After a resume: the re-exported fd, published until the gate is removed
   * (see publish_fd). */
  int pub_fd;
  /* Contents of a multicast-bound exporter freed across the checkpoint (see
   * do_suspend); NULL if none. */
  void* uc_content;
  /* Set once suspend has released the object; resume rebuilds only these.
   * Cleared as soon as the object is live again, so retries after a partial
   * failure redo exactly the remaining work. */
  int torn_down;
} Alloc;

typedef struct {
  int used;
  CUdeviceptr va;
  size_t size;
  size_t offset;
  int allocIdx; /* index into g_alloc of the mapped handle */
  /* Access set last applied to this mapping, replayed at resume. */
#define MAX_ACCESS 16
  CUmemAccessDesc access[MAX_ACCESS];
  int naccess;
  CUcontext ctx;
  /* Set as each unmap succeeds, cleared as each re-map succeeds (see
   * torn_down). */
  int suspended;
} Mapping;

typedef struct {
  int used;
  int groupIdx; /* index into g_alloc of the MC group */
  int by_addr;  /* 1 = cuMulticastBindAddr (replay by VA), 0 = BindMem */
  CUmemGenericAllocationHandle mem; /* BindMem: UC handle (stable) */
  CUdeviceptr va; /* BindAddr: bound VA (stable across restore) */
  size_t mcOffset;
  size_t memOffset;
  size_t size;
  CUcontext ctx;
  CUdevice dev; /* device hosting the memory (unbind is per-device) */
  /* Set as each unbind succeeds, cleared as each re-bind succeeds. */
  int unbound;
} Bind;

static Alloc g_alloc[MAXN];
static Mapping g_map[MAXN];
static Bind g_bind[MAXN];
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
/* All accesses are atomic or under g_gate_lock (see the suspend gate). */
static int g_suspended;

/* Sticky: some state could not be tracked (a table overflow), so do_suspend
 * refuses. Must hold g_lock. */
static int g_untracked;
static const char* g_untracked_why;

static void mark_untracked(const char* why) {
  if (!g_untracked) {
    g_untracked = 1;
    g_untracked_why = why;
  }
}

static int alloc_new(void) {
  for (int i = 0; i < MAXN; i++)
    if (g_alloc[i].kind == KIND_FREE) return i;
  if (!g_untracked)
    mclog(
        "FATAL: alloc table full (MAXN=%d); object untracked -- "
        "suspend is disabled for this process",
        MAXN);
  mark_untracked("alloc table overflow");
  return -1;
}

/* Find alloc index whose CURRENT handle matches h. */
static int alloc_find(CUmemGenericAllocationHandle h) {
  for (int i = 0; i < MAXN; i++)
    if (g_alloc[i].kind != KIND_FREE && g_alloc[i].handle == h) return i;
  return -1;
}

/* Translate a possibly stale handle to its object's current one: apps and NCCL
 * keep original handles in their structs, and a rebuild rotates them. */
static CUmemGenericAllocationHandle xlate_mc(CUmemGenericAllocationHandle h) {
  for (int i = 0; i < MAXN; i++)
    if (g_alloc[i].kind != KIND_FREE && g_alloc[i].orig == h)
      return g_alloc[i].handle;
  return h;
}

/* Must hold g_lock. Record h as a's current handle (a may be NULL for an
 * untracked handle). The driver reuses handle values, so an object whose
 * original handle is issued anew stops translating it (vLLM's sleep/wake churn
 * makes this routine). */
static void set_handle(Alloc* a, CUmemGenericAllocationHandle h) {
  for (int i = 0; i < MAXN; i++)
    if (g_alloc[i].kind != KIND_FREE && g_alloc[i].orig == h)
      g_alloc[i].orig = g_alloc[i].handle;
  if (a) a->handle = h;
}

/* Must hold g_lock. Reset slot i and record the current context + handle. */
static Alloc* alloc_init(int i, int kind, CUmemGenericAllocationHandle h) {
  Alloc* a = &g_alloc[i];
  set_handle(NULL, h);
  memset(a, 0, sizeof(*a));
  a->kind = kind;
  a->handle = a->orig = h;
  a->pub_fd = -1;
  r_cuCtxGetCurrent(&a->ctx);
  return a;
}

/* Locked translation of a possibly-stale MC handle (see xlate_mc). */
static CUmemGenericAllocationHandle xlate_locked(
    CUmemGenericAllocationHandle h) {
  pthread_mutex_lock(&g_lock);
  CUmemGenericAllocationHandle r = xlate_mc(h);
  pthread_mutex_unlock(&g_lock);
  return r;
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

/* Must hold g_lock. Record the rendezvous identity from nvproxy's fdinfo line;
 * without it the object cannot be rebuilt, so checkpoints are refused. The
 * first key wins. */
static void record_key(int i, int fd) {
  Alloc* a = &g_alloc[i];
  if (a->has_key) return;
  if (fd < 0 || fdinfo_oracle(fd, &a->key_client, &a->key_object) != 0) {
    mark_untracked("export without an nvproxy identity");
    return;
  }
  a->has_key = 1;
}

/* Interposed entry points. */

CUresult cuInit(unsigned int flags) {
  static CUresult (*real)(unsigned int);
  REAL(real, "cuInit");
  if (!real) return 3; /* CUDA_ERROR_NOT_INITIALIZED */
  /* Only processes that initialize CUDA take part in the protocol. */
  ensure_control_thread();
  return real(flags);
}

#define CU_MEM_HANDLE_TYPE_FABRIC 0x8

CUresult cuMemCreate(CUmemGenericAllocationHandle* h, size_t size,
                     const CUmemAllocationProp* prop,
                     unsigned long long flags) {
  resolve_reals();
  gate_wait();
  /* Strip the FABRIC handle type: it creates an NV_MEMORY_FABRIC (00f8) object
   * at allocation time, which cuda-checkpoint cannot serialize. On a single
   * node POSIX fds are equivalent. Masking the device attribute is not enough,
   * since statically linked runtimes bypass it. */
  CUmemAllocationProp fixed;
  if (prop && (prop->requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC) &&
      !getenv("MCSHIM_ALLOW_FABRIC")) {
    fixed = *prop;
    fixed.requestedHandleTypes &= ~CU_MEM_HANDLE_TYPE_FABRIC;
    if (!fixed.requestedHandleTypes)
      fixed.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FD;
    prop = &fixed;
    static int logged;
    if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
      mclog(
          "stripping CU_MEM_HANDLE_TYPE_FABRIC from "
          "cuMemCreate (-> handleTypes 0x%x): fabric-handle "
          "memory is not checkpointable; set "
          "MCSHIM_ALLOW_FABRIC=1 to keep it",
          fixed.requestedHandleTypes);
  }
  CUresult rc = r_cuMemCreate(h, size, prop, flags);
  if (rc == CUDA_SUCCESS) {
    pthread_mutex_lock(&g_lock);
    int i = alloc_new();
    if (i >= 0) {
      Alloc* a = alloc_init(i, KIND_UC, *h);
      a->size = size;
      if (prop) a->uprop = *prop;
    } else {
      /* Untracked, but its value must not translate to a stale object. */
      set_handle(NULL, *h);
    }
    pthread_mutex_unlock(&g_lock);
  }
  return rc;
}

CUresult cuMulticastCreate(CUmemGenericAllocationHandle* h,
                           const CUmulticastObjectProp* prop) {
  resolve_reals();
  gate_wait();
  CUresult rc = r_cuMulticastCreate(h, prop);
  if (rc == CUDA_SUCCESS) {
    pthread_mutex_lock(&g_lock);
    int i = alloc_new();
    if (i >= 0) {
      Alloc* a = alloc_init(i, KIND_MC, *h);
      if (prop) {
        a->mprop = *prop;
        a->size = prop->size;
      }
    } else {
      set_handle(NULL, *h);
    }
    pthread_mutex_unlock(&g_lock);
  }
  return rc;
}

#define CUDA_ERROR_NOT_SUPPORTED 801

CUresult cuMemExportToShareableHandle(void* shHandle,
                                      CUmemGenericAllocationHandle h, int type,
                                      unsigned long long flags) {
  resolve_reals();
  gate_wait();
  /* Refuse fabric exports. The driver fabric-exports memory that never asked
   * for fabric handles, so torch's export-as-FABRIC probe would succeed and
   * create one 00f8 object per pool chunk. Failing it makes torch fall back to
   * POSIX fds. */
  if (type == CU_MEM_HANDLE_TYPE_FABRIC && !getenv("MCSHIM_ALLOW_FABRIC")) {
    static int logged;
    if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
      mclog(
          "refusing fabric-typed cuMemExportToShareableHandle:"
          " fabric exports are not checkpointable; set "
          "MCSHIM_ALLOW_FABRIC=1 to permit them");
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUmemGenericAllocationHandle real_h = xlate_locked(h);
  CUresult rc = r_cuMemExportToShareableHandle(shHandle, real_h, type, flags);
  if (rc == CUDA_SUCCESS && type == CU_MEM_HANDLE_TYPE_POSIX_FD && shHandle) {
    pthread_mutex_lock(&g_lock);
    int i = alloc_find(real_h);
    /* Multicast groups and unicast (P2P) exports alike must be re-exported and
     * published on resume. */
    if (i >= 0 && (g_alloc[i].kind == KIND_MC || g_alloc[i].kind == KIND_UC)) {
      record_key(i, *(int*)shHandle);
    }
    pthread_mutex_unlock(&g_lock);
  }
  return rc;
}

CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* h,
                                        void* osHandle, int type) {
  resolve_reals();
  gate_wait();
  CUresult rc = r_cuMemImportFromShareableHandle(h, osHandle, type);
  if (rc == CUDA_SUCCESS && type == CU_MEM_HANDLE_TYPE_POSIX_FD && h) {
    pthread_mutex_lock(&g_lock);
    int i = alloc_new();
    if (i >= 0) {
      Alloc* a = alloc_init(i, KIND_IMP, *h);
      a->imported = 1;
      /* For POSIX-FD imports osHandle is the fd. */
      record_key(i, (int)(intptr_t)osHandle);
    } else {
      set_handle(NULL, *h);
    }
    pthread_mutex_unlock(&g_lock);
  }
  return rc;
}

/* Report no fabric-handle support unless MCSHIM_ALLOW_FABRIC=1, so that
 * frameworks choose POSIX fds. Fabric memory creates 00f8 objects that
 * cuda-checkpoint cannot serialize, and on a single node fds perform the same.
 */
CUresult cuDeviceGetAttribute(int* pi, int attrib, CUdevice dev) {
  resolve_reals();
  CUresult rc = r_cuDeviceGetAttribute(pi, attrib, dev);
  if (rc == CUDA_SUCCESS && pi &&
      attrib == CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED && *pi != 0 &&
      !getenv("MCSHIM_ALLOW_FABRIC")) {
    static int logged;
    if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
      mclog(
          "masking HANDLE_TYPE_FABRIC_SUPPORTED=0 (dev %d): "
          "fabric-handle exports are not checkpointable; "
          "set MCSHIM_ALLOW_FABRIC=1 to report the truth",
          dev);
    *pi = 0;
  }
  return rc;
}

CUresult cuMulticastAddDevice(CUmemGenericAllocationHandle h, CUdevice dev) {
  resolve_reals();
  gate_wait();
  CUmemGenericAllocationHandle real_h = xlate_locked(h);

  CUresult rc = r_cuMulticastAddDevice(real_h, dev);
  if (rc == CUDA_SUCCESS) {
    pthread_mutex_lock(&g_lock);
    int i = alloc_find(real_h);
    /* AddDevice proves that an imported handle is a multicast group. */
    if (i >= 0 && g_alloc[i].kind == KIND_IMP) g_alloc[i].kind = KIND_MC;
    if (i >= 0 && g_alloc[i].kind == KIND_MC && g_alloc[i].ndev < MAX_DEV)
      g_alloc[i].devs[g_alloc[i].ndev++] = dev;
    pthread_mutex_unlock(&g_lock);
  }
  return rc;
}

/* Must hold g_lock. Record a successful bind. */
static void bind_record(int gi, int by_addr, CUmemGenericAllocationHandle mem,
                        CUdeviceptr va, size_t mcOffset, size_t memOffset,
                        size_t size, CUdevice dev) {
  if (gi < 0) return;
  for (int b = 0; b < MAXN; b++) {
    if (g_bind[b].used) continue;
    g_bind[b].used = 1;
    g_bind[b].groupIdx = gi;
    g_bind[b].by_addr = by_addr;
    g_bind[b].mem = mem;
    g_bind[b].va = va;
    g_bind[b].mcOffset = mcOffset;
    g_bind[b].memOffset = memOffset;
    g_bind[b].size = size;
    g_bind[b].dev = dev;
    g_bind[b].unbound = 0;
    r_cuCtxGetCurrent(&g_bind[b].ctx);
    return;
  }
  mark_untracked("bind table overflow");
  mclog(
      "FATAL: bind table full (MAXN=%d); bind not tracked -- suspend "
      "is disabled for this process",
      MAXN);
}

CUresult cuMulticastBindMem(CUmemGenericAllocationHandle mc, size_t mcOffset,
                            CUmemGenericAllocationHandle mem, size_t memOffset,
                            size_t size, unsigned long long flags) {
  resolve_reals();
  gate_wait();
  CUmemGenericAllocationHandle real_mc = xlate_locked(mc);
  /* The bound memory may be a rotated import handle; record the current one. */
  CUmemGenericAllocationHandle real_mem = xlate_locked(mem);

  CUresult rc =
      r_cuMulticastBindMem(real_mc, mcOffset, real_mem, memOffset, size, flags);
  if (rc == CUDA_SUCCESS) {
    pthread_mutex_lock(&g_lock);
    /* Unbind is per device: the one hosting the memory. */
    CUdevice dev = -1;
    int mi = alloc_find(real_mem);
    if (mi >= 0 && g_alloc[mi].kind == KIND_UC)
      dev = g_alloc[mi].uprop.location.id;
    bind_record(alloc_find(real_mc), 0, real_mem, 0, mcOffset, memOffset, size,
                dev);
    pthread_mutex_unlock(&g_lock);
  }
  return rc;
}

CUresult cuMulticastBindAddr(CUmemGenericAllocationHandle mc, size_t mcOffset,
                             CUdeviceptr memptr, size_t size,
                             unsigned long long flags) {
  resolve_reals();
  gate_wait();
  CUmemGenericAllocationHandle real_mc = xlate_locked(mc);
  CUresult rc = r_cuMulticastBindAddr(real_mc, mcOffset, memptr, size, flags);
  if (rc == CUDA_SUCCESS) {
    pthread_mutex_lock(&g_lock);
    /* Replay is by VA; the hosting device is the caller's. */
    CUdevice dev = -1;
    r_cuCtxGetDevice(&dev);
    bind_record(alloc_find(real_mc), 1, 0, memptr, mcOffset, 0, size, dev);
    pthread_mutex_unlock(&g_lock);
  }
  return rc;
}

CUresult cuMulticastUnbind(CUmemGenericAllocationHandle mc, CUdevice dev,
                           size_t mcOffset, size_t size) {
  resolve_reals();
  gate_wait();
  CUmemGenericAllocationHandle real_mc = xlate_locked(mc);
  CUresult rc = r_cuMulticastUnbind(real_mc, dev, mcOffset, size);
  /* App-initiated: drop the recorded bind. The shim's own teardown calls the
   * reals. */
  if (rc == CUDA_SUCCESS && !__atomic_load_n(&g_suspended, __ATOMIC_SEQ_CST)) {
    pthread_mutex_lock(&g_lock);
    int gi = alloc_find(real_mc);
    for (int b = 0; b < MAXN; b++)
      if (g_bind[b].used && g_bind[b].groupIdx == gi && g_bind[b].dev == dev &&
          g_bind[b].mcOffset == mcOffset && g_bind[b].size == size)
        g_bind[b].used = 0;
    pthread_mutex_unlock(&g_lock);
  }
  return rc;
}

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle handle,
                  unsigned long long flags) {
  resolve_reals();
  gate_wait();
  CUmemGenericAllocationHandle real_h = xlate_locked(handle);
  CUresult rc = r_cuMemMap(ptr, size, offset, real_h, flags);
  if (rc == CUDA_SUCCESS && !__atomic_load_n(&g_suspended, __ATOMIC_SEQ_CST)) {
    pthread_mutex_lock(&g_lock);
    int ai = alloc_find(real_h);
    if (ai >= 0) {
      int placed = 0;
      for (int m = 0; m < MAXN; m++) {
        if (g_map[m].used) continue;
        g_map[m].used = 1;
        g_map[m].va = ptr;
        g_map[m].size = size;
        g_map[m].offset = offset;
        g_map[m].allocIdx = ai;
        g_map[m].naccess = 0;
        g_map[m].suspended = 0;
        r_cuCtxGetCurrent(&g_map[m].ctx);
        placed = 1;
        break;
      }
      if (!placed) {
        mark_untracked("mapping table overflow");
        mclog(
            "FATAL: mapping table full (MAXN=%d); "
            "mapping va=0x%llx untracked -- suspend "
            "is disabled for this process",
            MAXN, (unsigned long long)ptr);
      }
    }
    pthread_mutex_unlock(&g_lock);
  }
  return rc;
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
  resolve_reals();
  gate_wait();
  CUresult rc = r_cuMemUnmap(ptr, size);
  /* App-initiated: forget the mapping. */
  if (rc == CUDA_SUCCESS && !__atomic_load_n(&g_suspended, __ATOMIC_SEQ_CST)) {
    pthread_mutex_lock(&g_lock);
    for (int m = 0; m < MAXN; m++)
      if (g_map[m].used && g_map[m].va == ptr) g_map[m].used = 0;
    pthread_mutex_unlock(&g_lock);
  }
  return rc;
}

CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size,
                        const CUmemAccessDesc* desc, size_t count) {
  resolve_reals();
  gate_wait();
  CUresult rc = r_cuMemSetAccess(ptr, size, desc, count);
  if (rc == CUDA_SUCCESS && desc && count >= 1) {
    size_t n = count;
    if (n > MAX_ACCESS) {
      /* A prefix would silently narrow the access set at resume; log, and let
       * remap_alloc's owner-RW fallback apply. */
      mclog(
          "NOTE: cuMemSetAccess va=0x%llx count=%zu exceeds "
          "MAX_ACCESS=%d; access set NOT recorded (resume "
          "grants owner RW only)",
          (unsigned long long)ptr, count, MAX_ACCESS);
      n = 0;
    }
    /* Record on every tracked mapping in range: NCCL sets access once over a
     * reservation holding several maps. */
    pthread_mutex_lock(&g_lock);
    for (int m = 0; m < MAXN; m++) {
      if (!g_map[m].used || g_map[m].va < ptr ||
          g_map[m].va + g_map[m].size > ptr + size)
        continue;
      g_map[m].naccess = (int)n;
      if (n) memcpy(g_map[m].access, desc, n * sizeof(*desc));
    }
    pthread_mutex_unlock(&g_lock);
  }
  return rc;
}

static void unpublish_fd(Alloc* a);

/* Must hold g_lock. Forget alloc i and the binds and maps that reference it. */
static void alloc_forget(int i) {
  for (int b = 0; b < MAXN; b++)
    if (g_bind[b].used &&
        (g_bind[b].groupIdx == i || g_bind[b].mem == g_alloc[i].handle ||
         g_bind[b].mem == g_alloc[i].orig))
      g_bind[b].used = 0;
  for (int m = 0; m < MAXN; m++)
    if (g_map[m].used && g_map[m].allocIdx == i) g_map[m].used = 0;
  unpublish_fd(&g_alloc[i]);
  g_alloc[i].ctx = NULL; /* a freed slot must never look targetable */
  g_alloc[i].kind = KIND_FREE;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
  resolve_reals();
  gate_wait();
  CUmemGenericAllocationHandle real_h = xlate_locked(handle);
  CUresult rc = r_cuMemRelease(real_h);
  /* App-initiated: forget the alloc and its dependents. */
  if (rc == CUDA_SUCCESS && !__atomic_load_n(&g_suspended, __ATOMIC_SEQ_CST)) {
    pthread_mutex_lock(&g_lock);
    int i = alloc_find(real_h);
    if (i >= 0) alloc_forget(i);
    pthread_mutex_unlock(&g_lock);
  }
  return rc;
}

/* Cross-rank fd rendezvous. After a restore, each exporter re-exports its
 * object and publishes "<pid> <fd>" in $MCSHIM_DIR under the original export's
 * identity, and importers copy the fd with pidfd_getfd(2). That needs ptrace
 * access, which YAMA denies between sibling processes, so an exporter allows
 * any tracer (PR_SET_PTRACER_ANY) while it has fds published, which is until
 * the sentry removes the gate after every process has resumed. */

/* Default only; the sentry sets MCSHIM_DIR (see DefaultCudaMulticastShimDir).
 */
static char g_dir[512] = "/tmp/mcshim";
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

/* Suspend/resume helpers. */

/* Must hold g_lock. Unmap every VA that maps alloc gi, KEEPING the VA
 * reservations (cuMemUnmap only -- never cuMemAddressFree). */
static int unmap_alloc(int gi, const char* what, int* unmapped) {
  for (int m = 0; m < MAXN; m++) {
    if (!g_map[m].used || g_map[m].allocIdx != gi) continue;
    if (g_map[m].suspended)
      continue; /* already unmapped by an earlier attempt */
    if (g_map[m].ctx) r_cuCtxSetCurrent(g_map[m].ctx);
    CUresult rc = r_cuMemUnmap(g_map[m].va, g_map[m].size);
    if (rc != CUDA_SUCCESS) {
      mclog("SUSPEND: cuMemUnmap(%s 0x%llx) rc=%d", what,
            (unsigned long long)g_map[m].va, rc);
      return -1;
    }
    g_map[m].suspended = 1;
    (*unmapped)++;
  }
  return 0;
}

/* Must hold g_lock. Re-map every VA of alloc gi at the identical address,
 * backed by h, in the retained reservation. With partial set, gi is still live
 * after a failed suspend, so only what was unmapped is re-mapped. */
static int remap_alloc(int gi, CUmemGenericAllocationHandle h, const char* what,
                       int partial, int* remapped) {
  for (int m = 0; m < MAXN; m++) {
    if (!g_map[m].used || g_map[m].allocIdx != gi) continue;
    if (partial && !g_map[m].suspended)
      continue; /* still mapped; nothing to redo */
    if (g_map[m].ctx) r_cuCtxSetCurrent(g_map[m].ctx);
    CUresult rc = r_cuMemMap(g_map[m].va, g_map[m].size, g_map[m].offset, h, 0);
    if (rc != CUDA_SUCCESS) {
      mclog("RESUME: %s re-map at 0x%llx rc=%d", what,
            (unsigned long long)g_map[m].va, rc);
      return -1;
    }
    /* Replay the recorded access set. With none recorded, grant RW to the
     * owning device, which NCCL's imports and NVLS VAs need: an inaccessible
     * view faults the collective (719). */
    CUmemAccessDesc fallback;
    const CUmemAccessDesc* acc = g_map[m].access;
    size_t nacc = (size_t)g_map[m].naccess;
    if (nacc == 0) {
      CUdevice d = -1;
      r_cuCtxGetDevice(&d);
      memset(&fallback, 0, sizeof(fallback));
      fallback.location.type = 1 /* CU_MEM_LOCATION_TYPE_DEVICE */;
      fallback.location.id = d;
      fallback.flags = 3 /* CU_MEM_ACCESS_FLAGS_PROT_READWRITE */;
      acc = &fallback;
      nacc = 1;
    }
    CUresult ac = r_cuMemSetAccess(g_map[m].va, g_map[m].size, acc, nacc);
    if (ac != CUDA_SUCCESS) {
      mclog("RESUME: %s cuMemSetAccess(0x%llx) rc=%d", what,
            (unsigned long long)g_map[m].va, ac);
      return -1;
    }
    g_map[m].suspended = 0;
    (*remapped)++;
  }
  return 0;
}

/* Must hold g_lock. Re-export alloc gi's handle h and publish it for
 * importers. */
static int reexport(int gi, CUmemGenericAllocationHandle h) {
  /* A freshly restored allocation can transiently fail its export with
   * INVALID_VALUE; retry briefly (unretried, peers time out and fault with
   * 719). */
  int fd = -1;
  CUresult rc = 0;
  for (int attempt = 0; attempt < 100; attempt++) {
    rc = r_cuMemExportToShareableHandle(&fd, h, CU_MEM_HANDLE_TYPE_POSIX_FD, 0);
    if (rc == CUDA_SUCCESS && fd >= 0) break;
    if (attempt == 0) {
      CUcontext cur = NULL;
      r_cuCtxGetCurrent(&cur);
      mclog(
          "RESUME: re-export idx=%d kind=%d handle=0x%llx "
          "ctx=%p cur=%p rc=%d fd=%d, retrying",
          gi, g_alloc[gi].kind, (unsigned long long)h, g_alloc[gi].ctx, cur, rc,
          fd);
    }
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
static int reimport(int gi, CUmemGenericAllocationHandle* out) {
  if (!g_alloc[gi].has_key) {
    mclog("RESUME: imported idx=%d has no rendezvous key", gi);
    return -1;
  }
  for (int attempt = 0;; attempt++) {
    int fd = fetch_fd(&g_alloc[gi], 60 * 1000);
    if (fd < 0) return -1;
    CUresult rc = r_cuMemImportFromShareableHandle(out, (void*)(intptr_t)fd,
                                                   CU_MEM_HANDLE_TYPE_POSIX_FD);
    close(fd);
    if (rc == CUDA_SUCCESS) {
      if (attempt > 0)
        mclog(
            "RESUME: re-import idx=%d key=%lx:%lx ok "
            "after %d retries",
            gi, g_alloc[gi].key_client, g_alloc[gi].key_object, attempt);
      return 0;
    }
    /* Report the first failure as it happens, for correlation with the sentry's
     * logs. */
    if (attempt == 0) {
      CUdevice cur = -1;
      r_cuCtxGetDevice(&cur);
      mclog(
          "RESUME: re-import idx=%d key=%lx:%lx dev=%d "
          "rc=%d on first attempt",
          gi, g_alloc[gi].key_client, g_alloc[gi].key_object, cur, rc);
    }
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

/* Suspend. */

/* Must hold g_lock. Whether alloc gi's memory is bound into a tracked multicast
 * group (by handle, or for cuMulticastBindAddr by mapping VA). */
static int uc_is_mc_bound(int gi) {
  for (int b = 0; b < MAXN; b++) {
    if (!g_bind[b].used) continue;
    if (!g_bind[b].by_addr) {
      if (g_bind[b].mem == g_alloc[gi].handle ||
          g_bind[b].mem == g_alloc[gi].orig)
        return 1;
    } else {
      for (int m = 0; m < MAXN; m++)
        if (g_map[m].used && g_map[m].allocIdx == gi &&
            g_bind[b].va >= g_map[m].va &&
            g_bind[b].va < g_map[m].va + g_map[m].size)
          return 1;
    }
  }
  return 0;
}

static int do_suspend(void) {
  int groups = 0, imports = 0, unmapped = 0, unbound = 0, released = 0;
  CUcontext saved = NULL;
  r_cuCtxGetCurrent(&saved);

  if (g_untracked) {
    mclog("SUSPEND: refusing: untracked state (%s) cannot be torn down",
          g_untracked_why);
    return -1;
  }
  /* Withdraw the previous resume's fds: a held export fd blocks the
   * checkpoint. */
  unpublish_all();

  /* Multicast groups: unmap, unbind each device, release. Per-entry flags make
   * a retried suspend skip finished work. */
  for (int gi = 0; gi < MAXN; gi++) {
    if (g_alloc[gi].kind != KIND_MC) continue;
    if (g_alloc[gi].torn_down) continue; /* fully done by an earlier attempt */
    groups++;
    if (unmap_alloc(gi, "MC", &unmapped) != 0) return -1;
    for (int b = 0; b < MAXN; b++) {
      if (!g_bind[b].used || g_bind[b].groupIdx != gi) continue;
      if (g_bind[b].unbound)
        continue; /* already unbound by an earlier attempt */
      if (g_bind[b].dev < 0) {
        mclog("SUSPEND: bind %d has unknown device", b);
        return -1;
      }
      if (g_bind[b].ctx) r_cuCtxSetCurrent(g_bind[b].ctx);
      CUresult rc = r_cuMulticastUnbind(g_alloc[gi].handle, g_bind[b].dev,
                                        g_bind[b].mcOffset, g_bind[b].size);
      if (rc != CUDA_SUCCESS) {
        mclog(
            "SUSPEND: cuMulticastUnbind(mc=0x%llx, "
            "dev=%d, mcOff=0x%zx, size=0x%zx) rc=%d",
            (unsigned long long)g_alloc[gi].handle, g_bind[b].dev,
            g_bind[b].mcOffset, g_bind[b].size, rc);
        return -1;
      }
      g_bind[b].unbound = 1;
      unbound++;
    }
    CUresult rc = r_cuMemRelease(g_alloc[gi].handle);
    if (rc != CUDA_SUCCESS) {
      mclog("SUSPEND: cuMemRelease(MC 0x%llx) rc=%d",
            (unsigned long long)g_alloc[gi].handle, rc);
      return -1;
    }
    g_alloc[gi].torn_down = 1;
    released++;
  }

  /* Multicast-bound exporters: save contents, unmap (keeping reservations) and
   * release. Left resident, the next export after restore fails with
   * OBJECT_NOT_FOUND (R610, vLLM TP=4 and torch symmetric memory). Runs after
   * the group teardown, so the memory is already unbound. */
  int uc_freed = 0;
  for (int gi = 0; gi < MAXN; gi++) {
    if (g_alloc[gi].kind != KIND_UC || !g_alloc[gi].has_key) continue;
    if (g_alloc[gi].torn_down) continue; /* fully done by an earlier attempt */
    if (!uc_is_mc_bound(gi)) continue;
    if (!g_alloc[gi].uc_content) {
      void* buf = malloc(g_alloc[gi].size);
      if (!buf) {
        mclog(
            "SUSPEND: no memory for UC-export backup "
            "(0x%zx bytes)",
            g_alloc[gi].size);
        return -1;
      }
      int copied = 0;
      for (int m = 0; m < MAXN; m++) {
        if (!g_map[m].used || g_map[m].allocIdx != gi || g_map[m].suspended)
          continue;
        if (g_map[m].ctx) r_cuCtxSetCurrent(g_map[m].ctx);
        CUresult rc = r_cuMemcpyDtoH((char*)buf + g_map[m].offset, g_map[m].va,
                                     g_map[m].size);
        if (rc != CUDA_SUCCESS) {
          mclog(
              "SUSPEND: UC-export backup copy "
              "(va=0x%llx size=0x%zx) rc=%d",
              (unsigned long long)g_map[m].va, g_map[m].size, rc);
          free(buf);
          return -1;
        }
        copied++;
      }
      if (!copied) {
        free(buf);
        mclog("SUSPEND: UC-export idx=%d has no mapping to save", gi);
        return -1;
      }
      g_alloc[gi].uc_content = buf;
    }
    if (unmap_alloc(gi, "UC-export", &unmapped) != 0) return -1;
    CUresult rc = r_cuMemRelease(g_alloc[gi].handle);
    if (rc != CUDA_SUCCESS) {
      mclog("SUSPEND: cuMemRelease(UC-export 0x%llx) rc=%d",
            (unsigned long long)g_alloc[gi].handle, rc);
      return -1;
    }
    g_alloc[gi].torn_down = 1;
    released++;
    uc_freed++;
  }

  /* UC imports (P2P peer buffers): unmap and release. The memory is the
   * exporter's and cuda-checkpoint saves it; only the live import must go,
   * since cuda-checkpoint cannot restore it. */
  for (int ii = 0; ii < MAXN; ii++) {
    if (g_alloc[ii].kind != KIND_IMP) continue;
    if (g_alloc[ii].torn_down) continue; /* fully done by an earlier attempt */
    imports++;
    if (unmap_alloc(ii, "UC-import", &unmapped) != 0) return -1;
    CUresult rc = r_cuMemRelease(g_alloc[ii].handle);
    if (rc != CUDA_SUCCESS) {
      mclog("SUSPEND: cuMemRelease(import 0x%llx) rc=%d",
            (unsigned long long)g_alloc[ii].handle, rc);
      return -1;
    }
    g_alloc[ii].torn_down = 1;
    released++;
  }

  if (saved) r_cuCtxSetCurrent(saved);
  if (r_cuCtxSynchronize) r_cuCtxSynchronize();
  mclog(
      "SUSPEND done: groups=%d imports=%d uc_freed=%d unmapped=%d "
      "unbound=%d released=%d",
      groups, imports, uc_freed, unmapped, unbound, released);
  return 0;
}

/* Resume. A rank both exports and imports, so every exporter publishes (phase
 * 1) before anyone fetches (phase 2), which avoids deadlock; binds and mappings
 * follow (phase 3). */

/* Must hold g_lock. */
static int do_resume(void) {
  int groups = 0, imports = 0, remapped = 0, rebound = 0, published = 0;
  CUcontext saved = NULL;
  r_cuCtxGetCurrent(&saved);

  /* Snapshot which objects need a full rebuild. torn_down is cleared as each
   * object comes back, so phases 2 and 3 use this snapshot. */
  char full[MAXN];
  for (int gi = 0; gi < MAXN; gi++) full[gi] = (char)g_alloc[gi].torn_down;

  /* Phase 0: the first VMM call on a freshly restored context can fail with
   * CUDA_ERROR_UNKNOWN until the context is synchronized. */
  for (int i = 0; i < MAXN; i++) {
    if (g_alloc[i].kind == KIND_FREE || !g_alloc[i].ctx) continue;
    r_cuCtxSetCurrent(g_alloc[i].ctx);
    r_cuCtxSynchronize();
  }

  /* Phase 1: every exporter re-creates its object and publishes the re-exported
   * fd. */
  for (int gi = 0; gi < MAXN; gi++) {
    /* Switch context only after the kind checks, so a free slot never installs
     * a stale one. */
    if (g_alloc[gi].kind == KIND_MC && !g_alloc[gi].imported) {
      if (g_alloc[gi].ctx) r_cuCtxSetCurrent(g_alloc[gi].ctx);
      /* Multicast creator. After a partial suspend the group may still be live;
       * publish its existing handle so peers that did tear down can refetch. */
      CUmemGenericAllocationHandle newmc = g_alloc[gi].handle;
      if (full[gi]) {
        if (r_cuMulticastCreate(&newmc, &g_alloc[gi].mprop) != CUDA_SUCCESS) {
          mclog(
              "RESUME: cuMulticastCreate idx=%d "
              "failed",
              gi);
          return -1;
        }
        set_handle(&g_alloc[gi], newmc);
        /* Live again: a retried suspend must tear it down. */
        g_alloc[gi].torn_down = 0;
      }
      if (g_alloc[gi].has_key && reexport(gi, newmc) != 0) return -1;
      groups++;
    } else if (g_alloc[gi].kind == KIND_UC) {
      CUmemGenericAllocationHandle h = g_alloc[gi].handle;
      if (g_alloc[gi].uc_content && full[gi]) {
        /* Freed across the checkpoint (see do_suspend): recreate, re-map at the
         * identical VAs and restore the contents, before the re-export below
         * and the phase 3 binds. */
        if (g_alloc[gi].ctx) r_cuCtxSetCurrent(g_alloc[gi].ctx);
        CUmemGenericAllocationHandle nh = 0;
        CUresult rc =
            r_cuMemCreate(&nh, g_alloc[gi].size, &g_alloc[gi].uprop, 0);
        if (rc != CUDA_SUCCESS) {
          mclog(
              "RESUME: recreate UC-export idx=%d "
              "(size=0x%zx) rc=%d",
              gi, g_alloc[gi].size, rc);
          return -1;
        }
        set_handle(&g_alloc[gi], nh);
        g_alloc[gi].torn_down = 0;
        h = nh;
        if (remap_alloc(gi, nh, "UC-export", 0 /* full */, &remapped) != 0)
          return -1;
        for (int m = 0; m < MAXN; m++) {
          if (!g_map[m].used || g_map[m].allocIdx != gi) continue;
          if (g_map[m].ctx) r_cuCtxSetCurrent(g_map[m].ctx);
          rc = r_cuMemcpyHtoD(g_map[m].va,
                              (char*)g_alloc[gi].uc_content + g_map[m].offset,
                              g_map[m].size);
          if (rc != CUDA_SUCCESS) {
            mclog(
                "RESUME: UC-export content "
                "restore (va=0x%llx) rc=%d",
                (unsigned long long)g_map[m].va, rc);
            return -1;
          }
        }
        free(g_alloc[gi].uc_content);
        g_alloc[gi].uc_content = NULL;
      } else if (g_alloc[gi].uc_content) {
        /* Partial suspend: still live, so the device contents are
         * authoritative. Re-map what was unmapped and drop the backup. */
        if (remap_alloc(gi, h, "UC-export", 1 /* partial */, &remapped) != 0)
          return -1;
        free(g_alloc[gi].uc_content);
        g_alloc[gi].uc_content = NULL;
      }
      if (g_alloc[gi].has_key) {
        if (g_alloc[gi].ctx) r_cuCtxSetCurrent(g_alloc[gi].ctx);
        /* P2P exporter: publish the handle importers fetch. */
        if (reexport(gi, h) != 0) return -1;
        published++;
      }
    }
  }

  /* Phase 2: importers fetch and re-import (new handles). */
  for (int gi = 0; gi < MAXN; gi++) {
    if (!full[gi]) continue; /* still live (partial suspend); nothing to redo */
    if (g_alloc[gi].kind == KIND_MC && g_alloc[gi].imported) {
      if (g_alloc[gi].ctx) r_cuCtxSetCurrent(g_alloc[gi].ctx);
      CUmemGenericAllocationHandle newmc = 0;
      if (reimport(gi, &newmc) != 0) return -1;
      set_handle(&g_alloc[gi], newmc);
      g_alloc[gi].torn_down = 0; /* live again */
      groups++;
    } else if (g_alloc[gi].kind == KIND_IMP) {
      if (g_alloc[gi].ctx) r_cuCtxSetCurrent(g_alloc[gi].ctx);
      CUmemGenericAllocationHandle newh = 0;
      if (reimport(gi, &newh) != 0) return -1;
      set_handle(&g_alloc[gi], newh);
      g_alloc[gi].torn_down = 0; /* live again */
      imports++;
    }
  }

  /* Phase 3: rebuild binds and re-map every VA at its identical address.
   * Torn-down objects replay AddDevice, binds and mappings; objects a partial
   * suspend left live only redo what it undid. */
  for (int gi = 0; gi < MAXN; gi++) {
    if (g_alloc[gi].kind != KIND_MC && g_alloc[gi].kind != KIND_IMP) continue;
    int partial = !full[gi];
    if (g_alloc[gi].ctx) r_cuCtxSetCurrent(g_alloc[gi].ctx);
    if (g_alloc[gi].kind == KIND_MC) {
      CUmemGenericAllocationHandle mc = g_alloc[gi].handle;
      if (!partial)
        for (int d = 0; d < g_alloc[gi].ndev; d++)
          if (r_cuMulticastAddDevice(mc, g_alloc[gi].devs[d]) != CUDA_SUCCESS) {
            mclog(
                "RESUME: AddDevice dev=%d "
                "failed",
                g_alloc[gi].devs[d]);
            return -1;
          }
      /* cuMulticastBindMem blocks until every device joins: the binds are the
       * cross-rank barrier. */
      for (int b = 0; b < MAXN; b++) {
        if (!g_bind[b].used || g_bind[b].groupIdx != gi) continue;
        if (partial && !g_bind[b].unbound)
          continue; /* still bound; nothing to redo */
        if (g_bind[b].ctx) r_cuCtxSetCurrent(g_bind[b].ctx);
        CUresult rc =
            g_bind[b].by_addr
                ? r_cuMulticastBindAddr(mc, g_bind[b].mcOffset, g_bind[b].va,
                                        g_bind[b].size, 0)
                : r_cuMulticastBindMem(mc, g_bind[b].mcOffset,
                                       xlate_mc(g_bind[b].mem),
                                       g_bind[b].memOffset, g_bind[b].size, 0);
        if (rc != CUDA_SUCCESS) {
          mclog("RESUME: re-bind (%s) rc=%d",
                g_bind[b].by_addr ? "addr" : "mem", rc);
          return -1;
        }
        g_bind[b].unbound = 0;
        rebound++;
      }
      if (remap_alloc(gi, mc, "MC", partial, &remapped) != 0) return -1;
    } else {
      if (remap_alloc(gi, g_alloc[gi].handle, "UC-import", partial,
                      &remapped) != 0)
        return -1;
    }
  }

  /* Everything is rebuilt; reset every flag for the next suspend. */
  for (int gi = 0; gi < MAXN; gi++) g_alloc[gi].torn_down = 0;
  for (int b = 0; b < MAXN; b++) g_bind[b].unbound = 0;
  for (int m = 0; m < MAXN; m++) g_map[m].suspended = 0;

  if (saved) r_cuCtxSetCurrent(saved);
  if (r_cuCtxSynchronize) r_cuCtxSynchronize();
  mclog(
      "RESUME done: groups=%d imports=%d published=%d rebound=%d "
      "remapped=%d",
      groups, imports, published, rebound, remapped);
  return 0;
}

/* Control thread: polls $MCSHIM_DIR for markers. */

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

/* Lookup interposition: torch, NCCL and ctypes resolve driver entry points with
 * dlsym or cuGetProcAddress, bypassing symbol interposition, so tracked names
 * are redirected to the wrappers. */

/* Suspend gate. While suspended, multicast groups and imports are released and
 * their VAs unmapped: an app thread touching the GPU then faults its context
 * (700), and through a shared group every rank. cuda-checkpoint --toggle
 * restores and unlocks the app before the rebuild, so the entry points that
 * submit GPU work block until it completes. The shim's own work calls the
 * reals. */

static pthread_mutex_t g_gate_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_gate_cv = PTHREAD_COND_INITIALIZER;

/* Arm before the teardown, so that no app thread slips in before the first
 * unmap. gate_wait reads g_suspended locklessly, so transitions are atomic
 * stores. */
static void gate_arm(void) {
  pthread_mutex_lock(&g_gate_lock);
  __atomic_store_n(&g_suspended, 1, __ATOMIC_SEQ_CST);
  pthread_mutex_unlock(&g_gate_lock);
}

static void gate_disarm(void) {
  pthread_mutex_lock(&g_gate_lock);
  __atomic_store_n(&g_suspended, 0, __ATOMIC_SEQ_CST);
  pthread_cond_broadcast(&g_gate_cv);
  pthread_mutex_unlock(&g_gate_lock);
}

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

/* Entry points that submit GPU work or wait on it: block while suspended, then
 * forward. Tracked mutators call gate_wait() too, so that nothing creates or
 * frees shared state after the strict blocker check. Suspend and resume use the
 * reals, so gating cannot deadlock them. */
#define GATED(name, proto, args)                            \
  static CUresult(*r_##name) proto;                         \
  CUresult name proto;                                      \
  CUresult name proto {                                     \
    REAL(r_##name, #name);                                  \
    if (!r_##name) return 1 /* CUDA_ERROR_INVALID_VALUE */; \
    gate_wait();                                            \
    return r_##name args;                                   \
  }

typedef void* CUstream_t;
typedef void* CUfunction_t;
typedef void* CUgraphExec_t;
typedef void* CUhostFn_t;

GATED(cuLaunchKernel,
      (CUfunction_t f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
       unsigned by, unsigned bz, unsigned shmem, CUstream_t st, void** kp,
       void** extra),
      (f, gx, gy, gz, bx, by, bz, shmem, st, kp, extra))
GATED(cuLaunchKernelEx,
      (const void* cfg, CUfunction_t f, void** kp, void** extra),
      (cfg, f, kp, extra))
GATED(cuLaunchCooperativeKernel,
      (CUfunction_t f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
       unsigned by, unsigned bz, unsigned shmem, CUstream_t st, void** kp),
      (f, gx, gy, gz, bx, by, bz, shmem, st, kp))
GATED(cuGraphLaunch, (CUgraphExec_t g, CUstream_t st), (g, st))
GATED(cuMemsetD32_v2, (CUdeviceptr d, unsigned ui, size_t n), (d, ui, n))
GATED(cuMemsetD32Async, (CUdeviceptr d, unsigned ui, size_t n, CUstream_t st),
      (d, ui, n, st))
GATED(cuMemsetD8_v2, (CUdeviceptr d, unsigned char uc, size_t n), (d, uc, n))
GATED(cuMemsetD8Async,
      (CUdeviceptr d, unsigned char uc, size_t n, CUstream_t st),
      (d, uc, n, st))
GATED(cuMemcpyAsync,
      (CUdeviceptr dst, CUdeviceptr src, size_t n, CUstream_t st),
      (dst, src, n, st))
GATED(cuMemcpyHtoD_v2, (CUdeviceptr dst, const void* src, size_t n),
      (dst, src, n))
GATED(cuMemcpyDtoH_v2, (void* dst, CUdeviceptr src, size_t n), (dst, src, n))
GATED(cuMemcpyHtoDAsync_v2,
      (CUdeviceptr dst, const void* src, size_t n, CUstream_t st),
      (dst, src, n, st))
GATED(cuMemcpyDtoHAsync_v2,
      (void* dst, CUdeviceptr src, size_t n, CUstream_t st), (dst, src, n, st))
GATED(cuMemcpyDtoD_v2, (CUdeviceptr dst, CUdeviceptr src, size_t n),
      (dst, src, n))
GATED(cuMemcpyDtoDAsync_v2,
      (CUdeviceptr dst, CUdeviceptr src, size_t n, CUstream_t st),
      (dst, src, n, st))
GATED(cuMemcpy2D_v2, (const void* pCopy), (pCopy))
GATED(cuMemcpy2DAsync_v2, (const void* pCopy, CUstream_t st), (pCopy, st))
GATED(cuMemcpy3D_v2, (const void* pCopy), (pCopy))
GATED(cuMemcpy3DAsync_v2, (const void* pCopy, CUstream_t st), (pCopy, st))
GATED(cuMemsetD16_v2, (CUdeviceptr d, unsigned short us, size_t n), (d, us, n))
GATED(cuMemsetD16Async,
      (CUdeviceptr d, unsigned short us, size_t n, CUstream_t st),
      (d, us, n, st))
GATED(cuLaunchHostFunc, (CUstream_t st, CUhostFn_t fn, void* userData),
      (st, fn, userData))
GATED(cuStreamSynchronize, (CUstream_t st), (st))

CUresult cuGetProcAddress(const char*, void**, int, unsigned long long);
CUresult cuGetProcAddress_v2(const char*, void**, int, unsigned long long,
                             int*);
/* CUDA runtime resolvers (cudaError_t and the query-result enum are plain
 * ints in this toolkit-free build; 0 is success for both). */
int cudaGetDriverEntryPoint(const char*, void**, unsigned long long, int*);
int cudaGetDriverEntryPoint_ptsz(const char*, void**, unsigned long long, int*);
int cudaGetDriverEntryPointByVersion(const char*, void**, unsigned int,
                                     unsigned long long, int*);
int cudaGetDriverEntryPointByVersion_ptsz(const char*, void**, unsigned int,
                                          unsigned long long, int*);

typedef struct {
  const char* name;
  void* fn;
  /* Wrappers forward to the legacy-stream real, so PTDS lookups are not
   * redirected to them (see gpa_redirect). */
  int stream_sem;
} WrapEntry;

static const WrapEntry* wrap_table(void) {
  static WrapEntry t[] = {
      {"cuMemCreate", (void*)cuMemCreate, 0},
      {"cuMemRelease", (void*)cuMemRelease, 0},
      {"cuMemMap", (void*)cuMemMap, 0},
      {"cuMemUnmap", (void*)cuMemUnmap, 0},
      {"cuMemSetAccess", (void*)cuMemSetAccess, 0},
      {"cuMulticastCreate", (void*)cuMulticastCreate, 0},
      {"cuMulticastAddDevice", (void*)cuMulticastAddDevice, 0},
      {"cuMulticastBindMem", (void*)cuMulticastBindMem, 0},
      {"cuMulticastBindAddr", (void*)cuMulticastBindAddr, 0},
      {"cuMulticastUnbind", (void*)cuMulticastUnbind, 0},
      {"cuInit", (void*)cuInit, 0},
      {"cuMemExportToShareableHandle", (void*)cuMemExportToShareableHandle, 0},
      {"cuMemImportFromShareableHandle", (void*)cuMemImportFromShareableHandle,
       0},
      {"cuDeviceGetAttribute", (void*)cuDeviceGetAttribute, 0},

      {"cuLaunchKernel", (void*)cuLaunchKernel, 1},
      {"cuLaunchKernelEx", (void*)cuLaunchKernelEx, 1},
      {"cuLaunchCooperativeKernel", (void*)cuLaunchCooperativeKernel, 1},
      {"cuGraphLaunch", (void*)cuGraphLaunch, 1},
      {"cuLaunchHostFunc", (void*)cuLaunchHostFunc, 1},
      {"cuMemsetD32", (void*)cuMemsetD32_v2, 1},
      {"cuMemsetD32_v2", (void*)cuMemsetD32_v2, 1},
      {"cuMemsetD32Async", (void*)cuMemsetD32Async, 1},
      {"cuMemsetD16", (void*)cuMemsetD16_v2, 1},
      {"cuMemsetD16_v2", (void*)cuMemsetD16_v2, 1},
      {"cuMemsetD16Async", (void*)cuMemsetD16Async, 1},
      {"cuMemsetD8", (void*)cuMemsetD8_v2, 1},
      {"cuMemsetD8_v2", (void*)cuMemsetD8_v2, 1},
      {"cuMemsetD8Async", (void*)cuMemsetD8Async, 1},
      {"cuMemcpyAsync", (void*)cuMemcpyAsync, 1},
      {"cuMemcpyHtoD", (void*)cuMemcpyHtoD_v2, 1},
      {"cuMemcpyHtoD_v2", (void*)cuMemcpyHtoD_v2, 1},
      {"cuMemcpyDtoH", (void*)cuMemcpyDtoH_v2, 1},
      {"cuMemcpyDtoH_v2", (void*)cuMemcpyDtoH_v2, 1},
      {"cuMemcpyDtoD", (void*)cuMemcpyDtoD_v2, 1},
      {"cuMemcpyDtoD_v2", (void*)cuMemcpyDtoD_v2, 1},
      {"cuMemcpyHtoDAsync", (void*)cuMemcpyHtoDAsync_v2, 1},
      {"cuMemcpyHtoDAsync_v2", (void*)cuMemcpyHtoDAsync_v2, 1},
      {"cuMemcpyDtoHAsync", (void*)cuMemcpyDtoHAsync_v2, 1},
      {"cuMemcpyDtoHAsync_v2", (void*)cuMemcpyDtoHAsync_v2, 1},
      {"cuMemcpyDtoDAsync", (void*)cuMemcpyDtoDAsync_v2, 1},
      {"cuMemcpyDtoDAsync_v2", (void*)cuMemcpyDtoDAsync_v2, 1},
      {"cuMemcpy2D", (void*)cuMemcpy2D_v2, 1},
      {"cuMemcpy2D_v2", (void*)cuMemcpy2D_v2, 1},
      {"cuMemcpy2DAsync", (void*)cuMemcpy2DAsync_v2, 1},
      {"cuMemcpy2DAsync_v2", (void*)cuMemcpy2DAsync_v2, 1},
      {"cuMemcpy3D", (void*)cuMemcpy3D_v2, 1},
      {"cuMemcpy3D_v2", (void*)cuMemcpy3D_v2, 1},
      {"cuMemcpy3DAsync", (void*)cuMemcpy3DAsync_v2, 1},
      {"cuMemcpy3DAsync_v2", (void*)cuMemcpy3DAsync_v2, 1},
      {"cuStreamSynchronize", (void*)cuStreamSynchronize, 1},
      {"cuGetProcAddress", (void*)cuGetProcAddress, 0},
      {"cuGetProcAddress_v2", (void*)cuGetProcAddress_v2, 0},
      /* Covers apps that dlsym these from a dlopen'd libcudart. */
      {"cudaGetDriverEntryPoint", (void*)cudaGetDriverEntryPoint, 0},
      {"cudaGetDriverEntryPoint_ptsz", (void*)cudaGetDriverEntryPoint_ptsz, 0},
      {"cudaGetDriverEntryPointByVersion",
       (void*)cudaGetDriverEntryPointByVersion, 0},
      {"cudaGetDriverEntryPointByVersion_ptsz",
       (void*)cudaGetDriverEntryPointByVersion_ptsz, 0},
      {NULL, NULL, 0},
  };
  return t;
}

static const WrapEntry* wrap_entry(const char* name) {
  if (!name) return NULL;
  for (const WrapEntry* e = wrap_table(); e->name; e++)
    if (strcmp(e->name, name) == 0) return e;
  return NULL;
}

static void* wrapper_for(const char* name) {
  const WrapEntry* e = wrap_entry(name);
  return e ? e->fn : NULL;
}

/* Interposed dlsym: hand out wrappers for tracked driver symbols. Delegating
 * through a dlvsym-resolved dlsym re-anchors RTLD_NEXT at mcshim, which can
 * confuse interposers stacked after it. */
void* dlsym(void* handle, const char* symbol) {
  init_real_dlsym();
  if (!real_dlsym) return NULL;
  void* w = wrapper_for(symbol);
  if (w) {
    /* Redirect only if the library has the symbol, so feature probes still
     * work. */
    void* r = real_dlsym(handle, symbol);
    if (r) return w;
    return r;
  }
  return real_dlsym(handle, symbol);
}

/* Interposed cuGetProcAddress. A lookup of "cuGetProcAddress" at cudaVersion >=
 * 12000 expects the 5-argument v2 ABI, so it must get the v2 wrapper: the v1
 * wrapper leaves symbolStatus unwritten, and NCCL then treats every symbol as
 * missing. */

#define CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM (1ULL << 1)

static void* gpa_redirect(const char* symbol, int cudaVersion,
                          unsigned long long flags) {
  if (strcmp(symbol, "cuGetProcAddress") == 0)
    return cudaVersion >= 12000 ? (void*)cuGetProcAddress_v2
                                : (void*)cuGetProcAddress;
  const WrapEntry* e = wrap_entry(symbol);
  if (!e) return NULL;
  /* A PTDS lookup of a stream-sensitive entry gets the driver's own pfn, since
   * the wrapper would change NULL-stream semantics. Such apps are not gated on
   * these entries. */
  if (e->stream_sem && (flags & CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM))
    return NULL;
  return e->fn;
}

CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion,
                          unsigned long long flags) {
  static CUresult (*real)(const char*, void**, int, unsigned long long);
  REAL(real, "cuGetProcAddress");
  if (!real) return 3; /* CUDA_ERROR_NOT_INITIALIZED */
  CUresult rc = real(symbol, pfn, cudaVersion, flags);
  void* w;
  if (rc == CUDA_SUCCESS && pfn && *pfn &&
      (w = gpa_redirect(symbol, cudaVersion, flags))) {
    *pfn = w;
  }
  return rc;
}

CUresult cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion,
                             unsigned long long flags, int* symbolStatus) {
  static CUresult (*real)(const char*, void**, int, unsigned long long, int*);
  REAL(real, "cuGetProcAddress_v2");
  if (!real) return 3;
  CUresult rc = real(symbol, pfn, cudaVersion, flags, symbolStatus);
  void* w;
  if (rc == CUDA_SUCCESS && pfn && *pfn &&
      (w = gpa_redirect(symbol, cudaVersion, flags))) {
    *pfn = w;
  }
  return rc;
}

/* Interposed cudart resolvers. torch >= 2.11 resolves its driver API through
 * cudaGetDriverEntryPointByVersion, and cudart reaches libcuda by a path
 * neither hook sees, so interpose the resolver, let cudart look the symbol up,
 * and post-process like cuGetProcAddress. cudart's cudaEnable* values equal the
 * driver's CU_GET_PROC_ADDRESS_* bits, except that cudaEnableDefault means PTDS
 * in the _ptsz variants. The reals live in libcudart: resolve via RTLD_NEXT,
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

/* cudaEnableDefault in a _ptsz resolver means PTDS. */
static unsigned long long rt_eff_flags(unsigned long long flags, int ptsz) {
  if (ptsz && flags == 0) return CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM;
  return flags;
}

/* The unversioned resolver follows the runtime's own ABI generation, and cudart
 * >= 12.0 means the v2-era driver ABI. */
static void rt_gpa_post(const char* symbol, void** pfn, int cudaVersion,
                        unsigned long long flags) {
  void* w;
  if (pfn && *pfn && (w = gpa_redirect(symbol, cudaVersion, flags))) *pfn = w;
}

int cudaGetDriverEntryPoint(const char* symbol, void** pfn,
                            unsigned long long flags, int* driverStatus) {
  static int (*real)(const char*, void**, unsigned long long, int*);
  RTREAL(real, "cudaGetDriverEntryPoint");
  if (!real) return CUDA_ERROR_RT_SYMBOL_NOT_FOUND;
  int rc = real(symbol, pfn, flags, driverStatus);
  if (rc == 0) rt_gpa_post(symbol, pfn, 12000, rt_eff_flags(flags, 0));
  return rc;
}

int cudaGetDriverEntryPoint_ptsz(const char* symbol, void** pfn,
                                 unsigned long long flags, int* driverStatus) {
  static int (*real)(const char*, void**, unsigned long long, int*);
  RTREAL(real, "cudaGetDriverEntryPoint_ptsz");
  if (!real) return CUDA_ERROR_RT_SYMBOL_NOT_FOUND;
  int rc = real(symbol, pfn, flags, driverStatus);
  if (rc == 0) rt_gpa_post(symbol, pfn, 12000, rt_eff_flags(flags, 1));
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
  if (rc == 0)
    rt_gpa_post(symbol, pfn, (int)cudaVersion, rt_eff_flags(flags, 0));
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
  if (rc == 0)
    rt_gpa_post(symbol, pfn, (int)cudaVersion, rt_eff_flags(flags, 1));
  return rc;
}

/* Edge-triggered on marker existence: "suspend" appearing suspends and acks
 * suspended.<pid>, disappearing resumes and acks resumed.<pid>, and failures
 * ack error.<pid>. The marker is in the checkpoint image, so after a restore
 * the shim stays suspended until the sentry removes it. */
static void* control_thread(void* arg) {
  (void)arg;
  char ack_s[64], ack_r[64], ack_e[64];
  snprintf(ack_s, sizeof(ack_s), "suspended.%d", (int)getpid());
  snprintf(ack_r, sizeof(ack_r), "resumed.%d", (int)getpid());
  snprintf(ack_e, sizeof(ack_e), "error.%d", (int)getpid());
  char ack_g[64];
  snprintf(ack_g, sizeof(ack_g), "gated.%d", (int)getpid());
  /* present.<pid> tells the sentry that this process will ack. The sentry
   * selects CUDA processes by their open NVIDIA fds, a broader set. */
  char present[64];
  snprintf(present, sizeof(present), "present.%d", (int)getpid());
  /* Clear acks left by a dead process with the same pid. */
  marker_rm(ack_s);
  marker_rm(ack_r);
  marker_rm(ack_e);
  marker_rm(ack_g);
  marker_write(present, "ok");
  mclog("control thread started (dir=%s)", g_dir);
  int prev = 0;      /* treat startup as not-suspended */
  int prev_gate = 0; /* and not-gated */
  for (;;) {
    /* "gate" issues no CUDA calls, so the sentry can arm it while
     * cuda-checkpoint holds this process locked. */
    int wgate = marker_exists("gate");
    if (wgate != prev_gate) {
      prev_gate = wgate;
      if (wgate) {
        gate_arm();
        marker_write(ack_g, "ok");
      } else {
        /* Refuse to ungate while teardown state is outstanding (a failed
         * resume): the app would fault every coupled rank. Only a successful
         * resume clears it. */
        int torn = 0;
        pthread_mutex_lock(&g_lock);
        for (int i = 0; i < MAXN; i++) {
          if (g_alloc[i].kind != KIND_FREE && g_alloc[i].torn_down) torn++;
          if (g_map[i].used && g_map[i].suspended) torn++;
        }
        pthread_mutex_unlock(&g_lock);
        if (torn) {
          mclog(
              "FATAL: refusing to release the "
              "gate: %d object(s) still torn down "
              "after a failed resume; the "
              "application would run over unmapped "
              "GPU state",
              torn);
        } else {
          gate_disarm();
          marker_rm(ack_g);
          /* Every process has resumed: withdraw the published fds. */
          pthread_mutex_lock(&g_lock);
          unpublish_all();
          pthread_mutex_unlock(&g_lock);
        }
      }
    }
    int want = marker_exists("suspend");
    if (want != prev) {
      prev = want;
      /* Drop a stale error ack: the sentry fails fast on error.<pid>. */
      marker_rm(ack_e);
      pthread_mutex_lock(&g_lock);
      resolve_reals();
      int rc;
      if (want) {
        /* Arm first: no app thread may reach the GPU before the last unmap. */
        gate_arm();
        rc = do_suspend();
        if (rc != 0)
          /* A failed suspend leaves the app running, not blocked. The sentry's
           * unwind removes the marker, and the resume edge rebuilds any partial
           * teardown. */
          gate_disarm();
      } else {
        /* A failed resume keeps the gate armed: better blocked than corrupt.
         * Teardown and rebuild are re-enterable, so a retried edge converges.
         */
        rc = do_resume();
        if (rc == 0) gate_disarm();
      }
      pthread_mutex_unlock(&g_lock);
      if (rc != 0) {
        marker_write(ack_e, want ? "suspend failed" : "resume failed");
      } else if (want) {
        marker_rm(ack_r);
        marker_write(ack_s, "ok");
      } else {
        marker_rm(ack_s);
        marker_write(ack_r, "ok");
      }
    }
    /* Poll every 5 ms: the spread in when ranks see the gate bounds how often a
     * collective straddles it, which fails the sentry's lock. */
    struct timespec ts = {0, 5 * 1000 * 1000}; /* 5ms */
    nanosleep(&ts, NULL);
  }
  return NULL;
}

static int g_disabled;

/* Start the control thread only in processes that resolve a tracked entry
 * point, so that helpers inheriting LD_PRELOAD (shells, runsc exec,
 * cuda-checkpoint) never consume or ack markers. */
static void control_thread_start(void) {
  if (g_disabled) return;
  pthread_t t;
  int err = pthread_create(&t, NULL, control_thread, NULL);
  if (err == 0)
    pthread_detach(t);
  else
    /* Otherwise this surfaces as an unexplained sentry ack timeout. */
    mclog(
        "FATAL: control thread creation failed: %s -- this "
        "process will never acknowledge suspend/resume markers",
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
  __atomic_store_n(&g_suspended, 0, __ATOMIC_SEQ_CST);
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
  const char* d = getenv("MCSHIM_DIR");
  if (d && *d) snprintf(g_dir, sizeof(g_dir), "%s", d);
  /* Create the control dir, which may also hold MCSHIM_LOG, before the first
   * mclog. */
  mkdir(g_dir, 0777);
  /* Markers belong to the sentry; the control thread starts from cuInit. */
  mclog("loaded; control dir=%s", g_dir);
}
