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
 * pkg/sentry/control/state_cuda_shim.go). After a rebuild, exporters serve the
 * re-exported fd on a unix socket keyed by the original export's identity, and
 * importers fetch and re-import it. Unicast device memory stays
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
#include <poll.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

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
static CUresult (*r_cuMemAddressReserve)(CUdeviceptr*, size_t, size_t,
                                         CUdeviceptr, unsigned long long);
static CUresult (*r_cuMemAddressFree)(CUdeviceptr, size_t);
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
  REAL(r_cuMemAddressReserve, "cuMemAddressReserve");
  REAL(r_cuMemAddressFree, "cuMemAddressFree");
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

/* MCSHIM_VERBOSE=1 enables per-entry suspend/resume diagnostics. */
static int mcverbose(void) {
  static int v = -1;
  if (v < 0) v = getenv("MCSHIM_VERBOSE") != NULL;
  return v;
}

#define mcvlog(...)                      \
  do {                                   \
    if (mcverbose()) mclog(__VA_ARGS__); \
  } while (0)

/* Tracked state: the live object graph. Frees remove entries, so freed objects
 * drop out of the replay set. */

/* Static tables (about 3 MB per process) keep hot paths allocation-free. An
 * SGLang TP=8 rank tracks more than 512 objects. */
#define MAXN 4096
#define MAX_AKA 16
#define MAX_DEV 16

/* KIND_IMP is an import; cuMulticastAddDevice on it proves it a multicast group
 * and makes it KIND_MC. */
enum { KIND_FREE = 0, KIND_UC = 1, KIND_MC = 2, KIND_IMP = 3 };

typedef struct {
  int kind;
  CUmemGenericAllocationHandle handle;       /* current handle */
  CUmemGenericAllocationHandle aka[MAX_AKA]; /* all handles ever held */
  int naka;
  size_t size;
  CUcontext ctx;
  CUmemAllocationProp uprop;   /* KIND_UC */
  CUmulticastObjectProp mprop; /* KIND_MC */
  int devs[MAX_DEV];           /* KIND_MC: added devices */
  int ndev;
  /* Rendezvous identity of the export (see record_key). */
  int imported; /* 1 = handle came from an import */
  int has_key;
  unsigned long key_dev, key_ino;
  int key_ord;
  /* After a resume: the re-exported fd, served to importers until the next
   * suspend. */
  int serve_fd;
  int serve_sock;
  int serve_wake; /* write end of the serve thread's stop pipe */
  pthread_t serve_thread;
  char serve_path[104]; /* must fit sockaddr_un.sun_path (108) */
  int serving;
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
  for (int i = 0; i < MAXN; i++) {
    if (g_alloc[i].kind == KIND_FREE) continue;
    for (int a = 0; a < g_alloc[i].naka; a++)
      if (g_alloc[i].aka[a] == h) return g_alloc[i].handle;
  }
  return h;
}

/* Must hold g_lock. Drop h from every alias list. The driver reuses handle
 * values, so a stale alias could misroute a call on the new object to an
 * unrelated one (vLLM's sleep/wake churn makes this routine). Also called for
 * untracked handles. */
static void aka_purge(CUmemGenericAllocationHandle h) {
  for (int i = 0; i < MAXN; i++) {
    Alloc* o = &g_alloc[i];
    if (o->kind == KIND_FREE) continue;
    for (int k = 0; k < o->naka; k++) {
      if (o->aka[k] != h) continue;
      for (int j = k; j + 1 < o->naka; j++) o->aka[j] = o->aka[j + 1];
      o->naka--;
      k--;
    }
  }
}

/* Must hold g_lock. Record h as a's current handle, keeping previous values as
 * aliases (see xlate_mc) after purging h from every list. */
static int g_alias_overflow; /* sticky, see alloc_push_aka */

static void alloc_push_aka(Alloc* a, CUmemGenericAllocationHandle h) {
  aka_purge(h);
  a->handle = h;
  if (a->naka < MAX_AKA) {
    a->aka[a->naka++] = h;
    return;
  }
  /* Each rebuild rotates the handle once. Evict the oldest rotated value but
   * keep aka[0], the original the app holds; further suspends are refused. */
  if (!g_alias_overflow)
    mclog(
        "WARNING: alias list full (MAX_AKA=%d) for handle 0x%llx; "
        "evicting oldest rotated alias -- suspend is disabled for "
        "this process",
        MAX_AKA, (unsigned long long)h);
  g_alias_overflow = 1;
  for (int j = 1; j + 1 < MAX_AKA; j++) a->aka[j] = a->aka[j + 1];
  a->aka[MAX_AKA - 1] = h;
}

/* Must hold g_lock. Reset slot i and record the current context + handle. */
static Alloc* alloc_init(int i, int kind, CUmemGenericAllocationHandle h) {
  Alloc* a = &g_alloc[i];
  memset(a, 0, sizeof(*a));
  a->kind = kind;
  a->serve_fd = a->serve_sock = a->serve_wake = -1;
  r_cuCtxGetCurrent(&a->ctx);
  alloc_push_aka(a, h);
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

/* How many other live allocs carry this key, so that groups with colliding keys
 * still rendezvous when ranks create them in the same order. */
static int key_ordinal(unsigned long dev, unsigned long ino, int self) {
  int n = 0;
  for (int i = 0; i < MAXN; i++)
    if (i != self && g_alloc[i].kind != KIND_FREE && g_alloc[i].has_key &&
        g_alloc[i].key_dev == dev && g_alloc[i].key_ino == ino)
      n++;
  return n;
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

/* Must hold g_lock. Record the rendezvous identity: nvproxy's fdinfo line if
 * present, else st_dev:st_ino plus a creation ordinal (all NVIDIA export fds
 * share one inode, so that only works for a few groups). The first key wins. */
static void record_key(int i, int fd) {
  unsigned long kd, ki;
  struct stat st;
  if (fd < 0) return;
  if (fdinfo_oracle(fd, &kd, &ki) != 0) {
    if (fstat(fd, &st) != 0) return;
    kd = (unsigned long)st.st_dev;
    ki = (unsigned long)st.st_ino;
  }
  Alloc* a = &g_alloc[i];
  if (a->has_key) {
    mclog("idx=%d re-exported; keeping first key %lx:%lx", i, a->key_dev,
          a->key_ino);
    return;
  }
  a->has_key = 1;
  a->key_dev = kd;
  a->key_ino = ki;
  a->key_ord = key_ordinal(kd, ki, i);
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
      /* Untracked, but still purge stale aliases of its value. */
      aka_purge(*h);
    }
    pthread_mutex_unlock(&g_lock);
  }
  return rc;
}

CUresult cuMulticastCreate(CUmemGenericAllocationHandle* h,
                           const CUmulticastObjectProp* prop) {
  resolve_reals();
  gate_wait();
  /* Same strip as cuMemCreate: a FABRIC-typed group gets an NV_MEMORY_FABRIC
   * (00f8) companion object that outlives the multicast teardown. */
  CUmulticastObjectProp mfixed;
  if (prop && (prop->handleTypes & CU_MEM_HANDLE_TYPE_FABRIC) &&
      !getenv("MCSHIM_ALLOW_FABRIC")) {
    mfixed = *prop;
    mfixed.handleTypes &= ~(unsigned long long)CU_MEM_HANDLE_TYPE_FABRIC;
    if (!mfixed.handleTypes) mfixed.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FD;
    prop = &mfixed;
    static int logged;
    if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
      mclog(
          "stripping CU_MEM_HANDLE_TYPE_FABRIC from "
          "cuMulticastCreate (-> handleTypes 0x%llx): "
          "fabric-handle multicast is not checkpointable; "
          "set MCSHIM_ALLOW_FABRIC=1 to keep it",
          mfixed.handleTypes);
  }
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
      mclog("track MC group idx=%d handle=0x%llx size=0x%zx", i,
            (unsigned long long)*h, a->size);
    } else {
      aka_purge(*h);
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
     * served on resume. */
    if (i >= 0 && (g_alloc[i].kind == KIND_MC || g_alloc[i].kind == KIND_UC)) {
      record_key(i, *(int*)shHandle);
      mclog("track EXPORT idx=%d kind=%d key=%lx:%lx ord=%d", i,
            g_alloc[i].kind, g_alloc[i].key_dev, g_alloc[i].key_ino,
            g_alloc[i].key_ord);
    }
    pthread_mutex_unlock(&g_lock);
  }
  return rc;
}

CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* h,
                                        void* osHandle, int type) {
  resolve_reals();
  gate_wait();
  /* Mirror of the export refusal: an imported fabric ref is a 00fb object. */
  if (type == CU_MEM_HANDLE_TYPE_FABRIC && !getenv("MCSHIM_ALLOW_FABRIC")) {
    static int logged;
    if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
      mclog(
          "refusing fabric-typed cuMemImportFromShareableHandle"
          " (see export-side message)");
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  CUresult rc = r_cuMemImportFromShareableHandle(h, osHandle, type);
  if (rc == CUDA_SUCCESS && type == CU_MEM_HANDLE_TYPE_POSIX_FD && h) {
    pthread_mutex_lock(&g_lock);
    int i = alloc_new();
    if (i >= 0) {
      Alloc* a = alloc_init(i, KIND_IMP, *h);
      a->imported = 1;
      /* For POSIX-FD imports osHandle is the fd. */
      record_key(i, (int)(intptr_t)osHandle);
      mclog(
          "track IMPORT idx=%d handle=0x%llx key=%lx:%lx "
          "ord=%d",
          i, (unsigned long long)*h, a->key_dev, a->key_ino, a->key_ord);
    } else {
      aka_purge(*h);
    }
    pthread_mutex_unlock(&g_lock);
  }
  return rc;
}

/* torch >= 2.11 treats a failing nvmlDeviceGetGpuFabricInfoV as fatal. Report a
 * failure as NVML_SUCCESS with state NOT_SUPPORTED (0), as a fabric-less host
 * does. The struct size is in the version's low 24 bits. MCSHIM_ALLOW_FABRIC=1
 * opts out. */
static int nvml_fabric_info_smooth(void* info, int rc, const char* via) {
  if (rc == 0 || !info || getenv("MCSHIM_ALLOW_FABRIC")) return rc;
  unsigned int version = *(unsigned int*)info;
  unsigned int size = version & 0xffffffu;
  if (size < sizeof(unsigned int) || size > 4096)
    return rc; /* implausible; do not touch */
  memset((char*)info + sizeof(unsigned int), 0, size - sizeof(unsigned int));
  static int logged;
  if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
    mclog(
        "%s failed (rc=%d): reporting fabric NOT_SUPPORTED "
        "instead; set MCSHIM_ALLOW_FABRIC=1 to pass errors "
        "through",
        via, rc);
  return 0; /* NVML_SUCCESS */
}

static void* libnvml_sym(const char* name) {
  init_real_dlsym();
  if (!real_dlsym) return NULL;
  void* s = real_dlsym(RTLD_NEXT, name);
  if (s) return s;
  static void* h;
  if (!h) h = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_NOLOAD);
  return h ? real_dlsym(h, name) : NULL;
}

int nvmlDeviceGetGpuFabricInfoV(void* device, void* gpuFabricInfo);
int nvmlDeviceGetGpuFabricInfoV(void* device, void* gpuFabricInfo) {
  static int (*real)(void*, void*);
  if (!real) *(void**)(&real) = libnvml_sym("nvmlDeviceGetGpuFabricInfoV");
  if (!real) return 13; /* NVML_ERROR_FUNCTION_NOT_FOUND */
  int rc = real(device, gpuFabricInfo);
  return nvml_fabric_info_smooth(gpuFabricInfo, rc,
                                 "nvmlDeviceGetGpuFabricInfoV");
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
    if (i >= 0 && g_alloc[i].kind == KIND_IMP) {
      g_alloc[i].kind = KIND_MC;
      mclog("import idx=%d classified as MC group", i);
    }
    if (i >= 0 && g_alloc[i].kind == KIND_MC && g_alloc[i].ndev < MAX_DEV) {
      /* Avoid duplicate entries across resume re-adds. */
      int dup = 0;
      for (int d = 0; d < g_alloc[i].ndev; d++)
        if (g_alloc[i].devs[d] == dev) dup = 1;
      if (!dup) g_alloc[i].devs[g_alloc[i].ndev++] = dev;
    }
    pthread_mutex_unlock(&g_lock);
  }
  return rc;
}

/* Must hold g_lock. Record a successful bind unless it is already tracked. */
static void bind_record(int gi, int by_addr, CUmemGenericAllocationHandle mem,
                        CUdeviceptr va, size_t mcOffset, size_t memOffset,
                        size_t size, CUdevice dev) {
  if (gi < 0) return;
  for (int b = 0; b < MAXN; b++)
    if (g_bind[b].used && g_bind[b].groupIdx == gi &&
        g_bind[b].by_addr == by_addr && g_bind[b].mem == mem &&
        g_bind[b].va == va && g_bind[b].mcOffset == mcOffset &&
        g_bind[b].memOffset == memOffset && g_bind[b].size == size)
      return;
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
    mclog(
        "track BIND%s group=%d %s=0x%llx dev=%d mcOff=0x%zx "
        "size=0x%zx",
        by_addr ? "-ADDR" : "", gi, by_addr ? "va" : "mem",
        (unsigned long long)(by_addr ? va : mem), dev, mcOffset, size);
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

static void stop_serving(Alloc* a);

/* Must hold g_lock. Forget alloc i and the binds and maps that reference it. */
static void alloc_forget(int i) {
  for (int b = 0; b < MAXN; b++)
    if (g_bind[b].used &&
        (g_bind[b].groupIdx == i || g_bind[b].mem == g_alloc[i].handle))
      g_bind[b].used = 0;
  for (int m = 0; m < MAXN; m++)
    if (g_map[m].used && g_map[m].allocIdx == i) g_map[m].used = 0;
  stop_serving(&g_alloc[i]);
  g_alloc[i].ctx = NULL; /* a freed slot must never look targetable */
  g_alloc[i].naka = 0;
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

/* Cross-rank fd rendezvous: an exporter serves its re-exported fd on a unix
 * socket keyed by the original export identity, and importers receive it via
 * SCM_RIGHTS. */

/* Default only; the sentry sets MCSHIM_DIR (see DefaultCudaMulticastShimDir).
 */
static char g_dir[512] = "/tmp/mcshim";

/* Returns -1 if the path does not fit sun_path (keep MCSHIM_DIR short). */
static int group_sock_path(const Alloc* a, char* out, size_t n) {
  int w = snprintf(out, n, "%s/mcgrp-%lx-%lx-%d.sock", g_dir, a->key_dev,
                   a->key_ino, a->key_ord);
  if (w < 0 || (size_t)w >= n) {
    mclog("socket path too long (MCSHIM_DIR=%s)", g_dir);
    return -1;
  }
  return 0;
}

static int send_fd(int sock, int fd) {
  char b = 'F';
  struct iovec iov = {&b, 1};
  union {
    struct cmsghdr h;
    char buf[CMSG_SPACE(sizeof(int))];
  } u;
  struct msghdr msg;
  memset(&msg, 0, sizeof(msg));
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = u.buf;
  msg.msg_controllen = sizeof(u.buf);
  struct cmsghdr* c = CMSG_FIRSTHDR(&msg);
  c->cmsg_level = SOL_SOCKET;
  c->cmsg_type = SCM_RIGHTS;
  c->cmsg_len = CMSG_LEN(sizeof(int));
  memcpy(CMSG_DATA(c), &fd, sizeof(int));
  return sendmsg(sock, &msg, 0) == 1 ? 0 : -1;
}

static int recv_fd(int sock) {
  char b;
  struct iovec iov = {&b, 1};
  union {
    struct cmsghdr h;
    char buf[CMSG_SPACE(sizeof(int))];
  } u;
  struct msghdr msg;
  memset(&msg, 0, sizeof(msg));
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = u.buf;
  msg.msg_controllen = sizeof(u.buf);
  /* MSG_CMSG_CLOEXEC: an export fd leaked into a child blocks the next
   * checkpoint. */
  if (recvmsg(sock, &msg, MSG_CMSG_CLOEXEC) <= 0) return -1;
  struct cmsghdr* c = CMSG_FIRSTHDR(&msg);
  if (!c || c->cmsg_type != SCM_RIGHTS) return -1;
  int fd;
  memcpy(&fd, CMSG_DATA(c), sizeof(int));
  return fd;
}

/* Exporter accept loop. It owns its args, its own dup of the served fd and the
 * listening socket, so that stop_serving can never make a late accept send a
 * closed or reused fd number. Exits when stop_serving wakes it. */
typedef struct {
  int sock;
  int fd;
  int wake; /* read end of the stop pipe (see stop_serving) */
  char path[104];
} ServeArgs;

static void* serve_thread(void* arg) {
  ServeArgs* sa = arg;
  mclog("serving group fd on %s", sa->path);
  for (;;) {
    /* poll() with a stop pipe: shutdown() does not wake accept() under gVisor,
     * and a thread that never exits keeps its dup of the export fd open, which
     * blocks the checkpoint. */
    struct pollfd pfd[2] = {{sa->sock, POLLIN, 0}, {sa->wake, POLLIN, 0}};
    int pr = poll(pfd, 2, -1);
    if (pr < 0) {
      if (errno == EINTR) continue;
      break;
    }
    if (pfd[1].revents) break; /* stop_serving */
    int c = accept4(sa->sock, NULL, NULL, SOCK_CLOEXEC | SOCK_NONBLOCK);
    if (c < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      break;
    }
    if (send_fd(c, sa->fd) != 0)
      mclog("serve: send_fd failed: %s", strerror(errno));
    close(c);
  }
  mclog("serve thread for %s exiting", sa->path);
  close(sa->sock); /* thread-owned (see stop_serving) */
  close(sa->fd);   /* the thread's own dup (see start_serving) */
  close(sa->wake);
  free(sa);
  return NULL;
}

/* Must hold g_lock (mutates Alloc). */
static int start_serving(Alloc* a, int fd) {
  if (group_sock_path(a, a->serve_path, sizeof(a->serve_path)) != 0) return -1;
  struct sockaddr_un sa;
  unlink(a->serve_path);
  int s = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (s < 0) return -1;
  memset(&sa, 0, sizeof(sa));
  sa.sun_family = AF_UNIX;
  strcpy(sa.sun_path, a->serve_path);
  if (bind(s, (struct sockaddr*)&sa, sizeof(sa)) != 0 || listen(s, 64) != 0) {
    mclog("RESUME: bind/listen(%s) failed: %s", a->serve_path, strerror(errno));
    close(s);
    return -1;
  }
  ServeArgs* args = malloc(sizeof(*args));
  if (!args) {
    close(s);
    return -1;
  }
  /* The thread serves its own dup (see serve_thread). */
  int tfd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
  int wake[2];
  if (tfd < 0 || pipe2(wake, O_CLOEXEC) != 0) {
    close(s);
    if (tfd >= 0) close(tfd);
    free(args);
    return -1;
  }
  args->sock = s;
  args->fd = tfd;
  args->wake = wake[0];
  strcpy(args->path, a->serve_path);
  if (pthread_create(&a->serve_thread, NULL, serve_thread, args) != 0) {
    close(s);
    close(tfd);
    close(wake[0]);
    close(wake[1]);
    free(args);
    return -1;
  }
  a->serve_fd = fd;
  a->serve_sock = s;
  a->serve_wake = wake[1];
  a->serving = 1;
  return 0;
}

static void stop_serving(Alloc* a) {
  if (!a->serving) return;
  /* Wake and join the serve thread: its dup of the export fd blocks the
   * checkpoint until closed. The thread closes the listening fd itself. */
  if (a->serve_wake >= 0) {
    ssize_t n = write(a->serve_wake, "x", 1);
    (void)n;
    close(a->serve_wake);
    a->serve_wake = -1;
  }
  pthread_join(a->serve_thread, NULL);
  a->serve_sock = -1;
  if (a->serve_path[0]) unlink(a->serve_path);
  if (a->serve_fd >= 0) {
    close(a->serve_fd); /* exported fds are checkpoint blockers */
    a->serve_fd = -1;
  }
  a->serving = 0;
  mclog("stopped serving group fd");
}

/* The sentry waits 5 minutes for resumed.<pid>. One total resume budget inside
 * that caps every rendezvous wait, so the two sides cannot disagree about the
 * outcome. */

#define RESUME_DEADLINE_MS (240 * 1000L)

static long mono_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000L + ts.tv_nsec / 1000000L;
}

/* mono_ms() deadline for the resume in flight; set at do_resume entry. */
static long g_resume_deadline;

/* Remaining resume budget in ms; <= 0 once the deadline has passed. */
static long resume_budget_ms(void) { return g_resume_deadline - mono_ms(); }

/* Connect to the exporter's socket, retrying until it serves or the resume
 * deadline passes, and receive the fd. */
static int fetch_group_fd(const Alloc* a, int timeout_ms) {
  long budget = resume_budget_ms();
  if (budget <= 0) {
    mclog(
        "RESUME: resume deadline (%lds) exceeded before fetching "
        "group fd",
        RESUME_DEADLINE_MS / 1000);
    return -1;
  }
  if ((long)timeout_ms > budget) timeout_ms = (int)budget;
  char path[104];
  if (group_sock_path(a, path, sizeof(path)) != 0) return -1;
  struct sockaddr_un sa;
  memset(&sa, 0, sizeof(sa));
  sa.sun_family = AF_UNIX;
  strcpy(sa.sun_path, path);
  for (int waited = 0; waited < timeout_ms; waited += 100) {
    int s = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (s < 0) return -1;
    if (connect(s, (struct sockaddr*)&sa, sizeof(sa)) == 0) {
      int fd = recv_fd(s);
      close(s);
      if (fd >= 0) return fd;
    } else {
      close(s);
    }
    struct timespec ts = {0, 100 * 1000 * 1000};
    nanosleep(&ts, NULL);
  }
  mclog("timed out fetching group fd from %s", path);
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
 * backed by h, in the retained reservation (re-reserving at the fixed address
 * if needed). With partial set, gi is still live after a failed suspend, so
 * only what was unmapped is re-mapped. */
static int remap_alloc(int gi, CUmemGenericAllocationHandle h, const char* what,
                       int partial, int* remapped) {
  for (int m = 0; m < MAXN; m++) {
    if (!g_map[m].used || g_map[m].allocIdx != gi) continue;
    if (partial && !g_map[m].suspended)
      continue; /* still mapped; nothing to redo */
    if (g_map[m].ctx) r_cuCtxSetCurrent(g_map[m].ctx);
    const char* path = "retained-reservation";
    CUresult rc = r_cuMemMap(g_map[m].va, g_map[m].size, g_map[m].offset, h, 0);
    if (rc != CUDA_SUCCESS) {
      CUdeviceptr got = 0;
      CUresult rr =
          r_cuMemAddressReserve(&got, g_map[m].size, 0, g_map[m].va, 0);
      if (rr != CUDA_SUCCESS || got != g_map[m].va) {
        if (rr == CUDA_SUCCESS) r_cuMemAddressFree(got, g_map[m].size);
        mclog(
            "RESUME: %s re-map at identical VA 0x%llx "
            "failed (got 0x%llx rr=%d)",
            what, (unsigned long long)g_map[m].va, (unsigned long long)got, rr);
        return -1;
      }
      rc = r_cuMemMap(g_map[m].va, g_map[m].size, g_map[m].offset, h, 0);
      if (rc != CUDA_SUCCESS) {
        mclog(
            "RESUME: %s cuMemMap after re-reserve "
            "rc=%d",
            what, rc);
        return -1;
      }
      path = "re-reserved-fixed";
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
    mclog("RESUME: %s VA 0x%llx re-mapped IDENTICAL (%s)", what,
          (unsigned long long)g_map[m].va, path);
  }
  return 0;
}

/* Must hold g_lock. Synchronize every tracked context and log the result, to
 * bracket when a context faulted. Diagnostic only. */
static void ctx_probe(const char* tag) {
  CUcontext saved = NULL;
  r_cuCtxGetCurrent(&saved);
  for (int i = 0; i < MAXN; i++) {
    if (g_alloc[i].kind == KIND_FREE || !g_alloc[i].ctx) continue;
    int seen = 0;
    for (int j = 0; j < i; j++)
      if (g_alloc[j].kind != KIND_FREE && g_alloc[j].ctx == g_alloc[i].ctx) {
        seen = 1;
        break;
      }
    if (seen) continue;
    r_cuCtxSetCurrent(g_alloc[i].ctx);
    CUresult sy = r_cuCtxSynchronize ? r_cuCtxSynchronize() : 0;
    mclog("CTXPROBE[%s] ctx=%p sync=%d%s", tag, g_alloc[i].ctx, sy,
          sy ? "  <-- FAULTED" : "");
  }
  if (saved) r_cuCtxSetCurrent(saved);
}

/* Must hold g_lock. Re-export alloc gi's handle h and start serving it on the
 * rendezvous socket, so importers can fetch it. */
static int reexport_serve(int gi, CUmemGenericAllocationHandle h) {
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
  /* Keep the export fd out of forked children, where it blocks the next
   * checkpoint. */
  fcntl(fd, F_SETFD, FD_CLOEXEC);
  if (start_serving(&g_alloc[gi], fd) != 0) {
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
    if (resume_budget_ms() <= 0) {
      mclog(
          "RESUME: resume deadline (%lds) exceeded during "
          "re-import idx=%d (attempt %d)",
          RESUME_DEADLINE_MS / 1000, gi, attempt);
      return -1;
    }
    int fd = fetch_group_fd(&g_alloc[gi], 60 * 1000);
    if (fd < 0) return -1;
    CUresult rc = r_cuMemImportFromShareableHandle(out, (void*)(intptr_t)fd,
                                                   CU_MEM_HANDLE_TYPE_POSIX_FD);
    close(fd);
    if (rc == CUDA_SUCCESS) {
      if (attempt > 0)
        mclog(
            "RESUME: re-import idx=%d key=%lx:%lx ok "
            "after %d retries",
            gi, g_alloc[gi].key_dev, g_alloc[gi].key_ino, attempt);
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
          gi, g_alloc[gi].key_dev, g_alloc[gi].key_ino, cur, rc);
    }
    /* INVALID_DEVICE is never transient: this process cannot address the
     * exporter's device. */
    if (rc == CUDA_ERROR_INVALID_DEVICE) {
      mclog(
          "RESUME: re-import idx=%d gave up: INVALID_DEVICE "
          "(the importer cannot address the exporting device; "
          "after a restore onto different GPUs this means the "
          "sentry did not translate RM-reported device identity "
          "-- see nvproxy's GET_EXPORT_OBJECT_INFO handling)",
          gi);
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
      for (int k = 0; k < g_alloc[gi].naka; k++)
        if (g_bind[b].mem == g_alloc[gi].aka[k]) return 1;
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
  if (g_alias_overflow) {
    mclog(
        "SUSPEND: refusing: a handle alias list overflowed earlier "
        "(MAX_AKA=%d), so a stale handle the application holds "
        "may no longer translate",
        MAX_AKA);
    return -1;
  }

  ctx_probe("suspend-entry");

  /* Stop serving the previous resume's fds: a held export fd blocks the
   * checkpoint. */
  for (int i = 0; i < MAXN; i++)
    if (g_alloc[i].serving) stop_serving(&g_alloc[i]);

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
    mclog("SUSPEND: released MC group idx=%d handle=0x%llx", gi,
          (unsigned long long)g_alloc[gi].handle);
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
        /* No live mapping to copy from: leave it resident rather than lose its
         * contents. */
        free(buf);
        mclog(
            "SUSPEND: UC-export idx=%d has no live "
            "mapping; leaving resident",
            gi);
        continue;
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
    mclog(
        "SUSPEND: freed UC exporter idx=%d handle=0x%llx "
        "size=0x%zx (content saved)",
        gi, (unsigned long long)g_alloc[gi].handle, g_alloc[gi].size);
  }

  /* UC imports (P2P peer buffers): unmap and release. The memory is the
   * exporter's and cuda-checkpoint saves it; only the live import must go,
   * since cuda-checkpoint cannot restore it. */
  for (int ii = 0; ii < MAXN; ii++) {
    if (g_alloc[ii].kind != KIND_IMP) continue;
    if (g_alloc[ii].torn_down) continue; /* fully done by an earlier attempt */
    imports++;
    /* Layout diagnostics: a nonzero offset, or more than one mapping per
     * import, would need exact replay. */
    int nmap = 0;
    for (int m = 0; m < MAXN; m++) {
      if (!g_map[m].used || g_map[m].allocIdx != ii) continue;
      nmap++;
      if (g_map[m].offset != 0 || nmap > 1)
        mclog(
            "IMPORT-LAYOUT idx=%d map#%d va=0x%llx "
            "size=0x%zx offset=0x%zx",
            ii, nmap, (unsigned long long)g_map[m].va, g_map[m].size,
            g_map[m].offset);
    }
    if (nmap != 1) mclog("IMPORT-LAYOUT idx=%d has %d mappings", ii, nmap);
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

  ctx_probe("suspend-exit");

  if (saved) r_cuCtxSetCurrent(saved);
  if (r_cuCtxSynchronize) r_cuCtxSynchronize();
  mclog(
      "SUSPEND done: groups=%d imports=%d uc_freed=%d unmapped=%d "
      "unbound=%d released=%d",
      groups, imports, uc_freed, unmapped, unbound, released);
  return 0;
}

/* Resume. A rank both exports and imports, so every exporter serves (phase 1)
 * before anyone fetches (phase 2), which avoids deadlock; binds and mappings
 * follow (phase 3). */

/* Must hold g_lock. */
static int do_resume(void) {
  int groups = 0, imports = 0, remapped = 0, rebound = 0, served = 0;
  CUcontext saved = NULL;
  r_cuCtxGetCurrent(&saved);

  /* One budget for the whole rebuild, inside the sentry's 5-minute ack timeout.
   */
  g_resume_deadline = mono_ms() + RESUME_DEADLINE_MS;

  /* Snapshot which objects need a full rebuild. torn_down is cleared as each
   * object comes back, so phases 2 and 3 use this snapshot. */
  char full[MAXN];
  for (int gi = 0; gi < MAXN; gi++) full[gi] = (char)g_alloc[gi].torn_down;

  /* Phase 0: the first VMM call on a freshly restored context can fail with
   * CUDA_ERROR_UNKNOWN, and a synchronize clears it. Its result also shows
   * whether cuda-checkpoint's restore left the context faulted. */
  ctx_probe("resume-entry");
  ctx_probe("resume-warm");

  /* Phase 1: every exporter re-creates its object and serves the re-exported
   * fd. */
  int p1_mc = 0, p1_uc = 0;
  for (int gi = 0; gi < MAXN; gi++) {
    /* Switch context only after the kind checks, so a free slot never installs
     * a stale one. */
    if (g_alloc[gi].kind == KIND_MC && !g_alloc[gi].imported) {
      if (g_alloc[gi].ctx) r_cuCtxSetCurrent(g_alloc[gi].ctx);
      /* Multicast creator. After a partial suspend the group may still be live;
       * serve its existing handle so peers that did tear down can refetch. */
      CUmemGenericAllocationHandle newmc = g_alloc[gi].handle;
      if (full[gi]) {
        if (r_cuMulticastCreate(&newmc, &g_alloc[gi].mprop) != CUDA_SUCCESS) {
          mclog(
              "RESUME: cuMulticastCreate idx=%d "
              "failed",
              gi);
          return -1;
        }
        alloc_push_aka(&g_alloc[gi], newmc);
        /* Live again: a retried suspend must tear it down. */
        g_alloc[gi].torn_down = 0;
      }
      if (g_alloc[gi].has_key && reexport_serve(gi, newmc) != 0) return -1;
      groups++;
      p1_mc++;
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
        alloc_push_aka(&g_alloc[gi], nh);
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
        /* P2P exporter: serve the handle importers fetch. */
        if (reexport_serve(gi, h) != 0) return -1;
        served++;
        p1_uc++;
      }
    }
  }

  mclog("RESUME: phase1 done (%d MC creators, %d UC exporters served)", p1_mc,
        p1_uc);

  /* Phase 2: importers fetch and re-import (new handles). */
  for (int gi = 0; gi < MAXN; gi++) {
    if (!full[gi]) continue; /* still live (partial suspend); nothing to redo */
    if (g_alloc[gi].kind == KIND_MC && g_alloc[gi].imported) {
      if (g_alloc[gi].ctx) r_cuCtxSetCurrent(g_alloc[gi].ctx);
      CUmemGenericAllocationHandle newmc = 0;
      if (reimport(gi, &newmc) != 0) return -1;
      alloc_push_aka(&g_alloc[gi], newmc);
      g_alloc[gi].torn_down = 0; /* live again */
      groups++;
    } else if (g_alloc[gi].kind == KIND_IMP) {
      if (g_alloc[gi].ctx) r_cuCtxSetCurrent(g_alloc[gi].ctx);
      CUmemGenericAllocationHandle newh = 0;
      if (reimport(gi, &newh) != 0) return -1;
      alloc_push_aka(&g_alloc[gi], newh);
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
      "RESUME done: groups=%d imports=%d served=%d rebound=%d "
      "remapped=%d",
      groups, imports, served, rebound, remapped);
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
      /* NVML fabric-info smoothing (see nvml_fabric_info_smooth). */
      {"nvmlDeviceGetGpuFabricInfoV", (void*)nvmlDeviceGetGpuFabricInfoV, 0},
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
   * these entries (logged). */
  if (e->stream_sem &&
      (flags & CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)) {
    static int logged;
    if (!logged) {
      logged = 1;
      mcvlog(
          "cuGetProcAddress(%s, PTDS): declining redirect "
          "for per-thread-default-stream lookups",
          symbol);
    }
    return NULL;
  }
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
  char ack_g[64], ack_ug[64];
  snprintf(ack_g, sizeof(ack_g), "gated.%d", (int)getpid());
  snprintf(ack_ug, sizeof(ack_ug), "ungated.%d", (int)getpid());
  /* present.<pid> tells the sentry that this process will ack. The sentry
   * selects CUDA processes by their open NVIDIA fds, a broader set. */
  char present[64];
  snprintf(present, sizeof(present), "present.%d", (int)getpid());
  /* Clear acks left by a dead process with the same pid. */
  marker_rm(ack_s);
  marker_rm(ack_r);
  marker_rm(ack_e);
  marker_rm(ack_g);
  marker_rm(ack_ug);
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
        marker_rm(ack_ug);
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
          marker_write(ack_ug, "ok");
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
     * collective straddles it. At 100 ms, busy workloads exhausted the sentry's
     * lock retries. */
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

/* Fork: a child has no usable CUDA state, and a lock held by another parent
 * thread would deadlock it. Reset locks and tracking; the child starts its own
 * control thread if it initializes CUDA. */
static void mcshim_atfork_child(void) {
  g_loglock = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
  g_lock = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
  g_gate_lock = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
  g_gate_cv = (pthread_cond_t)PTHREAD_COND_INITIALIZER;
  /* Close inherited rendezvous fds first: a held export fd keeps the RM object
   * alive and blocks the parent's next checkpoint. The socket paths stay, since
   * the parent still serves them. Serve threads' own dups are unreachable and
   * linger until exec. */
  for (int i = 0; i < MAXN; i++) {
    if (g_alloc[i].kind != KIND_FREE) {
      if (g_alloc[i].serve_sock >= 0) close(g_alloc[i].serve_sock);
      if (g_alloc[i].serve_fd >= 0) close(g_alloc[i].serve_fd);
      if (g_alloc[i].serve_wake >= 0) close(g_alloc[i].serve_wake);
    }
    free(g_alloc[i].uc_content);
  }
  memset(g_alloc, 0, sizeof(g_alloc));
  memset(g_map, 0, sizeof(g_map));
  memset(g_bind, 0, sizeof(g_bind));
  g_untracked = 0;
  g_untracked_why = NULL;
  g_alias_overflow = 0;
  __atomic_store_n(&g_suspended, 0, __ATOMIC_SEQ_CST);
  g_control_started = 0;
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
