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

/* A fake libcuda.so.1 for the shim's table tests (tables.c), with no GPU. It
 * implements the VMM and multicast entry points the shim tracks and the
 * context calls its transitions make, and validates every handle and address
 * it gets, so a wrong translation in the shim fails the call. Like the real
 * driver, it reissues the handle values of freed objects, newest first.
 *
 * References: 1 per create, +1 per retain, +1 per mapping, and memory gets +1
 * per bind; an object is freed when none remain. Freeing a group drops its
 * binds. fake_live() reports what is still allocated. */

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long H;

enum { OK = 0, BAD = 1, NOT_FOUND = 500 };
enum { MAXOBJ = 1 << 20, NDEV = 2, VA_BUCKETS = 1 << 18 };

typedef struct {
  int live, refs, maps, mc;
  size_t size;
  int ndev;
} Obj;

typedef struct Map {
  CUdeviceptr va;
  size_t size;
  int obj;
  struct Map* next;
} Map;

typedef struct {
  int used, mc, mem;
  int dev;
  size_t mcOffset, size;
} Bind;

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static Obj* objs;
static int nobj;
static int* freeh;
static int nfreeh;
static Map* va_tab[VA_BUCKETS];
static Bind* binds;
static int nbinds, capbinds;
static int nlive, nmaps;
static int primary[NDEV];
static __thread int cur = -1;

static void init(void) {
  if (!objs) {
    objs = calloc(MAXOBJ, sizeof(Obj));
    freeh = calloc(MAXOBJ, sizeof(int));
  }
}

/* Handle values: index + 1, so 0 is never valid. */
static int obj_of(H h) {
  if (h == 0 || h > (H)nobj || !objs[h - 1].live) return -1;
  return (int)(h - 1);
}

static H new_obj(size_t size, int mc) {
  int i = nfreeh ? freeh[--nfreeh] : nobj++;
  objs[i] = (Obj){.live = 1, .refs = 1, .mc = mc, .size = size};
  nlive++;
  return (H)i + 1;
}

static void unref(int i);

static void drop_binds_of_group(int g) {
  for (int b = 0; b < nbinds; b++)
    if (binds[b].used && binds[b].mc == g) {
      binds[b].used = 0;
      unref(binds[b].mem);
    }
}

static void unref(int i) {
  if (--objs[i].refs > 0) return;
  if (objs[i].mc) drop_binds_of_group(i);
  objs[i].live = 0;
  freeh[nfreeh++] = i;
  nlive--;
}

static unsigned va_hash(CUdeviceptr va) {
  return (unsigned)((va >> 12) * 0x9e3779b97f4a7c15ULL >> 40) &
         (VA_BUCKETS - 1);
}

static Map* map_find(CUdeviceptr va) {
  for (Map* m = va_tab[va_hash(va)]; m; m = m->next)
    if (m->va == va) return m;
  return NULL;
}

int fake_live(void) { return nlive; }
int fake_maps(void) { return nmaps; }

#define LOCKED(body)          \
  do {                        \
    pthread_mutex_lock(&mu);  \
    init();                   \
    CUresult rc_ = (body);    \
    pthread_mutex_unlock(&mu); \
    return rc_;               \
  } while (0)

CUresult cuInit(unsigned flags) {
  (void)flags;
  LOCKED(OK);
}

CUresult cuDeviceGetCount(int* n) {
  *n = NDEV;
  return OK;
}

CUresult cuDeviceGetAttribute(int* pi, int attr, CUdevice d) {
  (void)attr;
  (void)d;
  *pi = 0;
  return OK;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext* c, CUdevice d) {
  if (d < 0 || d >= NDEV) return BAD;
  pthread_mutex_lock(&mu);
  primary[d]++;
  pthread_mutex_unlock(&mu);
  *c = (CUcontext)(long)(d + 1);
  return OK;
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice d) {
  if (d < 0 || d >= NDEV) return BAD;
  pthread_mutex_lock(&mu);
  CUresult rc = primary[d] > 0 ? (primary[d]--, OK) : BAD;
  pthread_mutex_unlock(&mu);
  return rc;
}

CUresult cuDevicePrimaryCtxGetState(CUdevice d, unsigned* flags, int* active) {
  if (d < 0 || d >= NDEV) return BAD;
  *flags = 0;
  *active = primary[d] > 0;
  return OK;
}

CUresult cuCtxSetCurrent(CUcontext c) {
  long d = (long)c - 1;
  if (c && (d < 0 || d >= NDEV)) return BAD;
  cur = c ? (int)d : -1;
  return OK;
}

CUresult cuCtxGetCurrent(CUcontext* c) {
  *c = cur < 0 ? NULL : (CUcontext)(long)(cur + 1);
  return OK;
}

CUresult cuCtxGetDevice(CUdevice* d) {
  if (cur < 0) return BAD;
  *d = cur;
  return OK;
}

CUresult cuCtxSynchronize(void) { return cur < 0 ? BAD : OK; }

CUresult cuPointerGetAttribute(void* data, int attr, CUdeviceptr va) {
  (void)attr;
  (void)va;
  *(int*)data = 0;
  return OK;
}

CUresult cuMemCreate(H* h, size_t size, const void* prop,
                     unsigned long long flags) {
  (void)prop;
  (void)flags;
  LOCKED((*h = new_obj(size, 0), OK));
}

CUresult cuMulticastCreate(H* h, const void* prop) {
  (void)prop;
  LOCKED((*h = new_obj(0, 1), OK));
}

static CUresult release(H h) {
  int i = obj_of(h);
  if (i < 0) return BAD;
  unref(i);
  return OK;
}

CUresult cuMemRelease(H h) { LOCKED(release(h)); }

static CUresult map(CUdeviceptr va, size_t size, H h) {
  int i = obj_of(h);
  if (i < 0 || !size || map_find(va)) return BAD;
  Map* m = malloc(sizeof(*m));
  *m = (Map){va, size, i, va_tab[va_hash(va)]};
  va_tab[va_hash(va)] = m;
  objs[i].refs++;
  objs[i].maps++;
  nmaps++;
  return OK;
}

CUresult cuMemMap(CUdeviceptr va, size_t size, size_t off, H h,
                  unsigned long long flags) {
  (void)off;
  (void)flags;
  LOCKED(map(va, size, h));
}

/* The range must be whole mappings, back to back. */
static CUresult unmap(CUdeviceptr va, size_t size) {
  CUdeviceptr p = va;
  while (p < va + size) {
    Map* m = map_find(p);
    if (!m || p + m->size > va + size) return BAD;
    p += m->size;
  }
  for (p = va; p < va + size;) {
    Map** pp = &va_tab[va_hash(p)];
    while ((*pp)->va != p) pp = &(*pp)->next;
    Map* m = *pp;
    *pp = m->next;
    p += m->size;
    objs[m->obj].maps--;
    nmaps--;
    unref(m->obj);
    free(m);
  }
  return OK;
}

CUresult cuMemUnmap(CUdeviceptr va, size_t size) { LOCKED(unmap(va, size)); }

static CUresult set_access(CUdeviceptr va, size_t size) {
  for (CUdeviceptr p = va; p < va + size;) {
    Map* m = map_find(p);
    if (!m) return BAD;
    p += m->size;
  }
  return OK;
}

CUresult cuMemSetAccess(CUdeviceptr va, size_t size, const void* desc,
                        size_t count) {
  (void)desc;
  (void)count;
  LOCKED(set_access(va, size));
}

static CUresult retain(H* h, void* addr) {
  Map* m = map_find((CUdeviceptr)(unsigned long)addr);
  if (!m) return BAD;
  objs[m->obj].refs++;
  *h = (H)m->obj + 1;
  return OK;
}

CUresult cuMemRetainAllocationHandle(H* h, void* addr) {
  LOCKED(retain(h, addr));
}

CUresult cuMemGetAllocationPropertiesFromHandle(void* prop, H h) {
  (void)prop;
  LOCKED(obj_of(h) < 0 ? BAD : OK);
}

static CUresult add_device(H h, CUdevice d) {
  int i = obj_of(h);
  if (i < 0 || !objs[i].mc || d < 0 || d >= NDEV) return BAD;
  objs[i].ndev++;
  return OK;
}

CUresult cuMulticastAddDevice(H h, CUdevice d) { LOCKED(add_device(h, d)); }

static CUresult bind(H mc, CUdevice dev, size_t mcOffset, H mem, size_t size) {
  int g = obj_of(mc), m = obj_of(mem);
  if (g < 0 || !objs[g].mc || m < 0 || objs[m].mc || !objs[g].ndev)
    return BAD;
  if (nbinds == capbinds) {
    capbinds = capbinds ? 2 * capbinds : 1024;
    binds = realloc(binds, (size_t)capbinds * sizeof(Bind));
  }
  binds[nbinds++] = (Bind){1, g, m, dev, mcOffset, size};
  objs[m].refs++;
  return OK;
}

CUresult cuMulticastBindMem(H mc, size_t mcOffset, H mem, size_t memOffset,
                            size_t size, unsigned long long flags) {
  (void)memOffset;
  (void)flags;
  LOCKED(bind(mc, cur, mcOffset, mem, size));
}

CUresult cuMulticastBindMem_v2(H mc, CUdevice dev, size_t mcOffset, H mem,
                               size_t memOffset, size_t size,
                               unsigned long long flags) {
  (void)memOffset;
  (void)flags;
  LOCKED(bind(mc, dev, mcOffset, mem, size));
}

static CUresult unbind(H mc, CUdevice dev, size_t mcOffset, size_t size) {
  int g = obj_of(mc);
  if (g < 0 || !objs[g].mc) return BAD;
  for (int b = 0; b < nbinds; b++)
    if (binds[b].used && binds[b].mc == g && binds[b].dev == dev &&
        binds[b].mcOffset >= mcOffset &&
        binds[b].mcOffset + binds[b].size <= mcOffset + size) {
      binds[b].used = 0;
      unref(binds[b].mem);
    }
  return OK;
}

CUresult cuMulticastUnbind(H mc, CUdevice dev, size_t mcOffset, size_t size) {
  LOCKED(unbind(mc, dev, mcOffset, size));
}

CUresult cuMemcpyDtoH_v2(void* dst, CUdeviceptr src, size_t n) {
  (void)dst;
  (void)n;
  LOCKED(map_find(src) ? OK : BAD);
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void* src, size_t n) {
  (void)src;
  (void)n;
  LOCKED(map_find(dst) ? OK : BAD);
}
