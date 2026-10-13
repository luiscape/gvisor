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

/* The shim's tracking tables, without a GPU: run against fakecuda.c, with the
 * shim (built with -DMCSHIM_TABLE_CHECKS, which checks every index as g_lock
 * is released; see check_tables_amortized) LD_PRELOADed. See run.sh's
 * "tables" test.
 *
 *   fuzz SEED OPS THREADS  random VMM and multicast traffic from THREADS
 *                          threads, each against its own model, with
 *                          suspend/resume cycles from the main thread while
 *                          they run; everything is released at the end
 *   scale N                N allocations mapped back to back, through a
 *                          cycle, released in random order and unmapped
 *                          many per call; prints the cost per call
 *
 * The fake driver rejects any handle or address it did not issue, so a wrong
 * translation fails the call. A retained handle must be the value the
 * application was given, a cycle must leave the driver with as many objects
 * and mappings as before, and nothing may remain at the end. */

#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long H;

typedef struct {
  int type, handleTypes;
  struct {
    int type, id;
  } loc;
  void* win32;
  unsigned char flags[8];
} AllocProp;
typedef struct {
  struct {
    int type, id;
  } loc;
  int flags;
} Access;
typedef struct {
  unsigned numDevices;
  size_t size;
  unsigned long long handleTypes, flags;
} McProp;

CUresult cuInit(unsigned);
CUresult cuDevicePrimaryCtxRetain(CUcontext*, CUdevice);
CUresult cuCtxSetCurrent(CUcontext);
CUresult cuMemCreate(H*, size_t, const AllocProp*, unsigned long long);
CUresult cuMemRelease(H);
CUresult cuMemMap(CUdeviceptr, size_t, size_t, H, unsigned long long);
CUresult cuMemUnmap(CUdeviceptr, size_t);
CUresult cuMemSetAccess(CUdeviceptr, size_t, const Access*, size_t);
CUresult cuMemRetainAllocationHandle(H*, void*);
CUresult cuMulticastCreate(H*, const McProp*);
CUresult cuMulticastAddDevice(H, CUdevice);
CUresult cuMulticastBindMem(H, size_t, H, size_t, size_t, unsigned long long);
CUresult cuMulticastBindMem_v2(H, CUdevice, size_t, H, size_t, size_t,
                               unsigned long long);
CUresult cuMulticastUnbind(H, CUdevice, size_t, size_t);
int fake_live(void);
int fake_maps(void);

#define SZ ((size_t)2 << 20)

static int g_failed;
#define CK(x)                                                              \
  do {                                                                     \
    CUresult rc_ = (x);                                                    \
    if (rc_ != 0) {                                                        \
      fprintf(stderr, "FAIL %s:%d: %s = %d\n", __FILE__, __LINE__, #x, rc_); \
      exit(1);                                                             \
    }                                                                      \
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

static CUcontext g_ctx[2];

static void use(int d) { CK(cuCtxSetCurrent(g_ctx[d])); }

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void msleep(int ms) {
  struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
  nanosleep(&ts, NULL);
}

static unsigned rnd(unsigned* s) {
  *s ^= *s << 13;
  *s ^= *s >> 17;
  *s ^= *s << 5;
  return *s;
}

static AllocProp prop(int d) {
  AllocProp p;
  memset(&p, 0, sizeof(p));
  p.type = 1;
  p.loc.type = 1;
  p.loc.id = d;
  return p;
}

/* The sentry's side of the marker protocol (see mcshim_test.c). */

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

/* 1: acked, 0: the shim refused (error.<pid>), -1: timeout. */
static int wait_ack(const char* kind) {
  char a[64], e[64];
  snprintf(a, sizeof(a), "%s.%d", kind, (int)getpid());
  snprintf(e, sizeof(e), "error.%d", (int)getpid());
  for (double end = now() + 60; now() < end; msleep(2)) {
    if (exists(e)) return 0;
    if (exists(a)) return 1;
  }
  return -1;
}

static void clear_acks(void) {
  static const char* const kinds[] = {"gated", "suspended", "resumed", "error"};
  for (int k = 0; k < 4; k++) {
    char a[64];
    snprintf(a, sizeof(a), "%s.%d", kinds[k], (int)getpid());
    rm(a);
  }
}

/* A checkpoint without the images. 1 if it ran, 0 if the gate refused (the
 * state may legitimately be uncarriable); a failure after the gate accepted
 * fails the test. */
static int cycle(void) {
  clear_acks();
  mk("gate");
  int r = wait_ack("gated");
  if (r != 1) {
    rm("gate");
    EXPECT(r == 0, "gate: no ack");
    msleep(10);
    return 0;
  }
  int live = fake_live(), maps = fake_maps();
  mk("suspend");
  r = wait_ack("suspended");
  EXPECT(r == 1, "suspend failed (%d)", r);
  rm("suspend");
  r = wait_ack("resumed");
  EXPECT(r == 1, "resume failed (%d)", r);
  EXPECT(fake_live() == live && fake_maps() == maps,
         "the driver holds %d objects and %d mappings, had %d and %d",
         fake_live(), fake_maps(), live, maps);
  rm("gate");
  msleep(10);
  if (g_failed) exit(1);
  return 1;
}

/* fuzz: each thread's model of what the application holds. */

enum { MAXH = 512, MAXM = 512, MAXG = 16 };

typedef struct {
  H h;
  int refs; /* the application's */
  int dev;
} Mem;

typedef struct {
  CUdeviceptr va;
  int n; /* back-to-back mappings of n allocations, one SZ each */
  int mem[8];
} Run;

typedef struct {
  H mc;
  int refs;
  CUdeviceptr va; /* 0 if unmapped */
  int bound[2];   /* mem slot bound on device d, or -1 */
} Group;

typedef struct {
  int id;
  unsigned seed;
  int ops;
  Mem mem[MAXH];
  int nmem;
  Run run[MAXM];
  int nrun;
  Group grp[MAXG];
  int ngrp;
  CUdeviceptr next_va;
} Model;

static CUdeviceptr new_va(Model* m, int n) {
  CUdeviceptr va = m->next_va;
  m->next_va += (CUdeviceptr)(n + 1) * SZ; /* a gap between runs */
  return va;
}

static int mem_held(Model* m, unsigned* s) {
  if (!m->nmem) return -1;
  for (int k = 0; k < 4; k++) {
    int i = (int)(rnd(s) % (unsigned)m->nmem);
    if (m->mem[i].refs > 0) return i;
  }
  return -1;
}

static int mapped(Model* m, int mi);
static int bound(Model* m, int mi);

/* Creates into a slot whose object is gone, or a new one. */
static void op_create(Model* m, unsigned* s) {
  int i = -1;
  for (int t = 0; t < 4 && m->nmem && i < 0; t++) {
    int c = (int)(rnd(s) % (unsigned)m->nmem);
    if (!m->mem[c].refs && !mapped(m, c) && !bound(m, c)) i = c;
  }
  if (i < 0 && m->nmem == MAXH) return;
  if (i < 0) i = m->nmem++;
  int d = (int)(rnd(s) % 2);
  AllocProp p = prop(d);
  use(d);
  Mem* x = &m->mem[i];
  CK(cuMemCreate(&x->h, SZ, &p, 0));
  x->refs = 1;
  x->dev = d;
}

/* Maps 1-8 held allocations back to back and grants access over the run in
 * one call. */
static void op_map(Model* m, unsigned* s) {
  if (m->nrun == MAXM) return;
  int n = 1 + (int)(rnd(s) % 8), mems[8], k = 0;
  for (int t = 0; t < 2 * n && k < n; t++) {
    int i = mem_held(m, s);
    if (i >= 0) mems[k++] = i;
  }
  if (!k) return;
  Run* r = &m->run[m->nrun++];
  r->va = new_va(m, k);
  r->n = k;
  use(m->mem[mems[0]].dev);
  for (int j = 0; j < k; j++) {
    r->mem[j] = mems[j];
    CK(cuMemMap(r->va + (CUdeviceptr)j * SZ, SZ, 0, m->mem[mems[j]].h, 0));
  }
  Access a = {{1, 0}, 3};
  CK(cuMemSetAccess(r->va, (size_t)k * SZ, &a, 1));
  a.loc.id = 1;
  CK(cuMemSetAccess(r->va, (size_t)k * SZ, &a, 1));
}

/* Retains through a mapping: the shim must return the application's value. */
static void op_retain(Model* m, unsigned* s) {
  if (!m->nrun) return;
  Run* r = &m->run[rnd(s) % (unsigned)m->nrun];
  int j = (int)(rnd(s) % (unsigned)r->n);
  H h;
  CK(cuMemRetainAllocationHandle(&h, (void*)(uintptr_t)(r->va + j * SZ)));
  Mem* x = &m->mem[r->mem[j]];
  EXPECT(h == x->h, "thread %d: retained 0x%llx, want 0x%llx", m->id, h, x->h);
  x->refs++;
}

/* Unmaps a whole run in one call, or its first mapping. */
static void op_unmap(Model* m, unsigned* s) {
  if (!m->nrun) return;
  int i = (int)(rnd(s) % (unsigned)m->nrun);
  Run* r = &m->run[i];
  if (r->n > 1 && rnd(s) % 2) {
    CK(cuMemUnmap(r->va, SZ));
    r->va += SZ;
    memmove(r->mem, r->mem + 1, (size_t)(r->n - 1) * sizeof(int));
    r->n--;
    return;
  }
  CK(cuMemUnmap(r->va, (size_t)r->n * SZ));
  m->run[i] = m->run[--m->nrun];
}

static int mapped(Model* m, int mi) {
  for (int i = 0; i < m->nrun; i++)
    for (int j = 0; j < m->run[i].n; j++)
      if (m->run[i].mem[j] == mi) return 1;
  return 0;
}

static int bound(Model* m, int mi) {
  for (int g = 0; g < m->ngrp; g++)
    if (m->grp[g].bound[0] == mi || m->grp[g].bound[1] == mi) return 1;
  return 0;
}

static void op_release(Model* m, unsigned* s) {
  int i = mem_held(m, s);
  if (i < 0) return;
  CK(cuMemRelease(m->mem[i].h));
  m->mem[i].refs--;
}

/* A group with both devices, memory bound on each (v1 on device 0, v2 on
 * device 1), mapped. */
static void op_group(Model* m, unsigned* s) {
  if (m->ngrp == MAXG) return;
  int u[2] = {-1, -1};
  for (int t = 0; t < 16 && (u[0] < 0 || u[1] < 0); t++) {
    int i = mem_held(m, s);
    if (i >= 0 && u[m->mem[i].dev] < 0 && !bound(m, i)) u[m->mem[i].dev] = i;
  }
  if (u[0] < 0 || u[1] < 0) return;
  Group* g = &m->grp[m->ngrp++];
  McProp p = {2, SZ, 0, 0};
  use(0);
  CK(cuMulticastCreate(&g->mc, &p));
  g->refs = 1;
  CK(cuMulticastAddDevice(g->mc, 0));
  CK(cuMulticastAddDevice(g->mc, 1));
  CK(cuMulticastBindMem(g->mc, 0, m->mem[u[0]].h, 0, SZ, 0));
  CK(cuMulticastBindMem_v2(g->mc, 1, 0, m->mem[u[1]].h, 0, SZ, 0));
  g->bound[0] = u[0];
  g->bound[1] = u[1];
  g->va = new_va(m, 1);
  CK(cuMemMap(g->va, SZ, 0, g->mc, 0));
  Access a = {{1, 0}, 3};
  CK(cuMemSetAccess(g->va, SZ, &a, 1));
  if (rnd(s) % 2) {
    H h;
    CK(cuMemRetainAllocationHandle(&h, (void*)(uintptr_t)g->va));
    EXPECT(h == g->mc, "thread %d: retained group 0x%llx, want 0x%llx",
           m->id, h, g->mc);
    g->refs++;
  }
}

static void group_destroy(Model* m, int gi) {
  Group* g = &m->grp[gi];
  use(0);
  if (g->va) CK(cuMemUnmap(g->va, SZ));
  for (int d = 0; d < 2; d++) CK(cuMulticastUnbind(g->mc, d, 0, SZ));
  while (g->refs-- > 0) CK(cuMemRelease(g->mc));
  m->grp[gi] = m->grp[--m->ngrp];
}

static void op_ungroup(Model* m, unsigned* s) {
  if (m->ngrp) group_destroy(m, (int)(rnd(s) % (unsigned)m->ngrp));
}

static void release_all(Model* m) {
  while (m->ngrp) group_destroy(m, 0);
  while (m->nrun) {
    CK(cuMemUnmap(m->run[0].va, (size_t)m->run[0].n * SZ));
    m->run[0] = m->run[--m->nrun];
  }
  for (int i = 0; i < m->nmem; i++)
    while (m->mem[i].refs-- > 0) CK(cuMemRelease(m->mem[i].h));
  m->nmem = 0;
}

static void* fuzz_thread(void* arg) {
  Model* m = arg;
  unsigned* s = &m->seed;
  m->next_va = 0x100000000000ULL + (CUdeviceptr)m->id * 0x10000000000ULL;
  for (int k = 0; k < m->ops; k++) {
    unsigned r = rnd(s) % 100;
    if (r < 25)
      op_create(m, s);
    else if (r < 45)
      op_map(m, s);
    else if (r < 55)
      op_retain(m, s);
    else if (r < 70)
      op_unmap(m, s);
    else if (r < 88)
      op_release(m, s);
    else if (r < 94)
      op_group(m, s);
    else
      op_ungroup(m, s);
  }
  release_all(m);
  return NULL;
}

static void setup(void) {
  CK(cuInit(0));
  for (int d = 0; d < 2; d++) CK(cuDevicePrimaryCtxRetain(&g_ctx[d], d));
  use(0);
  char p[64];
  snprintf(p, sizeof(p), "present.%d", (int)getpid());
  for (int i = 0; i < 500 && !exists(p); i++) msleep(10);
  EXPECT(exists(p), "the shim's control thread did not start");
}

static int t_fuzz(unsigned seed, int ops, int nthreads) {
  if (nthreads < 1 || nthreads > 64) return 2;
  setup();
  Model* m = calloc((size_t)nthreads, sizeof(Model));
  pthread_t t[64];
  for (int i = 0; i < nthreads; i++) {
    m[i].id = i;
    m[i].seed = seed * 2654435761u + (unsigned)i + 1;
    m[i].ops = ops;
    pthread_create(&t[i], NULL, fuzz_thread, &m[i]);
  }
  int ran = 0, tried = 0, joined[64] = {0};
  for (int alive = nthreads; alive;) {
    msleep(20);
    tried++;
    ran += cycle();
    alive = 0;
    for (int i = 0; i < nthreads; i++) {
      if (!joined[i]) joined[i] = pthread_tryjoin_np(t[i], NULL) == 0;
      alive += !joined[i];
    }
  }
  EXPECT(ran > 0, "no cycle ran (%d tried)", tried);
  EXPECT(fake_live() == 0 && fake_maps() == 0,
         "left %d objects and %d mappings in the driver", fake_live(),
         fake_maps());
  printf("fuzz: seed %u, %d ops x %d threads, %d of %d cycles ran: %s\n", seed,
         ops, nthreads, ran, tried, g_failed ? "FAILED" : "ok");
  free(m);
  return g_failed;
}

static int t_scale(int n) {
  setup();
  enum { RUN = 64 };
  H* h = malloc((size_t)n * sizeof(H));
  int* order = malloc((size_t)n * sizeof(int));
  CUdeviceptr base = 0x200000000000ULL;
  AllocProp p = prop(0);
  Access a = {{1, 0}, 3};
  unsigned s = 12345;
  double t0 = now();
  for (int i = 0; i < n; i++) CK(cuMemCreate(&h[i], SZ, &p, 0));
  double t1 = now();
  for (int i = 0; i < n; i++)
    CK(cuMemMap(base + (CUdeviceptr)i * SZ, SZ, 0, h[i], 0));
  double t2 = now();
  for (int i = 0; i < n; i += RUN) {
    int k = n - i < RUN ? n - i : RUN;
    CK(cuMemSetAccess(base + (CUdeviceptr)i * SZ, (size_t)k * SZ, &a, 1));
  }
  double t3 = now();
  for (int k = 0; k < 1000; k++) {
    int i = (int)(rnd(&s) % (unsigned)n);
    H r;
    CK(cuMemRetainAllocationHandle(&r, (void*)(uintptr_t)(base + i * SZ)));
    EXPECT(r == h[i], "retained 0x%llx, want 0x%llx", r, h[i]);
    CK(cuMemRelease(r));
  }
  double t4 = now();
  EXPECT(cycle() == 1, "the gate refused %d plain allocations", n);
  double t5 = now();
  /* Release every handle in random order; the mappings keep the objects. */
  for (int i = 0; i < n; i++) order[i] = i;
  for (int i = n - 1; i > 0; i--) {
    int j = (int)(rnd(&s) % (unsigned)(i + 1)), x = order[i];
    order[i] = order[j];
    order[j] = x;
  }
  for (int k = 0; k < n; k++) CK(cuMemRelease(h[order[k]]));
  double t6 = now();
  /* Unmap RUN mappings per call; each object dies with its mapping. */
  for (int i = 0; i < n; i += RUN) {
    int k = n - i < RUN ? n - i : RUN;
    CK(cuMemUnmap(base + (CUdeviceptr)i * SZ, (size_t)k * SZ));
  }
  double t7 = now();
  EXPECT(fake_live() == 0 && fake_maps() == 0,
         "left %d objects and %d mappings in the driver", fake_live(),
         fake_maps());
  double us = 1e6 / n;
  printf(
      "scale: %d allocations; per call (us): create %.2f, map %.2f, "
      "set access over %d %.2f, retain+release %.2f, release %.2f, "
      "unmap of %d %.2f; cycle %.1f ms: %s\n",
      n, (t1 - t0) * us, (t2 - t1) * us, RUN, (t3 - t2) * us * RUN,
      (t4 - t3) * 1e3, (t6 - t5) * us, RUN, (t7 - t6) * us * RUN,
      (t5 - t4) * 1e3, g_failed ? "FAILED" : "ok");
  free(h);
  free(order);
  return g_failed;
}

int main(int argc, char** argv) {
  rm("gate");
  rm("suspend");
  if (argc == 5 && !strcmp(argv[1], "fuzz"))
    return t_fuzz((unsigned)atoi(argv[2]), atoi(argv[3]), atoi(argv[4])) ? 1
                                                                          : 0;
  if (argc == 3 && !strcmp(argv[1], "scale"))
    return t_scale(atoi(argv[2])) ? 1 : 0;
  fprintf(stderr, "usage: %s fuzz SEED OPS THREADS | scale N\n", argv[0]);
  return 2;
}
