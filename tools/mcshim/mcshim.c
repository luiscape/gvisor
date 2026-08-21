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

/*
 * mcshim.c -- generic libcuda-level multicast suspend/resume interposer.
 *
 * An LD_PRELOAD shim that interposes the CUDA driver's multicast +
 * virtual-memory-management (VMM) entry points, tracks every multicast group
 * (its handle, participating devices, backing unicast allocations, VA
 * reservations, bindings, and mappings), and on request tears that state
 * down and later rebuilds it -- transparently, for ANY multicast owner (NCCL
 * NVLS, torch _symmetric_memory, raw cuMulticast).
 *
 * Why in-process: freeing the 0x00fd (NV_MEMORY_MULTICAST_FABRIC) objects from
 * nvproxy makes cuda-checkpoint's SAVE succeed but its RESTORE toggle refuse,
 * because libcuda's userspace bookkeeping still lists the multicast allocation.
 * Running the teardown through libcuda (as this shim does) keeps application
 * structs, libcuda tables, and kernel RM state consistent.
 *
 * Orchestration (driven by the gVisor sentry; see
 * pkg/sentry/control/state_cuda_shim.go):
 *   (a) the app is quiesced (the gate marker plus cuda-checkpoint's lock),
 *   (b) SUSPEND: unmap MC VAs keeping the reservations -> unbind -> release the
 *       0x00fd handles.  The multicast blocker set is now empty.
 *   (c) cuda-checkpoint checkpoint/restore.
 *   (d) RESUME: recreate the group (new handle) -> re-addDevice -> re-bind the
 *       same (cuda-checkpoint-restored) unicast handles -> re-map at the
 *       IDENTICAL VAs.  Captured CUDA graphs and app pointers stay valid because
 *       every VA is byte-identical; only the opaque MC handle changes, which
 *       only teardown paths observe (and the shim translates stale handles).
 *
 * Trigger: a background control thread polls $MCSHIM_DIR for an
 * existence-based marker, which makes the protocol race-free for any number
 * of rank processes sharing the control dir:
 *   "suspend" appears    -> perform (b), ack "suspended.<pid>"
 *   "suspend" disappears -> perform (d), ack "resumed.<pid>"
 * The marker lives in the container's filesystem, so it is part of the
 * checkpoint image: after a restore it still exists and the shim stays
 * suspended until the orchestrator (the sentry) removes it.
 *
 * Multi-process ranks (one process per GPU, the vLLM/SGLang TP topology):
 * rank 0 creates the group and exports it (cuMemExportToShareableHandle);
 * peers import it (cuMemImportFromShareableHandle). The shim records the
 * export/import relationship using the exported object's identity as the
 * rendezvous key (see record_key) -- SCM_RIGHTS passes the same open file
 * description, so exporter and importers observe the same identity. On
 * resume the creator re-exports the recreated group and serves the new fd on
 * a unix socket ($MCSHIM_DIR/mcgrp-<key>.sock); importers reconnect,
 * re-import, and every rank re-adds its device and re-binds.
 * cuMulticastBindMem blocks until all devices have joined, so the binds
 * themselves are the cross-rank barrier. Unicast device memory stays
 * cuda-checkpoint's responsibility; the shim only manages the multicast
 * layer, VMM/legacy-IPC imports, and the VA mappings that reference the
 * released handles.
 *
 * Interposition mechanism: plain LD_PRELOAD symbol interposition only catches
 * calls resolved through the global scope (build-time linking). Real CUDA
 * consumers -- torch, NCCL, ctypes -- dlopen libcuda.so.1 and dlsym the
 * entry points (or use cuGetProcAddress), which bypasses symbol
 * interposition. So the shim ALSO interposes dlsym itself (resolving the
 * real dlsym via dlvsym, which we do not wrap) and cuGetProcAddress/_v2,
 * redirecting lookups of the tracked entry points to the shim's wrappers.
 * A third resolution path exists: the CUDA *runtime*'s public
 * cudaGetDriverEntryPoint[ByVersion] exports (torch >= 2.11 resolves its
 * entire c10 DriverAPI table this way), which reach libcuda through
 * libcudart's internal dlvsym/export-table bootstrap that neither hook
 * sees. The torch->libcudart hop is a normal cross-DSO call, though, so
 * the shim interposes those four cudart exports as well.
 *
 * Build:  ./build.sh   (toolkit-free: CUDA types are declared locally)
 */

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
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* Minimal CUDA driver ABI (x86_64), mirrored from cuda.h.            */
/* ------------------------------------------------------------------ */

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUmemGenericAllocationHandle;

#define CUDA_SUCCESS 0
/* Returned by an import when the process cannot address the exporting device. */
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
	void *win32HandleMetaData;
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

/* Legacy (pre-VMM) CUDA IPC. A separate API from cuMemExport/Import: it
 * shares cuMemAlloc'd memory through an opaque 64-byte blob rather than an OS
 * handle, and the blob is passed to cuIpcOpenMemHandle BY VALUE. */
#define CU_IPC_HANDLE_SIZE 64

typedef struct {
	unsigned char reserved[CU_IPC_HANDLE_SIZE];
} CUipcMemHandle;

/* ------------------------------------------------------------------ */
/* Logging                                                            */
/* ------------------------------------------------------------------ */

static FILE *g_log;
static pthread_mutex_t g_loglock = PTHREAD_MUTEX_INITIALIZER;

/* Defined with the control thread below; started lazily from cuInit so
 * only real CUDA consumers poll for control markers. */
static void ensure_control_thread(void);
static void gate_wait(void);

static void mclog(const char *fmt, ...) {
	pthread_mutex_lock(&g_loglock);
	if (!g_log) {
		const char *p = getenv("MCSHIM_LOG");
		g_log = p && *p ? fopen(p, "a") : stderr;
		if (!g_log)
			g_log = stderr;
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

/* ------------------------------------------------------------------ */
/* Real dlsym: resolved via dlvsym (which we do not interpose), so our
 * own dlsym wrapper below can delegate without recursing.            */
/* ------------------------------------------------------------------ */

static void *(*real_dlsym)(void *, const char *);

static void init_real_dlsym(void) {
	if (real_dlsym)
		return;
	/* glibc >= 2.34 moved dlsym into libc under GLIBC_2.34; older
	 * installs export it as GLIBC_2.2.5 (compat alias also present on
	 * new glibc, but prefer the current version). */
	*(void **)(&real_dlsym) = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
	if (!real_dlsym)
		*(void **)(&real_dlsym) =
		    dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
	if (!real_dlsym)
		mclog("FATAL: could not resolve real dlsym via dlvsym");
}

/* ------------------------------------------------------------------ */
/* Real driver symbol resolution.                                     */
/*                                                                    */
/* Resolve against an explicit libcuda.so.1 handle, NOT RTLD_NEXT:    */
/* consumers commonly dlopen libcuda with RTLD_LOCAL (ctypes, torch), */
/* which keeps it out of the global scope RTLD_NEXT searches. dlopen  */
/* of an already-loaded soname just bumps its refcount and returns    */
/* the same handle. Must use real_dlsym here: the interposed dlsym    */
/* below would redirect these names back to our own wrappers.         */
/* ------------------------------------------------------------------ */

static void *libcuda_handle(void) {
	static void *h;
	if (!h)
		h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
	if (!h)
		h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
	if (!h)
		mclog("FATAL: dlopen(libcuda.so.1) failed: %s", dlerror());
	return h;
}

#define REAL(var, name)                                                        \
	do {                                                                   \
		if (!(var)) {                                                  \
			init_real_dlsym();                                     \
			void *h_ = libcuda_handle();                           \
			if (real_dlsym && h_)                                  \
				*(void **)(&(var)) = real_dlsym(h_, name);     \
		}                                                              \
	} while (0)

static CUresult (*r_cuMemCreate)(CUmemGenericAllocationHandle *, size_t,
                                 const CUmemAllocationProp *, unsigned long long);
static CUresult (*r_cuMemRelease)(CUmemGenericAllocationHandle);
static CUresult (*r_cuMemMap)(CUdeviceptr, size_t, size_t,
                              CUmemGenericAllocationHandle, unsigned long long);
static CUresult (*r_cuMemUnmap)(CUdeviceptr, size_t);
static CUresult (*r_cuMemAddressReserve)(CUdeviceptr *, size_t, size_t,
                                         CUdeviceptr, unsigned long long);
static CUresult (*r_cuMemAddressFree)(CUdeviceptr, size_t);
static CUresult (*r_cuMemSetAccess)(CUdeviceptr, size_t, const CUmemAccessDesc *,
                                    size_t);
static CUresult (*r_cuMulticastCreate)(CUmemGenericAllocationHandle *,
                                       const CUmulticastObjectProp *);
static CUresult (*r_cuMulticastAddDevice)(CUmemGenericAllocationHandle, CUdevice);
static CUresult (*r_cuMulticastBindMem)(CUmemGenericAllocationHandle, size_t,
                                        CUmemGenericAllocationHandle, size_t,
                                        size_t, unsigned long long);
static CUresult (*r_cuMulticastBindAddr)(CUmemGenericAllocationHandle, size_t,
                                         CUdeviceptr, size_t,
                                         unsigned long long);
static CUresult (*r_cuMulticastUnbind)(CUmemGenericAllocationHandle, CUdevice,
                                       size_t, size_t);
static CUresult (*r_cuCtxGetDevice)(CUdevice *);
static CUresult (*r_cuMemExportToShareableHandle)(void *,
                                                  CUmemGenericAllocationHandle,
                                                  int, unsigned long long);
static CUresult (*r_cuMemImportFromShareableHandle)(
    CUmemGenericAllocationHandle *, void *, int);
static CUresult (*r_cuCtxGetCurrent)(CUcontext *);
static CUresult (*r_cuCtxSetCurrent)(CUcontext);
static CUresult (*r_cuCtxSynchronize)(void);
static CUresult (*r_cuIpcGetMemHandle)(CUipcMemHandle *, CUdeviceptr);
static CUresult (*r_cuIpcOpenMemHandle)(CUdeviceptr *, CUipcMemHandle,
                                        unsigned int);
static CUresult (*r_cuIpcCloseMemHandle)(CUdeviceptr);
static CUresult (*r_cuMemGetAddressRange)(CUdeviceptr *, size_t *, CUdeviceptr);
static CUresult (*r_cuDeviceGetAttribute)(int *, int, CUdevice);

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
	REAL(r_cuIpcGetMemHandle, "cuIpcGetMemHandle");
	/* cuda.h #defines cuIpcOpenMemHandle to the _v2 ABI; resolve that
	 * first and fall back for drivers that only export the old name. */
	REAL(r_cuIpcOpenMemHandle, "cuIpcOpenMemHandle_v2");
	REAL(r_cuIpcOpenMemHandle, "cuIpcOpenMemHandle");
	REAL(r_cuIpcCloseMemHandle, "cuIpcCloseMemHandle");
	REAL(r_cuMemGetAddressRange, "cuMemGetAddressRange_v2");
	REAL(r_cuMemGetAddressRange, "cuMemGetAddressRange");
}


/* MCSHIM_VERBOSE=1 opts into per-entry suspend/resume diagnostics (which
 * import moved where, reservation outcomes). Off by default: per-entry
 * output across ranks floods the application's stderr. */
static int mcverbose(void) {
	static int v = -1;
	if (v < 0)
		v = getenv("MCSHIM_VERBOSE") != NULL;
	return v;
}

/* Whether to tear legacy CUDA IPC imports down across the checkpoint.
 *
 * On the supported drivers cuda-checkpoint checkpoints processes
 * individually and cannot carry a live legacy IPC import across the
 * per-process restore toggle, so the teardown is required; gVisor sets
 * MCSHIM_IPC_SUSPEND=1 when it preloads the interposer. It stays an env
 * gate (rather than unconditional) because on driver modes where IPC IS
 * covered by cuda-checkpoint itself, touching the imports would be harmful
 * -- cuIpcOpenMemHandle has no address hint, so anything the interposer
 * closes it cannot put back at the same VA (measured: 0 of 58 returned)
 * -- and because the gate is the only way to A/B the question. */
static int ipc_suspend_enabled(void) {
	static int v = -1;
	if (v < 0)
		v = getenv("MCSHIM_IPC_SUSPEND") != NULL;
	return v;
}

/* MCSHIM_IPC_REPLAY_FLOOR=<hex VA> (default 0x40000000000): imports whose
 * range sits BELOW this address are left live across the checkpoint instead
 * of closed and replayed.
 *
 * The populations separate by ADDRESS REGION, not size (a size threshold was
 * tried and misclassified TP=4, whose small imports live high and replay
 * fine). Every unreplayable import observed sits in a low driver region
 * (0x31..-0x3a.., placements the driver makes once in a young process and
 * never repeats -- even a same-blob reopen in the same live process moves);
 * every replayable one sits in the high per-mapping area (~0x7e..). 4 TB
 * splits them with orders of magnitude to spare on both sides. */
static CUdeviceptr ipc_replay_floor(void) {
	static unsigned long long v;
	static int init;
	if (!init) {
		const char *e = getenv("MCSHIM_IPC_REPLAY_FLOOR");
		v = e ? strtoull(e, NULL, 16) : 0x40000000000ULL;
		init = 1;
	}
	return (CUdeviceptr)v;
}



#define mcvlog(...)                                                            \
	do {                                                                   \
		if (mcverbose())                                               \
			mclog(__VA_ARGS__);                                    \
	} while (0)

/* ------------------------------------------------------------------ */
/* Tracked state (the live object graph, at the libcuda layer).       */
/*                                                                    */
/* This is live state, not an ioctl log: app-initiated frees/unbinds  */
/* remove entries, so they drop out of the replay set automatically.  */
/* ------------------------------------------------------------------ */

/* Sized for the largest observed footprint with headroom: an SGLang TP=8
 * rank tracks >512 objects (its per-peer buffers scale with world size),
 * which overflowed the previous MAXN of 512 and disabled suspend. Static
 * tables keep the interposer allocation-free on hot paths; at ~200 bytes per
 * entry the four tables cost ~3 MB per process, which is noise next to a
 * CUDA context. */
#define MAXN 4096
#define MAX_AKA 8
#define MAX_DEV 16

/* KIND_IMP: handle came from cuMemImportFromShareableHandle but has not been
 * classified yet; upgraded to KIND_MC when cuMulticastAddDevice is called on
 * it. (Imports of unicast memory would stay KIND_IMP and are not suspended;
 * cross-rank UC import replay is future work.) */
enum { KIND_FREE = 0, KIND_UC = 1, KIND_MC = 2, KIND_IMP = 3 };

typedef struct {
	int kind;
	CUmemGenericAllocationHandle handle; /* current handle */
	CUmemGenericAllocationHandle aka[MAX_AKA]; /* all handles ever held */
	int naka;
	size_t size;
	CUcontext ctx;
	CUmemAllocationProp uprop; /* KIND_UC */
	CUmulticastObjectProp mprop; /* KIND_MC */
	int devs[MAX_DEV]; /* KIND_MC: added devices */
	int ndev;
	/* Cross-process rendezvous identity (KIND_MC/KIND_IMP): the exported
	 * fd's st_dev:st_ino + a per-key ordinal for the (unlikely) case of
	 * key collisions across multiple groups. */
	int imported; /* 1 = handle came from an import */
	int has_key;
	unsigned long key_dev, key_ino;
	int key_ord;
	/* Creator-side, post-resume: the re-exported fd being served to
	 * importers over a unix socket until the next suspend. */
	int serve_fd;
	int serve_sock;
	char serve_path[104]; /* must fit sockaddr_un.sun_path (108) */
	int serving;
	/* Set by do_suspend once this object is fully torn down (released on
	 * the device); do_resume rebuilds only objects with this set, so a
	 * resume after a PARTIAL suspend failure unwinds exactly what was torn
	 * down instead of double-creating live objects. Cleared the moment the
	 * object is live again (recreated or re-imported), so a suspend
	 * retried after a FAILED resume tears rebuilt objects back down
	 * instead of skipping them. */
	int torn_down;
} Alloc;

typedef struct {
	int used;
	CUdeviceptr va;
	size_t size;
	size_t offset;
	int allocIdx; /* index into g_alloc of the mapped handle */
	CUmemAccessDesc access;
	int has_access;
	CUcontext ctx;
	/* Set by unmap_alloc as each unmap succeeds, so a suspend retried
	 * after a partial failure skips work already done, and a resume after
	 * a partial suspend re-maps exactly what was unmapped. Cleared as
	 * each re-map succeeds and in do_resume's success epilogue. */
	int suspended;
} Mapping;

typedef struct {
	int used;
	int groupIdx; /* index into g_alloc of the MC group */
	int by_addr; /* 1 = cuMulticastBindAddr (replay by VA), 0 = BindMem */
	CUmemGenericAllocationHandle mem; /* BindMem: UC handle (stable) */
	CUdeviceptr va; /* BindAddr: bound VA (stable across restore) */
	size_t mcOffset;
	size_t memOffset;
	size_t size;
	CUcontext ctx;
	CUdevice dev; /* device hosting the memory (unbind is per-device) */
	/* Set by do_suspend as each unbind succeeds (see Mapping.suspended
	 * for the rationale); cleared as each re-bind succeeds and in
	 * do_resume's success epilogue. */
	int unbound;
} Bind;

/* Legacy CUDA IPC participation. Kept in its own table rather than folded
 * into Alloc: legacy IPC has no CUmemGenericAllocationHandle to key on (only
 * a device pointer), and none of Alloc's mapping/bind machinery applies.
 *
 * The rendezvous key is the ORIGINAL blob. The blob a re-export produces
 * after a restore is different (measured), so it cannot identify anything
 * across the checkpoint; but both the exporter and its importers saw the
 * same original bytes, which makes those bytes a cross-process name that
 * survives. */
typedef struct {
	int used;
	int is_import; /* 1 = we opened a peer's handle; 0 = we exported */
	CUdeviceptr ptr; /* import: VA from cuIpcOpenMemHandle
	                  * export: the local pointer we exported */
	CUipcMemHandle blob0; /* the original blob: the rendezvous key */
	unsigned int flags; /* import: flags passed to cuIpcOpenMemHandle */
	CUcontext ctx;
	int seq; /* open order; imports must be reopened in it */
	/* Exporter-side serving state (post-resume). */
	int serving;
	int serve_sock;
	char serve_path[104];
	/* Address reservation held across the checkpoint so the range cannot be
	 * taken while the import is closed (the same trick the VMM path uses --
	 * see remap_alloc's "retained-reservation"). */
	CUdeviceptr resv;
	size_t resv_size;
	/* The mapping's true extent (cuMemGetAddressRange), captured just
	 * before the close; the reopen must land on range_base. */
	CUdeviceptr range_base;
	size_t range_size;
	int closed; /* 1 = closed by suspend, not yet reopened; only these
	             * are reopened, each clearing it as its reopen lands */
} IpcEnt;

static Alloc g_alloc[MAXN];
static Mapping g_map[MAXN];
static Bind g_bind[MAXN];
static IpcEnt g_ipc[MAXN];
static int g_ipc_seq;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
/* All accesses are atomic or under g_gate_lock (see the suspend gate). */
static int g_suspended;

/* Sticky: an object the shim failed to track cannot be torn down at suspend,
 * and an untracked live import turns a restore failure into silent corruption.
 * Set on ANY tracking-table overflow (allocs, mappings, binds, legacy IPC);
 * once set, do_suspend refuses (fail loudly at checkpoint, not after). */
static int g_track_overflow;

static int alloc_new(void) {
	for (int i = 0; i < MAXN; i++)
		if (g_alloc[i].kind == KIND_FREE)
			return i;
	if (!g_track_overflow) {
		g_track_overflow = 1;
		mclog("FATAL: alloc table full (MAXN=%d); object untracked -- "
		      "suspend is disabled for this process", MAXN);
	}
	return -1;
}

/* Find alloc index whose CURRENT handle matches h. */
static int alloc_find(CUmemGenericAllocationHandle h) {
	for (int i = 0; i < MAXN; i++)
		if (g_alloc[i].kind != KIND_FREE && g_alloc[i].handle == h)
			return i;
	return -1;
}

/* Translate a possibly-stale handle to its object's current handle. The app
 * (or NCCL) may retain the original handle in its structs; after a resume a
 * multicast group or an imported allocation has a new handle, so rewrite calls
 * that reference an old value. Applied to every kind: it is the identity for
 * objects whose handle did not rotate, and callers cannot always tell which
 * kind a handle belongs to. */
static CUmemGenericAllocationHandle xlate_mc(CUmemGenericAllocationHandle h) {
	for (int i = 0; i < MAXN; i++) {
		if (g_alloc[i].kind == KIND_FREE)
			continue;
		for (int a = 0; a < g_alloc[i].naka; a++)
			if (g_alloc[i].aka[a] == h)
				return g_alloc[i].handle;
	}
	return h;
}

/* Must hold g_lock. Drop handle value h from every object's alias list: the
 * driver reuses handle values, so when a value is issued anew, any alias
 * still carrying it elsewhere is stale. Leaving it would make xlate_mc
 * rewrite a legitimate reference to the new object into a reference to an
 * unrelated one. Application churn makes such collisions routine -- vLLM's
 * sleep/wake releases and re-creates every weight allocation -- and the
 * symptom is some later, unrelated cuMem* call failing ("operation not
 * supported" out of vLLM's own allocator), intermittently and only after a
 * rebuild has created aliases. Called on every newly issued handle, even
 * ones the shim fails to track (table overflow): an untracked handle must
 * not be misrouted through a stale alias either. */
static void aka_purge(CUmemGenericAllocationHandle h) {
	for (int i = 0; i < MAXN; i++) {
		Alloc *o = &g_alloc[i];
		if (o->kind == KIND_FREE)
			continue;
		for (int k = 0; k < o->naka; k++) {
			if (o->aka[k] != h)
				continue;
			for (int j = k; j + 1 < o->naka; j++)
				o->aka[j] = o->aka[j + 1];
			o->naka--;
			k--;
		}
	}
}

/* Record h as a's current handle, remembering the previous values so that a
 * later call still referring to one of them can be rewritten (see xlate_mc).
 * Must hold g_lock. Purges h from every alias list first (see aka_purge). */
static void alloc_push_aka(Alloc *a, CUmemGenericAllocationHandle h) {
	aka_purge(h);
	a->handle = h;
	if (a->naka < MAX_AKA)
		a->aka[a->naka++] = h;
	else
		a->aka[MAX_AKA - 1] = h;
}

/* Must hold g_lock. Reset slot i and record the current context + handle. */
static Alloc *alloc_init(int i, int kind, CUmemGenericAllocationHandle h) {
	Alloc *a = &g_alloc[i];
	memset(a, 0, sizeof(*a));
	a->kind = kind;
	a->serve_fd = a->serve_sock = -1;
	r_cuCtxGetCurrent(&a->ctx);
	alloc_push_aka(a, h);
	return a;
}

/* Locked translation of a possibly-stale MC handle (see xlate_mc). */
static CUmemGenericAllocationHandle xlate_locked(CUmemGenericAllocationHandle h) {
	pthread_mutex_lock(&g_lock);
	CUmemGenericAllocationHandle r = xlate_mc(h);
	pthread_mutex_unlock(&g_lock);
	return r;
}

/* Per-key ordinal: how many other live allocs already carry this key. Lets
 * multiple groups that hash to the same fd identity still rendezvous, as
 * long as ranks create/import them in the same order. */
static int key_ordinal(unsigned long dev, unsigned long ino, int self) {
	int n = 0;
	for (int i = 0; i < MAXN; i++)
		if (i != self && g_alloc[i].kind != KIND_FREE &&
		    g_alloc[i].has_key && g_alloc[i].key_dev == dev &&
		    g_alloc[i].key_ino == ino)
			n++;
	return n;
}

/* Identity oracle: under gVisor, nvproxy exposes the exported RM object's
 * identity in /proc/self/fdinfo/<fd> as
 *   nvproxy_exported_object:\tclient=0x... object=0x... class=0x...
 * The (client, object) pair is globally unique and identical for exporter
 * and every SCM_RIGHTS recipient (same FileDescription), so it scales to
 * arbitrarily many exported objects. Returns 0 and fills client/object on
 * success. */
static int fdinfo_oracle(int fd, unsigned long *client, unsigned long *object) {
	char path[64], line[256];
	snprintf(path, sizeof(path), "/proc/self/fdinfo/%d", fd);
	FILE *f = fopen(path, "r");
	if (!f)
		return -1;
	int found = -1;
	while (fgets(line, sizeof(line), f)) {
		if (sscanf(line, "nvproxy_exported_object: client=%lx object=%lx",
		           client, object) == 2) {
			found = 0;
			break;
		}
	}
	fclose(f);
	return found;
}

/* Must hold g_lock. Record the rendezvous identity: the nvproxy fdinfo
 * oracle when available (gVisor; scales to any number of objects), else the
 * fd's st_dev:st_ino (native; all NVIDIA export fds are opens of
 * /dev/nvidiactl and share one inode, so only the per-key creation ordinal
 * disambiguates -- fine for few groups only). First key wins: peers
 * rendezvous against the identity of the export they actually received. */
static void record_key(int i, int fd) {
	unsigned long kd, ki;
	struct stat st;
	if (fd < 0)
		return;
	if (fdinfo_oracle(fd, &kd, &ki) != 0) {
		if (fstat(fd, &st) != 0)
			return;
		kd = (unsigned long)st.st_dev;
		ki = (unsigned long)st.st_ino;
	}
	Alloc *a = &g_alloc[i];
	if (a->has_key) {
		mclog("idx=%d re-exported; keeping first key %lx:%lx", i,
		      a->key_dev, a->key_ino);
		return;
	}
	a->has_key = 1;
	a->key_dev = kd;
	a->key_ino = ki;
	a->key_ord = key_ordinal(kd, ki, i);
}

/* ------------------------------------------------------------------ */
/* Interposed entry points                                            */
/* ------------------------------------------------------------------ */

CUresult cuInit(unsigned int flags) {
	static CUresult (*real)(unsigned int);
	REAL(real, "cuInit");
	if (!real)
		return 3; /* CUDA_ERROR_NOT_INITIALIZED */
	/* Only processes that actually initialize CUDA participate in the
	 * suspend/resume protocol (launchers/helpers that merely load libcuda
	 * must not ack markers). */
	ensure_control_thread();
	return real(flags);
}

#define CU_MEM_HANDLE_TYPE_FABRIC 0x8

CUresult cuMemCreate(CUmemGenericAllocationHandle *h, size_t size,
                     const CUmemAllocationProp *prop, unsigned long long flags) {
	resolve_reals();
	gate_wait();
	/* Strip the fabric handle type: it creates an NV_MEMORY_FABRIC (00f8)
	 * object at ALLOCATION time -- before any export -- which
	 * cuda-checkpoint cannot serialize and this shim does not suspend, so
	 * one such allocation blocks every checkpoint. Single-node, fabric
	 * handles buy nothing over POSIX fds (the multicast/NVLS path is
	 * identical), and frameworks request them merely because the device
	 * advertises support (torch >= 2.11 symmetric memory does). Masking
	 * the capability bit in cuDeviceGetAttribute is not sufficient:
	 * statically linked CUDA runtimes reach the driver through paths this
	 * shim cannot interpose, so the request itself must be rewritten. */
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
			mclog("stripping CU_MEM_HANDLE_TYPE_FABRIC from "
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
			Alloc *a = alloc_init(i, KIND_UC, *h);
			a->size = size;
			if (prop)
				a->uprop = *prop;
		} else {
			/* Untracked, but its handle value must still not be
			 * misrouted through a stale alias (see aka_purge). */
			aka_purge(*h);
		}
		pthread_mutex_unlock(&g_lock);
	}
	return rc;
}

CUresult cuMulticastCreate(CUmemGenericAllocationHandle *h,
                           const CUmulticastObjectProp *prop) {
	resolve_reals();
	gate_wait();
	/* Same strip as cuMemCreate: a multicast group created with the
	 * FABRIC handle type gets a companion NV_MEMORY_FABRIC (00f8) object
	 * that cuda-checkpoint cannot serialize and that outlives the shim's
	 * multicast teardown (it belongs to the group's fabric export, not to
	 * the binds). torch 2.11 requests FD|FABRIC unconditionally on
	 * fabric-attached systems; single-node, FD-only is equivalent. */
	CUmulticastObjectProp mfixed;
	if (prop && (prop->handleTypes & CU_MEM_HANDLE_TYPE_FABRIC) &&
	    !getenv("MCSHIM_ALLOW_FABRIC")) {
		mfixed = *prop;
		mfixed.handleTypes &= ~(unsigned long long)CU_MEM_HANDLE_TYPE_FABRIC;
		if (!mfixed.handleTypes)
			mfixed.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FD;
		prop = &mfixed;
		static int logged;
		if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
			mclog("stripping CU_MEM_HANDLE_TYPE_FABRIC from "
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
			Alloc *a = alloc_init(i, KIND_MC, *h);
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

CUresult cuMemExportToShareableHandle(void *shHandle,
                                      CUmemGenericAllocationHandle h, int type,
                                      unsigned long long flags) {
	resolve_reals();
	gate_wait();
	/* Refuse fabric-typed exports. Stripping the handle type at
	 * cuMemCreate is not enough: the driver happily fabric-exports
	 * memory that never requested fabric handles, so torch 2.11's
	 * empirical isFabricSupported probe (create, then export-as-FABRIC)
	 * still succeeds and symm-mem then exchanges CUmemFabricHandles --
	 * one un-serializable NV_MEMORY_FABRIC (00f8) object per pool chunk.
	 * Failing the export makes the probe conclude fabric is unavailable
	 * and fall back to POSIX fds, which the shim fully supports; on a
	 * single node that costs nothing. (Not sufficient by itself: on
	 * fabric-attached systems libcuda also creates driver-internal 00f8
	 * FLA registrations over FD-shared memory once the RM fabric probe
	 * has succeeded -- suppressing those is nvproxy's job, by gating
	 * NV2080_CTRL_CMD_GET_GPU_FABRIC_PROBE_INFO on fabric-imex-mgmt.) */
	if (type == CU_MEM_HANDLE_TYPE_FABRIC && !getenv("MCSHIM_ALLOW_FABRIC")) {
		static int logged;
		if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
			mclog("refusing fabric-typed cuMemExportToShareableHandle:"
			      " fabric exports are not checkpointable; set "
			      "MCSHIM_ALLOW_FABRIC=1 to permit them");
		return CUDA_ERROR_NOT_SUPPORTED;
	}
	CUmemGenericAllocationHandle real_h = xlate_locked(h);
	CUresult rc = r_cuMemExportToShareableHandle(shHandle, real_h, type, flags);
	if (rc == CUDA_SUCCESS && type == CU_MEM_HANDLE_TYPE_POSIX_FD && shHandle) {
		pthread_mutex_lock(&g_lock);
		int i = alloc_find(real_h);
		/* Record the rendezvous identity for BOTH multicast groups and
		 * unicast allocations: a UC export is a P2P peer buffer whose
		 * importers must re-fetch it after restore, so its exporter
		 * must re-export + serve on resume (has_key drives that). */
		if (i >= 0 &&
		    (g_alloc[i].kind == KIND_MC || g_alloc[i].kind == KIND_UC)) {
			record_key(i, *(int *)shHandle);
			mclog("track EXPORT idx=%d kind=%d key=%lx:%lx ord=%d", i,
			      g_alloc[i].kind, g_alloc[i].key_dev,
			      g_alloc[i].key_ino, g_alloc[i].key_ord);
		}
		pthread_mutex_unlock(&g_lock);
	}
	return rc;
}

CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *h,
                                        void *osHandle, int type) {
	resolve_reals();
	gate_wait();
	/* Mirror of the export-side refusal: an imported fabric ref is an
	 * NV_MEMORY_FABRIC_IMPORTED_REF (00fb) object, equally
	 * un-serializable. Unreachable when exports are refused too (peers
	 * then never see a fabric handle); kept for defense in depth. */
	if (type == CU_MEM_HANDLE_TYPE_FABRIC && !getenv("MCSHIM_ALLOW_FABRIC")) {
		static int logged;
		if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
			mclog("refusing fabric-typed cuMemImportFromShareableHandle"
			      " (see export-side message)");
		return CUDA_ERROR_NOT_SUPPORTED;
	}

	CUresult rc = r_cuMemImportFromShareableHandle(h, osHandle, type);
	if (rc == CUDA_SUCCESS && type == CU_MEM_HANDLE_TYPE_POSIX_FD && h) {
		pthread_mutex_lock(&g_lock);
		int i = alloc_new();
		if (i >= 0) {
			Alloc *a = alloc_init(i, KIND_IMP, *h);
			a->imported = 1;
			/* For POSIX-FD imports osHandle is the fd value cast
			 * to void*. */
			record_key(i, (int)(intptr_t)osHandle);
			mclog("track IMPORT idx=%d handle=0x%llx key=%lx:%lx "
			      "ord=%d",
			      i, (unsigned long long)*h, a->key_dev, a->key_ino,
			      a->key_ord);
		} else {
			aka_purge(*h);
		}
		pthread_mutex_unlock(&g_lock);
	}
	return rc;
}

/* NVML fabric-info smoothing.
 *
 * torch >= 2.11's isFabricSupported() TORCH_CHECKs that
 * nvmlDeviceGetGpuFabricInfoV returns NVML_SUCCESS and only then inspects
 * fabricInfo.state -- an NVML error is a hard crash, not a fallback. On any
 * stack where the query errors (measured on a sandbox that denied the RM
 * fabric probe; host/driver variance can produce the same), torch dies at
 * symm-mem init. Translate failure into what a fabric-less host reports:
 * NVML_SUCCESS with state=NVML_GPU_FABRIC_STATE_NOT_SUPPORTED (0), which
 * torch handles gracefully. The versioned-struct size travels in the
 * version field's low 24 bits (NVML_STRUCT_VERSION), so the payload can be
 * zeroed exactly. Opt out with MCSHIM_ALLOW_FABRIC=1. */
static int nvml_fabric_info_smooth(void *info, int rc, const char *via) {
	if (rc == 0 || !info || getenv("MCSHIM_ALLOW_FABRIC"))
		return rc;
	unsigned int version = *(unsigned int *)info;
	unsigned int size = version & 0xffffffu;
	if (size < sizeof(unsigned int) || size > 4096)
		return rc; /* implausible; do not touch */
	memset((char *)info + sizeof(unsigned int), 0,
	       size - sizeof(unsigned int));
	static int logged;
	if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
		mclog("%s failed (rc=%d): reporting fabric NOT_SUPPORTED "
		      "instead; set MCSHIM_ALLOW_FABRIC=1 to pass errors "
		      "through", via, rc);
	return 0; /* NVML_SUCCESS */
}

static void *libnvml_sym(const char *name) {
	init_real_dlsym();
	if (!real_dlsym)
		return NULL;
	void *s = real_dlsym(RTLD_NEXT, name);
	if (s)
		return s;
	static void *h;
	if (!h)
		h = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_NOLOAD);
	return h ? real_dlsym(h, name) : NULL;
}

int nvmlDeviceGetGpuFabricInfoV(void *device, void *gpuFabricInfo);
int nvmlDeviceGetGpuFabricInfoV(void *device, void *gpuFabricInfo) {
	static int (*real)(void *, void *);
	if (!real)
		*(void **)(&real) = libnvml_sym("nvmlDeviceGetGpuFabricInfoV");
	if (!real)
		return 13; /* NVML_ERROR_FUNCTION_NOT_FOUND */
	int rc = real(device, gpuFabricInfo);
	return nvml_fabric_info_smooth(gpuFabricInfo, rc,
	                               "nvmlDeviceGetGpuFabricInfoV");
}

/* Hide fabric-handle support from the application unless explicitly allowed.
 *
 * Memory created with CU_MEM_HANDLE_TYPE_FABRIC becomes an NV_MEMORY_FABRIC
 * (00f8) object that cuda-checkpoint cannot serialize and this shim does not
 * suspend, so it blocks every checkpoint. Frameworks pick the handle type by
 * querying this attribute (torch >= 2.11's symmetric memory prefers fabric
 * handles wherever the device advertises them), and on a single node fabric
 * handles buy nothing over POSIX fds: the multicast/NVLS performance path is
 * identical. Masking the capability steers allocators to fds, which the shim
 * fully supports. MCSHIM_ALLOW_FABRIC=1 restores truthful reporting (e.g.
 * for multi-node, where checkpointing is out of scope anyway). */
#define CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED 132

CUresult cuDeviceGetAttribute(int *pi, int attrib, CUdevice dev) {
	resolve_reals();
	CUresult rc = r_cuDeviceGetAttribute(pi, attrib, dev);
	if (rc == CUDA_SUCCESS && pi &&
	    attrib == CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED &&
	    *pi != 0 && !getenv("MCSHIM_ALLOW_FABRIC")) {
		static int logged;
		if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
			mclog("masking HANDLE_TYPE_FABRIC_SUPPORTED=0 (dev %d): "
			      "fabric-handle exports are not checkpointable; "
			      "set MCSHIM_ALLOW_FABRIC=1 to report the truth",
			      dev);
		*pi = 0;
	}
	/* Opt-in multicast hiding (MCSHIM_HIDE_MULTICAST=1) for workloads whose
	 * multicast path cannot survive restore. Today that is exactly torch
	 * symm-mem multimem: its userspace caches the FLA registration handle
	 * of the covered workspace across exports, the registration cannot be
	 * checkpointed or recreated (NVIDIA-side gap; see nvproxy
	 * fla_registration.go), so its post-restore re-export fails. Masking
	 * this attribute makes torch select its two-shot symm-mem kernel,
	 * which round-trips checkpoints. Do NOT set this for NCCL-NVLS or
	 * FlashInfer-fusion workloads: their multicast state is fully
	 * suspend/resumable and the mask would only cost performance. */
	if (rc == CUDA_SUCCESS && pi &&
	    attrib == CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED && *pi != 0 &&
	    getenv("MCSHIM_HIDE_MULTICAST")) {
		static int logged;
		if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
			mclog("masking MULTICAST_SUPPORTED=0 (dev %d) per "
			      "MCSHIM_HIDE_MULTICAST=1", dev);
		*pi = 0;
	}
	return rc;
}

/* ------------------------------------------------------------------ */
/* Legacy CUDA IPC interposition                                      */
/*                                                                    */
/* In job mode -- the only mode where legacy IPC is checkpointable at  */
/* all -- a live import is the sole thing blocking restore. Exporting  */
/* is fine and needs no teardown; the importer must close, and reopen  */
/* afterwards. (Measured.)                                             */
/* ------------------------------------------------------------------ */

static int ipc_new(void) {
	for (int i = 0; i < MAXN; i++)
		if (!g_ipc[i].used)
			return i;
	return -1;
}

/* Find the live import that owns a device pointer. */
static int ipc_find_import(CUdeviceptr p) {
	for (int i = 0; i < MAXN; i++)
		if (g_ipc[i].used && g_ipc[i].is_import && g_ipc[i].ptr == p)
			return i;
	return -1;
}

static int ipc_find_export(CUdeviceptr p) {
	for (int i = 0; i < MAXN; i++)
		if (g_ipc[i].used && !g_ipc[i].is_import && g_ipc[i].ptr == p)
			return i;
	return -1;
}

/* FNV-1a over the original blob: a short, stable, cross-process name for one
 * shared allocation, usable as a socket path. */
static unsigned long blob_key(const CUipcMemHandle *b) {
	unsigned long h = 1469598103934665603UL;
	for (int i = 0; i < CU_IPC_HANDLE_SIZE; i++) {
		h ^= b->reserved[i];
		h *= 1099511628211UL;
	}
	return h;
}

CUresult cuIpcGetMemHandle(CUipcMemHandle *handle, CUdeviceptr dptr) {
	resolve_reals();
	gate_wait();
	CUresult rc = r_cuIpcGetMemHandle(handle, dptr);
	if (rc != CUDA_SUCCESS || !handle)
		return rc;
	pthread_mutex_lock(&g_lock);
	/* Re-exporting the same pointer is idempotent from our point of view:
	 * keep the FIRST blob, since that is the one peers used as the key. */
	if (ipc_find_export(dptr) < 0) {
		int i = ipc_new();
		if (i >= 0) {
			IpcEnt *e = &g_ipc[i];
			memset(e, 0, sizeof(*e));
			e->used = 1;
			e->is_import = 0;
			e->ptr = dptr;
			e->blob0 = *handle;
			e->serve_sock = -1;
			e->seq = g_ipc_seq++;
			r_cuCtxGetCurrent(&e->ctx);
			mclog("track IPC-EXPORT idx=%d ptr=0x%llx key=%016lx", i,
			      (unsigned long long)dptr, blob_key(handle));
		} else {
			g_track_overflow = 1;
			mclog("IPC-EXPORT table full; ptr=0x%llx UNTRACKED -- "
			      "suspend is disabled for this process",
			      (unsigned long long)dptr);
		}
	}
	pthread_mutex_unlock(&g_lock);
	return rc;
}

CUresult cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle,
                            unsigned int flags) {
	resolve_reals();
	gate_wait();
	CUresult rc = r_cuIpcOpenMemHandle(pdptr, handle, flags);
	if (rc != CUDA_SUCCESS || !pdptr)
		return rc;
	pthread_mutex_lock(&g_lock);
	int i = ipc_new();
	if (i >= 0) {
		IpcEnt *e = &g_ipc[i];
		memset(e, 0, sizeof(*e));
		e->used = 1;
		e->is_import = 1;
		e->ptr = *pdptr;
		e->blob0 = handle;
		e->flags = flags;
		e->serve_sock = -1;
		e->seq = g_ipc_seq++;
		r_cuCtxGetCurrent(&e->ctx);
		/* Record the placement's CONTEXT, not just the address. The
		 * driver places imports inside internal arenas, and which arena
		 * it picks at first-open decides whether a replay can ever get
		 * the address back (TP=8 places some imports in a low arena that
		 * a fresh open never chooses again). The neighbours say what the
		 * arena was created next to. */
		CUdeviceptr rb = 0, nb_lo = 0, nb_hi = 0;
		size_t rs = 0, ns_lo = 0, ns_hi = 0;
		CUresult lo_rc = 999, hi_rc = 999;
		if (r_cuMemGetAddressRange &&
		    r_cuMemGetAddressRange(&rb, &rs, *pdptr) == CUDA_SUCCESS) {
			if (rb >= (2u << 20))
				lo_rc = r_cuMemGetAddressRange(
				    &nb_lo, &ns_lo, rb - 1);
			hi_rc = r_cuMemGetAddressRange(&nb_hi, &ns_hi,
			                               rb + rs);
		}
		mclog("track IPC-IMPORT idx=%d seq=%d va=0x%llx key=%016lx "
		      "range=0x%llx+0x%zx below=[rc=%d 0x%llx+0x%zx] "
		      "above=[rc=%d 0x%llx+0x%zx]",
		      i, e->seq, (unsigned long long)*pdptr, blob_key(&handle),
		      (unsigned long long)rb, rs,
		      lo_rc, (unsigned long long)nb_lo, ns_lo,
		      hi_rc, (unsigned long long)nb_hi, ns_hi);
	} else {
		/* An untracked import is not merely unsupported, it is a
		 * silent restore failure later, so say so loudly now. */
		g_track_overflow = 1;
		mclog("IPC-IMPORT table full; va=0x%llx UNTRACKED -- restore "
		      "WILL fail",
		      (unsigned long long)*pdptr);
	}
	pthread_mutex_unlock(&g_lock);
	return rc;
}

/* cuda.h maps cuIpcOpenMemHandle to the _v2 ABI; both names must resolve to
 * the wrapper or an app linking the versioned symbol bypasses the shim. */
CUresult cuIpcOpenMemHandle_v2(CUdeviceptr *pdptr, CUipcMemHandle handle,
                               unsigned int flags) {
	return cuIpcOpenMemHandle(pdptr, handle, flags);
}

CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) {
	resolve_reals();
	gate_wait();
	CUresult rc = r_cuIpcCloseMemHandle(dptr);
	if (rc == CUDA_SUCCESS) {
		pthread_mutex_lock(&g_lock);
		int i = ipc_find_import(dptr);
		if (i >= 0) {
			g_ipc[i].used = 0;
			mcvlog("untrack IPC-IMPORT idx=%d va=0x%llx", i,
			       (unsigned long long)dptr);
		}
		pthread_mutex_unlock(&g_lock);
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
		/* An AddDevice on an imported handle proves it is a multicast
		 * group: classify it so suspend/resume manages it. */
		if (i >= 0 && g_alloc[i].kind == KIND_IMP) {
			g_alloc[i].kind = KIND_MC;
			mclog("import idx=%d classified as MC group", i);
		}
		if (i >= 0 && g_alloc[i].kind == KIND_MC &&
		    g_alloc[i].ndev < MAX_DEV) {
			/* Avoid duplicate entries across resume re-adds. */
			int dup = 0;
			for (int d = 0; d < g_alloc[i].ndev; d++)
				if (g_alloc[i].devs[d] == dev)
					dup = 1;
			if (!dup)
				g_alloc[i].devs[g_alloc[i].ndev++] = dev;
		}
		pthread_mutex_unlock(&g_lock);
	}
	return rc;
}

/* Must hold g_lock. Record a successful bind unless it is already tracked
 * (resume re-binds existing entries through the reals, but a paranoid app
 * re-binding the same range must not duplicate). */
static void bind_record(int gi, int by_addr, CUmemGenericAllocationHandle mem,
                        CUdeviceptr va, size_t mcOffset, size_t memOffset,
                        size_t size, CUdevice dev) {
	if (gi < 0)
		return;
	for (int b = 0; b < MAXN; b++)
		if (g_bind[b].used && g_bind[b].groupIdx == gi &&
		    g_bind[b].by_addr == by_addr && g_bind[b].mem == mem &&
		    g_bind[b].va == va && g_bind[b].mcOffset == mcOffset &&
		    g_bind[b].memOffset == memOffset && g_bind[b].size == size)
			return;
	for (int b = 0; b < MAXN; b++) {
		if (g_bind[b].used)
			continue;
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
		mclog("track BIND%s group=%d %s=0x%llx dev=%d mcOff=0x%zx "
		      "size=0x%zx",
		      by_addr ? "-ADDR" : "", gi, by_addr ? "va" : "mem",
		      (unsigned long long)(by_addr ? va : mem), dev, mcOffset,
		      size);
		return;
	}
	g_track_overflow = 1;
	mclog("FATAL: bind table full (MAXN=%d); bind not tracked -- suspend "
	      "is disabled for this process", MAXN);
}

CUresult cuMulticastBindMem(CUmemGenericAllocationHandle mc, size_t mcOffset,
                            CUmemGenericAllocationHandle mem, size_t memOffset,
                            size_t size, unsigned long long flags) {
	resolve_reals();
	gate_wait();
	CUmemGenericAllocationHandle real_mc = xlate_locked(mc);
	/* The bound memory may itself be referenced by a stale handle (an
	 * import whose handle rotated across a rebuild); translate it too, and
	 * record the translated value so dedupe and later unbind/rebind key on
	 * the current handle. */
	CUmemGenericAllocationHandle real_mem = xlate_locked(mem);

	CUresult rc = r_cuMulticastBindMem(real_mc, mcOffset, real_mem,
	                                   memOffset, size, flags);
	if (rc == CUDA_SUCCESS) {
		pthread_mutex_lock(&g_lock);
		/* The unbind is per-device: the device hosting the memory
		 * comes from the UC alloc's prop. */
		CUdevice dev = -1;
		int mi = alloc_find(real_mem);
		if (mi >= 0 && g_alloc[mi].kind == KIND_UC)
			dev = g_alloc[mi].uprop.location.id;
		bind_record(alloc_find(real_mc), 0, real_mem, 0, mcOffset,
		            memOffset, size, dev);
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
		/* Replay is by VA (stable across restore); the hosting device
		 * is the caller's current device. */
		CUdevice dev = -1;
		r_cuCtxGetDevice(&dev);
		bind_record(alloc_find(real_mc), 1, 0, memptr, mcOffset, 0,
		            size, dev);
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
	/* App-initiated: drop this device's recorded bind so it leaves the
	 * replay set. (While suspended the app is quiesced by protocol; the
	 * shim's own teardown calls the reals directly and never gets here.) */
	if (rc == CUDA_SUCCESS && !__atomic_load_n(&g_suspended, __ATOMIC_SEQ_CST)) {
		pthread_mutex_lock(&g_lock);
		int gi = alloc_find(real_mc);
		for (int b = 0; b < MAXN; b++)
			if (g_bind[b].used && g_bind[b].groupIdx == gi &&
			    g_bind[b].dev == dev &&
			    g_bind[b].mcOffset == mcOffset &&
			    g_bind[b].size == size)
				g_bind[b].used = 0;
		pthread_mutex_unlock(&g_lock);
	}
	return rc;
}

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle handle, unsigned long long flags) {
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
				if (g_map[m].used)
					continue;
				g_map[m].used = 1;
				g_map[m].va = ptr;
				g_map[m].size = size;
				g_map[m].offset = offset;
				g_map[m].allocIdx = ai;
				g_map[m].has_access = 0;
				g_map[m].suspended = 0;
				r_cuCtxGetCurrent(&g_map[m].ctx);
				placed = 1;
				break;
			}
			if (!placed) {
				g_track_overflow = 1;
				mclog("FATAL: mapping table full (MAXN=%d); "
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
			if (g_map[m].used && g_map[m].va == ptr)
				g_map[m].used = 0;
		pthread_mutex_unlock(&g_lock);
	}
	return rc;
}

CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size,
                        const CUmemAccessDesc *desc, size_t count) {
	resolve_reals();
	gate_wait();
	CUresult rc = r_cuMemSetAccess(ptr, size, desc, count);
	if (rc == CUDA_SUCCESS && desc && count >= 1) {
		if (count > 1)
			mclog("NOTE: cuMemSetAccess va=0x%llx count=%zu "
			      "(only desc[0] recorded!)",
			      (unsigned long long)ptr, count);
		pthread_mutex_lock(&g_lock);
		for (int m = 0; m < MAXN; m++)
			if (g_map[m].used && g_map[m].va == ptr) {
				g_map[m].access = desc[0];
				g_map[m].has_access = 1;
			}
		pthread_mutex_unlock(&g_lock);
	}
	return rc;
}

static void stop_serving(Alloc *a);

/* Must hold g_lock. Forget alloc i and everything that references it, so a
 * reused slot can never inherit stale binds/maps (freed objects must drop
 * out of the replay set, and dependents die with their object). */
static void alloc_forget(int i) {
	for (int b = 0; b < MAXN; b++)
		if (g_bind[b].used &&
		    (g_bind[b].groupIdx == i ||
		     g_bind[b].mem == g_alloc[i].handle))
			g_bind[b].used = 0;
	for (int m = 0; m < MAXN; m++)
		if (g_map[m].used && g_map[m].allocIdx == i)
			g_map[m].used = 0;
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
		if (i >= 0)
			alloc_forget(i);
		pthread_mutex_unlock(&g_lock);
	}
	return rc;
}

/* ------------------------------------------------------------------ */
/* Cross-rank fd rendezvous: the creator serves the re-exported fd on */
/* a unix socket keyed by the original export identity; importers     */
/* connect and receive it via SCM_RIGHTS.                             */
/* ------------------------------------------------------------------ */

/* Fallback only: the sentry normally sets MCSHIM_DIR explicitly (see
 * DefaultCudaMulticastShimDir in pkg/sentry/control/state_cuda_shim.go,
 * which this default matches). */
static char g_dir[512] = "/tmp/mcshim";

/* Returns 0, or -1 if the path would not fit sockaddr_un.sun_path (keep
 * MCSHIM_DIR short, e.g. /tmp/mcshim). */
static int group_sock_path(const Alloc *a, char *out, size_t n) {
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
	struct cmsghdr *c = CMSG_FIRSTHDR(&msg);
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
	/* MSG_CMSG_CLOEXEC: a received export fd leaking into a forked child
	 * keeps the RM object alive there and blocks the NEXT checkpoint. */
	if (recvmsg(sock, &msg, MSG_CMSG_CLOEXEC) <= 0)
		return -1;
	struct cmsghdr *c = CMSG_FIRSTHDR(&msg);
	if (!c || c->cmsg_type != SCM_RIGHTS)
		return -1;
	int fd;
	memcpy(&fd, CMSG_DATA(c), sizeof(int));
	return fd;
}

/* Creator-side accept loop: hand the re-exported fd to each connecting
 * importer. Owns its heap-allocated args (never the Alloc, whose slot may be
 * freed and reused while this thread runs), its OWN dup of the served fd
 * (so stop_serving closing the original can never turn a late-accepted
 * connection into an SCM_RIGHTS send of a closed -- possibly reused -- fd
 * number), and the LISTENING socket (stop_serving only shutdown()s it; if it
 * closed it, the number could be reused and a late accept would steal an
 * unrelated fd). Exits when stop_serving shuts the socket down. */
typedef struct {
	int sock;
	int fd;
	char path[104];
} ServeArgs;

static void *serve_thread(void *arg) {
	ServeArgs *sa = arg;
	mclog("serving group fd on %s", sa->path);
	for (;;) {
		int c = accept4(sa->sock, NULL, NULL, SOCK_CLOEXEC);
		if (c < 0) {
			/* A stray signal (SIGCHLD is routine under Python
			 * multiprocessing) must not kill the exporter while
			 * peers are still fetching. */
			if (errno == EINTR)
				continue;
			break; /* stop_serving's shutdown(), or a real error */
		}
		if (send_fd(c, sa->fd) != 0)
			mclog("serve: send_fd failed: %s", strerror(errno));
		close(c);
	}
	mclog("serve thread for %s exiting", sa->path);
	close(sa->sock); /* thread-owned (see stop_serving) */
	close(sa->fd); /* the thread's own dup (see start_serving) */
	free(sa);
	return NULL;
}

/* Must hold g_lock (mutates Alloc). */
static int start_serving(Alloc *a, int fd) {
	if (group_sock_path(a, a->serve_path, sizeof(a->serve_path)) != 0)
		return -1;
	struct sockaddr_un sa;
	unlink(a->serve_path);
	int s = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
	if (s < 0)
		return -1;
	memset(&sa, 0, sizeof(sa));
	sa.sun_family = AF_UNIX;
	strcpy(sa.sun_path, a->serve_path);
	if (bind(s, (struct sockaddr *)&sa, sizeof(sa)) != 0 ||
	    listen(s, 64) != 0) {
		mclog("RESUME: bind/listen(%s) failed: %s", a->serve_path,
		      strerror(errno));
		close(s);
		return -1;
	}
	ServeArgs *args = malloc(sizeof(*args));
	if (!args) {
		close(s);
		return -1;
	}
	/* The thread serves its own dup, decoupled from a->serve_fd (see
	 * serve_thread). F_DUPFD_CLOEXEC: a plain dup would drop CLOEXEC. */
	int tfd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
	if (tfd < 0) {
		close(s);
		free(args);
		return -1;
	}
	args->sock = s;
	args->fd = tfd;
	strcpy(args->path, a->serve_path);
	pthread_t t;
	if (pthread_create(&t, NULL, serve_thread, args) != 0) {
		close(s);
		close(tfd);
		free(args);
		return -1;
	}
	pthread_detach(t);
	a->serve_fd = fd;
	a->serve_sock = s;
	a->serving = 1;
	return 0;
}

static void stop_serving(Alloc *a) {
	if (!a->serving)
		return;
	if (a->serve_sock >= 0) {
		/* close() alone does not reliably wake a thread blocked in
		 * accept(); shutdown() does (AF_UNIX), making the thread exit
		 * deterministic. The LISTENING fd is closed by the serve
		 * thread itself, never here: closing a possibly-still-
		 * accepting fd would let the number be reused and a late
		 * accept steal an unrelated socket. */
		shutdown(a->serve_sock, SHUT_RDWR);
		a->serve_sock = -1;
	}
	if (a->serve_path[0])
		unlink(a->serve_path);
	if (a->serve_fd >= 0) {
		close(a->serve_fd); /* exported fds are checkpoint blockers */
		a->serve_fd = -1;
	}
	a->serving = 0;
	mclog("stopped serving group fd");
}

/* ------------------------------------------------------------------ */
/* Legacy IPC rendezvous: same shape as the fd rendezvous above, but   */
/* the payload is 64 opaque bytes, so no SCM_RIGHTS is involved.       */
/* ------------------------------------------------------------------ */

static int ipc_sock_path(const IpcEnt *e, char *out, size_t n) {
	int w = snprintf(out, n, "%s/ipcblob-%016lx.sock", g_dir,
	                 blob_key(&e->blob0));
	if (w < 0 || (size_t)w >= n) {
		mclog("IPC socket path too long (MCSHIM_DIR=%s)", g_dir);
		return -1;
	}
	return 0;
}

typedef struct {
	int sock;
	CUipcMemHandle blob;
	char path[104];
} IpcServeArgs;

static void *ipc_serve_thread(void *arg) {
	IpcServeArgs *sa = arg;
	mclog("serving IPC blob on %s", sa->path);
	for (;;) {
		int c = accept4(sa->sock, NULL, NULL, SOCK_CLOEXEC);
		if (c < 0) {
			if (errno == EINTR)
				continue; /* see serve_thread */
			break;
		}
		ssize_t w = write(c, sa->blob.reserved, CU_IPC_HANDLE_SIZE);
		if (w != CU_IPC_HANDLE_SIZE)
			mclog("IPC serve: short write (%zd): %s", w,
			      strerror(errno));
		close(c);
	}
	mclog("IPC serve thread for %s exiting", sa->path);
	close(sa->sock); /* thread-owned (see ipc_stop_serving) */
	free(sa);
	return NULL;
}

/* Must hold g_lock. */
static int ipc_start_serving(IpcEnt *e, const CUipcMemHandle *blob) {
	if (ipc_sock_path(e, e->serve_path, sizeof(e->serve_path)) != 0)
		return -1;
	struct sockaddr_un sa;
	unlink(e->serve_path);
	int s = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
	if (s < 0)
		return -1;
	memset(&sa, 0, sizeof(sa));
	sa.sun_family = AF_UNIX;
	strcpy(sa.sun_path, e->serve_path);
	if (bind(s, (struct sockaddr *)&sa, sizeof(sa)) != 0 ||
	    listen(s, 64) != 0) {
		mclog("RESUME: IPC bind/listen(%s) failed: %s", e->serve_path,
		      strerror(errno));
		close(s);
		return -1;
	}
	IpcServeArgs *args = malloc(sizeof(*args));
	if (!args) {
		close(s);
		return -1;
	}
	args->sock = s;
	args->blob = *blob;
	strcpy(args->path, e->serve_path);
	pthread_t t;
	if (pthread_create(&t, NULL, ipc_serve_thread, args) != 0) {
		close(s);
		free(args);
		return -1;
	}
	pthread_detach(t);
	e->serve_sock = s;
	e->serving = 1;
	return 0;
}

static void ipc_stop_serving(IpcEnt *e) {
	if (!e->serving)
		return;
	if (e->serve_sock >= 0) {
		/* shutdown() reliably wakes the thread out of accept();
		 * close() alone does not, and the listening fd itself is
		 * thread-owned (see stop_serving for why). */
		shutdown(e->serve_sock, SHUT_RDWR);
		e->serve_sock = -1;
	}
	if (e->serve_path[0])
		unlink(e->serve_path);
	e->serving = 0;
}

/* ------------------------------------------------------------------ */
/* Resume deadline.                                                    */
/*                                                                     */
/* The sentry gives up waiting for resumed.<pid> after 5 minutes. The  */
/* shim's per-fetch timeouts and re-import retries could exceed that   */
/* in aggregate, leaving the two sides disagreeing about the outcome.  */
/* So every resume gets one total budget, comfortably inside the       */
/* sentry's, and every rendezvous wait is capped by what remains.      */
/* ------------------------------------------------------------------ */

#define RESUME_DEADLINE_MS (240 * 1000L)

static long mono_ms(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000L + ts.tv_nsec / 1000000L;
}

/* mono_ms() deadline for the resume in flight; set at do_resume entry. */
static long g_resume_deadline;

/* Remaining resume budget in ms; <= 0 once the deadline has passed. */
static long resume_budget_ms(void) {
	return g_resume_deadline - mono_ms();
}

/* Importer-side: connect (with retry; the exporter may not be serving yet)
 * and read the re-exported blob. */
static int ipc_fetch_blob(const IpcEnt *e, CUipcMemHandle *out, int timeout_ms) {
	char path[104];
	if (ipc_sock_path(e, path, sizeof(path)) != 0)
		return -1;
	long budget = resume_budget_ms();
	if (budget <= 0) {
		mclog("RESUME: resume deadline (%lds) exceeded before fetching "
		      "IPC blob", RESUME_DEADLINE_MS / 1000);
		return -1;
	}
	if ((long)timeout_ms > budget)
		timeout_ms = (int)budget;
	struct sockaddr_un sa;
	memset(&sa, 0, sizeof(sa));
	sa.sun_family = AF_UNIX;
	strcpy(sa.sun_path, path);
	for (int waited = 0; waited < timeout_ms; waited += 100) {
		int s = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
		if (s < 0)
			return -1;
		if (connect(s, (struct sockaddr *)&sa, sizeof(sa)) == 0) {
			size_t got = 0;
			while (got < CU_IPC_HANDLE_SIZE) {
				ssize_t r = read(s, out->reserved + got,
				                 CU_IPC_HANDLE_SIZE - got);
				if (r <= 0)
					break;
				got += (size_t)r;
			}
			close(s);
			if (got == CU_IPC_HANDLE_SIZE)
				return 0;
		} else {
			close(s);
		}
		struct timespec ts = {0, 100 * 1000 * 1000};
		nanosleep(&ts, NULL);
	}
	mclog("RESUME: timed out fetching IPC blob from %s", path);
	return -1;
}

/* Importer-side: connect (with retry; the creator may not be serving yet)
 * and receive the new group fd. */
static int fetch_group_fd(const Alloc *a, int timeout_ms) {
	char path[104];
	if (group_sock_path(a, path, sizeof(path)) != 0)
		return -1;
	long budget = resume_budget_ms();
	if (budget <= 0) {
		mclog("RESUME: resume deadline (%lds) exceeded before fetching "
		      "group fd", RESUME_DEADLINE_MS / 1000);
		return -1;
	}
	if ((long)timeout_ms > budget)
		timeout_ms = (int)budget;
	struct sockaddr_un sa;
	memset(&sa, 0, sizeof(sa));
	sa.sun_family = AF_UNIX;
	strcpy(sa.sun_path, path);
	for (int waited = 0; waited < timeout_ms; waited += 100) {
		int s = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
		if (s < 0)
			return -1;
		if (connect(s, (struct sockaddr *)&sa, sizeof(sa)) == 0) {
			int fd = recv_fd(s);
			close(s);
			if (fd >= 0)
				return fd;
		} else {
			close(s);
		}
		struct timespec ts = {0, 100 * 1000 * 1000};
		nanosleep(&ts, NULL);
	}
	mclog("RESUME: timed out fetching group fd from %s", path);
	return -1;
}

/* ------------------------------------------------------------------ */
/* Suspend / resume shared helpers                                    */
/* ------------------------------------------------------------------ */

/* Must hold g_lock. Unmap every VA that maps alloc gi, KEEPING the VA
 * reservations (cuMemUnmap only -- never cuMemAddressFree). */
static int unmap_alloc(int gi, const char *what, int *unmapped) {
	for (int m = 0; m < MAXN; m++) {
		if (!g_map[m].used || g_map[m].allocIdx != gi)
			continue;
		if (g_map[m].suspended)
			continue; /* already unmapped by an earlier attempt */
		if (g_map[m].ctx)
			r_cuCtxSetCurrent(g_map[m].ctx);
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

/* Must hold g_lock. Re-map every VA of alloc gi at its IDENTICAL address,
 * backed by handle h. Prefers the retained reservation; re-reserves at the
 * fixed address if it did not survive restore.
 *
 * partial != 0 means alloc gi is still LIVE (a suspend failed before tearing
 * it down): re-map only the mappings that suspend actually unmapped
 * (.suspended), the rest are still mapped. For torn-down objects every
 * recorded mapping is re-mapped, exactly as before partial unwind existed. */
static int remap_alloc(int gi, CUmemGenericAllocationHandle h,
                       const char *what, int partial, int *remapped) {
	for (int m = 0; m < MAXN; m++) {
		if (!g_map[m].used || g_map[m].allocIdx != gi)
			continue;
		if (partial && !g_map[m].suspended)
			continue; /* still mapped; nothing to redo */
		if (g_map[m].ctx)
			r_cuCtxSetCurrent(g_map[m].ctx);
		const char *path = "retained-reservation";
		CUresult rc = r_cuMemMap(g_map[m].va, g_map[m].size,
		                         g_map[m].offset, h, 0);
		if (rc != CUDA_SUCCESS) {
			CUdeviceptr got = 0;
			CUresult rr = r_cuMemAddressReserve(
			    &got, g_map[m].size, 0, g_map[m].va, 0);
			if (rr != CUDA_SUCCESS || got != g_map[m].va) {
				if (rr == CUDA_SUCCESS)
					r_cuMemAddressFree(got, g_map[m].size);
				mclog("RESUME: %s re-map at identical VA 0x%llx "
				      "failed (got 0x%llx rr=%d)",
				      what, (unsigned long long)g_map[m].va,
				      (unsigned long long)got, rr);
				return -1;
			}
			rc = r_cuMemMap(g_map[m].va, g_map[m].size,
			                g_map[m].offset, h, 0);
			if (rc != CUDA_SUCCESS) {
				mclog("RESUME: %s cuMemMap after re-reserve "
				      "rc=%d", what, rc);
				return -1;
			}
			path = "re-reserved-fixed";
		}
		/* Grant access. NCCL frequently sets access once over a whole
		 * reservation range rather than per sub-map, so a per-map
		 * capture (has_access) misses it and the re-mapped view ends up
		 * inaccessible -> the collective kernel faults (719) on the rank
		 * whose import lost access. Always (re)grant RW for the mapping's
		 * owning device, which is what NCCL's P2P imports and NVLS VAs
		 * need; fall back to any captured descriptor. */
		CUmemAccessDesc acc;
		if (g_map[m].has_access) {
			acc = g_map[m].access;
		} else {
			CUdevice d = -1;
			r_cuCtxGetDevice(&d);
			memset(&acc, 0, sizeof(acc));
			acc.location.type = 1 /* CU_MEM_LOCATION_TYPE_DEVICE */;
			acc.location.id = d;
			acc.flags = 3 /* CU_MEM_ACCESS_FLAGS_PROT_READWRITE */;
		}
		CUresult ac = r_cuMemSetAccess(g_map[m].va, g_map[m].size, &acc, 1);
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

/* Must hold g_lock. Synchronize every distinct tracked context and log the
 * result. CUDA latches an unrecoverable fault into the context, so the first
 * probe that reports non-zero brackets exactly when the context died. Purely
 * diagnostic: never fatal, and every context is probed so all ranks report. */
static void ctx_probe(const char *tag) {
	CUcontext saved = NULL;
	r_cuCtxGetCurrent(&saved);
	for (int i = 0; i < MAXN; i++) {
		if (g_alloc[i].kind == KIND_FREE || !g_alloc[i].ctx)
			continue;
		int seen = 0;
		for (int j = 0; j < i; j++)
			if (g_alloc[j].kind != KIND_FREE &&
			    g_alloc[j].ctx == g_alloc[i].ctx) {
				seen = 1;
				break;
			}
		if (seen)
			continue;
		r_cuCtxSetCurrent(g_alloc[i].ctx);
		CUresult sy = r_cuCtxSynchronize ? r_cuCtxSynchronize() : 0;
		mclog("CTXPROBE[%s] ctx=%p sync=%d%s", tag, g_alloc[i].ctx, sy,
		      sy ? "  <-- FAULTED" : "");
	}
	if (saved)
		r_cuCtxSetCurrent(saved);
}

/* Must hold g_lock. Re-export alloc gi's handle h and start serving it on the
 * rendezvous socket, so importers can fetch it. */
static int reexport_serve(int gi, CUmemGenericAllocationHandle h) {
	/* cuMemExportToShareableHandle can transiently return INVALID_VALUE (1)
	 * on a freshly cuda-checkpoint-restored allocation: the handle is valid
	 * (it survived restore) but the driver's export path is briefly not
	 * ready. Left unretried this aborts the rank's resume, so peers time out
	 * fetching the buffers it should serve -> ~10% one-rank 719. Retry with a
	 * short backoff, bounded so a genuine failure stays loud. Mirrors the
	 * re-import 304 retry. */
	int fd = -1;
	CUresult rc = 0;
	for (int attempt = 0; attempt < 100; attempt++) {
		rc = r_cuMemExportToShareableHandle(
		    &fd, h, CU_MEM_HANDLE_TYPE_POSIX_FD, 0);
		if (rc == CUDA_SUCCESS && fd >= 0)
			break;
		if (attempt == 0) {
			CUcontext cur = NULL;
			r_cuCtxGetCurrent(&cur);
			mclog("RESUME: re-export idx=%d kind=%d handle=0x%llx "
			      "ctx=%p cur=%p rc=%d fd=%d, retrying",
			      gi, g_alloc[gi].kind, (unsigned long long)h,
			      g_alloc[gi].ctx, cur, rc, fd);
		}
		struct timespec ts = {0, 100 * 1000 * 1000}; /* 100ms */
		nanosleep(&ts, NULL);
	}
	if (rc != CUDA_SUCCESS || fd < 0) {
		mclog("RESUME: re-export idx=%d gave up rc=%d fd=%d", gi, rc, fd);
		return -1;
	}
	/* The export fd must not leak into forked children, where it keeps
	 * the RM object alive and blocks the NEXT checkpoint. */
	fcntl(fd, F_SETFD, FD_CLOEXEC);
	if (start_serving(&g_alloc[gi], fd) != 0) {
		close(fd);
		return -1;
	}
	return 0;
}

/* Must hold g_lock. Fetch alloc gi's re-exported fd from its exporter and
 * re-import it into *out. Concurrent imports of the same object can
 * transiently fail (CUDA_ERROR_OPERATING_SYSTEM=304 when several ranks import
 * within ~1ms), so retry with a fresh fd, bounded so a real failure is loud. */
static int reimport(int gi, CUmemGenericAllocationHandle *out) {
	if (!g_alloc[gi].has_key) {
		mclog("RESUME: imported idx=%d has no rendezvous key", gi);
		return -1;
	}
	for (int attempt = 0;; attempt++) {
		if (resume_budget_ms() <= 0) {
			mclog("RESUME: resume deadline (%lds) exceeded during "
			      "re-import idx=%d (attempt %d)",
			      RESUME_DEADLINE_MS / 1000, gi, attempt);
			return -1;
		}
		int fd = fetch_group_fd(&g_alloc[gi], 60 * 1000);
		if (fd < 0)
			return -1;
		CUresult rc = r_cuMemImportFromShareableHandle(
		    out, (void *)(intptr_t)fd, CU_MEM_HANDLE_TYPE_POSIX_FD);
		close(fd);
		if (rc == CUDA_SUCCESS) {
			if (attempt > 0)
				mclog("RESUME: re-import idx=%d key=%lx:%lx ok "
				      "after %d retries",
				      gi, g_alloc[gi].key_dev,
				      g_alloc[gi].key_ino, attempt);
			return 0;
		}
		/* Report the FIRST failure as it happens. Retrying silently for
		 * 20s and only then reporting hides both the timing (which is
		 * what correlates this against the sentry's ioctl log) and the
		 * fact that the very first attempt already failed. */
		if (attempt == 0) {
			CUdevice cur = -1;
			r_cuCtxGetDevice(&cur);
			mclog("RESUME: re-import idx=%d key=%lx:%lx dev=%d "
			      "rc=%d on first attempt",
			      gi, g_alloc[gi].key_dev, g_alloc[gi].key_ino,
			      cur, rc);
		}
		/* CUDA_ERROR_INVALID_DEVICE is never transient: the importing
		 * process cannot address the exporter's device at all, which
		 * retrying cannot change. Fail immediately instead of burning
		 * the resume budget. */
		if (rc == CUDA_ERROR_INVALID_DEVICE) {
			mclog("RESUME: re-import idx=%d gave up: INVALID_DEVICE "
			      "(the importer cannot address the exporting device; "
			      "after a restore onto different GPUs this means the "
			      "sentry did not translate RM-reported device identity "
			      "-- see nvproxy's GET_EXPORT_OBJECT_INFO handling)",
			      gi);
			return -1;
		}
		if (attempt >= 100) {
			mclog("RESUME: re-import idx=%d rc=%d after %d attempts",
			      gi, rc, attempt);
			return -1;
		}
		struct timespec ts = {0, 200 * 1000 * 1000};
		nanosleep(&ts, NULL);
	}
}

/* ------------------------------------------------------------------ */
/* Suspend                                                            */
/* ------------------------------------------------------------------ */

/* Must hold g_lock. */
static int do_suspend(void) {
	int groups = 0, imports = 0, unmapped = 0, unbound = 0, released = 0;
	CUcontext saved = NULL;
	r_cuCtxGetCurrent(&saved);

	if (g_track_overflow) {
		mclog("SUSPEND: refusing: tracking table overflowed earlier, "
		      "so some object is untracked and cannot be torn down");
		return -1;
	}

	ctx_probe("suspend-entry");

	/* Stop serving any previous resume's re-exported fds first: a held
	 * export fd is itself a checkpoint blocker. */
	for (int i = 0; i < MAXN; i++)
		if (g_alloc[i].serving)
			stop_serving(&g_alloc[i]);

	/* Multicast groups: unmap MC VAs, unbind each device, release the
	 * 0x00fd handle. Objects already torn down, binds already unbound and
	 * mappings already unmapped (per-entry flags) are skipped, so a
	 * suspend retried after a partial failure resumes where it left off
	 * instead of re-tearing (and failing on) work already done. */
	for (int gi = 0; gi < MAXN; gi++) {
		if (g_alloc[gi].kind != KIND_MC)
			continue;
		if (g_alloc[gi].torn_down)
			continue; /* fully done by an earlier attempt */
		groups++;
		if (unmap_alloc(gi, "MC", &unmapped) != 0)
			return -1;
		for (int b = 0; b < MAXN; b++) {
			if (!g_bind[b].used || g_bind[b].groupIdx != gi)
				continue;
			if (g_bind[b].unbound)
				continue; /* already unbound by an earlier attempt */
			if (g_bind[b].dev < 0) {
				mclog("SUSPEND: bind %d has unknown device", b);
				return -1;
			}
			if (g_bind[b].ctx)
				r_cuCtxSetCurrent(g_bind[b].ctx);
			CUresult rc = r_cuMulticastUnbind(
			    g_alloc[gi].handle, g_bind[b].dev,
			    g_bind[b].mcOffset, g_bind[b].size);
			if (rc != CUDA_SUCCESS) {
				mclog("SUSPEND: cuMulticastUnbind(mc=0x%llx, "
				      "dev=%d, mcOff=0x%zx, size=0x%zx) rc=%d",
				      (unsigned long long)g_alloc[gi].handle,
				      g_bind[b].dev, g_bind[b].mcOffset,
				      g_bind[b].size, rc);
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

	/* UC imports (P2P peer buffers): unmap the views and release the
	 * imported handle. The backing physical memory is the EXPORTER's
	 * resident allocation, saved by cuda-checkpoint, so no content backup
	 * is needed here -- only the local import must be released so this
	 * process holds no live VMM import across the checkpoint (which R610
	 * cuda-checkpoint cannot restore). */
	for (int ii = 0; ii < MAXN; ii++) {
		if (g_alloc[ii].kind != KIND_IMP)
			continue;
		if (g_alloc[ii].torn_down)
			continue; /* fully done by an earlier attempt */
		imports++;
		/* Layout diagnostics: how many mappings back this import, and
		 * their (va, size, offset). NCCL mapping imports at nonzero
		 * offsets or an import with != 1 mapping would mean the shim's
		 * replay must reproduce that exactly. */
		int nmap = 0;
		for (int m = 0; m < MAXN; m++) {
			if (!g_map[m].used || g_map[m].allocIdx != ii)
				continue;
			nmap++;
			if (g_map[m].offset != 0 || nmap > 1)
				mclog("IMPORT-LAYOUT idx=%d map#%d va=0x%llx "
				      "size=0x%zx offset=0x%zx",
				      ii, nmap,
				      (unsigned long long)g_map[m].va,
				      g_map[m].size, g_map[m].offset);
		}
		if (nmap != 1)
			mclog("IMPORT-LAYOUT idx=%d has %d mappings", ii, nmap);
		if (unmap_alloc(ii, "UC-import", &unmapped) != 0)
			return -1;
		CUresult rc = r_cuMemRelease(g_alloc[ii].handle);
		if (rc != CUDA_SUCCESS) {
			mclog("SUSPEND: cuMemRelease(import 0x%llx) rc=%d",
			      (unsigned long long)g_alloc[ii].handle, rc);
			return -1;
		}
		g_alloc[ii].torn_down = 1;
		released++;
	}

	/* Legacy CUDA IPC imports. Only the import blocks the restore -- the
	 * exporter keeps its allocation and needs no teardown -- so close
	 * every live one and leave the exports alone.
	 *
	 * Closing order does not matter; REOPEN order does, and is replayed
	 * from each entry's seq on resume. */
	int ipc_closed = 0, ipc_live = 0;
	for (int i = 0; i < MAXN; i++) {
		if (g_ipc[i].serving)
			ipc_stop_serving(&g_ipc[i]);
		if (!g_ipc[i].used || !g_ipc[i].is_import)
			continue;
		if (!ipc_suspend_enabled()) {
			/* Left for the driver's job-mode IPC support to carry
			 * across the checkpoint. Counted so the log still says
			 * how much legacy IPC is in play. */
			ipc_live++;
			continue;
		}
		if (g_ipc[i].closed) {
			/* Closed by an earlier attempt (a partial suspend, or a
			 * failed resume that never reopened it). Its range and
			 * any held reservation are still valid: re-closing would
			 * fail, and resetting resv would leak the hold. */
			ipc_closed++;
			continue;
		}
		if (g_ipc[i].ctx)
			r_cuCtxSetCurrent(g_ipc[i].ctx);
		/* Learn the mapping's extent before closing it (the reopen must
		 * land back on it), then close. The RESERVE happens in a second
		 * pass below, after every import is closed: reserving here, one
		 * import at a time, asks for a granule-rounded range while the
		 * neighbouring imports are still mapped, and any overlap makes
		 * the reservation silently land elsewhere. */
		g_ipc[i].resv = 0;
		g_ipc[i].resv_size = 0;
		g_ipc[i].closed = 0;
		if (r_cuMemGetAddressRange &&
		    r_cuMemGetAddressRange(&g_ipc[i].range_base,
		                           &g_ipc[i].range_size,
		                           g_ipc[i].ptr) != CUDA_SUCCESS) {
			g_ipc[i].range_base = g_ipc[i].ptr;
			g_ipc[i].range_size = 0;
		}
		/* Classify BEFORE closing, because closing an unreplayable
		 * import is unrecoverable: even an immediate reopen of the same
		 * blob in the same live process lands ~75 TB away (measured --
		 * the placement is a one-time young-process decision the driver
		 * never repeats). Such imports must cross the checkpoint live,
		 * carried by the driver's own (intermittent) job-mode support.
		 *
		 * The classifier is SIZE. Small exporter allocations are
		 * suballocated inside driver-owned regions where user
		 * reservations are not honored (fixed-address reserves of
		 * provably free space get bumped), so no replay can ever place
		 * them; allocations with dedicated mappings live in user space
		 * where close+hold+walk is measured to work. Every observation
		 * so far separates cleanly: 0x41300 and 0x200000 imports are
		 * driver-owned (TP=8's signal pads), 0x801300 and 0x1000000
		 * ones are replayable (TP=2/4/8 data buffers). An adjacency-
		 * probing classifier was tried and misfires in the high arena;
		 * a wrong guess here is loud, not corrupting (live imports can
		 * fail the toggle; replayed ones verify their address). */
		if (g_ipc[i].range_base < ipc_replay_floor()) {
			ipc_live++;
			mclog("SUSPEND: leaving IPC import seq=%d va=0x%llx "
			      "(low-region placement -> unreplayable) live "
			      "for the driver to carry",
			      g_ipc[i].seq, (unsigned long long)g_ipc[i].ptr);
			continue;
		}
		CUresult rc = r_cuIpcCloseMemHandle(g_ipc[i].ptr);
		if (rc != CUDA_SUCCESS) {
			mclog("SUSPEND: cuIpcCloseMemHandle(0x%llx) rc=%d",
			      (unsigned long long)g_ipc[i].ptr, rc);
			return -1;
		}
		g_ipc[i].closed = 1;
		ipc_closed++;
		mcvlog("SUSPEND: closed IPC import idx=%d seq=%d va=0x%llx "
		       "range=0x%llx+0x%zx",
		       i, g_ipc[i].seq, (unsigned long long)g_ipc[i].ptr,
		       (unsigned long long)g_ipc[i].range_base,
		       g_ipc[i].range_size);
	}

	/* Second pass: hold every closed import's range across the checkpoint,
	 * so the restore cannot put something else there (which is what pushed
	 * reopened imports 21.6 GB away). Now that ALL imports are closed, a
	 * granule-rounded reservation cannot collide with a still-mapped
	 * neighbour.
	 *
	 * cuMemAddressReserve's addr argument is a HINT: on contention it
	 * SUCCEEDS at a different address rather than failing. Both outcomes
	 * are logged distinctly, with what occupies the wanted range, because
	 * conflating them cost a day of theorizing already. */
	int ipc_held = 0;
	for (int i = 0; i < MAXN; i++) {
		if (!ipc_suspend_enabled())
			break;
		if (!g_ipc[i].used || !g_ipc[i].is_import ||
		    !g_ipc[i].closed || !g_ipc[i].range_size)
			continue;
		if (g_ipc[i].resv) {
			ipc_held++; /* still held by an earlier attempt */
			continue;
		}
		if (g_ipc[i].ctx)
			r_cuCtxSetCurrent(g_ipc[i].ctx);
		const size_t gran = 2u << 20;
		CUdeviceptr base = g_ipc[i].range_base;
		size_t sz = (g_ipc[i].range_size + gran - 1) & ~(gran - 1);
		CUdeviceptr r = 0;
		CUresult rc = r_cuMemAddressReserve(&r, sz, 0, base, 0);
		if (rc == CUDA_SUCCESS && r == base) {
			g_ipc[i].resv = r;
			g_ipc[i].resv_size = sz;
			ipc_held++;
			continue;
		}
		/* The hold attempt is the CLASSIFIER. If the hint was honored,
		 * this import lives in reservable address space and the replay
		 * machinery can put it back. If not -- reserve failed, or
		 * succeeded somewhere else while the wanted range sits provably
		 * free -- then the range is inside a driver-owned region
		 * (0x31..-0x3b.. here) where user reservations are not honored,
		 * fences cannot be built, and no replay can work. There is no
		 * revival: the import stays torn down with no held range, and
		 * the resume walk fails loudly if it cannot place it. */
		if (rc == CUDA_SUCCESS) {
			mcvlog("SUSPEND: reserve for seq=%d mislanded: wanted "
			       "0x%llx+0x%zx, got 0x%llx -> driver-owned range",
			       g_ipc[i].seq, (unsigned long long)base, sz,
			       (unsigned long long)r);
			r_cuMemAddressFree(r, sz);
		} else {
			mcvlog("SUSPEND: reserve for seq=%d failed rc=%d "
			       "(wanted 0x%llx+0x%zx) -> driver-owned range",
			       g_ipc[i].seq, rc, (unsigned long long)base, sz);
		}
	}

	if (ipc_closed || ipc_live)
		mclog("SUSPEND: legacy IPC: %d closed (%d held; replayable), "
		      "%d left live (driver-owned or suspend disabled)",
		      ipc_closed, ipc_held, ipc_live);

	ctx_probe("suspend-exit");

	if (saved)
		r_cuCtxSetCurrent(saved);
	if (r_cuCtxSynchronize)
		r_cuCtxSynchronize();
	mclog("SUSPEND done: groups=%d imports=%d unmapped=%d unbound=%d "
	      "released=%d ipc_closed=%d ipc_left_live=%d",
	      groups, imports, unmapped, unbound, released, ipc_closed,
	      ipc_live);
	return 0;
}

/* ------------------------------------------------------------------ */
/* Resume (three phases)                                              */
/*                                                                    */
/* A rank is simultaneously an EXPORTER (of its multicast groups and its    */
/* P2P peer buffers) and an IMPORTER (of its peers'). If it fetched before  */
/* it served, two ranks could deadlock waiting on each other. So every      */
/* exporter starts serving (phase 1) before anyone fetches (phase 2), and   */
/* bindings/mappings that need all handles resolved come last (phase 3).    */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Multicast proxy (pre-R610 drivers).                                */
/*                                                                    */
/* On R580, a cuda-checkpoint-restored process can IMPORT a multicast */
/* group fd, BIND its memory to it, and MAP the multicast VA -- but   */
/* cuMulticastCreate and cuMulticastAddDevice fail with               */
/* CUDA_ERROR_INVALID_DEVICE, and cuCtxCreate fails with OOM: the     */
/* restore blocks fresh device admission at the PROCESS level         */
/* (measured: gpu_mem_snapshots/phase0/native_mc_after_restore.py).   */
/* A never-checkpointed helper (mcshim-helper, exec'd here) performs  */
/* exactly create+attach on the rank's behalf; the group persists     */
/* once the rank holds its own import (measured:                      */
/* native_mc_proxy_restore.py), so the helper exits right after the   */
/* rebuild. Enabled by MCSHIM_MC_PROXY (the gVisor loader sets it).   */
/* ------------------------------------------------------------------ */

static int g_helper_sock = -1;
static pid_t g_helper_pid;

static int mc_proxy_enabled(void) {
	static int v = -1;
	if (v < 0)
		v = getenv("MCSHIM_MC_PROXY") != NULL;
	return v;
}

/* Resolve the helper binary: $MCSHIM_HELPER, else "mcshim-helper" next to
 * this library. */
static int helper_path(char *buf, size_t n) {
	const char *env = getenv("MCSHIM_HELPER");
	if (env && *env) {
		snprintf(buf, n, "%s", env);
		return 0;
	}
	Dl_info info;
	if (!dladdr((void *)&helper_path, &info) || !info.dli_fname)
		return -1;
	snprintf(buf, n, "%s", info.dli_fname);
	char *slash = strrchr(buf, '/');
	if (!slash)
		return -1;
	snprintf(slash + 1, n - (slash + 1 - buf), "mcshim-helper");
	return 0;
}

static int helper_start(void) {
	if (g_helper_sock >= 0)
		return 0;
	char path[512];
	if (helper_path(path, sizeof(path)) != 0) {
		mclog("RESUME: cannot locate mcshim-helper");
		return -1;
	}
	int sp[2];
	/* The child's end crosses exec, so no SOCK_CLOEXEC on creation; the
	 * parent's end gets FD_CLOEXEC below. */
	if (socketpair(AF_UNIX, SOCK_STREAM, 0, sp) != 0) {
		mclog("RESUME: helper socketpair: %s", strerror(errno));
		return -1;
	}
	pid_t pid = fork();
	if (pid < 0) {
		mclog("RESUME: helper fork: %s", strerror(errno));
		close(sp[0]);
		close(sp[1]);
		return -1;
	}
	if (pid == 0) {
		/* Child: exec the helper with only its socket end. The shim in
		 * the child (ld.so.preload) is silenced; the helper resolves
		 * the driver itself. */
		close(sp[0]);
		char fdstr[16];
		snprintf(fdstr, sizeof(fdstr), "%d", sp[1]);
		setenv("MCSHIM_DISABLE", "1", 1);
		execl(path, path, fdstr, (char *)NULL);
		_exit(127);
	}
	close(sp[1]);
	fcntl(sp[0], F_SETFD, FD_CLOEXEC);
	g_helper_sock = sp[0];
	g_helper_pid = pid;
	mclog("RESUME: multicast proxy helper started (pid=%d, %s)",
	      (int)pid, path);
	return 0;
}

/* One command round-trip: send line (+ optional fd), await "OK"/"ERR" (+
 * optional fd). Bounded by the resume budget. */
static int helper_txn(const char *cmd, int fd_send, int *fd_recv) {
	if (helper_start() != 0)
		return -1;
	struct iovec iov = {(void *)cmd, strlen(cmd)};
	char cbuf[CMSG_SPACE(sizeof(int))];
	struct msghdr mh;
	memset(&mh, 0, sizeof(mh));
	mh.msg_iov = &iov;
	mh.msg_iovlen = 1;
	if (fd_send >= 0) {
		mh.msg_control = cbuf;
		mh.msg_controllen = sizeof(cbuf);
		struct cmsghdr *c = CMSG_FIRSTHDR(&mh);
		c->cmsg_level = SOL_SOCKET;
		c->cmsg_type = SCM_RIGHTS;
		c->cmsg_len = CMSG_LEN(sizeof(int));
		memcpy(CMSG_DATA(c), &fd_send, sizeof(int));
	}
	if (sendmsg(g_helper_sock, &mh, 0) < 0) {
		mclog("RESUME: helper send (%s): %s", cmd, strerror(errno));
		return -1;
	}
	long budget = resume_budget_ms();
	if (budget <= 0 || budget > 60000)
		budget = 60000;
	struct pollfd pfd = {g_helper_sock, POLLIN, 0};
	int pr = poll(&pfd, 1, (int)budget);
	if (pr <= 0) {
		mclog("RESUME: helper reply timeout (%s)", cmd);
		return -1;
	}
	char rbuf[128];
	struct iovec riov = {rbuf, sizeof(rbuf) - 1};
	char rcbuf[CMSG_SPACE(sizeof(int))];
	memset(&mh, 0, sizeof(mh));
	mh.msg_iov = &riov;
	mh.msg_iovlen = 1;
	mh.msg_control = rcbuf;
	mh.msg_controllen = sizeof(rcbuf);
	ssize_t r = recvmsg(g_helper_sock, &mh, MSG_CMSG_CLOEXEC);
	if (r <= 0) {
		mclog("RESUME: helper died (%s)", cmd);
		return -1;
	}
	rbuf[r] = 0;
	if (fd_recv) {
		*fd_recv = -1;
		for (struct cmsghdr *c = CMSG_FIRSTHDR(&mh); c;
		     c = CMSG_NXTHDR(&mh, c))
			if (c->cmsg_level == SOL_SOCKET &&
			    c->cmsg_type == SCM_RIGHTS)
				memcpy(fd_recv, CMSG_DATA(c), sizeof(int));
	}
	if (strncmp(rbuf, "OK", 2) != 0) {
		mclog("RESUME: helper %s -> %s", cmd, rbuf);
		return -1;
	}
	return 0;
}

static void helper_stop(void) {
	if (g_helper_sock < 0)
		return;
	/* Best-effort EXIT; EOF on our end makes the helper exit anyway. */
	ssize_t n = write(g_helper_sock, "EXIT\n", 5);
	(void)n;
	close(g_helper_sock);
	g_helper_sock = -1;
	if (g_helper_pid > 0) {
		for (int i = 0; i < 50; i++) {
			if (waitpid(g_helper_pid, NULL, WNOHANG) != 0)
				break;
			struct timespec ts = {0, 100 * 1000 * 1000};
			nanosleep(&ts, NULL);
		}
		g_helper_pid = 0;
	}
}

/* Proxy a group's AddDevice list through the helper. */
static int helper_add_devices(int gi) {
	for (int d = 0; d < g_alloc[gi].ndev; d++) {
		char cmd[64];
		snprintf(cmd, sizeof(cmd), "ADDDEV %d\n", g_alloc[gi].devs[d]);
		if (helper_txn(cmd, -1, NULL) != 0)
			return -1;
	}
	return 0;
}

static int resume_reopen_ipc(int p1_ipc);

/* Must hold g_lock. */
static int do_resume(void) {
	int groups = 0, imports = 0, remapped = 0, rebound = 0, served = 0;
	CUcontext saved = NULL;
	r_cuCtxGetCurrent(&saved);

	/* One total budget for the whole rebuild, comfortably inside the
	 * sentry's 5-minute ack timeout: fail loudly here rather than let the
	 * two sides time out disagreeing about the outcome. */
	g_resume_deadline = mono_ms() + RESUME_DEADLINE_MS;

	/* Snapshot which objects need the FULL rebuild (torn down by suspend).
	 * torn_down itself is cleared below the moment an object is live
	 * again, so a suspend retried after a FAILED resume tears rebuilt
	 * objects back down instead of skipping them, and a retried resume
	 * cannot double-create them. Phases 2 and 3 therefore decide off this
	 * snapshot, never the live flag. */
	char full[MAXN];
	for (int gi = 0; gi < MAXN; gi++)
		full[gi] = (char)g_alloc[gi].torn_down;

	/* Phase 0: warm each context. The first VMM/IPC call issued from this
	 * (control) thread on a freshly cuda-checkpoint-restored context can
	 * return CUDA_ERROR_UNKNOWN; a synchronize first clears it. (Rank 0
	 * was masked because its first op is cuMulticastCreate; importer ranks,
	 * whose first op was a re-export, hit the stale-context error.)
	 *
	 * The synchronize result is also the earliest possible health probe:
	 * it runs before the shim has issued any rebuild work, so a sticky
	 * error here means cuda-checkpoint's restore itself left the context
	 * faulted, and no rebuild policy can be responsible. Logged, not
	 * fatal, so every rank reports. */
	ctx_probe("resume-entry");
	ctx_probe("resume-warm");

	/* Phase 1: every exporter re-establishes its object and starts
	 * serving the re-exported fd. */
	int p1_mc = 0, p1_uc = 0;
	for (int gi = 0; gi < MAXN; gi++) {
		/* The context switch happens after the kind checks so a
		 * KIND_FREE slot can never install a stale context. */
		if (g_alloc[gi].kind == KIND_MC && !g_alloc[gi].imported) {
			if (g_alloc[gi].ctx)
				r_cuCtxSetCurrent(g_alloc[gi].ctx);
			/* Multicast creator: new group handle, then serve it.
			 * After a PARTIAL suspend failure a group may still be
			 * live (torn_down unset); serve its existing handle
			 * without recreating, so peers that DID tear down can
			 * still refetch during the unwind. */
			CUmemGenericAllocationHandle newmc = g_alloc[gi].handle;
			int need_serve = 1;
			if (full[gi] && mc_proxy_enabled()) {
				/* Pre-R610: create+attach in the helper, then hold
				 * the group via our own import (binds and maps are
				 * measured to work in the restored process). The
				 * HELPER's export fd is also what gets served to
				 * peers: re-exporting an imported group handle from
				 * a restored process fails on R580. */
				char cmd[160];
				snprintf(cmd, sizeof(cmd),
				         "CREATE %u %zu %llu %llu\n",
				         g_alloc[gi].mprop.numDevices,
				         (size_t)g_alloc[gi].mprop.size,
				         (unsigned long long)g_alloc[gi].mprop.handleTypes,
				         (unsigned long long)g_alloc[gi].mprop.flags);
				int mcfd = -1;
				if (helper_txn(cmd, -1, &mcfd) != 0 || mcfd < 0) {
					mclog("RESUME: proxied cuMulticastCreate "
					      "idx=%d failed", gi);
					return -1;
				}
				if (helper_add_devices(gi) != 0) {
					close(mcfd);
					return -1;
				}
				CUresult rc = r_cuMemImportFromShareableHandle(
				    &newmc, (void *)(intptr_t)mcfd,
				    CU_MEM_HANDLE_TYPE_POSIX_FD);
				if (rc != CUDA_SUCCESS) {
					close(mcfd);
					mclog("RESUME: import of proxied group "
					      "idx=%d rc=%d", gi, rc);
					return -1;
				}
				alloc_push_aka(&g_alloc[gi], newmc);
				g_alloc[gi].torn_down = 0;
				if (g_alloc[gi].has_key) {
					fcntl(mcfd, F_SETFD, FD_CLOEXEC);
					if (start_serving(&g_alloc[gi], mcfd) != 0) {
						close(mcfd);
						return -1;
					}
				} else {
					close(mcfd);
				}
				need_serve = 0;
			} else if (full[gi]) {
				if (r_cuMulticastCreate(&newmc,
				                        &g_alloc[gi].mprop) !=
				    CUDA_SUCCESS) {
					mclog("RESUME: cuMulticastCreate idx=%d "
					      "failed", gi);
					return -1;
				}
				alloc_push_aka(&g_alloc[gi], newmc);
				/* Live again: a retried suspend must tear it
				 * down, not skip it. */
				g_alloc[gi].torn_down = 0;
			}
			if (need_serve && g_alloc[gi].has_key &&
			    reexport_serve(gi, newmc) != 0)
				return -1;
			groups++;
			p1_mc++;
		} else if (g_alloc[gi].kind == KIND_UC) {
			CUmemGenericAllocationHandle h = g_alloc[gi].handle;
			if (g_alloc[gi].has_key) {
				if (g_alloc[gi].ctx)
					r_cuCtxSetCurrent(g_alloc[gi].ctx);
				/* Exporter of a P2P peer buffer: serve the handle
				 * importers must fetch. */
				if (reexport_serve(gi, h) != 0)
					return -1;
				served++;
				p1_uc++;
			}
		}
	}
	/* Legacy IPC exporters re-export and serve here too, in phase 1, for
	 * the same anti-deadlock reason: a rank that is an exporter to one
	 * peer and an importer from another must be serving before anybody
	 * starts fetching. */
	int p1_ipc = 0;
	for (int i = 0; i < MAXN; i++) {
		if (!ipc_suspend_enabled())
			break; /* no importer will fetch; do not re-export */
		if (!g_ipc[i].used || g_ipc[i].is_import)
			continue;
		if (g_ipc[i].ctx)
			r_cuCtxSetCurrent(g_ipc[i].ctx);
		/* The blob a re-export produces differs from the original, so
		 * importers cannot reuse what they have; serve the new one
		 * under the original blob's key, which both sides still know. */
		CUipcMemHandle nb;
		CUresult rc = r_cuIpcGetMemHandle(&nb, g_ipc[i].ptr);
		if (rc != CUDA_SUCCESS) {
			mclog("RESUME: cuIpcGetMemHandle(0x%llx) rc=%d",
			      (unsigned long long)g_ipc[i].ptr, rc);
			return -1;
		}
		if (ipc_start_serving(&g_ipc[i], &nb) != 0)
			return -1;
		p1_ipc++;
	}

	mclog("RESUME: phase1 done (%d MC creators, %d UC exporters served, "
	      "%d IPC exporters served)",
	      p1_mc, p1_uc, p1_ipc);

	/* Phase 2: importers fetch the re-exported fd and re-import (new
	 * handle). Serving is already up for every exporter, so no deadlock. */
	for (int gi = 0; gi < MAXN; gi++) {
		if (!full[gi])
			continue; /* still live (partial suspend); nothing to redo */
		if (g_alloc[gi].kind == KIND_MC && g_alloc[gi].imported) {
			if (g_alloc[gi].ctx)
				r_cuCtxSetCurrent(g_alloc[gi].ctx);
			CUmemGenericAllocationHandle newmc = 0;
			if (reimport(gi, &newmc) != 0)
				return -1;
			alloc_push_aka(&g_alloc[gi], newmc);
			g_alloc[gi].torn_down = 0; /* live again */
			/* Pre-R610: this rank's AddDevice calls must also run in
			 * the helper. Re-exporting our import fails on R580, so
			 * fetch a SECOND fd from the creator (still serving) and
			 * hand that to the helper. */
			if (mc_proxy_enabled() && g_alloc[gi].ndev > 0) {
				int fd = fetch_group_fd(&g_alloc[gi], 60 * 1000);
				if (fd < 0) {
					mclog("RESUME: refetch for proxied "
					      "AddDevice idx=%d failed", gi);
					return -1;
				}
				int ok = helper_txn("IMPORT\n", fd, NULL) == 0 &&
				         helper_add_devices(gi) == 0;
				close(fd);
				if (!ok)
					return -1;
			}
			groups++;
		} else if (g_alloc[gi].kind == KIND_IMP) {
			if (g_alloc[gi].ctx)
				r_cuCtxSetCurrent(g_alloc[gi].ctx);
			CUmemGenericAllocationHandle newh = 0;
			if (reimport(gi, &newh) != 0)
				return -1;
			alloc_push_aka(&g_alloc[gi], newh);
			g_alloc[gi].torn_down = 0; /* live again */
			imports++;
		}
	}

	/* Phase 3: rebuild bindings and re-map every VA at its IDENTICAL
	 * address, now that all handles (local and imported) are resolved.
	 *
	 * Torn-down objects take the full path: AddDevice replay, every
	 * recorded bind, every recorded mapping (exactly the validated
	 * behavior). Objects a partial suspend left LIVE need only their
	 * per-entry unwind: re-bind the binds it unbound (.unbound) and
	 * re-map the mappings it unmapped (.suspended), against the LIVE
	 * handle -- no AddDevice replay, since no device was ever detached
	 * from a group that was never released. */
	for (int gi = 0; gi < MAXN; gi++) {
		if (g_alloc[gi].kind != KIND_MC && g_alloc[gi].kind != KIND_IMP)
			continue;
		int partial = !full[gi];
		if (g_alloc[gi].ctx)
			r_cuCtxSetCurrent(g_alloc[gi].ctx);
		if (g_alloc[gi].kind == KIND_MC) {
			CUmemGenericAllocationHandle mc = g_alloc[gi].handle;
			/* In proxy mode every AddDevice already ran in the helper
			 * (phase 1 for creators, phase 2 for importers). */
			if (!partial && !mc_proxy_enabled())
				for (int d = 0; d < g_alloc[gi].ndev; d++)
					if (r_cuMulticastAddDevice(
					        mc, g_alloc[gi].devs[d]) !=
					    CUDA_SUCCESS) {
						mclog("RESUME: AddDevice dev=%d "
						      "failed",
						      g_alloc[gi].devs[d]);
						return -1;
					}
			/* cuMulticastBindMem blocks until every device has
			 * joined -- the binds are the cross-rank barrier. */
			for (int b = 0; b < MAXN; b++) {
				if (!g_bind[b].used || g_bind[b].groupIdx != gi)
					continue;
				if (partial && !g_bind[b].unbound)
					continue; /* still bound; nothing to redo */
				if (g_bind[b].ctx)
					r_cuCtxSetCurrent(g_bind[b].ctx);
				CUresult rc =
				    g_bind[b].by_addr
				        ? r_cuMulticastBindAddr(
				              mc, g_bind[b].mcOffset,
				              g_bind[b].va, g_bind[b].size, 0)
				        : r_cuMulticastBindMem(
				              mc, g_bind[b].mcOffset,
				              /* The bound memory may itself be an
				               * imported handle, and those do
				               * rotate across a rebuild; identity
				               * for anything that did not. */
				              xlate_mc(g_bind[b].mem),
				              g_bind[b].memOffset,
				              g_bind[b].size, 0);
				if (rc != CUDA_SUCCESS) {
					mclog("RESUME: re-bind (%s) rc=%d",
					      g_bind[b].by_addr ? "addr" : "mem",
					      rc);
					return -1;
				}
				g_bind[b].unbound = 0;
				rebound++;
			}
			if (remap_alloc(gi, mc, "MC", partial, &remapped) != 0)
				return -1;
		} else {
			if (remap_alloc(gi, g_alloc[gi].handle, "UC-import",
			                partial, &remapped) != 0)
				return -1;
		}
	}

	/* Phase 4: reopen legacy IPC imports.
	 *
	 * Last, and in ascending open order. cuIpcOpenMemHandle takes no
	 * address hint -- unlike cuMemAddressReserve, the driver picks the
	 * address, and it picks the next free slot. So the original VA comes
	 * back only if the allocation state it sees matches what it saw the
	 * first time, which is why this runs after every VMM mapping has been
	 * restored to its identical address, and why nothing may allocate in
	 * between (and why the whole cuMemAlloc* family is gated). Measured:
	 * an interposed allocation moves the import by exactly one slot.
	 *
	 * A moved import is silent corruption: the application still holds the
	 * old pointer and nothing returns an error. So verify, and fail loudly
	 * rather than hand back a working-looking process. */
	if (resume_reopen_ipc(p1_ipc) != 0)
		return -1;

	/* Everything is rebuilt; the next suspend starts from a clean slate.
	 * (torn_down was already cleared per-object as each rebuild landed;
	 * the sweeps keep every flag consistent regardless.) */
	for (int gi = 0; gi < MAXN; gi++)
		g_alloc[gi].torn_down = 0;
	for (int b = 0; b < MAXN; b++)
		g_bind[b].unbound = 0;
	for (int m = 0; m < MAXN; m++)
		g_map[m].suspended = 0;

	if (saved)
		r_cuCtxSetCurrent(saved);
	if (r_cuCtxSynchronize)
		r_cuCtxSynchronize();
	mclog("RESUME done: groups=%d imports=%d served=%d rebound=%d "
	      "remapped=%d",
	      groups, imports, served, rebound, remapped);
	return 0;
}

/* Reopen every legacy IPC import, in its original open order, and verify each
 * lands where it was. Must hold g_lock. */
/* Temporary reservations plugging arena holes during the reopen walk. Freed
 * once every import is placed; until then they are what keeps import N+1 from
 * falling into the holes already probed for import N. */
#define IPC_MAX_PLUGS 16384
static struct {
	CUdeviceptr base;
	size_t size;
} g_plugs[IPC_MAX_PLUGS];

static int resume_reopen_ipc(int p1_ipc) {
	int ipc_reopened = 0, ipc_moved = 0, ipc_replaced = 0;
	int nplugs = 0;
	if (!ipc_suspend_enabled())
		return 0; /* nothing was torn down, so nothing to rebuild */


	for (int pass_seq = 0; pass_seq < g_ipc_seq; pass_seq++) {
		for (int i = 0; i < MAXN; i++) {
			if (!g_ipc[i].used || !g_ipc[i].is_import ||
			    !g_ipc[i].closed || g_ipc[i].seq != pass_seq)
				continue;
			if (g_ipc[i].ctx)
				r_cuCtxSetCurrent(g_ipc[i].ctx);
			CUipcMemHandle nb;
			if (ipc_fetch_blob(&g_ipc[i], &nb, 60000) != 0)
				goto fail;
			/* Release the held range immediately before THIS reopen
			 * and no earlier: every other target must stay reserved,
			 * or this import can squat on a later import's address.
			 * (Freeing them all up front was tried while chasing the
			 * TP=8 low-arena problem; it did not help that and broke
			 * the TP=4 walk.) */
			if (g_ipc[i].resv) {
				CUresult frc = r_cuMemAddressFree(
				    g_ipc[i].resv, g_ipc[i].resv_size);
				if (frc != CUDA_SUCCESS)
					mclog("RESUME: freeing held range for "
					      "seq=%d rc=%d",
					      g_ipc[i].seq, frc);
				g_ipc[i].resv = 0;
			}
			CUdeviceptr np = 0;
			CUresult rc =
			    r_cuIpcOpenMemHandle(&np, nb, g_ipc[i].flags);
			if (rc != CUDA_SUCCESS) {
				mclog("RESUME: cuIpcOpenMemHandle(seq=%d) rc=%d",
				      g_ipc[i].seq, rc);
				goto fail;
			}

			/* Walk it back if it landed low.
			 *
			 * cuIpcOpenMemHandle takes no address hint; it takes the
			 * lowest free hole in its arena. The import's own range is
			 * protected (we held a reservation over it across the
			 * checkpoint), but the arena has OTHER free holes below it
			 * -- between the import clusters, and where /sleep freed
			 * the weights and KV cache -- and the driver prefers those.
			 *
			 * A single fence over [landed, target) cannot work: that
			 * span crosses the other imports' held reservations, and a
			 * reservation cannot overlap an existing one. So plug the
			 * holes one at a time instead. Wherever the open lands IS,
			 * by construction, the lowest free hole: close it, reserve
			 * exactly there, and open again. Each iteration eliminates
			 * one hole, so this terminates, and once nothing below the
			 * target is free the open lands exactly on it. Plugs stay
			 * until every import is placed (they are what stops import
			 * N+1 falling into the same holes), then all are freed. */
			int hops = 0;
			while (np < g_ipc[i].ptr) {
				if (nplugs >= IPC_MAX_PLUGS) {
					mclog("RESUME: seq=%d still 0x%llx short "
					      "of target after %d hole plugs; "
					      "giving up",
					      g_ipc[i].seq,
					      (unsigned long long)(g_ipc[i].ptr - np),
					      nplugs);
					break;
				}
				if (r_cuIpcCloseMemHandle(np) != CUDA_SUCCESS) {
					mclog("RESUME: close during re-place "
					      "(seq=%d) failed", g_ipc[i].seq);
					goto fail;
				}
				/* Plug the hole it fell into. Sized like the
				 * mapping (that is how much of the hole the open
				 * proved free), capped so it cannot spill past the
				 * target range. */
				const size_t gran = 2u << 20;
				size_t psz = (g_ipc[i].range_size + gran - 1) &
				             ~(gran - 1);
				if (psz > (size_t)(g_ipc[i].ptr - np))
					psz = (size_t)(g_ipc[i].ptr - np);
				if (psz < gran)
					psz = gran;
				CUdeviceptr plug = 0;
				CUresult prc = r_cuMemAddressReserve(&plug, psz, 0,
				                                     np, 0);
				if (prc != CUDA_SUCCESS) {
					mclog("RESUME: plugging hole at 0x%llx+0x%zx "
					      "failed rc=%d (seq=%d)",
					      (unsigned long long)np, psz, prc,
					      g_ipc[i].seq);
					goto fail;
				}
				if (plug != np) {
					/* Hint not honored: the hole is smaller than
					 * psz. Take what we got anyway (it plugs SOME
					 * hole) and keep walking. */
					mcvlog("RESUME: plug mislanded 0x%llx -> "
					       "0x%llx (hole smaller than 0x%zx)",
					       (unsigned long long)np,
					       (unsigned long long)plug, psz);
				}
				g_plugs[nplugs].base = plug;
				g_plugs[nplugs].size = psz;
				nplugs++;
				hops++;
				np = 0;
				rc = r_cuIpcOpenMemHandle(&np, nb, g_ipc[i].flags);
				if (rc != CUDA_SUCCESS) {
					mclog("RESUME: re-place open seq=%d rc=%d "
					      "after %d plugs",
					      g_ipc[i].seq, rc, hops);
					goto fail;
				}
			}
			if (hops && np == g_ipc[i].ptr) {
				ipc_replaced++;
				mcvlog("RESUME: seq=%d walked back to 0x%llx in %d "
				       "hole plugs",
				       g_ipc[i].seq, (unsigned long long)np, hops);
			}

			if (np != g_ipc[i].ptr) {
				/* Keep going rather than stopping at the first
				 * mismatch: how MANY move, and by how much, is
				 * what distinguishes a placement-ordering bug
				 * from the approach being unworkable. The resume
				 * still fails below -- the application holds the
				 * old pointers, so a moved import is silent
				 * corruption, not a warning. */
				ipc_moved++;
				mclog("RESUME: IPC import seq=%d MOVED: 0x%llx -> "
				      "0x%llx (delta %+lld MiB)",
				      g_ipc[i].seq,
				      (unsigned long long)g_ipc[i].ptr,
				      (unsigned long long)np,
				      ((long long)np - (long long)g_ipc[i].ptr) /
				          (1024 * 1024));
				continue;
			}
			/* Live again at the right address: a retried
			 * suspend re-closes it, and the gate-release
			 * check no longer counts it as outstanding. */
			g_ipc[i].closed = 0;
			ipc_reopened++;
			mcvlog("RESUME: reopened IPC import seq=%d va=0x%llx",
			       g_ipc[i].seq, (unsigned long long)np);
		}
	}
	/* Every import is placed (or we are about to fail); the hole plugs
	 * have served their purpose. */
	for (int p = 0; p < nplugs; p++)
		r_cuMemAddressFree(g_plugs[p].base, g_plugs[p].size);

	if (ipc_reopened || p1_ipc || ipc_moved)
		mclog("RESUME: legacy IPC done (%d reopened at identical VAs, "
		      "of which %d walked back via %d hole plugs; %d MOVED, "
		      "%d served)",
		      ipc_reopened, ipc_replaced, nplugs, ipc_moved, p1_ipc);
	if (ipc_moved) {
		mclog("RESUME: FATAL: %d of %d legacy IPC imports did not return "
		      "to their original address. cuIpcOpenMemHandle takes no "
		      "address hint, so placement depends on the allocation "
		      "state at reopen time matching the original.",
		      ipc_moved, ipc_moved + ipc_reopened);
		return -1;
	}
	return 0;

fail:
	/* The plugs are only ever temporary; a failure return must not leak
	 * them into the (still gated, possibly retried) address space. */
	for (int p = 0; p < nplugs; p++)
		r_cuMemAddressFree(g_plugs[p].base, g_plugs[p].size);
	return -1;
}

/* ------------------------------------------------------------------ */
/* Control thread: poll $MCSHIM_DIR for suspend/resume markers.       */
/* ------------------------------------------------------------------ */

static void marker(const char *name, char *out, size_t n) {
	snprintf(out, n, "%s/%s", g_dir, name);
}

static int marker_exists(const char *name) {
	char p[600];
	marker(name, p, sizeof(p));
	return access(p, F_OK) == 0;
}

static void marker_rm(const char *name) {
	char p[600];
	marker(name, p, sizeof(p));
	unlink(p);
}

static void marker_write(const char *name, const char *body) {
	char p[600];
	marker(name, p, sizeof(p));
	FILE *f = fopen(p, "w");
	if (f) {
		fputs(body, f);
		fputc('\n', f);
		fclose(f);
	}
}

/* ------------------------------------------------------------------ */
/* Lookup interposition: dlsym + cuGetProcAddress.                    */
/*                                                                    */
/* torch/NCCL/ctypes resolve driver entry points with                 */
/* dlsym(dlopen("libcuda.so.1"), name) or cuGetProcAddress, which     */
/* bypasses classic symbol interposition. Redirect lookups of the     */
/* tracked entry points to the shim's wrappers.                       */
/* ------------------------------------------------------------------ */

/* All wrappers above are already defined; only the cuGetProcAddress pair is
 * defined below the table and needs declarations. */

/* ------------------------------------------------------------------ */
/* Suspend gate.                                                      */
/*                                                                    */
/* Between suspend and resume the multicast groups and peer imports    */
/* are released and their VAs are unmapped. Any application thread     */
/* that reaches the GPU in that window touches an unmapped VA and      */
/* faults its context (CUDA_ERROR_ILLEGAL_ADDRESS, 700) -- and because */
/* the ranks share a multicast group, one rank doing so can fault      */
/* every rank. The acceptance harness avoids this by pausing the       */
/* workload, but a real workload (vLLM, SGLang) has no such pause:     */
/* `cuda-checkpoint --toggle` restores AND unlocks each process, so the */
/* application becomes runnable while the rebuild is still in flight.  */
/*                                                                    */
/* So the shim makes the guarantee itself: while suspended, the        */
/* interposed entry points through which GPU work is submitted block   */
/* until the rebuild completes. The application observes a pause, not  */
/* a fault, and needs no cooperation. The shim's own suspend/resume    */
/* work calls the real entry points (r_*) directly and is never gated. */
/* ------------------------------------------------------------------ */

static pthread_mutex_t g_gate_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_gate_cv = PTHREAD_COND_INITIALIZER;

/* Arm before the teardown begins, so no app thread can slip between the
 * decision to suspend and the first unmap. g_suspended is read locklessly on
 * the gate_wait fast path, so all transitions are atomic stores. */
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
	if (!__atomic_load_n(&g_suspended, __ATOMIC_SEQ_CST))
		return;
	static __thread int logged;
	pthread_mutex_lock(&g_gate_lock);
	if (g_suspended && !logged) {
		logged = 1;
		mclog("GATE: app thread blocked until resume");
	}
	while (g_suspended)
		pthread_cond_wait(&g_gate_cv, &g_gate_lock);
	pthread_mutex_unlock(&g_gate_lock);
}

/* Entry points through which the application submits GPU work or waits on
 * it. Each one blocks while suspended, then forwards unchanged. The tracked
 * mutators above (cuMemCreate, the cuMulticast and cuIpc families, exports,
 * imports, map/unmap, release) call gate_wait() directly for the same
 * reason plus one more: a thread reaching them in the unlocked teardown
 * window would create or free shared state AFTER the strict blocker gate
 * verified there was none, yielding a snapshot that hangs the checkpoint or
 * only fails at restore. The suspend/resume machinery itself always calls
 * the r_ reals, never these wrappers, so gating them cannot deadlock the
 * transition. */
#define GATED(name, proto, args)                                               \
	static CUresult (*r_##name) proto;                                     \
	CUresult name proto;                                                   \
	CUresult name proto {                                                  \
		REAL(r_##name, #name);                                         \
		if (!r_##name)                                                 \
			return 1 /* CUDA_ERROR_INVALID_VALUE */;               \
		gate_wait();                                                   \
		return r_##name args;                                          \
	}

typedef void *CUstream_t;
typedef void *CUfunction_t;
typedef void *CUgraphExec_t;
typedef void *CUhostFn_t;

GATED(cuLaunchKernel,
      (CUfunction_t f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
       unsigned by, unsigned bz, unsigned shmem, CUstream_t st, void **kp,
       void **extra),
      (f, gx, gy, gz, bx, by, bz, shmem, st, kp, extra))
GATED(cuLaunchKernelEx,
      (const void *cfg, CUfunction_t f, void **kp, void **extra),
      (cfg, f, kp, extra))
GATED(cuLaunchCooperativeKernel,
      (CUfunction_t f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
       unsigned by, unsigned bz, unsigned shmem, CUstream_t st, void **kp),
      (f, gx, gy, gz, bx, by, bz, shmem, st, kp))
GATED(cuGraphLaunch, (CUgraphExec_t g, CUstream_t st), (g, st))
GATED(cuMemsetD32_v2, (CUdeviceptr d, unsigned ui, size_t n), (d, ui, n))
GATED(cuMemsetD32Async,
      (CUdeviceptr d, unsigned ui, size_t n, CUstream_t st), (d, ui, n, st))
GATED(cuMemsetD8_v2, (CUdeviceptr d, unsigned char uc, size_t n), (d, uc, n))
GATED(cuMemsetD8Async,
      (CUdeviceptr d, unsigned char uc, size_t n, CUstream_t st),
      (d, uc, n, st))
GATED(cuMemcpyAsync,
      (CUdeviceptr dst, CUdeviceptr src, size_t n, CUstream_t st),
      (dst, src, n, st))
GATED(cuMemcpyHtoD_v2, (CUdeviceptr dst, const void *src, size_t n),
      (dst, src, n))
GATED(cuMemcpyDtoH_v2, (void *dst, CUdeviceptr src, size_t n), (dst, src, n))
GATED(cuMemcpyHtoDAsync_v2,
      (CUdeviceptr dst, const void *src, size_t n, CUstream_t st),
      (dst, src, n, st))
GATED(cuMemcpyDtoHAsync_v2,
      (void *dst, CUdeviceptr src, size_t n, CUstream_t st), (dst, src, n, st))
GATED(cuMemcpyDtoD_v2, (CUdeviceptr dst, CUdeviceptr src, size_t n),
      (dst, src, n))
GATED(cuMemcpyDtoDAsync_v2,
      (CUdeviceptr dst, CUdeviceptr src, size_t n, CUstream_t st),
      (dst, src, n, st))
GATED(cuMemcpy2D_v2, (const void *pCopy), (pCopy))
GATED(cuMemcpy2DAsync_v2, (const void *pCopy, CUstream_t st), (pCopy, st))
GATED(cuMemcpy3D_v2, (const void *pCopy), (pCopy))
GATED(cuMemcpy3DAsync_v2, (const void *pCopy, CUstream_t st), (pCopy, st))
GATED(cuMemsetD16_v2, (CUdeviceptr d, unsigned short us, size_t n),
      (d, us, n))
GATED(cuMemsetD16Async,
      (CUdeviceptr d, unsigned short us, size_t n, CUstream_t st),
      (d, us, n, st))
GATED(cuLaunchHostFunc, (CUstream_t st, CUhostFn_t fn, void *userData),
      (st, fn, userData))
GATED(cuStreamSynchronize, (CUstream_t st), (st))
/* The allocation family is gated for a different reason than the
 * submission entries: the legacy-IPC reopen walk (resume_reopen_ipc)
 * requires that NOTHING allocates device VA while it runs, or the walked
 * imports land one slot off their original addresses. */
GATED(cuMemAlloc_v2, (CUdeviceptr *dptr, size_t n), (dptr, n))
GATED(cuMemFree_v2, (CUdeviceptr dptr), (dptr))
GATED(cuMemAllocAsync, (CUdeviceptr *dptr, size_t n, CUstream_t st),
      (dptr, n, st))
GATED(cuMemFreeAsync, (CUdeviceptr dptr, CUstream_t st), (dptr, st))
GATED(cuMemAllocPitch_v2,
      (CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height,
       unsigned int ElementSizeBytes),
      (dptr, pPitch, WidthInBytes, Height, ElementSizeBytes))

CUresult cuGetProcAddress(const char *, void **, int, unsigned long long);
CUresult cuGetProcAddress_v2(const char *, void **, int, unsigned long long,
                             int *);
/* CUDA runtime resolvers (cudaError_t and the query-result enum are plain
 * ints in this toolkit-free build; 0 is success for both). */
int cudaGetDriverEntryPoint(const char *, void **, unsigned long long, int *);
int cudaGetDriverEntryPoint_ptsz(const char *, void **, unsigned long long,
                                 int *);
int cudaGetDriverEntryPointByVersion(const char *, void **, unsigned int,
                                     unsigned long long, int *);
int cudaGetDriverEntryPointByVersion_ptsz(const char *, void **, unsigned int,
                                          unsigned long long, int *);

typedef struct {
	const char *name;
	void *fn;
	/* The wrapper forwards to the LEGACY-stream real, so redirecting a
	 * per-thread-default-stream (PTDS) cuGetProcAddress lookup to it
	 * would silently change the app's NULL-stream semantics; such
	 * lookups are declined (see gpa_redirect). State-tracking entries
	 * have no stream semantics and are always redirected. */
	int stream_sem;
} WrapEntry;

static const WrapEntry *wrap_table(void) {
	static WrapEntry t[] = {
	    {"cuMemCreate", (void *)cuMemCreate, 0},
	    {"cuMemRelease", (void *)cuMemRelease, 0},
	    {"cuMemMap", (void *)cuMemMap, 0},
	    {"cuMemUnmap", (void *)cuMemUnmap, 0},
	    {"cuMemSetAccess", (void *)cuMemSetAccess, 0},
	    {"cuMulticastCreate", (void *)cuMulticastCreate, 0},
	    {"cuMulticastAddDevice", (void *)cuMulticastAddDevice, 0},
	    {"cuMulticastBindMem", (void *)cuMulticastBindMem, 0},
	    {"cuMulticastBindAddr", (void *)cuMulticastBindAddr, 0},
	    {"cuMulticastUnbind", (void *)cuMulticastUnbind, 0},
	    {"cuInit", (void *)cuInit, 0},
	    {"cuMemExportToShareableHandle",
	     (void *)cuMemExportToShareableHandle, 0},
	    {"cuMemImportFromShareableHandle",
	     (void *)cuMemImportFromShareableHandle, 0},
	    {"cuDeviceGetAttribute", (void *)cuDeviceGetAttribute, 0},

	    {"cuIpcGetMemHandle", (void *)cuIpcGetMemHandle, 0},
	    {"cuIpcOpenMemHandle", (void *)cuIpcOpenMemHandle_v2, 0},
	    {"cuIpcOpenMemHandle_v2", (void *)cuIpcOpenMemHandle_v2, 0},
	    {"cuIpcCloseMemHandle", (void *)cuIpcCloseMemHandle, 0},
	    {"cuLaunchKernel", (void *)cuLaunchKernel, 1},
	    {"cuLaunchKernelEx", (void *)cuLaunchKernelEx, 1},
	    {"cuLaunchCooperativeKernel", (void *)cuLaunchCooperativeKernel, 1},
	    {"cuGraphLaunch", (void *)cuGraphLaunch, 1},
	    {"cuLaunchHostFunc", (void *)cuLaunchHostFunc, 1},
	    {"cuMemsetD32", (void *)cuMemsetD32_v2, 1},
	    {"cuMemsetD32_v2", (void *)cuMemsetD32_v2, 1},
	    {"cuMemsetD32Async", (void *)cuMemsetD32Async, 1},
	    {"cuMemsetD16", (void *)cuMemsetD16_v2, 1},
	    {"cuMemsetD16_v2", (void *)cuMemsetD16_v2, 1},
	    {"cuMemsetD16Async", (void *)cuMemsetD16Async, 1},
	    {"cuMemsetD8", (void *)cuMemsetD8_v2, 1},
	    {"cuMemsetD8_v2", (void *)cuMemsetD8_v2, 1},
	    {"cuMemsetD8Async", (void *)cuMemsetD8Async, 1},
	    {"cuMemcpyAsync", (void *)cuMemcpyAsync, 1},
	    {"cuMemcpyHtoD", (void *)cuMemcpyHtoD_v2, 1},
	    {"cuMemcpyHtoD_v2", (void *)cuMemcpyHtoD_v2, 1},
	    {"cuMemcpyDtoH", (void *)cuMemcpyDtoH_v2, 1},
	    {"cuMemcpyDtoH_v2", (void *)cuMemcpyDtoH_v2, 1},
	    {"cuMemcpyDtoD", (void *)cuMemcpyDtoD_v2, 1},
	    {"cuMemcpyDtoD_v2", (void *)cuMemcpyDtoD_v2, 1},
	    {"cuMemcpyHtoDAsync", (void *)cuMemcpyHtoDAsync_v2, 1},
	    {"cuMemcpyHtoDAsync_v2", (void *)cuMemcpyHtoDAsync_v2, 1},
	    {"cuMemcpyDtoHAsync", (void *)cuMemcpyDtoHAsync_v2, 1},
	    {"cuMemcpyDtoHAsync_v2", (void *)cuMemcpyDtoHAsync_v2, 1},
	    {"cuMemcpyDtoDAsync", (void *)cuMemcpyDtoDAsync_v2, 1},
	    {"cuMemcpyDtoDAsync_v2", (void *)cuMemcpyDtoDAsync_v2, 1},
	    {"cuMemcpy2D", (void *)cuMemcpy2D_v2, 1},
	    {"cuMemcpy2D_v2", (void *)cuMemcpy2D_v2, 1},
	    {"cuMemcpy2DAsync", (void *)cuMemcpy2DAsync_v2, 1},
	    {"cuMemcpy2DAsync_v2", (void *)cuMemcpy2DAsync_v2, 1},
	    {"cuMemcpy3D", (void *)cuMemcpy3D_v2, 1},
	    {"cuMemcpy3D_v2", (void *)cuMemcpy3D_v2, 1},
	    {"cuMemcpy3DAsync", (void *)cuMemcpy3DAsync_v2, 1},
	    {"cuMemcpy3DAsync_v2", (void *)cuMemcpy3DAsync_v2, 1},
	    {"cuStreamSynchronize", (void *)cuStreamSynchronize, 1},
	    /* Gated (the legacy-IPC reopen walk requires that nothing
	     * allocates device VA while it runs) but without PTDS/PTSZ
	     * variants, so always redirected. */
	    {"cuMemAlloc", (void *)cuMemAlloc_v2, 0},
	    {"cuMemAlloc_v2", (void *)cuMemAlloc_v2, 0},
	    {"cuMemFree", (void *)cuMemFree_v2, 0},
	    {"cuMemFree_v2", (void *)cuMemFree_v2, 0},
	    {"cuMemAllocPitch", (void *)cuMemAllocPitch_v2, 0},
	    {"cuMemAllocPitch_v2", (void *)cuMemAllocPitch_v2, 0},
	    /* Stream-ordered allocators DO have _ptsz variants. */
	    {"cuMemAllocAsync", (void *)cuMemAllocAsync, 1},
	    {"cuMemFreeAsync", (void *)cuMemFreeAsync, 1},
	    {"cuGetProcAddress", (void *)cuGetProcAddress, 0},
	    {"cuGetProcAddress_v2", (void *)cuGetProcAddress_v2, 0},
	    /* NVML fabric-info smoothing (see nvml_fabric_info_smooth). */
	    {"nvmlDeviceGetGpuFabricInfoV",
	     (void *)nvmlDeviceGetGpuFabricInfoV, 0},
	    /* CUDA runtime resolvers: covers apps that dlsym them from a
	     * dlopen'd libcudart rather than linking it. */
	    {"cudaGetDriverEntryPoint", (void *)cudaGetDriverEntryPoint, 0},
	    {"cudaGetDriverEntryPoint_ptsz",
	     (void *)cudaGetDriverEntryPoint_ptsz, 0},
	    {"cudaGetDriverEntryPointByVersion",
	     (void *)cudaGetDriverEntryPointByVersion, 0},
	    {"cudaGetDriverEntryPointByVersion_ptsz",
	     (void *)cudaGetDriverEntryPointByVersion_ptsz, 0},
	    {NULL, NULL, 0},
	};
	return t;
}

static const WrapEntry *wrap_entry(const char *name) {
	if (!name)
		return NULL;
	for (const WrapEntry *e = wrap_table(); e->name; e++)
		if (strcmp(e->name, name) == 0)
			return e;
	return NULL;
}

static void *wrapper_for(const char *name) {
	const WrapEntry *e = wrap_entry(name);
	return e ? e->fn : NULL;
}

/* Interposed dlsym: hand out shim wrappers for tracked driver symbols.
 * Note: delegating through a dlvsym-resolved dlsym re-anchors RTLD_NEXT at
 * mcshim's link map, which can confuse interposers stacked after it
 * (inherent to the technique). */
void *dlsym(void *handle, const char *symbol) {
	init_real_dlsym();
	if (!real_dlsym)
		return NULL;
	void *w = wrapper_for(symbol);
	if (w) {
		/* Only redirect if the real library actually has the symbol
		 * (so feature probes against old drivers still behave). */
		void *r = real_dlsym(handle, symbol);
		if (r)
			return w;
		return r;
	}
	return real_dlsym(handle, symbol);
}

/* Interposed cuGetProcAddress (CUDA 11.3+ runtime/apps resolve driver
 * entry points through this): same redirection -- but the resolver itself
 * needs ABI-aware handling. A query for "cuGetProcAddress" at
 * cudaVersion >= 12000 returns the 5-arg v2 ABI, so it must get our v2
 * wrapper: handing out the 4-arg v1 wrapper leaves the caller's
 * symbolStatus unwritten (uninitialized), which made stock NCCL mark every
 * symbol missing and call NULL pfns. */

#define CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM (1ULL << 1)

static void *gpa_redirect(const char *symbol, int cudaVersion,
                          unsigned long long flags) {
	if (strcmp(symbol, "cuGetProcAddress") == 0)
		return cudaVersion >= 12000 ? (void *)cuGetProcAddress_v2
		                            : (void *)cuGetProcAddress;
	const WrapEntry *e = wrap_entry(symbol);
	if (!e)
		return NULL;
	/* A PTDS lookup of a stream-semantics-sensitive entry gets the
	 * driver's own _ptds/_ptsz pfn unmodified: our wrapper forwards to
	 * the legacy-stream real and would silently change the app's
	 * NULL-stream semantics. (Such an app is then not gated on these
	 * entries -- a known limitation, logged so it is diagnosable.) */
	if (e->stream_sem &&
	    (flags & CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)) {
		static int logged;
		if (!logged) {
			logged = 1;
			mcvlog("cuGetProcAddress(%s, PTDS): declining redirect "
			       "for per-thread-default-stream lookups", symbol);
		}
		return NULL;
	}
	return e->fn;
}

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
                          unsigned long long flags) {
	static CUresult (*real)(const char *, void **, int, unsigned long long);
	REAL(real, "cuGetProcAddress");
	if (!real)
		return 3; /* CUDA_ERROR_NOT_INITIALIZED */
	CUresult rc = real(symbol, pfn, cudaVersion, flags);
	void *w;
	if (rc == CUDA_SUCCESS && pfn && *pfn &&
	    (w = gpa_redirect(symbol, cudaVersion, flags))) {
		*pfn = w;
	}
	return rc;
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                             unsigned long long flags, int *symbolStatus) {
	static CUresult (*real)(const char *, void **, int, unsigned long long,
	                        int *);
	REAL(real, "cuGetProcAddress_v2");
	if (!real)
		return 3;
	CUresult rc = real(symbol, pfn, cudaVersion, flags, symbolStatus);
	void *w;
	if (rc == CUDA_SUCCESS && pfn && *pfn &&
	    (w = gpa_redirect(symbol, cudaVersion, flags))) {
		*pfn = w;
	}
	return rc;
}

/* Interposed CUDA *runtime* driver-entry-point resolvers.
 *
 * torch >= 2.11 resolves its entire c10 DriverAPI table -- cuMemCreate,
 * cuMulticastCreate, cuMemExportToShareableHandle, ... -- through
 * cudaGetDriverEntryPointByVersion (a public libcudart export). libcudart
 * reaches libcuda through an internal dlvsym/cuGetExportTable bootstrap that
 * neither the dlsym hook nor the cuGetProcAddress wrappers ever see, so
 * every symbol resolved this way was invisible to the shim. The
 * torch->libcudart hop itself is an ordinary cross-DSO binding, which
 * LD_PRELOAD wins: interpose the resolver, let the real runtime do the
 * lookup, then post-process the result exactly like cuGetProcAddress.
 *
 * The runtime's cudaEnable{Default,LegacyStream,PerThreadDefaultStream}
 * flag values (0, 1, 2) coincide numerically with the driver's
 * CU_GET_PROC_ADDRESS_* bits, so gpa_redirect's PTDS decline logic applies
 * unchanged -- except in the _ptsz variants, where cudaEnableDefault means
 * PTDS (the caller was compiled with per-thread default streams), so the
 * effective flags for the redirect decision are mapped first.
 *
 * The real functions live in libcudart, not libcuda: resolve via RTLD_NEXT
 * (app linked cudart into the global scope), then by soname against an
 * already-loaded copy (torch dlopens its bundled cudart RTLD_LOCAL). Never
 * force-load: if no cudart is loaded, no one can be calling us. If two
 * different-major cudarts are loaded, the newest wins the forward; their
 * resolvers differ only in the default-version mapping of the unversioned
 * variants, and no supported image ships two. */

#define CUDA_ERROR_RT_SYMBOL_NOT_FOUND 500 /* cudaErrorSymbolNotFound */

static void *libcudart_sym(const char *name) {
	init_real_dlsym();
	if (!real_dlsym)
		return NULL;
	void *s = real_dlsym(RTLD_NEXT, name);
	if (s)
		return s;
	static void *h;
	if (!h) {
		static const char *const sonames[] = {
		    "libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0",
		    "libcudart.so", NULL};
		for (int i = 0; !h && sonames[i]; i++)
			h = dlopen(sonames[i], RTLD_NOW | RTLD_NOLOAD);
	}
	return h ? real_dlsym(h, name) : NULL;
}

#define RTREAL(var, name)                                                      \
	do {                                                                   \
		if (!(var))                                                    \
			*(void **)(&(var)) = libcudart_sym(name);              \
	} while (0)

/* cudaEnableDefault in a _ptsz resolver means PTDS. */
static unsigned long long rt_eff_flags(unsigned long long flags, int ptsz) {
	if (ptsz && flags == 0)
		return CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM;
	return flags;
}

/* The unversioned resolver returns entry points matching the runtime's own
 * ABI generation; every cudart >= 12.0 (the supported floor) maps to the
 * v2-era driver ABI, hence version 12000 for the redirect decision. */
static void rt_gpa_post(const char *symbol, void **pfn, int cudaVersion,
                        unsigned long long flags) {
	void *w;
	if (pfn && *pfn && (w = gpa_redirect(symbol, cudaVersion, flags)))
		*pfn = w;
}

int cudaGetDriverEntryPoint(const char *symbol, void **pfn,
                            unsigned long long flags, int *driverStatus) {
	static int (*real)(const char *, void **, unsigned long long, int *);
	RTREAL(real, "cudaGetDriverEntryPoint");
	if (!real)
		return CUDA_ERROR_RT_SYMBOL_NOT_FOUND;
	int rc = real(symbol, pfn, flags, driverStatus);
	if (rc == 0)
		rt_gpa_post(symbol, pfn, 12000, rt_eff_flags(flags, 0));
	return rc;
}

int cudaGetDriverEntryPoint_ptsz(const char *symbol, void **pfn,
                                 unsigned long long flags,
                                 int *driverStatus) {
	static int (*real)(const char *, void **, unsigned long long, int *);
	RTREAL(real, "cudaGetDriverEntryPoint_ptsz");
	if (!real)
		return CUDA_ERROR_RT_SYMBOL_NOT_FOUND;
	int rc = real(symbol, pfn, flags, driverStatus);
	if (rc == 0)
		rt_gpa_post(symbol, pfn, 12000, rt_eff_flags(flags, 1));
	return rc;
}

int cudaGetDriverEntryPointByVersion(const char *symbol, void **pfn,
                                     unsigned int cudaVersion,
                                     unsigned long long flags,
                                     int *driverStatus) {
	static int (*real)(const char *, void **, unsigned int,
	                   unsigned long long, int *);
	RTREAL(real, "cudaGetDriverEntryPointByVersion");
	if (!real)
		return CUDA_ERROR_RT_SYMBOL_NOT_FOUND;
	int rc = real(symbol, pfn, cudaVersion, flags, driverStatus);
	if (rc == 0)
		rt_gpa_post(symbol, pfn, (int)cudaVersion, rt_eff_flags(flags, 0));
	return rc;
}

int cudaGetDriverEntryPointByVersion_ptsz(const char *symbol, void **pfn,
                                          unsigned int cudaVersion,
                                          unsigned long long flags,
                                          int *driverStatus) {
	static int (*real)(const char *, void **, unsigned int,
	                   unsigned long long, int *);
	RTREAL(real, "cudaGetDriverEntryPointByVersion_ptsz");
	if (!real)
		return CUDA_ERROR_RT_SYMBOL_NOT_FOUND;
	int rc = real(symbol, pfn, cudaVersion, flags, driverStatus);
	if (rc == 0)
		rt_gpa_post(symbol, pfn, (int)cudaVersion, rt_eff_flags(flags, 1));
	return rc;
}

/* Existence-based, edge-triggered protocol (race-free for N ranks sharing
 * the control dir):
 *   "suspend" appears    -> suspend once, ack "suspended.<pid>"
 *   "suspend" disappears -> resume once,  ack "resumed.<pid>"
 * Failures ack "error.<pid>" and wait for the next edge (no retry storm).
 * The marker file is part of the checkpoint image (container /tmp), so after
 * a restore the shim stays suspended until the orchestrator removes it. */
static void *control_thread(void *arg) {
	(void)arg;
	char ack_s[64], ack_r[64], ack_e[64];
	snprintf(ack_s, sizeof(ack_s), "suspended.%d", (int)getpid());
	snprintf(ack_r, sizeof(ack_r), "resumed.%d", (int)getpid());
	snprintf(ack_e, sizeof(ack_e), "error.%d", (int)getpid());
	char ack_g[64], ack_ug[64];
	snprintf(ack_g, sizeof(ack_g), "gated.%d", (int)getpid());
	snprintf(ack_ug, sizeof(ack_ug), "ungated.%d", (int)getpid());
	/* Announce that this process is interposer-managed. Only processes that
	 * resolved a tracked CUDA entry point get a control thread, so only they
	 * can ever acknowledge a transition. The orchestrator selects CUDA
	 * processes by looking for open NVIDIA device FDs, which is a broader set
	 * (a vLLM API server or engine-core process holds them without ever
	 * touching multicast). Without this file the orchestrator would wait
	 * forever for acknowledgements from processes that have no interposer. */
	char present[64];
	snprintf(present, sizeof(present), "present.%d", (int)getpid());
	/* Clear any acks a dead predecessor left behind first: pids are
	 * reused inside a container, so a stale suspended.<pid> (etc.) could
	 * otherwise satisfy the sentry's ack wait with a dead process's
	 * answer. */
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
		/* "gate" bars the application from the GPU. Handled first and
		 * separately from "suspend" because it issues no CUDA calls, so
		 * the orchestrator can arm it while this process is locked by
		 * cuda-checkpoint -- which is what lets the lock, rather than
		 * the gate, be the thing that quiesces coupled ranks. */
		int wgate = marker_exists("gate");
		if (wgate != prev_gate) {
			prev_gate = wgate;
			if (wgate) {
				gate_arm();
				marker_rm(ack_ug);
				marker_write(ack_g, "ok");
			} else {
				/* Refuse to release the gate while teardown state
				 * is outstanding (a resume failed partway):
				 * ungated, the app would run over unmapped GPU
				 * state and fault every coupled rank. The sentry's
				 * unwind removing the marker does not override
				 * this; only a successful resume clears the flags.
				 * (prev_gate was updated above, so a re-created
				 * marker still re-arms cleanly.) No ungated ack is
				 * written: the process is still gated. */
				int torn = 0;
				pthread_mutex_lock(&g_lock);
				for (int i = 0; i < MAXN; i++) {
					if (g_alloc[i].kind != KIND_FREE &&
					    g_alloc[i].torn_down)
						torn++;
					if (g_map[i].used && g_map[i].suspended)
						torn++;
					if (g_ipc[i].used && g_ipc[i].is_import &&
					    g_ipc[i].closed)
						torn++;
				}
				pthread_mutex_unlock(&g_lock);
				if (torn) {
					mclog("FATAL: refusing to release the "
					      "gate: %d object(s) still torn down "
					      "after a failed resume; the "
					      "application would run over unmapped "
					      "GPU state", torn);
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
			/* Drop any stale error ack before starting this
			 * transition: the sentry fast-fails on error.<pid>, so
			 * only errors from the transition in flight may be
			 * visible. */
			marker_rm(ack_e);
			pthread_mutex_lock(&g_lock);
			resolve_reals();
			int rc;
			if (want) {
				/* Arm first: an app thread must not reach the
				 * GPU between here and the last unmap. */
				gate_arm();
				rc = do_suspend();
				if (rc != 0)
					/* Deliberate: a failed checkpoint must
					 * leave the application running, not
					 * blocked. Whatever partial teardown
					 * happened is rebuilt when the sentry's
					 * unwind removes the suspend marker (the
					 * resume edge below); the window until
					 * then is the price of not deadlocking
					 * the app if the orchestrator goes away. */
					gate_disarm();
			} else {
				/* On a FAILED resume the gate stays armed: the
				 * GPU state is partially rebuilt, and blocking
				 * the app beats letting it compute on it (better
				 * blocked than corrupt). Removing the "gate"
				 * marker does not override this -- the gate edge
				 * above refuses to release while teardown state
				 * is outstanding. Teardown and rebuild are
				 * re-enterable (per-entry flags, cleared as each
				 * entry is rebuilt), so a retried suspend/resume
				 * edge converges once the underlying cause
				 * clears. */
				rc = do_resume();
				/* The multicast-proxy helper (if any) is only
				 * needed while the rebuild runs; the groups
				 * persist through this process's imports. */
				helper_stop();
				if (rc == 0)
					gate_disarm();
			}
			pthread_mutex_unlock(&g_lock);
			if (rc != 0) {
				marker_write(ack_e, want ? "suspend failed"
				                         : "resume failed");
			} else if (want) {
				marker_rm(ack_r);
				marker_write(ack_s, "ok");
			} else {
				marker_rm(ack_s);
				marker_write(ack_r, "ok");
			}
		}
		/* Poll fast. The orchestrator arms the gate on every rank by
		 * creating one marker, so the spread in when the ranks observe
		 * it bounds how likely a peer is to enter a collective after
		 * another rank has already gated -- a straddled collective that
		 * neither the gate nor the lock can then quiesce. At 100ms the
		 * spread was wide enough that a workload with no idle gap
		 * exhausted the orchestrator's lock retries; a few ms of skew
		 * makes it rare. Two stat()s per interval is negligible. */
		struct timespec ts = {0, 5 * 1000 * 1000}; /* 5ms */
		nanosleep(&ts, NULL);
	}
	return NULL;
}

static int g_disabled;

/* Start the control thread lazily, only in processes that actually resolve
 * a tracked CUDA entry point. Every process spawned in the container (shell
 * helpers, `runsc exec touch`, cuda-checkpoint itself) may inherit
 * LD_PRELOAD and share $MCSHIM_DIR; if they all polled for markers, a
 * short-lived helper could consume (or falsely ack) a suspend/resume meant
 * for the CUDA workload. */
static void control_thread_start(void) {
	if (g_disabled)
		return;
	pthread_t t;
	int err = pthread_create(&t, NULL, control_thread, NULL);
	if (err == 0)
		pthread_detach(t);
	else
		/* Left silent this surfaces as an inexplicable sentry ack
		 * timeout: no control thread, no present.<pid>, no acks. */
		mclog("FATAL: control thread creation failed: %s -- this "
		      "process will never acknowledge suspend/resume markers",
		      strerror(err));
}

/* Not pthread_once: the fork child handler must be able to reset this so a
 * forked child that later calls cuInit starts its OWN control thread
 * (pthread_once state cannot be reset portably). Guarded by g_lock, which
 * the child handler re-initializes. */
static int g_control_started;

static void ensure_control_thread(void) {
	pthread_mutex_lock(&g_lock);
	int need = !g_control_started;
	g_control_started = 1;
	pthread_mutex_unlock(&g_lock);
	if (need)
		control_thread_start();
}

/* Fork safety. A forked child has fresh CUDA state (a CUDA context is not
 * usable across fork), so tracking inherited from the parent would describe
 * objects the child does not hold; and any shim mutex held by another parent
 * thread at fork time would deadlock the child's first tracked call. Re-init
 * every lock, drop all tracking, and let the child start its own control
 * thread if it ever initializes CUDA itself. */
static void mcshim_atfork_child(void) {
	g_loglock = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
	g_lock = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
	g_gate_lock = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
	g_gate_cv = (pthread_cond_t)PTHREAD_COND_INITIALIZER;
	/* Close inherited rendezvous fds BEFORE dropping the tables that name
	 * them: a held export fd keeps the RM object alive in this child
	 * (CLOEXEC only helps across exec) and would block the parent's NEXT
	 * checkpoint. Do NOT unlink the socket paths: the parent still serves
	 * them. The serve threads' own fds (ServeArgs) are unreachable by
	 * design -- fork does not replicate the owning threads -- and remain
	 * a known residue in a child that never execs. */
	for (int i = 0; i < MAXN; i++) {
		if (g_alloc[i].kind != KIND_FREE) {
			if (g_alloc[i].serve_sock >= 0)
				close(g_alloc[i].serve_sock);
			if (g_alloc[i].serve_fd >= 0)
				close(g_alloc[i].serve_fd);
		}
		if (g_ipc[i].used && g_ipc[i].serve_sock >= 0)
			close(g_ipc[i].serve_sock);
	}
	memset(g_alloc, 0, sizeof(g_alloc));
	memset(g_map, 0, sizeof(g_map));
	memset(g_bind, 0, sizeof(g_bind));
	memset(g_ipc, 0, sizeof(g_ipc));
	g_ipc_seq = 0;
	g_track_overflow = 0;
	__atomic_store_n(&g_suspended, 0, __ATOMIC_SEQ_CST);
	g_control_started = 0;
}

__attribute__((constructor)) static void mcshim_init(void) {
	pthread_atfork(NULL, NULL, mcshim_atfork_child);
	if (getenv("MCSHIM_DISABLE")) {
		/* Silent by design: the disabling parent may be collecting this
		 * process's stderr and parsing it (the sentry disables the shim
		 * in the cuda-checkpoint processes it execs, then reads their
		 * output -- a banner would corrupt e.g. --get-state's "running").
		 */
		g_disabled = 1;
		return;
	}
	const char *d = getenv("MCSHIM_DIR");
	if (d && *d)
		snprintf(g_dir, sizeof(g_dir), "%s", d);
	/* The control dir (which also hosts MCSHIM_LOG) may not exist yet:
	 * the constructor runs before the app's main(). Create it so the
	 * first mclog doesn't silently fall back to stderr. */
	mkdir(g_dir, 0777);
	/* Do NOT clear markers here, and do NOT start the control thread yet
	 * (see ensure_control_thread). The runner owns marker hygiene. */
	mclog("loaded; control dir=%s", g_dir);
}
