/*
 * mcshim.c -- generic libcuda-level multicast suspend/resume interposer.
 *
 * This is the single-process prototype of "Idea D": instead of forking NCCL or
 * hooking the inference engine, an LD_PRELOAD shim interposes the CUDA driver's
 * multicast + virtual-memory-management (VMM) entry points, tracks every
 * multicast group (its handle, participating devices, backing unicast
 * allocations, VA reservations, bindings, and mappings), and performs the same
 * in-process teardown/rebuild that the patched NCCL does -- but transparently,
 * for ANY multicast owner (NCCL NVLS, torch _symmetric_memory, raw cuMulticast).
 *
 * Why in-process: freeing the 0x00fd (NV_MEMORY_MULTICAST_FABRIC) objects from
 * nvproxy makes cuda-checkpoint's SAVE succeed but its RESTORE toggle refuse,
 * because libcuda's userspace bookkeeping still lists the multicast allocation.
 * Running the teardown through libcuda (as this shim does) keeps NCCL structs,
 * libcuda tables, and kernel RM state consistent.
 *
 * Orchestration (matches NCCL_SUSPEND_RESULTS.md order):
 *   (a) the app is quiesced (issues no new CUDA/multicast work),
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
 * The marker lives in the container's /tmp, so it is part of the checkpoint
 * image: after a restore it still exists and the shim stays suspended until
 * the orchestrator removes it. gVisor drives this the same way it drives
 * cuda-checkpoint phases (e.g. via --save-restore-exec-argv around
 * control/state_cuda.go, or `runsc exec`).
 *
 * Multi-process ranks (one process per GPU, the vLLM/SGLang TP topology):
 * rank 0 creates the group and exports it (cuMemExportToShareableHandle);
 * peers import it (cuMemImportFromShareableHandle). The shim records the
 * export/import relationship using the fd's st_dev:st_ino as the rendezvous
 * key -- SCM_RIGHTS passes the same open file description, so exporter and
 * importers observe the same identity. On resume the creator re-exports the
 * recreated group and serves the new fd on an abstract-path-free unix socket
 * ($MCSHIM_DIR/mcgrp-<key>.sock); importers reconnect, re-import, and every
 * rank re-adds its device and re-binds. cuMulticastBindMem blocks until all
 * devices have joined, so the binds themselves are the cross-rank barrier.
 *
 * Interposition mechanism: plain LD_PRELOAD symbol interposition only catches
 * calls resolved through the global scope (build-time linking). Real CUDA
 * consumers -- torch, NCCL, and this repo's ctypes harnesses -- dlopen
 * libcuda.so.1 and dlsym the entry points (or use cuGetProcAddress), which
 * bypasses symbol interposition. So the shim ALSO interposes dlsym itself
 * (resolving the real dlsym via dlvsym, which we do not wrap) and
 * cuGetProcAddress/_v2, redirecting lookups of the tracked entry points to
 * the shim's wrappers.
 *
 * Scope of THIS prototype: single process.  Multi-rank export/import rendezvous
 * (cross-process cuMemImportFromShareableHandle replay) is deliberately out of
 * scope here -- see README.md.  Unicast device memory stays cuda-checkpoint's
 * responsibility; the shim only manages the multicast layer + the MC VA
 * mappings that reference the released handle.
 *
 * Build:  ./build.sh   (toolkit-free: CUDA types are declared locally)
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
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

/* ------------------------------------------------------------------ */
/* Minimal CUDA driver ABI (x86_64), mirrored from _cuda.py / cuda.h. */
/* ------------------------------------------------------------------ */

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUmemGenericAllocationHandle;

#define CUDA_SUCCESS 0

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

/* ------------------------------------------------------------------ */
/* Logging                                                            */
/* ------------------------------------------------------------------ */

static FILE *g_log;
static pthread_mutex_t g_loglock = PTHREAD_MUTEX_INITIALIZER;

/* Defined with the control thread below; started lazily from the lookup
 * wrappers so only real CUDA consumers poll for control markers. */
static void ensure_control_thread(void);

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

#define CU_MEM_HANDLE_TYPE_POSIX_FD 0x1

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
}

/* ------------------------------------------------------------------ */
/* Tracked state (the live object graph, at the libcuda layer).       */
/*                                                                    */
/* This is live state, not an ioctl log: app-initiated frees/unbinds  */
/* remove entries, so they drop out of the replay set automatically   */
/* (same invariant TASK.md requires of nvproxy's object graph).       */
/* ------------------------------------------------------------------ */

#define MAXN 512
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
} Bind;

static Alloc g_alloc[MAXN];
static Mapping g_map[MAXN];
static Bind g_bind[MAXN];
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static volatile int g_suspended;

static int alloc_new(void) {
	for (int i = 0; i < MAXN; i++)
		if (g_alloc[i].kind == KIND_FREE)
			return i;
	return -1;
}

/* Find alloc index whose CURRENT handle matches h. */
static int alloc_find(CUmemGenericAllocationHandle h) {
	for (int i = 0; i < MAXN; i++)
		if (g_alloc[i].kind != KIND_FREE && g_alloc[i].handle == h)
			return i;
	return -1;
}

/* Translate a possibly-stale MC handle to the group's current handle. The app
 * (or NCCL) may retain the original handle in its structs; after a resume the
 * group has a new handle, so rewrite calls that reference an old value. */
static CUmemGenericAllocationHandle xlate_mc(CUmemGenericAllocationHandle h) {
	for (int i = 0; i < MAXN; i++) {
		if (g_alloc[i].kind != KIND_MC)
			continue;
		for (int a = 0; a < g_alloc[i].naka; a++)
			if (g_alloc[i].aka[a] == h)
				return g_alloc[i].handle;
	}
	return h;
}

static void alloc_push_aka(Alloc *a, CUmemGenericAllocationHandle h) {
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

CUresult cuMemCreate(CUmemGenericAllocationHandle *h, size_t size,
                     const CUmemAllocationProp *prop, unsigned long long flags) {
	resolve_reals();
	CUresult rc = r_cuMemCreate(h, size, prop, flags);
	if (rc == CUDA_SUCCESS) {
		pthread_mutex_lock(&g_lock);
		int i = alloc_new();
		if (i >= 0) {
			Alloc *a = alloc_init(i, KIND_UC, *h);
			a->size = size;
			if (prop)
				a->uprop = *prop;
		}
		pthread_mutex_unlock(&g_lock);
	}
	return rc;
}

CUresult cuMulticastCreate(CUmemGenericAllocationHandle *h,
                           const CUmulticastObjectProp *prop) {
	resolve_reals();
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
		}
		pthread_mutex_unlock(&g_lock);
	}
	return rc;
}

CUresult cuMemExportToShareableHandle(void *shHandle,
                                      CUmemGenericAllocationHandle h, int type,
                                      unsigned long long flags) {
	resolve_reals();
	CUmemGenericAllocationHandle real_h = xlate_locked(h);
	CUresult rc = r_cuMemExportToShareableHandle(shHandle, real_h, type, flags);
	if (rc == CUDA_SUCCESS && type == CU_MEM_HANDLE_TYPE_POSIX_FD && shHandle) {
		pthread_mutex_lock(&g_lock);
		int i = alloc_find(real_h);
		if (i >= 0 && g_alloc[i].kind == KIND_MC) {
			record_key(i, *(int *)shHandle);
			mclog("track EXPORT group=%d key=%lx:%lx ord=%d", i,
			      g_alloc[i].key_dev, g_alloc[i].key_ino,
			      g_alloc[i].key_ord);
		}
		pthread_mutex_unlock(&g_lock);
	}
	return rc;
}

CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *h,
                                        void *osHandle, int type) {
	resolve_reals();
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
		}
		pthread_mutex_unlock(&g_lock);
	}
	return rc;
}

CUresult cuMulticastAddDevice(CUmemGenericAllocationHandle h, CUdevice dev) {
	resolve_reals();
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
		r_cuCtxGetCurrent(&g_bind[b].ctx);
		mclog("track BIND%s group=%d %s=0x%llx dev=%d mcOff=0x%zx "
		      "size=0x%zx",
		      by_addr ? "-ADDR" : "", gi, by_addr ? "va" : "mem",
		      (unsigned long long)(by_addr ? va : mem), dev, mcOffset,
		      size);
		return;
	}
	mclog("WARNING: bind table full; bind not tracked");
}

CUresult cuMulticastBindMem(CUmemGenericAllocationHandle mc, size_t mcOffset,
                            CUmemGenericAllocationHandle mem, size_t memOffset,
                            size_t size, unsigned long long flags) {
	resolve_reals();
	CUmemGenericAllocationHandle real_mc = xlate_locked(mc);
	CUresult rc =
	    r_cuMulticastBindMem(real_mc, mcOffset, mem, memOffset, size, flags);
	if (rc == CUDA_SUCCESS) {
		pthread_mutex_lock(&g_lock);
		/* The unbind is per-device: the device hosting the memory
		 * comes from the UC alloc's prop. */
		CUdevice dev = -1;
		int mi = alloc_find(mem);
		if (mi >= 0 && g_alloc[mi].kind == KIND_UC)
			dev = g_alloc[mi].uprop.location.id;
		bind_record(alloc_find(real_mc), 0, mem, 0, mcOffset, memOffset,
		            size, dev);
		pthread_mutex_unlock(&g_lock);
	}
	return rc;
}

CUresult cuMulticastBindAddr(CUmemGenericAllocationHandle mc, size_t mcOffset,
                             CUdeviceptr memptr, size_t size,
                             unsigned long long flags) {
	resolve_reals();
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
	CUmemGenericAllocationHandle real_mc = xlate_locked(mc);
	CUresult rc = r_cuMulticastUnbind(real_mc, dev, mcOffset, size);
	/* App-initiated: drop this device's recorded bind so it leaves the
	 * replay set. (While suspended the app is quiesced by protocol; the
	 * shim's own teardown calls the reals directly and never gets here.) */
	if (rc == CUDA_SUCCESS && !g_suspended) {
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
	CUmemGenericAllocationHandle real_h = xlate_locked(handle);
	CUresult rc = r_cuMemMap(ptr, size, offset, real_h, flags);
	if (rc == CUDA_SUCCESS && !g_suspended) {
		pthread_mutex_lock(&g_lock);
		int ai = alloc_find(real_h);
		if (ai >= 0) {
			for (int m = 0; m < MAXN; m++) {
				if (g_map[m].used)
					continue;
				g_map[m].used = 1;
				g_map[m].va = ptr;
				g_map[m].size = size;
				g_map[m].offset = offset;
				g_map[m].allocIdx = ai;
				g_map[m].has_access = 0;
				r_cuCtxGetCurrent(&g_map[m].ctx);
				break;
			}
		}
		pthread_mutex_unlock(&g_lock);
	}
	return rc;
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
	resolve_reals();
	CUresult rc = r_cuMemUnmap(ptr, size);
	/* App-initiated: forget the mapping. */
	if (rc == CUDA_SUCCESS && !g_suspended) {
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
	CUresult rc = r_cuMemSetAccess(ptr, size, desc, count);
	if (rc == CUDA_SUCCESS && desc && count >= 1) {
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
	g_alloc[i].kind = KIND_FREE;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
	resolve_reals();
	CUmemGenericAllocationHandle real_h = xlate_locked(handle);
	CUresult rc = r_cuMemRelease(real_h);
	/* App-initiated: forget the alloc and its dependents. */
	if (rc == CUDA_SUCCESS && !g_suspended) {
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

static char g_dir[512] = "/tmp";

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
	if (recvmsg(sock, &msg, 0) <= 0)
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
 * freed and reused while this thread runs); exits when stop_serving closes
 * the listening socket. */
typedef struct {
	int sock;
	int fd;
	char path[104];
} ServeArgs;

static void *serve_thread(void *arg) {
	ServeArgs *sa = arg;
	mclog("serving group fd on %s", sa->path);
	for (;;) {
		int c = accept(sa->sock, NULL, NULL);
		if (c < 0)
			break;
		if (send_fd(c, sa->fd) != 0)
			mclog("serve: send_fd failed: %s", strerror(errno));
		close(c);
	}
	mclog("serve thread for %s exiting", sa->path);
	free(sa);
	return NULL;
}

/* Must hold g_lock (mutates Alloc). */
static int start_serving(Alloc *a, int fd) {
	if (group_sock_path(a, a->serve_path, sizeof(a->serve_path)) != 0)
		return -1;
	struct sockaddr_un sa;
	unlink(a->serve_path);
	int s = socket(AF_UNIX, SOCK_STREAM, 0);
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
	args->sock = s;
	args->fd = fd;
	strcpy(args->path, a->serve_path);
	pthread_t t;
	if (pthread_create(&t, NULL, serve_thread, args) != 0) {
		close(s);
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
		close(a->serve_sock); /* unblocks accept -> thread exits */
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

/* Importer-side: connect (with retry; the creator may not be serving yet)
 * and receive the new group fd. */
static int fetch_group_fd(const Alloc *a, int timeout_ms) {
	char path[104];
	if (group_sock_path(a, path, sizeof(path)) != 0)
		return -1;
	struct sockaddr_un sa;
	memset(&sa, 0, sizeof(sa));
	sa.sun_family = AF_UNIX;
	strcpy(sa.sun_path, path);
	for (int waited = 0; waited < timeout_ms; waited += 100) {
		int s = socket(AF_UNIX, SOCK_STREAM, 0);
		if (s < 0)
			return -1;
		if (connect(s, (struct sockaddr *)&sa, sizeof(sa)) == 0) {
			int fd = recv_fd(s);
			close(s);
			if (fd >= 0)
				return fd;
		}
		close(s);
		struct timespec ts = {0, 100 * 1000 * 1000};
		nanosleep(&ts, NULL);
	}
	mclog("RESUME: timed out fetching group fd from %s", path);
	return -1;
}

/* ------------------------------------------------------------------ */
/* Suspend / resume of the multicast layer                            */
/* ------------------------------------------------------------------ */

/* Must hold g_lock. */
static int do_suspend(void) {
	int groups = 0, unmapped = 0, unbound = 0, released = 0;
	CUcontext saved = NULL;
	r_cuCtxGetCurrent(&saved);

	for (int gi = 0; gi < MAXN; gi++) {
		if (g_alloc[gi].kind != KIND_MC)
			continue;
		groups++;

		/* 0. Stop serving a previous resume's re-exported fd; a held
		 *    export fd is itself a checkpoint blocker. */
		stop_serving(&g_alloc[gi]);

		/* 1. Unmap every MC VA that maps this group; KEEP the VA
		 *    reservation (cuMemUnmap only -- never cuMemAddressFree). */
		for (int m = 0; m < MAXN; m++) {
			if (!g_map[m].used || g_map[m].allocIdx != gi)
				continue;
			if (g_map[m].ctx)
				r_cuCtxSetCurrent(g_map[m].ctx);
			CUresult rc = r_cuMemUnmap(g_map[m].va, g_map[m].size);
			if (rc != CUDA_SUCCESS) {
				mclog("SUSPEND: cuMemUnmap(0x%llx) failed rc=%d",
				      (unsigned long long)g_map[m].va, rc);
				return -1;
			}
			unmapped++;
			mclog("SUSPEND: unmapped MC VA 0x%llx (reservation kept)",
			      (unsigned long long)g_map[m].va);
		}

		/* 2. Unbind each recorded bind once, on the device hosting
		 *    its memory. */
		for (int b = 0; b < MAXN; b++) {
			if (!g_bind[b].used || g_bind[b].groupIdx != gi)
				continue;
			if (g_bind[b].dev < 0) {
				mclog("SUSPEND: bind %d has unknown device", b);
				return -1;
			}
			CUresult crc = 0;
			if (g_bind[b].ctx)
				crc = r_cuCtxSetCurrent(g_bind[b].ctx);
			CUresult rc = r_cuMulticastUnbind(
			    g_alloc[gi].handle, g_bind[b].dev,
			    g_bind[b].mcOffset, g_bind[b].size);
			if (rc != CUDA_SUCCESS) {
				mclog("SUSPEND: cuMulticastUnbind(mc=0x%llx, "
				      "dev=%d, mcOff=0x%zx, size=0x%zx) rc=%d "
				      "(setctx=%p rc=%d)",
				      (unsigned long long)g_alloc[gi].handle,
				      g_bind[b].dev, g_bind[b].mcOffset,
				      g_bind[b].size, rc, g_bind[b].ctx, crc);
				return -1;
			}
			unbound++;
		}

		/* 3. Release the 0x00fd multicast handle. */
		CUresult rc = r_cuMemRelease(g_alloc[gi].handle);
		if (rc != CUDA_SUCCESS) {
			mclog("SUSPEND: cuMemRelease(MC 0x%llx) rc=%d",
			      (unsigned long long)g_alloc[gi].handle, rc);
			return -1;
		}
		released++;
		mclog("SUSPEND: released MC group idx=%d handle=0x%llx", gi,
		      (unsigned long long)g_alloc[gi].handle);
	}

	if (saved)
		r_cuCtxSetCurrent(saved);
	if (r_cuCtxSynchronize)
		r_cuCtxSynchronize();
	mclog("SUSPEND done: groups=%d unmapped=%d unbound=%d released=%d",
	      groups, unmapped, unbound, released);
	return 0;
}

/* Must hold g_lock. */
static int do_resume(void) {
	int groups = 0, remapped = 0, rebound = 0;
	CUcontext saved = NULL;
	r_cuCtxGetCurrent(&saved);

	for (int gi = 0; gi < MAXN; gi++) {
		if (g_alloc[gi].kind != KIND_MC)
			continue;
		groups++;

		/* 1. Re-establish the multicast object (new handle is fine).
		 *    Creator: cuMulticastCreate, then re-export + serve the fd
		 *    BEFORE AddDevice/Bind, so importers are never starved
		 *    while our own bind blocks waiting for them to join.
		 *    Importer: fetch the creator's new fd and re-import. */
		if (g_alloc[gi].ctx)
			r_cuCtxSetCurrent(g_alloc[gi].ctx);
		CUmemGenericAllocationHandle newmc = 0;
		CUresult rc;
		if (!g_alloc[gi].imported) {
			rc = r_cuMulticastCreate(&newmc, &g_alloc[gi].mprop);
			if (rc != CUDA_SUCCESS) {
				mclog("RESUME: cuMulticastCreate rc=%d", rc);
				return -1;
			}
			if (g_alloc[gi].has_key) {
				int fd = -1;
				rc = r_cuMemExportToShareableHandle(
				    &fd, newmc, CU_MEM_HANDLE_TYPE_POSIX_FD, 0);
				if (rc != CUDA_SUCCESS || fd < 0) {
					mclog("RESUME: re-export rc=%d fd=%d",
					      rc, fd);
					return -1;
				}
				if (start_serving(&g_alloc[gi], fd) != 0) {
					close(fd);
					return -1;
				}
			}
		} else {
			if (!g_alloc[gi].has_key) {
				mclog("RESUME: imported group idx=%d has no "
				      "rendezvous key; cannot re-import",
				      gi);
				return -1;
			}
			/* Concurrent imports of the same group can transiently
			 * fail (observed: CUDA_ERROR_OPERATING_SYSTEM=304 when
			 * several ranks import within ~1ms). Retry with a fresh
			 * fd, bounded so a real failure stays loud. */
			int attempt = 0;
			for (;;) {
				int fd = fetch_group_fd(&g_alloc[gi], 60 * 1000);
				if (fd < 0)
					return -1;
				rc = r_cuMemImportFromShareableHandle(
				    &newmc, (void *)(intptr_t)fd,
				    CU_MEM_HANDLE_TYPE_POSIX_FD);
				close(fd);
				if (rc == CUDA_SUCCESS)
					break;
				if (++attempt >= 100) {
					mclog("RESUME: re-import rc=%d after "
					      "%d attempts, giving up",
					      rc, attempt);
					return -1;
				}
				mclog("RESUME: re-import rc=%d (attempt %d), "
				      "retrying",
				      rc, attempt);
				struct timespec ts = {0, 200 * 1000 * 1000};
				nanosleep(&ts, NULL);
			}
		}
		CUmemGenericAllocationHandle oldmc = g_alloc[gi].handle;
		alloc_push_aka(&g_alloc[gi], newmc);
		mclog("RESUME: re-established MC group idx=%d 0x%llx -> 0x%llx "
		      "(%s)",
		      gi, (unsigned long long)oldmc, (unsigned long long)newmc,
		      g_alloc[gi].imported ? "re-imported" : "recreated+served");

		/* 2. Re-add the participating devices. */
		for (int d = 0; d < g_alloc[gi].ndev; d++) {
			rc = r_cuMulticastAddDevice(newmc, g_alloc[gi].devs[d]);
			if (rc != CUDA_SUCCESS) {
				mclog("RESUME: cuMulticastAddDevice dev=%d rc=%d",
				      g_alloc[gi].devs[d], rc);
				return -1;
			}
		}

		/* 3. Re-bind the same memory (restored verbatim by
		 *    cuda-checkpoint) into the group: by handle for BindMem
		 *    entries, by (stable) VA for BindAddr entries. */
		for (int b = 0; b < MAXN; b++) {
			if (!g_bind[b].used || g_bind[b].groupIdx != gi)
				continue;
			if (g_bind[b].ctx)
				r_cuCtxSetCurrent(g_bind[b].ctx);
			if (g_bind[b].by_addr)
				rc = r_cuMulticastBindAddr(newmc,
				                           g_bind[b].mcOffset,
				                           g_bind[b].va,
				                           g_bind[b].size, 0);
			else
				rc = r_cuMulticastBindMem(newmc,
				                          g_bind[b].mcOffset,
				                          g_bind[b].mem,
				                          g_bind[b].memOffset,
				                          g_bind[b].size, 0);
			if (rc != CUDA_SUCCESS) {
				mclog("RESUME: re-bind (%s) rc=%d",
				      g_bind[b].by_addr ? "addr" : "mem", rc);
				return -1;
			}
			rebound++;
		}

		/* 4. Re-map every MC VA at its IDENTICAL address. Prefer the
		 *    retained reservation; if it did not survive restore,
		 *    re-reserve at the fixed address. */
		for (int m = 0; m < MAXN; m++) {
			if (!g_map[m].used || g_map[m].allocIdx != gi)
				continue;
			if (g_map[m].ctx)
				r_cuCtxSetCurrent(g_map[m].ctx);
			const char *path = "retained-reservation";
			rc = r_cuMemMap(g_map[m].va, g_map[m].size,
			                g_map[m].offset, newmc, 0);
			if (rc != CUDA_SUCCESS) {
				CUdeviceptr got = 0;
				CUresult rr = r_cuMemAddressReserve(
				    &got, g_map[m].size, 0, g_map[m].va, 0);
				if (rr != CUDA_SUCCESS || got != g_map[m].va) {
					if (rr == CUDA_SUCCESS)
						r_cuMemAddressFree(got,
						                   g_map[m].size);
					mclog("RESUME: could not re-map at "
					      "identical VA 0x%llx (got 0x%llx "
					      "rr=%d)",
					      (unsigned long long)g_map[m].va,
					      (unsigned long long)got, rr);
					return -1;
				}
				rc = r_cuMemMap(g_map[m].va, g_map[m].size,
				                g_map[m].offset, newmc, 0);
				if (rc != CUDA_SUCCESS) {
					mclog("RESUME: cuMemMap after re-reserve "
					      "rc=%d",
					      rc);
					return -1;
				}
				path = "re-reserved-fixed";
			}
			if (g_map[m].has_access)
				r_cuMemSetAccess(g_map[m].va, g_map[m].size,
				                 &g_map[m].access, 1);
			remapped++;
			mclog("RESUME: MC VA 0x%llx re-mapped IDENTICAL (%s)",
			      (unsigned long long)g_map[m].va, path);
		}
	}

	if (saved)
		r_cuCtxSetCurrent(saved);
	if (r_cuCtxSynchronize)
		r_cuCtxSynchronize();
	mclog("RESUME done: groups=%d rebound=%d remapped=%d", groups, rebound,
	      remapped);
	return 0;
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
CUresult cuGetProcAddress(const char *, void **, int, unsigned long long);
CUresult cuGetProcAddress_v2(const char *, void **, int, unsigned long long,
                             int *);

typedef struct {
	const char *name;
	void *fn;
} WrapEntry;

static const WrapEntry *wrap_table(void) {
	static WrapEntry t[] = {
	    {"cuMemCreate", (void *)cuMemCreate},
	    {"cuMemRelease", (void *)cuMemRelease},
	    {"cuMemMap", (void *)cuMemMap},
	    {"cuMemUnmap", (void *)cuMemUnmap},
	    {"cuMemSetAccess", (void *)cuMemSetAccess},
	    {"cuMulticastCreate", (void *)cuMulticastCreate},
	    {"cuMulticastAddDevice", (void *)cuMulticastAddDevice},
	    {"cuMulticastBindMem", (void *)cuMulticastBindMem},
	    {"cuMulticastBindAddr", (void *)cuMulticastBindAddr},
	    {"cuMulticastUnbind", (void *)cuMulticastUnbind},
	    {"cuInit", (void *)cuInit},
	    {"cuMemExportToShareableHandle",
	     (void *)cuMemExportToShareableHandle},
	    {"cuMemImportFromShareableHandle",
	     (void *)cuMemImportFromShareableHandle},
	    {"cuGetProcAddress", (void *)cuGetProcAddress},
	    {"cuGetProcAddress_v2", (void *)cuGetProcAddress_v2},
	    {NULL, NULL},
	};
	return t;
}

static void *wrapper_for(const char *name) {
	if (!name)
		return NULL;
	for (const WrapEntry *e = wrap_table(); e->name; e++)
		if (strcmp(e->name, name) == 0)
			return e->fn;
	return NULL;
}

/* Interposed dlsym: hand out shim wrappers for tracked driver symbols. */
void *dlsym(void *handle, const char *symbol) {
	init_real_dlsym();
	if (!real_dlsym)
		return NULL;
	void *w = wrapper_for(symbol);
	if (w) {
		/* Only redirect if the real library actually has the symbol
		 * (so feature probes against old drivers still behave). */
		void *r = real_dlsym(handle, symbol);
		if (r) {
			mclog("dlsym(%s) -> shim wrapper", symbol);
			return w;
		}
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
static void *gpa_redirect(const char *symbol, int cudaVersion) {
	if (strcmp(symbol, "cuGetProcAddress") == 0)
		return cudaVersion >= 12000 ? (void *)cuGetProcAddress_v2
		                            : (void *)cuGetProcAddress;
	return wrapper_for(symbol);
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
	    (w = gpa_redirect(symbol, cudaVersion))) {
		mclog("cuGetProcAddress(%s, ver=%d) -> shim wrapper", symbol,
		      cudaVersion);
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
	    (w = gpa_redirect(symbol, cudaVersion))) {
		mclog("cuGetProcAddress_v2(%s, ver=%d) -> shim wrapper", symbol,
		      cudaVersion);
		*pfn = w;
	}
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
	mclog("control thread started (dir=%s)", g_dir);
	int prev = 0; /* treat startup as not-suspended */
	for (;;) {
		int want = marker_exists("suspend");
		if (want != prev) {
			prev = want;
			pthread_mutex_lock(&g_lock);
			resolve_reals();
			int rc;
			if (want) {
				rc = do_suspend();
				if (rc == 0)
					g_suspended = 1;
			} else {
				rc = do_resume();
				if (rc == 0)
					g_suspended = 0;
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
		struct timespec ts = {0, 100 * 1000 * 1000}; /* 100ms */
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
	if (pthread_create(&t, NULL, control_thread, NULL) == 0)
		pthread_detach(t);
}

static void ensure_control_thread(void) {
	static pthread_once_t once = PTHREAD_ONCE_INIT;
	pthread_once(&once, control_thread_start);
}

__attribute__((constructor)) static void mcshim_init(void) {
	if (getenv("MCSHIM_DISABLE")) {
		g_disabled = 1;
		mclog("disabled via MCSHIM_DISABLE");
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
