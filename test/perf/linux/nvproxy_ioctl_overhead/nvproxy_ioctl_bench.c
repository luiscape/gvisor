// Copyright 2024 The gVisor Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// nvproxy_ioctl_bench — measures per-ioctl latency on real NVIDIA kernel
// driver file descriptors to quantify nvproxy Sentry overhead.
//
// Strategy
// --------
// The benchmark opens /dev/nvidiactl and /dev/nvidia0 directly (no CUDA
// runtime, no libcuda), performs the minimum RM object tree needed to have a
// valid client/device/subdevice context, then hammers the ioctls that make up
// the pageable cudaMemcpy hot loop:
//
//   Phase 0 – version handshake    NV_ESC_CHECK_VERSION_STR    nvidiactl
//   Phase 1 – RM root client       NV_ESC_RM_ALLOC             nvidiactl
//   Phase 2 – attach GPU to fd     NV_ESC_ATTACH_GPUS_TO_FD    nvidia0
//   Phase 3 – register fd          NV_ESC_REGISTER_FD          nvidia0
//   Phase 4 – alloc device         NV_ESC_RM_ALLOC             nvidiactl
//   Phase 5 – alloc subdevice      NV_ESC_RM_ALLOC             nvidiactl
//
// Hot-loop targets (all on the live fds after setup):
//
//   T1 – NV_ESC_RM_CONTROL  (NV_ESC_RM_CONTROL, cmd=NV2080_CTRL_CMD_TIMER_GET_TIME)
//        This is the "wait/complete" ioctl in the DMA triplet.  It is a
//        simple passthrough through rmControlSimple — no embedded pointers,
//        no object tracking.  Ideal fast-path candidate.
//
//   T2 – NV_ESC_RM_CONTROL  (cmd=NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS)
//        Another plain rmControlSimple call on nvidiactl.
//
//   T3 – NV_ESC_RM_ALLOC / NV_ESC_RM_FREE pair on nvidiactl
//        Allocates and immediately frees an event object — the alloc/free
//        teardown pattern seen at the end of each transfer chunk.
//
//   T4 – NV_ESC_CHECK_VERSION_STR repeated
//        Cheapest possible frontend ioctl, gives the floor for Sentry
//        overhead with minimal driver work.
//
// Compile
// -------
//   # dynamic (host/runc):
//   gcc -O2 -o nvproxy_ioctl_bench nvproxy_ioctl_bench.c -lm
//
//   # static (works in any container including busybox):
//   gcc -O2 -static -o nvproxy_ioctl_bench_static nvproxy_ioctl_bench.c -lm
//
// Run
// ---
//   # native / runc:
//   ./nvproxy_ioctl_bench
//
//   # gVisor stock:
//   docker run --rm --runtime=runsc --gpus all \
//     -v $PWD/nvproxy_ioctl_bench_static:/bench:ro \
//     --entrypoint /bench busybox:latest
//
//   # gVisor fast-path:
//   docker run --rm --runtime=runsc-fastpath --gpus all \
//     -v $PWD/nvproxy_ioctl_bench_static:/bench:ro \
//     --entrypoint /bench busybox:latest

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// NVIDIA RM ioctl interface — derived from open-source kernel module headers
// and the gVisor nvgpu package definitions.
// ---------------------------------------------------------------------------

#define NV_IOCTL_MAGIC      'F'
#define NV_IOCTL_BASE       200

// Frontend ioctl numbers (IOC_NR component only; full cmd built via _IOWR).
#define NV_ESC_CARD_INFO              (NV_IOCTL_BASE + 0)   // 200 = 0xc8
#define NV_ESC_REGISTER_FD            (NV_IOCTL_BASE + 1)   // 201 = 0xc9
#define NV_ESC_CHECK_VERSION_STR      (NV_IOCTL_BASE + 10)  // 210 = 0xd2
#define NV_ESC_ATTACH_GPUS_TO_FD      (NV_IOCTL_BASE + 12)  // 212 = 0xd4
#define NV_ESC_SYS_PARAMS             (NV_IOCTL_BASE + 14)  // 214 = 0xd6

// From nv_escape.h / gVisor nvgpu package:
#define NV_ESC_RM_FREE                0x29
#define NV_ESC_RM_CONTROL             0x2a
#define NV_ESC_RM_ALLOC               0x2b

// Build a full _IOWR ioctl command from (type, nr, struct_size).
#define NV_IOWR(nr, size) \
    ((unsigned long)(((3U)<<30)|((NV_IOCTL_MAGIC)<<8)|((nr)&0xff)|((size)<<16)))

// NV status codes.
#define NV_OK               0x00000000
#define NV_ERR_NOT_SUPPORTED 0x00000056

// RM object class IDs.
#define NV01_ROOT_CLIENT    0x00000041
#define NV01_DEVICE_0       0x00000080
#define NV20_SUBDEVICE_0    0x00002080
#define NV01_EVENT_OS_EVENT 0x00000079

// Control command IDs used in the hot loops.
// NV2080_CTRL_CMD_TIMER_GET_TIME — subdevice timer read; plain rmControlSimple.
#define NV2080_CTRL_CMD_TIMER_GET_TIME          0x20800401
// NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS — client-level query; plain rmControlSimple.
#define NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS    0x00000201
// NV2080_CTRL_CMD_GPU_GET_ID — subdevice GPU ID query.
#define NV2080_CTRL_CMD_GPU_GET_ID              0x20800142
// NV2080_CTRL_CMD_MC_SERVICE_INTERRUPTS — simple fire-and-forget control.
#define NV2080_CTRL_CMD_MC_SERVICE_INTERRUPTS   0x20801702

// ---------------------------------------------------------------------------
// Parameter structs (field layout matches NVIDIA driver ABI exactly).
// All structs use only uint32_t and uint64_t to avoid compiler padding
// surprises; fields match the gVisor nvgpu Go definitions byte-for-byte.
// ---------------------------------------------------------------------------

// NV_ESC_CHECK_VERSION_STR parameter.
typedef struct {
    uint32_t cmd;
    uint32_t reply;
    char     versionString[64];
} RMAPIVersion;

// NV_ESC_SYS_PARAMS parameter.
typedef struct {
    uint64_t memSize;
} IoctlSysParams;

// NV_ESC_REGISTER_FD parameter.
typedef struct {
    int32_t ctlFD;
} IoctlRegisterFD;

// NV_ESC_RM_ALLOC parameter (NVOS64 variant — 48 bytes, matches driver usage).
// The driver always uses NVOS64 on modern drivers; NVOS21 is an older alias.
typedef struct {
    uint32_t hRoot;
    uint32_t hObjectParent;
    uint32_t hObjectNew;
    uint32_t hClass;
    uint64_t pAllocParms;           // pointer to class-specific alloc params
    uint64_t pRightsRequested;      // NULL for most allocations
    uint32_t paramsSize;
    uint32_t flags;
    uint32_t status;
    uint32_t _pad;
} NVOS64_PARAMETERS;

// Keep NVOS21 name as alias so existing call sites compile.
#define NVOS21_PARAMETERS NVOS64_PARAMETERS

// NV_ESC_RM_FREE parameter.
typedef struct {
    uint32_t hRoot;
    uint32_t hObjectParent;
    uint32_t hObjectOld;
    uint32_t status;
} NVOS00_PARAMETERS;

// NV_ESC_RM_CONTROL parameter (NVOS54).
typedef struct {
    uint32_t hClient;
    uint32_t hObject;
    uint32_t cmd;
    uint32_t flags;
    uint64_t params;        // pointer to cmd-specific params
    uint32_t paramsSize;
    uint32_t status;
} NVOS54_PARAMETERS;

// NV01_DEVICE_0 alloc params.
typedef struct {
    uint32_t deviceId;
    uint32_t hClientShare;
    uint32_t hTargetClient;
    uint32_t hTargetDevice;
    uint32_t flags;
    uint32_t pad0;
    uint64_t vaSpaceSize;
    uint64_t vaStartInternal;
    uint64_t vaLimitInternal;
    uint32_t vaMode;
    uint32_t pad1;
} NV0080_ALLOC_PARAMETERS;

// NV20_SUBDEVICE_0 alloc params.
typedef struct {
    uint32_t subDeviceId;
} NV2080_ALLOC_PARAMETERS;

// NV2080_CTRL_CMD_TIMER_GET_TIME params (16 bytes).
typedef struct {
    uint64_t time_nsec;
    uint32_t flags;
    uint32_t pad;
} NV2080_CTRL_TIMER_GET_TIME_PARAMS;

// NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS params.
#define NV0000_CTRL_SYSTEM_MAX_ATTACHED_GPUS 32
typedef struct {
    uint32_t gpuIds[NV0000_CTRL_SYSTEM_MAX_ATTACHED_GPUS];
} NV0000_CTRL_GPU_GET_ATTACHED_IDS_PARAMS;

// NV2080_CTRL_CMD_GPU_GET_ID params.
typedef struct {
    uint32_t gpuId;
} NV2080_CTRL_GPU_GET_ID_PARAMS;

// ---------------------------------------------------------------------------
// ioctl cmd builders
// ---------------------------------------------------------------------------
static unsigned long fe_cmd(unsigned int nr, unsigned int sz) {
    // Cap size field to 14 bits (ioctl encoding limit).
    return NV_IOWR(nr, sz & 0x3fff);
}

// Convenience: ioctl command for NV_ESC_RM_ALLOC using NVOS64 size (48 bytes).
#define RM_ALLOC_CMD  fe_cmd(NV_ESC_RM_ALLOC,  (unsigned int)sizeof(NVOS64_PARAMETERS))
#define RM_FREE_CMD   fe_cmd(NV_ESC_RM_FREE,   (unsigned int)sizeof(NVOS00_PARAMETERS))
#define RM_CTRL_CMD   fe_cmd(NV_ESC_RM_CONTROL,(unsigned int)sizeof(NVOS54_PARAMETERS))

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline double ns2us(uint64_t ns) { return (double)ns / 1e3; }
static inline double ns2ms(uint64_t ns) { return (double)ns / 1e6; }

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------
typedef struct {
    uint64_t *samples;
    size_t    n;
    double    mean_ns;
    double    min_ns;
    double    max_ns;
    double    p50_ns;
    double    p90_ns;
    double    p99_ns;
    double    p999_ns;
    double    stddev_ns;
} stats_t;

static int cmp_u64(const void *a, const void *b) {
    uint64_t va = *(const uint64_t *)a, vb = *(const uint64_t *)b;
    return (va > vb) - (va < vb);
}

static void compute_stats(stats_t *s) {
    if (!s->n) return;
    qsort(s->samples, s->n, sizeof(uint64_t), cmp_u64);
    s->min_ns = (double)s->samples[0];
    s->max_ns = (double)s->samples[s->n - 1];
    double sum = 0;
    for (size_t i = 0; i < s->n; i++) sum += (double)s->samples[i];
    s->mean_ns = sum / (double)s->n;
    double var = 0;
    for (size_t i = 0; i < s->n; i++) {
        double d = (double)s->samples[i] - s->mean_ns;
        var += d * d;
    }
    s->stddev_ns = sqrt(var / (double)s->n);
    s->p50_ns  = (double)s->samples[(size_t)(s->n * 0.50)];
    s->p90_ns  = (double)s->samples[(size_t)(s->n * 0.90)];
    s->p99_ns  = (double)s->samples[(size_t)(s->n * 0.99)];
    s->p999_ns = (double)s->samples[(size_t)(s->n * 0.999 < s->n ? s->n * 0.999 : s->n - 1)];
}

static stats_t alloc_stats(size_t n) {
    stats_t s = {0};
    s.n = n;
    s.samples = calloc(n, sizeof(uint64_t));
    if (!s.samples) { perror("calloc"); exit(1); }
    return s;
}

static void free_stats(stats_t *s) { free(s->samples); s->samples = NULL; }

static void print_stats_row(const char *label, const stats_t *s) {
    printf("  %-46s  min=%7.2f  p50=%7.2f  p90=%7.2f  p99=%7.2f  p99.9=%7.2f  max=%8.2f  mean=%7.2f ± %.2f us\n",
           label,
           ns2us((uint64_t)s->min_ns),
           ns2us((uint64_t)s->p50_ns),
           ns2us((uint64_t)s->p90_ns),
           ns2us((uint64_t)s->p99_ns),
           ns2us((uint64_t)s->p999_ns),
           ns2us((uint64_t)s->max_ns),
           s->mean_ns / 1e3,
           s->stddev_ns / 1e3);
}

// ---------------------------------------------------------------------------
// RM handle allocator — monotonically increasing, starting at 0xDEAD0000
// so they're recognisable in strace output.
// ---------------------------------------------------------------------------
static uint32_t g_next_handle = 0xDEAD0001;
static uint32_t alloc_handle(void) { return g_next_handle++; }

// ---------------------------------------------------------------------------
// RM context — holds open fds and allocated handles.
// ---------------------------------------------------------------------------
typedef struct {
    int      ctl_fd;        // /dev/nvidiactl
    int      dev_fd;        // /dev/nvidia0
    uint32_t h_client;      // root RM client
    uint32_t h_device;      // NV01_DEVICE_0
    uint32_t h_subdevice;   // NV20_SUBDEVICE_0
    uint32_t gpu_id;        // GPU ID from attached IDs list
} rm_ctx_t;

// ---------------------------------------------------------------------------
// Helper: issue NV_ESC_RM_ALLOC and return status.
// ---------------------------------------------------------------------------
static uint32_t rm_alloc(int fd, uint32_t h_root, uint32_t h_parent,
                          uint32_t h_new, uint32_t h_class,
                          void *alloc_params, uint32_t params_size) {
    NVOS64_PARAMETERS p = {0};
    p.hRoot              = h_root;
    p.hObjectParent      = h_parent;
    p.hObjectNew         = h_new;
    p.hClass             = h_class;
    p.pAllocParms        = (uint64_t)(uintptr_t)alloc_params;
    p.pRightsRequested   = 0;
    p.paramsSize         = params_size;
    p.flags              = 0;
    p.status             = ~0u;
    if (ioctl(fd, RM_ALLOC_CMD, &p) < 0) return (uint32_t)errno;
    return p.status;
}

// ---------------------------------------------------------------------------
// Helper: issue NV_ESC_RM_FREE and return status.
// ---------------------------------------------------------------------------
static uint32_t rm_free(int fd, uint32_t h_root, uint32_t h_parent,
                         uint32_t h_object) {
    NVOS00_PARAMETERS p = {0};
    p.hRoot         = h_root;
    p.hObjectParent = h_parent;
    p.hObjectOld    = h_object;
    p.status        = ~0u;
    if (ioctl(fd, RM_FREE_CMD, &p) < 0) return (uint32_t)errno;
    return p.status;
}

// ---------------------------------------------------------------------------
// Helper: issue NV_ESC_RM_CONTROL and return status.
// ---------------------------------------------------------------------------
static uint32_t rm_control(int fd, uint32_t h_client, uint32_t h_object,
                            uint32_t cmd_id, void *ctrl_params,
                            uint32_t params_size) {
    NVOS54_PARAMETERS p = {0};
    p.hClient    = h_client;
    p.hObject    = h_object;
    p.cmd        = cmd_id;
    p.flags      = 0;
    p.params     = (uint64_t)(uintptr_t)ctrl_params;
    p.paramsSize = params_size;
    p.status     = ~0u;
    if (ioctl(fd, RM_CTRL_CMD, &p) < 0) return (uint32_t)errno;
    return p.status;
}

// ---------------------------------------------------------------------------
// IoctlCardInfo — layout of a single entry returned by NV_ESC_CARD_INFO.
// The driver passes an array of these; we read slot 0 to get the GPU ID.
// Field offsets come from nv-ioctl.h / gVisor IoctlCardInfo definition.
// Total size per entry: 1+3+12+4+2+2+8+8+8+8+4+10+2 = 72 bytes.
// The ioctl transfers NV_MAX_CARDS (32) entries = 32×72 = 2304 = 0x900 bytes,
// which matches what strace shows: _IOC(…, 0x46, 0xc8, 0x900).
// ---------------------------------------------------------------------------
#define NV_MAX_CARDS      32
#define CARD_INFO_ENTRY   72   // bytes per nv_ioctl_card_info_t entry

typedef struct {
    uint8_t  valid;
    uint8_t  _pad0[3];
    // PCIInfo: domain(4) bus(1) slot(1) func(1) pad(1) vendorId(2) deviceId(2) = 12
    uint8_t  pci[12];
    uint32_t gpuId;
    uint16_t interruptLine;
    uint8_t  _pad1[2];
    uint64_t regAddress;
    uint64_t regSize;
    uint64_t fbAddress;
    uint64_t fbSize;
    uint32_t minorNumber;
    char     devName[10];
    uint8_t  _pad2[2];
} CardInfoEntry;

// Compile-time size assertion.
typedef char _card_info_size_check[sizeof(CardInfoEntry) == CARD_INFO_ENTRY ? 1 : -1];

// ---------------------------------------------------------------------------
// Open and initialise the RM context.
// Returns 0 on success, -1 on error (with message to stderr).
//
// All hot-path ioctls (RM_ALLOC, RM_FREE, RM_CONTROL) go to /dev/nvidiactl.
// We do NOT use NV_ESC_ATTACH_GPUS_TO_FD — that ioctl requires real GPU IDs
// (not indices) obtained from card_info, and is only needed for operations
// on /dev/nvidia0.  The benchmarked RM_CONTROL calls all target nvidiactl.
// ---------------------------------------------------------------------------
static int rm_ctx_open(rm_ctx_t *ctx) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->ctl_fd = -1;
    ctx->dev_fd = -1;

    // Open control fd — all benchmarked ioctls go here.
    ctx->ctl_fd = open("/dev/nvidiactl", O_RDWR);
    if (ctx->ctl_fd < 0) { perror("open /dev/nvidiactl"); return -1; }

    // Open device fd (needed later for ATTACH/REGISTER if desired, but not
    // for the hot-loop bench; we open it so it's available).
    ctx->dev_fd = open("/dev/nvidia0", O_RDWR);
    if (ctx->dev_fd < 0) { perror("open /dev/nvidia0"); return -1; }

    // Phase 0: version handshake — driver requires this before RM_ALLOC.
    {
        RMAPIVersion ver = {0};
        ver.cmd = 0x32;   // NV_RM_API_VERSION_CMD_STRICT
        snprintf(ver.versionString, sizeof(ver.versionString), "%s", "580.95.05");
        unsigned long cmd = fe_cmd(NV_ESC_CHECK_VERSION_STR, (unsigned int)sizeof(ver));
        if (ioctl(ctx->ctl_fd, cmd, &ver) < 0) {
            perror("NV_ESC_CHECK_VERSION_STR"); return -1;
        }
        // reply != 0 → version mismatch; driver may still proceed, so warn only.
        if (ver.reply != 0)
            fprintf(stderr, "warning: version mismatch reply=0x%x\n", ver.reply);
    }

    // Phase 1: read card info to discover the real GPU ID.
    {
        CardInfoEntry cards[NV_MAX_CARDS];
        memset(cards, 0, sizeof(cards));
        // Size = 32 × 72 = 2304 = 0x900, matching what strace shows.
        unsigned long cmd = fe_cmd(NV_ESC_CARD_INFO, (unsigned int)sizeof(cards));
        if (ioctl(ctx->ctl_fd, cmd, cards) == 0 && cards[0].valid)
            ctx->gpu_id = cards[0].gpuId;
    }

    // Phase 2: allocate root RM client on nvidiactl.
    ctx->h_client = alloc_handle();
    {
        uint32_t st = rm_alloc(ctx->ctl_fd,
                               ctx->h_client, ctx->h_client, ctx->h_client,
                               NV01_ROOT_CLIENT, NULL, 0);
        if (st != NV_OK) {
            fprintf(stderr, "NV_ESC_RM_ALLOC NV01_ROOT_CLIENT: status=0x%x\n", st);
            return -1;
        }
    }

    // Phase 3: allocate NV01_DEVICE_0 under the client.
    ctx->h_device = alloc_handle();
    {
        NV0080_ALLOC_PARAMETERS dp = {0};
        dp.deviceId      = 0;
        dp.hClientShare  = ctx->h_client;
        dp.hTargetClient = ctx->h_client;
        uint32_t st = rm_alloc(ctx->ctl_fd,
                               ctx->h_client, ctx->h_client, ctx->h_device,
                               NV01_DEVICE_0, &dp, (uint32_t)sizeof(dp));
        if (st != NV_OK) {
            fprintf(stderr, "NV_ESC_RM_ALLOC NV01_DEVICE_0: status=0x%x\n", st);
            return -1;
        }
    }

    // Phase 4: allocate NV20_SUBDEVICE_0 under the device.
    ctx->h_subdevice = alloc_handle();
    {
        NV2080_ALLOC_PARAMETERS sdp = {0};
        sdp.subDeviceId = 0;
        uint32_t st = rm_alloc(ctx->ctl_fd,
                               ctx->h_client, ctx->h_device, ctx->h_subdevice,
                               NV20_SUBDEVICE_0, &sdp, (uint32_t)sizeof(sdp));
        if (st != NV_OK) {
            fprintf(stderr, "NV_ESC_RM_ALLOC NV20_SUBDEVICE_0: status=0x%x\n", st);
            return -1;
        }
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Tear down the RM context.
// ---------------------------------------------------------------------------
static void rm_ctx_close(rm_ctx_t *ctx) {
    if (ctx->h_subdevice)
        rm_free(ctx->ctl_fd, ctx->h_client, ctx->h_device, ctx->h_subdevice);
    if (ctx->h_device)
        rm_free(ctx->ctl_fd, ctx->h_client, ctx->h_client, ctx->h_device);
    if (ctx->h_client)
        rm_free(ctx->ctl_fd, ctx->h_client, ctx->h_client, ctx->h_client);
    if (ctx->dev_fd >= 0) close(ctx->dev_fd);
    if (ctx->ctl_fd >= 0) close(ctx->ctl_fd);
    memset(ctx, 0, sizeof(*ctx));
    ctx->ctl_fd = ctx->dev_fd = -1;
}

// ---------------------------------------------------------------------------
// Benchmark: NV_ESC_RM_CONTROL — subdevice timer read.
//
// This is the canonical "RM_CONTROL wait/complete" ioctl in the DMA transfer
// hot loop. It goes through rmControl → rmControlSimple (GSS-legacy mask is
// set on timer commands, or it falls through to a known-simple handler).
// Parameter size is 16 bytes — fits comfortably in the pool buffer.
// ---------------------------------------------------------------------------
static void bench_rmcontrol_timer(rm_ctx_t *ctx, size_t warmup, stats_t *s) {
    NV2080_CTRL_TIMER_GET_TIME_PARAMS tp = {0};
    NVOS54_PARAMETERS p = {0};
    p.hClient    = ctx->h_client;
    p.hObject    = ctx->h_subdevice;
    p.cmd        = NV2080_CTRL_CMD_TIMER_GET_TIME;
    p.params     = (uint64_t)(uintptr_t)&tp;
    p.paramsSize = (uint32_t)sizeof(tp);
    for (size_t i = 0; i < warmup; i++) {
        p.status = 0;
        ioctl(ctx->ctl_fd, RM_CTRL_CMD, &p);
    }
    for (size_t i = 0; i < s->n; i++) {
        p.status = 0;
        uint64_t t0 = now_ns();
        ioctl(ctx->ctl_fd, RM_CTRL_CMD, &p);
        s->samples[i] = now_ns() - t0;
    }
}

// ---------------------------------------------------------------------------
// Benchmark: NV_ESC_RM_CONTROL — GPU-get-ID on subdevice.
// Small params (4 bytes), simple passthrough.
// ---------------------------------------------------------------------------
static void bench_rmcontrol_gpu_id(rm_ctx_t *ctx, size_t warmup, stats_t *s) {
    NV2080_CTRL_GPU_GET_ID_PARAMS gp = {0};
    NVOS54_PARAMETERS p = {0};
    p.hClient    = ctx->h_client;
    p.hObject    = ctx->h_subdevice;
    p.cmd        = NV2080_CTRL_CMD_GPU_GET_ID;
    p.params     = (uint64_t)(uintptr_t)&gp;
    p.paramsSize = (uint32_t)sizeof(gp);
    for (size_t i = 0; i < warmup; i++) {
        p.status = 0; ioctl(ctx->ctl_fd, RM_CTRL_CMD, &p);
    }
    for (size_t i = 0; i < s->n; i++) {
        p.status = 0;
        uint64_t t0 = now_ns();
        ioctl(ctx->ctl_fd, RM_CTRL_CMD, &p);
        s->samples[i] = now_ns() - t0;
    }
}

// ---------------------------------------------------------------------------
// Benchmark: NV_ESC_RM_CONTROL — get-attached-IDs on root client.
// 128-byte params, client-level simple passthrough.
// ---------------------------------------------------------------------------
static void bench_rmcontrol_attached_ids(rm_ctx_t *ctx, size_t warmup, stats_t *s) {
    NV0000_CTRL_GPU_GET_ATTACHED_IDS_PARAMS ap = {{0}};
    NVOS54_PARAMETERS p = {0};
    p.hClient    = ctx->h_client;
    p.hObject    = ctx->h_client;
    p.cmd        = NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS;
    p.params     = (uint64_t)(uintptr_t)&ap;
    p.paramsSize = (uint32_t)sizeof(ap);
    for (size_t i = 0; i < warmup; i++) {
        p.status = 0; ioctl(ctx->ctl_fd, RM_CTRL_CMD, &p);
    }
    for (size_t i = 0; i < s->n; i++) {
        p.status = 0;
        uint64_t t0 = now_ns();
        ioctl(ctx->ctl_fd, RM_CTRL_CMD, &p);
        s->samples[i] = now_ns() - t0;
    }
}

// ---------------------------------------------------------------------------
// Benchmark: NV_ESC_RM_ALLOC + NV_ESC_RM_FREE paired.
// Allocates and immediately frees an NV01_ROOT_CLIENT sub-client each
// iteration, exercising both the alloc and free paths without accumulating
// handles. This models the cleanup portion of each DMA chunk.
// ---------------------------------------------------------------------------
static void bench_alloc_free_client(rm_ctx_t *ctx, size_t warmup, stats_t *s) {
    for (size_t i = 0; i < warmup; i++) {
        uint32_t h = alloc_handle();
        NVOS64_PARAMETERS ap = {0};
        ap.hRoot = ctx->h_client; ap.hObjectParent = ctx->h_client;
        ap.hObjectNew = h; ap.hClass = NV01_ROOT_CLIENT;
        ioctl(ctx->ctl_fd, RM_ALLOC_CMD, &ap);
        NVOS00_PARAMETERS fp = {0};
        fp.hRoot = ctx->h_client; fp.hObjectParent = ctx->h_client; fp.hObjectOld = h;
        ioctl(ctx->ctl_fd, RM_FREE_CMD, &fp);
    }
    for (size_t i = 0; i < s->n; i++) {
        uint32_t h = alloc_handle();
        NVOS64_PARAMETERS ap = {0};
        ap.hRoot = ctx->h_client; ap.hObjectParent = ctx->h_client;
        ap.hObjectNew = h; ap.hClass = NV01_ROOT_CLIENT;
        NVOS00_PARAMETERS fp = {0};
        fp.hRoot = ctx->h_client; fp.hObjectParent = ctx->h_client; fp.hObjectOld = h;

        uint64_t t0 = now_ns();
        ioctl(ctx->ctl_fd, RM_ALLOC_CMD, &ap);
        ioctl(ctx->ctl_fd, RM_FREE_CMD,  &fp);
        s->samples[i] = now_ns() - t0;
    }
}

// ---------------------------------------------------------------------------
// Benchmark: NV_ESC_CHECK_VERSION_STR — cheapest frontend ioctl.
// Goes through feHandler (or feHandlerFast) directly to a simple
// frontendIoctlSimpleNoStatus call. Gives the floor on Sentry overhead.
// ---------------------------------------------------------------------------
static void bench_version_str(rm_ctx_t *ctx, size_t warmup, stats_t *s) {
    RMAPIVersion ver = {0};
    ver.cmd = 0x32;
    snprintf(ver.versionString, sizeof(ver.versionString), "%s", "580.95.05");
    unsigned long cmd = fe_cmd(NV_ESC_CHECK_VERSION_STR, (unsigned int)sizeof(ver));

    for (size_t i = 0; i < warmup; i++) ioctl(ctx->ctl_fd, cmd, &ver);
    for (size_t i = 0; i < s->n; i++) {
        uint64_t t0 = now_ns();
        ioctl(ctx->ctl_fd, cmd, &ver);
        s->samples[i] = now_ns() - t0;
    }
}

// ---------------------------------------------------------------------------
// Benchmark: raw getpid — pure syscall round-trip, the absolute floor.
// ---------------------------------------------------------------------------
static void bench_getpid(size_t warmup, stats_t *s) {
    for (size_t i = 0; i < warmup; i++) getpid();
    for (size_t i = 0; i < s->n; i++) {
        uint64_t t0 = now_ns();
        getpid();
        s->samples[i] = now_ns() - t0;
    }
}

// ---------------------------------------------------------------------------
// Benchmark: the DMA transfer hot-loop triplet, end-to-end.
//
// Replicates the repeating 3-ioctl-per-chunk pattern observed via strace
// during pageable cudaMemcpy:
//
//   1. NV_ESC_RM_CONTROL  subdevice → timer (DMA notify / setup)
//   2. NV_ESC_RM_CONTROL  subdevice → gpu_id (pin + kick DMA)
//   3. NV_ESC_RM_CONTROL  client   → attached_ids (RM_CONTROL wait/complete)
//
// We use three cheap, always-succeeding control commands that mirror the
// actual ioctl path (rmControl → rmControlSimple / versioned handler) as
// closely as possible without requiring an actual DMA engine.
// ---------------------------------------------------------------------------
static void bench_dma_triplet(rm_ctx_t *ctx, size_t chunks, size_t warmup,
                               stats_t *total_s, stats_t *per_ioctl_s,
                               size_t reps) {
    NV2080_CTRL_TIMER_GET_TIME_PARAMS        tp  = {0};
    NV2080_CTRL_GPU_GET_ID_PARAMS            gp  = {0};
    NV0000_CTRL_GPU_GET_ATTACHED_IDS_PARAMS  ap  = {{0}};

    unsigned long ctrl_cmd = fe_cmd(NV_ESC_RM_CONTROL, (unsigned int)sizeof(NVOS54_PARAMETERS));

    NVOS54_PARAMETERS p1 = {0};
    p1.hClient = ctx->h_client; p1.hObject = ctx->h_subdevice;
    p1.cmd = NV2080_CTRL_CMD_TIMER_GET_TIME;
    p1.params = (uint64_t)(uintptr_t)&tp; p1.paramsSize = (uint32_t)sizeof(tp);

    NVOS54_PARAMETERS p2 = {0};
    p2.hClient = ctx->h_client; p2.hObject = ctx->h_subdevice;
    p2.cmd = NV2080_CTRL_CMD_GPU_GET_ID;
    p2.params = (uint64_t)(uintptr_t)&gp; p2.paramsSize = (uint32_t)sizeof(gp);

    NVOS54_PARAMETERS p3 = {0};
    p3.hClient = ctx->h_client; p3.hObject = ctx->h_client;
    p3.cmd = NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS;
    p3.params = (uint64_t)(uintptr_t)&ap; p3.paramsSize = (uint32_t)sizeof(ap);

    // Warmup.
    for (size_t i = 0; i < warmup; i++) {
        p1.status = p2.status = p3.status = 0;
        ioctl(ctx->ctl_fd, RM_CTRL_CMD, &p1);
        ioctl(ctx->ctl_fd, RM_CTRL_CMD, &p2);
        ioctl(ctx->ctl_fd, RM_CTRL_CMD, &p3);
    }

    size_t total_ioctls = chunks * 3;
    for (size_t r = 0; r < reps; r++) {
        uint64_t t0 = now_ns();
        for (size_t c = 0; c < chunks; c++) {
            p1.status = p2.status = p3.status = 0;
            ioctl(ctx->ctl_fd, RM_CTRL_CMD, &p1);
            ioctl(ctx->ctl_fd, RM_CTRL_CMD, &p2);
            ioctl(ctx->ctl_fd, RM_CTRL_CMD, &p3);
        }
        uint64_t elapsed = now_ns() - t0;
        total_s->samples[r]     = elapsed;
        per_ioctl_s->samples[r] = elapsed / total_ioctls;
    }
}

// ---------------------------------------------------------------------------
// Sustained throughput — 10 000 calls, no per-sample overhead.
// ---------------------------------------------------------------------------
static void throughput_burst(rm_ctx_t *ctx, size_t n) {
    NV2080_CTRL_TIMER_GET_TIME_PARAMS tp = {0};
    NV2080_CTRL_GPU_GET_ID_PARAMS     gp = {0};
    RMAPIVersion ver = {0};
    ver.cmd = 0x32;
    snprintf(ver.versionString, sizeof(ver.versionString), "%s", "580.95.05");

    unsigned long ctrl_cmd  = fe_cmd(NV_ESC_RM_CONTROL,      (unsigned int)sizeof(NVOS54_PARAMETERS));
    unsigned long check_cmd = fe_cmd(NV_ESC_CHECK_VERSION_STR,(unsigned int)sizeof(ver));

    NVOS54_PARAMETERS timer_p = {0};
    timer_p.hClient = ctx->h_client; timer_p.hObject = ctx->h_subdevice;
    timer_p.cmd = NV2080_CTRL_CMD_TIMER_GET_TIME;
    timer_p.params = (uint64_t)(uintptr_t)&tp; timer_p.paramsSize = (uint32_t)sizeof(tp);

    NVOS54_PARAMETERS gpuid_p = {0};
    gpuid_p.hClient = ctx->h_client; gpuid_p.hObject = ctx->h_subdevice;
    gpuid_p.cmd = NV2080_CTRL_CMD_GPU_GET_ID;
    gpuid_p.params = (uint64_t)(uintptr_t)&gp; gpuid_p.paramsSize = (uint32_t)sizeof(gp);

    // getpid baseline.
    uint64_t t0 = now_ns();
    for (size_t i = 0; i < n; i++) getpid();
    uint64_t e_getpid = now_ns() - t0;

    // NV_ESC_CHECK_VERSION_STR — cheapest nvidia ioctl.
    t0 = now_ns();
    for (size_t i = 0; i < n; i++) ioctl(ctx->ctl_fd, check_cmd, &ver);
    uint64_t e_check = now_ns() - t0;

    // NV_ESC_RM_CONTROL timer — the canonical RM_CONTROL path.
    t0 = now_ns();
    for (size_t i = 0; i < n; i++) { timer_p.status=0; ioctl(ctx->ctl_fd, RM_CTRL_CMD, &timer_p); }
    uint64_t e_timer = now_ns() - t0;

    // NV_ESC_RM_CONTROL gpu_get_id — minimal params.
    t0 = now_ns();
    for (size_t i = 0; i < n; i++) { gpuid_p.status=0; ioctl(ctx->ctl_fd, RM_CTRL_CMD, &gpuid_p); }
    uint64_t e_gpuid = now_ns() - t0;

    printf("\n  %zu-call sustained burst (no per-sample timing overhead):\n", n);
    printf("  %-46s  per-call=%6.2f us  rate=%8.0f /s\n",
           "getpid (syscall floor)",
           ns2us(e_getpid) / (double)n,
           (double)n / (ns2ms(e_getpid) / 1e3));
    printf("  %-46s  per-call=%6.2f us  rate=%8.0f /s\n",
           "NV_ESC_CHECK_VERSION_STR (feHandlerFast floor)",
           ns2us(e_check) / (double)n,
           (double)n / (ns2ms(e_check) / 1e3));
    printf("  %-46s  per-call=%6.2f us  rate=%8.0f /s\n",
           "NV_ESC_RM_CONTROL timer_get_time",
           ns2us(e_timer) / (double)n,
           (double)n / (ns2ms(e_timer) / 1e3));
    printf("  %-46s  per-call=%6.2f us  rate=%8.0f /s\n",
           "NV_ESC_RM_CONTROL gpu_get_id",
           ns2us(e_gpuid) / (double)n,
           (double)n / (ns2ms(e_gpuid) / 1e3));
}

// ---------------------------------------------------------------------------
// Bandwidth projection table.
// ---------------------------------------------------------------------------
static void print_projection(double measured_p50_us, size_t total_ioctls,
                              double native_ms) {
    printf("\n  Bandwidth projection (%.0f-ioctl transfer, native baseline=%.1f ms):\n",
           (double)total_ioctls, native_ms);
    printf("  %-24s  %12s  %12s  %10s\n",
           "Per-ioctl overhead", "Total overhead", "Transfer time", "Bandwidth");
    printf("  %-24s  %12s  %12s  %10s\n",
           "------------------------", "------------", "------------", "----------");

    double overheads[] = {0, 0.5, 1, 2, 5, 10, 20, 30, 50, 75, 100};
    int n = (int)(sizeof(overheads)/sizeof(overheads[0]));
    int printed_measured = 0;
    for (int i = 0; i < n; i++) {
        if (!printed_measured && measured_p50_us < overheads[i]) {
            double oh_ms    = (double)total_ioctls * measured_p50_us / 1e3;
            double total_ms = native_ms + oh_ms;
            double bw_gbs   = 1.0 / (total_ms / 1e3);
            printf("  %6.2f us  \033[1m(MEASURED)\033[0m        %10.2f ms  %10.2f ms  %6.1f GB/s  ◄\n",
                   measured_p50_us, oh_ms, total_ms, bw_gbs);
            printed_measured = 1;
        }
        const char *tag = "";
        if (overheads[i] == 0)  tag = "(native)";
        if (overheads[i] == 10) tag = "(fast-path goal)";
        if (overheads[i] == 50) tag = "(typical gVisor)";
        double oh_ms    = (double)total_ioctls * overheads[i] / 1e3;
        double total_ms = native_ms + oh_ms;
        double bw_gbs   = 1.0 / (total_ms / 1e3);
        printf("  %6.1f us  %-16s  %10.2f ms  %10.2f ms  %6.1f GB/s\n",
               overheads[i], tag, oh_ms, total_ms, bw_gbs);
    }
    if (!printed_measured) {
        double oh_ms    = (double)total_ioctls * measured_p50_us / 1e3;
        double total_ms = native_ms + oh_ms;
        double bw_gbs   = 1.0 / (total_ms / 1e3);
        printf("  %6.2f us  \033[1m(MEASURED)\033[0m        %10.2f ms  %10.2f ms  %6.1f GB/s  ◄\n",
               measured_p50_us, oh_ms, total_ms, bw_gbs);
    }
}

// ---------------------------------------------------------------------------
// Detect runtime
// ---------------------------------------------------------------------------
static const char *detect_runtime(void) {
    struct stat st;
    // gVisor exposes this path inside the sandbox.
    if (stat("/proc/self/gvisor", &st) == 0) return "gVisor";
    FILE *f = fopen("/proc/self/cgroup", "r");
    if (f) {
        char line[256]; int found = 0;
        while (fgets(line, sizeof(line), f)) {
            if (strstr(line, "docker") || strstr(line, "containerd"))
                { found = 1; break; }
        }
        fclose(f);
        if (found) return "container (runc/docker)";
    }
    return "native";
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
static void usage(const char *prog) {
    printf("Usage: %s [options]\n\n"
           "  -n N   iterations for per-ioctl latency tests (default 5000)\n"
           "  -c C   chunks for DMA triplet simulation (default 256 = 1 GB at 4 MB/chunk)\n"
           "  -r R   repetitions for DMA triplet simulation (default 100)\n"
           "  -w W   warmup iterations (default 200)\n"
           "  -b B   burst size for throughput section (default 10000)\n"
           "  -h     show this help\n\n"
           "Run in three environments and compare:\n"
           "  1. native:                ./nvproxy_ioctl_bench\n"
           "  2. runc+GPU:              docker run --rm --runtime=runc --gpus all ...\n"
           "  3. gVisor stock+GPU:      docker run --rm --runtime=runsc --gpus all ...\n"
           "  4. gVisor fast-path+GPU:  docker run --rm --runtime=runsc-fastpath --gpus all ...\n",
           prog);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    size_t n_iters   = 5000;
    size_t n_chunks  = 256;
    size_t n_reps    = 100;
    size_t n_warmup  = 200;
    size_t n_burst   = 10000;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i+1 < argc) n_iters  = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i+1 < argc) n_chunks = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "-r") && i+1 < argc) n_reps   = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "-w") && i+1 < argc) n_warmup = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "-b") && i+1 < argc) n_burst  = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    printf("========================================================================\n");
    printf(" nvproxy real-ioctl latency benchmark  (NVIDIA A10G / driver 580.95.05)\n");
    printf("========================================================================\n");
    printf(" Runtime:     %s\n", detect_runtime());
    printf(" Iterations:  %zu per latency test\n", n_iters);
    printf(" Chunks:      %zu  (%.1f GB at 4 MB/chunk → %zu ioctls)\n",
           n_chunks, n_chunks * 4.0 / 1024.0, n_chunks * 3);
    printf(" Triplet reps:%zu\n", n_reps);
    printf(" Warmup:      %zu\n", n_warmup);
    struct timespec res; clock_getres(CLOCK_MONOTONIC, &res);
    printf(" Clock res:   %ld ns\n", res.tv_sec * 1000000000L + res.tv_nsec);
    printf("========================================================================\n\n");

    // Open the RM context.
    rm_ctx_t ctx;
    if (rm_ctx_open(&ctx) < 0) {
        fprintf(stderr,
                "\nFailed to initialise NVIDIA RM context.\n"
                "Is /dev/nvidia0 accessible? Are you in a --gpus all container?\n");
        return 1;
    }
    printf(" RM context: client=0x%x  device=0x%x  subdevice=0x%x  gpu_id=0x%x\n\n",
           ctx.h_client, ctx.h_device, ctx.h_subdevice, ctx.gpu_id);

    // -----------------------------------------------------------------------
    // Part 1: Per-ioctl latency — individual calls with nanosecond timing.
    // -----------------------------------------------------------------------
    printf("--- Part 1: Per-ioctl latency (real nvidia ioctls, %zu samples) ---\n\n",
           n_iters);

    stats_t s = alloc_stats(n_iters);

    // 1a. getpid — pure syscall floor.
    bench_getpid(n_warmup, &s);
    compute_stats(&s);
    print_stats_row("getpid (syscall floor)", &s);
    double getpid_p50 = s.p50_ns;

    // 1b. NV_ESC_CHECK_VERSION_STR — cheapest nvidia frontend ioctl.
    bench_version_str(&ctx, n_warmup, &s);
    compute_stats(&s);
    print_stats_row("NV_ESC_CHECK_VERSION_STR (feHandlerFast)", &s);

    // 1c. NV_ESC_RM_CONTROL timer_get_time — 16-byte params, subdevice.
    bench_rmcontrol_timer(&ctx, n_warmup, &s);
    compute_stats(&s);
    print_stats_row("NV_ESC_RM_CONTROL timer_get_time (16B)", &s);
    double rmcontrol_timer_p50 = s.p50_ns;

    // 1d. NV_ESC_RM_CONTROL gpu_get_id — 4-byte params, subdevice.
    bench_rmcontrol_gpu_id(&ctx, n_warmup, &s);
    compute_stats(&s);
    print_stats_row("NV_ESC_RM_CONTROL gpu_get_id (4B)", &s);

    // 1e. NV_ESC_RM_CONTROL attached_ids — 128-byte params, client.
    bench_rmcontrol_attached_ids(&ctx, n_warmup, &s);
    compute_stats(&s);
    print_stats_row("NV_ESC_RM_CONTROL get_attached_ids (128B)", &s);

    // 1f. NV_ESC_RM_ALLOC + NV_ESC_RM_FREE pair.
    bench_alloc_free_client(&ctx, n_warmup, &s);
    compute_stats(&s);
    print_stats_row("NV_ESC_RM_ALLOC + NV_ESC_RM_FREE (pair)", &s);

    printf("\n  getpid p50 = %.2f us — the minimum Sentry syscall round-trip.\n"
           "  Subtract from nvidia ioctl p50 to isolate nvproxy dispatch overhead.\n",
           getpid_p50 / 1e3);
    free_stats(&s);

    // -----------------------------------------------------------------------
    // Part 2: DMA transfer simulation — the hot triplet loop.
    // -----------------------------------------------------------------------
    size_t total_ioctls = n_chunks * 3;
    printf("\n--- Part 2: DMA transfer triplet simulation (%zu chunks × 3 = %zu ioctls) ---\n\n",
           n_chunks, total_ioctls);

    stats_t total_s     = alloc_stats(n_reps);
    stats_t per_ioctl_s = alloc_stats(n_reps);

    bench_dma_triplet(&ctx, n_chunks, n_warmup, &total_s, &per_ioctl_s, n_reps);
    compute_stats(&total_s);
    compute_stats(&per_ioctl_s);

    printf("  Total transfer time:\n");
    print_stats_row("  768-ioctl transfer (ms scale)", &total_s);
    printf("  Per-ioctl breakdown:\n");
    print_stats_row("  per-ioctl latency", &per_ioctl_s);

    printf("\n  Transfer p50 = %.3f ms  (%.2f us per ioctl)\n",
           ns2ms((uint64_t)total_s.p50_ns),
           per_ioctl_s.p50_ns / 1e3);

    // -----------------------------------------------------------------------
    // Part 3: Bandwidth projection.
    // -----------------------------------------------------------------------
    printf("\n--- Part 3: Bandwidth projection ---\n");
    // 57 ms is the measured 1 GB pageable transfer time under native runc
    // on this A10G instance (16+ GB/s PCIe bandwidth).
    print_projection(per_ioctl_s.p50_ns / 1e3, total_ioctls, 57.0);

    free_stats(&total_s);
    free_stats(&per_ioctl_s);

    // -----------------------------------------------------------------------
    // Part 4: Sustained throughput — pure rate, no per-sample timer cost.
    // -----------------------------------------------------------------------
    printf("\n--- Part 4: Sustained throughput (%zu-call bursts) ---", n_burst);
    throughput_burst(&ctx, n_burst);

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    printf("\n========================================================================\n");
    printf(" Key numbers to compare across runtimes\n");
    printf("========================================================================\n\n");
    printf("  %-46s  %s\n", "Metric", "Current run");
    printf("  %-46s  %s\n",
           "----------------------------------------------", "-----------");
    printf("  %-46s  %.2f us\n", "getpid p50 (syscall floor)",
           getpid_p50 / 1e3);
    printf("  %-46s  %.2f us\n",
           "NV_ESC_RM_CONTROL timer_get_time p50",
           rmcontrol_timer_p50 / 1e3);
    printf("  %-46s  %.2f us\n",
           "nvproxy overhead (rmcontrol - getpid p50)",
           (rmcontrol_timer_p50 - getpid_p50) / 1e3);
    printf("\n");
    printf("  Run identically under runc, runsc (stock), and runsc-fastpath to\n");
    printf("  compare. The nvproxy overhead row is the Sentry dispatch cost.\n");
    printf("  Multiply by 768 for total overhead on a 1 GB pageable transfer.\n");
    printf("========================================================================\n");

    rm_ctx_close(&ctx);
    return 0;
}