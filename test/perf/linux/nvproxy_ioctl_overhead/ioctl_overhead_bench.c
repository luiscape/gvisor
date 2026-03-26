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

// ioctl_overhead_bench measures per-ioctl syscall overhead to quantify
// the latency nvproxy's Sentry round-trip adds to pageable GPU memory
// transfers.
//
// Background:
//   CUDA pageable cudaMemcpy uses a repeating 3-ioctl pattern per ~4MB
//   chunk:
//     ioctl(dev, 0x49, ...)   — DMA setup/notify     (~20 us native)
//     ioctl(dev, 0x21, ...)   — pin pages + kick DMA  (~100-1700 us native)
//     ioctl(ctl, 0x2b, ...)   — RM_CONTROL wait       (~30-1500 us native)
//
//   For 1 GB at ~4 MB/chunk: 256 chunks x 3 ioctls = 768 transfer ioctls.
//   Under runc the total transfer takes ~57 ms (17.6 GB/s).
//   If nvproxy adds ~50 us per ioctl for the Sentry round-trip:
//     768 x 50 us = ~38 ms added => 57+38 = 95 ms => ~10.5 GB/s
//   This matches gVisor's actual 8-10 GB/s.
//
// This benchmark measures that per-ioctl overhead without a GPU by using
// lightweight ioctls on universally-available file descriptors (eventfd,
// /dev/null, pipes, timerfd). It then projects the bandwidth impact on
// pageable transfers.
//
// Usage:
//   gcc -O2 -o ioctl_overhead_bench ioctl_overhead_bench.c -lm
//   ./ioctl_overhead_bench [--iterations N] [--chunks C] [--warmup W]
//
// Run under different runtimes to compare:
//   # Native / runc:
//   ./ioctl_overhead_bench
//   # gVisor:
//   docker run --runtime=runsc -v $PWD:/bench /bench/ioctl_overhead_bench

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/timerfd.h>
#include <sys/types.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Configuration defaults
// ---------------------------------------------------------------------------

#define DEFAULT_ITERATIONS  10000   // per-ioctl latency sample count
#define DEFAULT_CHUNKS      256     // 1 GB / 4 MB per chunk
#define DEFAULT_WARMUP      500     // warmup iterations before measuring
#define IOCTLS_PER_CHUNK    3       // the hot triplet pattern
#define NATIVE_TRANSFER_MS  57.0    // baseline 1 GB native transfer time
#define TRANSFER_SIZE_GB    1.0

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline double ns_to_us(uint64_t ns) {
    return (double)ns / 1000.0;
}

static inline double ns_to_ms(uint64_t ns) {
    return (double)ns / 1000000.0;
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

typedef struct {
    uint64_t *samples;
    size_t    count;
    // Computed by compute_stats:
    double    mean_ns;
    double    median_ns;
    double    p50_ns;
    double    p90_ns;
    double    p99_ns;
    double    p999_ns;
    uint64_t  min_ns;
    uint64_t  max_ns;
    double    stddev_ns;
} stats_t;

static int cmp_uint64(const void *a, const void *b) {
    uint64_t va = *(const uint64_t *)a;
    uint64_t vb = *(const uint64_t *)b;
    return (va > vb) - (va < vb);
}

static void compute_stats(stats_t *s) {
    if (s->count == 0) return;

    qsort(s->samples, s->count, sizeof(uint64_t), cmp_uint64);

    s->min_ns = s->samples[0];
    s->max_ns = s->samples[s->count - 1];

    double sum = 0;
    for (size_t i = 0; i < s->count; i++) {
        sum += (double)s->samples[i];
    }
    s->mean_ns = sum / (double)s->count;

    double var_sum = 0;
    for (size_t i = 0; i < s->count; i++) {
        double d = (double)s->samples[i] - s->mean_ns;
        var_sum += d * d;
    }
    s->stddev_ns = sqrt(var_sum / (double)s->count);

    s->p50_ns  = (double)s->samples[(size_t)(s->count * 0.50)];
    s->median_ns = s->p50_ns;
    s->p90_ns  = (double)s->samples[(size_t)(s->count * 0.90)];
    s->p99_ns  = (double)s->samples[(size_t)(s->count * 0.99)];
    if (s->count >= 1000)
        s->p999_ns = (double)s->samples[(size_t)(s->count * 0.999)];
    else
        s->p999_ns = s->p99_ns;
}

static void print_stats(const char *label, stats_t *s) {
    printf("  %-32s  min=%7.2f  p50=%7.2f  p90=%7.2f  p99=%7.2f  max=%7.2f  mean=%7.2f +/- %.2f us\n",
           label,
           ns_to_us(s->min_ns),
           s->p50_ns / 1000.0,
           s->p90_ns / 1000.0,
           s->p99_ns / 1000.0,
           ns_to_us(s->max_ns),
           s->mean_ns / 1000.0,
           s->stddev_ns / 1000.0);
}

static stats_t alloc_stats(size_t count) {
    stats_t s;
    memset(&s, 0, sizeof(s));
    s.count = count;
    s.samples = (uint64_t *)calloc(count, sizeof(uint64_t));
    if (!s.samples) {
        fprintf(stderr, "Failed to allocate %zu samples\n", count);
        exit(1);
    }
    return s;
}

static void free_stats(stats_t *s) {
    free(s->samples);
    s->samples = NULL;
}

// ---------------------------------------------------------------------------
// Test: getpid() — pure syscall baseline (no fd, no ioctl)
// ---------------------------------------------------------------------------

static void bench_getpid(size_t warmup, stats_t *s) {
    for (size_t i = 0; i < warmup; i++) {
        syscall(SYS_getpid);
    }
    for (size_t i = 0; i < s->count; i++) {
        uint64_t t0 = now_ns();
        syscall(SYS_getpid);
        s->samples[i] = now_ns() - t0;
    }
}

// ---------------------------------------------------------------------------
// Test: ioctl(TCGETS) on /dev/null — lightweight, fails with ENOTTY
// This is the simplest ioctl path through the Sentry.
// ---------------------------------------------------------------------------

static void bench_devnull_tcgets(int fd, size_t warmup, stats_t *s) {
    struct termios t;
    for (size_t i = 0; i < warmup; i++) {
        ioctl(fd, TCGETS, &t);
    }
    for (size_t i = 0; i < s->count; i++) {
        uint64_t t0 = now_ns();
        ioctl(fd, TCGETS, &t);
        s->samples[i] = now_ns() - t0;
    }
}

// ---------------------------------------------------------------------------
// Test: ioctl(FIONREAD) on eventfd — returns number of readable bytes
// ---------------------------------------------------------------------------

static void bench_eventfd_fionread(int efd, size_t warmup, stats_t *s) {
    int nbytes;
    for (size_t i = 0; i < warmup; i++) {
        ioctl(efd, FIONREAD, &nbytes);
    }
    for (size_t i = 0; i < s->count; i++) {
        uint64_t t0 = now_ns();
        ioctl(efd, FIONREAD, &nbytes);
        s->samples[i] = now_ns() - t0;
    }
}

// ---------------------------------------------------------------------------
// Test: ioctl(FIONREAD) on pipe read end
// ---------------------------------------------------------------------------

static void bench_pipe_fionread(int pipefd, size_t warmup, stats_t *s) {
    int nbytes;
    for (size_t i = 0; i < warmup; i++) {
        ioctl(pipefd, FIONREAD, &nbytes);
    }
    for (size_t i = 0; i < s->count; i++) {
        uint64_t t0 = now_ns();
        ioctl(pipefd, FIONREAD, &nbytes);
        s->samples[i] = now_ns() - t0;
    }
}

// ---------------------------------------------------------------------------
// Test: ioctl(FIONREAD) on /dev/null
// ---------------------------------------------------------------------------

static void bench_devnull_fionread(int fd, size_t warmup, stats_t *s) {
    int nbytes;
    for (size_t i = 0; i < warmup; i++) {
        ioctl(fd, FIONREAD, &nbytes);
    }
    for (size_t i = 0; i < s->count; i++) {
        uint64_t t0 = now_ns();
        ioctl(fd, FIONREAD, &nbytes);
        s->samples[i] = now_ns() - t0;
    }
}

// ---------------------------------------------------------------------------
// Test: timerfd_gettime via syscall — measures a non-ioctl fd-based syscall
// ---------------------------------------------------------------------------

static void bench_timerfd_gettime(int tfd, size_t warmup, stats_t *s) {
    struct itimerspec its;
    for (size_t i = 0; i < warmup; i++) {
        timerfd_gettime(tfd, &its);
    }
    for (size_t i = 0; i < s->count; i++) {
        uint64_t t0 = now_ns();
        timerfd_gettime(tfd, &its);
        s->samples[i] = now_ns() - t0;
    }
}

// ---------------------------------------------------------------------------
// Test: eventfd write+read round-trip — measures fd syscall overhead with
// actual data movement (comparable to ioctl with payload)
// ---------------------------------------------------------------------------

static void bench_eventfd_write_read(int efd, size_t warmup, stats_t *s) {
    uint64_t val = 1;
    uint64_t rval;
    for (size_t i = 0; i < warmup; i++) {
        write(efd, &val, sizeof(val));
        read(efd, &rval, sizeof(rval));
    }
    for (size_t i = 0; i < s->count; i++) {
        uint64_t t0 = now_ns();
        write(efd, &val, sizeof(val));
        read(efd, &rval, sizeof(rval));
        s->samples[i] = now_ns() - t0;
    }
}

// ---------------------------------------------------------------------------
// Simulated DMA transfer pattern
//
// This simulates the exact 3-ioctl-per-chunk pattern that CUDA uses for
// pageable memory transfers. We use three different fd+ioctl combos to
// represent the three distinct ioctl codes:
//
//   "ioctl 0x49" (DMA setup)     => ioctl(devnull_fd, TCGETS)
//   "ioctl 0x21" (pin+kick DMA)  => ioctl(eventfd, FIONREAD)
//   "ioctl 0x2b" (RM_CONTROL)    => ioctl(pipe_fd, FIONREAD)
//
// This exercises:
//   - Multiple distinct fds per triplet (as the real driver uses)
//   - Different ioctl dispatch paths in the Sentry
//   - The sequential dependency pattern (each ioctl starts after the
//     previous completes)
// ---------------------------------------------------------------------------

typedef struct {
    uint64_t total_ns;
    uint64_t per_chunk_ns;
    double   total_ms;
    double   per_ioctl_us;
    size_t   num_chunks;
    size_t   total_ioctls;
} transfer_result_t;

static transfer_result_t simulate_transfer(
    int fd_setup,      // "dev" fd for ioctl 0x49  (DMA setup)
    int fd_kick,       // "dev" fd for ioctl 0x21  (pin+kick)
    int fd_ctrl,       // "ctl" fd for ioctl 0x2b  (RM_CONTROL)
    size_t num_chunks,
    size_t warmup
) {
    struct termios tios;
    int nbytes;
    transfer_result_t result;
    memset(&result, 0, sizeof(result));

    result.num_chunks = num_chunks;
    result.total_ioctls = num_chunks * IOCTLS_PER_CHUNK;

    // Warmup: run a few triplets to prime caches, TLBs, etc.
    for (size_t i = 0; i < warmup; i++) {
        ioctl(fd_setup, TCGETS, &tios);
        ioctl(fd_kick,  FIONREAD, &nbytes);
        ioctl(fd_ctrl,  FIONREAD, &nbytes);
    }

    // Timed run: execute the full transfer pattern.
    uint64_t t0 = now_ns();
    for (size_t chunk = 0; chunk < num_chunks; chunk++) {
        // Step 1: DMA setup/notify on device fd
        ioctl(fd_setup, TCGETS, &tios);
        // Step 2: Pin pages + kick DMA on device fd
        ioctl(fd_kick, FIONREAD, &nbytes);
        // Step 3: RM_CONTROL wait/complete on control fd
        ioctl(fd_ctrl, FIONREAD, &nbytes);
    }
    uint64_t t1 = now_ns();

    result.total_ns = t1 - t0;
    result.per_chunk_ns = result.total_ns / num_chunks;
    result.total_ms = ns_to_ms(result.total_ns);
    result.per_ioctl_us = ns_to_us(result.total_ns) / (double)result.total_ioctls;

    return result;
}

// Run the transfer simulation multiple times and report statistics.
static void bench_transfer_pattern(
    int fd_setup, int fd_kick, int fd_ctrl,
    size_t num_chunks, size_t warmup, size_t repetitions
) {
    stats_t total_stats = alloc_stats(repetitions);
    stats_t per_ioctl_stats = alloc_stats(repetitions);

    for (size_t rep = 0; rep < repetitions; rep++) {
        transfer_result_t r = simulate_transfer(
            fd_setup, fd_kick, fd_ctrl, num_chunks, warmup);
        total_stats.samples[rep] = r.total_ns;
        // Store per-ioctl as integer nanoseconds.
        per_ioctl_stats.samples[rep] = r.total_ns / r.total_ioctls;
    }

    compute_stats(&total_stats);
    compute_stats(&per_ioctl_stats);

    printf("\n");
    printf("  Transfer simulation: %zu chunks x %d ioctls = %zu total ioctls\n",
           num_chunks, IOCTLS_PER_CHUNK, num_chunks * IOCTLS_PER_CHUNK);
    printf("  Total time: min=%.3f  p50=%.3f  p99=%.3f  max=%.3f ms\n",
           ns_to_ms(total_stats.min_ns),
           total_stats.p50_ns / 1e6,
           total_stats.p99_ns / 1e6,
           ns_to_ms(total_stats.max_ns));
    printf("  Per-ioctl:  min=%.3f  p50=%.3f  p99=%.3f  max=%.3f us\n",
           ns_to_us(per_ioctl_stats.min_ns),
           per_ioctl_stats.p50_ns / 1e3,
           per_ioctl_stats.p99_ns / 1e3,
           ns_to_us(per_ioctl_stats.max_ns));

    free_stats(&total_stats);
    free_stats(&per_ioctl_stats);
}

// ---------------------------------------------------------------------------
// Homogeneous triplet benchmark: same ioctl 3 times (isolates per-ioctl cost)
// ---------------------------------------------------------------------------

static void bench_triplet_same_fd(int fd, unsigned long request,
                                  void *arg, size_t warmup, stats_t *s) {
    for (size_t i = 0; i < warmup; i++) {
        ioctl(fd, request, arg);
        ioctl(fd, request, arg);
        ioctl(fd, request, arg);
    }
    for (size_t i = 0; i < s->count; i++) {
        uint64_t t0 = now_ns();
        ioctl(fd, request, arg);
        ioctl(fd, request, arg);
        ioctl(fd, request, arg);
        s->samples[i] = now_ns() - t0;
    }
}

// ---------------------------------------------------------------------------
// Mixed triplet benchmark: 3 different fds, as the real transfer does
// ---------------------------------------------------------------------------

static void bench_triplet_mixed(
    int fd1, unsigned long req1, void *arg1,
    int fd2, unsigned long req2, void *arg2,
    int fd3, unsigned long req3, void *arg3,
    size_t warmup, stats_t *s
) {
    for (size_t i = 0; i < warmup; i++) {
        ioctl(fd1, req1, arg1);
        ioctl(fd2, req2, arg2);
        ioctl(fd3, req3, arg3);
    }
    for (size_t i = 0; i < s->count; i++) {
        uint64_t t0 = now_ns();
        ioctl(fd1, req1, arg1);
        ioctl(fd2, req2, arg2);
        ioctl(fd3, req3, arg3);
        s->samples[i] = now_ns() - t0;
    }
}

// ---------------------------------------------------------------------------
// Detect runtime environment
// ---------------------------------------------------------------------------

static const char *detect_runtime(void) {
    // Check for gVisor by looking at /sys/module/gvisor.
    struct stat st;
    if (stat("/proc/self/gvisor", &st) == 0) {
        return "gVisor (detected via /proc/self/gvisor)";
    }
    // Check cgroup for docker/containerd hints.
    FILE *f = fopen("/proc/1/cgroup", "r");
    if (f) {
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            if (strstr(line, "docker") || strstr(line, "containerd")) {
                fclose(f);
                return "container (docker/containerd)";
            }
        }
        fclose(f);
    }
    // Check if /dev/nvidia0 exists.
    if (stat("/dev/nvidia0", &st) == 0) {
        return "native (GPU present)";
    }
    return "native";
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

typedef struct {
    size_t iterations;
    size_t chunks;
    size_t warmup;
    size_t transfer_reps;
    int    verbose;
} bench_opts_t;

static bench_opts_t parse_args(int argc, char **argv) {
    bench_opts_t opts = {
        .iterations    = DEFAULT_ITERATIONS,
        .chunks        = DEFAULT_CHUNKS,
        .warmup        = DEFAULT_WARMUP,
        .transfer_reps = 100,
        .verbose       = 0,
    };

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--iterations") == 0 || strcmp(argv[i], "-n") == 0) {
            if (++i < argc) opts.iterations = (size_t)atol(argv[i]);
        } else if (strcmp(argv[i], "--chunks") == 0 || strcmp(argv[i], "-c") == 0) {
            if (++i < argc) opts.chunks = (size_t)atol(argv[i]);
        } else if (strcmp(argv[i], "--warmup") == 0 || strcmp(argv[i], "-w") == 0) {
            if (++i < argc) opts.warmup = (size_t)atol(argv[i]);
        } else if (strcmp(argv[i], "--transfer-reps") == 0 || strcmp(argv[i], "-r") == 0) {
            if (++i < argc) opts.transfer_reps = (size_t)atol(argv[i]);
        } else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            opts.verbose = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [OPTIONS]\n", argv[0]);
            printf("\nMeasures per-ioctl syscall overhead to quantify nvproxy Sentry latency.\n");
            printf("\nOptions:\n");
            printf("  -n, --iterations N      Per-ioctl latency samples (default: %d)\n", DEFAULT_ITERATIONS);
            printf("  -c, --chunks C          Simulated 4MB chunks for transfer (default: %d = 1GB)\n", DEFAULT_CHUNKS);
            printf("  -w, --warmup W          Warmup iterations (default: %d)\n", DEFAULT_WARMUP);
            printf("  -r, --transfer-reps R   Transfer simulation repetitions (default: 100)\n");
            printf("  -v, --verbose           Show per-sample histograms\n");
            printf("  -h, --help              Show this help\n");
            printf("\nRun under different runtimes to compare:\n");
            printf("  Native:  ./ioctl_overhead_bench\n");
            printf("  gVisor:  docker run --runtime=runsc -v $PWD:/bench /bench/ioctl_overhead_bench\n");
            exit(0);
        } else {
            fprintf(stderr, "Unknown option: %s (try --help)\n", argv[i]);
            exit(1);
        }
    }
    return opts;
}

// ---------------------------------------------------------------------------
// Histogram printer (optional with --verbose)
// ---------------------------------------------------------------------------

static void print_histogram(stats_t *s, size_t num_buckets) {
    if (s->count == 0) return;

    double lo = (double)s->min_ns;
    double hi = (double)s->max_ns;
    // Use log-scale buckets.
    double log_lo = log(lo > 0 ? lo : 1);
    double log_hi = log(hi > 0 ? hi : 1);
    double log_step = (log_hi - log_lo) / (double)num_buckets;
    if (log_step <= 0) log_step = 1;

    size_t *buckets = calloc(num_buckets, sizeof(size_t));
    size_t max_count = 0;
    for (size_t i = 0; i < s->count; i++) {
        double lv = log((double)(s->samples[i] > 0 ? s->samples[i] : 1));
        size_t b = (size_t)((lv - log_lo) / log_step);
        if (b >= num_buckets) b = num_buckets - 1;
        buckets[b]++;
        if (buckets[b] > max_count) max_count = buckets[b];
    }

    int bar_width = 40;
    for (size_t b = 0; b < num_buckets; b++) {
        if (buckets[b] == 0) continue;
        double edge_lo = exp(log_lo + log_step * (double)b);
        double edge_hi = exp(log_lo + log_step * (double)(b + 1));
        int filled = max_count > 0
            ? (int)((double)buckets[b] / (double)max_count * bar_width)
            : 0;
        printf("    [%7.1f - %7.1f us] %6zu |", edge_lo / 1000.0,
               edge_hi / 1000.0, buckets[b]);
        for (int j = 0; j < filled; j++) putchar('#');
        putchar('\n');
    }
    free(buckets);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    bench_opts_t opts = parse_args(argc, argv);

    printf("========================================================================\n");
    printf(" nvproxy ioctl overhead benchmark\n");
    printf("========================================================================\n");
    printf(" Runtime:     %s\n", detect_runtime());
    printf(" Iterations:  %zu per test\n", opts.iterations);
    printf(" Warmup:      %zu\n", opts.warmup);
    printf(" Chunks:      %zu (simulating %.1f GB at 4 MB/chunk)\n",
           opts.chunks, (double)opts.chunks * 4.0 / 1024.0);
    printf(" Transfer repetitions: %zu\n", opts.transfer_reps);

    // Print clock resolution for reference.
    struct timespec res;
    clock_getres(CLOCK_MONOTONIC, &res);
    printf(" Clock resolution: %ld ns\n", res.tv_sec * 1000000000L + res.tv_nsec);
    printf("========================================================================\n\n");

    // ------------------------------------------------------------------
    // Open file descriptors
    // ------------------------------------------------------------------

    int fd_devnull = open("/dev/null", O_RDWR);
    if (fd_devnull < 0) {
        perror("open /dev/null");
        return 1;
    }

    int fd_eventfd = eventfd(0, EFD_NONBLOCK);
    if (fd_eventfd < 0) {
        perror("eventfd");
        return 1;
    }

    int pipefd[2];
    if (pipe(pipefd) < 0) {
        perror("pipe");
        return 1;
    }

    int fd_timerfd = timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK);
    if (fd_timerfd < 0) {
        perror("timerfd_create");
        return 1;
    }

    // Open a second /dev/null to simulate having two distinct nvidia fds
    // (one for /dev/nvidia0, one for /dev/nvidiactl).
    int fd_devnull2 = open("/dev/null", O_RDWR);
    if (fd_devnull2 < 0) {
        perror("open /dev/null (2)");
        return 1;
    }

    printf("File descriptors: devnull=%d, eventfd=%d, pipe_r=%d, timerfd=%d, devnull2=%d\n\n",
           fd_devnull, fd_eventfd, pipefd[0], fd_timerfd, fd_devnull2);

    // ------------------------------------------------------------------
    // Part 1: Individual ioctl latency (single calls)
    // ------------------------------------------------------------------

    printf("--- Part 1: Single syscall / ioctl latency (us) ---\n\n");

    {
        stats_t s = alloc_stats(opts.iterations);

        // 1a. getpid baseline (pure syscall, no fd)
        bench_getpid(opts.warmup, &s);
        compute_stats(&s);
        print_stats("getpid (syscall baseline)", &s);
        double getpid_median_us = s.p50_ns / 1000.0;
        if (opts.verbose) print_histogram(&s, 20);

        // 1b. ioctl(TCGETS) on /dev/null
        bench_devnull_tcgets(fd_devnull, opts.warmup, &s);
        compute_stats(&s);
        print_stats("ioctl(devnull, TCGETS)", &s);
        if (opts.verbose) print_histogram(&s, 20);

        // 1c. ioctl(FIONREAD) on /dev/null
        bench_devnull_fionread(fd_devnull, opts.warmup, &s);
        compute_stats(&s);
        print_stats("ioctl(devnull, FIONREAD)", &s);
        if (opts.verbose) print_histogram(&s, 20);

        // 1d. ioctl(FIONREAD) on eventfd
        bench_eventfd_fionread(fd_eventfd, opts.warmup, &s);
        compute_stats(&s);
        print_stats("ioctl(eventfd, FIONREAD)", &s);
        if (opts.verbose) print_histogram(&s, 20);

        // 1e. ioctl(FIONREAD) on pipe
        bench_pipe_fionread(pipefd[0], opts.warmup, &s);
        compute_stats(&s);
        print_stats("ioctl(pipe, FIONREAD)", &s);
        if (opts.verbose) print_histogram(&s, 20);

        // 1f. timerfd_gettime
        bench_timerfd_gettime(fd_timerfd, opts.warmup, &s);
        compute_stats(&s);
        print_stats("timerfd_gettime", &s);
        if (opts.verbose) print_histogram(&s, 20);

        // 1g. eventfd write+read pair
        bench_eventfd_write_read(fd_eventfd, opts.warmup, &s);
        compute_stats(&s);
        print_stats("eventfd write+read pair", &s);
        if (opts.verbose) print_histogram(&s, 20);

        printf("\n  [getpid median = %.2f us — subtract this from ioctl medians\n"
               "   to estimate ioctl-specific dispatch overhead]\n", getpid_median_us);

        free_stats(&s);
    }

    // ------------------------------------------------------------------
    // Part 2: Triplet latency (3 ioctls back-to-back)
    // ------------------------------------------------------------------

    printf("\n--- Part 2: Triplet latency (3 back-to-back ioctls, us) ---\n\n");

    {
        stats_t s = alloc_stats(opts.iterations);
        struct termios tios;
        int nbytes;

        // 2a. Homogeneous triplet: devnull TCGETS x3
        bench_triplet_same_fd(fd_devnull, TCGETS, &tios, opts.warmup, &s);
        compute_stats(&s);
        print_stats("3x ioctl(devnull, TCGETS)", &s);
        if (opts.verbose) print_histogram(&s, 20);

        // 2b. Homogeneous triplet: eventfd FIONREAD x3
        bench_triplet_same_fd(fd_eventfd, FIONREAD, &nbytes, opts.warmup, &s);
        compute_stats(&s);
        print_stats("3x ioctl(eventfd, FIONREAD)", &s);
        if (opts.verbose) print_histogram(&s, 20);

        // 2c. Mixed triplet: devnull TCGETS + eventfd FIONREAD + pipe FIONREAD
        //     This most closely models the real CUDA transfer pattern.
        bench_triplet_mixed(
            fd_devnull,  TCGETS,   &tios,
            fd_eventfd,  FIONREAD, &nbytes,
            pipefd[0],   FIONREAD, &nbytes,
            opts.warmup, &s);
        compute_stats(&s);
        print_stats("mixed triplet (3 diff fds)", &s);
        double mixed_triplet_p50_us = s.p50_ns / 1000.0;
        if (opts.verbose) print_histogram(&s, 20);

        // 2d. Mixed triplet variant: two devnull fds + pipe
        //     Models the real pattern where 0x49 and 0x21 go to the same
        //     nvidia device fd, and 0x2b goes to the control fd.
        bench_triplet_mixed(
            fd_devnull,  TCGETS,   &tios,
            fd_devnull,  FIONREAD, &nbytes,
            fd_devnull2, FIONREAD, &nbytes,
            opts.warmup, &s);
        compute_stats(&s);
        print_stats("mixed triplet (2 devnull fds)", &s);
        if (opts.verbose) print_histogram(&s, 20);

        printf("\n  [mixed triplet p50 = %.2f us => per-ioctl = %.2f us]\n",
               mixed_triplet_p50_us, mixed_triplet_p50_us / 3.0);

        free_stats(&s);
    }

    // ------------------------------------------------------------------
    // Part 3: Full transfer simulation
    // ------------------------------------------------------------------

    printf("\n--- Part 3: Simulated pageable transfer (%zu chunks x 3 ioctls) ---\n",
           opts.chunks);

    bench_transfer_pattern(fd_devnull, fd_eventfd, pipefd[0],
                           opts.chunks, opts.warmup, opts.transfer_reps);

    // Also test with the "2 devnull + pipe" pattern that more closely
    // matches the real fd topology.
    printf("\n  [variant: 2x devnull + pipe fd topology]\n");
    bench_transfer_pattern(fd_devnull, fd_devnull, pipefd[0],
                           opts.chunks, opts.warmup, opts.transfer_reps);

    // ------------------------------------------------------------------
    // Part 4: Bandwidth projection
    // ------------------------------------------------------------------

    printf("\n--- Part 4: Projected pageable transfer bandwidth ---\n\n");

    {
        // Run one more simulation to get the median overhead.
        size_t total_ioctls = opts.chunks * IOCTLS_PER_CHUNK;
        stats_t s = alloc_stats(opts.transfer_reps);

        for (size_t rep = 0; rep < opts.transfer_reps; rep++) {
            transfer_result_t r = simulate_transfer(
                fd_devnull, fd_eventfd, pipefd[0],
                opts.chunks, opts.warmup);
            s.samples[rep] = r.total_ns / r.total_ioctls;  // per-ioctl ns
        }
        compute_stats(&s);

        double per_ioctl_p50_us = s.p50_ns / 1000.0;
        double measured_overhead_ms = (double)total_ioctls * per_ioctl_p50_us / 1000.0;

        printf("  Measured per-ioctl overhead (p50): %.2f us\n", per_ioctl_p50_us);
        printf("  Total ioctls for 1 GB transfer:   %zu\n", total_ioctls);
        printf("  Total ioctl overhead:              %.2f ms\n", measured_overhead_ms);
        printf("  Native transfer time (reference):  %.1f ms\n", NATIVE_TRANSFER_MS);
        printf("\n");

        // Projection table for various overhead scenarios.
        printf("  %-20s  %-14s  %-14s  %-10s\n",
               "Per-ioctl overhead", "Total overhead", "Transfer time", "Bandwidth");
        printf("  %-20s  %-14s  %-14s  %-10s\n",
               "--------------------", "--------------", "--------------", "----------");

        double overheads_us[] = {0, 1, 2, 5, 10, 20, 30, 50, 75, 100};
        size_t num_overheads = sizeof(overheads_us) / sizeof(overheads_us[0]);

        for (size_t i = 0; i < num_overheads; i++) {
            double oh_us = overheads_us[i];
            double oh_ms = (double)total_ioctls * oh_us / 1000.0;
            double total_ms = NATIVE_TRANSFER_MS + oh_ms;
            double bw = TRANSFER_SIZE_GB / (total_ms / 1000.0);
            const char *marker = "";
            // Mark the row closest to our measurement.
            if (i > 0 && per_ioctl_p50_us >= overheads_us[i - 1] &&
                per_ioctl_p50_us < overheads_us[i]) {
                // Insert measured row before this one.
                double moh_ms = (double)total_ioctls * per_ioctl_p50_us / 1000.0;
                double mtotal = NATIVE_TRANSFER_MS + moh_ms;
                double mbw = TRANSFER_SIZE_GB / (mtotal / 1000.0);
                printf("  %6.1f us  (MEASURED)  %10.2f ms    %10.2f ms    %6.1f GB/s  <---\n",
                       per_ioctl_p50_us, moh_ms, mtotal, mbw);
            }
            if (oh_us == 0) marker = "(native)";
            else if (oh_us == 50) marker = "(typical gVisor)";
            else if (oh_us == 10) marker = "(fast-path goal)";
            printf("  %6.1f us  %-12s  %10.2f ms    %10.2f ms    %6.1f GB/s\n",
                   oh_us, marker, oh_ms, total_ms, bw);
        }

        // If measured overhead is >= 100, print it at the end.
        if (per_ioctl_p50_us >= overheads_us[num_overheads - 1]) {
            double moh_ms = (double)total_ioctls * per_ioctl_p50_us / 1000.0;
            double mtotal = NATIVE_TRANSFER_MS + moh_ms;
            double mbw = TRANSFER_SIZE_GB / (mtotal / 1000.0);
            printf("  %6.1f us  (MEASURED)  %10.2f ms    %10.2f ms    %6.1f GB/s  <---\n",
                   per_ioctl_p50_us, moh_ms, mtotal, mbw);
        }

        free_stats(&s);
    }

    // ------------------------------------------------------------------
    // Part 5: Batching projection
    // ------------------------------------------------------------------

    printf("\n--- Part 5: Optimization projections ---\n\n");

    {
        size_t total_ioctls = opts.chunks * IOCTLS_PER_CHUNK;

        // Measure the single ioctl overhead one more time to get a stable value.
        stats_t s = alloc_stats(opts.transfer_reps);
        for (size_t rep = 0; rep < opts.transfer_reps; rep++) {
            transfer_result_t r = simulate_transfer(
                fd_devnull, fd_eventfd, pipefd[0],
                opts.chunks, opts.warmup);
            s.samples[rep] = r.total_ns / r.total_ioctls;
        }
        compute_stats(&s);
        double measured_us = s.p50_ns / 1000.0;

        printf("  Measured per-ioctl overhead: %.2f us (p50)\n\n", measured_us);

        printf("  Strategy A: Fast-path the hot triplet (reduce per-ioctl overhead)\n");
        printf("  %-30s  %-14s  %-10s\n", "Target per-ioctl", "Total overhead", "Bandwidth");
        double targets_a[] = {1, 2, 5, 10, 20};
        for (size_t i = 0; i < sizeof(targets_a)/sizeof(targets_a[0]); i++) {
            double oh_ms = total_ioctls * targets_a[i] / 1000.0;
            double total_ms = NATIVE_TRANSFER_MS + oh_ms;
            double bw = TRANSFER_SIZE_GB / (total_ms / 1000.0);
            printf("  %6.0f us                        %10.2f ms    %6.1f GB/s\n",
                   targets_a[i], oh_ms, bw);
        }

        printf("\n  Strategy B: Batch triplet into 1 Sentry transition (3x fewer transitions)\n");
        printf("  %-30s  %-14s  %-10s\n", "Per-transition overhead", "Total overhead", "Bandwidth");
        double targets_b[] = {10, 20, 30, 50};
        size_t transitions = opts.chunks;  // one per chunk instead of 3
        for (size_t i = 0; i < sizeof(targets_b)/sizeof(targets_b[0]); i++) {
            double oh_ms = (double)transitions * targets_b[i] / 1000.0;
            double total_ms = NATIVE_TRANSFER_MS + oh_ms;
            double bw = TRANSFER_SIZE_GB / (total_ms / 1000.0);
            printf("  %6.0f us                        %10.2f ms    %6.1f GB/s\n",
                   targets_b[i], oh_ms, bw);
        }

        printf("\n  Strategy C: ioctl passthrough (bypass Sentry dispatch entirely)\n");
        double oh_ms = 0;
        double total_ms = NATIVE_TRANSFER_MS + oh_ms;
        double bw = TRANSFER_SIZE_GB / (total_ms / 1000.0);
        printf("  Per-ioctl overhead: ~0 us => %.1f GB/s (matches native)\n", bw);

        // With measured overhead:
        printf("\n  With current measured overhead (%.1f us):\n", measured_us);
        double cur_oh_ms = total_ioctls * measured_us / 1000.0;
        double cur_total = NATIVE_TRANSFER_MS + cur_oh_ms;
        double cur_bw = TRANSFER_SIZE_GB / (cur_total / 1000.0);
        printf("    Projected: %.2f ms overhead => %.2f ms total => %.1f GB/s\n",
               cur_oh_ms, cur_total, cur_bw);

        free_stats(&s);
    }

    // ------------------------------------------------------------------
    // Part 6: Rapid-fire ioctl throughput
    // ------------------------------------------------------------------

    printf("\n--- Part 6: Sustained ioctl throughput ---\n\n");

    {
        struct termios tios;
        int nbytes;
        size_t burst = 10000;

        // Warmup.
        for (size_t i = 0; i < opts.warmup; i++) {
            ioctl(fd_devnull, TCGETS, &tios);
        }

        // Measure burst of ioctls — no per-sample timing overhead.
        uint64_t t0 = now_ns();
        for (size_t i = 0; i < burst; i++) {
            ioctl(fd_devnull, TCGETS, &tios);
        }
        uint64_t elapsed_devnull = now_ns() - t0;

        t0 = now_ns();
        for (size_t i = 0; i < burst; i++) {
            ioctl(fd_eventfd, FIONREAD, &nbytes);
        }
        uint64_t elapsed_eventfd = now_ns() - t0;

        // Mixed triplet burst.
        size_t triplet_burst = burst / 3;
        t0 = now_ns();
        for (size_t i = 0; i < triplet_burst; i++) {
            ioctl(fd_devnull, TCGETS, &tios);
            ioctl(fd_eventfd, FIONREAD, &nbytes);
            ioctl(pipefd[0], FIONREAD, &nbytes);
        }
        uint64_t elapsed_mixed = now_ns() - t0;

        // getpid burst for comparison.
        t0 = now_ns();
        for (size_t i = 0; i < burst; i++) {
            syscall(SYS_getpid);
        }
        uint64_t elapsed_getpid = now_ns() - t0;

        printf("  %zu-call burst (no per-sample timing):\n", burst);
        printf("  %-32s  total=%.3f ms  per-call=%.2f us  rate=%.0f calls/s\n",
               "getpid",
               ns_to_ms(elapsed_getpid),
               ns_to_us(elapsed_getpid) / burst,
               burst / (ns_to_ms(elapsed_getpid) / 1000.0));
        printf("  %-32s  total=%.3f ms  per-call=%.2f us  rate=%.0f calls/s\n",
               "ioctl(devnull, TCGETS)",
               ns_to_ms(elapsed_devnull),
               ns_to_us(elapsed_devnull) / burst,
               burst / (ns_to_ms(elapsed_devnull) / 1000.0));
        printf("  %-32s  total=%.3f ms  per-call=%.2f us  rate=%.0f calls/s\n",
               "ioctl(eventfd, FIONREAD)",
               ns_to_ms(elapsed_eventfd),
               ns_to_us(elapsed_eventfd) / burst,
               burst / (ns_to_ms(elapsed_eventfd) / 1000.0));
        printf("  %-32s  total=%.3f ms  per-ioctl=%.2f us  rate=%.0f triplets/s\n",
               "mixed triplet",
               ns_to_ms(elapsed_mixed),
               ns_to_us(elapsed_mixed) / (triplet_burst * 3),
               triplet_burst / (ns_to_ms(elapsed_mixed) / 1000.0));
    }

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------

    printf("\n========================================================================\n");
    printf(" Summary\n");
    printf("========================================================================\n\n");
    printf("  To compare nvproxy overhead, run this benchmark in three environments:\n\n");
    printf("    1. Native Linux:         ./ioctl_overhead_bench\n");
    printf("    2. runc (Docker):        docker run --rm -v $PWD:/bench ubuntu /bench/ioctl_overhead_bench\n");
    printf("    3. gVisor (runsc):       docker run --rm --runtime=runsc -v $PWD:/bench ubuntu /bench/ioctl_overhead_bench\n");
    printf("\n");
    printf("  The difference in per-ioctl latency between (2) and (3) is the\n");
    printf("  nvproxy Sentry overhead. Multiply by 768 to get the total overhead\n");
    printf("  added to a 1 GB pageable cudaMemcpy.\n");
    printf("\n");
    printf("  Key metrics to compare across runtimes:\n");
    printf("    - Part 1: single ioctl p50 latency\n");
    printf("    - Part 2: mixed triplet p50 latency\n");
    printf("    - Part 4: projected bandwidth\n");
    printf("    - Part 6: sustained throughput rate\n");
    printf("========================================================================\n");

    // Cleanup.
    close(fd_devnull);
    close(fd_devnull2);
    close(fd_eventfd);
    close(pipefd[0]);
    close(pipefd[1]);
    close(fd_timerfd);

    return 0;
}