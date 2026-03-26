# nvproxy ioctl overhead benchmark

## Background

When CUDA copies pageable (non-pinned) memory between host and device, the
NVIDIA driver uses a repeating 3-ioctl pattern per ~4 MB chunk:

| Step | ioctl | Target fd | Typical native latency | Purpose |
|------|-------|-----------|----------------------|---------|
| 1 | `0x49` | `/dev/nvidia0` | ~20 μs | DMA setup/notify |
| 2 | `0x21` | `/dev/nvidia0` | 100–1700 μs | Pin pages + kick DMA |
| 3 | `0x2b` | `/dev/nvidiactl` | 30–1500 μs | RM_CONTROL wait/complete |

For a 1 GB transfer at ~4 MB/chunk this produces **256 chunks × 3 = 768
ioctls**. Under `runc` the transfer completes in ~57 ms (17.6 GB/s).

In gVisor, every one of those ioctls takes a round-trip through the Sentry
(nvproxy intercepts, validates, then forwards to the host kernel). If that
round-trip adds ~50 μs per ioctl:

```
768 ioctls × 50 μs = 38.4 ms added overhead
57 ms + 38.4 ms ≈ 95 ms → ~10.5 GB/s
```

This matches gVisor's observed 8–10 GB/s for pageable transfers and explains
virtually the entire bandwidth gap versus native/runc.

## What this benchmark measures

This benchmark quantifies the **per-ioctl syscall overhead** without requiring
a GPU. It uses lightweight ioctls on universally-available file descriptors
(`/dev/null`, `eventfd`, `pipe`, `timerfd`) to measure the Sentry dispatch
cost, then projects the impact on pageable GPU memory transfers.

The benchmark does **not** measure:
- Actual GPU DMA latency (that's hardware-bound and identical across runtimes)
- CUDA library initialization overhead
- Pinned memory transfers (which bypass this ioctl pattern entirely)

It **does** measure:
- Raw per-ioctl round-trip latency across different fd types
- The 3-ioctl triplet pattern that mirrors the real CUDA transfer hot loop
- Sustained ioctl throughput under burst conditions
- The difference between native, runc, and gVisor syscall dispatch paths

## Building

```sh
# Dynamic build (standard):
make

# Static build (runs in any container, including scratch/busybox):
make static
```

Requirements: a C compiler (`gcc` or `clang`) and `libm`. No GPU or NVIDIA
drivers needed.

## Running

### Quick run

```sh
./ioctl_overhead_bench
```

### With more samples for higher confidence

```sh
./ioctl_overhead_bench --iterations 50000 --transfer-reps 500 --warmup 2000
```

### Full three-way comparison (native vs runc vs gVisor)

```sh
make run-compare
```

This requires Docker with the `runsc` runtime installed. It builds a static
binary and runs it natively, under `runc`, and under `runsc` in sequence.

### Manual comparison

```sh
# 1. Native
./ioctl_overhead_bench_static

# 2. runc
docker run --rm \
  -v $PWD/ioctl_overhead_bench_static:/bench:ro \
  --entrypoint /bench \
  busybox:latest

# 3. gVisor
docker run --rm --runtime=runsc \
  -v $PWD/ioctl_overhead_bench_static:/bench:ro \
  --entrypoint /bench \
  busybox:latest
```

## Command-line options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--iterations` | `-n` | 10000 | Per-ioctl latency sample count |
| `--chunks` | `-c` | 256 | Simulated 4 MB chunks (256 = 1 GB) |
| `--warmup` | `-w` | 500 | Warmup iterations before measuring |
| `--transfer-reps` | `-r` | 100 | Transfer simulation repetitions |
| `--verbose` | `-v` | off | Show per-test log-scale histograms |
| `--help` | `-h` | — | Print usage |

## Interpreting the output

The benchmark has six parts:

### Part 1: Single syscall/ioctl latency

Measures individual ioctl calls on different fd types. The `getpid` baseline
shows raw syscall entry/exit cost with no fd dispatch at all. Subtract the
`getpid` median from each ioctl median to isolate the fd-specific dispatch
overhead.

**What to look for:** Under gVisor, all ioctl medians will be significantly
higher than under native/runc. The *difference* is the Sentry overhead.

### Part 2: Triplet latency

Measures three back-to-back ioctls, which is what happens per chunk in a real
CUDA transfer. The "mixed triplet (3 diff fds)" test most closely models the
real pattern where each of the three ioctls hits a different fd/dispatch path.

**What to look for:** Divide the mixed triplet p50 by 3 to get the effective
per-ioctl cost in the hot loop.

### Part 3: Simulated pageable transfer

Runs the full 256-chunk × 3-ioctl pattern and reports total time. This is the
pure syscall overhead portion of a 1 GB pageable `cudaMemcpy`—the time the CPU
spends in ioctl dispatch, excluding actual DMA.

**What to look for:** This number plus 57 ms (native DMA time) gives the
projected total transfer time.

### Part 4: Projected pageable transfer bandwidth

Uses the measured overhead to project real-world transfer bandwidth. A
reference table shows what bandwidth you'd get at various per-ioctl overhead
levels, with your measured value marked.

**What to look for:**
- `0 μs` row = native performance (17.6 GB/s)
- `50 μs` row = typical gVisor overhead (~10.5 GB/s)
- `10 μs` row = fast-path optimization goal (~15.4 GB/s)
- `MEASURED` row = your actual measurement

### Part 5: Optimization projections

Projects the bandwidth impact of three proposed nvproxy optimizations:

| Strategy | Approach | Mechanism |
|----------|----------|-----------|
| **A** | Fast-path the hot triplet | Reduce per-ioctl Sentry overhead for the three DMA ioctls |
| **B** | Batch the triplet | Recognize the repeating 3-ioctl pattern and execute all three in one Sentry→host round-trip |
| **C** | ioctl passthrough | Bypass Sentry dispatch entirely for nvidia device fds |

### Part 6: Sustained ioctl throughput

Measures burst throughput (10,000 calls with no per-sample timing) to show the
sustained ioctl rate without measurement overhead inflating the numbers. This
gives the most accurate per-call cost.

**What to look for:** The "rate" column shows calls/second. Native Linux
typically achieves 2–5 million ioctls/s. gVisor will be lower; the ratio tells
you the overhead factor.

## Example results

### Native Linux (no container)

```
Per-ioctl p50: ~0.2–0.5 μs
Mixed triplet p50: ~1.0–1.5 μs
Projected 1 GB transfer: 57.4 ms → 17.4 GB/s
Sustained rate: ~3M ioctls/s
```

### runc (Docker default)

```
Per-ioctl p50: ~0.3–0.6 μs (essentially identical to native)
Mixed triplet p50: ~1.0–2.0 μs
Projected 1 GB transfer: 57.5 ms → 17.4 GB/s
Sustained rate: ~2.5M ioctls/s
```

### gVisor (runsc) — expected

```
Per-ioctl p50: ~30–80 μs (Sentry round-trip dominates)
Mixed triplet p50: ~100–250 μs
Projected 1 GB transfer: ~80–100 ms → ~10–12 GB/s
Sustained rate: ~15–30K ioctls/s
```

The gap between runc and gVisor numbers is the nvproxy optimization target.

## Limitations

1. **No GPU hardware interaction.** The ioctls measured here are lightweight
   kernel operations, not real NVIDIA driver calls. The benchmark measures
   *dispatch overhead*, not driver processing time. The real DMA ioctls
   (0x21, 0x2b) take 100–1700 μs on the GPU side, but that time is identical
   across runtimes—only the Sentry round-trip overhead differs.

2. **Different Sentry code paths.** nvproxy has specific handling for each
   NVIDIA ioctl number (parameter validation, fd translation, object tracking).
   The benchmark's `/dev/null` and `eventfd` ioctls exercise the generic Sentry
   ioctl path, which may be faster or slower than the nvproxy-specific path.
   The benchmark provides a *lower bound* on nvproxy overhead since nvproxy
   does additional work per ioctl.

3. **No seccomp filter overhead.** In production gVisor, seccomp-bpf filters
   add per-syscall overhead. This benchmark doesn't install seccomp filters.
   To include that component, run the benchmark inside an actual gVisor
   container rather than natively.

4. **Clock resolution.** The benchmark reports CLOCK_MONOTONIC resolution at
   startup. On most Linux systems this is 1 ns, but under some virtualization
   layers it may be coarser, affecting the accuracy of sub-microsecond
   measurements.

## Relationship to nvproxy code

The key nvproxy code paths exercised during a real pageable transfer:

- **Ioctl dispatch:** `pkg/sentry/devices/nvproxy/frontend.go` →
  `frontendFD.Ioctl()` — looks up handler by `IOC_NR(cmd)`
- **Host forwarding:** `pkg/sentry/devices/nvproxy/frontend_unsafe.go` →
  `frontendIoctlInvokeNoStatus()` — `unix.RawSyscall(SYS_IOCTL, ...)`
- **RM_CONTROL path:** `frontend.go` → `rmControl()` → `rmControlSimple()` →
  copies params in from guest, invokes host ioctl, copies params out
- **Seccomp filters:** `seccomp_filters.go` — allowlist for nvidia ioctl cmds

Each of these steps adds latency relative to a direct host ioctl. The total
per-ioctl path is roughly:

```
guest syscall → Sentry trap → fd lookup → ioctl handler dispatch →
param CopyIn → seccomp check → host RawSyscall → param CopyOut →
Sentry return → guest resume
```

This benchmark measures the end-to-end cost of an analogous path for
non-nvidia ioctls, giving a realistic estimate of the Sentry round-trip
component.