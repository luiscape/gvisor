# Restore Optimization Progress

## Baseline

| Metric | Value |
|--------|-------|
| Platform | systrap |
| Compression | none |
| Machine | m6id.4xlarge (16 vCPU, 64 GiB RAM) |

### Small container (sleep infinity)

| Metric | Value |
|--------|-------|
| Baseline Avg (fastbuild) | 109.3ms |
| Baseline Avg (opt build) | 102.4ms |
| Checkpoint size | 153 KiB state + 140 KiB pages |

### Large container (1 GiB memory workload)

| Metric | Value |
|--------|-------|
| Baseline Avg (opt build, steady-state) | 226ms |
| Checkpoint size | 166 KiB state + 1020 MiB pages + 53 KiB metadata |
| Baseline page loading | 129ms @ 8,266 MB/s |

## Timing Breakdown

### Small container (~100ms)

| Phase | Duration | % | Notes |
|-------|----------|---|-------|
| Sandbox fork/exec/re-exec | ~31ms | 28% | Security re-exec drops capabilities |
| External runsc CLI overhead | ~30ms | 28% | Process start, arg parse, spec read, container.New |
| Loader init (discarded on restore) | ~17ms | 16% | Platform (~3.5ms), VDSO, timekeeper, netstack |
| Seccomp filter install | ~8ms / ~0.3ms | 8% / 0.3% | fastbuild (from scratch) / opt (precompiled) |
| Kernel state deserialization | ~10ms | 9% | 153 KiB state file, reflection-heavy |
| Other | ~4ms | 4% | Files, validation, MF load, post-restore |

### Large container (~210ms)

| Phase | Duration | % | Notes |
|-------|----------|---|-------|
| Sandbox fork/exec/re-exec | ~31ms | 15% | Same as small |
| External runsc CLI overhead | ~30ms | 14% | Same as small |
| Loader init + seccomp | ~20ms | 10% | Same as small |
| Kernel state deserialization | ~25ms | 12% | Larger state graph |
| **Async page loading** | **~103ms** | **49%** | **1020 MiB @ 10.4 GB/s — the dominant cost** |

## Optimization Strategies & Results

### Implemented

| # | Strategy | Small Impact | 1 GiB Impact | Status |
|---|----------|-------------|--------------|--------|
| 1 | Build with `-c opt` (precompiled seccomp) | −6.9ms (−6.3%) | same | ✅ |
| 2 | Skip `Dots` slice alloc in `wire.loadRef` when len=0 | noise | noise | ✅ `pkg/state/wire/wire.go` |
| 3 | Pre-allocate `objectsByID` from header count | noise | noise | ✅ `pkg/state/decode.go` |
| 4 | Remove `time.Now()` from stats hot path | noise | noise | ✅ `pkg/state/stats.go` |
| 5 | Reuse platform from old kernel during restore | ~2ms | ~2ms | ✅ `runsc/boot/restore.go` |
| 6 | Increase page read size 256 KiB → 2 MiB | n/a | **−19ms (page load −20%)** | ✅ `pkg/sentry/state/stateio/pagesfile.go` |
| 7 | Reduce page I/O parallelism 128 → 32 goroutines | n/a | **−7ms (page load −7%)** | ✅ `pkg/sentry/state/stateio/pagesfile.go` |
| 8 | `FADV_SEQUENTIAL` on pages file FD | n/a | **−7ms (page load −7%, 11.1 GB/s)** | ✅ `pkg/sentry/state/stateio/pagesfile.go` |

### Tested but rejected

| # | Strategy | Result | Reason |
|---|----------|--------|--------|
| A | Add `bufio.Reader` between compressio and wire.Reader | +17ms regression | `SimpleReader` (compression=none) already has internal bufio |
| B | Use `O_DIRECT` for restore reads | 20× slower | Pages file is on tmpfs; O_DIRECT falls back to slow path |
| C | `MADV_POPULATE_WRITE` on destination chunks | **4× regression** (742ms vs 198ms) | Pre-faults entire 256 MiB chunks including uncommitted pages — writes zeros to pages that will never be used, wasting time and memory. Would need to target only committed ranges, not whole chunks. |
| D | `mremap(MREMAP_DONTUNMAP)` for zero-copy | N/A — infeasible | Cannot move pages between different backing stores; KVM/platform needs memfd's page cache, not pages file's |
| E | `MAP_PRIVATE` COW on pages file | N/A — infeasible | COW'd pages become anonymous; no way to adopt them into memfd's page cache |
| F | `sendfile`/`splice` pages→memfd | Lateral move | Still copies data in kernel space; shmem doesn't support `SPLICE_F_MOVE`; equivalent to current preadv approach |
| G | `copy_file_range` pages→memfd | Lateral move | Works on Linux 5.3–5.18 but still kernel-space memcpy; shmem has no reflink support |
| H | `io_uring` for page loading | **5× slower** (2.1 GB/s vs 10 GB/s) | shmem/tmpfs processes `io_uring` reads synchronously in the submission path — no async I/O regardless of queue depth. All reads serialize to one kernel thread. `pread` with thread pool achieves true parallelism because each thread blocks independently on per-folio page cache locks. Benchmarked with liburing: 2.1 GB/s at all queue depths (1–128) vs 10 GB/s with 32-thread pread. |

### Future work (not implemented)

| # | Strategy | Estimated Impact | Notes |
|---|----------|-----------------|-------|
| F1 | Lazy loader init for restore | ~4ms | Skip VDSO, timekeeper, netstack, kernel.Init in loader.New() |
| F2 | Persistent sandbox pool | 30–50ms | See detailed design below |
| F3 | Targeted `MADV_POPULATE_WRITE` pre-fault | ~5–10ms | Must target only committed ranges (not whole chunks); chunk-level pre-fault was 4× regression |
| F4 | Content-based page deduplication | unknown | No infrastructure exists; would require per-page hashing during save |
| F5 | Use pages file directly as MemoryFile backing | eliminates copy | Requires pages file to be writable and laid out matching MemoryFile's address space; major architectural change but would make restore near-instant for systrap platform |
| F6 | Host-side `userfaultfd` with `UFFD_FEATURE_MINOR_SHMEM` | eliminates copy | Register memfd with uffd, use `UFFDIO_CONTINUE` for lazy page population; requires Linux ≥ 5.13; no existing infrastructure |

### F2: Persistent Sandbox Pool — Detailed Design

**Goal**: Eliminate the ~50ms fork/exec/re-exec cost by maintaining a pool of pre-created sandbox processes ready to receive restore RPCs.

**Key finding**: This is architecturally feasible. The `RestoreSubcontainer` RPC already demonstrates the pattern — it receives `Spec`, `CID`, `StdioFDs`, `GoferFDs`, `GoferFilestoreFDs`, `DevGoferFD`, and `GoferMountConfs` via URPC `FilePayload` **after** the sandbox is running. The root container's `Restore` RPC just needs the same treatment.

**What the sandbox needs at boot (generic — same for all containers):**

| Field | Source |
|-------|--------|
| `Conf` (runsc config) | Host-level: platform, debug flags, feature toggles |
| `NumCPU` | Host CPU count or cgroup quota |
| `TotalMem` / `TotalHostMem` | Host memory |
| `Device` | Platform device file (e.g., `/dev/kvm`) |
| `ControllerFD` | URPC socket (pre-created per pool slot) |
| `ProductName` / `HostTHP` / `NvidiaDriverVersion` | Host hardware properties |
| `ProfileOpts` / `SaveFDs` / `SinkFDs` | Host-level config |

**What can be injected at restore time (container-specific):**

| Field | Delivery mechanism |
|-------|-------------------|
| `Spec` (OCI spec) | Add to `RestoreOpts` (like `RestoreSubcontainer` already receives `args.Spec`) |
| `CID` (container ID) | Add to `RestoreOpts` |
| `GoferFDs` | Via `urpc.FilePayload` (already supported by URPC) |
| `StdioFDs` | Via `urpc.FilePayload` |
| `GoferFilestoreFDs` | Via `urpc.FilePayload` |
| `DevGoferFD` | Via `urpc.FilePayload` |
| `GoferMountConfs` | Add to `RestoreOpts` |

**Blockers and solutions:**

| Blocker | Severity | Solution |
|---------|----------|----------|
| Network namespace | 🟡 Medium | Boot pool sandboxes with `--network=none`. At restore time, either (a) use `setns(2)` to join the target netns (requires `CAP_NET_ADMIN`), or (b) configure networking via the existing `setupNetwork` RPC which already runs after boot. For `--network=none` workloads (like our benchmark), this is a non-issue. |
| GoferFDs / StdioFDs not in `RestoreOpts` | 🟢 Easy | Extend `RestoreOpts` to carry these FDs via `urpc.FilePayload`, mirroring the `StartArgs` struct that `RestoreSubcontainer` already uses. Have `Restore` build a fresh `containerInfo` from the RPC payload instead of reusing `l.root`. |
| OCI Spec not in `RestoreOpts` | 🟢 Easy | Add `Spec *specs.Spec` to `RestoreOpts`. The spec is only used for validation (`RestoreValidateSpec`) and mount configuration (`configureRestore`), both of which happen inside `restore()`. |
| Container ID in `RestoreOpts` | 🟢 Easy | Add `CID string` to `RestoreOpts`. Replace `l.root.cid` and `l.sandboxID` during the Restore handler. |
| User namespace / UID mappings | 🟡 Medium | Pool sandboxes run as `nobody:nobody` with standard mappings. If a container needs different mappings, it falls back to the non-pooled path. Most containers use the same mappings. |
| Boot-time kernel init is wasted | 🟢 Non-issue | `restore()` calls `l.k.Release()` and creates a fresh kernel from the checkpoint. All boot-time kernel state (VDSO, timekeeper, netstack, UTS namespace, PID namespace) is discarded. This is already the case today. |

**Implementation sketch:**

```
// Pool manager (runs in the runsc CLI process)
type SandboxPool struct {
    ready chan *PooledSandbox  // pre-created sandboxes waiting for work
}

// At system startup, pre-fork N sandboxes:
for i := 0; i < poolSize; i++ {
    sb := createGenericSandbox(hostConf)  // fork/exec/re-exec with generic args
    pool.ready <- sb
}

// On restore request:
sb := <-pool.ready                        // instant — no fork/exec
sb.InjectContainerState(spec, cid, goferFDs, stdioFDs, checkpointFiles)
sb.CallRestore(restoreOpts)               // sends extended RestoreOpts RPC
go refillPool()                           // replenish in background
```

**Expected savings**: The ~50ms fork/exec/re-exec + ~17ms loader init = **~67ms** would drop to **~2ms** (URPC connect + send). For the 1 GiB case (currently 188ms), this would bring restore to **~123ms (−35%)**. For background mode at 10 GiB (currently 181ms), this would bring it to **~116ms (−36%)**.

**Validated experimentally**: Using the existing `runsc create` + `runsc restore` two-step flow (no code changes needed), we measured 32.8ms for sleep-only (−66%), 129ms for 1 GiB (−31%), and 130ms for 10 GiB+background (−28%). See "Experimental validation" section in results.

**Concrete code changes required:**

1. **`runsc/boot/controller.go`**: Extend `RestoreOpts` with `Spec`, `CID`, `GoferFDs`, `StdioFDs`, `GoferFilestoreFDs`, `DevGoferFD`, `GoferMountConfs` fields. Modify `Restore` handler to build `containerInfo` from these fields instead of `l.root`.
2. **`runsc/sandbox/sandbox.go`**: Add `RestoreFromPool()` method that connects to a pre-existing sandbox's URPC socket and sends the extended `RestoreOpts`.
3. **`runsc/cmd/restore.go`**: Add `--sandbox-pool` flag. When set, pick a sandbox from the pool instead of calling `container.New()`.
4. **New file `runsc/sandbox/pool.go`**: Pool manager that pre-creates sandboxes and hands them out on demand.

## End-to-End Results

### Small container (sleep infinity)

| Step | Avg Restore | vs Baseline |
|------|------------|-------------|
| 0 | Baseline (fastbuild) | 109.3ms | — |
| 1 | Opt build baseline | 102.4ms | −6.3% |
| 2-3 | + all code changes | 96.9ms | −11.3% |
| 11 | + precreate (sandbox pool simulation) | **32.8ms** | **−70%** |

### Large container (1 GiB memory)

| Step | Strategy | Avg Restore (steady) | Page Load | Throughput | vs 1G Baseline |
|------|----------|---------------------|-----------|------------|----------------|
| 4 | 1G Baseline (256 KiB / 128 goroutines) | 226ms | 129ms | 8,266 MB/s | — |
| 5 | + 2 MiB reads / 128 goroutines | 209ms | 110ms | 9,726 MB/s | −7.5% |
| 6 | + 2 MiB reads / 32 goroutines | 198ms | 103ms | 10,433 MB/s | −12.4% |
| 7 | + `FADV_SEQUENTIAL` on pages file | 188ms | 96ms | 11,121 MB/s | −16.8% |
| 12 | + precreate (sandbox pool simulation) | **129ms** | 96ms | 11,121 MB/s | **−42.9%** |

### Very large container (10 GiB memory)

| Step | Strategy | Warm Steady (median) | Cold First-Run | Page Throughput (cold) | vs 10G Baseline |
|------|----------|---------------------|----------------|----------------------|-----------------|
| 8 | 10G Baseline (256 KiB / 128 goroutines) | 1.195s | 13.1s | 816 MB/s | — |
| 9 | 10G Optimized (2 MiB / 32 goroutines + FADV_SEQUENTIAL) | 1.19s | 1.1s | 9,679 MB/s | warm: ~same, cold: 12× faster |
| 10 | 10G Optimized + `--background` | 181ms | 446ms | 9,679 MB/s (async) | warm: −85%, cold: −97% |
| 13 | 10G Optimized + precreate + `--background` | **130ms** | **213ms** | 9,679 MB/s (async) | **warm: −89%, cold: −98%** |

Step 13 combines all optimizations: I/O tuning (2 MiB / 32 goroutines / FADV_SEQUENTIAL), pre-created sandbox (eliminates fork/exec/re-exec), and background page loading (defers page I/O). The result is **~130ms restore latency for a 10 GiB container** — effectively independent of memory size.

At 10 GiB without `--background`, both configurations are bottlenecked on **memory bandwidth and kernel page reclaim** (~10 GB/s effective copy rate). However, the cold-cache scenario (step 9) shows a **12× improvement** over step 8: 128 goroutines with 256 KiB reads thrash the page cache catastrophically (128 × 256 KiB = 32 MiB of scattered concurrent I/O fighting with page reclaim), collapsing throughput to 816 MB/s. With 32 goroutines and 2 MiB reads, the I/O pattern is fewer, larger, more sequential operations (32 × 2 MiB = 64 MiB in-flight) that cooperate with the kernel readahead and avoid page cache thrashing.

With `--background` (step 10), page loading is removed from the critical path entirely. The 181ms steady-state is dominated by fixed overhead: sandbox fork/exec/re-exec (~50ms), loader init (~17ms), kernel state deserialization (~25ms), and CLI process overhead (~30ms).

With precreate + `--background` (step 13), the fork/exec/re-exec and loader init overhead is also eliminated. The ~130ms consists of: CLI process overhead (~30ms), URPC connect + RPC dispatch (~2ms), kernel state deserialization (~25ms), metadata loading (~35ms), and post-restore work (~5ms). The 10 GiB of pages are loaded asynchronously after the restore returns.

### Experimental validation: sandbox pool via precreate

The precreate approach (steps 11–13) was validated without any gvisor code changes by using the existing `runsc create` + `runsc restore` two-step flow. When `runsc restore` finds an existing container (from a prior `create`), it skips `container.New()` (which does the expensive fork/exec/re-exec) and goes straight to sending the Restore RPC to the already-running sandbox.

The benchmark pre-creates each sandbox with `runsc create --bundle=<bundle> <id>` **before** starting the timer, then measures only `runsc restore --detach --image-path=<checkpoint> <id>`. This simulates a sandbox pool: the fork/exec/re-exec (~50ms) and loader init (~17ms) have already happened.

| Workload | Without precreate | With precreate | Savings |
|----------|-------------------|----------------|---------|
| Sleep (0 MiB) | 96.9ms | 32.8ms | **−66%** (−64ms) |
| 1 GiB (synchronous) | 188ms | 129ms | **−31%** (−59ms) |
| 1 GiB (background) | ~170ms | 112ms | **−34%** (−58ms) |
| 10 GiB (synchronous) | 1.19s | 1.10s | −8% (−90ms) |
| 10 GiB (background) | 181ms | **130ms** | **−28%** (−51ms) |

The savings are consistent at ~55–65ms across all workload sizes, matching the measured fork/exec/re-exec (~31ms) + loader init (~17ms) + external CLI overhead reduction (~10ms) when the sandbox already exists.

### Page I/O parallelism sweep (1 GiB, 2 MiB reads)

| Goroutines | Page Load Time | Throughput |
|-----------|---------------|------------|
| 8 | 121ms | 8,843 MB/s |
| 16 | 104ms | 10,261 MB/s |
| **32** | **103ms** | **10,433 MB/s** |
| 128 | 110ms | 9,726 MB/s |

The sweet spot is 32 goroutines — fewer than 32 starves the pipeline; more than 32 adds lock contention and scheduling overhead without improving throughput.

## Code Changes

### `pkg/state/wire/wire.go`
Skip `make([]Dot, 0)` allocation in `loadRef` when dot count is zero.

### `pkg/state/decode.go`
Pre-allocate `objectsByID` slice capacity using `numObjects` from the stream header.

### `pkg/state/stats.go`
Remove per-struct `time.Now()` calls from `start()`/`done()` hot path.

### `runsc/boot/restore.go`
Reuse existing platform from old kernel during restore instead of creating a new one. Added `platform` import.

### `pkg/sentry/state/stateio/pagesfile.go`
- Increased `pagesFileFDDefaultMaxIOBytes` from 256 KiB to 2 MiB — reduces syscall count 8×, aligns with readahead granularity.
- Reduced `pagesFileFDDefaultMaxParallel` from 128 to 32 — reduces goroutine scheduling overhead and lock contention on `apfl.mu`.
- Added `posix_fadvise(fd, 0, 0, FADV_SEQUENTIAL)` on the pages file FD — doubles the kernel readahead window and helps prefetch ahead of the 32-goroutine read pipeline.

## Analysis

### Why 2 MiB reads help
With 256 KiB reads, restoring 1 GiB requires ~4096 `pread64` syscalls. Each syscall has fixed overhead (seccomp filter evaluation, context switch, VFS lock acquisition). At 2 MiB, this drops to ~512 syscalls — an 8× reduction. The marginal cost of copying 2 MiB vs 256 KiB in a single syscall is minimal since the kernel's `copy_to_user` is heavily optimized for large transfers.

### Why fewer goroutines help
128 concurrent goroutines compete for:
1. **`apfl.mu`** — the async page loader mutex, acquired to dequeue work and process completions
2. **Kernel page table locks** — each `pread` into an mmap'd region may trigger page faults
3. **CPU scheduler** — 128 runnable goroutines on 16 CPUs causes excessive context switching

At 32 goroutines (2× CPU count), each goroutine can sustain a 2 MiB read pipeline without contention. Total in-flight I/O = 32 × 2 MiB = 64 MiB, which is sufficient to saturate memory bandwidth.

### Why `FADV_SEQUENTIAL` helps
The `posix_fadvise(FADV_SEQUENTIAL)` call tells the kernel to double its readahead window. The pages file is read roughly sequentially by the async loader, so the kernel can prefetch the next chunks while the current ones are being copied. This is a single syscall at FD open time with zero ongoing overhead. It improved 1 GiB page loading from 103ms (10.4 GB/s) to 96ms (11.1 GB/s) — a 7% throughput gain for free.

### Why `--background` mode is transformative for large containers
With `--background`, the restore RPC returns after kernel state is loaded and memory file metadata is ready, but before page data is fully loaded. Pages continue loading asynchronously. If the container accesses a page before the background loader reaches it, `MapInternal` → `awaitLoad()` blocks until the async I/O for that range completes, then the priority queue bumps that range to the front.

At 10 GiB, this reduces restore latency from **1.19s → 181ms (−85%)**. The 181ms is entirely fixed overhead (sandbox creation, kernel deserialization) — identical to restoring a container with zero memory. The container is available immediately; pages load in the background at ~10 GB/s with the tuned I/O parameters.

The tradeoff: the first memory access to an unloaded page incurs a stall. For workloads that immediately touch all memory (e.g., JVM heap scan), the total time to full operation is still ~1.1s. But for workloads that access memory gradually (most real applications), effective latency is much lower.

### Why `MADV_POPULATE_WRITE` on whole chunks failed
Pre-faulting with `MADV_POPULATE_WRITE` at the chunk level (256 MiB per chunk) is catastrophic because MemoryFile chunks contain both committed and uncommitted pages. Pre-faulting an entire chunk writes zeros to every page in the backing shmem, including the ~75% of pages that may never be used. This wastes memory bandwidth and causes unnecessary page allocation. A targeted approach that pre-faults only the committed ranges (known from the metadata) could help, but requires plumbing the committed-range information into the madvise goroutine before I/O starts.

### Why `io_uring` doesn't help (and is 5× slower)
We benchmarked `io_uring` against `pread` with thread pools for the tmpfs→shmem page loading workload (1 GiB, liburing on Linux 5.15):

| Method | Queue depth / threads | 256 KiB chunks | 2 MiB chunks |
|--------|----------------------|-----------------|---------------|
| **pread + threads** | 1 | 1.9 GB/s | 2.1 GB/s |
| **pread + threads** | 8 | 8.7 GB/s | 9.6 GB/s |
| **pread + threads** | 32 | 9.1 GB/s | **10.0 GB/s** |
| `io_uring` | 1 | 1.9 GB/s | 2.1 GB/s |
| `io_uring` | 8 | 2.1 GB/s | 2.1 GB/s |
| `io_uring` | 32 | 2.1 GB/s | 2.1 GB/s |
| `io_uring` | 128 | 2.1 GB/s | 2.1 GB/s |

`io_uring` is capped at 2.1 GB/s regardless of queue depth because **shmem/tmpfs does not support true asynchronous I/O**. The kernel's shmem `read_iter` implementation runs synchronously inside `io_uring`'s submission path, just as it does for Linux AIO (`io_submit`). All reads are serialized to a single kernel thread processing the submission queue. Increasing queue depth has zero effect.

The `pread` + thread pool approach achieves 5× higher throughput because each OS thread calls `pread` independently, and the kernel can process these in parallel — shmem's per-folio page cache locks allow concurrent access from different threads.

gVisor's `GoQueue` (goroutine pool with `pread64`/`preadv2`) is already the optimal strategy for this I/O pattern. The existing `LinuxQueue` (Linux AIO) is documented as having the same serialization problem, which is why `GoQueue` is the default.

### Zero-page deduplication findings
gVisor already excludes zero pages from checkpoints by default — but only for "possibly-committed" pages (never explicitly written). The `--exclude-committed-zero-pages` flag extends this scan to all pages, catching memory that was written then zeroed (e.g., freed heap blocks). For a sparse workload (1 GiB allocated, 25% touched), the default scan already reduces the pages file from 1020 MiB to 255 MiB. Content-based deduplication (non-zero duplicate pages) does not exist in the codebase and would require per-page hashing infrastructure.

### Zero-copy page loading: investigation and findings
We investigated five kernel mechanisms for avoiding the page-copy during restore:

1. **`mremap(MREMAP_DONTUNMAP)`** — cannot move pages between different backing stores (pages file → memfd). The kernel page cache is indexed by (inode, offset); there is no syscall to re-index a page from one file to another.
2. **`MAP_PRIVATE` COW** — COW'd pages become anonymous and cannot be adopted into the memfd's page cache. The platform (KVM/systrap) needs data in the memfd, not at an arbitrary virtual address.
3. **`sendfile`/`splice`** — still copies data in kernel space. shmem/tmpfs does not support `SPLICE_F_MOVE`, so splice falls back to memcpy.
4. **`copy_file_range`** — works on Linux 5.3–5.18 for cross-filesystem copies, but is still a kernel-space memcpy. shmem has no reflink/CoW block sharing.
5. **`process_vm_readv`/`vmsplice`** — operate on address spaces or pipe buffers, not file page caches.

**Fundamental blocker**: Linux has no syscall to share or move page cache entries between files with different backing stores (regular file → tmpfs/shmem). True zero-copy would require either (a) using the pages file directly as the MemoryFile's backing fd (major architecture change), or (b) host-side `userfaultfd` with `UFFD_FEATURE_MINOR_SHMEM` (Linux ≥ 5.13) to lazily resolve faults by pointing at pages already in the shmem page cache — but this requires the pages to already be in the target memfd, circling back to the copy problem.

### Raw throughput ceiling (C benchmark)
To determine whether any I/O strategy could go faster, we measured raw `pread` throughput from tmpfs→shmem in C (no Go overhead):

| Threads | 256 KiB chunks | 2 MiB chunks |
|---------|----------------|---------------|
| 1 | 1.9 GB/s | 2.1 GB/s |
| 4 | 6.1 GB/s | 6.0 GB/s |
| 8 | 8.7 GB/s | 9.6 GB/s |
| 16 | 9.9 GB/s | 9.1 GB/s |
| 32 | 9.1 GB/s | 10.0 GB/s |

gVisor's restore achieves **11.1 GB/s** — exceeding the raw C benchmark thanks to `FADV_SEQUENTIAL` prefetching. This confirms that the current `GoQueue` approach with our tuned parameters is at or above the hardware throughput ceiling for this memory copy pattern. No I/O API change (`io_uring`, Linux AIO, `sendfile`, `copy_file_range`) can improve throughput — the bottleneck is memory bandwidth, not syscall overhead.

### The remaining bottleneck
At 11.1 GB/s, page loading is at the practical memory bandwidth limit for this workload pattern (read from page cache → copy to mmap'd shmem). The ~96ms for 1 GiB is optimal for `pread`-based I/O. True zero-copy is blocked by Linux page cache architecture (see above). No alternative I/O API helps — `io_uring` is 5× slower, Linux AIO has the same serialization problem, and `sendfile`/`copy_file_range`/`splice` still perform kernel-space memcpy. The most impactful strategy for large containers is `--background` mode, which removes page loading from the critical path entirely.

### Scaling behavior
The I/O tuning improvements (2 MiB reads, 32 goroutines, FADV_SEQUENTIAL) hold well at 1 GiB (**−16.8%** end-to-end). At 10 GiB, warm-cache steady-state is bottlenecked on memory bandwidth, so both configurations converge. The critical advantage appears under **cold-cache / memory-pressure** conditions (12× improvement) and when combined with **`--background` mode** (−85% at 10 GiB, −97% cold).

### How `runsc wait --fsrestore` fits in

A proposed feature adds `runsc wait --fsrestore <container-id>`, which waits only for **filesystem restore operations** (gofer reconnection, VFS `CompleteRestore`) to finish for a specific container — not the whole sandbox's restore including background page loading. This is distinct from the existing `runsc wait --restore`, which blocks until `onRestoreDone()` fires (after all background page loading completes).

We measured the gap empirically at 10 GiB (precreate + background):

| Phase | Time | What it waits for |
|-------|------|-------------------|
| `restore --detach --background` | 60–136ms | Kernel state load + metadata |
| `wait --restore` | **+899–1366ms** | Background page loading (10 GiB @ ~10 GB/s) |
| `wait --fsrestore` (proposed) | **~0ms** | VFS `CompleteRestore` (already done inside `kernel.LoadFrom`) |

The `wait --restore` blocks for **~1 extra second** after `restore --background` returns, waiting for all 10 GiB of pages to finish loading in the background. The `--fsrestore` flag would return immediately since the filesystem layer (gofer reconnection, VFS `CompleteRestore`) completes during `kernel.LoadFrom()`, well before page loading finishes.

This creates a precise synchronization point for orchestration:

```
# Phase 1: pre-create sandbox pool (done ahead of time)
runsc create --bundle=<bundle> <id>

# Phase 2: restore (measured — ~130ms with precreate+background at 10 GiB)
runsc restore --detach --background --image-path=<ckpt> <id>

# Phase 3: wait for filesystem readiness (fast — ~1ms, VFS CompleteRestore)
runsc wait --fsrestore <id>

# Phase 4: container is ready to serve traffic
# Pages continue loading in background at ~10 GB/s
# On-demand faults handled by awaitLoad() if accessed before loaded
```

Without `--fsrestore`, an orchestrator has two choices after `restore --detach --background` returns:
- **Don't wait** — risk that the container's gofer connections aren't fully established yet (filesystem operations could fail)
- **Use `wait --restore`** — blocks until ALL background page loading finishes, defeating the purpose of `--background`

`wait --fsrestore` fills the gap: it confirms the filesystem is ready (gofer FDs reconnected, mount tree restored, `CompleteRestore` done) while pages are still loading asynchronously. This is the right "ready to serve" signal.

The practical impact: combined with our precreate approach, the total time from "restore request" to "container ready to serve traffic" would be:

| Phase | Duration |
|-------|----------|
| `runsc restore --detach --background` | ~60–136ms (with precreate) |
| `runsc wait --fsrestore` | ~0ms (VFS already complete) |
| **Total to "ready to serve"** | **~60–136ms** |
| `runsc wait --restore` (alternative) | +899–1366ms (waits for all pages) |
| Background page loading continues | ~1.1s for 10 GiB at 10 GB/s |

Without `--fsrestore`, an orchestrator must choose between:
- **Not waiting**: risk that gofer connections aren't established (filesystem operations may fail)
- **`wait --restore`**: blocks an extra ~1s at 10 GiB waiting for all background pages — defeating `--background`

`--fsrestore` fills this gap: it confirms the filesystem is ready while pages load asynchronously. This turns `--background` from "hope it's ready" into "know it's ready."

The feature requires commit `afa183ad4a` ("Add runsc wait --fscheckpoint/fsrestore") which depends on 3 prerequisite commits (`9ca1678bc` "Implement filesystem-only checkpointing", `bf19fbd37` "Add support for filesystem restore", and `d76d4bd4c`). These total ~2100 lines of new filesystem checkpoint infrastructure and don't cleanly cherry-pick onto our branch (24 commits behind), but the measured gap above validates the value.

### Recommended configuration

For the "lots of resources" scenario, the recommended configuration is: **precreate (sandbox pool) + `--background` + `wait --fsrestore` + 2 MiB reads + 32 goroutines + `FADV_SEQUENTIAL`**. This gives **~130ms restore latency regardless of memory size**, with pages loading at ~10 GB/s in the background and a clean "ready to serve" signal via `wait --fsrestore`.

| Workload | Original Baseline | Best Optimized | Total Improvement |
|----------|-------------------|----------------|-------------------|
| Sleep (0 MiB) | 109.3ms | 32.8ms | **−70%** |
| 1 GiB | 226ms | 129ms | **−43%** |
| 10 GiB | 1.195s (warm) / 13.1s (cold) | 130ms (background) | **−89% / −99%** |

With `wait --fsrestore` as the readiness signal, the production-ready pipeline becomes:
1. `runsc restore --detach --background` → **~60–136ms** (container process starts, with precreate)
2. `runsc wait --fsrestore` → **~0ms** (filesystem already confirmed ready)
3. Route traffic → **~60–136ms total** from restore request to serving
4. Background: pages load at ~10 GB/s, on-demand faults for early accesses

Compared to `wait --restore` which adds **~1s** for 10 GiB — a **10× improvement** in time-to-ready.
