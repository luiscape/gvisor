# Access-order-traced checkpoint restore benchmark

This harness measures the benefit of writing a checkpoint's **pages file in
expected post-restore access order** rather than in MemoryFile offset order,
an idea adapted from *"Rethinking Process Snapshots for Near-Warm Serverless
Cold Starts"* (Spice/SHELF, OSDI '26).

## Mechanism under test

`runsc restore --pages-trace` performs a *profiling restore*:

1. Pages are only loaded from the pages file when demanded, and the order in
   which `MemoryFile` ranges are first demanded is recorded
   (`pgalloc.AsyncPagesFileLoadOpts.TraceAccess`).
2. A subsequent `runsc checkpoint` of that sandbox writes its pages file in
   the recorded access order — working set first, in first-touch order,
   followed by the remaining (cold) pages in offset order — and records the
   explicit range order in the pages metadata
   (`MemoryFileMetadataProto.pages_file_ranges`, metadata version 2).
3. Future restores of the resulting checkpoint load pages **sequentially in
   expected access order**, so the application rides just behind the
   background load sweep and rarely blocks on demand loads. While loading in
   access order, the sentry maps memory at fine granularity but
   opportunistically extends each mapping over contiguous already-loaded
   pages (`MemoryFile.AsyncLoadedSpan`), avoiding both long waits on
   never-accessed pages and repeated faults on loaded ones.

## Workload

`workload/main.go` emulates a snapshottable serverless function: it allocates
`TOTAL_MB` of memory as `REGION_KB`-sized mappings (bounding demand-map
granularity, like a fragmented runtime address space), fills every page, then
repeatedly touches a working set of `HOT_MB` worth of regions chosen and
traversed in a fixed pseudo-random order (access order ≠ address order). Each
pass appends a `PASS <n> dur_ms=...` line to a bind-mounted log; the harness
writes a trigger file to start a pass immediately after restore.

## Running

Build runsc and run the harness (root required):

```
mkdir -p bin
make copy TARGETS=runsc DESTINATION=bin/
sudo tools/mem-restore-bench/run.sh
```

The harness: (1) runs the workload and takes an initial checkpoint;
(2) creates a **baseline** image (restore + one pass + checkpoint) and a
**reordered** image (identical, but restored with `--pages-trace`); then
(3) for each variant, repeatedly drops the host page cache and measures
`runsc restore -detach --background --direct` followed by the first
workload pass.

Metrics per trial:

*   `restore_cmd_ms`: latency of the `runsc restore` command itself.
*   `time_to_pass_ms`: restore start → first post-restore pass complete
    (≈ time-to-first-request-served).
*   `pass_dur_ms`: duration of that first pass (demand-load stalls).

It also extracts the sentry's async page loader stats (`waiters waited X for
B bytes`) from the debug logs.

Knobs (env vars): `TOTAL_MB` (default 4096), `HOT_MB` (1024), `REGION_KB`
(256), `TRIALS` (5), `PLATFORM` (systrap), `WORK` (scratch dir), `RUNSC`.

## Representative results

c6id-class host, 48 cores, local NVMe (~3.1 GB/s sequential), cold page
cache, `--direct`; 16 GiB image, 4 GiB scattered working set:

| REGION_KB | metric          | baseline | reordered | speedup |
|-----------|-----------------|----------|-----------|---------|
| 256       | time_to_pass_ms | 6388     | 2504      | 2.55×   |
| 256       | pass_dur_ms     | 5591     | 1789      | 3.12×   |
| 1024      | time_to_pass_ms | 6423     | 1666      | 3.85×   |
| 1024      | pass_dur_ms     | 5639     | 870       | 6.48×   |

Async page loader stats for the first pass: baseline blocked ~5.3 s waiting
on ~7.6 GB of scattered demand loads; reordered blocked ~150 ms on <5 MB.
