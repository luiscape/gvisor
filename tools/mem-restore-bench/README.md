# Memory restore benchmark (access-order-traced pages files)

This harness benchmarks gVisor checkpoint restore with **access-order-traced
pages files**, a mechanism inspired by the SHELF snapshot format from
*"Rethinking Process Snapshots for Near-Warm Serverless Cold Starts"* (Spice,
OSDI '26): instead of storing checkpointed pages in MemoryFile-offset order,
pages are stored in the order in which the application is expected to touch
them after restore, so that both demand loading and sequential background
loading deliver pages approximately in first-touch order.

## Mechanism under test

1. `runsc restore --pages-trace` restores a checkpoint in *profiling* mode:
   pages are only loaded from the pages file when demanded, and the order in
   which they are first demanded is recorded
   (`pgalloc.AsyncPagesFileLoadOpts.TraceAccess`).
2. A subsequent `runsc checkpoint` of that sandbox writes its pages file in
   the recorded access order — traced pages first, in first-touch order,
   then all remaining pages in offset order — and records the layout in the
   pages metadata (`MemoryFileMetadataProto.pages_file_ranges`, version 2).
3. Restores of the resulting checkpoint load the pages file sequentially,
   which now *is* the expected access order; demand faults wait only for the
   pages they need (fine-grained mapping), and already-loaded spans are
   opportunistically mapped to avoid later faults.

## Workload

`workload/main.go` emulates a snapshottable serverless function: it allocates
`TOTAL_MB` of memory as separate `REGION_KB`-sized mappings and fills every
page, then repeatedly touches a working set of `HOT_MB` worth of regions,
chosen pseudo-randomly from across the whole allocation and traversed in a
fixed scrambled order (access order is unrelated to address order, as in real
snapshots). Each pass appends a `PASS <n> dur_ms=...` line to a bind-mounted
log; a trigger file lets the harness request a pass immediately after
restore.

## Running

Build runsc and run the harness (as root; it drops page caches):

```
make copy TARGETS=runsc DESTINATION=bin/
sudo tools/mem-restore-bench/run.sh
```

Knobs (environment variables): `TOTAL_MB` (default 4096), `HOT_MB` (1024),
`REGION_KB` (256), `TRIALS` (5), `WORK` (scratch dir, defaults under `/data`
if present), `RUNSC`, `PLATFORM` (systrap).

The harness:

1. Runs the workload and takes an initial checkpoint (`img-orig`).
2. Restores it *without* tracing, runs one pass, and checkpoints again
   (`img-base`: same contents, offset-ordered pages file).
3. Restores it *with* `--pages-trace`, runs one pass, and checkpoints again
   (`img-reordered`: same contents, access-ordered pages file).
4. Restores each image `TRIALS` times with a cold page cache
   (`--direct --background`), measuring the `runsc restore` command latency,
   the time from restore start until the first post-restore pass completes,
   and that pass's duration.

## Example results

c6in-class EC2 host, local NVMe (~3.4 GB/s sequential), 16 GiB checkpoint,
4 GiB scattered working set, 5 trials:

| metric (mean) | baseline | reordered | speedup |
|---|---|---|---|
| `restore` cmd latency | 519 ms | 488 ms | 1.06× |
| time to first pass | 6388 ms | 2504 ms | **2.55×** |
| first pass duration | 5591 ms | 1789 ms | **3.12×** |

With `REGION_KB=1024` (larger mappings, closer to language-runtime heaps):
time to first pass 6423 ms → 1666 ms (**3.9×**), first pass duration
5639 ms → 870 ms (**6.5×**).

The sentry's async page loader stats (in the boot debug logs) show the
underlying effect: per restore, baseline trials blocked ~5.3 s waiting on
~7.6 GB of scattered demand reads, while reordered trials blocked ~150 ms
waiting on well under 1 MB — the application rides just behind the sequential
load of the access-ordered pages file.
