# Multi-GPU CUDA checkpoint/restore: status and reviewer reply

Status as of 2026-10-11. Branch `luis/multicast-xgpu-upstream`, merged with
upstream master at `2834ad887` (`dc93d60ef`). This is a dev-only file: drop
it before opening upstream PRs.

## Summary

-   **Third interposer review:** every item and every optional item is
    addressed. The reply is at the end of this file. Before the engineering
    brief work started, the branch passed every test on H100 ×8 with driver
    610.57.04: the shim suite, all 15 engine cells and the 3 cells without
    `CAP_SYS_PTRACE`.
-   **Engineering brief, P0 tier:** all seven tasks are done and pushed. On
    the final binary (`ea443eb56`), the shim suite, all 15 engine cells, the
    3 cells without `CAP_SYS_PTRACE`, and Qwen2.5-72B on all 8 GPUs pass.
-   **Round after P0 (CPU only, this machine has no GPU):** the review of
    `45bd3f7b0` found three problems that hit processes that never
    checkpoint, since `/etc/ld.so.preload` puts the shim in every process:
    `dlsym` re-anchoring, missed cudart resolvers, and a glibc floor of 2.34.
    All three are fixed, with the small sentry items, and every check that
    runs on a CPU passes (below). GPU validation of the new binary is the
    next round.
-   **GPU round on the merged binary (`dc93d60ef`): everything passes**
    (below): the full shim suite, the 15 engine cells, the 3 without
    `CAP_SYS_PTRACE`, the three Qwen2.5-72B cells, the tokens/s A/B, and a
    new largest-model run, Qwen3-235B-A22B (bf16, 490 GB image) at TP=8 on
    both engines, with throughput under load before and after restore.
-   **Scope for the first merge:** validated on R610 only. runsc enables the
    interposer and the job on R610 or newer (an R615-opt-in flag was added
    this round and then removed), so keeping unvalidated releases off is up
    to the scheduler. R615 comes after the first version is merged.

## Growable, indexed tables (2026-10-12)

The interposer's three 4096-entry tables are replaced by tables that grow in
chunks of 1024 entries that never move (up to 1M each), so the cap is gone
without invalidating the indices and pointers that calls dropping `g_lock`
keep. Wrappers no longer scan: hash indexes on the application value and the
driver handle, per-object lists of mappings and binds, and an interval treap
for lookups by address. A process that never uses CUDA allocates no table
memory (the `.so`'s bss went from 2.2 MB to 31 KB).

Cost per call with N live allocations (fake driver, so shim cost only):

| Call | Old, N = 3,000 | Old, N = 20,000 | New, N = 3,000 | New, N = 100,000 |
| :--- | ---: | ---: | ---: | ---: |
| `cuMemCreate` | 2.2 µs | 6.2 µs | 0.28 µs | 0.30 µs |
| `cuMemMap` | 4.5 µs | 9.0 µs | 1.1 µs | 0.47 µs |
| `cuMemRelease` | 4.4 µs | 10.0 µs | 0.18 µs | 0.21 µs |
| retain + release | 3.7 µs | 11.8 µs | 0.10 µs | 0.37 µs |
| `cuMemUnmap` of 64 mappings | 477 µs | n/a | 25 µs | 26 µs |
| Checkpoint gate | accepts | **refuses** (overflow) | accepts | accepts |

Tests:

-   New `run.sh tables` (no GPU): a fake `libcuda` (`test/fakecuda.c`) that
    rejects any handle or address it did not issue, and `test/tables.c`:
    random VMM and multicast traffic from 3-16 threads with suspend/resume
    cycles running concurrently, with the shim built with
    `MCSHIM_TABLE_CHECKS` so every index is checked each time `g_lock` is
    released, plus a 100k-allocation scale run. About 9,000 cycles across
    seven seeds pass, also under UBSan.
-   New `run.sh torch-expandable` (GPU, under runsc): PyTorch with
    `expandable_segments:True` holds 12 GiB of small tensors, then runs a
    full suspend/resume; contents must survive. The old shim refuses it
    ("object table overflow"); the new one passes.
-   The full shim suite passes on the new shim (all 19 tests), and so do
    `sglang_tp4_symm`, `vllm_tp4_72b_xgpu` and `sglang_tp8_72b` with a
    release build embedding it.

## GPU round on the merged binary (`dc93d60ef`, 2026-10-10)

H100 ×8, 610.57.04. Every item in "For the GPU round" below, on one binary,
with no Xid and the fabric `Completed/Success` after every phase.

-   **Shim suite:** all 17 pass (abi gate mc refcount refuse mapwait silent
    preload glibc rtres ipc orphan deadline reason optout torch-kernel
    torch-symm). `glibc` loads on ubuntu:20.04, debian:11, rockylinux:8 and
    centos:7. `rtres` now also covers a vendored copy
    (`libcudart-1a2b3c4d.so.12`, set by `patchelf`) next to
    `libcudart.so.13`: each caller reaches its own runtime. `refuse fabric`
    skipped (no IMEX channel).
-   **Engine cells:** all 15 pass, and so do the 3 without `CAP_SYS_PTRACE`
    (`vllm_tp4`, `sglang_tp4_symm`, `sglang_tp4_nvls_xgpu`). Timings match
    the last round (`vllm_tp2` 44x, `sglang_tp4_symm` 18x, `sglang_tp8`
    9.5x).
-   **Qwen2.5-72B:** `vllm_tp8_72b` 9.6x (checkpoint 60.9 s, 172 GB),
    `sglang_tp8_72b` 5.6x (86.9 s, 169 GB), `vllm_tp4_72b_xgpu` 14.8x onto
    GPUs 4-7.
-   **Tokens/s A/B** (Qwen2.5-72B, vLLM TP=8, `--enforce-eager`, 512
    prompts, 1024 in / 256 out, 64 concurrent, two boots each): output
    816 / 805 tok/s without the interposer, 808 / 807 with it. The
    difference (-0.4%) is smaller than the spread between boots (1.4%).

### Largest model: Qwen3-235B-A22B-Instruct-2507, bf16, TP=8

The largest open model that fits 8 × 80 GB in bf16 (438 GiB of weights, MoE).
One cycle per engine: load test, sleep, checkpoint, restore, wake, identical
output at temperature 0, the same load test again. Then the same image is
restored a second time after `drop_caches`, as a restore from storage would
be (the first restore reads an image the checkpoint just left in the page
cache). Both **PASS**.

| Step | vLLM 0.29.0 | SGLang 0.5.20 |
| :--- | :--- | :--- |
| Cold boot | 1049 s | 582 s |
| Load before checkpoint (output tok/s, TPOT mean) | 2130, 27.2 ms | 2142, 24.3 ms |
| Checkpoint (engine asleep) | 134 s, image 490 GB | 174 s, image 496 GB |
| `cuda-checkpoint` lock and checkpoint | 45 s | 90 s |
| Sentry memory save | 86 s (6.1 GB/s) | 79 s (6.7 GB/s) |
| `runsc restore` returns (warm / cold cache) | 48 s / 79 s | 135 s / 82 s |
| `cuda-checkpoint` restore toggle (warm / cold) | 87 s / 62 s | 72 s / 38 s |
| First inference after restore (warm / cold) | 142 s / 148 s | 230 s / 141 s |
| Speedup over cold boot (warm / cold) | 7.3x / 7.0x | 2.5x / 4.1x |
| Load after restore (output tok/s, TPOT mean) | 2278, 25.3 ms | 2153, 24.2 ms |
| Processes managed by the interposer | 10 | 10 |
| Suspend / resume | 0.8 s / 1.5 s | 0.2 s / 0.8 s |

Notes:

-   Throughput after restore is at least what it was before (vLLM's +7% is
    likely warm-up), so the rebuilt multicast and all-reduce paths run at
    full speed.
-   SGLang's warm restore was slower than its cold one. That was memory
    reclaim on NUMA node 0 caused by the harness's `--membind=0`, not
    SGLang or the interposer: see the NUMA rerun below.
-   The cold boots read weights from the page cache, so the speedups are
    conservative. vLLM's boot includes `torch.compile` and graph capture.
-   Image and page loading now dominate (about 500 GB at 6-11 GB/s), along
    with the sequential `cuda-checkpoint` calls (P2.5).
-   Harness: `~/bench/bench.sh` gained `LOAD`, `LOAD_ONLY`, `COLD_RESTORE`,
    `GPU_UTIL`, `MAX_LEN`, `gpu.csv`/`host.csv` sampling and the sentry's
    timing lines in `summary.txt`. Logs: `/data/bench/overnight/` and
    `/data/bench/runs/on-*`.

### NUMA rerun: the slow SGLang restore was reclaim on node 0

`bench.sh` runs the sandbox, `runsc checkpoint` and `runsc restore` under
`numactl --cpunodebind=N --membind=N`. Node 0 has 1 TB; one 235B SGLang cycle
needs more than that on it: about 470 GB of pinned CPU weight backup while
asleep, the 496 GB image in the page cache, then a new 500 GB sentry. Strict
binding cannot spill to node 1 (also 1 TB, idle), so the kernel reclaims on
node 0 instead, in the middle of the checkpoint and the restore.

Three SGLang TP=8 cycles of Qwen3-235B, settings as in the table above
(`/data/bench/numa/run.sh`). "Pre-filled" reads the vLLM image into node 0's
page cache first, which is the state the original run started in. All three
**PASS** with identical output and the same throughput before and after
restore (2127-2169 output tok/s).

| | No pinning, pre-filled | Pinned, pre-filled (original) | Pinned, clean |
| :--- | :--- | :--- | :--- |
| `cuda-checkpoint` lock and checkpoint | 49 s | 94 s | 126 s |
| Checkpoint total | 138 s | 173 s | 215 s |
| Page loading on restore | 18 s (29 GB/s) | 121 s (4.4 GB/s) | 71 s (7.5 GB/s) |
| `cuda-checkpoint` restore toggle | 36 s | 73 s | 59 s |
| First inference after restore | **65 s** | 219 s | 163 s |
| Cold-cache restore: page loading / first inference | 122 s / 170 s | 81 s / 144 s | 81 s / 155 s |
| Direct reclaim over the cycle (pages scanned, alloc stalls) | 0, 4 | 38.0 M, 579 k | 2.3 M, 36 k |
| File refaults over the cycle | 1.2 M | 106 M | 58 M |
| Cold boot | 773 s | 580 s | 583 s |

What this shows:

-   **Confirmed.** Without strict binding there is no direct reclaim, the
    warm restore reads the image from the page cache at 29 GB/s, and the
    first inference comes 3.4x sooner (65 s against 219 s; 11.8x faster than
    a cold boot). The original run's 230 s reproduces (219 s) when pinned
    with the same pre-fill.
-   **It is not only the restore.** `cuda-checkpoint`'s lock and checkpoint
    (49 s against 94-126 s) and its restore toggle (36 s against 59-73 s)
    slow down under the same pressure: both move hundreds of GB between the
    GPUs and host memory. So the earlier "SGLang's `cuda-checkpoint` is
    twice vLLM's" (90 s against 45 s) was also the harness.
-   **Pinned, clean still reclaims.** Even from empty caches, node 0
    overflows during one cycle (kswapd scanned 199 M pages). Dropping caches
    is not a fix.
-   **But removing pinning altogether costs elsewhere.** Unpinned, the cold
    boot is 33% slower and the cold-cache restore reads at 4.4 GB/s instead
    of 6.5 GB/s, likely because the threads then run on both sockets, far
    from GPUs 0-3 and the memory. Not verified.
-   **Not a gVisor problem,** but a deployment one: anything that pins a
    sandbox with a large host footprint to one node with strict binding
    will hit this. The likely best setting is `--cpunodebind` with a
    preferred (not strict) memory policy, or interleaving across both nodes
    for TP=8 (its GPUs span both). Untested.
-   **Earlier numbers.** The other cells and the 72B runs have far smaller
    host footprints (under 175 GB) and showed no such effect; the vLLM 235B
    cycle fit in node 0 by a small margin. The SGLang 235B row in the table
    above (2.5x) should be read as the pinned-harness number.

### NUMA policy rerun: interleave removes the pathology

CPUs stay on the GPUs' node (`--cpunodebind=0`); only the memory policy
changes (`bench.sh`'s new `NUMA_MEM`). Every 235B cycle starts pre-filled
with the other engine's image in node 0's page cache, the state that made
the original SGLang restore slow (`/data/bench/numa2/run.sh`). One cycle per
cell; repeats of the same cell vary by about 10%. All **PASS**, with
identical output and the same throughput after restore as before.

First inference after restore, warm cache / cold cache (s):

| Policy | SGLang | vLLM |
| :--- | :--- | :--- |
| `--membind=0` (old default) | 219 / 144 | 139 / 144 |
| No `numactl` at all | 65 / 170 | - |
| `--preferred=0` | 83 / 143 | 187 / 168 |
| `--interleave=all` | **75 / 134** | 156 / 145 |

Breakdown (s; direct reclaim in pages scanned over the cycle):

| | SGLang membind | SGLang preferred | SGLang interleave | vLLM membind | vLLM preferred | vLLM interleave |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cuda-checkpoint` lock and checkpoint | 94 | 87 | 53 | 52 | 55 | 46 |
| Checkpoint total | 173 | 181 | 150 | 142 | 152 | 141 |
| Warm page loading | 121 | 19 | 24 | 52 | 43 | 18 |
| Warm `cuda-checkpoint` restore toggle | 73 | 41 | 40 | 79 | 137 | 131 |
| Cold page loading | 81 | 79 | 88 | 79 | 79 | 87 |
| Cold `cuda-checkpoint` restore toggle | 40 | 41 | 35 | 57 | 81 | 52 |
| Direct reclaim | 38 M | 0 | 0 | 0.96 M | 0 | 0.83 M |
| Cold boot | 580 | 579 | 599 | 1049 | 1043 | 1088 |

(SGLang membind is `numa-pin-prefill` from the previous section: same
binary, node and pre-fill.)

Findings:

-   **SGLang: both policies fix it, interleave is best.** Warm restore drops
    from 219 s to 75 s, and the cold restore improves too (134 s against
    144 s). Interleave also has the fastest `cuda-checkpoint` lock and
    checkpoint (53 s against 94 s), which matches running fully unpinned
    (49 s) while keeping the cold restore fast.
-   **vLLM: a new, smaller effect.** Without strict binding, page loading
    gets faster (18 s against 52 s with interleave), but the warm
    `cuda-checkpoint` restore toggle gets slower (131-137 s against 79 s),
    so the warm first inference is 17 s worse with interleave. Its cold
    restore is unchanged (145 s against 144 s). `preferred` is worst for
    vLLM: its cold toggle is slow too (81 s). I do not know why the toggle
    is sensitive to placement for vLLM and not SGLang. vLLM's
    interleave cycle did see direct reclaim (the host had about 1.5 TB in
    use with the pre-fill), so this may be global memory pressure rather
    than placement. Not verified.
-   **Smaller cells are fine with interleave.** `vllm_tp2` 48.6x (44.4x
    with membind), `sglang_tp4_symm` 20.2x (18.4x), `vllm_tp4_72b_xgpu`
    15.5x (14.8x), `sglang_tp8_72b` 7.1x (5.6x). All pass. Checkpoints of the
    smaller cells are slightly slower (`vllm_tp2` 9.4 s against 8.5 s,
    `vllm_tp4_72b_xgpu` 42.0 s against 35.5 s); `sglang_tp8_72b`'s is
    faster (72.5 s against 86.9 s).

Decision: `bench.sh` now defaults to `--cpunodebind=N --interleave=all`
(`NUMA_MEM=membind` restores the old behavior). It removes the 3x outlier,
is best or tied on every cold restore, and its only loss is vLLM's warm
restore under a pre-filled cache. Timings from before this change were
taken with `--membind` and are not directly comparable. For deployment:
strict single-node binding of a sandbox whose host footprint approaches
the node's memory should be avoided; interleaving is a reasonable default
for TP=8 on this topology (GPUs 0-3 on node 0, 4-7 on node 1).

Open: why vLLM's restore toggle slows down without strict binding. A
repeat of the vLLM cells without the pre-fill, with `numastat -p` on the
sandbox during the toggle, would tell placement from pressure.

## Round after P0: processes that never checkpoint

Rule adopted this round: a problem in the shim may refuse a checkpoint, but
it must never change the behavior of a process that does not checkpoint.
The gate, `suspend_locked`, `resume_locked`, the marker protocol and the
R610 workarounds are untouched.

| Item | Commit | CPU evidence | Left for the GPU round |
| :--- | :--- | :--- | :--- |
| `dlsym` re-anchoring: every `RTLD_NEXT` lookup, and every name not rewritten (`cu[A-Z]*`, `cudaGetDriverEntryPoint*`), is forwarded with a tail call | `d1b1d1823` | `run.sh preload` passes (interposer before and after the shim). `mcshim-repro.sh` checks 1 and 2 pass on the fastbuild and `-c opt` Bazel shims and on `build.sh`'s; on `45bd3f7b0`'s shim, check 1 recursed (exit 99) and check 2 failed. The patched source at `-O0` under GCC 11 fails check 1, which is what `copts = ["-O2"]` is for; clang passes at `-O0` through `musttail`. | `abi` and the suite: every redirected lookup still redirects. |
| cudart resolvers per caller: whatever the caller's dependencies define, unless it is the shim's wrapper; vendored copies (`libcudart-<hash>.so.12`) count; a per-thread depth counter treats a call that came back through another interposer as unplaced | `d1b1d1823` | `mcshim-repro.sh` checks 3 (vendored alone, static alone, each next to `libcudart.so.12`) and 4 (unplaced caller with two runtimes refuses) pass on every shim; all five failed on `45bd3f7b0`'s. | `rtres`, extended with a renamed copy (`patchelf --set-soname libcudart-1a2b3c4d.so.12`). |
| umask: `/tmp/mcshim` is 01777 and the log 0666 regardless of the first process's umask | `d1b1d1823` | centos:7 under the preload: `/tmp/mcshim` is `drwxrwxrwt`. | Acks from a process under another uid, in a cell. |
| glibc floor of the embedded shim: GLIBC_2.17 | `c0ebf9590` | Newest `GLIBC_` version in the `-c opt` shim `make` embeds: 2.17 (`clock_gettime`), was 2.34; NEEDED gains `libdl.so.2` and `libpthread.so.0`. `//runsc/mcshimbin:mcshimbin_test` checks that on the embedded bytes (fails on the old shim). `run.sh glibc` and `test/glibc_smoke.sh`: `sh -c true` and `/bin/true` exit 0 with empty output on ubuntu:20.04 (2.31), debian:11 (2.31), rockylinux:8 (2.28) and centos:7 (2.17), through `LD_PRELOAD` and through `/etc/ld.so.preload`; also fine on the host (2.34) and ubuntu:24.04 (2.39) with `LD_BIND_NOW=1`. gcc 11 and clang 20 build it with `-Wall -Wextra` and no warnings. | The suite on the new binary. |
| `--get-state` timeout fails the checkpoint instead of skipping the process | `c42b42acd` | Build, nogo and `control_test` pass. The `deadline-state` case is written (`ckpt_stub` now hangs on `--get-state` too). | `run.sh deadline` (needs a CUDA process). |
| Unlock every process after a failed lock or re-lock | `bcb3c6c7e` | Build, nogo and `control_test` pass. | `deadline-lock`: the lock that was killed is unlocked. |
| R615 opt-in: on by default only on R610; `--cuda-multicast-shim-unvalidated-driver=ALLOW` for anything newer, with a warning | `2a284f526` | `TestCudaMulticastShimEnabled`: 10 cases, including 615 refused, 615 allowed, 580 refused even when allowed. The flag is an enum (REFUSE/ALLOW), since `runsc/config` rejects new boolean flags. **Superseded:** the flag was removed afterwards; runsc enables the interposer on R610 or newer, and the scheduler decides which releases to use. | A run on R615. |
| Append the shim to `LD_PRELOAD` | `6888d9e0f` | `TestCudaMulticastShimPreloadOrder`. Same order as `/etc/ld.so.preload` already gave. | `torch-symm` under SGLang, whose launcher rewrites `LD_PRELOAD`. |

Notes on the floor. The approach in the brief (`.symver` on the seven
symbols, `-l:libdl.so.2 -l:libpthread.so.0`) builds and drops every 2.34
requirement, and loads under lazy binding, but fails on glibc 2.17 and 2.28
under `-z now`, which Bazel's toolchain links with: the version need then
names `libc.so.6`, and the old loader reports `symbol dlopen, version
GLIBC_2.2.5 not defined in file libc.so.6 with link time reference`. Linking
against stubs of `libdl.so.2` and `libpthread.so.0` that define the seven
symbols at the base version (`tools/mcshim/glibc_stubs.sh`, a genrule in
Bazel) records the right file, and both old and new loaders accept it.
`sscanf`, `fscanf` and `strtoul` are replaced by hand parsers, so a host
with glibc 2.38+ builds the same floor. `build.sh` and `run.sh` link the same
way. 2.17 was practical, so the 2.28 fallback was not needed.

`mcshim-fixes.patch` lacked a trailing newline (`git apply`: "corrupt patch
at line 311"); applied from a copy with one added. `mcshim-repro.sh`'s
`unplaced_main.c` needs `#define _GNU_SOURCE` for `RTLD_DEFAULT` with this
glibc; run from a copy with that line added, no other change.

Go validation: `//runsc` builds; `control_test`, `boot_test`, `config_test`,
`mcshimbin_test` and the `nogo` targets of those four packages pass.

### For the GPU round (not run; one binary)

Run everything on the new binary:

-   The full shim suite, including `preload` and `glibc`.
-   `rtres`, extended with a renamed copy
    (`patchelf --set-soname libcudart-1a2b3c4d.so.12`).
-   The stub tests, including the `--get-state` hang (`deadline-state`).
-   The 15 engine cells and the 3 without `CAP_SYS_PTRACE`.
-   Qwen2.5-72B at TP=8 on vLLM and SGLang.
-   72B restored onto other GPUs.
-   A tokens/s A/B with and without the shim, on one config that does not
    use CUDA graphs.

R615 is a separate run:
`abi` with lookups up to 13.4, the suite, the engine cells and the fabric
measurement.

Not this round: P1 table work (the 4096-entry cap fails safely). **Done
since:** see "Growable, indexed tables" below.

## P0 status

| Task | Status | Commit | Validation |
| :--- | :--- | :--- | :--- |
| P0.1 cudart resolvers per caller | Done | `ea443eb56` | `rtres` (cudart 12.8 and 13.0 in one process): each caller gets its own runtime's `cuMemcpyBatchAsync` ABI, by call and through `dlsym`; the old shim gave the 12.8 caller `cuMemcpyBatchAsync_v2`. An unplaced caller with two runtimes refuses checkpoints. A runtime renamed `libcudart.so.99` resolves. `torch-kernel` and `torch-symm` still pass. |
| P0.2 track the resolver hooks | Done | `5fd7d9c25` | `abi`: an unknown `cuGetProcAddress_v3` refuses the gate; the old shim accepted. 1,132 lookups up to CUDA 13.4 still redirect. |
| P0.3 silent preload | Done | `b86d14b83` | `silent`: `sh -c true` under the preload prints nothing; logs go to `/tmp/mcshim/mcshim.log`. `reason`: a refusal's cause and last log lines reach `runsc checkpoint`'s error. `TestContainerFileExists` covers the IMAGE-mode check. |
| P0.4 bound every cuda-checkpoint call | Done | `3592a60f1` | `deadline`: a hung lock, and a hung checkpoint, are killed after the timeout (5 s in the test); the checkpoint fails with the reason, and the application keeps running. |
| P0.5 confine job mode to opt-in | Done | `41a023ea2` | `optout`: without the interposer, no `--launch-job`, and exactly `--get-state` then `--toggle`. `TestCudaMulticastShimEnabled`. |
| P0.6 `MCSHIM_ALLOW_FABRIC` and 00f8 | Done | `9815084a2` | Measured on R610 (below). `refuse fabric`: refused while a fabric-capable allocation is alive, accepted once it is released. |
| P0.7 stale comments | Done | `8a0fb2db0` | Includes a fourth stale `do_suspend` reference the brief missed. |

## Decisions taken

### P0.5: one opt-in

The interposer flag (`--cuda-multicast-shim-path` or
`--cuda-multicast-shim-source=EMBEDDED`, which already requires
`--cuda-checkpoint-path`, on R610+) is the only switch. It turns on all of
the following together:

-   the `cuda-checkpoint --launch-job` wrap;
-   sequential `cuda-checkpoint`, as jobs require;
-   two-phase lock/checkpoint;
-   the interposer protocol (gate, suspend, resume);
-   the blocker inventory;
-   bounded invocations before GPU state is saved (`--get-state`, lock,
    unlock: 2 minutes each, fixed). The `--cuda-checkpoint-timeout` flag
    was removed after this round: checkpoint and restore invocations are
    unbounded again, since killing one partway cannot be undone.

Without the flag, CUDA checkpoints are exactly as upstream's: one unbounded
`--toggle` per process, parallel unless `--cuda-checkpoint-sequential`, toggled
back if any fails. There is no job and no blocker inventory. The runtime
`--cuda-checkpoint-path` then only sets the default binary for `runsc
checkpoint` (upstream has no runtime flag, so no existing configuration
changes).

Why:

-   **Simple to review and canary:** nothing changes unless you opt in.
-   **No exception needed to decision 5.**
-   **Avoids new refusals for existing configurations.** The blocker
    inventory now also counts fd imports. For configurations that don't opt
    in, it would turn today's "checkpoint succeeds, restore fails" for VMM
    IPC into a refusal at checkpoint time. That's better, but it is visible,
    and the import tracking has only been validated on the 15 engine cells.

Shipping the inventory and the deadlines to everyone is a small follow-up
once the canary shows no false positives.

### P0.6: what `cuda-checkpoint` does with fabric allocations (R610)

Measured on 610.57.04, H100, single process, `cuda-checkpoint --toggle`
twice, contents verified after restore:

| Allocation | Result |
| :--- | :--- |
| `POSIX_FD` | checkpoint and restore OK |
| `POSIX_FD`, live POSIX export | OK |
| `FABRIC\|POSIX_FD` | OK |
| `FABRIC\|POSIX_FD`, live FABRIC export | OK |
| `FABRIC\|POSIX_FD`, live POSIX export | OK |
| `FABRIC\|POSIX_FD` without an IMEX channel | `cuMemCreate` fails with `NOT_PERMITTED` (800) |

So the interposer's comment ("00f8 cannot be serialized") was wrong, and the
blocker inventory's comment was right. Both comments now record the
measurement. FABRIC stays stripped by default, since the shim cannot rebuild
IMEX sharing. Under `MCSHIM_ALLOW_FABRIC`, the gate refuses while a
fabric-capable allocation is alive, because other drivers are unmeasured.

### P0.2

R610's `libcuda` exports only `cuGetProcAddress` and `cuGetProcAddress_v2`.
There is no `_v2_ptsz`, so no new wrapper is needed.

## Open questions

-   **The blocker inventory for configurations that don't opt in.** I
    recommend leaving it out of the first merge (see P0.5).
-   **The fdinfo line.** Resolved (R14): the `nvproxy_exported_object` line,
    and export/import tracking, are on only with the interposer
    (`nvproxy.Options.TrackExports`), and the line needs ptrace access.
-   **Correction to an earlier status.** Upstream master has #14850 (FLA) but
    not #14817 (`--device-map`): master's `state_cuda.go` still has the FIXME
    for it.

## Validation on this node (H100 ×8, 610.57.04)

-   **Go unit tests and nogo:** `nvproxy`, `control`, `state`, `boot`,
    `config` all pass.
-   **Native interposer tests:** `abi gate mc refcount refuse mapwait silent`
    all pass. `refuse fabric` needs an IMEX channel, which I created only for
    that run.
-   **Under runsc:** `ipc orphan deadline reason optout` all pass. `deadline`,
    `reason` and `optout` use a stub `cuda-checkpoint`
    (`tools/mcshim/test/ckpt_stub.c`) that logs its calls and hangs or fails
    on request.
-   **Engine cells on the final binary (`ea443eb56`):** all 15 pass, and so
    do the 3 without `CAP_SYS_PTRACE`, with timings as before (for example
    `vllm_tp2` 45x faster to first inference than a cold boot,
    `sglang_tp4_symm` 19x, `sglang_tp8` 9x).
-   **Silent preload in a real container (P0.3):** in a running vLLM cell,
    `/etc/ld.so.preload` lists the shim, and `runsc exec ... sh -c true`
    prints nothing.
-   **Shim suite on the final binary:** all 15 tests pass (abi, gate, mc,
    refcount, refuse, mapwait, silent, rtres, ipc, orphan, deadline, reason,
    optout, torch-kernel, torch-symm).

### Large model on all 8 GPUs

Qwen2.5-72B-Instruct (bf16, 145 GB) at TP=8 with the final binary: **PASS**
with both engines, identical output at temperature 0, restored onto all 8
GPUs.

| Step | vLLM 0.29.0 | SGLang 0.5.20 |
| :--- | :--- | :--- |
| Cold boot | 386 s | 245 s |
| Checkpoint (engine asleep) | 61.8 s, image 172 GB | 85.9 s, image 169 GB |
| `runsc restore` returns | 8.3 s | 7.9 s |
| First inference after restore | 35.5 s (10.8x faster than a cold boot) | 45.2 s (5.4x) |
| Processes managed by the interposer | 10 | 10 |
| Multicast (NVLS) objects created | 10 | 4 |
| Gate / teardown / rebuild | 0.1 s / 0.8 s / 1.2 s | 0.1 s / 0.2 s / 0.6 s |
| `cuda-checkpoint` lock and checkpoint | 31.8 s | 54.7 s |
| `cuda-checkpoint` restore toggle | 23.7 s | 25.6 s |

`cuda-checkpoint` runs one process at a time, as job mode requires, and that
is most of the checkpoint and of the time to first inference. Running it in
parallel (which job mode forbids) is the main remaining lever (P2.5).

Also passing on the final binary:

-   **Large model onto other GPUs:** vLLM with Qwen2.5-72B at TP=4,
    checkpointed on GPUs 0-3 and restored onto GPUs 4-7 (`--device-map`).
    Checkpoint 34.9 s (image 154 GB); first inference after restore 23.0 s
    (14.6x faster than a cold boot).
-   **Single GPU without the interposer** (the existing path: `runsc
    checkpoint --cuda-checkpoint-path` only): vLLM 70x, SGLang 52x. The
    sentry ran exactly `--get-state`, then `--toggle` per process, and
    `--toggle` again on restore, with no job wrap.
-   **Single GPU with the interposer:** vLLM 68x.

## Next

1.  ~~The GPU round on the new binary~~ done on `dc93d60ef` (above).
2.  After the first merge: R615 (P2.4), then P1.

---

## Reply to the third interposer review

Thanks again. All five items and all the optional ones are addressed.
Branch: `luis/multicast-xgpu-upstream`.

**1. Release while torn down.** Done as suggested.

-   The release is now derived from state: the gate marker is gone, the gate
    is armed, and the process is neither torn down nor broken. The `armed`
    flag is gone.
-   The preflight disarms only when the process is neither torn down nor
    broken.
-   `on_gate_up` removes a stale `error.<pid>` before acking.
-   A suspend that arrives without the gate now refuses, so that preflight
    path is deleted.

Test: `gate` removes and re-creates the gate while the process is torn down,
once accepted and once refused, and checks that the application is neither
released early nor stranded.

**2. Multicast `cuMemMap` under `g_lock`.** Agreed: the rule is "no call that
can wait on the GPU or a peer runs under the lock", not "calls the header
marks synchronous".

-   `cuMemMap` now translates under the lock, calls the driver without it,
    and commits only if the handle's slot still holds the same object (each
    slot has a generation).
-   The preflight returns before taking the lock when the drain fails.
-   I also took the optional unmap handle-reuse fix: `cuMemUnmap` marks the
    objects it frees as dying, and lookups skip them.

Test: `mapwait`. With the old shim, two processes deadlock: each maps its own
group on one thread while another thread adds its device to the peer's group.
The test also checks that a map stuck in flight makes the gate refuse after
the drain timeout.

**3. Import check in the sentry.** Conceded, and done as suggested.

-   nvproxy now records objects imported through `IMPORT_OBJECT(S)_FROM_FD`,
    together with the exported object they came from.
-   Once every process is gated, the sentry checks that each import's
    exported object still exists in a process that runs the interposer. This
    happens before anything is locked or torn down.
-   The shim's `exp-*` announcements and its `kill(pid, 0)` probe are gone.

The end-to-end test caught two things that reading the code would not have:

-   libcuda reuses a freed RM handle immediately: the exporter's next
    allocation took it. A handle-based check therefore passed an orphaned
    import, so the check now compares nvproxy's object records instead of
    handles.
-   libcuda exports some allocations with a companion object in a later slot
    of the fd, and the importer imports both. Only slot 0 has a recorded
    identity, so imports from any slot are attributed to slot 0, the
    allocation the interposer re-exports.

Writing the import handlers also turned up a latent bug:
`NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM` was 0, while the driver defines it
as 1.

New test, `orphan`: one process maps memory that another process exported and
then freed. `run.sh` runs a real `runsc checkpoint`, expects the refusal
("imports that could not be rebuilt after a restore: ... which will not be
re-exported"), and checks that both processes keep running.

**4. Post-suspend backstop.** It already existed: `CheckpointBlockers` runs
after every process acks `suspended`, before `cuda-checkpoint`. But nvproxy
had no record of fd imports, so it could not see a leaked one. With item 3,
imports are blockers too. The up-front check refuses them in processes
without the interposer, and the backstop catches any the interposer left
behind.

**5. `mem_dev`.** Measured on R610: the driver refuses every multicast bind of
imported memory (`CUDA_ERROR_INVALID_VALUE`), by handle or by address, v1 or
v2, while the same binds of local memory succeed. So an import is never a
bind target, and the only wrong case was a v1 bind by address of memory the
shim never saw.

-   That case now asks the driver (`CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL`).
-   Tracked memory uses its allocation's location.

`mc` covers it: a group bound by address, with device 0 current, over memory
on both devices that the shim never saw. With the old shim, suspend unbinds
the wrong device (device 0) and fails.

**Optional items.**

-   **Unmap handle-reuse window:** done (see 2).
-   **Bind flags:** recorded and replayed.
-   **Unmapped export contents:** documented in the README's known
    limitations.
-   **`devs[]` in `devs_in_use`:** dropped. A transition re-adds a group's
    devices from the group's own device, so a device only added to a group
    needs no context. Test: a `refuse` case that adds a device from another
    device's context is accepted and survives a cycle.
-   **Suspend without a gate:** now refuses (see 1).
-   **Table scans:** done. I measured before deciding: a miss scanned all
    4096 slots in about 2.5 µs, since a slot is 232 to 240 bytes, and VMM
    calls do several scans. Slots fill first-free from 0, so a grow-only
    high-water mark per table bounds scans by the peak number of live
    entries: 0.18 µs at 300 entries. Launches never scan; the gate only
    checks a flag.

**Also hardened on the sentry side.** A checkpoint attempt after a failed
teardown used to wait out the 5-minute ack timeout for acks no process would
send. It is now refused at once. Before each request, the sentry also clears
every stale ack of the kind it waits for, not only error acks.

**Validation (H100 ×8, R610.57.04).**

-   Unit tests pass, and so does the shim suite (abi, gate, mc, refcount,
    refuse, mapwait, ipc, orphan, torch-kernel, torch-symm).
-   All 15 engine cells pass a checkpoint/restore with identical output:
    vLLM TP=2/4/8, SGLang TP=4/8, NVLS, symmetric memory, FlashInfer fusion,
    and restores onto other GPUs. So do the 3 cells without
    `CAP_SYS_PTRACE`.
-   Timings are unchanged from the last round (design doc updated). The
    interposer's share is about 0.1 s to arm the gate, 0.1 to 0.7 s for the
    teardown, and 0.4 to 1.1 s for the rebuild.

The first gate run of the import check refused every vLLM cell and the
symmetric-memory cells with "imported from an unknown export". That was the
companion object from item 3; it is fixed, and those cells were re-run.

### Changes since this reply

These came from our own hardening pass, not from the review, but you may want
to look at them:

-   **Silent preload.** The shim prints nothing (it is preloaded into every
    process in the container). It logs to `/tmp/mcshim/mcshim.log`, and the
    sentry quotes a refusal's reason and the shim's last log lines in its
    error. In IMAGE mode, a missing interposer file is not added to
    `/etc/ld.so.preload`.
-   **Bounded `cuda-checkpoint`.** With the interposer, every invocation that
    runs before GPU state is saved has a fixed 2-minute deadline; one still
    running is killed, and the checkpoint fails with the application
    running. (This replaced a `--cuda-checkpoint-timeout` flag that bounded
    checkpoint and restore too.)
-   **Opt-in only.** Without the interposer, checkpoints are exactly as
    upstream's: no job, one `--toggle` per process, no blocker inventory.
-   **Resolvers tracked.** A lookup returning an unknown `cuGetProcAddress`
    ABI refuses checkpoints.
-   **Fabric allocations.** Measured on R610: `cuda-checkpoint` carries a
    process's fabric-capable allocations. Under `MCSHIM_ALLOW_FABRIC`, the
    gate still refuses while one is alive.
