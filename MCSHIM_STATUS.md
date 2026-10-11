# Multi-GPU CUDA checkpoint/restore: status and reviewer reply

Status as of 2026-10-10. Branch `luis/multicast-xgpu-upstream`, merged with
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

Next: rerun the 235B cycles with `numactl --cpunodebind=N --preferred=N`
and with `--interleave=all`, and make the better one `bench.sh`'s default.

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

Not this round: P1 table work (the 4096-entry cap fails safely).

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
-   **The fdinfo line.** nvproxy now adds an `nvproxy_exported_object` line to
    `/proc/<pid>/fdinfo` for exported `/dev/nvidiactl` fds, for every nvproxy
    user. Applications can see it, but it is harmless (dmabuf has the same
    pattern); this is open question 4 in the design doc. Gating it on the flag
    means plumbing the flag into nvproxy's options.
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

<!-- numa2-raw:begin -->

### NUMA policy rerun (raw, in progress)

Unanalyzed output of `/data/bench/numa2/run.sh` (see its header). Times in s; reclaim columns are deltas over the cycle (pages, stalls, refaults); n1 peak is node 1 use above its start.

| Engine | Policy | ckpt lock+ckpt | ckpt total | warm page load | warm toggle | warm first inf | cold page load | cold first inf | direct reclaim | alloc stalls | refaults | cold boot | n1 peak | result |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| sglang | preferred | 87 | 181.3 | 19 | 41 | 83.3 | 79 | 143.2 | 0 | 1 | 26560 | 579.4 | 506G | PASS |
| sglang | interleave | 53 | 149.5 | 24 | 40 | 74.5 | 88 | 133.7 | 0 | 3 | 5665182 | 598.7 | 903G | PASS |
| vllm | membind | 52 | 141.7 | 52 | 79 | 138.9 | 79 | 144.4 | 961954 | 13252 | 39416805 | 1049.2 | 104G | PASS |
| vllm | preferred | 55 | 152.3 | 43 | 137 | 187.0 | 79 | 167.8 | 0 | 0 | 114088 | 1043.1 | 720G | PASS |
| vllm | interleave | 46 | 141.4 | 18 | 131 | 156.0 | 87 | 145.4 | 830333 | 808 | 41821 | 1087.7 | 981G | PASS |

```
[2026-10-11 16:56:28] START runsc=runsc version dc93d60ef9dd node0 free=989G file=0G, node1 free=995G file=0G
[2026-10-11 16:57:47] numa2-sglang-preferred: start, node0 free=500G file=489G, node1 free=995G file=0G
[2026-10-11 17:19:24] numa2-sglang-preferred: rc=0, end node0 free=504G file=483G, node1 free=989G file=6G
    engine=sglang tp=8 gpus=0,1,2,3,4,5,6,7 restore-gpus=0,1,2,3,4,5,6,7 shim=1 sleep=1 runsc=dc93d60ef9dd pin=numactl --cpunodebind=0 --preferred=0 job=1 noptrace=0 ckpt_flags=none model=Qwen/Qwen3-2
    cold boot 579.4s
    load pre: n=512 c=64 in=1024 out=256 completed=512 output_tok/s=2138.3 total_tok/s=10691.6 req/s=8.3 ttft_ms mean/p99=1425.7/2396.5 tpot_ms mean/p99=24.3/29.7 itl_ms p99=24.7
    checkpoint 181.3s image 496G
    restore returned 18.7s
    first inference after restore 83.3s
    load post: n=512 c=64 in=1024 out=256 completed=512 output_tok/s=2157.1 total_tok/s=10785.8 req/s=8.4 ttft_ms mean/p99=1394.4/2244.6 tpot_ms mean/p99=24.2/29.3 itl_ms p99=24.1
    cold restore returned 79.3s
    cold first inference after restore 143.2s (speedup 4.0x)
    sentry: 17:11:14 cuda-checkpoint lock+checkpoint on 10 processes took [1m27.167251838s]
    sentry: 17:12:44 Async page saving completed in 1m28.894s (531613372416 bytes, 5980.290 MB/s); queue full for 1m26.218s (529958031360 bytes, 6146.739 MB/s)
    sentry: 17:13:09 Async page loading completed in 18.519s (531613372416 bytes, 28707.034 MB/s); 0 waiters waited 0s~0s for 0 bytes
    sentry: 17:13:51 cuda-checkpoint restore toggle on 10 processes took [41.47205483s]
    sentry: 17:18:12 Async page loading completed in 1m19.016s (531613372416 bytes, 6727.915 MB/s); 0 waiters waited 0s~0s for 0 bytes
    sentry: 17:18:53 cuda-checkpoint restore toggle on 10 processes took [40.606863059s]
    RESULT: PASS (speedup to first inference 6.9x)
[2026-10-11 17:19:24]   vmstat deltas: pgscan_direct=0 allocstall_normal=0 allocstall_movable=1 workingset_refault_file=26560 pgscan_kswapd=130959731 
[2026-10-11 17:21:32] numa2-sglang-interleave: start, node0 free=500G file=489G, node1 free=995G file=0G
[2026-10-11 17:43:00] numa2-sglang-interleave: rc=0, end node0 free=741G file=247G, node1 free=747G file=247G
    engine=sglang tp=8 gpus=0,1,2,3,4,5,6,7 restore-gpus=0,1,2,3,4,5,6,7 shim=1 sleep=1 runsc=dc93d60ef9dd pin=numactl --cpunodebind=0 --interleave=all job=1 noptrace=0 ckpt_flags=none model=Qwen/Qwen
    cold boot 598.7s
    load pre: n=512 c=64 in=1024 out=256 completed=512 output_tok/s=2138.2 total_tok/s=10691.4 req/s=8.3 ttft_ms mean/p99=1422.5/2399.2 tpot_ms mean/p99=24.3/29.8 itl_ms p99=25
    checkpoint 149.5s image 495G
    restore returned 24.3s
    first inference after restore 74.5s
    load post: n=512 c=64 in=1024 out=256 completed=512 output_tok/s=2157 total_tok/s=10785.3 req/s=8.4 ttft_ms mean/p99=1390.4/2274.5 tpot_ms mean/p99=24.2/30 itl_ms p99=24.9
    cold restore returned 87.9s
    cold first inference after restore 133.7s (speedup 4.4x)
    sentry: 17:34:46 cuda-checkpoint lock+checkpoint on 10 processes took [52.527013513s]
    sentry: 17:36:20 Async page saving completed in 1m33.765s (530799964160 bytes, 5660.988 MB/s); queue full for 1m31.018s (528977580032 bytes, 5811.781 MB/s)
    sentry: 17:36:52 Async page loading completed in 24.112s (530799964160 bytes, 22014.220 MB/s); 0 waiters waited 0s~0s for 0 bytes
    sentry: 17:37:31 cuda-checkpoint restore toggle on 10 processes took [39.656319608s]
    sentry: 17:42:06 Async page loading completed in 1m27.638s (530799964160 bytes, 6056.728 MB/s); 0 waiters waited 0s~0s for 0 bytes
    sentry: 17:42:41 cuda-checkpoint restore toggle on 10 processes took [35.051086266s]
    RESULT: PASS (speedup to first inference 8.0x)
[2026-10-11 17:43:00]   vmstat deltas: pgscan_direct=0 allocstall_normal=0 allocstall_movable=3 workingset_refault_file=5665182 pgscan_kswapd=21915783 
[2026-10-11 17:45:16] numa2-vllm-membind: start, node0 free=494G file=495G, node1 free=994G file=0G
[2026-10-11 18:16:42] numa2-vllm-membind: rc=0, end node0 free=681G file=306G, node1 free=995G file=0G
    engine=vllm tp=8 gpus=0,1,2,3,4,5,6,7 restore-gpus=0,1,2,3,4,5,6,7 shim=1 sleep=1 runsc=dc93d60ef9dd pin=numactl --cpunodebind=0 --membind=0 job=1 noptrace=0 ckpt_flags=none model=Qwen/Qwen3-235B-
    cold boot 1049.2s
    load pre: n=512 c=64 in=1024 out=256 completed=512 output_tok/s=2140.3 total_tok/s=10701.8 req/s=8.3 ttft_ms mean/p99=721.8/3377.9 tpot_ms mean/p99=27.1/32.5 itl_ms p99=233.5
    checkpoint 141.7s image 490G
    restore returned 52.0s
    first inference after restore 138.9s
    load post: n=512 c=64 in=1024 out=256 completed=512 output_tok/s=2248.1 total_tok/s=11240.5 req/s=8.7 ttft_ms mean/p99=779.9/2103.7 tpot_ms mean/p99=25.4/28 itl_ms p99=233.4
    cold restore returned 79.6s
    cold first inference after restore 144.4s (speedup 7.2x)
    sentry: 18:06:49 cuda-checkpoint lock+checkpoint on 10 processes took [52.35179543s]
    sentry: 18:08:15 Async page saving completed in 1m26.372s (525365977088 bytes, 6082.597 MB/s); queue full for 1m25.791s (524916318208 bytes, 6118.562 MB/s)
    sentry: 18:09:15 Async page loading completed in 51.825s (525365977088 bytes, 10137.263 MB/s); 0 waiters waited 0s~0s for 0 bytes
    sentry: 18:10:35 cuda-checkpoint restore toggle on 10 processes took [1m19.467284156s]
    sentry: 18:15:17 Async page loading completed in 1m19.408s (525365977088 bytes, 6616.060 MB/s); 0 waiters waited 0s~0s for 0 bytes
    sentry: 18:16:15 cuda-checkpoint restore toggle on 10 processes took [57.296115494s]
    RESULT: PASS (speedup to first inference 7.5x)
[2026-10-11 18:16:42]   vmstat deltas: pgscan_direct=961954 allocstall_normal=411 allocstall_movable=12841 workingset_refault_file=39416805 pgscan_kswapd=390166360 
[2026-10-11 18:18:39] numa2-vllm-preferred: start, node0 free=494G file=495G, node1 free=995G file=0G
[2026-10-11 18:52:15] numa2-vllm-preferred: rc=0, end node0 free=505G file=482G, node1 free=991G file=3G
    engine=vllm tp=8 gpus=0,1,2,3,4,5,6,7 restore-gpus=0,1,2,3,4,5,6,7 shim=1 sleep=1 runsc=dc93d60ef9dd pin=numactl --cpunodebind=0 --preferred=0 job=1 noptrace=0 ckpt_flags=none model=Qwen/Qwen3-235
    cold boot 1043.1s
    load pre: n=512 c=64 in=1024 out=256 completed=512 output_tok/s=2133.4 total_tok/s=10667.3 req/s=8.3 ttft_ms mean/p99=881.6/3362.3 tpot_ms mean/p99=26.5/32.4 itl_ms p99=234.8
    checkpoint 152.3s image 490G
    restore returned 42.8s
    first inference after restore 187.0s
    load post: n=512 c=64 in=1024 out=256 completed=512 output_tok/s=2306.1 total_tok/s=11530.8 req/s=9 ttft_ms mean/p99=690.7/2114.7 tpot_ms mean/p99=25/27 itl_ms p99=220
    cold restore returned 79.0s
    cold first inference after restore 167.8s (speedup 6.2x)
    sentry: 18:40:30 cuda-checkpoint lock+checkpoint on 10 processes took [55.0004109s]
    sentry: 18:42:04 Async page saving completed in 1m34.429s (525421940736 bytes, 5564.193 MB/s); queue full for 1m33.865s (524967165952 bytes, 5592.773 MB/s)
    sentry: 18:43:03 Async page loading completed in 42.701s (525421940736 bytes, 12304.705 MB/s); 0 waiters waited 0s~0s for 0 bytes
    sentry: 18:45:20 cuda-checkpoint restore toggle on 10 processes took [2m16.899522815s]
    sentry: 18:50:26 Async page loading completed in 1m18.794s (525421940736 bytes, 6668.335 MB/s); 0 waiters waited 0s~0s for 0 bytes
    sentry: 18:51:48 cuda-checkpoint restore toggle on 10 processes took [1m21.330849965s]
    RESULT: PASS (speedup to first inference 5.5x)
[2026-10-11 18:52:15]   vmstat deltas: pgscan_direct=0 allocstall_normal=0 allocstall_movable=0 workingset_refault_file=114088 pgscan_kswapd=159445080 
[2026-10-11 18:54:22] numa2-vllm-interleave: start, node0 free=494G file=495G, node1 free=995G file=0G
[2026-10-11 19:26:51] numa2-vllm-interleave: rc=0, end node0 free=743G file=244G, node1 free=749G file=245G
    engine=vllm tp=8 gpus=0,1,2,3,4,5,6,7 restore-gpus=0,1,2,3,4,5,6,7 shim=1 sleep=1 runsc=dc93d60ef9dd pin=numactl --cpunodebind=0 --interleave=all job=1 noptrace=0 ckpt_flags=none model=Qwen/Qwen3-
    cold boot 1087.7s
    load pre: n=512 c=64 in=1024 out=256 completed=512 output_tok/s=2128.7 total_tok/s=10643.9 req/s=8.3 ttft_ms mean/p99=926/3393.6 tpot_ms mean/p99=26.4/32.7 itl_ms p99=235.9
    checkpoint 141.4s image 490G
    restore returned 18.5s
    first inference after restore 156.0s
    load post: n=512 c=64 in=1024 out=256 completed=512 output_tok/s=2291.5 total_tok/s=11457.9 req/s=8.9 ttft_ms mean/p99=795.8/2128.9 tpot_ms mean/p99=24.8/27.3 itl_ms p99=234.1
    cold restore returned 87.1s
    cold first inference after restore 145.4s (speedup 7.4x)
    sentry: 19:16:24 cuda-checkpoint lock+checkpoint on 10 processes took [46.272787684s]
    sentry: 19:17:56 Async page saving completed in 1m32.255s (525100027904 bytes, 5691.861 MB/s); queue full for 1m31.905s (524856332288 bytes, 5710.829 MB/s)
    sentry: 19:18:23 Async page loading completed in 18.354s (525100027904 bytes, 28609.038 MB/s); 0 waiters waited 0s~0s for 0 bytes
    sentry: 19:20:34 cuda-checkpoint restore toggle on 10 processes took [2m10.84093841s]
    sentry: 19:25:33 Async page loading completed in 1m26.922s (525100027904 bytes, 6041.070 MB/s); 0 waiters waited 0s~0s for 0 bytes
    sentry: 19:26:25 cuda-checkpoint restore toggle on 10 processes took [51.635819189s]
    RESULT: PASS (speedup to first inference 6.9x)
[2026-10-11 19:26:51]   vmstat deltas: pgscan_direct=830333 allocstall_normal=808 allocstall_movable=0 workingset_refault_file=41821 pgscan_kswapd=37376400 
[2026-10-11 19:27:47] better policy: interleave (score preferred=581.3 interleave=509.6)
[2026-10-11 19:27:47] CELLS with NUMA_MEM=interleave
    vllm_tp2 | RESULT: PASS (speedup to first inference 48.6x) | devmap=0 0xfd=8 | checkpoint 9.4s image 12G|restore returned 1.3s|first inference after restore 5.2s|RESULT: PASS (speedup to first inference 48.6x)
    sglang_tp4_symm | RESULT: PASS (speedup to first inference 20.2x) | devmap=0 0xfd=6 | checkpoint 16.9s image 17G|restore returned 1.9s|first inference after restore 10.1s|RESULT: PASS (speedup to first inference 20.2x)
    vllm_tp4_72b_xgpu | RESULT: PASS (speedup to first inference 15.5x) | devmap=1 0xfd=8 | checkpoint 42.0s image 154G|restore returned 5.9s|first inference after restore 25.2s|RESULT: PASS (speedup to first inference 15.5x)
    sglang_tp8_72b | RESULT: PASS (speedup to first inference 7.1x) | devmap=0 0xfd=4 | checkpoint 72.5s image 169G|restore returned 7.9s|first inference after restore 37.4s|RESULT: PASS (speedup to first inference 7.1x)
    GATE_DONE
[2026-10-11 19:51:34] ALL DONE
```

<!-- numa2-raw:end -->
