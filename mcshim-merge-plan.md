# Multi-GPU CUDA checkpoint/restore: architecture and upstream merge plan

| | |
| :--- | :--- |
| Status | Draft for review |
| Date | 2026-10-12 |
| Branch | `luis/multicast-xgpu-upstream` at `43da4f2fb` (github.com/luiscape/gvisor) |
| Upstream | `google/gvisor` `master` at `a676ebdd0` (merged into the branch) |
| Design doc | `g3doc/proposals/cuda_multigpu_checkpoint.md` (on the branch) |

## 1. Summary

The branch makes `runsc checkpoint` and `runsc restore` work for single-node,
multi-process, multi-GPU CUDA workloads (tensor-parallel vLLM and SGLang),
including restore onto different GPUs. It has been validated end to end on
8x H100 with driver 610.57.04, including Qwen2.5-72B at TP=8 on both engines.

It is not ready to merge upstream. A five-part review of the whole branch
found no design problems, but found fixes that must land first, in four
groups:

1. **The interposer changes processes that never checkpoint.** It is
   preloaded into every process of the container, and six bugs alter how
   ordinary processes behave (crashes, signal handling, `dlsym` results, one
   file-permission issue).
2. **Privileged file operations follow container-controlled paths.** The
   sentry and runsc act as root on paths inside the container that a
   container user can redirect with a symlink.
3. **Users who never enable the interposer still see changes.** nvproxy
   tracking, the restore path and the job wrap reach them today.
4. **Hygiene.** Dev-only files, stale docs and upstream drift.

Since the first draft, R1-R6, R9-R24 have been fixed or resolved on the
branch (section 4.1), R8 has been dropped, and R7 is resolved by keeping
the current behavior. That closes all four groups above. What remains is
applying R21 when each PR is cut, the two open decisions in section 8
(R25), and the GPU validation round (section 7.4).

This document describes the architecture, lists the fixes as requirements
(must land with or before the code they fix) and nice-to-haves (follow-ups),
and proposes 10 PRs in merge order. Section 8 lists the decisions reviewers
need to make.

## 2. Problem

gVisor already checkpoints CUDA processes by running NVIDIA's
`cuda-checkpoint` on each of them. That works for processes that share no GPU
state. Multi-GPU workloads do share it, and `cuda-checkpoint` cannot handle
that state. Measured at TP=2 and TP=4 on 8x H100 with NVSwitch, driver
610.57.04:

| Cross-process GPU state | Created by | Result without this work |
| :--- | :--- | :--- |
| NVLink multicast group (`NV_MEMORY_MULTICAST_FABRIC`, 0x00fd) | NCCL NVLS, PyTorch symmetric memory, FlashInfer all-reduce fusion | `cuda-checkpoint` hangs |
| VMM IPC import (`cuMemImportFromShareableHandle`) | NCCL peer-to-peer with `NCCL_CUMEM_ENABLE=1` | Checkpoint succeeds, restore fails |
| Legacy CUDA IPC import (`cuIpcOpenMemHandle`) | NCCL with cuMem disabled, the engines' custom all-reduce | `cuda-checkpoint` hangs |

Driver R610's *job mode* (`cuda-checkpoint --launch-job`) fixes the last row
only. vLLM and SGLang hit the first two in their default configurations.

Non-goals: multi-node NVLink (IMEX), drivers older than R610, checkpointing
device memory contents (still `cuda-checkpoint`'s job), and isolation (the
interposer runs in the container's trust domain).

## 3. Architecture

### 3.1 Components

```
 runsc (host)                       Sandbox (sentry)                     Container
 ------------                       ----------------                     ---------
 flags, Validate                    pkg/sentry/control                   every process:
 boot/loader.go:                      state_cuda.go                        LD_PRELOAD and
   - LD_PRELOAD + ld.so.preload         cuda-checkpoint phases              /etc/ld.so.preload
     of --cuda-multicast-shim-path      (lock / checkpoint / restore)       load mcshim.so,
   - wrap command in                  state_cuda_shim.go
     cuda-checkpoint --launch-job       marker protocol  <---------------> /tmp/mcshim/
 boot/restore.go:                   pkg/sentry/devices/nvproxy             gate, suspend,
   - save() options                   export/import tracking               present.<pid>, acks
                                      checkpoint-blocker inventory       CUDA processes:
                                    pkg/sentry/fsimpl/proc                 mcshim control thread
                                      fdinfo: nvproxy_exported_object      (started at cuInit)

 (mcshim.so is provided by the operator at --cuda-multicast-shim-path,
  in the image or through a mount; runsc does not carry it.)
```

| Component | Location | Role |
| :--- | :--- | :--- |
| Interposer (`mcshim.so`) | `tools/mcshim` (C, about 2,900 lines) | Tracks multicast groups, VMM allocations, exports, imports, binds and mappings at the libcuda layer. On request, gates the application off the GPU, tears that state down, and rebuilds it later at identical virtual addresses, translating handles so the application only sees the values it was given. Interposes `cu*` entry points, `dlsym`, `cuGetProcAddress*` and cudart's `cudaGetDriverEntryPoint*`. |
| Sentry sequencing | `pkg/sentry/control/state_cuda.go`, `state_cuda_shim.go` | Runs the `cuda-checkpoint` phases and drives the interposer through marker files. With the interposer, the invocations that run before GPU state is saved (`--get-state`, lock, unlock) are bounded at 2 minutes; checkpoint and restore invocations are not (section 3.5). |
| nvproxy additions | `pkg/sentry/devices/nvproxy`, `pkg/sentry/fsimpl/proc` | Records exported RM objects and fd imports; reports an exported object's identity on a `nvproxy_exported_object:` line in `/proc/<pid>/fdinfo/<fd>`; provides the *checkpoint-blocker inventory* (objects `cuda-checkpoint` cannot handle), so a checkpoint is refused up front instead of hanging. |
| runsc delivery | `runsc/boot`, `runsc/config` | Flags; preloads the interposer, which the operator provides at `--cuda-multicast-shim-path`, through `LD_PRELOAD` and `/etc/ld.so.preload`; wraps the container command in `cuda-checkpoint --launch-job`. |

### 3.2 Configuration

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--cuda-checkpoint-path` | empty | `cuda-checkpoint` inside the container. Required by the interposer. |

| `--cuda-multicast-shim-path` | empty | Interposer path in the container. Setting it enables the interposer. |


Without the interposer flags, CUDA checkpoints are unchanged from upstream
with #14817: no job wrap, one `cuda-checkpoint --toggle` per process (R12).

### 3.3 Control protocol

The sentry and the interposer communicate only through files in
`/tmp/mcshim` inside the container. Markers are existence-based: the
interposer reacts to a file appearing or disappearing, never to its content.

| File | Written by | Meaning |
| :--- | :--- | :--- |
| `gate` | sentry | Created: stop new GPU submissions, wait for calls in flight. Removed: release the application. |
| `suspend` | sentry | Created: tear down tracked state. Removed: rebuild it. |
| `present.<pid>` | interposer | This process runs a control thread and will acknowledge. |
| `gated.<pid>`, `suspended.<pid>`, `resumed.<pid>` | interposer | Acknowledgements. |
| `error.<pid>` | interposer | The transition failed; the sentry fails fast. |

The directory is part of the checkpoint image, so `suspend` and `gate` still
exist after a restore, and the interposer stays torn down and gated until the
sentry removes them.

### 3.4 Checkpoint and restore sequence

Checkpoint:

0. Refuse if a process that does not run the interposer holds multicast
   objects or imported memory, or if an earlier attempt's teardown failed.
1. Create `gate`; wait for `gated.<pid>` from every process. With everyone
   gated, refuse if an import's source object no longer exists.
2. `cuda-checkpoint --action lock` on all processes in parallel (drains GPU
   work).
3. `--action unlock` (the teardown issues CUDA calls, which a locked process
   cannot make).
4. Create `suspend`; wait for `suspended.<pid>`.
5. Verify the blocker inventory is now empty.
6. Re-lock, then `--action checkpoint` one process at a time (job mode
   requires it); save the sandbox.

Restore:

7. `--toggle`, or `--action restore --device-map` then `--action unlock` when
   the GPUs changed.
8. Remove `suspend`; wait for `resumed.<pid>` (rebuild at identical virtual
   addresses).
9. Remove `gate`, releasing the application.

Any failure before step 6 completes unwinds: unlock, rebuild, ungate, and the
application keeps running. A failed teardown or rebuild deliberately leaves
the application gated ("better blocked than corrupt").

### 3.5 Design principles

- **Opt-in and inert.** Nothing changes unless the interposer flags are set.
- **Refuse, never corrupt.** A problem may refuse a checkpoint; it must never
  change the behavior of a process that does not checkpoint. The interposer
  is preloaded into every process in the container, most of which never
  checkpoint.
- **Bounded where it can be undone.** With the interposer, every
  acknowledgement wait, and every `cuda-checkpoint` invocation that runs
  before GPU state is saved (`--get-state`, lock, unlock), has a fixed
  deadline; on expiry the checkpoint fails and the application keeps
  running. `--action checkpoint` and every restore invocation wait without a
  bound, as without the interposer: their duration grows with GPU memory,
  killing one partway leaves the process in a state `cuda-checkpoint` does
  not define, and a failed restore kills the sandbox anyway. There is no
  flag; an earlier `--cuda-checkpoint-timeout` flag was removed.
- **Same trust domain.** The interposer and its marker directory belong to the
  container. A container process may break its own checkpoint, but must not
  gain anything outside the container.
- **Portable.** The interposer needs no glibc symbol version newer than
  GLIBC_2.17, so it loads in old images (CentOS 7, Amazon Linux 2).

## 4. Current state of the branch

| | |
| :--- | :--- |
| Size against upstream | About 52 files and 11k added lines, of which `MCSHIM_STATUS.md` and this plan (about 1k lines) stay out of every PR (R21). |
| Upstream drift | None: upstream `master` at `a676ebdd0` was merged in `daa5614de` with no conflicts, and builds and tests pass on it. The `nvproxy/BUILD` conflict seen earlier no longer occurs. |
| Related open upstream PRs | #14817 (`--device-map` on restore): labelled *ready to pull*, not merged; the branch contains it, and without the interposer runs exactly its code (R12). #13987 (job wrap): unreviewed, and its flag names no longer match the branch. |
| History | WIP commits and files added then removed. PRs should be cut from the final tree, not cherry-picked. |
| Validation so far | H100 x8, driver 610.57.04: the interposer suite, 15 engine configurations, 3 more without `CAP_SYS_PTRACE`, Qwen2.5-72B at TP=8 on vLLM and SGLang, 72B restored onto other GPUs. The latest round (dlsym fix, cudart resolvers, glibc floor, sentry fixes) is CPU-validated only. |

### 4.1 Requirements fixed so far

Each fix is CPU-validated; the GPU round (section 7.4) still has to run on
the new binary.

| Requirement | Commit | Evidence |
| :--- | :--- | :--- |
| R2 (interposer control directory) | `3f8f69a7c` | `run.sh ctlsec` (Docker, real control thread over the fake libcuda): a planted log symlink, log hard link, marker symlink, symlinked control directory and `MCSHIM_LOG` in a setuid process. The previous shim fails 5 of 6 cases; the fixed one passes all 6, including the shared log and acknowledgements across uids. |
| R16, R17, R20 (job wrap scope, missing `cuda-checkpoint`, `--exec-fd`, path validation) | `182960ccb` | `TestSetupCudaCheckpointJob` (GPU and non-GPU containers, old and newer drivers, missing or directory `cuda-checkpoint`); `config_test` cases for relative paths and separators. |
| R11, R13 (undoing a failed checkpoint; reaping comment) | `72756209d` | `TestCheckpointLocked`: 8 failure points against a fake `cuda-checkpoint`. New `undo` stub case in `run.sh` for the GPU round. |
| R15 (export/import recording) | `09df770cc` | `TestRecordExportImport` covers the recording. The lock held across the host call, and recording before the copy-out, need a GPU to exercise. |
| R1, R3, R4, R5, R6 (interposer effects on processes that never checkpoint) | `96841776a` | New `run.sh noeffect` (`noeffect.sh`): seven programs, each run with and without the shim. The previous shim changes six of them; the fixed one none. |
| R14 (nvproxy tracking for every user) | `6b3db1560` | `TestProcFDInfoExtraTracking`: the fdinfo line is empty with tracking off. With tracking off the four export/import handlers delegate to `ctrlHasFrontendFD`, upstream's handler, which a code read confirms; no CPU test exercises the host call. |
| R9, R10 (the sentry's marker operations) | `26e35715b` | `TestCudaShimFS` on a tmpfs root: the control directory as a symlink, and symlinks, a hard link and FIFOs planted as markers, acks and the log. Against the old resolution rules it fails 9 checks. |
| R19 (single-architecture embedded interposer) | `cf1bec3e8` | EMBEDDED mode and `runsc/mcshimbin` removed. Boot, config, control and nvproxy tests, nogo, and `//tools/mcshim:glibc_floor_test` (fastbuild and `-c opt`) pass; `//runsc` builds and rejects `--cuda-multicast-shim-source`. |
| R22 (upstream drift) | `daa5614de` | Merge of `a676ebdd0` (76 upstream commits) with no conflicts. Seven files changed on both sides (`Makefile`, three nvproxy files, three `runsc/config` files); the tests and nogo below pass on the merge, and `//runsc` builds. |
| R12 (the path without the interposer) | `83c1fd194` | Without the interposer, `preSaveCuda` and `postRestoreCuda` run #14817's code unchanged; a diff of `state_cuda.go` against #14817 shows only additions behind `CudaMulticastShim`. Control, boot, config and nvproxy tests and nogo pass. No CPU test runs `cuda-checkpoint`; the GPU round covers both paths. |
| R24 (CI on the shipped artifact) | `9e6d8a05a` | New `//tools/mcshim:preload_test` passes on fastbuild and `-c opt`, and fails (exit 99) on the shim built at `-O0`. With `_FORTIFY_SOURCE` pinned, fastbuild and opt import the same 60 glibc symbols, all within 2.17. `run.sh preload noeffect glibc ctlsec` passes. |
| R23 (stale docs) | `43da4f2fb` | The one remaining wrong statement (the proposal said `--cuda-checkpoint-path` alone wraps containers in a job) is corrected. The other four had already been fixed by earlier commits, which a re-read confirms. |

How each was fixed:

- **R2.** The interposer opens `/tmp/mcshim` once with `O_NOFOLLOW` and
  reaches every file relative to it; it creates files only with
  `O_EXCL | O_NOFOLLOW`; it appends to or reads an existing file only if
  it is regular with one link; it changes the mode only of a log it just
  created. setuid processes skip the interposer, and `MCSHIM_*` is read
  with `secure_getenv`. The glibc floor stays at 2.17.
- **R16, R17, R20.** The wrap and preload apply only to containers for
  which `GPUFunctionalityRequested` is true. `Validate` requires absolute
  paths, and no colons or whitespace in the interposer path. A GPU
  container that cannot be wrapped (missing `cuda-checkpoint`, or
  `--exec-fd`) starts without the job or the interposer, and checkpoints
  are refused, naming it, until it is destroyed.
- **R11, R13.** The checkpoint phase stops at the first failure, asks each
  process that did not succeed for its state (bounded `--get-state`),
  restores only checkpointed ones, and unlocks all. What cannot be undone
  returns `errCudaUnrecoverable`, which skips the interposer unwind.
- **R15.** The client lock is held across the host call, as for
  allocations, and the result is recorded before anything is copied back
  to the application.

- **R1, R3-R6.** The rewritten `dlsym` lookups find what the caller would
  find without the shim, then return the wrapper: never the shim's own
  definition, the caller's own dependencies searched for `RTLD_DEFAULT`,
  and `dlerror` left clear on success. A cudart resolver called from the
  main program goes to the runtime the main program binds to. The shim
  never loads libcuda, and its wrappers return
  `CUDA_ERROR_NOT_INITIALIZED` without it. The control thread blocks every
  signal.
- **R9, R10.** The sentry resolves `/tmp/mcshim` once, refusing a symlink;
  reaches each marker as one component without following a final symlink;
  and opens only regular files with one link, non-blocking.

A behavior change to know about: a `dlsym(RTLD_DEFAULT)` lookup of a
resolver or driver entry point that only the shim defines (no runtime or
libcuda loaded globally, or loaded `RTLD_LOCAL` by another library) now
returns NULL, as without the shim. Before, it returned the shim's wrapper.

- **R19.** runsc no longer carries the interposer. The operator provides
  `mcshim.so` inside each GPU container, in the image or through a mount,
  at the path given in `--cuda-multicast-shim-path`. That file must stay available at the same
  path for restores, so use a versioned path per release. There is no
  version handshake between the interposer and the sentry yet (N22).

- **R12.** Without the interposer, checkpoint and restore run #14817's
  `runCudaCheckpointOp` unchanged: its timeline markers, stopping at the
  first exec failure, re-pausing before the SIGSEGV unlock, and undoing a
  failed toggle as upstream does. Only with the interposer does
  `cuda-checkpoint` run with `MCSHIM_DISABLE=1` (recorded in the checkpoint
  as `cuda-multicast-shim` so restore does the same), does a failed save
  with `Resume` undo the CUDA checkpoint, and does restore resume the
  interposer. Item 3 of the original finding (a failed toggle no longer
  re-toggled) referred to restore, where #14817 itself stops undoing; that
  change is #14817's and already reviewed.
- **R24.** `//tools/mcshim:preload_test` (an `sh_test`) runs the `RTLD_NEXT`
  check on the interposer Bazel builds; `run.sh preload` calls the same
  script. `_FORTIFY_SOURCE` is pinned to 2 in `BUILD`, `build.sh` and
  `run.sh`.

- **R14.** `nvproxy.Options.TrackExports`, set only when the interposer is
  enabled, turns on export/import tracking and the fdinfo line. The line is
  shown only to readers with ptrace access to the task. One small change
  remains for every nvproxy user: each client records the thread group that
  created it (`rootClient.tgid`, used by the blocker inventory). It is
  bookkeeping only and changes no behavior.

R8 (musl images) is dropped, not fixed: since R16, a GPU container is wrapped
and preloaded only if its image has `cuda-checkpoint`, which a musl image
does not (CUDA is glibc-only). What remains is an image with a file at that
path that cannot run, which is a misconfiguration.

## 5. Requirements

These must be fixed in, or before, the PR that carries the affected code.

Status:
- **Verified**: reproduced, or confirmed by reading the code.
- **Reported**: a reviewer reproduced it; not re-run.
- **Regression**: introduced by the latest round of work.
- **Fixed**: fixed on the branch (see section 4.1).

### 5.1 Interposer (`tools/mcshim`)

| ID | Problem | Status | Fix |
| :--- | :--- | :--- | :--- |
| R1 | The exported `cu*`/`cuda*` wrappers make CUDA look installed: `dlsym(RTLD_DEFAULT, "cuInit")` succeeds in a process without CUDA, and calling a wrapper without libcuda segfaults. With libcuda installed, a process that only probes for it ends up loading it. | **Fixed** (`96841776a`); was: Verified (probe, crash); reported (forced load) | Return NULL from `dlsym` when the only definition is the interposer's own. Never `dlopen` libcuda without `RTLD_NOLOAD`. Check every real pointer and return `CUDA_ERROR_NOT_INITIALIZED`. |
| R2 | A root process run with `MCSHIM_LOG=<file>`, or a log symlink planted in `/tmp/mcshim`, makes the interposer `chmod 0666` that file. `/etc/ld.so.preload` applies to setuid programs too. | **Fixed** (`3f8f69a7c`); was: Verified; regression | `secure_getenv`; do nothing when `AT_SECURE` is set; open with `O_NOFOLLOW`, create with `O_EXCL`, change modes only on files just created. Same for marker and publish files. |
| R3 | The control thread inherits the caller's signal mask, so a program waiting for SIGTERM with `sigwait` is killed by it instead. | **Fixed** (`96841776a`); was: Verified | Block all signals around `pthread_create`. |
| R4 | `RTLD_DEFAULT` lookups of `cu*` names from `RTLD_LOCAL` libraries (how Python loads extensions) still search the interposer's scope. The latest round fixed this only for other names. | **Fixed** (`96841776a`); was: Reported; regression | Resolve in the caller's scope. |
| R5 | When the caller is the main program, cudart's resolver can be answered by the newest loaded runtime instead of the caller's own (wrong ABI). | **Fixed** (`96841776a`); was: Reported; regression | Fall back to the next definition after the interposer. |
| R6 | A successful `cu*` lookup can leave `dlerror()` set. | **Fixed** (`96841776a`); was: Reported | Clear it on success. |
| R7 | Fabric handle types are stripped in every CUDA process, so multi-node NVLink workloads that never checkpoint lose them. | **Resolved: kept as is**; was: Verified (code) | None. The interposer is turned off at the runtime level for multi-node workloads, the only ones the rewrite would harm; on a single node it is what lets PyTorch and NCCL fall back to POSIX fds, which the shim can carry. Documented in the README. |
| R8 | On musl images (Alpine), preloading the glibc interposer makes every process fail to start. | **Dropped**; was: Reported (reproduced on alpine:3.20 through `LD_PRELOAD`; musl ignores `/etc/ld.so.preload`) | None. Since R16 the interposer is preloaded only where `cuda-checkpoint` exists, which excludes musl images in practice. |

### 5.2 Sentry (`pkg/sentry/control`)

| ID | Problem | Status | Fix |
| :--- | :--- | :--- | :--- |
| R9 | Marker operations run as root on `/tmp/mcshim/<name>`, protecting only the last path component from symlinks. A container user who makes `/tmp/mcshim` a symlink first gets the sentry to create and delete files as root elsewhere, including on writable bind mounts. | **Fixed** (`26e35715b`); was: Verified (code) | Open the directory once without following symlinks and resolve each marker relative to it; consider the process's own credentials. |
| R10 | A FIFO planted as a marker, error file or log hangs the checkpoint RPC forever. | **Fixed** (`26e35715b`); was: Reported | Open non-blocking and require a regular file. |
| R11 | If the checkpoint phase fails, restore runs on every locked process, including those that did not checkpoint; the error says "unrecoverable" but the interposer is unwound anyway. | **Fixed** (`72756209d`); was: Verified (first half); reported (second) | Restore only processes that checkpointed, then unlock all; skip the unwind on the unrecoverable path. |
| R12 | The path without the interposer differs from upstream: `--device-map` on restore, a failed save-and-resume now undoes CUDA state, a failed toggle is no longer re-toggled, exec failures continue instead of stopping, an extra environment variable, unlock/pause order, missing timeline markers. | **Fixed** (`83c1fd194`); was: Verified | Without the interposer the code is #14817's, unchanged; everything else is behind the interposer option. `--device-map` is resolved by landing #14817. |
| R13 | A code comment from the latest round claims every `cuda-checkpoint` invocation is reaped; untrue when a killed one never exits. | **Fixed** (`72756209d`); was: Reported; regression | Correct it (real fix: N4). |

### 5.3 nvproxy

| ID | Problem | Status | Fix |
| :--- | :--- | :--- | :--- |
| R14 | Export/import tracking and the new fdinfo line are active for every nvproxy user. | **Fixed** (`6b3db1560`); was: Verified | Gate both on a new nvproxy option set only when the interposer is configured; add the ptrace-access check Linux applies to fdinfo. |
| R15 | Imports and exports are recorded without holding the object lock across the host call; a concurrent free and reallocation corrupts the object graph. | **Fixed** (`09df770cc`); was: Reported | Hold the lock across the call, as allocation does. |

### 5.4 runsc

| ID | Problem | Status | Fix |
| :--- | :--- | :--- | :--- |
| R16 | The job wrap and preload apply to every container when nvproxy is enabled globally, including the pause container; a container whose image lacks `cuda-checkpoint` fails to start. | **Fixed** (`182960ccb`); was: Verified (scope); reported (failure) | Apply only to containers that request GPUs; skip with a warning if the binary is missing; require absolute paths. |
| R17 | With `--exec-fd`, the job wrap drops the user's program. | **Fixed** (`182960ccb`); was: Reported | Skip the wrap in that case. |
| R18 | The interposer and `ld.so.preload` are written as root, following intermediate symlinks; without a rootfs overlay they land in the user's rootfs on the host, contrary to the docs. | **Docs fixed** (`c201dcf24`); was: Reported (two reviewers) | Docs now say where the writes land. Guard deferred (N21): it only matters with `--overlay2=none` or bind mounts over `/etc` or the interposer's directory, which Modal does not use. Since EMBEDDED's removal (`cf1bec3e8`), only the `ld.so.preload` write remains. |
| R19 | The embedded interposer is single-architecture; source distributions could embed x86 bytes into arm64 builds. | **Resolved** (`cf1bec3e8`): EMBEDDED mode removed; was: Reported | None: runsc no longer carries the interposer. |
| R20 | Flag validation misses relative paths and separators in the interposer path. | **Fixed** (`182960ccb`); was: Reported | Add the checks and tests. |

### 5.5 Build, docs and hygiene

| ID | Problem | Status | Fix |
| :--- | :--- | :--- | :--- |
| R21 | `MCSHIM_STATUS.md` is a tracked dev-only file naming internal people and companies; other scratch files are untracked but not ignored. | **Resolved: a checklist item** when cutting each PR | Leave out `MCSHIM_STATUS.md` and `mcshim-merge-plan.md`; no untracked scratch files remain, and `.gitignore` covers built `tools/mcshim/*.so`. Keep `tools/mcshim/build.sh` and `test/torch_gate_test.py` in PR3: the user guide tells operators to build with `build.sh` (now that runsc does not carry the interposer), and `run.sh` uses both. |
| R22 | Upstream drift and BUILD conventions. | **Fixed** (`daa5614de`) | Merged `a676ebdd0` with no conflicts. `tools/mcshim/BUILD` is the only new BUILD file and has no `go_library`, so the `embed =` convention does not apply. PR branches are cut from `master`. |
| R23 | Stale or wrong docs: the README threat model says marker content is never parsed (one file is); the proposal says any `--cuda-checkpoint-path` triggers the job wrap; flag help says "R610+"; two docs disagree on whether `ld.so.preload` survives restore; the README overstates what one test catches (regression). | **Fixed** (`43da4f2fb` and earlier) | Corrected; the flag help's "R610+" became accurate when the driver gate became "R610 or newer". |
| R24 | CI never tests the artifact that ships. | **Fixed** (`9e6d8a05a`) | `//tools/mcshim:preload_test` on the built interposer; `//tools/mcshim:glibc_floor_test` checks its glibc versions and library dependencies; `_FORTIFY_SOURCE` pinned. |
| R25 | Open decisions block the runsc PR. | Open | See section 8 (decisions 3 and 4). |

## 6. Nice-to-haves

These can follow the merge as separate PRs.

| ID | Item |
| :--- | :--- |
| N1 | Unit tests for the sentry sequence with an injected `cuda-checkpoint` runner (one case per failure point), marker functions on tmpfs including symlink and FIFO cases, and old/new checkpoint keys. |
| N2 | nvproxy handler tests, an fdinfo hook test, and the cascade when an import's parent is freed. |
| N3 | Match processes by global ID instead of in-namespace PID (multi-container pods can skip the blocker check today). |
| N4 | A killed `cuda-checkpoint` that never exits: mark unrecoverable, skip that process, avoid an unbounded pause. |
| N5 | Check that `/tmp/mcshim` is part of the checkpoint image. |
| N6 | Stale markers: a `gate` left after a failed teardown blocks later CUDA processes; `present.<pid>` is never removed. |
| N7 | Possible deadlock between the gate and the dynamic-linker lock (unverified). |
| N8 | Performance: the control thread polls at 200 Hz in every CUDA process; one release path is quadratic. |
| N9 | Interposer details: `errno` clobbered at load, cached log fd reuse, path truncation, resolver cache invalidation, `-z nodelete`. |
| N10 | Possible blocker false negative through open export fds (needs a GPU to confirm). |
| N11 | Split `mcshim.c` into files and add an explicit export list. |
| N12 | Move the proposal's result tables to PR descriptions. |
| N13 | Keep `build.sh` and `make mcshim`, or build with Bazel only. |
| N14 | Document that `runsc exec` processes are not in the job. |
| N15 | Per-container vs sandbox-wide configuration of the interposer flags. |
| N16 | Revisit the fixed bounds (2-minute pre-save invocations, 5-minute acknowledgements) if they prove too tight; add a flag only then. |
| N17 | A hardware-gated multi-GPU end-to-end test target. |
| N18 | Test-runner robustness (unconditional `sudo`, shared log path). |
| N19 | Environment slice aliasing when appending to the spec's environment. |
| N20 | nvproxy cleanups: a dead parameter check, a duplicated handler, structured blocker results. |
| N21 | R18's guard: write the interposer and the `ld.so.preload` entry only into the rootfs overlay (refuse without one, and refuse a path that crosses a bind mount), treating a refusal like a missing `cuda-checkpoint` (no wrap, no interposer, checkpoints refused). Falling back to `LD_PRELOAD` alone would be worse: SGLang's workers would lose the interposer silently. |
| N22 | A protocol version handshake between the interposer and the sentry (for example, a version in each `gated.<pid>` marker), now that runsc and the interposer ship separately. A mismatch refuses the checkpoint. |

## 7. Merge plan

### 7.1 Dependencies

```
PR0 --device-map (#14817) ---> PR2 control refactor ---> PR7 bounded two-phase checkpoint --+
                                                                                             |
PR4 nvproxy tracking ---+---> PR5 fdinfo line                                                |
                        +---> PR6 blocker inventory ------------------------------------------+--> PR8 interposer protocol
                                                                                                          |
PR3 tools/mcshim ---------------------------------------------------------------------------+            |
                                                                                            +--> PR9 runsc flags, job wrap, preload
PR1 design proposal (settles open questions; informs PR4, PR8, PR9)
```

PR3 depends on nothing and is the largest review, so it should start first,
in parallel with PR0, PR1 and PR2.

### 7.2 PRs

| PR | Contents | Requirements carried | Inert unless enabled? | Size |
| :--- | :--- | :--- | :--- | :--- |
| PR0 | `--device-map` on restore: the existing #14817. Land it, then rebase. | Part of R12 | No: restore onto different GPUs changes, by design and already reviewed | Small, done |
| PR1 | Design proposal, trimmed. Settles: interposer in-tree or not, writing `ld.so.preload`, the fdinfo line as a `/proc` contract, fabric policy. | R23 (proposal), N12 | Docs only | Small |
| PR2 | `state_cuda` groundwork for the interposer path (bounded `wait`, `runCudaAction`, `openCudaCheckpointNullFD`), with the path without the interposer left as #14817 has it; nil-leader fix. | R12 | Yes | Small |
| PR3 | `tools/mcshim`: sources, glibc stubs, BUILD, README, CPU tests, GPU harness. | R1-R6, R23 (README), R24 | Nothing links it | Large |
| PR4 | nvproxy export/import tracking, behind a new option. | R14, R15, R22 | Yes, once gated | Medium |
| PR5 | fdinfo hook for exported objects. | R14 (gating, access check) | Yes | Small |
| PR6 | Checkpoint-blocker inventory. | None | No callers yet | Small |
| PR7 | Bounded pre-save invocations (fixed 2-minute bound on `--get-state`, lock and unlock), lock-then-checkpoint behind the interposer option, the `--get-state` timeout and unlock fixes. No new flag. | R11, R13 | Yes | Medium |
| PR8 | Interposer protocol: `state_cuda_shim.go`, the "suspended" and `cuda-multicast-shim` checkpoint keys, `MCSHIM_DISABLE`. | R9, R10 | Yes | Medium |
| PR9 | runsc: flags, validation, job wrap (supersedes #13987), `LD_PRELOAD` and `ld.so.preload`, save options, the R610-or-newer driver check. May split into job wrap and preload. | R16, R17, R20, R25 | Yes | Medium |

R21 applies to every PR. Each PR branch is cut from `master` at the time,
not from this branch's history.

### 7.3 After the merge

Nice-to-haves as follow-up PRs: N1 and N2 as test-only PRs; N3 to N7 as
robustness fixes; N8 and N11 as interposer refactors; N17 as the GPU target.

### 7.4 Validation

R1-R6, R11, R12, R15 and R16 change code paths that run on GPUs. Before
PR2, PR3, PR4, PR7 and PR9 merge, rerun on the fixed binary:

- The full interposer suite, including the CPU-only `preload`, `noeffect`,
  `glibc` and `ctlsec` tests. (`tables` at its default size, `fuzz 20000 8`, runs past
  400 s with the shim both before and after these fixes; at 2,000 operations
  it passes.)
- `rtres` with a renamed cudart (`libcudart-<hash>.so.12`).
- The stub `cuda-checkpoint` tests, including a hung `--get-state` and the
  `undo` case for a failed checkpoint.
- A checkpoint and restore with `--cuda-checkpoint-path` alone (no
  interposer), in parallel and sequential mode, and one that fails partway,
  to confirm the path without the interposer behaves as upstream (R12).
- A GPU container whose image lacks `cuda-checkpoint` starts, and
  `runsc checkpoint` refuses with its name; a pod's pause container gets
  neither the wrap nor the interposer.
- The 15 engine configurations and the 3 without `CAP_SYS_PTRACE`.
- Qwen2.5-72B at TP=8 on vLLM and SGLang, and restored onto other GPUs.
- A tokens/s comparison with and without the interposer, on one
  configuration without CUDA graphs.

R615 is a separate run. runsc enables the interposer on any driver R610 or
newer, so R615 must be validated before it is deployed there.

## 8. Decisions needed

1. **Fabric handles (R7). Resolved: keep the rewrite.** The shim keeps
   removing fabric handle types in every CUDA process of a container it is
   enabled for. It changes nothing a single-node workload can observe
   (POSIX fds do the same job), and the deployment turns the interposer off
   for multi-node NVLink workloads, which are out of scope and the only ones
   it would affect.
2. **Driver gate (R25). Resolved.** runsc enables the interposer on driver
   R610 or newer and keeps it off, with a warning, on older drivers. There
   is no flag; whether an unvalidated newer release is used is left to the
   deployment (for example, the scheduler).
3. **Where the interposer lives.** In-tree (PR3), or in its own repository.
   See section 9. Either way runsc no longer carries it: `EMBEDDED` mode was
   removed, and the operator provides the interposer at
   `--cuda-multicast-shim-path`.
4. **#13987.** Update it to match PR9, or close it in favour of PR9.

## 9. Alternative under consideration: a separate repository for the interposer

Tabled for now; recorded here so reviewers can weigh it.

The coupling is narrow: the marker directory and file names, the
`MCSHIM_DISABLE` variable, and the fdinfo line format. runsc no longer
embeds the binary.
The interposer needs nothing from gVisor to build, and its test runner
already takes `runsc` as an argument.

| | |
| :--- | :--- |
| Gains | The largest and riskiest upstream review (PR3) leaves gVisor. The Bazel-specific build work (glibc stubs) disappears. The interposer gets its own CI (GPU runners, Docker smoke tests) and release cadence. |
| Costs | The protocol spans two repositories and needs a version handshake (also worth having now that runsc and the interposer ship separately). Restore must find the same interposer bytes at the same path, as it must already. Upstream would want a stub interposer to test the protocol on CPU. |
| Effect on the plan | PR3 leaves the upstream plan, with R1-R6, the README part of R23 and the interposer part of R24. PR8 gains the protocol version check. |
