# Multi-GPU CUDA Checkpoint/Restore

Status as of 2026-10-14: Draft, seeking feedback on the
[open questions](#open-questions).

## Synopsis

Make `runsc checkpoint` / `runsc restore` work for single-node, multi-process,
multi-GPU CUDA workloads, e.g. tensor-parallel (TP) inference engines like
vLLM and SGLang, including restore onto a different set of GPUs, without
modifying the application, its libraries (NCCL, PyTorch, FlashInfer), or the
NVIDIA driver.

gVisor already checkpoints CUDA processes by running NVIDIA's
`cuda-checkpoint` on each of them (`runsc checkpoint --cuda-checkpoint-path`).
That works for processes that share no GPU state. Multi-GPU workloads do share
it: ranks map each other's GPU memory and, on NVSwitch systems, create NVLink
multicast groups, and `cuda-checkpoint` can neither checkpoint nor restore
that state.

This proposal adds three small, generic pieces to gVisor:

1.  **`cuda-checkpoint` jobs** (#13987): run each GPU container in a
    `cuda-checkpoint --launch-job` job, which lets R610's `cuda-checkpoint`
    carry legacy CUDA IPC.
2.  **A checkpoint hook**: a program inside the container that runsc runs in
    place of the built-in `cuda-checkpoint` handling, at fixed points of a
    checkpoint and a restore.
3.  **Exported-object identity in fdinfo** (nvproxy, #15761): which RM object an
    exported fd carries, as Linux reports for dma-buf.

The rest, an interposer that releases multicast groups and imported memory
before the checkpoint and rebuilds them at identical virtual addresses after
restore, and the hook that drives it and `cuda-checkpoint`, lives outside
gVisor, in [mcshim](https://github.com/modal-projects/mcshim), much as
`cuda-checkpoint` does.

Non-goals:

-   Multi-node / MNNVL (IMEX, fabric-handle memory shared across hosts).
-   NVIDIA drivers older than R610.
-   Checkpointing device memory contents. That remains `cuda-checkpoint`'s job.
-   Isolation. Hooks and interposers run in the container's trust domain.

## Background

### What `cuda-checkpoint` cannot do

Measured with a PyTorch/NCCL test program at TP=2 and TP=4 on 8x H100 with
NVSwitch, driver 610.57.04, without the interposer:

Cross-process GPU state                                    | Created by                                                         | Result
---------------------------------------------------------- | ------------------------------------------------------------------ | ------
NVLink multicast group (`NV_MEMORY_MULTICAST_FABRIC`)      | NCCL NVLS, PyTorch symmetric memory, FlashInfer all-reduce fusion  | `cuda-checkpoint` hangs (killed after 240 s)
VMM IPC import (`cuMemImportFromShareableHandle`)          | NCCL peer-to-peer with `NCCL_CUMEM_ENABLE=1`                       | checkpoint succeeds, restore fails (`invalid argument`)
Legacy CUDA IPC import (`cuIpcOpenMemHandle`)              | NCCL with cuMem disabled, the engines' custom all-reduce           | `cuda-checkpoint` hangs (killed after 300 s)

R610's job mode (`cuda-checkpoint --launch-job`) fixes the last row, and only
that one. In job mode `cuda-checkpoint` also carries IPC events and dma-buf
exports; it cannot carry managed memory ("operation not supported").

Both engines hit this in their default configurations. vLLM 0.29.0 at TP=2
dispatches all-reduce to FlashInfer, custom and symmetric-memory backends
ahead of NCCL, and one rank holds 4 multicast objects; SGLang 0.5.20 at TP=4
holds 2. Without the interposer, `cuda-checkpoint` hangs in both cases. Avoiding all of this means disabling
NVLS, custom all-reduce and symmetric memory in the engine and forcing NCCL off
its IPC transports, which gives up the intra-node NVLink paths these engines
are built around.

### Why not in the sentry

nvproxy already keeps a live RM object graph and replays it on restore, so
replaying multicast groups and imports from below was the first approach we
tried. It does not work:

1.  `cuda-checkpoint`'s restore rebuilds a process's RM client through paths
    nvproxy never sees, and RM rejects sentry-issued allocations into the
    restored client with `NV_ERR_INSUFFICIENT_PERMISSIONS` (measured on R580).
2.  libcuda keeps its own bookkeeping for this state, which goes stale if the
    objects change underneath it. For example, an allocation bound into a
    multicast group and left resident across the checkpoint restores fine,
    but its next export fails with `NV_ERR_OBJECT_NOT_FOUND` (measured on
    R610).

The CUDA driver API is the lowest layer at which this state can be torn down
and rebuilt consistently. An interposer there covers NCCL, PyTorch, FlashInfer
and the engines without patching any of them. Nothing in this proposal has the
sentry free or recreate RM objects on the application's behalf.

## Design

### `cuda-checkpoint` jobs (runsc)

`--cuda-checkpoint-job=launch`, with `--cuda-checkpoint-path`, wraps the
command of each container given GPU access in `cuda-checkpoint --launch-job`
on driver R610 or newer, and makes `runsc checkpoint` run `cuda-checkpoint`
one process at a time, as jobs require (in parallel, SGLang's restore fails).
A GPU container that cannot be wrapped (its image lacks `cuda-checkpoint`, or
its entrypoint is `--exec-fd`) still starts, outside a job, and checkpoints
are refused while it exists: `cuda-checkpoint` hangs on legacy IPC outside a
job. Containers without GPU access (a pod's pause container, sidecars) are
never wrapped.

The wrap is a convenience: an operator that builds the spec can write
`cuda-checkpoint --launch-job` into the entrypoint itself, and with the
checkpoint hook below, which runs `cuda-checkpoint` itself, needs no runsc
flag at all. runsc then cannot refuse a checkpoint of a GPU container left
outside the job; the operator guarantees there is none.

### The checkpoint hook (`pkg/sentry/control`)

The root container's annotation `dev.gvisor.internal.checkpoint.hook=PATH`
names, by absolute path, a program inside each GPU container. It adds no runsc
flag: like the other checkpoint annotations
(`dev.gvisor.internal.checkpoint.*`), it is set in the spec by the operator,
per sandbox, and read when a checkpoint starts; the path is kept in the
checkpoint for the restore. With it, the sentry does not run
`cuda-checkpoint` itself; for each container with a
process holding an NVIDIA device open, it runs the hook with the container's
mount and PID namespaces and environment:

Phase                               | When                                                         | Page faults
----------------------------------- | ------------------------------------------------------------ | -----------
`<hook> checkpoint`                 | before the sandbox is saved                                  | held
`<hook> restore [--device-map MAP]` | after a restore, or a save with `--resume` (successful or not) | held
`<hook> resume`                     | after `restore` has finished in every container              | released

`MAP` is the `--device-map` value `cuda-checkpoint` takes (#14817), passed
when the sandbox was restored onto different GPUs.

Page faults of the GPU processes are held (`ThreadGroup.SigsegvLock`) while
their GPU memory is away, as for the built-in path. That is the one ordering
constraint the sentry imposes: work that touches GPU memory faults forever
while they are held (every rank failed with `CUDA_ERROR_ILLEGAL_ADDRESS` when
an interposer rebuilt its state before they were released), so it belongs in
`resume`.

`checkpoint` exits 0 on success. Status 1 means it failed and undid its work,
except for what `resume` completes, which the sentry then runs, after
releasing page faults; the application keeps running. Any other status (or a
crash) means it could not undo its work: the sentry does not run `resume`, and
the application may be left blocked. If one container's `checkpoint` fails
after another's succeeded, the sentry runs `restore` and `resume` in the
latter. A failed `restore` or `resume` fails the restore, which kills the
sandbox, as a failed `cuda-checkpoint` restore does today. The hook has no
time limit: its duration grows with GPU memory, and it bounds its own steps.

When the kernel is paused (Docker pauses before it checkpoints), the sentry
unpauses it around the hook, as it does for `cuda-checkpoint`. The hook's
path and the processes it ran for are kept in the checkpoint, so `runsc
restore` needs no flags.

### Exported-object identity (nvproxy)

Every fd that `cuMemExportToShareableHandle` returns is an open of
`/dev/nvidiactl`: `fstat` cannot tell them apart, and a process that receives
one over `SCM_RIGHTS` cannot tell which allocation it carries. nvproxy records,
when an RM object is exported into an fd
(`NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT(S)_TO_FD`), the object's identity, and
reports it in `/proc/[pid]/fdinfo/[fd]` to readers allowed to ptrace the
process, as Linux does for dma-buf:

```
nvproxy_exported_object:	client=0xc1d00922 object=0x5c000123
```

Exporters and importers read the same line, since they share the file
description. An interposer uses it to pair each import with its export when
it rebuilds them after a restore.

### mcshim (out of tree)

[mcshim](https://github.com/modal-projects/mcshim) is the hook this was built
for: an `LD_PRELOAD` interposer, which the operator preloads into the GPU
containers (`LD_PRELOAD`, and `/etc/ld.so.preload` for launchers that rewrite
the environment), and `mcshim-ckpt`, its hook. On `checkpoint`, `mcshim-ckpt`
gates the application's GPU work, locks every process with `cuda-checkpoint`,
has the interposer release multicast groups and imported memory (keeping the
virtual address reservations), re-locks, and checkpoints one process at a
time. On `restore` it restores every process with `cuda-checkpoint`; on
`resume` it has the interposer rebuild its state at the same addresses and
releases the gate. Its design, tests and limitations are documented there.

## Validation

Two hosts, both with driver and fabric manager 610.57.04: 8x H100 80GB with
NVSwitch, and 8x B300 (NVLink 5). The H100 results are from the current
revision; the B300 results are from the revision before the interposer
reviews. vLLM 0.29.0 and SGLang 0.5.20 serve
Qwen2.5-1.5B-Instruct in their default configurations plus the listed flags.
Each run boots the engine, records a completion at temperature 0, puts the
engine to sleep, checkpoints, restores (onto other GPUs where listed), wakes
it, and repeats the completion. A run passes if the output is identical and
the sandbox is on the requested GPUs.

All runs use job mode and the engines' custom all-reduce. TP=8 serves Qwen2.5-3B-Instruct, since the 1.5B model's 12
attention heads do not split 8 ways.

H100:

Workload                                  | GPUs       | Checkpoint (image) | Restore | First inference after restore | vs. cold boot
----------------------------------------- | ---------- | ------------------ | ------- | ----------------------------- | -------------
vLLM TP=2                                 | 0,1        | 8.7 s (12G)        | 1.4 s   | 5.3 s                         | 42x
vLLM TP=4                                 | 0-3        | 14.4 s (18G)       | 2.3 s   | 9.6 s                         | 26x
vLLM TP=8                                 | 0-7        | 36.6 s (35G)       | 4.5 s   | 22.9 s                        | 14x
vLLM TP=2                                 | 0,1 -> 4,5 | 8.7 s (12G)        | 1.5 s   | 5.6 s                         | 41x
vLLM TP=4                                 | 0-3 -> 4-7 | 14.2 s (18G)       | 2.4 s   | 10.5 s                        | 24x
vLLM TP=2                                 | 0,1 -> 1,2 | 8.5 s (12G)        | 1.4 s   | 5.6 s                         | 40x
SGLang TP=4                               | 0-3        | 15.1 s (17G)       | 2.2 s   | 9.6 s                         | 19x
SGLang TP=4                               | 0-3 -> 4-7 | 15.0 s (17G)       | 2.2 s   | 10.8 s                        | 17x
SGLang TP=4 `--enable-nccl-nvls`          | 0-3        | 15.3 s (17G)       | 2.2 s   | 10.1 s                        | 18x
SGLang TP=4 `--enable-torch-symm-mem`     | 0-3        | 15.4 s (17G)       | 2.2 s   | 10.0 s                        | 18x
SGLang TP=4 FlashInfer all-reduce fusion  | 0-3        | 16.7 s (18G)       | 2.3 s   | 10.5 s                        | 17x
SGLang TP=4 FlashInfer all-reduce fusion  | 0-3 -> 4-7 | 16.9 s (18G)       | 2.3 s   | 11.7 s                        | 15x
SGLang TP=4 `--enable-nccl-nvls`          | 0-3 -> 4-7 | 15.2 s (17G)       | 2.3 s   | 11.1 s                        | 16x
SGLang TP=4 `--enable-torch-symm-mem`     | 0-3 -> 4-7 | 15.3 s (17G)       | 2.3 s   | 11.1 s                        | 16x
SGLang TP=8                               | 0-7        | 42.5 s (33G)       | 4.3 s   | 23.2 s                        | 9x

B300:

Workload                                  | GPUs       | Checkpoint (image) | Restore | First inference after restore | vs. cold boot
----------------------------------------- | ---------- | ------------------ | ------- | ----------------------------- | -------------
vLLM TP=2                                 | 0,1        | 8.6 s (14G)        | 1.6 s   | 6.3 s                         | 26x
vLLM TP=4                                 | 0-3        | 15.2 s (21G)       | 2.1 s   | 12.0 s                        | 15x
vLLM TP=8                                 | 0-7        | 37.0 s (41G)       | 3.7 s   | 32.1 s                        | 8x
vLLM TP=2                                 | 0,1 -> 4,5 | 8.6 s (14G)        | 1.6 s   | 6.4 s                         | 25x
vLLM TP=4                                 | 0-3 -> 4-7 | 15.1 s (21G)       | 2.0 s   | 12.8 s                        | 14x
vLLM TP=2                                 | 0,1 -> 1,2 | 8.6 s (14G)        | 1.6 s   | 6.5 s                         | 25x
SGLang TP=4                               | 0-3        | 18.2 s (19G)       | 2.1 s   | 13.8 s                        | 13x
SGLang TP=4                               | 0-3 -> 4-7 | 18.0 s (19G)       | 2.1 s   | 14.5 s                        | 12x
SGLang TP=4 `--enable-nccl-nvls`          | 0-3        | 18.4 s (19G)       | 2.1 s   | 13.7 s                        | 13x
SGLang TP=4 `--enable-torch-symm-mem`     | 0-3        | 18.5 s (19G)       | 2.1 s   | 13.9 s                        | 13x
SGLang TP=4 FlashInfer all-reduce fusion  | 0-3        | 20.7 s (20G)       | 2.2 s   | 14.5 s                        | 12x
SGLang TP=4 FlashInfer all-reduce fusion  | 0-3 -> 4-7 | 20.6 s (20G)       | 2.1 s   | 15.1 s                        | 12x
SGLang TP=4 `--enable-nccl-nvls`          | 0-3 -> 4-7 | 18.5 s (19G)       | 2.0 s   | 14.5 s                        | 12x
SGLang TP=4 `--enable-torch-symm-mem`     | 0-3 -> 4-7 | 18.7 s (19G)       | 2.1 s   | 14.5 s                        | 12x
SGLang TP=8                               | 0-7        | 43.9 s (37G)       | 3.4 s   | 34.3 s                        | 6x

Timings are from a single run per host; repeat runs varied by up to about
30%. The interposer's share is small: on H100, arming the gate takes about
0.1 s, the teardown 0.1 to 0.4 s (0.7 s at TP=8), and the rebuild 0.4 to
0.8 s (1.1 s at TP=8). vLLM TP=4, SGLang TP=4 with symmetric memory, and
the vLLM TP=2 cross-GPU restore also pass on H100 in containers without
`CAP_SYS_PTRACE`, where `pidfd_getfd` relies on the exporters'
`PR_SET_PTRACER_ANY`.

Job mode requires running `cuda-checkpoint` one process at a time. Against
running it in parallel (which an earlier version allowed; see
[Alternatives](#alternatives-considered)), that costs 2 to 3 s at TP=4. In an
alternating comparison with the page cache dropped before each run (3 runs
each of vLLM TP=4 and SGLang TP=4 with and without fusion), checkpoints took
14.5 to 16.2 s instead of 12.1 to 13.2 s, and the first inference after
restore 10.9 to 12.0 s instead of 8.6 to 9.2 s.

These results are from the previous revision of this proposal, in which the
sentry drove the interposer itself (see
[Alternatives](#alternatives-considered)). The hook runs the same sequence
from inside the container; the matrix is to be rerun with it before the hook
merges.
## Alternatives considered

-   **RM-layer replay in nvproxy.** Rejected: see
    [Why not in the sentry](#why-not-in-the-sentry).
-   **Patching the libraries.** A fork of NCCL with communicator
    suspend/resume, driven by the engines' sleep endpoints, handles NCCL's own
    state but not PyTorch symmetric memory, the engines' custom all-reduce, or
    FlashInfer. Each would need its own patch, maintained against upstream.
-   **`cuda-checkpoint` job mode alone** (`--launch-job`, R610). It carries
    legacy IPC, but still hangs on multicast and fails to restore VMM
    imports: with the interposer handling only multicast, vLLM TP=4's restore
    fails with `invalid argument` on every rank holding VMM imports.
-   **Legacy IPC in the interposer.** An earlier version replaced each
    `cudaMalloc`'d range the application exported, at its original address,
    by an exported VMM allocation, making legacy IPC ordinary VMM IPC. It
    worked in every configuration tested and allowed parallel
    `cuda-checkpoint` (2 to 3 s faster at TP=4, see [Validation](#validation)),
    but cost ~400 lines of interposer code that job mode makes unnecessary.
-   **Disabling the features.** Works today, at the performance cost described
    above.
-   **The interposer's protocol in the sentry.** The previous revision of
    this proposal had the sentry drive the interposer directly: marker files
    in the container, the lock/teardown/checkpoint sequence, a protocol
    version check, and an nvproxy inventory of objects `cuda-checkpoint`
    would hang on (about 1,700 lines in `pkg/sentry/control` and 450 in
    nvproxy). It worked, but put a CUDA-specific protocol in the sentry that
    would have to shrink with every `cuda-checkpoint` release that carries
    more state. The hook moves all of it into the container.
-   **Waiting for NVIDIA.** Preferred long term. With the hook, gVisor carries
    nothing interposer-specific, so nothing in gVisor has to be removed when
    `cuda-checkpoint` handles this state itself.

## Open questions

1.  **A generic hook.** Is the hook annotation acceptable as a runtime
    feature, given that it runs a container-provided program during a
    checkpoint? Should it be CUDA-specific in name
    (`dev.gvisor.internal.checkpoint.cuda-hook`), since it runs only in GPU
    containers and its `restore`/`resume` split exists for GPU memory?
2.  **Where the hook runs.** In each container with a process holding an
    NVIDIA device open. Should it instead run in every container, or only in
    containers that opt in through an annotation?
3.  **The fdinfo line as a `/proc` contract.** Always shown, to readers with
    ptrace access, like dma-buf's. Acceptable?
4.  **Testing.** The sentry's hook sequence has unit tests against a fake
    hook; the job wrap and fdinfo line have unit tests; mcshim tests its own
    sequence on CPU against fakes, and end to end on GPUs. Is a
    hardware-gated `test/gpu` target wanted in gVisor as well?

## Rollout

As independent PRs, each inert unless its flag is set:

1.  `--device-map` on restore (#14817).
2.  `cuda-checkpoint` jobs (#13987, updated).
3.  Exported-object identity and the fdinfo line (#15761).
4.  The checkpoint hook (`pkg/sentry/control`, `runsc`).

## Appendix: operational notes

-   vLLM and SGLang are checkpointed after putting the engine to sleep (vLLM
    `/sleep?level=1`; SGLang `release_memory_occupation` with
    `--enable-memory-saver --enable-weights-cpu-backup`; without the backup
    flag the restored engine produced garbage output). A checkpoint under
    load may fail the lock phase; the application keeps running.
-   Only a single checkpoint then restore is validated. Chained restores,
    resuming after a successful save, and the failure unwinds have not been
    re-validated on R610, nor anything with the hook.
-   Restore performance depends on host settings:
    `transparent_hugepage/shmem_enabled=always` and pinning the sandbox to the
    GPUs' NUMA node.
