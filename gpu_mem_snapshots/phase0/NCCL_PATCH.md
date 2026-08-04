# NCCL NVLS suspend/resume patch — design & rationale

Companion to `NCCL_SUSPEND_RESULTS.md` (results) and `nccl-nvls-suspend.patch`
(the exact diff, applyable with `git apply` on base commit
`5067397c2676d5aed50042fc39e5c8ee96eb0027`, tag `nccl4py-v0.3.1`, NCCL
2.30.7). The patch lives as uncommitted working-tree changes in `nccl/`; the
`.patch` file preserves it independently of that tree.

**Status:** proven PASS (native + gVisor, single- and multi-process, NVLS +
captured CUDA graphs), but **superseded as the preferred path** by the
generic `mcshim` interposer (`mcshim/README.md`), which needs no NCCL fork.
The patch remains valuable as (a) the reference implementation the shim
mirrors, (b) the only currently-working path for stock-engine stacks *if*
paired with an engine hook, and (c) an upstreaming candidate (it is a natural
extension of NCCL's own `ncclCommSuspend`).

## Problem

`ncclCommSuspend(comm, NCCL_SUSPEND_MEM)` / `ncclCommResume` (upstream NCCL
>= 2.29.7, `src/mem_manager.cc`) release NCCL's *dynamic* GPU allocations so
a process can be checkpointed. But the mem manager does not touch the **NVLS
multicast layer**: the multicast groups (`NV_MEMORY_MULTICAST_FABRIC`, RM
class 0x00fd) and their UC backing are tracked as `ncclMemPersist` and stay
live. `cuCheckpointProcessCheckpoint` **hangs** on any live multicast object,
so an NVLS comm was un-checkpointable even suspended.

## What the patch adds

Three files, ~194 added lines, no deletions, no behavior change unless
`ncclCommSuspend` is called:

### `src/include/transport.h`
`struct ncclNvlsSharedRes` gains suspend state:
- `int mcSuspended` — idempotence latch,
- `void *buffCpuBackup, *creditCpuBackup` — CPU copies of UC contents while
  the physical memory is released.

Declares `ncclNvlsSuspend(comm)` / `ncclNvlsResume(comm)`.

### `src/transport/nvls.cc`
The implementation, plus no-op stubs in the `CUDART_VERSION < 12010` branch
(so non-NVLS builds still link).

NVLS has (up to) two multicast-backed regions per comm, handled identically
by `nvlsSuspendOne`/`nvlsResumeOne`: **buff** (the NVLS collective buffer)
and **credit** (flow-control credits). For each region there is a UC
(unicast, per-rank physical) allocation mapped at `ucPtr` and a multicast
mapping of the group at `mcPtr`.

**Suspend (per region):**
1. `cudaMemcpy` UC contents to a pinned CPU backup — the FIFO/credit state is
   *live transport state* referenced by persistent conn structs; losing it
   corrupts the comm after resume.
2. `cuMemUnmap(mcPtr)` — the VA **reservation is retained** (never
   `cuMemAddressFree`).
3. `cuMulticastUnbind(dev)` + `cuMemRelease(mcHandle)` — the 0x00fd object is
   gone; the checkpoint blocker disappears.
4. `cuMemUnmap(ucPtr)` + `cuMemRelease(ucHandle)` — reservation retained.

Teardown uses `CUCHECKIGNORE`: suspend must make progress even if the driver
reports an already-degraded state.

**Resume (per region), mirroring `ncclNvlsBufferSetup`'s rendezvous:**
1. Local rank 0 `ncclNvlsGroupCreate` (create + export shareable handle) and
   `bootstrapIntraNodeBroadcast`s the handle; peers `ncclNvlsGroupConnect`
   (import). Same bootstrap channel as initial setup — this is what a generic
   interposer has to reinvent (the mcshim's unix-socket fd service).
2. `cuMulticastAddDevice` on every rank.
3. `cuMemCreate` new UC memory, `cuMemMap` at the **identical** `ucPtr`,
   `cuMemSetAccess`, restore contents from the CPU backup, free the backup.
4. `bootstrapIntraNodeBarrier`, then `cuMulticastBindMem` — the bind blocks
   until all devices joined; the barrier prevents a rank from binding before
   a peer has re-created its UC memory (same ordering as setup).
5. `cuMemMap` the group at the **identical** `mcPtr` + `cuMemSetAccess`.

**Wrappers `ncclNvlsSuspend`/`ncclNvlsResume`:**
- no-op when `nvlsSupport == 0`, resources are NULL, or already
  suspended/resumed (idempotent);
- suspend refuses shared NVLS resources (`refCount > 1`, comm-split sharing)
  with `ncclInvalidUsage` rather than corrupting a sibling comm — a real,
  documented limitation;
- `ncclMemUntrack`/`ncclMemTrack` keep the mem manager's accounting
  consistent so `ncclCommMemStats` and a later real free stay correct.

### `src/mem_manager.cc`
Wires the pair into the existing API: `ncclCommMemSuspend` calls
`ncclNvlsSuspend` as its final step; `ncclCommMemResume` calls
`ncclNvlsResume` as its final step (after peer-import re-setup, before the
final barrier). So the public API surface is unchanged — apps/engines call
the *upstream* `ncclCommSuspend`/`ncclCommResume` entry points.

## Why the VAs must be identical (the core invariant)

Captured CUDA graphs, kernel launch params, and NCCL's persistent conn
structs all reference `ucPtr`/`mcPtr` values captured at setup. CRIU/
cuda-checkpoint restore the process image verbatim, so if resume re-maps at
the same addresses, nothing above libcuda ever observes the teardown. Only
the opaque CUmem/multicast *handles* change, and only teardown paths look at
those (which the patch updates in `ncclNvlsSharedRes`). This is the same
invariant TASK.md demands and the same one the mcshim enforces.

## Ordering contract (who calls what, when)

```
(a) engine idle: no in-flight collectives, no new comms
(b) every rank: ncclCommSuspend(comm, NCCL_SUSPEND_MEM)   <- releases NVLS MC
(c) cuda-checkpoint lock -> checkpoint   (per rank; job-wrapped under gVisor)
    ... save / restore ...
    cuda-checkpoint restore -> unlock
(d) every rank: ncclCommResume(comm)     <- collective! see below
(e) engine resumes issuing work
```

`ncclCommResume` is **collective across the comm's local ranks** and must be
in flight on all ranks concurrently: `cuMulticastBindMem` blocks until every
device has re-joined the group. Serialized resume across ranks deadlocks
(TASK.md work item 4's concern, confirmed).

## Limitations

- `refCount > 1` (shared NVLS resources across split comms) is rejected at
  suspend.
- NVLS **user-buffer registrations** (`tryRegisterBuffer` /
  `ncclNvlsGraphRegisterBuffer`, which bind app buffers via
  `cuMulticastBindAddr`) are not covered — the harnesses don't exercise
  registration, and engines using it would need those regRecords suspended
  too.
- Requires the engine (or a wrapper) to call suspend/resume around the
  checkpoint — the integration gap that motivated the mcshim
  (`NCCL_SUSPEND_RESULTS.md` § "Testing the actual vLLM/SGLang cases").

## Cross-rank UC imports — why the patch passes where the bare shim doesn't

Beyond NVLS, `ncclCommMemSuspend`'s *upstream* steps already unmap and
release the P2P buffers each rank **imported from its peers**
(`cuMemImportFromShareableHandle`), and `ncclCommMemResume` re-imports them.
R610 cuda-checkpoint (job mode included) refuses to restore a process holding
live VMM UC imports (proven minimally: `ipc_taint.py --mode hold` under
`--launch-job`). The patched-NCCL flow is therefore *complete*: upstream
suspend handles UC imports, this patch handles multicast. The mcshim
currently handles only multicast — closing its UC-import gap
(`mcshim/README.md` § next steps) reaches parity.

## Build

Host glibc 2.41 + CUDA 13 device compile is incompatible; build in a
container (see `NCCL_SUSPEND_RESULTS.md`):

```
cd nccl && rm -rf build
sudo docker run --rm -v "$PWD":/nccl -w /nccl nvidia/cuda:13.0.1-devel-ubuntu24.04 \
  bash -c 'apt-get update -qq && apt-get install -y -qq git python3 && \
           make src.build -j64 NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"'
cp build/lib/libnccl.so.2.30.7 /opt/phase0/nccl/nvidia/nccl/lib/libnccl.so.2
```

Deploying into an engine image: replace the torch-bundled `libnccl.so.2` or
`LD_PRELOAD` it (ABI-compatible for torch's public-API use).

## Re-applying / upstreaming

- Recreate on a clean tree: `cd nccl && git apply ../gpu_mem_snapshots/phase0/nccl-nvls-suspend.patch`
- Upstreaming: the change is additive, gated behind the existing
  `ncclCommSuspend` entry point, and no-ops on non-NVLS comms — a natural
  proposal to NVIDIA once the shared-resources (`refCount > 1`) and
  registered-buffer cases are addressed.
