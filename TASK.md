# Task: Multicast object suspend/restore in gVisor nvproxy

## Goal

Make a single-node, multi-GPU (4–8x Hopper) CUDA process checkpointable and
restorable while it holds **multicast fabric objects**, without patching NCCL,
PyTorch, or the application.

Today `cuCheckpointProcessCheckpoint` refuses to checkpoint a process holding
IPC memory created via `cuMemExportToShareableHandle()`, and multicast objects
are never released by any userspace suspend path. Both NCCL's NVLS and
PyTorch's `torch.distributed._symmetric_memory` create the *same* RM objects,
so one fix at the nvproxy layer covers both.

The invariant that makes this work: after restore, every RM handle and every
GPU virtual address must be **identical** to its pre-checkpoint value. CRIU
restores libcuda's internal structures verbatim, so if handles and VAs match,
userspace never observes the teardown. Captured CUDA graphs remain valid for
the same reason — their node parameters reference VAs that come back unchanged.

## Existing mechanism (do not rebuild)

`pkg/sentry/devices/nvproxy/`:

- `object.go` — live object graph per `rootClient`: `objAdd`/`objFree`,
  per-object `class`, `handle`, `parent`, `deps`/`rdeps`.
- `save_restore.go`:
  - `nvproxy.beforeSave()` panics if any live object's `impl` does not
    implement `restorableObjectImpl`.
  - `nvproxy.afterLoad()` topologically sorts live objects by `deps`, then
    calls `Restore(ctx)` on each in dependency order.
  - `rmAllocObject.Restore()` replays the saved `NV_ESC_RM_ALLOC` params.
  - `rootClient.Restore()` re-requests the original handle via `HRoot`.
  - `DeviceRemapping` / `DeviceRemapID` handle GPU identity across restore.
- `version.go` — `NV_MEMORY_MULTICAST_FABRIC` (00FD), `NV_MEMORY_FABRIC` (00F8),
  `NV_MEMORY_FABRIC_IMPORTED_REF` (00FB) are already allowlisted, with a
  `rmAllocMulticastFabric` handler and V545/V590 param variants.
- `handlers.go` — per-command `NV_ESC_RM_CONTROL` dispatch.
- `frontend_mmap.go` — sentry-owned device memory mappings.

The object graph is **live state, not an ioctl log**. Do not add a warmup
recording phase or a replayable ioctl stream. Freed objects must drop out of
the replay set automatically, which the existing graph already guarantees.

## Work items

### 1. Blocker inventory and gate (do this first, ship independently)

Add to `nvproxy`:

```go
type CheckpointBlocker struct {
    ClientHandle nvgpu.Handle
    ObjectHandle nvgpu.Handle
    Class        nvgpu.ClassID
    TaskID       int32   // recorded at objAdd
    Kind         string  // "multicast" | "fabric-import" | "exported-fd"
}

func (nvp *nvproxy) CheckpointBlockers() []CheckpointBlocker
```

- Walk `nvp.clients[*].resources`; report objects of class 00FD, 00F8, 00FB.
- Maintain a counter for `NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD`,
  decremented when the exported FD is closed.
- Record the creating task in `objAdd` so blockers are attributable to a rank.
- Wire into `pkg/sentry/control/state_cuda.go` `preSaveCuda`, before the
  existing checkpoint sequence: poll with a configurable timeout, then fail
  with a **per-client** message (`"rank 5 (client 0x...): 3 multicast objects"`),
  not an aggregate count.

**Acceptance:** running the harness below on 8x H100 and triggering a
checkpoint prints a non-empty, per-rank blocker list naming both the NCCL and
symmetric-memory multicast objects.

### 2. Record state-mutating control calls

`rmAllocObject.Restore()` replays only the allocation. A multicast object's
state is built afterward by control calls, so a replayed 00FD object comes back
empty.

- Add a **narrow allowlist** of state-mutating controls whose effects are
  recorded on the target object: at minimum `NV00FD_CTRL_CMD_ATTACH_GPU` and
  `NV00FD_CTRL_CMD_ATTACH_MEM` (and their detach counterparts, which remove
  the recorded entry).
- Store the recorded set on the object; replay it inside that object's
  `Restore` after the alloc replay.
- Do not record controls outside the allowlist.

**Acceptance:** unit test asserts that alloc-then-attach-then-save produces a
recorded attach set that survives the state round-trip.

### 3. Multicast suspend before `cuda-checkpoint`

`cuCheckpointProcessCheckpoint` releases GPU-side state and will still refuse
while multicast objects are live. nvproxy must therefore **tear the multicast
layer down before** the checkpoint and replay it after restore.

Sequence in `preSaveCuda`:

1. Sentry unmaps multicast VAs (retain the VA reservations — never release the
   address range).
2. Issue detach controls; free the 00FD objects and any 00FB imported refs.
3. Blocker list is now empty → run `cuda-checkpoint`.
4. Save.

On restore, `afterLoad` replays alloc + recorded attach controls with identical
handles, and remaps at identical VAs.

**Acceptance:** VA values printed by the harness are byte-identical before and
after restore; captured CUDA graphs replay to correct results.

### 4. Batched attach across clients

The `afterLoad` topological sort restores objects serially across all clients.
If RM-level `ATTACH_GPU` inherits the blocking semantics of
`cuMulticastAddDevice` (blocks until all participating GPUs join), a serial
replay across 8 ranks deadlocks.

- **Measure this with 2 ranks before implementing.**
- If blocking: group attaches per multicast object and issue them across
  clients before advancing the sort. Add a timeout with a clear panic message
  rather than allowing an indefinite hang.

## Measure before implementing: IPC taint

Does a `cuMemCreate` allocation with
`requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` become
checkpointable again once every export FD is closed and every peer import is
released — or is it permanently IPC-tainted?

Standalone test, two processes, one GPU each, no NCCL, no PyTorch:

1. `cuMemCreate` with POSIX_FD handle type, `cuMemMap`, write a known pattern.
2. Export, pass the FD, peer imports and maps.
3. Peer unmaps and `cuMemRelease`s the imported handle; both sides close all FDs.
4. `cuda-checkpoint` the exporter, restore, verify the pattern.

**If it passes:** work items 1–4 are sufficient; unicast allocations stay
resident and device memory remains `cuda-checkpoint`'s responsibility.

**If it fails:** nvproxy must also tear down and replay the unicast
allocations, which makes device memory content nvproxy's problem. Stop and
escalate — that is a materially larger design.

## Non-goals

- Multi-node / MNNVL. Single-node keeps `ncclCuMemHandleType` at
  `POSIX_FILE_DESCRIPTOR`; the `CU_MEM_HANDLE_TYPE_FABRIC` path is out of scope.
- Ampere and Blackwell. Ampere has no multicast support at all; Blackwell comes
  after Hopper works.
- Any patch to NCCL, PyTorch, or the application.
- Multicast slot contention on restore. Accepted risk for now — but note that
  attach failure manifests as a **hang, not an error**, so every replay path
  needs a timeout and a loud failure.
- Network transport, IB QPs, GDRCopy.

## Verification harness

`symmem_nccl_ckpt_test.py` (accompanying). Run under `runsc` with nvproxy on
4–8 H100s. It captures CUDA graphs containing both NCCL NVLS collectives and
PyTorch symmetric-memory multicast collectives, then verifies continuously in a
loop. Checkpoint and restore the sandbox mid-loop; the harness prints VA
inventories and pass/fail per iteration, and detects the restore via a
wall-clock jump.

Pass criteria: no verification failure across the restore boundary, and
identical VA inventory before and after.
