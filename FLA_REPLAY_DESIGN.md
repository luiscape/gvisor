# Design: suspend/replay of persistent FLA registrations (NV_MEMORY_FABRIC 00f8)

Status: designed, not implemented. This is the single remaining blocker class
for checkpointing engaged multicast features (torch symm-mem multimem,
FlashInfer all-reduce fusion) under `fabric-imex-mgmt`.

## Problem (fully characterized, 2026-08-19)

On fabric-attached GPUs with the RM fabric probe allowed, libcuda lazily
creates one driver-internal `NV_MEMORY_FABRIC` (00f8) object per
exporter-side VMM allocation that peers import: an FLA *registration* whose
alloc params carry `Map.HVidMem != 0` pointing at the covered vidmem. It is
metadata over content that lives elsewhere; nothing in userspace ever frees
it while the allocation lives.

Evidence chain:
- torch symm-mem workspace: 1 live 00f8/rank at gate time (57/rank in pause
  mode), `allocSize=0x20200000 map.hVidMem=0x5c000328` (the workspace).
- FlashInfer `--flashinfer-allreduce-fusion-backend trtllm`: identical
  signature, 1 live/rank at TP=4 (252 created / 248 shim-released) and TP=8
  (632 / 624).
- cuda-checkpoint *checkpoints* a live 00f8 but cannot *restore* it: with the
  gate bypassed (arm_d6 experiment), checkpoint succeeded and the restored
  process died in `NV_ERR_OBJECT_NOT_FOUND` (0x57) storms; the restore boot
  log contains **zero** 00f8 allocations (cuda-checkpoint never recreated it,
  libcuda's restored bookkeeping still references its handle).
- Importer-side/transient 00f8s are already released by the interposer's
  suspend (imports + unbinds); only the exporter-side registration survives.
- No CUDA API frees it. Only nvproxy has the handle and the saved params.

## Fix shape: nvproxy frees before checkpoint, replays after the toggle

Mirrors the interposer's philosophy (tear down before cuda-checkpoint,
rebuild after), but at the RM level where this object lives. Invariant
preserved: identical client + handle + parent + params at replay, so
libcuda's restored bookkeeping never notices (same argument as TASK.md).

### Checkpoint side (state_cuda.go, checkpointCudaProcs, shim path only)

Sequence today: lock -> unlock -> shim suspend -> strict gate -> re-lock ->
checkpoint. Insert between re-lock and checkpoint (procs locked, GPU
quiesced, transient 00f8s already gone):

1. `nvproxy.SuspendFabricRegistrations(vfsObj)`:
   - Walk `nvp.clients[*].resources` for live 00f8 objects whose saved
     alloc params (`rmAllocObject`, already retained) have `Map.HVidMem != 0`
     (the registration flavor -- never free content-bearing fabric memory,
     which only exists under `MCSHIM_ALLOW_FABRIC=1`).
   - Host `NV_ESC_RM_FREE` each via the owning frontendFD's hostFD (new
     small helper next to the existing host-invoke helpers in
     `save_restore_unsafe.go` / `multicast_unsafe.go`).
   - Do NOT objFree from the graph: mark the object `hostFreed=true`
     (new field, savable). The graph entry carries everything replay needs.
2. Re-verify zero blockers (gate must skip `hostFreed` objects), then
   checkpoint.
3. Unwind path (checkpoint fails afterwards): immediately re-alloc the freed
   registrations via the replay helper and clear `hostFreed`, before the
   restore-toggle/unlock unwind.

### Restore side (postRestoreCuda)

Sequence today: afterLoad (nvproxy object replay) -> cuda toggle
(cuda-checkpoint recreates process GPU state, incl. the vidmem handles the
registrations reference) -> shim resume. Changes:

1. afterLoad's topological restore SKIPS `hostFreed` objects (their
   dependencies -- subdevice parent, hVidMem -- do not exist until the
   toggle).
2. After `restoreCudaProcs` (toggle) and before the shim resume: new
   `nvproxy.ReplayDeferredObjects(ctx)` -- host `NV_ESC_RM_ALLOC` with the
   saved hObjectNew/parent/params on each owning client's hostFD (the
   existing rmAllocObject replay path, invoked post-toggle instead of in
   afterLoad), then clear `hostFreed`.

### Gate change

`checkpointBlockers()` skips objects with `hostFreed=true`. No new exemption
kinds; the strict gate stays strict.

## Empirical unknowns to verify first (cheap, in order)

1. Host RM_FREE of a 00f8 on a locked process's client: accepted by RM?
   (Expected yes -- RM ioctls are client-scoped, not process-scoped.)
2. Post-toggle RM_ALLOC of 00f8 with the original hObjectNew: does the
   toggle restore the referenced vidmem under the same handle (it must --
   cuda-checkpoint's contract with libcuda), and does RM accept the alloc
   from the sentry-held fd? Probe with a 2-rank torch symm-mem multimem run
   under CB_IMEX=1.
3. Ordering vs. shim resume: the fusion/symm-mem multicast groups are
   rebuilt by the shim AFTER this replay; the registration does not
   reference them (only hVidMem), so no cycle. Verify empirically.

## Acceptance

- SGLang TP=4/TP=8 `--flashinfer-allreduce-fusion-backend trtllm` under
  `CB_IMEX=1`: checkpoint proceeds (0 blockers), restore passes, fusion
  engaged post-restore (no "Skipping allreduce fusion"), inference matches.
- torch symm-mem multimem (bf16, no compile) under `CB_IMEX=1`: same, with
  `multimem_all_reduce_` correctness.
- Existing matrix unchanged (no-fabric runs never enter the new path:
  zero live 00f8 at that point).
