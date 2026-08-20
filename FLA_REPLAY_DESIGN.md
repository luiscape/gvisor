# FLA registrations (NV_MEMORY_FABRIC 00f8) and checkpoint/restore

Status: implemented and validated. Code: `pkg/sentry/devices/nvproxy/`
`fla_registration.go` (+ `_unsafe.go`), wired in
`pkg/sentry/control/state_cuda.go`. This document is the mechanism summary
and the record of measured dead ends, so nobody re-walks them.

## The object

On fabric-attached GPUs (NVSwitch + fabric manager; the RM fabric probe
succeeds in default sandboxes), libcuda lazily allocates one
`NV_MEMORY_FABRIC` (00f8) object per VMM allocation shared between
processes: an FLA *registration*, identified by alloc params with
`Map.HVidMem != 0` (naming the covered vidmem). Properties, all measured:

- Created by libcuda internally; the covered `cuMemCreate` asked for plain
  POSIX-FD handles. Not application-requested, not application-freeable,
  no CUDA API touches it.
- Lives as long as the covered allocation.
- cuda-checkpoint *checkpoints* it but cannot *restore* it: the restored
  process dies in `NV_ERR_OBJECT_NOT_FOUND` storms. (This is why the
  checkpoint blocker gate refuses fabric objects.)
- Importer-side and bind-time 00f8 companions are transient and already
  released by the multicast interposer's suspend; only the exporter-side
  registration persists.

Affected workloads: anything that keeps peer-shared VMM allocations across
a checkpoint under an NVSwitch fabric -- torch symm-mem's workspace,
FlashInfer's all-reduce-fusion workspace.

## The mechanism

**Checkpoint** (between the cuda-checkpoint suspend window's teardown and
the checkpoint phase): `SuspendFLARegistrations` host-frees every live
registration via the owning client's allocating fd, and records
(client, handle, parent, hVidMem, params, fd) on
`nvproxy.suspendedFLARegs`. The strict blocker gate then sees the truth
(zero fabric objects) and cuda-checkpoint never meets one.

The record lives on `nvproxy`, NOT in the object graph: cuda-checkpoint's
own checkpoint phase frees the process's remaining RM objects through
nvproxy, cascading graph entries away before `state.Save`.

**Afterwards**, by fate of the process's driver state:

| Path | Driver state | Action |
| --- | --- | --- |
| Checkpoint-failure unwind | survived in place | `ReplayFLARegistrations`: host RM_ALLOC with identical client/handle/params, on the same fd that carried the free (app frozen since; fd necessarily open) |
| Resume-after-save | survived in place | same replay, after the restore toggle |
| True restore | rebuilt by cuda-checkpoint | `afterLoad` drops the record; libcuda lazily re-registers on the next export |

## Measured dead ends (do not re-walk)

1. **Replay bookkeeping on graph objects** (`hostFreed` flag): cascaded
   away by cuda-checkpoint's checkpoint-phase frees before `state.Save`.
2. **Denying the 00f8 class instead of suspending**: once the fabric probe
   has succeeded, libcuda treats FLA-registration failure during FD export
   as fatal (`CUDA driver error: unknown error`).
3. **Denying the fabric probe** (gating on `fabric-imex-mgmt`): kills
   single-node NVLS/multicast entirely -- libcuda couples
   `MULTICAST_SUPPORTED` to the probe. Reverted; the probe is default-
   allowed and IMEX is out of the C/R picture.
4. **Replaying after a true restore**: the toggle rebuilds the app's client
   through privileged debugger paths nvproxy never sees (the post-toggle
   graph contains only cuda-checkpoint's utility clients); no sentry-held
   fd is RM-associated with the restored client (foreign fds:
   `INVALID_CLIENT`; saved fds: dead); sentry RM_ALLOC into it fails
   `NV_ERR_INSUFFICIENT_PERMISSIONS` regardless of carrier (probed across
   every live frontendFD).

## The residual gap (NVIDIA-side)

Userspace that *caches* the registration handle across exports cannot
survive a true restore: torch symm-mem multimem's post-restore re-export
retries `EXPORT_OBJECT_TO_FD` against the stale handle into
`OBJECT_NOT_FOUND`. The checkpointed image believes in an object that
cannot be checkpointed; only cuda-checkpoint can reconcile that. Until
then, torch symm-mem workloads should set `MCSHIM_HIDE_MULTICAST=1`
(interposer masks `MULTICAST_SUPPORTED`; torch selects its two-shot
symm-mem kernel, which round-trips checkpoints -- validated). FlashInfer
fusion does not cache and passes end-to-end without the mask.

## Validation (8x H100, R580, default caps, no IMEX)

| Config | Result |
| --- | --- |
| SGLang TP=4 forced `--enable-nccl-nvls` | PASS, full C/R |
| SGLang TP=4 `--flashinfer-allreduce-fusion-backend trtllm`, fusion engaged | PASS, full C/R (FLA registrations host-freed at checkpoint, re-registered lazily post-restore) |
| SGLang TP=4 torch symm-mem bf16 + `MCSHIM_HIDE_MULTICAST=1` | PASS, two-shot |
| phase0 multicast harness, no-fabric matrix, nvproxy/control unit tests | PASS, unaffected |
