# Merge plan: multi-GPU CUDA checkpoint/restore

Branch `luis/r580-multicast-snapshots` is ~9.3k lines against `master`, ~4.7k
of which is the C interposer (`tools/mcshim`). It will not be reviewed as one
PR. This document splits it into independently useful stages, records the
measurements each stage rests on, and lists the questions that must be
settled with maintainers before the later stages are proposed.

`PROGRESS.md` is the chronological session log; everything durable from it
is folded in here.

## 1. What the branch does

`runsc checkpoint` / `restore` of a single-node multi-GPU inference
container (vLLM, SGLang; TP/DP/PP/EP up to 8 GPUs), preserving CUDA graphs
and torch.compile state, including restore onto a *different* set of GPUs.
Validated on 8x H100 + NVSwitch, drivers 580.173.02 and 610.57.04, with
custom all-reduce, NCCL NVLS, torch symmetric memory, FlashInfer fusion,
MoE / expert parallelism, FP8, and three quiesce geometries (engine sleeps
and offloads weights; engine frees KV only; nothing freed).

Two cooperating layers:

| Layer | Piece | Job |
| --- | --- | --- |
| sentry `control/state_cuda.go` | orchestration | admission gate -> lock all ranks -> unlock -> interposer suspend -> FLA suspend -> strict blocker gate -> re-lock -> `cuda-checkpoint`; inverse after restore |
| sentry `control/state_cuda_shim.go` | marker-file protocol | existence-based, edge-triggered, per-pid acks, timeouts |
| sentry `nvproxy/checkpoint_blockers.go` | blocker inventory | fabric/multicast RM objects and exported fds, attributed per rank; `/proc/[pid]/fdinfo` identity oracle |
| sentry `nvproxy/fla_registration.go` | FLA registrations | host-frees driver-internal 00f8 objects in the suspend window; replays on failure paths |
| sentry `nvproxy/cuda_admission.go` | admission gate | a process initializing CUDA during a checkpoint is held until the sequence ends |
| sentry `nvproxy` device remap | cross-GPU | sandbox-visible minors stable across restore; RM device instances translated where the driver needs it |
| container `tools/mcshim/mcshim.c` | LD_PRELOAD interposer | tracks multicast groups / VMM exports+imports / legacy IPC / mappings at the libcuda layer; suspend tears down, resume rebuilds at byte-identical VAs; promotes legacy IPC to VMM IPC at export |

## 2. Stages

Each stage is one clean commit carved from the tree (`git checkout -p` onto
a branch off current `master`), gate-tested on its own. Stages 1-4 have no
interposer dependency and help every gVisor GPU-checkpoint user; they can
land while the interposer design conversation (stages 6-7) happens.

| # | Content | ~Lines | Standalone value | Risk / review focus |
| --- | --- | --- | --- | --- |
| 1 | **Save-side correctness.** `frontendFD`/`uvmFD` mapping tracking + `InvalidateUnsavable` (fixes `Can't save pma with non-MemoryFile of type *nvproxy.frontendFDMemmapFile`); `cudaProcs` walks every task's FD table; `cudaProcs` sorted; `NV2080_CTRL_CMD_FLA_GET_FABRIC_MEM_STATS` handler | 150 | fixes a real save panic for any GPU checkpoint | low; pure fixes, vfio `pciDeviceFD` pattern |
| 2 | **Blocker inventory + fdinfo oracle.** `checkpoint_blockers.go` (per-slot export accounting), `procFDInfoExtra` hook in `fsimpl/proc`, `--cuda-checkpoint-blocker-timeout` / `SaveOpts.CudaBlockerTimeout` | 500 | "hangs inside cuda-checkpoint" becomes an attributed refusal; the fdinfo line is a generic dmabuf-style facility | low-medium; new `/proc` surface needs a maintainer nod; the fdinfo format is a contract (locked by test) |
| 3 | **Admission gate.** `cuda_admission.go`, hook in `rmAllocRootClient`, open/close in `preSaveCuda` / `postResumeCuda` | 200 | late-initializing CUDA processes are no longer silently left out of the snapshot (their objects then fail replay at restore) | medium; blocks an ioctl on a channel (`ERESTARTNOINTR`); must open before any interposer rebuild (found by the recovery trial) |
| 4 | **Cross-GPU restore.** `DeviceRemapping` device-view stability (`hostMinorByMinor`, `appDevInstByHostDevInst` gated to R<610), `createRemappedNvproxyDeviceFiles`; `--device-map` FIXME resolved | 250 | restore onto different GPUs without `--device-map` | medium; the 580/610 divergence (section 4) needs an NVIDIA-savvy reviewer |
| 5 | **FLA registration suspend/replay.** `fla_registration.go`, `rmAllocMemoryFabric`, `cudaSaveFailedKey` recovery in `state.go` | 450 | needed on any fabric-attached host, shim or not | medium-high; host-frees RM objects from the sentry; single-pass scope guard |
| 6 | **Interposer integration.** `state_cuda_shim.go`, loader LD_PRELOAD + `/etc/ld.so.preload` + env injection (`MCSHIM_*`, `NCCL_CUMEM_ENABLE=1`), `--cuda-multicast-shim-path` / `--cuda-multicast-shim-source=IMAGE\|EMBEDDED`, `runsc/mcshimbin` embedding | 1200 | the multi-GPU enabler | high; propose as a design doc first (section 3 is most of it) |
| 7 | **The interposer.** `tools/mcshim/{mcshim.c,mcshim_helper.c,build.sh,BUILD}`, `make mcshim` | 4900 | -- | open question whether C code of this size belongs in-tree (section 6) |

Not upstream-bound (drop from every stage): `PROGRESS.md`, `NEW_UPDATES.md`,
`R580_VALIDATION.md`, `FLA_REPLAY_DESIGN.md`, `e2e_*.sh`, `cr-bench/`,
`gpu_mem_snapshots/`, `NVIDIA-*.run`.

Before carving: rebase onto current master (last merge: `e606870d3`; note
upstream now forbids new boolean flags and requires the sidecar layout,
both already accommodated).

## 3. Mechanisms and why each exists (for the design doc)

Every mechanism below fixes a failure that was measured end to end, and the
alternatives listed were tried and failed. Reviewers will ask "why not X";
these are the answers.

1. **Multicast (0x00fd; NCCL NVLS, torch symm-mem, FlashInfer fusion).**
   `cuda-checkpoint` cannot serialize it. Interposer teardown before the
   checkpoint, rebuild after. On R580/R610 a restored process cannot
   `cuMulticastCreate`/`AddDevice` (INVALID_DEVICE), so a never-checkpointed
   helper process (`mcshim-helper`) does create+attach on the rank's behalf;
   the group persists once the rank holds its own import.
2. **Shared VMM P2P (NCCL `cuMem`).** Live imports cannot cross the
   per-process restore toggle. Close/re-import; the exporter re-serves the
   fd over a unix socket keyed by the fdinfo oracle (all export fds are opens
   of `/dev/nvidiactl` and share one inode, so `fstat` cannot identify them).
3. **Legacy CUDA IPC (custom all-reduce, `cudaMalloc` buffers).** Promoted to
   VMM IPC at `cuIpcGetMemHandle`: the exporter's segment is replaced in
   place by a `cuMemCreate`'d one at the same VA, exported as a POSIX fd; the
   blob names the fd. **Why not replay:** `cuIpcOpenMemHandle` takes no
   address hint, and the driver's legacy VA allocator is top-down first-fit
   and refuses an exact-fit hole (`gpu_mem_snapshots/probes/
   ipc_reopen_probe.py`), so packed imports (32 MiB at 32 MiB stride) can
   never reopen at their VA. Per-import reservations, releasing all holds,
   hole plugging, size/reverse ordering were all measured; none returns
   packed imports. Three details that mattered: `cuMemRetainAllocationHandle`
   must report promoted ranges as non-VMM (SGLang probes it per buffer set);
   sub-2 MiB buffers are granule-rounded; the shim's serve threads must be
   joinable (`shutdown()` does not wake `accept()` under gVisor). The legacy
   close/replay path is retained behind `MCSHIM_IPC_PROMOTE=0` as a fallback.
4. **FLA registrations (0x00f8).** libcuda-internal, one per peer-shared
   allocation on fabric-attached hosts, no userspace API frees them,
   checkpointable but not restorable. The sentry host-frees them in the
   suspend window; libcuda re-registers lazily after a true restore; on a
   failed save they are replayed by client identity. Denying the 00f8 class
   is not a substitute (libcuda treats registration failure during export as
   fatal); denying the fabric probe disables single-node NVLS.
5. **Stale FLA cache on resident allocations (torch multimem).** libcuda
   caches the registration handle per allocation; a resident allocation
   restored after its registration was freed fails its next export with
   `OBJECT_NOT_FOUND`. Multicast-bound UC exporters are therefore freed
   across the checkpoint (contents saved into process memory) and recreated
   at identical VAs. Default on; `MCSHIM_KEEP_UC_EXPORTS=1` opts out.
6. **Quiescence.** Gate first (stops new submissions), then
   `cuda-checkpoint --action lock` in parallel (drains in-flight work);
   teardown sandwiched inside lock/unlock/re-lock because the interposer must
   issue libcuda calls, which a locked process cannot. Gating alone
   deadlocks collectives (a gated rank starves peers spinning in the
   collective it has not submitted), hence bounded gate/lock retries.
7. **Late CUDA initialization.** Engines spawn helpers that reach `cuInit`
   seconds after the process set is collected; their RM objects would be
   replayed blind on restore. The admission gate holds a process's first
   RM root-client allocation until the sequence ends.
8. **Cross-GPU restore.** Device minors stay stable to the application;
   translated at the RM boundary. See section 4 for the driver divergence.

## 4. Driver-dependent behaviour (must be stated in stage 4's PR)

`NV0000_CTRL_CMD_OS_UNIX_GET_EXPORT_OBJECT_INFO` reports a device instance.
Through R580 libcuda resolves it against its own (pre-restore) device table,
so after a restore onto other GPUs the sentry must translate it back to the
old instance or the import fails with `CUDA_ERROR_INVALID_DEVICE`. From R610
libcuda resolves it against current state and the translation causes exactly
that failure. All four cells measured (vLLM TP=2, GPUs 0,1 -> 4,5):

| Driver | translation on | translation off |
| --- | --- | --- |
| 580.173.02 | PASS | FAIL |
| 610.57.04 | FAIL | PASS |

Gated to `Major() < 610`. The boundary within (580.173, 610.57] is not pinned
down; R590/R595 are untested. Everything else on the branch is
driver-independent between 580 and 610 (measured back to back).

## 5. Performance facts that belong in the user guide

- **Host `transparent_hugepage/shmem_enabled=always` is load-bearing.**
  With `never`, every never-written page of the memory file that the driver
  reads or writes during the toggle takes a 4 KiB `shmem_fault` +
  zero-fill; for a fully resident 2x60 GB checkpoint that is 100 s of restore
  toggle vs **14 s** with `always` (checkpoint action 92 s vs 21 s). `advise`
  is refused by runsc unless `defrag` is `always|defer|never`. This should
  become a runsc warning at GPU checkpoint time.
- **Pin the sandbox to the GPUs' NUMA node.** Cross-socket, the checkpoint
  action is 3.5x slower (73 s vs 21 s for 2x60 GB); restore is
  placement-independent (pages placed at load). `runsc run` under
  `numactl --cpunodebind=N --membind=N`; for cross-socket restore pin
  `runsc restore` to the *destination* node.
- With both settings, fully resident ("pause") quiesce is the fastest
  geometry: 2x60 GB resident -> ~21 s checkpoint action, ~14 s toggle, ~15 s
  to first inference; image stays small because `--exclude-committed-zero-
  pages` drops the zero KV pool. Release quiesce (vLLM TP=2, 1.5B): 5 s to
  first inference.
- **Retracted claims** (so nobody re-derives them): "~1 s/GB carry cost" was
  the 4 KiB fault storm; "R610 copies 2x faster than R580" was NUMA
  placement of an unpinned native probe. Neither is a driver property.
- `cuda-checkpoint` native copy rate, pinned near: ~2 GB/s D2H, ~5 GB/s H2D
  (60 GB: 32 s / 12 s); 2x slower D2H cross-socket.
- Pinned host memory adds ~70 ms/GB at restore; shim resume is ~1.5 s at
  100+ imports/rank; neither is material.

## 6. Open questions for maintainers (settle before stages 6-7)

1. Can a ~4.7k-line C interposer live in-tree, or should `tools/mcshim` be
   its own repository with gVisor embedding a released artifact?
2. Is embedding the interposer binary in `runsc`
   (`--cuda-multicast-shim-source=EMBEDDED`, materialized into the
   container's overlay at creation) acceptable, or must images carry it?
3. The interposer injects `LD_PRELOAD` and appends to the container's
   `/etc/ld.so.preload` (launchers such as SGLang's rewrite `LD_PRELOAD`).
   Is writing into the container's root acceptable under the default
   overlay? (It never touches the host rootfs.)
4. Promotion changes what the application allocated (VMM-backed where it
   asked for `cudaMalloc`). It is masked at `cuMemRetainAllocationHandle`;
   no engine observed the difference. Is that an acceptable contract?
5. Stage 5 host-frees RM objects the application still references (FLA
   registrations). It is correct on every measured path, but it is the
   sentry reaching into driver state; reviewers should know.

## 7. Known limits

- Single pass only: checkpoint -> restore. Resume-after-save with fabric
  users fails loudly by design; chained cross-GPU restores panic by design.
- Multi-node / IMEX / MNNVL out of scope (fabric handle types are stripped).
- Promotion migrates the whole `cudaMalloc` segment containing an exported
  buffer, once, at export time (tens of ms at 1.5B-MoE scale; unmeasured on
  30B-class).
- Pause geometry saves the resident KV pool; correct, but image/time scale
  with resident memory.
- Not validated: 30B-class models, sleep/wake churn stress, R590/R595.

## 8. Operational pitfalls (for whoever re-runs validation)

- Never host `bazel`; `make build TARGETS=//:release` via the container.
  Since the upstream sidecar change, install the whole `bazel-bin/release/`
  layout (`runsc` + `gvisor-bin/`); a bare `runsc` copy does not boot.
- Never background `&&` chains; `nohup` only standalone scripts. `pgrep -f`
  self-matches.
- `apt-daily-upgrade` restarted the fabric manager once and orphaned every
  GPU registration (all `cuMulticastBindMem` fail with 401, natively too).
  Recover with FM stop -> `nvidia-smi -r` -> FM start; disable the timers.
- Run dirs eat the disk; keep `rootfs-*`, delete `cr-bench-*-multi-*`.
- `date +%s%3N` is wrong on uutils coreutils; a private netns has `lo` down
  (use `--network=none`); the legacy docker builder ignores heredocs.
- Verify engagement, not just PASS: fusion self-disables silently, fp16
  disables symm-mem, `uc_freed` / `PROMOTE:` / `host-freed FLA` counts in
  the logs are the evidence.
