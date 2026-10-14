# Multi-GPU CUDA checkpoint/restore: plan

Dev-only: not part of any upstream PR.

| | |
| :--- | :--- |
| Status | Implemented on the branch, CPU-tested; GPU validation, design review and PR cutting remain (section 8) |
| Date | 2026-10-14 |
| Branch | `luis/multicast-xgpu-upstream` (github.com/luiscape/gvisor), current with upstream `master` |
| mcshim | github.com/modal-projects/mcshim (interposer and `mcshim-ckpt`) |
| Upstream proposal | `g3doc/proposals/cuda_multigpu_checkpoint.md` |
| History | The earlier merge plan, split plan and status file (requirements R1-R25, N1-N22, the sentry-driven design) were removed in the commit after `28c39e73c`; see git history |

## 1. Summary

The sentry used to carry mcshim's whole checkpoint protocol: it gates the
application, locks every process, has the interposer tear down its state,
re-locks, checkpoints, and reverses all of it on restore, through marker files
in `/tmp/mcshim`. That is about 1,700 lines in `pkg/sentry/control`, plus a
blocker inventory in nvproxy, and every future `cuda-checkpoint` release that
learns to carry more state would mean removing more of it.

It was replaced by a **generic hook**: a program inside the
container that runsc runs at fixed points of a checkpoint and a restore. gVisor
keeps only what it alone can do (run the hook with the container's namespaces,
hold page faults while GPU state is away, unpause a paused kernel, compute the
device map). The CUDA sequence and the marker protocol move into a hook that
mcshim ships, `mcshim-ckpt`, so the protocol becomes internal to the mcshim
repository.

Decisions taken: `cuda-checkpoint --launch-job` stays in gVisor (#13987); the
blocker inventory is removed.

## 2. What stays in gVisor

| Piece | Why gVisor | PR |
| :--- | :--- | :--- |
| `--device-map` on restore | Only runsc knows how GPUs were remapped | #14817 (unchanged) |
| `cuda-checkpoint --launch-job` wrap, its checks, and refusing checkpoints while a GPU container could not be wrapped | Must happen when the container starts | #13987, updated (GPU containers only; skip with a warning if `cuda-checkpoint` is missing or the entrypoint is `--exec-fd`; absolute paths) |
| Export identity in nvproxy and the `nvproxy_exported_object` fdinfo line | Only the sentry sees RM objects | nvproxy PR (reduced, section 6) |
| The checkpoint hook | This design | New |

## 3. The hook

### 3.1 Configuration

A per-container runsc flag, like `--cuda-checkpoint-path`:

```
--checkpoint-hook=/opt/mcshim/mcshim-ckpt    # absolute path inside the container
```

Validation as for `--cuda-checkpoint-path` (absolute, a regular file in the
container at checkpoint time). A container with a hook is excluded from the
built-in `cuda-checkpoint` handling: the hook owns its CUDA processes. The
built-in path is unchanged for everyone else (#14817's code).

If CUDA processes exist in a container without a hook while another container
has one, the checkpoint is refused: the two cannot be ordered against each
other.

### 3.2 Phases

The sentry runs `<hook> <phase> [args]` once per container that has a hook,
with the container's mount and PID namespaces and the environment from its
spec (as it runs `cuda-checkpoint` today), stdin `/dev/null`, stdout and
stderr captured into the error and the log.

| Phase | When | What `mcshim-ckpt` does | On failure |
| :--- | :--- | :--- | :--- |
| `checkpoint` | Before the kernel is saved, after the sentry holds page faults (3.3) | Find CUDA processes; gate; lock all; unlock; suspend the interposer; re-lock; `cuda-checkpoint --action checkpoint` one process at a time | The hook undoes what it did (restore the checkpointed processes, unlock all, unwind the interposer) before it exits non-zero. The sentry releases page faults and fails the checkpoint |
| `restore [--device-map M]` | After a restore, or after a save with `--resume`, or after a save that failed with `--resume`; page faults still held | `--toggle`, or `--action restore --device-map M` then `--action unlock`, on every checkpointed process | The sentry fails the restore (the sandbox is killed, as today) |
| `resume` | After the sentry releases page faults | Remove `suspend`, wait for every interposer to rebuild, remove `gate` | Same |

Exit status 0 is success; anything else is failure. A hook that cannot undo a
failed `checkpoint` exits 2 instead of 1, so the error can say the application
is left blocked; the sentry does nothing else differently.

The split between `restore` and `resume` is the one constraint the sentry
imposes on the sequence: the interposer's rebuild touches GPU memory, and
faults it takes while the sentry holds them are never serviced (every rank
fails with `CUDA_ERROR_ILLEGAL_ADDRESS`, observed). So GPU state comes back
first, page faults are released, and only then does the rebuild run.

### 3.3 What the sentry does around the hook

`preSaveCuda`, for a container with a hook:

1. If the kernel is paused (Docker's `checkpoint` pauses first), unpause it,
   as today.
2. `SigsegvLock` every thread group with an NVIDIA device fd (no
   `--get-state` filter: the lock is harmless for NVML-only processes).
3. Run `<hook> checkpoint`.
4. On failure: `SigsegvUnlock`, re-pause if it was paused, return the error
   with the hook's output.
5. On success: re-pause if needed, and record in the checkpoint the hook path,
   the containers it ran for, and the locked thread groups.

`postRestoreCuda` (restore, resume after a save, and resume after a failed
save with `--resume`, under the same conditions as today):

1. Pop the recorded state; nothing to do if absent.
2. Run `<hook> restore`, with `--device-map` if `cudaCheckpointDeviceMap`
   returns one (R580 or newer, as today).
3. `SigsegvUnlock` the recorded thread groups.
4. Run `<hook> resume`.

No timeout: like `--action checkpoint` today, the hook's duration grows with
GPU memory. The hook bounds its own steps.

### 3.4 What leaves gVisor

| Removed | Lines (code + tests, approx.) | Goes to |
| :--- | :--- | :--- |
| `state_cuda_shim.go` (markers, acks, protocol version, `cudaShimFS`) and tests | 950 | `mcshim-ckpt` |
| The interposer sequence in `state_cuda.go` (`checkpointCudaProcs`, `checkpointLocked`, bounded `wait`, `runCudaAction`, `cudaProcState`) and tests | 650 | `mcshim-ckpt` |
| `MCSHIM_DISABLE` on `cuda-checkpoint`, the `cuda-multicast-shim*` checkpoint keys, `SaveOpts.CudaMulticastShim` | 60 | Gone (the hook sets the variable for its own children) |
| The blocker inventory and nvproxy's import tracking | 450 | Partly `mcshim-ckpt` (section 5) |
| `--cuda-multicast-shim-path`, the preload, and the driver gate on it | 250 | The operator (decision D1) |
| `Options.TrackExports` and its gating | 60 | Gone if the fdinfo line is always on (decision D2) |

Added: the hook flag, its validation, the hook runner (generalizing
`invokeCudaCheckpoint` from "one process" to "one container"), the sequence
above, and tests, about 450 lines.

Net: the gVisor changes go from about 3,600 lines (without #14817) to about
1,300, of which about 300 are #13987 and 400 are nvproxy's export identity.

## 4. `mcshim-ckpt` (in the mcshim repository)

Implemented in mcshim (`cmd/mcshim-ckpt`, `internal/ckpt`), CPU-tested.

A static Go binary (`cmd/mcshim-ckpt`): static so that `/etc/ld.so.preload`
does not load the interposer into it, and Go so that the existing sentry
logic and its tests port nearly as they are.

- **Finding processes.** Scan `/proc/[0-9]*/fd` for links to `/dev/nvidia*`,
  then `cuda-checkpoint --get-state` each, bounded at 2 minutes, as the
  sentry does today. (To verify on a GPU host: what `readlink` shows for an
  nvproxy fd inside gVisor.)
- **The sequence.** As in section 3.2, with today's code: parallel lock with
  `--timeout 30000`, sequential checkpoint stopping at the first failure,
  `--get-state` of the rest, restore only the checkpointed, unlock all
  (`checkpointLocked`); the 2-minute bound on `--get-state`, lock and
  unlock; the unwind and the "torn down" refusal.
- **The markers.** Opened as `cudaShimFS` does today (directory without
  following a symlink, one component at a time, `O_NOFOLLOW | O_NONBLOCK`,
  regular files with one link): the hook runs as root in a directory any
  container user can write to.
- **The protocol version** (`mcshim protocol N` in `present.<pid>`) stays, now between `mcshim-ckpt` and
  `libmcshim` from the same repository; it catches an image whose `.so`
  and hook come from different releases.
- **`cuda-checkpoint`'s path**: an argument or environment variable set by
  the operator (default `/usr/local/bin/cuda-checkpoint`), and every child
  runs with `MCSHIM_DISABLE=1`.

Tests run without a GPU: a fake `cuda-checkpoint` (as `TestCheckpointLocked`
uses) and a fake interposer answering the markers. The stub tests in
`test/run.sh` (`deadline`, `reason`, `undo`, `optout`, `orphan`) move to
driving runsc with `--checkpoint-hook`.

## 5. Replacing the blocker inventory

| Check the sentry did | Replacement in `mcshim-ckpt` |
| :--- | :--- |
| A CUDA process not running the interposer holds multicast or imported memory | Refuse any process that `--get-state` reports as CUDA but that never wrote `present.<pid>`. Stricter, and needs no view of RM objects |
| Something remains after the interposer's teardown | No exact replacement. Backstop: a generous bound on `--action checkpoint` (configurable, default 30 minutes), so a hang becomes a failed, unrecoverable checkpoint instead of a stuck RPC |
| An import whose exporter no longer exists (`UnresolvableImports`) | Each interposer lists the export identities it holds, and the identities of its imports, when it acks the gate; the hook refuses an import nobody exports. A small interposer change |

## 6. nvproxy after the change

Kept: when an object is exported into an fd (`EXPORT_OBJECT(S)_TO_FD`), record
its (client, object, class) on the fd, and report it in
`/proc/[pid]/fdinfo/[fd]` to readers with ptrace access, as Linux does for
dma-buf. Removed: import tracking (`importedObject`, the two import
handlers, which return to upstream's `ctrlHasFrontendFD`), `rootClient.tgid`,
holding the client lock across the host call, and the import constants.

## 7. Decisions

| # | Decision | Outcome |
| :--- | :--- | :--- |
| D1 | Does runsc keep preloading the interposer? | **No.** The operator preloads it: `LD_PRELOAD` in the spec and `/etc/ld.so.preload` as a read-only bind mount. `--cuda-multicast-shim-path` is removed |
| D2 | Is the fdinfo line always on? | **Yes**, like dma-buf's. `TrackExports` is removed; import tracking and the blocker inventory too |
| D3 | Name and phases | `--checkpoint-hook`; `checkpoint`, `restore [--device-map]`, `resume`. Runs in each container with a process holding an NVIDIA device open |
| D4 | How #13987 is enabled | `--cuda-checkpoint-job=launch` (enum), with `--cuda-checkpoint-path`, R610 or newer |
| D5 | Maintainer review before merging | Pending: PR1 review of `g3doc/proposals/cuda_multigpu_checkpoint.md` (rewritten for the hook) |

Implementation notes (differences from sections 3-6):

- On a `checkpoint` failure with status 1, the hook does not unwind the
  interposer itself: the sentry releases page faults and runs `resume`,
  which rebuilds what was torn down and releases the gate. The rebuild
  touches GPU memory, so it cannot run while page faults are held. Status 2
  or a crash: no `resume`.
- A hook that could not be started changed nothing; it is reported as such,
  and is not resumed.
- The CUDA path without a hook is #14817's code exactly, plus a shared
  `execInContainerOf` helper (also used by the hook) that skips a thread
  group whose leader has exited.

## 8. Remaining work

### 8.1 GPU validation

Nothing above has run on a GPU. On an H100 x8 host with driver R610, with
`runsc --nvproxy --cuda-checkpoint-path=... --cuda-checkpoint-job=launch
--checkpoint-hook=/opt/mcshim/mcshim-ckpt`, the operator preloading
`mcshim.so` (`LD_PRELOAD` and a read-only `/etc/ld.so.preload` bind mount):

1. **One checkpoint and restore first**, to confirm what CPU tests cannot:
   `mcshim-ckpt` finds CUDA processes through `/proc/<pid>/fd` inside gVisor;
   the `/etc/ld.so.preload` bind mount works (it may need a mount point in
   the image); page faults are held across `checkpoint` and `restore` and
   released before `resume`.
2. **mcshim's suite**: `make test`, `make test-docker`, and `test/run.sh`
   with `-r RUNSC` (stub tests: `deadline`, `reason`, `undo`, `optout`).
3. **Failure paths through runsc**: a hook exiting 1 (resume runs, the
   application keeps running), exiting 2 (no resume), a hung `--get-state`,
   and Docker's pause-then-checkpoint flow.
4. **Without the hook**: `--cuda-checkpoint-path` alone (one `--toggle` per
   process, as #14817), and `--cuda-checkpoint-job=launch` without a hook; a
   GPU container whose image lacks `cuda-checkpoint` starts and checkpoints
   are refused naming it; a pod's pause container is not wrapped.
5. **The matrix**: the 15 engine configurations and the 3 without
   `CAP_SYS_PTRACE`; Qwen2.5-72B at TP=8 on vLLM and SGLang; restores onto
   other GPUs (`--device-map` through the hook); a tokens/s comparison with
   and without the interposer.
6. Update the proposal's validation tables, which are from the
   sentry-driven design.

R615 is a separate run before deploying there.

### 8.2 Upstream

- **#14817** (`--device-map`): open, labelled *ready to pull*, no review
  comments. Ping a maintainer; PR4 depends on its `cudaCheckpointDeviceMap`.
- **#13987** (job wrap): its flags (`nvproxy-cuda-checkpoint-*`) no longer
  match. Update it to PR2 below, or close it in favour of PR2.
- **Design review (D5)** of the proposal before opening PR4.

### 8.3 Cutting the PRs

Each from `master`, not from this branch's history; this file stays out.

| PR | Files from the branch |
| :--- | :--- |
| PR1 | #14817 as is |
| PR2 | `runsc/config` (`CUDACheckpointJob`, `--cuda-checkpoint-job`, validation, tests); `runsc/boot/loader.go` (`setupCudaCheckpointJob`, `cudaJobSkipped`, `containerFileExists`) and `loader_test.go`; `runsc/boot/restore.go` (sequential and refusal in `save`) |
| PR3 | `pkg/sentry/devices/nvproxy/exported_objects.go` and test, the export handlers in `version.go`, `frontendFD.exportedObj`, `pkg/abi/nvgpu/ctrl.go` constants, `pkg/sentry/fsimpl/proc/task_fds.go` |
| PR4 | `pkg/sentry/control/state_hook.go` and test, `execInContainerOf`/`openNullFD` in `state_cuda.go`, `state.go`, `pkg/sentry/state/state.go`, `--checkpoint-hook` (config, flags, `save`), the proposal and user guide |

### 8.4 Gaps

- The hook's kernel-backed code (`gpuProcs`, `kernelHookOps`,
  `preSaveHook`, `postRestoreHook`, the failed-save undo) has no automated
  test: it needs NVIDIA device fds. Reviewers may ask for a runsc container
  test with a fake hook.
- The hook path is sandbox-wide (the root container's config), and runs only
  in containers with an NVIDIA device open.
- mcshim no longer refuses an import whose exporter is gone at checkpoint
  time (formerly nvproxy's `UnresolvableImports`): the rebuild fails in
  `resume` instead, leaving the application gated. TODO in `mcshim-ckpt`
  (interposers listing their export and import identities).
## 9. Resulting PRs

| PR | Contents | Approx. lines |
| :--- | :--- | :--- |
| PR1 | #14817, `--device-map` on restore | 300 (open) |
| PR2 | #13987, the `cuda-checkpoint --launch-job` wrap, with its checks and enable flag | 300 |
| PR3 | nvproxy: export identity and the fdinfo line | 400 |
| PR4 | The checkpoint hook | 450 |

Four code PRs (the design doc goes with PR4), none of which mentions
multicast or mcshim.
