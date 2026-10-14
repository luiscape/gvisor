# Design: a generic in-container checkpoint hook

| | |
| :--- | :--- |
| Status | Draft for review |
| Date | 2026-10-14 |
| Branch | `luis/multicast-xgpu-upstream` at `12146bebf` (upstream `a9f9fd444` merged) |
| Related | `mcshim-merge-plan.md`, `mcshim-split-plan.md` |

## 1. Summary

Today the sentry carries mcshim's whole checkpoint protocol: it gates the
application, locks every process, has the interposer tear down its state,
re-locks, checkpoints, and reverses all of it on restore, through marker files
in `/tmp/mcshim`. That is about 1,700 lines in `pkg/sentry/control`, plus a
blocker inventory in nvproxy, and every future `cuda-checkpoint` release that
learns to carry more state would mean removing more of it.

This design replaces it with a **generic hook**: a program inside the
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
| `cuda-checkpoint --launch-job` wrap, its checks, and refusing checkpoints while a GPU container could not be wrapped | Must happen when the container starts | #13987, updated with R16, R17, R20 |
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
  (`checkpointLocked`, R11); the 2-minute bound on `--get-state`, lock and
  unlock; the unwind and the "torn down" refusal.
- **The markers.** Opened as `cudaShimFS` does today (directory without
  following a symlink, one component at a time, `O_NOFOLLOW | O_NONBLOCK`,
  regular files with one link): the hook runs as root in a directory any
  container user can write to.
- **The protocol version** (N22) stays, now between `mcshim-ckpt` and
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

## 7. Decisions needed

| # | Decision | Recommendation |
| :--- | :--- | :--- |
| D1 | Does runsc keep preloading the interposer (`--cuda-multicast-shim-path`, `LD_PRELOAD`, `/etc/ld.so.preload`)? | Move it to the operator: `LD_PRELOAD` in the spec's environment, and `/etc/ld.so.preload` as a read-only bind mount of a one-line file. Then gVisor knows nothing about mcshim but the fdinfo line. Keeping it costs about 250 lines and a mcshim-specific flag |
| D2 | Is the fdinfo line always on, or behind an option? | Always on, as for dma-buf in Linux: one record per export, shown only to tracers. Removes `TrackExports` and the second code path in four handlers. If upstream objects, keep the option |
| D3 | Name and phases of the hook | `--checkpoint-hook`, phases `checkpoint`, `restore`, `resume`. Upstream may prefer a CUDA-specific name (`--cuda-checkpoint-hook`), since the `restore`/`resume` split exists for GPU memory |
| D4 | How #13987 is enabled now that it is not tied to the interposer flag | An enum (new boolean flags are refused), e.g. `--cuda-checkpoint-job=launch`, requiring `--cuda-checkpoint-path`; R610 or newer |
| D5 | Ask a gVisor maintainer before building it | Yes: in the PR1 design review, with this document |

## 8. Plan

1. **Design review** with a gVisor maintainer (D3, D5).
2. **mcshim**: write `mcshim-ckpt` by porting the Go code, with CPU tests
   against fakes; add the interposer change for section 5.
3. **gVisor**, on the branch:
   - remove the blocker inventory, import tracking and, per D2,
     `TrackExports`;
   - add the hook (flag, runner, sequence, checkpoint keys) and tests with a
     fake hook (a shell script per phase);
   - remove `state_cuda_shim.go`, the interposer sequence, and, per D1, the
     preload flag;
   - give #13987 its own enable flag (D4).
4. **GPU validation**: the full matrix from the merge plan's section 7.4,
   now with `runsc --checkpoint-hook` and `mcshim-ckpt`.
5. **Merge plan**: rewrite the PR list (section 9).

Rough effort (estimates, not measured): `mcshim-ckpt` 3-5 days; gVisor
changes 2-3 days; GPU validation 2 days.

## 9. Resulting PRs

| PR | Contents | Approx. lines |
| :--- | :--- | :--- |
| PR0 | #14817, `--device-map` on restore | 300 (done) |
| PR1 | Design: the hook, the fdinfo line, the job wrap | Docs |
| PR2 | #13987, the `cuda-checkpoint --launch-job` wrap, with R16, R17, R20 and its enable flag | 300 |
| PR3 | nvproxy: export identity and the fdinfo line | 400 |
| PR4 | The checkpoint hook | 450 |

Down from nine code PRs to four, none of which mentions multicast or mcshim.
