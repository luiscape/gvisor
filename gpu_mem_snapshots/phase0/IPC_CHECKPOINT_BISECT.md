# What actually blocks cuda-checkpoint: a two-API bisect

Reproducer: `ipc_scale_probe.py`. Native (no gVisor, no runc, no CRIU), no
NCCL, no PyTorch, no multicast, no CUDA graphs, no inference engine. W
processes, one GPU each, ~140 lines of `ctypes` against `libcuda.so`.

Environment: 8x H100 (NVSwitch), driver **610.57.04**, `cuda-checkpoint`
driving `lock -> checkpoint -> restore -> unlock` (the same phased sequence as
`pkg/sentry/control/state_cuda.go`).

```
# standalone: one --action per pid
python3 ipc_scale_probe.py --world 2 --allocs 1 --stage everything --trials 3

# job mode: what gVisor actually does (--cuda-checkpoint-path wraps the
# container command, so every CUDA process shares one job file)
cuda-checkpoint --launch-job python3 ipc_scale_probe.py --stage everything ...
```

**Run both.** Job mode is not a packaging detail: it does not change the VMM
result at all, but it changes the legacy CUDA IPC result completely, and the
job-mode answer is the one that applies to gVisor.

Every stage is a strict superset of the one above it, so the first failing
stage names the precise step that arms the failure. Results below are
**3/3 deterministic** in each direction -- there is no flakiness to average.

## Result: two independent defects, in two different APIs

### VMM sharing (`cuMemCreate` + `cuMemExportToShareableHandle`)

| stage | what the process holds at checkpoint time | result |
| --- | --- | --- |
| `plain` | VMM alloc, not exportable | pass |
| `alloc` | exportable handle type, never exported | pass |
| `export` | + `cuMemExportToShareableHandle`, FD held locally | pass |
| `share` | + FD passed to peers, which never import it | pass |
| `nomap` | + peers `cuMemImportFromShareableHandle`, **never mapped** | **FAIL** — restore: `"out of memory"` |
| `import` | + peers import **and** `cuMemMap` | **FAIL** — restore: `"invalid argument"` |
| `unmap` | imported, then unmapped; **handle still held** | **FAIL** — restore: `"out of memory"` |
| `release` | imported, then `cuMemRelease`d; export FDs still open | pass |
| `teardown` | imported, then fully released everywhere | pass |

Reading down the table:

- **The trigger is one thing: a live imported allocation handle.** Not
  exportability, not the export FD, not FD possession by a peer, not the
  mapping.
- **Mapping is irrelevant.** `nomap` fails without a single `cuMemMap`.
- **Unmapping is not a remedy.** `unmap` still fails; only `cuMemRelease` of
  the imported handle clears it.
- **It is fully reversible.** `release` passes with the exporter's FDs still
  open, so `cuMemRelease` on the importer is both *necessary and sufficient*.
- Checkpoint always succeeds. The refusal lands on **restore**, which is why
  it has been so expensive to diagnose: the failure surfaces one phase after
  its cause.

The VMM ladder is **identical in job mode** — same stages fail, same phase,
same error strings (verified: `share`, `nomap`, `import`, `unmap`, `release`).

### Legacy CUDA IPC (`cuIpcGetMemHandle` / `cuIpcOpenMemHandle`)

Here, and only here, job mode matters:

| stage | what the process holds at checkpoint time | standalone | **job mode** |
| --- | --- | --- | --- |
| `legacy-alloc` | `cuMemAlloc`, never shared | pass | pass |
| `legacy-export` | + `cuIpcGetMemHandle` | **FAIL** — checkpoint: `"OS call failed or operation not supported on this OS"` | **pass** |
| `legacy-import` | + peers `cuIpcOpenMemHandle` | **FAIL** — checkpoint | **FAIL** — restore: `"unknown error"` |
| `legacy-close` | peers `cuIpcCloseMemHandle` before the checkpoint | **FAIL** — checkpoint | **pass** |
| `legacy-free` | peers close **and** exporter `cuMemFree`s | pass | pass |

Standalone, `cuda-checkpoint` refuses any process that has ever called
`cuIpcGetMemHandle` — it cannot reason about peers it was not told about, so
exporting alone is fatal and only destroying the allocation clears it.

**Given the job file, that refusal disappears.** Exporting is fine, and
closing the imports restores checkpointability. What remains is a live
*import*, failing at *restore* — i.e. **exactly the same defect as the VMM
path**, in the same phase, with the same remedy.

## The unified conclusion (job mode, which is what gVisor uses)

> A live cross-process **import** — VMM or legacy — breaks `restore`.
> Releasing the import before the checkpoint is necessary and sufficient.
> Nothing else in either sharing sequence matters.

This also explains the two error strings that made the "toggle bug" look like
one flaky failure with an inconsistent signature:

| observed | cause |
| --- | --- |
| `"invalid argument"` | live **VMM** import, mapped |
| `"out of memory"` | live **VMM** import, unmapped |
| `"unknown error"` | live **legacy IPC** import |

and its shape: only the ranks actually holding a live import fail, which is
why some workers toggled fine and others did not (`legacy-import` in job mode
fails 1 of 2 processes).

## Why this matters here

The two tables explain, and cleanly separate, results that had been lumped
together as one flaky "toggle bug":

- **mcshim already does the right thing for VMM**, and for a principled
  reason we can now state: it releases every `KIND_IMP` import at suspend.
  Per the `release` row that is exactly the necessary and sufficient action.
- **`NCCL_CUMEM_ENABLE=1` reproducing at TP=2** is not a scaling effect.
  Enabling cuMem makes NCCL use VMM allocations and take live imports; one
  import is enough. The earlier "pass rate tracks import count" reading was
  an artifact of partial teardown, not of count.
- **vLLM's custom all-reduce is reachable after all** — but only in job mode,
  and only if something closes its legacy imports. An earlier revision of this
  document concluded the opposite, from the standalone table alone; that was
  wrong, and it is exactly the error that running only the convenient
  configuration produces.
- **TASK.md's "measure before implementing: IPC taint" gate is answered, and
  it passes** for the VMM path (`teardown` and `release` both pass). Work
  items 1-4 are therefore sufficient; device memory content stays
  cuda-checkpoint's responsibility and does **not** become nvproxy's problem.

## Coverage gap this exposes, and the work item it implies

`mcshim.c` interposes the VMM API only — it contains **zero** references to
`cuIpc*`. So today any workload using legacy CUDA IPC is uncovered.

The job-mode table says that gap is **closable, by the mechanism already
built**. The interposer's VMM handling is `cuMemRelease` at suspend and
re-import at resume; the legacy path needs the same shape:

- intercept `cuIpcOpenMemHandle` / `cuIpcCloseMemHandle`, tracking each live
  import the way `KIND_IMP` already tracks VMM ones;
- `cuIpcCloseMemHandle` every live import at suspend (`legacy-close` proves
  this is sufficient);
- re-open at resume. Unlike VMM, the handle is an opaque 64-byte blob rather
  than an OS handle, so the existing FD-based rendezvous does not apply and
  the blob must be re-fetched from the exporter after restore.

### VA identity: measured, and it holds

`cuIpcOpenMemHandle` picks its own address — unlike `cuMemAddressReserve` it
takes no hint — so whether a reopened import lands where it was is the
driver's choice, not ours. Since a moved buffer is silent corruption rather
than an error, this had to be measured before building anything.

`legacy_va_probe.py`, job mode, 3/3 identical runs:

| question | answer |
| --- | --- |
| plain close then reopen | **same VA** |
| close, allocate something else, reopen | **moves** |
| close, checkpoint, restore, reopen | **same VA**, contents intact |
| is the 64-byte IPC blob still valid after restore? | **no, it changes** |

So the driver hands out the next free slot rather than a stable address. The
VA is preserved across checkpoint/restore *provided nothing perturbs the
allocation state in between*, which yields two rules for the interposer:

1. **Reopen in the original order, with no allocations interposed.** This is
   the same class of ordering discipline mcshim already enforces for
   multicast, so it fits the existing structure rather than fighting it.
2. **Re-fetch the blob from the exporter after restore.** The existing
   rendezvous serves an FD; legacy IPC needs a variant that serves 64 bytes.

Rule 1 is a real constraint, not a formality: the interposed-allocation case
moves the import by exactly one slot, so any allocation slipped into the
resume path — by the interposer itself, or by a rank that is not fully
quiesced — would silently relocate a buffer the application still points at.

## Notes for anyone re-running this

- **Always run both standalone and job mode.** The legacy result inverts
  between them. Testing only the convenient one produced a confidently stated
  and wrong conclusion here once already.
- The two error strings for the same VMM defect (`"invalid argument"` when
  mapped, `"out of memory"` when not) are the same refusal; do not treat them
  as separate bugs.
- The probe framed its messages over a `SOCK_STREAM` socketpair. A single
  `recv_fds` can return two messages, and dropping the remainder deadlocks
  the peer that waits for the swallowed one. That is a race: it passed at
  `--world 2` and hung at `--world 4`. `Chan` in the probe buffers properly;
  `_cuda.recv_msg` still has the naive behaviour and is only safe when every
  message is separated by a round trip.
- Legacy IPC handles are 64-byte structs passed **by value**. They need a real
  `ctypes.Structure`; passing a pointer or `bytes` fails with
  `CUDA_ERROR_INVALID_VALUE`, as does letting ctypes default a `CUdeviceptr`
  argument to 32-bit `int`.
