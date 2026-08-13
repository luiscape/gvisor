# What actually blocks cuda-checkpoint: a two-API bisect

Reproducer: `ipc_scale_probe.py`. Native (no gVisor, no runc, no CRIU), no
NCCL, no PyTorch, no multicast, no CUDA graphs, no inference engine. W
processes, one GPU each, ~140 lines of `ctypes` against `libcuda.so`.

Environment: 8x H100 (NVSwitch), driver **610.57.04**, `cuda-checkpoint`
driving `lock -> checkpoint -> restore -> unlock` (the same phased sequence as
`pkg/sentry/control/state_cuda.go`).

```
python3 ipc_scale_probe.py --world 2 --allocs 1 --stage everything --trials 3
```

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

### Legacy CUDA IPC (`cuIpcGetMemHandle` / `cuIpcOpenMemHandle`)

| stage | what the process holds at checkpoint time | result |
| --- | --- | --- |
| `legacy-alloc` | `cuMemAlloc`, never shared | pass |
| `legacy-export` | + `cuIpcGetMemHandle` (handle never even leaves the process) | **FAIL** — checkpoint: `"OS call failed or operation not supported on this OS"` |
| `legacy-import` | + peers `cuIpcOpenMemHandle` | **FAIL** — checkpoint |
| `legacy-close` | peers `cuIpcCloseMemHandle` before the checkpoint | **FAIL** — checkpoint |
| `legacy-free` | peers close **and** the exporter `cuMemFree`s the allocation | pass |

This is a different defect with an inverted shape:

- It fires on the **exporter**, at **checkpoint** rather than restore.
- Merely *calling* `cuIpcGetMemHandle` is enough. No peer has to do anything;
  the handle need not leave the process.
- The importer closing its handle does **not** clear it. Only destroying the
  exported allocation does, and an application cannot do that — it is its data.

So legacy IPC is **not** recoverable by any suspend/resume protocol short of
freeing and repopulating device memory.

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
- **vLLM's custom all-reduce is out of reach of any interposer.** It uses the
  legacy `cuIpc*` path, whose taint cannot be released. This retroactively
  justifies requiring `DISABLE_CUSTOM_ALL_REDUCE=1`: it is not a workaround
  we have failed to remove, it is a hard consequence of the second defect.
- **TASK.md's "measure before implementing: IPC taint" gate is answered, and
  it passes** for the VMM path (`teardown` and `release` both pass). Work
  items 1-4 are therefore sufficient; device memory content stays
  cuda-checkpoint's responsibility and does **not** become nvproxy's problem.

## Coverage gap this exposes

`mcshim.c` interposes the VMM API only — it contains **zero** references to
`cuIpc*`. Any workload using legacy CUDA IPC is uncovered, and per the second
table cannot be covered by release-and-replay. The only options there are to
disable the feature in the application or to have the *allocation itself*
recreated across the checkpoint.

## Notes for anyone re-running this

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
