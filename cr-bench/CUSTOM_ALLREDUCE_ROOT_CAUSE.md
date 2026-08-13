# Why vLLM custom all-reduce breaks restore, and what would fix it

vLLM's custom all-reduce is the one remaining feature that costs reliability in
the sleep/checkpoint/restore/wake workflow: **3/8 pass with it on, 8/8 with it
off** (TP=4, `NCCL_CUMEM_ENABLE=1`, otherwise identical). This is what is
actually going on.

## The chain

1. **vLLM registers peer buffers over legacy CUDA IPC at init.**
   `CustomAllreduce.__init__` calls `cudaIpcGetMemHandle` and exchanges the
   handles, so every rank imports every peer's buffer. The count is
   `buffers x (world - 1)`: 10 live imports per worker at TP=2, 30 at TP=4.
   This happens **whether or not the collective is ever used**, so it is not
   avoidable by making NCCL win the runtime dispatch (e.g. with NVLS).

2. **`/sleep` does not release them.** Sleep level 1 offloads weights and drops
   the KV cache through `CuMemAllocator`; the custom all-reduce buffers are
   registered once at init, outside that scope. They are still live at
   checkpoint time.

3. **`cuda-checkpoint --toggle` then has to reconstruct them at restore, and
   often cannot.** The failure is always the same shape:

   ```
   Error toggling CUDA in process ID 683: "unknown error"
   ```

   always on TP workers -- never on the API server or the EngineCore -- and it
   takes down a varying subset of them.

## What it is not

**It is not toggle ordering.** `cudaProcs` is built by
`TaskSet.ForEachThreadGroup`, which iterates `ts.Root.tgids` -- a **map**, so
the order is different on every run. That looked like the obvious cause of the
intermittency, and it is not:

| toggle order | failed |
| --- | --- |
| `681 457 679 680 682 1` | `681` |
| `682 681 684 1 458 683` | `682 681` |
| `459 683 684 685 1 682` | `683 684 685` |

No position rule fits all three. And making the toggle **parallel** instead of
sequential does not help either: 1/3 parallel vs 3/8 sequential.

(The nondeterministic order is still worth fixing on its own account -- it makes
failures unreproducible -- but it is not the cause.)

**It is not the import count.** No threshold survives contact with the data:
126 live imports passed, 10 live imports failed.

## What it is

**Live legacy IPC imports at toggle time.** Removing them removes the failure,
exactly:

| legacy imports live at checkpoint | toggle failures |
| --- | --- |
| yes (custom all-reduce on) | ~60% of runs, `"unknown error"` on TP workers |
| **no** (interposer closes them, `MCSHIM_IPC_SUSPEND=1`) | **0 of 3 runs** |

That is the whole diagnosis. The driver's job-mode IPC support does carry live
imports -- it demonstrably works, including 126 of them in one run -- but not
reliably at this scale, and nothing about *how* we drive the toggle changes
that.

## Why the obvious fix does not work yet

The interposer can close the imports before the checkpoint (which makes the
toggle 100% reliable) but cannot put them back where they were:

```
legacy IPC done (0 reopened at identical VAs, 30 MOVED, 10 served)
```

`cuIpcOpenMemHandle` takes no address hint. The driver hands out the lowest
free address in its region, and by resume time the address space is not what it
was when the application first opened these handles -- `/sleep` has freed the
weights and KV cache that used to sit below them. So a fresh open packs low,
while the originals sit high, and every import moves. The application still
holds the old pointers, in host structures and inside captured CUDA graphs, so
the interposer fails the resume loudly rather than letting that through.

## The fix, and the measurement that makes it plausible

Placement has no *hint*, but it is **steerable**: `cuMemAddressReserve` does
take a fixed address, and a reservation counts as occupied for
`cuIpcOpenMemHandle`'s placement. Measured (`legacy_va_probe.py`, job mode):

```
open #1                       va=0x77b999e00000
open #3b (natural slot 0x77b999e00000 reserved) va=0x77b99a200000
        AVOIDED the reservation -> steerable
placement_steerable=True
```

So the driver's choice can be fenced. The proposed replay, per import, in
original open order:

1. open the import; if it lands on the recorded address, done;
2. otherwise it landed **below** the target (the measured failure direction).
   Close it, `cuMemAddressReserve` the gap `[landed, target)`, and reopen. With
   everything below the target fenced off, the lowest free address in the
   region is the target itself;
3. free the reservation once the import is placed.

This is unbuilt. What is measured is that each step's primitive behaves as
required -- reservations steer placement, closing imports makes the toggle
reliable, and the interposer already tracks every import with its original
address and open order. What is not yet known is whether step 2 lands exactly
on the target rather than merely above it, which is the thing to test first,
and cheaply, in `legacy_va_probe.py` before touching `mcshim.c`.

## Current recommendation

Until that is built, `DISABLE_CUSTOM_ALL_REDUCE=1`. The cost of leaving it on
is roughly a third of restores, and the failures are clean toggle failures
rather than corruption, so a retry loop is a legitimate alternative if the
inference speedup matters more than restore latency.
