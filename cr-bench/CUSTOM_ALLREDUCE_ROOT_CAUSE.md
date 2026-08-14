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

### Step 2 works in isolation

`legacy_va_probe.py` reproduces the failure and repairs it. A filler occupies
the low slot so the import lands high; the filler is freed (as `/sleep` frees
the weights and KV cache); the reopen then packs low, exactly as in vLLM; and
fencing the gap puts it back on the nose:

```
reopen after freeing the filler   va=0x7a4e39e00000 (target 0x7a4e3a200000, MOVED DOWN)
reopen with gap 0x400000 fenced   va=0x7a4e3a200000 EXACT -- placement is controllable
placement_steerable=True exact_placement=True
```

### ...and does not survive contact with vLLM yet

Implemented in `mcshim.c` behind `MCSHIM_IPC_SUSPEND=1` and run against vLLM
TP=4, it does not fire, for a reason the probe could not have shown:

- With the interposer closing the imports, they come back **+21.6 GB above**
  their original addresses, not below. The fence-the-gap-underneath repair
  only applies to a downward move, so it never triggers (`0 needed
  re-placing`).
- The upward move is self-inflicted. Closing an import **frees its VA range**
  before the checkpoint, so the restore is free to put something else there.
  The VMM path does not have this problem because it *retains* the address
  reservation across the checkpoint -- that is what "re-mapped IDENTICAL
  (retained-reservation)" in its log means.
- Applying the same retention to legacy IPC -- `cuMemGetAddressRange` before
  closing, then `cuMemAddressReserve` over the range -- currently **fails to
  take the reservation** (72 attempts, all refused), including after rounding
  the size up to a 2 MB granule (an IPC allocation's size is whatever the
  exporter asked for; vLLM's is `0x801300`). `cuMemAddressReserve`'s address
  argument is a hint rather than a requirement, so it can also silently return
  a different address.

So the diagnosis is complete and the repair is not. The next things to try,
in order:

1. Log the actual return code and returned address from the failing
   `cuMemAddressReserve`, rather than inferring. The current message does not
   distinguish "call failed" from "succeeded at the wrong address".
2. Close **all** imports first, then reserve all the ranges in a second pass.
   The current code closes and reserves one import at a time, so a rounded
   reservation is taken while neighbouring imports are still mapped.
3. Only then revisit the fence-the-gap repair, which is what handles whatever
   residual drift the retention does not.

`MCSHIM_IPC_SUSPEND` remains **off by default**, so none of this is in the
shipped path, and the recommended configuration still passes (verified after
these changes: TP=4, exact match).

## Current recommendation

Until that is built, `DISABLE_CUSTOM_ALL_REDUCE=1`. The cost of leaving it on
is roughly a third of restores, and the failures are clean toggle failures
rather than corruption, so a retry loop is a legitimate alternative if the
inference speedup matters more than restore latency.
