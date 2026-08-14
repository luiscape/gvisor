# Plan: make the custom all-reduce use-case checkpointable

`CUSTOM_ALLREDUCE_ROOT_CAUSE.md` establishes the diagnosis. This is the plan to
close the gap. Status quo: the workflow is 8/8 with custom all-reduce off and
~60%-failing with it on; every failure is a restore-toggle failure caused by
live legacy CUDA IPC imports.

## Framing: three routes, not one

The *use-case* is a fast small-message all-reduce that survives
checkpoint/restore with captured CUDA graphs intact. There are three distinct
ways to get it, they are not exclusive, and they have very different risk
profiles:

| | Route | Mechanism | Transparent? | Risk |
| --- | --- | --- | --- | --- |
| **T1** | Interposer close + retained VA + exact re-place | ours | yes | one unexplained failure left |
| **T2** | Make the driver's live-import restore reliable | NVIDIA's | yes | not in our control |
| **T3** | Serve the use-case over VMM instead (symm-mem / FlashInfer all-reduce) | vLLM's | config-only | perf parity + engagement unknowns |

Work T1 as the main line, T3 in parallel because it is cheap and might satisfy
the use-case outright, and T2 opportunistically because the reproducer already
exists.

A hard constraint shared by every route: **the peer-buffer addresses are baked
into captured CUDA graphs.** Any solution that brings a buffer back at a
different address is not a solution, whether it errors or not. This is why
engine-side re-registration on wake (a fourth route) is not listed: it has the
same VA-identity requirement as T1 but with vLLM's cooperation needed too, and
its only escape hatch -- re-capturing graphs after wake -- forfeits the warmed
state that makes snapshotting valuable. It becomes interesting only if T1's
mechanism is shown to be impossible.

---

## Track 1: interposer close + retain + re-place (main line)

### Where it stands

Everything is proven except one step:

- Closing imports before the checkpoint makes the toggle reliable
  (0 failures/3 runs). **Proven.**
- `cuIpcOpenMemHandle` placement is steerable and exactly controllable with
  `cuMemAddressReserve` fences (`legacy_va_probe.py`: `exact_placement=True`).
  **Proven in isolation.**
- VA reservations survive checkpoint/restore -- the VMM path's
  "retained-reservation" remap depends on exactly this. **Proven by the VMM
  path.**
- Taking the reservation over a just-closed import's range **fails in vLLM**
  (72/72 refused-or-mislanded). **The open problem.** Once the range is held,
  the rest is machinery that already exists.

### Step 1.0 — instrument before theorizing (half a day)

The current failure log conflates three cases. Add to `mcshim.c`:

- the `cuMemAddressReserve` return code AND returned address (the address
  argument is a hint; "succeeded elsewhere" and "failed" are different bugs);
- at suspend, after each refusal: probe the range with `cuMemGetAddressRange`
  to learn what occupies it;
- at resume, before each reopen: the same probe on the target address.

Exit criterion: for one failing vLLM run, we can say for every import whether
the reserve failed or mislanded, and what sat on the target at reopen time.

### Step 1.1 — two-pass suspend (half a day)

Current code closes and reserves one import at a time, so a granule-rounded
reservation is requested while neighbouring imports are still mapped
(vLLM's are `0x801300` bytes at `0xA00000` spacing -- rounding to `0xA00000`
abuts the neighbour exactly, and any driver-side padding makes that an
overlap). Restructure: close **all** imports, then reserve **all** ranges.

### Step 1.2 — a fast iteration rig (half a day, pays for itself immediately)

Every T1 iteration currently costs an 8-minute vLLM cycle. Build the
vLLM-shaped case into the native probes instead: W processes, each holding
~10 legacy IPC imports at vLLM's real size/spacing **plus** VMM imports and a
multicast group, driven through mcshim suspend/checkpoint/restore/resume under
`--launch-job`, with a large allocation freed before the checkpoint to mimic
`/sleep`. Cycle time ~1 minute. `mcshim_mp.py` / `ipc_scale_probe.py` provide
most of the parts.

### Step 1.3 — close the placement loop

With 1.0's data, apply the fix the data indicates. Ranked hypotheses:

1. Reserve mislands because the hint is unsatisfiable at that instant
   (neighbour overlap -- fixed by 1.1).
2. The range is consumed between close and reserve by the driver's own
   allocator; if so, reserve *bigger* regions spanning whole import clusters
   rather than per-import granules.
3. The IPC arena tolerates reservations for *steering* (probe) but the closed
   range itself is not reservable in a process with the full vLLM allocation
   history; if so, invert the approach -- do not reserve the target, fence
   everything *around* it at resume, which the probe proved works.

Fallback if all placement control fails in situ: T1 is impossible as designed;
escalate to T3 as the supported path and T2 as the long-term fix, and document
why with the 1.0 evidence.

### Execution log (updated as it happens)

- **1.0 + 1.1 landed and paid off immediately.** The distinct-outcome logging
  plus the two-pass suspend took the reservation problem from 72/72
  refused-or-mislanded to **30/30 ranges held**. The one-at-a-time
  close/reserve interaction was the whole story on the suspend side.
- **Retention alone was not sufficient** (as suspected): with targets held,
  reopens landed in *other* free holes below -- between import clusters and
  where `/sleep` freed memory. A single fence over `[landed, target)` is
  unbuildable (it would cross the other imports' held reservations), so the
  walk was added: wherever an open lands is by construction the lowest free
  hole; close it, plug exactly that hole, reopen; plugs persist until every
  import is placed. ~183 plugs, 2 walked-back imports per worker in practice.
- **Results:** TP=4 custom-AR-on **5/5** (was 3/5), TP=2 **3/3**, recommended
  config and PyTorch NVLS tier regression-free. `MCSHIM_IPC_SUSPEND=1` is the
  switch; still opt-in.
- **TP=8 exposes a residual sub-problem, distinct from everything above:**
  `7 reopened at identical VAs, 1 walked back, 21 MOVED`. The 21 movers'
  *original* addresses live in a **low VA region** (`0x31ce400000`-ish),
  nowhere near the `0x7e...` arena where reopens land -- so they land *above*
  target and the walk (which only handles landing below) never engages.
  Plausibly these buffers were originally placed adjacent to vLLM's sleep-pool
  VMM reservations at init. Next moves, in order: (a) confirm what the low
  region adjoins (log `cuMemGetAddressRange` neighbours at *track* time, not
  suspend time); (b) if the region is reconstructible, hold-and-walk works
  there too once the arena choice is understood -- the open question is what
  makes cuIpcOpenMemHandle *choose* the low region at init but not at resume.
  TP=2/TP=4 do not exhibit this (all their imports live in the high arena).

### Step 1.4 — acceptance

- Native rig: 10/10 with all imports at identical VAs.
- vLLM TP=2 and TP=4, custom AR on, `MCSHIM_IPC_SUSPEND=1`: 5/5 each
  (`vllm_trials.sh`), exact match.
- TP=8 single run, then 5 trials.
- Regression: the recommended config (custom AR off) stays green; the PyTorch
  NVLS tier stays green; `MCSHIM_IPC_SUSPEND` becomes default-on only after
  all of the above.

Estimated total: 2-4 working days, dominated by 1.3's unknowns.

---

## Track 2: driver reliability (opportunistic, parallel)

The driver *does* restore live legacy imports -- 58 and even 126 of them -- just
not reliably. Two moves:

1. **File the NVIDIA bug now.** The reproducer already exists and is native:
   `ipc_scale_probe.py --stage legacy-import` under `--launch-job` passes
   ~30% (2/6). No gVisor, no CRIU, ~200 lines. This is precisely the kind of
   artifact that gets driver bugs fixed, and every other track benefits if it
   is.
2. **Second-pass toggle retry** in `restoreCudaProcs`: after the first pass,
   wait ~2 s, retry failures (bounded). The handoff says retry "had no
   effect", but that was measured before the interposer handled multicast and
   VMM imports -- the failure landscape has since changed completely, and the
   failures are clean per-process errors, not corruption. One gVisor rebuild
   (~20 min) + 5 trials answers it. If retries converge, this alone might make
   live-import mode acceptable without T1 -- worth knowing even if T1 lands,
   as belt-and-braces.

Also fix while in there (cheap, unconditional): `cudaProcs` ordering comes
from a Go map iteration, so failures are unreproducible run to run. Sort by
PID. Not the cause of anything, but nondeterminism in a checkpoint path is a
debugging tax everyone pays.

---

## Track 3: serve the use-case over VMM (cheap, parallel)

The point of custom all-reduce is fast small-message all-reduce. vLLM has a
second implementation of the same idea on a checkpointable substrate:
`VLLM_ALLREDUCE_USE_SYMM_MEM=1` (torch symmetric memory -- VMM + multicast,
which the interposer already handles; the PyTorch tier proves symm-mem
suspend/resume at WORLD=2/4/8).

Known trap from the earlier sweep: the vLLM symm-mem cell **never actually
engaged** (size-gated), so that cell proved nothing. The work here is:

1. Confirm engagement: `NCCL_DEBUG` off, instrument or log
   `torch.ops.symm_mem` calls / check `multicast_ptr != 0` per worker; make
   the harness fail the cell if symm-mem did not engage rather than
   vacuously pass.
2. C/R correctness with symm-mem on and custom AR off: 5 trials, TP=4.
3. Perf: tokens/s and latency, custom AR vs symm-mem vs neither, TP=4/TP=8.
   If symm-mem is within a few percent of custom AR, the *use-case* is served
   today, and T1 becomes about generality (SGLang, other engines, FlashInfer)
   rather than urgency.

FlashInfer all-reduce (`VLLM_ALLREDUCE_USE_FLASHINFER`) belongs here too but is
gated on the CUDA-toolchain mismatch noted in the handoff, and its IPC basis is
unknown -- classify it (legacy vs VMM) before spending anything on it.

---

## Sequencing and decision points

```
week 1:  T1 1.0-1.2 (instrument, two-pass, rig)     T3.1-3.2 (engage + C/R)
         T2.1 (file NVIDIA bug)
decide:  if T1 1.3 lands -> T1 1.4 acceptance, ship MCSHIM_IPC_SUSPEND default
         if T1 blocked   -> T3 becomes the supported answer for the use-case;
                            keep custom AR behind a retry loop (T2.2) with its
                            measured rate; revisit T1 when NVIDIA responds
```

The honest bottom line as of today: **T1 is one placement bug away from a full
transparent solution**, and everything else about it is already measured to
work. But it is exactly the kind of bug that can hide driver policy we cannot
override, which is why T3 runs in parallel rather than after.
