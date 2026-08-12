# mcshim — generic libcuda-level multicast checkpoint interposer (Idea D)

**PASS native + PASS under gVisor** for raw-multicast workloads, single- and
multi-process (WORLD=2/4, driver 610.57.04, H100 NVSwitch), **and for stock
NCCL NVLS + CUDA graphs end-to-end under a full `runsc checkpoint`/`restore`**
(WORLD=4) — see the resolution below.

---

## RESOLVED (2026-08-10, later) — supersedes both sections below

The stock-NCCL `719` was **not** a shim limitation, not IPC taint, and not
cuda-checkpoint. It was an **orchestration race in the harness**, and the
transparent shim is sufficient for NCCL after all.

### Root cause

`runsc restore -detach` returns as soon as tasks are runnable, but
`postRestoreCuda`'s `cuda-checkpoint --toggle` keeps rebuilding GPU state for
**~8.5 s longer**, and the application's non-CUDA threads — including the
shim's marker-polling control thread — run concurrently with it. The harness
removed the `suspend` marker ~2 s after restore returned, so whichever rank
happened *not* to be frozen by the toggle at that instant resumed and rebuilt
multicast on a context whose device state had not been restored yet. That
rank's context latched an unrecoverable fault (sticky `719`); every other rank
resumed after the toggle and was fine.

This exactly matches every previously-confusing symptom: one rank, a
*different* rank each run, NCCL silent, resume reporting full success
(`served=48 rebound=1 remapped=49` — VMM control-plane calls succeed on a
faulted context because they never touch the device), and sensitivity to
machine state (which only shifted the timing of the race).

### Evidence

A `CTXPROBE` (a `cuCtxSynchronize` whose result is logged) at suspend entry,
suspend exit, and resume entry localises the fault precisely:

| Probe | faulting rank | other ranks |
| --- | --- | --- |
| `suspend-entry` | 0 | 0 |
| `suspend-exit` | 0 | 0 |
| `resume-entry` | **719** | 0 |

So the shim's teardown is innocent — every context is healthy after suspend.
Two further controls close it out:

- `TOGGLE_ONLY=1` (the same `cuda-checkpoint` lock/checkpoint/restore/unlock
  sequence, driven in place, with **no** gVisor save/restore): **2/2 PASS**,
  16/16 clean probes. Exonerates cuda-checkpoint.
- In the failing runs the single `status=0x57` frontend-ioctl failure in the
  restore sandbox is on the faulting tgid at the exact microsecond of its
  `719` probe, ~6.4 s *before* `cuda-checkpoint --toggle for PID N succeeded`
  was logged for it. The resume provably ran mid-toggle.

### Fix and result

Wait for the toggle to finish on every rank before signalling resume (the
harness now polls the boot log for `cuda-checkpoint --toggle for PID .*
succeeded` × WORLD). With that ordering:

| Flow | Before | After |
| --- | --- | --- |
| full `runsc checkpoint` + `restore`, stock NCCL NVLS + CUDA graph, WORLD=4 | 4/4 FAIL | **3/3 PASS**, 0 context faults |

**Conclusion: a purely transparent, fork-free shim IS sufficient for stock
NCCL NVLS.** There is no "stale NCCL device descriptor" ceiling — consistent
with the NCCL patch itself, whose `nvlsResumeOne` rebuilds *no* NCCL
descriptors and relies solely on VAs being identical.

The real requirement is **staging**: the resume must not begin until
cuda-checkpoint has finished restoring GPU state. In production this ordering
should be owned by gVisor (drive the shim from `postResumeCuda` *after* the
toggle) rather than by an external poller racing `runsc restore`.

### Disproven along the way

`MCSHIM_UC_REBUILD` (1 = re-create IPC-exported unicast allocations fresh at
resume, 2 = all; default 0) was added to test the hypothesis that
cuda-checkpoint-restored allocations are "IPC-tainted" and cannot be cleanly
re-exported — mirroring NCCL's policy of never carrying IPC-exported memory
through a checkpoint. **It did not help** (mode 1 failed *worse*: the faulting
rank aborted resume at its first `HtoD`, starving peers into `700`). The knob
is retained only as a documented negative result; the default path is correct.

---

## gVisor-DRIVEN integration (2026-08-11) -- WORKING

The interposer is driven by gVisor itself; nothing outside gVisor touches it.
`runsc --cuda-multicast-shim-path=<mcshim.so>` makes the sentry LD_PRELOAD it
into a GPU container and export `MCSHIM_DIR` (`runsc/boot/loader.go`), and
`pkg/sentry/control/state_cuda_shim.go` drives both transitions around the
cuda-checkpoint phases.

Harness: `run_nccl_shim_gvisor_driven.sh` (stock NCCL NVLS + CUDA graph,
WORLD=4). Result: **PASS, 0 context faults, reproducible.**

### Ordering rules, all established empirically

1. **Suspend before `cuda-checkpoint --action lock`.** The interposer tears
   down through libcuda, and a locked process cannot submit that work
   (observed: no rank acknowledges and the suspend times out).
2. **Verify, do not trust.** After the suspend, the blocker gate is re-run with
   nothing exempt, so anything left unreleased fails the checkpoint loudly
   instead of producing a snapshot that only misbehaves after restore.
3. **Rebuild after the restore toggle has completed on every rank**, which is
   where `postResumeCuda` already sits.
4. **The application must not run GPU work until the rebuild completes.**

### Rule 4, and a warning about how easy it is to misdiagnose

Rule 4 cost the most to find, and produced a long chain of wrong conclusions
worth recording so they are not repeated.

The symptom was that a gVisor-driven rebuild faulted **every** rank with
`CUDA_ERROR_ILLEGAL_ADDRESS` (700), while an externally-driven rebuild was
clean. Ruled out, each by direct experiment: the `SigsegvLock` window (moving
the rebuild after `SigsegvUnlock` changed nothing); waiting for
`cuda-checkpoint --get-state` to report every process `running`; a settle
delay; running the rebuild asynchronously; the interposer's unicast policy
(`MCSHIM_UC_REBUILD`); async MemoryFile page loading (the log shows
`MFs loaded` completing ~10 s *before* the rebuild); and the injection method
(spec env vs `procArgs.Envv`, i.e. whether the cuda-checkpoint helpers are
themselves preloaded).

The actual cause was in the **harness**: it removed the workload's `pause`
marker about two seconds after `runsc restore` returned, but gVisor's rebuild
runs in `postResumeCuda`, which finishes *later* than the restore RPC. The
ranks therefore resumed collectives against multicast VAs that were still
unmapped. Every rank faulting, and the fault being immune to every delay added
*inside* gVisor, are both explained by this -- a longer internal delay makes it
strictly worse, because the application is unpaused even earlier relative to
the rebuild.

Two traps made this hard to see:

* An `EXTERNAL_RESUME=1` bisect appeared to prove the sentry-side rebuild was
  at fault. It proved nothing: **`runsc` clears the sandbox environment**
  (`cmd.Env = []string{}` in `runsc/sandbox/sandbox.go`), so that knob -- and
  every other env-based knob -- never reached the sentry. gVisor rebuilt
  inline in *all* of those runs. What the flag actually changed was that the
  harness waited for the toggle before unpausing.
* `CTXPROBE` at resume entry reports the context already faulted, which looks
  like "something before the rebuild broke it". It was the *application*,
  running ahead of the rebuild.

Fix: wait for the interposer's per-rank `resumed` acknowledgements before
letting the workload run.

### Production implication -- ADDRESSED, and now verified

This section previously read "not yet addressed". It is stale: the gate was
implemented (`GATED(...)` wrappers around `cuLaunchKernel`,
`cuLaunchKernelEx`, `cuLaunchCooperativeKernel`, `cuGraphLaunch`, the memset
and memcpy families, and `cuStreamSynchronize`), so a suspended interposer
blocks the application in libcuda rather than relying on it being idle.

What was missing was a test: every harness still paused the application
itself, so the gate was never the thing doing the quiescing. That gap is now
closed by `run_torch_nccl_gvisor.sh NO_PAUSE=1 MECH=mcshim`, which runs the
PyTorch tier with **no application-level pause at all** -- a workload
continuously replaying a captured CUDA graph of an NCCL collective, which is
the hardest case, since a graph replay does not re-enter NCCL and cannot be
stopped by anything above libcuda.

Result: **3/3 PASS**, with all four ranks logging
`GATE: app thread blocked until resume` during the checkpoint and returning
correct results afterwards at unchanged VAs. So the guarantee really is
app-transparent, which is what a real engine needs -- `cuda-checkpoint
--toggle` restores *and unlocks* each process, so the application becomes
runnable while gVisor is still rebuilding.

One harness note worth keeping, because it produced a spurious failure first:
`runsc restore -detach` returns before `postRestore` has finished rebuilding,
and under `NO_PAUSE` the ranks are still gated at that moment. A fixed sleep
before checking their status is therefore a race; poll for the post-restore
line instead.

---

## Earlier correction (2026-08-10) — superseded by the RESOLVED section above

The earlier conclusion that a cuda-checkpoint-dropped `ncclCommInitRank`
`/dev/nvidiactl` page was the root cause of the stock-NCCL `719` was **WRONG**
(correlation, not causation). Decisive evidence: a run with `pre=5 post=4`
(page dropped) **PASSED**. The page is a benign transient init mapping that
drops in passing runs too. (The `nccl_commninit_page_probe.py` observation is
real but irrelevant to the fault; the reverted nvproxy replay fix was aimed at
a red herring — correctly reverted.)

The real picture: the transparent stock-NCCL path has **two** failure modes.

- **Mode 1 — re-export race (FIXED).** During resume phase-1,
  `cuMemExportToShareableHandle` on a freshly-restored UC buffer transiently
  returns `INVALID_VALUE`; the rank aborts resume, peers time out fetching its
  buffers → one-rank failure (~10%). Fixed by a bounded retry in
  `reexport_serve` (mirrors the import-side `304` retry). With it, stock NCCL
  reached **15/15 PASS** in one environment state.
- **Mode 2 — stale NCCL device state (OPEN, likely intrinsic).** Resume
  completes fully on all ranks (`served=48 rebound=1 remapped=49`), NCCL logs
  nothing (`NCCL_DEBUG=WARN` silent), yet one rank's collective kernel faults
  `719`. It is **device-state-sensitive**: 15/15 PASS in a "warmed" box,
  reliably FAIL (4/4) right after a clean `nvidia-smi -r` + fabric reinit. The
  **patched-NCCL flow is robust across all these states**, so mode 2 is
  specific to the transparent approach, not the environment.

Interpretation (the answer to "can a purely transparent shim be universal for
NCCL?"): **no, there is a ceiling.** The shim restores *memory* — every VA
and mapping byte-identical, proven sound to 256 buffers with real NVLink
read+write kernels by `p2p_reexport_probe.py` — but it cannot rebuild NCCL's
*internal device-side bookkeeping* (channel/connection descriptors, FIFO
state) that references those buffers and that `ncclCommSuspend`/`Resume`
re-derives. When the shim silently releases + re-imports underneath NCCL, VAs
match (steady-state kernels work) but any descriptor encoding more than a VA
can be stale → the intermittent, state-dependent `719`. This is exactly why
the NCCL-cooperative path exists and is robust.

**Consequences for the roadmap:**
1. **Robust solution today = the NCCL patch** (`NCCL_PATCH.md`): reliable
   across every environment state tested. It is the path to a working vLLM
   e2e now.
2. **Transparent shim** remains the right generic mechanism for owners that
   rely only on VAs (raw `cuMulticast`, and — to be checked — torch
   `_symmetric_memory`), and its memory-replay core (identity oracle,
   VA-stable multicast/import rebuild) is validated. For NCCL specifically it
   needs to stop being *purely* transparent: a **hybrid** — the shim (or
   gVisor) additionally drives NCCL's own `ncclCommSuspend`/`Resume` around
   the checkpoint so NCCL rebuilds its descriptors, while the shim handles
   any non-NCCL multicast owners. That removes the mode-2 ceiling without
   forking NCCL's internals.
3. Everything below this line predates the correction; the memory-mechanism
   findings stand, but the "cuda-checkpoint page = root cause" attribution
   does not.

---

## What this is

An `LD_PRELOAD` shim that makes a process holding **live multicast (0x00fd)
objects** checkpointable by `cuda-checkpoint` — with **no NCCL fork, no
PyTorch/engine hooks, and no application changes**. It is the generic
equivalent of the NVLS suspend/resume this exploration previously added to
NCCL (`NCCL_SUSPEND_RESULTS.md`), moved one layer down so it covers *every*
multicast owner: NCCL NVLS, torch `_symmetric_memory`, and raw `cuMulticast`.

The shim interposes the CUDA driver multicast + VMM entry points, maintains a
live table of every multicast group (prop, devices, bindings, backing UC
handles, VA mappings + access descriptors, owning contexts), and on demand:

- **suspend**: unmap MC VAs (**keeping the VA reservations**), unbind each
  bound device, `cuMemRelease` the 0x00fd handles. The multicast blocker set
  is now empty, so `cuda-checkpoint checkpoint` succeeds instead of hanging.
- **resume**: `cuMulticastCreate` (new handle) → re-`AddDevice` → re-bind the
  same UC handles (restored verbatim by cuda-checkpoint) → re-map at the
  **identical VAs** (retained reservation first; fixed-address re-reserve as
  fallback) → re-apply access.

Because every VA is byte-identical after restore, app pointers and captured
CUDA graphs stay valid; only the opaque MC handle changes, and the shim
translates stale handle values in later calls (aka-handle table), so even
app-side teardown after a resume keeps working.

## Why in-process (the constraint this satisfies)

Freeing the 0x00fd objects from nvproxy makes the checkpoint SAVE succeed but
the restore toggle proactively refuse (`"unknown error"`): libcuda's userspace
bookkeeping still lists the allocation. The teardown must run **through
libcuda in the process**, keeping app structs, libcuda tables, and kernel RM
state consistent. An LD_PRELOAD shim is in-process by construction — same
reason the NCCL patch worked, without the fork.

## Key implementation findings

1. **Symbol interposition alone does not intercept anything.** Real CUDA
   consumers (torch, NCCL, and this repo's ctypes harnesses) `dlopen`
   `libcuda.so.1` and `dlsym` the entry points, or use `cuGetProcAddress` —
   both bypass classic LD_PRELOAD symbol resolution. The shim therefore
   interposes **`dlsym` itself** (real `dlsym` recovered via `dlvsym`,
   trying `GLIBC_2.34` then `GLIBC_2.2.5`) plus **`cuGetProcAddress`/`_v2`**,
   redirecting lookups of tracked names to its wrappers. Verified: without
   this, zero calls were seen; with it, all calls route through the shim.
2. **Resolve real driver symbols against an explicit `dlopen("libcuda.so.1")`
   handle, not `RTLD_NEXT`** — consumers load libcuda with `RTLD_LOCAL`, so
   it may not be in the scope `RTLD_NEXT` searches.
3. **Unbind is per-device-hosting-the-memory**: record the device from the UC
   alloc's `location.id` at bind time and unbind each bind exactly once
   (unbinding a bind on a non-hosting device returns `INVALID_VALUE`).
4. **Control thread starts lazily**, only in processes that actually resolve
   a tracked CUDA symbol. Everything in the container (exec helpers,
   cuda-checkpoint itself via the sentry's spec-env provisioning) may inherit
   `LD_PRELOAD` + `MCSHIM_DIR`; a short-lived helper must not consume a
   suspend/resume marker meant for the CUDA workload.
5. The tracked state is a **live object graph, not an ioctl log**:
   app-initiated `cuMemRelease`/`cuMulticastUnbind`/`cuMemUnmap` (outside the
   shim's own suspend) drop entries, so freed objects leave the replay set —
   the same invariant TASK.md requires of nvproxy's graph.

## Control protocol

Existence-based marker in `$MCSHIM_DIR` (default `/tmp`), race-free for any
number of rank processes sharing the dir:

- `suspend` **appears** → each shim suspends once and acks `suspended.<pid>`
- `suspend` **disappears** → each shim resumes once and acks `resumed.<pid>`
- failures ack `error.<pid>` and wait for the next edge (no retry storm)

The marker lives in the container `/tmp`, so it is part of the checkpoint
image: after a restore the shims stay suspended until the orchestrator
removes it. Only processes that actually call `cuInit` join the protocol
(launchers and exec helpers that merely load libcuda never ack). Env:
`MCSHIM_DIR`, `MCSHIM_LOG`, `MCSHIM_DISABLE`. gVisor can drive this via
`runsc exec` (as the harness does) or `--save-restore-exec-argv` around the
`state_cuda.go` cuda-checkpoint phases; injection is env-only, the same way
`runsc/boot/loader.go` wraps the command in `cuda-checkpoint --launch-job`.

## Multi-process: cross-rank fd rendezvous (brokered by the shim itself)

In the process-per-GPU topology rank 0 creates + exports the group
(`cuMemExportToShareableHandle`) and peers import it
(`cuMemImportFromShareableHandle`) over the app's own channel (NCCL
bootstrap / torch rendezvous — the harness uses a launcher socketpair).
The shim records this graph using the **fd's `st_dev:st_ino` as the group
identity**: SCM_RIGHTS passes the same open file description, so exporter
and importers observe the same key (verified: `key=14:12` on all 4 ranks
under gVisor) plus a per-key ordinal for collisions.

On resume:

1. The creator's shim recreates the group, re-exports it, and **serves the
   new fd on a unix socket** `$MCSHIM_DIR/mcgrp-<key>.sock` (SCM_RIGHTS,
   accept loop on its own thread) — started *before* its own AddDevice/Bind
   so importers are never starved while the creator's bind blocks.
2. Peer shims connect (bounded retry), receive the fd, re-import. Concurrent
   imports can transiently fail (`CUDA_ERROR_OPERATING_SYSTEM=304` when
   several ranks import within ~1ms — seen under gVisor at WORLD=4); the
   shim retries with a fresh fd, bounded at 100 attempts so real failures
   stay loud.
3. Every rank re-adds its device and re-binds its local memory.
   `cuMulticastBindMem` blocks until all devices have joined, so **the binds
   themselves are the cross-rank barrier** — no extra synchronization needed.
4. Every rank re-maps its MC VA at the IDENTICAL address.

At the next suspend the creator closes the served fd and socket first — a
held export fd is itself a checkpoint blocker.

The served socket + fd exist only between resume and the next suspend, so
they are never part of a checkpoint image.

## Results (610.57.04, H100 NVSwitch)

### Native — `run_mcshim_native.py` — PASS

| leg | result |
|-----|--------|
| CONTROL (`CONTROL=1`): live MC, no suspend | lock ok; `checkpoint` **HANGS** (60s timeout) |
| MAIN: pause → shim suspend → lock/checkpoint/restore/unlock → shim resume → unpause | checkpoint **rc=0 (2.2s)**, restore **rc=0 (3.2s)**, MC VA re-mapped **IDENTICAL** (retained reservation), `post-restore+mc-live pass failures=0` |

### gVisor — `run_mcshim_gvisor.sh` — PASS

Workload under `runsc` (job-wrapped via `--cuda-checkpoint-path`), shim
injected purely via container env (`LD_PRELOAD` + `MCSHIM_*`):

```
pause -> PAUSED                       (quiesce; the idle-engine step)
suspend -> SUSPEND done: groups=1 unmapped=1 unbound=2 released=1
runsc checkpoint                      rc=0 (2s)   zero multicast blockers
runsc restore                         rc=0
resume -> RESUME: MC VA 0x406000000 re-mapped IDENTICAL (retained-reservation)
          RESUME done: groups=1 rebound=2 remapped=1
post-restore+mc-live pass failures=0
==== RESULT: PASS ====
```

Verification is R610-safe: unicast (bound vidmem) patterns survive the
round-trip, and the VA inventory — including the multicast VA — is
byte-identical. (Plain load/store through an MC VA is rejected on R610; only
multimem PTX exercises it, covered by the torch-based `symmem_nccl_ckpt.py`.)

### Multi-process — `run_mcshim_mp_native.py` / `run_mcshim_mp_gvisor.sh` — PASS

Launcher forks one rank per GPU under one `cuda-checkpoint --launch-job`;
rank 0 creates+exports, peers import; every rank binds local vidmem and maps
unicast + MC VAs; zero suspend logic in the workload.

| leg | result |
|-----|--------|
| native WORLD=2 | lock/checkpoint/restore/unlock rc=0 on every rank pid; all ranks `post-restore+mc-live pass failures=0` |
| native WORLD=4 | same, PASS |
| gVisor WORLD=2 | `runsc checkpoint` rc=0, restore rc=0; both shims re-established the group (creator recreated+served, peer re-imported); MC VA identical; PASS |
| gVisor WORLD=4 | same; one round of transient import-race retries absorbed; all 4 ranks `failures=0`; PASS |

## Files

- `mcshim.c` / `build.sh` — the interposer (toolkit-free; CUDA types declared
  locally, builds with plain gcc on a bare driver install).
- `../mcshim_workload.py` — transparent single-process workload: creates MC +
  binds + maps, loops verifying; contains **zero** suspend/resume logic. Only
  cooperation is the existence-based `pause` quiesce marker (the (a) step —
  an idle inference engine).
- `../mcshim_mp.py` — transparent multi-process workload: launcher forks one
  rank per GPU; rank 0 creates+exports, peers import (app-level socketpair
  handoff standing in for NCCL bootstrap).
- `../run_mcshim_native.py` — single-process native e2e (CONTROL hang + MAIN).
- `../run_mcshim_gvisor.sh` — single-process gVisor e2e.
- `../run_mcshim_mp_native.py` — multi-process native e2e (`--world N`,
  `CONTROL=1` for the hang control leg).
- `../run_mcshim_mp_gvisor.sh` — multi-process gVisor e2e (`WORLD=N`).
- `../fd_identity_probe.py` / `../run_fd_identity_gvisor.sh` — step-0
  measurement: are exported-object fd identities distinct? (Answer: no via
  fstat — all opens of `/dev/nvidiactl` — yes via the nvproxy fdinfo oracle.)
- `../reexport_probe.py` — single-alloc: does a VMM POSIX-FD allocation
  survive re-export after cuda-checkpoint? (Yes.)
- `../p2p_reexport_probe.py` / `../run_p2p_reexport_gvisor.sh` — the
  peer-access oracle: bidirectional N-buffer VMM P2P import with real SM
  NVLink read/write kernels, release+re-import across checkpoint. Proves the
  mechanism sound (all configs PASS), bounding the mcshim `719` to shim
  replay fidelity. Knobs: `P2P_NBUF`, `P2P_THREADED`.
- `../run_nccl_mcshim_gvisor.sh` — stock-NCCL acceptance under gVisor
  (`NCCL_NVLS_ENABLE`, `GRAPH` bisection knobs).

## Stock-NCCL NVLS validation (`run_nccl_mcshim_native.sh`) — boundary found

Ran the `nccl_suspend_mp.py` topology (4 ranks, NVLS, eager allreduce +
captured CUDA graph, `--pause-only` so **NCCL is stock and never calls any
suspend API**; stock build from `git archive HEAD` of `nccl/`, no NVLS-suspend
patch, staged at `/opt/phase0/nccl-stock/libnccl.so.2`). To rebuild it
(pristine sources; the working tree's patch is never touched):

```
mkdir -p /tmp/nccl-stock-src && (cd nccl && git archive HEAD) | tar -x -C /tmp/nccl-stock-src
sudo docker run --rm -v /tmp/nccl-stock-src:/nccl -w /nccl nvidia/cuda:13.0.1-devel-ubuntu24.04 \
  bash -c 'apt-get update -qq && apt-get install -y -qq git python3 && \
           make src.build -j64 NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"'
sudo mkdir -p /opt/phase0/nccl-stock
sudo cp /tmp/nccl-stock-src/build/lib/libnccl.so.2.30.7 /opt/phase0/nccl-stock/libnccl.so.2
```

| step | result |
|------|--------|
| CONTROL (no shim) | checkpoint **HANGS** — stock NVLS blocks, as expected |
| shim multicast suspend | ✅ all 4 ranks: `groups=1 unmapped=1 unbound=1 released=1` |
| `cuda-checkpoint checkpoint` | ✅ rc=0 on all ranks (multicast gone) |
| `cuda-checkpoint restore` | ❌ `"invalid argument"` on **every** rank |

Root cause isolated to a **minimal 2-process probe**: `ipc_taint.py --mode
hold` run inside a `cuda-checkpoint --launch-job` job — exporter restores
fine, the process **holding a live VMM POSIX-FD unicast import** fails
restore with the same `"invalid argument"`. So R610 job mode does **not**
cover live `cuMemImportFromShareableHandle` state, and stock NCCL ranks hold
~195 such imports each (P2P transport buffers). The patched NCCL passed
because `ncclCommSuspend(MEM)` releases those imports too; the shim so far
only releases the multicast layer.

Encouragingly, Phase 0's taint measurement already proved **released**
imports are fully checkpointable, and the importer side needs no content
backup (the content lives in the exporter's resident allocation). So
UC-import suspend/replay is mechanically identical to the MC path the shim
already implements: unmap keeping the reservation → release the imported
handle → re-import → re-map at the identical VA.

**The real blocker is identity.** This run disproved the fd `st_dev:st_ino`
rendezvous key at scale: **every NVIDIA export fd shares one anon inode**
(all groups and imports logged `key=7:55e`). The per-key ordinal saved the
few-groups mcshim_mp case, but cannot match ~195 interleaved, multi-exporter
imports. A generic shim has no robust cross-process buffer identity — but
**nvproxy does**: it proxies the export/import ioctls and fds and can hand
the shim (or perform itself) a true object-identity mapping. The gVisor
integration is therefore not just injection plumbing; it supplies the
missing oracle.

Bonus fix from this run: stock NCCL resolves `cuGetProcAddress` *through*
`cuGetProcAddress_v2`; handing back the 4-arg v1 wrapper for a v2-ABI
request left `symbolStatus` unwritten and crashed `ncclCommInitRank`
(ip=0 SIGSEGV). The resolver redirection is now ABI-aware (`gpa_redirect`).
Also added `cuMulticastBindAddr` tracking/replay (stock NCCL's NVLS
user-buffer registration path; not exercised by this workload but required
for engine workloads that register buffers during graph capture).

## Stock-NCCL under gVisor with the identity oracle + UC-import replay

With the fdinfo oracle (above) and UC-import suspend/replay implemented
(`do_suspend`/`do_resume` now handle `KIND_IMP` alongside `KIND_MC`;
`run_nccl_mcshim_gvisor.sh` is the acceptance runner, with `NCCL_NVLS_ENABLE`
and `GRAPH` bisection knobs), stock 4-rank NCCL under gVisor now gets **much
further** but is **not yet passing**:

| stage | result |
|-------|--------|
| boot, NVLS engaged, stock NCCL 2.30.7 | ok |
| shim suspend (all ranks) | ok: `groups=1 imports=48 unmapped=49 unbound=1 released=49` |
| `runsc checkpoint` | **rc=0** |
| `runsc restore` | **rc=0** |
| shim resume (all ranks) | ok: `groups=1 imports=48 served=48 rebound=1 remapped=49`, every rank |
| post-restore collective | **FAIL: one (varying) rank faults `CUDA_ERROR_LAUNCH_FAILED` (719)** |

This is a large step past the prior wall (checkpoint hung / restore refused):
the **entire mechanical pipeline works** for stock NCCL — suspend, checkpoint,
restore, and a fully-successful resume that re-establishes every multicast
group and all 48 P2P imports/exports per rank at identical VAs.

Three bugs were found and fixed getting here, each verified by the run
advancing past it:
1. **ABI-aware `cuGetProcAddress` redirection** (`gpa_redirect`): stock NCCL
   resolves `cuGetProcAddress` *through* `cuGetProcAddress_v2`; returning the
   4-arg v1 wrapper for a v2 request left `symbolStatus` unwritten and
   crashed `ncclCommInitRank` (ip=0 SIGSEGV).
2. **UC exporters must record the rendezvous key too** (not just multicast):
   `cuMemExportToShareableHandle` now records for `KIND_UC` as well, so P2P
   peer buffers are re-exported+served on resume (else importers time out).
3. **Fresh-context warmup** (resume phase 0): the first VMM/IPC call from the
   shim's control thread on a freshly cuda-checkpoint-restored context returns
   `CUDA_ERROR_UNKNOWN` (999); a per-context `cuCtxSynchronize` first clears
   it. (Rank 0 was masked because its first op is `cuMulticastCreate`.)

### The remaining fault (open)

Symptom: after a fully-successful resume, exactly one rank (which rank varies
run-to-run) faults `719` on its first post-restore collective; NCCL itself
logs no error (`NCCL_DEBUG=WARN` is silent), so from NCCL's view all VAs and
structs are consistent — the GPU access itself faults.

Ruled out by bisection (`run_nccl_mcshim_gvisor.sh`):
- **Not multicast/NVLS**: reproduces with `NCCL_NVLS_ENABLE=0` (`groups=0`,
  pure P2P UC imports).
- **Not CUDA graphs**: reproduces with `GRAPH=0` (plain eager allreduce).
- **Not the 304 import-retry path**: reproduces with zero retries logged.
- **Not VA/reservation overlap**: all 192 imports re-map on
  `retained-reservation` (gVisor preserves the empty reservations exactly);
  none fall back to fixed re-reserve.
- **Not access grants** (as first guessed): always re-granting RW for the
  mapping's device on remap did not help, and no `count>1`
  `cuMemSetAccess` was observed.

### The mechanism is sound — proven by a bare probe (`p2p_reexport_probe.py`)

To decide whether `719` is a driver/gVisor limitation or a shim bug, a
minimal probe reproduces the *mechanism* with **no NCCL and no shim**: two
processes (one GPU each) create VMM POSIX-FD buffers, export/import them over
a socketpair, and read/write every peer buffer with a **real PTX SM kernel
over NVLink** (loaded via `cuModuleLoadData`, no nvcc). Then it mirrors the
shim flow: release each import (unmap keeping the reservation + `cuMemRelease`)
→ checkpoint/restore → re-export → re-import → re-map at the IDENTICAL VA →
re-run the peer kernel.

**Every configuration PASSES** (`baseline=OK`, `post=OK`):

| dimension | native | gVisor |
|-----------|--------|--------|
| 1 buffer, unidirectional, read | PASS | PASS |
| 48 buffers, **bidirectional** (each rank exporter+importer) | PASS | PASS |
| suspend/resume on a **background control thread** (`P2P_THREADED=1`, exactly like the shim) | PASS | PASS |
| kernel **writes** to peer memory (not just reads) | PASS | PASS |

So peer NVLink access to a re-imported allocation **does** survive
cuda-checkpoint and gVisor C/R, bidirectionally, at NCCL's buffer count, from
a control thread, for reads and writes. The `719` is therefore **not** a
driver, gVisor, cuda-checkpoint, threading, or fundamental-mechanism
limitation.

### So the remaining fault is shim replay fidelity vs real NCCL

Ruled out further: the shim's per-import layout matches the probe exactly —
`IMPORT-LAYOUT` diagnostics show **every NCCL import is a single mapping at
offset 0** (no shared-reservation offsets, no multi-mapping). So the tracked
set is structurally simple and correct-looking, yet one random rank still
faults.

### Root cause localized: one NCCL RM control page is not restored

Dumping `/proc/self/maps` (GPU/UVM ranges) in each rank immediately before the
shim suspend and again after resume gives a clean, decisive signal:

| run | GPU maps pre → post | `r--s /dev/nvidiactl` pages | collective |
|-----|--------------------|----------------------------|-----------|
| passing probe, 48 buffers | 39 → 39 | 2 → 2 | OK |
| passing probe, **256** buffers | 39 → 39 | 2 → 2 | OK |
| **stock NCCL, WORLD=4** | **46 → 45** | **5 → 4** | **719 (one rank)** |

Across the shim's suspend/checkpoint/restore/resume, stock NCCL loses exactly
**one 4 KiB read-only-shared `/dev/nvidiactl` control page** on every rank
(the first page of a contiguous 4-page block) — and the passing probe loses
none, even at 256 buffers. So:

- It is **not** a cuda-checkpoint scaling issue (256-buffer probe is stable).
- It is **not** the shim's VMM replay: that page is **not** device memory the
  shim manages (not a `cuMemCreate`/import/multicast VA). All of the shim's
  tracked ranges (`-w-s /dev/nvidia0` device memory + import VAs) round-trip
  intact; every import re-maps at its identical VA.
- It **is** an NCCL-specific RM control object: NCCL has 5 such `r--s` control
  pages (from objects the probe never creates — events, proxy/FIFO, semaphores,
  notifiers, …); one is not re-established after restore, and its loss
  correlates 1:1 with the `719`.

This strongly implies a **cuda-checkpoint (or nvproxy replay) limitation for a
specific NCCL RM control object**, surfaced only now that the shim cleared the
import blocker — the same class of cuda-checkpoint gap the original native A/B
hit (`PROGRESS.md`), one layer deeper. It is *not* a defect in the shim's
suspend/replay logic.

### ATTRIBUTED (2026-08-10): a cuda-checkpoint gap in `ncclCommInitRank`

Phase-tagged maps snapshots pin the page's origin, and a minimal repro pins
the layer:

- **Origin — `ncclCommInitRank`.** The `r--s /dev/nvidiactl` page count per
  rank goes 4 (after ctx+stream+alloc) → **5 after `ncclCommInitRank`** → 5
  through warmup → **4 after restore**. The exact page added by
  `ncclCommInitRank` is byte-for-byte the one lost on restore, identical on
  all 4 ranks.
- **Layer — cuda-checkpoint, not gVisor/nvproxy, not the shim.** Two
  controls:
  - Native (no gVisor): reading `/proc/<rank>/maps` from the host around a
    bare `cuda-checkpoint` lock/checkpoint/restore/unlock cycle shows the
    same drop (9 → 8 per rank; native /proc shows more host mappings, hence
    9 not 5, but the delta is the same −1).
  - **Minimal shim-free, gVisor-free, single-GPU repro**
    (`nccl_commninit_page_probe.py`): one process, WORLD=1 NCCL comm (no
    peers, no imports, no multicast, no mcshim), bare `cuda-checkpoint`
    cycle → **9 → 8**. `VERDICT: cuda-checkpoint DROPS the ncclCommInitRank
    control page`.

So the post-restore `719` is rooted in **cuda-checkpoint failing to
re-establish an RM control mapping that `ncclCommInitRank` creates** —
entirely upstream of gVisor, nvproxy, and this project's shim. It is the same
*class* of cuda-checkpoint limitation the original native A/B hit
(`PROGRESS.md`), now precisely localized to a single control page. (The page
drops even for a single-GPU comm, where it is harmless because no collective
kernel dereferences the state behind it; in multi-GPU it manifests as the
one-rank `719`.)

### The page is an `NV_ESC_RM_MAP_MEMORY` mapping nvproxy can replay

A temporary debug log in nvproxy `rmMapMemory` (reverted after use) named the
mappings created during `ncclCommInitRank`: they are `NV_ESC_RM_MAP_MEMORY`
calls on RM memory objects (classes `0x3e` `NV01_MEMORY_SYSTEM`, `0x40`,
`0xde`, `0xc661`), and **every one reports `found=true`** — i.e. the mapped
object is in nvproxy's live object graph. The lost `r--s` page is one of the
small (`len=0x1000`) read-only control mappings among these.

Crucially, gVisor's own `frontend_mmap.go` documents the gap directly:

> `mmapLength ... state:"nosave"` — "we do not automatically reinvoke
> `NV_ESC_RM_MAP_MEMORY` after restore, so restored FDs have no
> mmap_context."

So under gVisor, after restore the guest VA exists but its nvproxy
mmap_context is gone, and nothing re-issues the map — matching the observed
page loss.

### The nvproxy workaround was attempted and does NOT apply (measured)

An nvproxy-side fix was implemented and tested: record `NV_ESC_RM_MAP_MEMORY`
params (savable) and replay the ioctl in `afterLoad` to re-establish the
mmap_context. **It is a no-op for this flow**, for a decisive reason:

> At restore, `nvproxy.afterLoad` logged **`0/0 frontend FDs`** and **0
> restored objects**.

With the cuda-checkpoint **job** integration, `control/state_cuda.go`
`preSaveCuda` runs `cuda-checkpoint` *before* the gVisor save. That releases
the process's GPU FDs and RM objects, so at gVisor-save time nvproxy holds
**zero** GPU state; on restore, `cuda-checkpoint` (running in-sandbox)
re-creates 100% of the GPU state itself — the FDs and mappings are re-opened
*after* `afterLoad` runs. nvproxy's save/restore machinery (object graph,
mmap replay) is therefore entirely **out of the GPU-state restore path** when
cuda-checkpoint job mode is used, and it cannot re-establish a mapping
cuda-checkpoint dropped (the app VMA itself is gone — the workload's own
`/proc/self/maps` shows the page absent post-restore). The nvproxy change was
reverted; the `nosave` mmap_context gap it targets is real but only matters
for a pure-nvproxy restore path that GPU C/R does not use (cuda-checkpoint is
always in the loop for live CUDA).

### The fix must come from NVIDIA (cuda-checkpoint)

This is now conclusive: **cuda-checkpoint fails to re-establish, on restore,
an RM control mapping created by `ncclCommInitRank`**, and no gVisor/nvproxy
layer can compensate because cuda-checkpoint owns the entire GPU-state
teardown/rebuild in job mode. `nccl_commninit_page_probe.py` is the clean,
~90-line, dependency-light repro to file (stock NCCL + cuda-checkpoint only —
no gVisor, no fork, no shim: a bare cuda-checkpoint cycle drops the page
9→8).

The only conceivable in-sandbox workaround would be for gVisor to *observe*
cuda-checkpoint's restore ioctls and inject the missing `NV_ESC_RM_MAP_MEMORY`
— but it would also have to synthesize the app-side `mmap(2)`/VMA that
cuda-checkpoint dropped, which is above nvproxy's layer and fragile. Not
recommended; escalate to NVIDIA.

Diagnostic aid (reverted, re-add if needed): a one-line `fi.ctx.Debugf` in
`rmMapMemory` logging `hClient/hMemory/class/found/len/flags` names every
mapped object at init; `restoreMmap`'s `%d/%d frontend FDs` log revealed the
0/0 above.

The probes remain the reusable oracles: `p2p_reexport_probe.py` proves the
shim's memory mechanism sound (all configs PASS to 256 buffers, read+write
over NVLink, control-thread); `nccl_commninit_page_probe.py` isolates the
cuda-checkpoint control-page gap with no shim in the picture.

## Scope and next steps

Done: single-process suspend/replay core; multi-rank rendezvous +
export/import replay for multicast (WORLD=2/4, native + gVisor, PASS);
fdinfo identity oracle (sentry, PASS); UC-import suspend/replay implemented
and mechanically complete for stock NCCL under gVisor (checkpoint+restore+
resume all succeed); one open post-restore peer-access fault (above).

Remaining, in de-risking order:

0. **MEASURED (2026-08-08) — the identity collision exists under gVisor
   too; an oracle is REQUIRED.** `fd_identity_probe.py` /
   `run_fd_identity_gvisor.sh`: two MC groups + one UC alloc exported —
   native keys all `0x7:0x55e`, gVisor keys all `0x14:0x12`. The probe also
   corrected the root-cause theory: export fds are not anon inodes, they are
   **opens of `/dev/nvidiactl`** (`readlink /proc/self/fd/N`), so every
   export fd shares the device *node's* inode in any environment. `kcmp(2)`
   (the only oracle-free open-file-description identity) is unimplemented in
   gVisor (`linux64.go` CapError) and has the wrong access pattern anyway
   (needs both fds alive simultaneously).
1. **UC-import suspend/replay**: extend the KIND_IMP path (never-AddDevice'd
   imports) with the same unmap-keeping-reservation → release → re-import →
   re-map-at-identical-VA cycle the MC path already implements. Exporter
   side: on resume, re-export each still-imported-by-peers UC handle and
   serve it (same socket machinery; UC handles survive restore, so only
   importers need replay). Unblocks stock NCCL NVLS and torch
   `_symmetric_memory` (same import topology).
2. **DONE (2026-08-08) — identity oracle via fdinfo.** Implemented as a
   `show_fdinfo`-style extension (Linux precedent: dmabuf):
   - `pkg/sentry/fsimpl/proc/task_fds.go`: optional `procFDInfoExtra`
     interface (duck-typed, so proc does not import nvproxy); `Generate`
     appends its lines.
   - `pkg/sentry/devices/nvproxy/fabric.go`: `frontendFD.ProcFDInfoExtra`
     emits `nvproxy_exported_object:\tclient=0x… object=0x… class=0x…` from
     the `exportedObj` recorded at export time. No import-side handling
     needed anywhere: SCM_RIGHTS passes the same FileDescription, so
     importers read the identical line.
   - `mcshim.c` `record_key`: prefers the oracle (`key = client:object`,
     globally unique — RM client handles are host-global), falls back to
     fstat natively.
   Verified: probe shows 3/3 distinct keys under gVisor + cross-process
   SCM_RIGHTS match (`oracle=[client=… object=0x5c0000a8/a9/aa]`);
   `run_mcshim_mp_gvisor.sh` WORLD=2/4 PASS on oracle keys
   (`key=c1d02d05:5c0000a8`); native fallback PASS unchanged (`key=7:55e`);
   `proc_test` + `nvproxy_test` pass. The fdinfo `class` field also lets the
   shim classify imports (0xfd multicast vs 0x40 memory) without waiting for
   an AddDevice — useful for item 1. Native remains ordinal-only
   (CONTROL-leg duty; acceptance lives under gVisor).
3. **gVisor integration**: inject `LD_PRELOAD`/`MCSHIM_*` next to
   `setupCudaCheckpointJob` in `runsc/boot/loader.go`; drive the marker edge
   from `state_cuda.go` around the cuda-checkpoint phases; assert
   `CheckpointBlockers()` is empty after suspend instead of trusting acks.
4. **Acceptance**: a gVisor variant of `run_nccl_mcshim_native.sh` (same
   flow as `run_mcshim_mp_gvisor.sh`, stock NCCL lib staged into
   `/opt/phase0`). Per measurement 0, scalable identity is gVisor-only, so
   the native runner's MAIN leg is retired to CONTROL duty and acceptance
   lives under gVisor (the production environment). Then the torch
   `symmem_nccl_ckpt.py` harness (repo root) on 4–8 GPUs is the full-stack
   pass criterion from TASK.md.

### Known limitations (current shim)

- Rendezvous identity is fd `st_dev:st_ino` + per-key creation ordinal:
  sufficient for few groups with deterministic setup order; insufficient for
  many interleaved imports natively (see above).
- `MAX_AKA=8` old handles per group: stale-handle translation degrades after
  8 suspend/resume cycles in one process lifetime.
- Fixed tables (`MAXN=512` allocs/binds/maps per process); bind-table
  exhaustion logs a warning and stops tracking (loud in the log, not fatal).
- `cuMemSetAccess` records only the first access descriptor (count>1 not
  replayed).
- Suspend requires the app quiesced (step (a)); the shim does not fence
  concurrent CUDA calls itself.
