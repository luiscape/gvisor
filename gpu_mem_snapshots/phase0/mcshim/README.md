# mcshim — generic libcuda-level multicast checkpoint interposer (Idea D)

**PASS native + PASS under gVisor**, single-process (2 GPUs) **and**
multi-process — one process per GPU, the vLLM/SGLang TP topology — at
WORLD=2 and WORLD=4 (driver 610.57.04, H100 NVSwitch).

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
  measurement: are exported-object fd identities distinct? (Answer: no,
  neither natively nor under gVisor — they are all opens of
  `/dev/nvidiactl`.)

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

## Scope and next steps

Done: single-process suspend/replay core; multi-rank rendezvous +
export/import replay for multicast (WORLD=2/4, native + gVisor, PASS);
stock-NCCL multicast suspend verified; UC-import gap isolated to a minimal
repro with a proven-sufficient remediation shape.

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
