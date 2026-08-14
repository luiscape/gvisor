# What changed in gVisor, and how the interposer works

State of branch `luis/experiment-nccl-state` vs upstream (`05b57c92a`).
Sentry/runsc delta: ~2,850 lines across 28 files. The interposer (`mcshim.c`)
is ~2,700 lines of standalone C, LD_PRELOADed into the container, not part of
the sentry.

The one-sentence architecture: **gVisor orchestrates, the interposer
executes.** The sentry decides *when* GPU shared-memory state must be torn
down and rebuilt (around the cuda-checkpoint phases, while tasks are frozen);
the interposer is the only component that can actually do it, because the
teardown and rebuild must be issued *through libcuda inside the application's
own process* so that libcuda's userspace bookkeeping stays consistent.

---

## 1. gVisor changes, by layer

### runsc (flags, loader)

- `--cuda-checkpoint-path=<bin>` — wraps the container command in
  `cuda-checkpoint --launch-job`, so every CUDA process in the container
  shares one checkpoint *job*. Job mode is required: it is what makes the
  driver's own handling of legacy CUDA IPC work at all.
- `--cuda-multicast-shim-path=<lib>` — LD_PRELOADs the interposer into the
  container by appending to the initial process's `LD_PRELOAD`
  (`runsc/boot/loader.go: setupCudaMulticastShim`). Known limitation: a
  launcher that rewrites `LD_PRELOAD` for its children (SGLang's
  torch_memory_saver) evicts it; the benches work around this with
  `/etc/ld.so.preload`, and moving the sentry's injection there is a candidate
  change.
- `runsc checkpoint --cuda-checkpoint-sequential` — toggle job members one at
  a time on restore.

### `pkg/sentry/control/state_cuda.go` — the orchestrator

Runs during save/restore, with application tasks frozen. Checkpoint sequence:

1. Enumerate CUDA processes (thread groups holding nvproxy device FDs,
   filtered via `cuda-checkpoint --get-state`).
2. **Blocker gate**: poll `nvproxy.CheckpointBlockers()` until no resource
   cuda-checkpoint cannot serialize remains live (fabric/multicast objects,
   exported-object FDs), with per-client attribution in the failure message.
   Multicast blockers are exempt when the interposer will release them
   mid-sequence.
3. **Gate + lock, retried**: arm the interposer's gate (bars the app from the
   GPU — including captured-graph replays, which nothing else can stop), then
   `cuda-checkpoint --action lock` on all processes in parallel (coupled
   ranks only quiesce together). Gating can deadlock a mid-flight collective,
   so on lock timeout the gate is released to let it drain, then retried.
4. **Interposer teardown, sandwiched inside the lock**: unlock (a locked
   process cannot execute CUDA calls), tell the interposer to suspend, verify
   with the blocker gate that nothing it claimed to release is still live,
   re-lock.
5. `cuda-checkpoint --action checkpoint` (GPU memory drains to 0), then the
   normal sentry save.

Restore sequence (`postResumeCuda`): sentry load → `cuda-checkpoint --toggle`
per process → interposer resume (rebuild) → un-gate. Ordering is
load-bearing: the rebuild must run after *every* toggle (a rank rebuilding on
a half-restored peer latches sticky context errors) and after `SigsegvUnlock`
(the rebuild faults pages).

### `pkg/sentry/control/state_cuda_shim.go` — driving the interposer

The sentry↔interposer contract is an existence-based marker protocol in a
shared directory (chosen because it needs no channel into a frozen process):
sentry creates `gate` / `suspend`, each interposer-managed process acks with
`gated.<pid>` / `suspended.<pid>` (or `error.<pid>`); removal triggers resume,
acked with `resumed.<pid>`. Only processes that announced `present.<pid>` are
waited on — an engine's API server holds GPU FDs but never touches multicast,
and must not be waited on forever.

### `pkg/sentry/devices/nvproxy` — visibility, not mechanism

- **`fabric.go` — the blocker inventory** (the part of TASK.md that shipped
  and stayed): walks the live object graph and reports objects
  cuda-checkpoint cannot serialize (classes 00F8/00FD/00FB, exported-object
  FDs), attributable per client/rank.
- **The fdinfo oracle**: `/proc/self/fdinfo/<fd>` on an nvproxy-exported FD
  shows `nvproxy_exported_object: client=... object=...` — a globally unique
  identity the interposer uses to key its cross-rank rendezvous under gVisor.
  (Natively all export FDs share one inode; the fallback ordinal scheme only
  disambiguates simple topologies.)
- Driver 610.57.04 support, a handful of new-in-610 ioctls
  (`NV2080_CTRL_CMD_FLA_GET_FABRIC_MEM_STATS` was required for TP workers to
  reach checkpoint at all).
- **`multicast.go` / `fabric_unsafe.go` — kept as a negative result.**
  Sentry-side multicast suspend/replay is *disproven*: freeing the RM objects
  from nvproxy makes the save succeed but the restore toggle refuses, because
  libcuda's own userspace bookkeeping still lists them. This is the
  measurement that dictates the entire architecture — teardown must happen
  through libcuda, in-process, which is why the interposer exists.

### `pkg/sentry/mm` — `MAP_FIXED_NOREPLACE` (cherry-pick of PR #14008)

Correct Linux semantics (exact placement or `EEXIST`) instead of treating the
address as a hint. The NVIDIA driver's checkpoint path depends on `PROT_NONE`
placeholders over released ranges. Tested against the residual live-import
toggle intermittency: **not** the cause (1/5 with the fix), but correct and
kept.

---

## 2. How the interposer (`mcshim.c`) works

### Interposition

Wraps ~40 libcuda entry points via three capture routes (direct link,
`dlsym`, and `cuGetProcAddress`/`_v2` — CUDA 12 runtimes resolve through the
latter, with ABI-version-aware handling). Wrapped calls fall into three
groups: **tracking** (allocation/mapping/IPC lifecycle), **translation**
(handles change across a rebuild; stale handles the app retained are rewritten
via an alias table), and **gating** (kernel launches, graph launches, memcpys
block while the gate marker is set).

### Tracked state: a live object graph, not a log

Four tables, mirroring TASK.md's invariant for nvproxy's graph — app-initiated
frees drop entries out of the replay set automatically:

- `Alloc` — VMM allocations & multicast groups: kind (UC / MC / imported),
  size, properties, device list, rendezvous key, handle alias history.
- `Mapping` — every `cuMemMap`: VA, size, offset, access descriptors.
- `Bind` — every `cuMulticastBind{Mem,Addr}`: group, device, offsets.
- `IpcEnt` — legacy CUDA IPC: exporter ptr or importer VA, the **original**
  64-byte blob (the only identity that survives a restore — re-exported blobs
  differ), open order, held VA reservation.

### Suspend (between lock and checkpoint)

1. Multicast groups: unmap VAs (**retaining** the `cuMemAddressReserve`
   reservations), unbind each device, release the 00FD handle.
2. VMM imports: unmap views, release imported handles. One live import is
   sufficient to break restore (measured natively, deterministic); backing
   memory is the exporter's and stays resident.
3. Legacy IPC imports, classified by `MCSHIM_IPC_REPLAY_FLOOR` (default
   4 TB): **high-region** imports (dedicated mappings) are closed and their
   exact ranges held with reservations across the checkpoint; **low-region**
   ones (suballocated into driver-owned VA where user reservations are not
   honored and even a same-blob reopen in the same live process relocates)
   are left live for the driver's job-mode support to carry.

Nothing the application owns is modified; after resume it observes identical
handles and identical addresses.

### Resume (after every toggle)

1. **Serve**: each exporter recreates its object (`cuMulticastCreate` /
   re-export) and serves the new FD (SCM_RIGHTS) or new IPC blob (64 bytes)
   on a unix socket named by the object's stable identity — fdinfo oracle
   under gVisor, original-blob hash for legacy IPC. All serving starts before
   any fetching (a rank is usually both exporter and importer; ordering is
   the deadlock avoidance).
2. **Fetch + reimport**: importers fetch and re-import; new handles are
   pushed onto the alias table so stale app-held handles keep translating.
3. **Rebind + remap**: multicast `AddDevice` + binds replayed (binds are the
   cross-rank barrier), then every mapping remapped **into its retained
   reservation** — identical VA by construction, verified.
4. **Legacy IPC reopen, in original open order**: release the held
   reservation immediately before each reopen (never earlier). Since
   `cuIpcOpenMemHandle` takes no address hint but allocates lowest-free,
   a wrong landing is walked home: close, reserve exactly the hole it fell
   into, reopen; plugs persist until every import is placed, then are freed.
   Every reopened VA is compared to the recorded one; any mismatch fails the
   resume loudly — the app holds the old pointers (including inside captured
   CUDA graphs), so a moved import is silent corruption, not an error.

### Why this division of labor

- The **driver** (R610 job mode) carries: process memory, contexts, streams,
  captured graph state, resident allocations, and live *legacy* IPC imports
  (intermittently — the one unattributed reliability gap).
- The **interposer** carries: everything the driver refuses or breaks on —
  multicast/NVLS groups, VMM IPC imports, replayable legacy IPC imports.
- The **sentry** carries: sequencing, quiesce, verification gates, and the
  sandbox snapshot itself.

The invariant the whole design serves: **after restore, every handle and every
GPU virtual address the application can observe is identical**, so CRIU-style
state restoration and cuda-checkpoint's device restore compose without the
application ever knowing.
