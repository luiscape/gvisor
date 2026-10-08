# mcshim: CUDA multicast suspend/resume interposer

`mcshim.so` is an `LD_PRELOAD` interposer that gVisor injects into CUDA
processes so that `cuda-checkpoint` (NVIDIA driver R610+)
can checkpoint and restore workloads it otherwise refuses:

*   **Multicast (NVLS).** Processes holding live `NV_MEMORY_MULTICAST_FABRIC`
    (0x00fd) objects -- created by NCCL NVLS and torch `_symmetric_memory` --
    cannot be checkpointed. The shim tracks every multicast group at the
    libcuda layer, releases it before the checkpoint, and rebuilds it at
    byte-identical virtual addresses afterwards, so application pointers and
    captured CUDA graphs stay valid.
*   **VMM imports.** Live `cuMemImportFromShareableHandle` imports (NCCL P2P
    buffers) cannot be restored either; the shim releases and re-imports them
    the same way, re-fetching the re-exported fd from the exporting rank.

Legacy CUDA IPC (`cuIpcGetMemHandle` / `cuIpcOpenMemHandle`, used by the
engines' custom all-reduce) is not the shim's concern: `cuda-checkpoint`
carries it when the processes share a job, which runsc arranges whenever the
interposer is enabled. runsc therefore rejects the interposer flags without
`--cuda-checkpoint-path`. Without the interposer, CUDA checkpoints are
unchanged: one `cuda-checkpoint --toggle` per process, no job.

Build with `./build.sh` (toolkit-free; runs in a pinned ubuntu:22.04 container
by default so the result loads under older glibc). The Bazel target
`//tools/mcshim:mcshim` builds the same artifact, and `//runsc` embeds it
(through `//runsc/mcshimbin`) so that a stock runsc binary can inject the
interposer into containers whose images do not carry it
(`--cuda-multicast-shim-source=EMBEDDED`).

## How it gets into a container

With `runsc --cuda-checkpoint-path=... --cuda-multicast-shim-path=/path/to/mcshim.so`
(plus nvproxy and an R610+ driver) -- or with
`--cuda-multicast-shim-source=EMBEDDED`, in which case runsc first writes its
embedded copy of `mcshim.so` into the container filesystem at that path
(default `/usr/local/lib/mcshim.so`) -- `Loader.setupCudaMulticastShim`
(`runsc/boot/loader.go`) prepends the shim to the container's `LD_PRELOAD`
**and** appends it to `/etc/ld.so.preload` through the container's VFS
(launchers like SGLang's `torch_memory_saver` rewrite `LD_PRELOAD` for exactly
the worker processes that matter; `ld.so.preload` is immune). In IMAGE mode,
a path that is not a file in the container is not preloaded at all, since the
dynamic loader would print an error on every exec. `Loader.setupCudaCheckpointJob`
wraps the container's command in `cuda-checkpoint --launch-job`.

The shim and the sentry (`pkg/sentry/control/state_cuda_shim.go`)
rendezvous in `/tmp/mcshim`, which must be part of the checkpoint image (not
a host mount). The sentry drives the shim in every process that announced
itself there (`present.<pid>`).

The shim is inert until a process calls `cuInit`; only then does it start its
control thread and participate in the protocol. Helpers that merely inherit
the preload (shells, `cuda-checkpoint` itself) never ack anything.

## Control protocol

Everything goes through `/tmp/mcshim`. Markers are **existence-based and
edge-triggered**: the shim reacts to a marker appearing or disappearing, never
to its content, which keeps the protocol race-free for any number of rank
processes sharing one directory.

| File                  | Written by | Meaning                                                        |
| :-------------------- | :--------- | :------------------------------------------------------------- |
| `gate`                | sentry     | created: arm the gate, wait for tracked calls in flight, check the state can be carried (else `error.<pid>` and disarm); removed: release the application, once this process is not torn down (refused after a failed transition) |
| `suspend`             | sentry     | created: tear down tracked state; removed: rebuild it          |
| `present.<pid>`       | shim       | this pid runs a control thread and will ack transitions        |
| `gated.<pid>`         | shim       | gate armed and drained                                         |
| `suspended.<pid>` / `resumed.<pid>` | shim | teardown / rebuild finished                      |
| `error.<pid>`         | shim       | the transition in flight failed (sentry fails fast on this)    |

The sentry waits up to 5 minutes for acks, and removes stale acks of the kind
it waits for before each request. On startup the control thread removes any
stale acks a dead predecessor with the same (reused) pid left behind, then
writes `present.<pid>`.

The markers live in the container filesystem, so they are **part of the
checkpoint image**: after a restore they still exist, and the shim stays
suspended until the sentry removes `suspend` to trigger the rebuild, and gated
until the sentry removes `gate`.

## The sentry's sequence

From `pkg/sentry/control/state_cuda.go` / `state_cuda_shim.go`:

0.  refuse if a process that does not run the shim holds multicast objects or
    imported memory, or if an earlier attempt's teardown failed (its markers
    remain),
1.  create `gate`, wait for `gated.<pid>`. Arming waits (up to 30 s) for
    application threads to leave tracked calls, so it must precede the lock;
    a refusal here fails the checkpoint before anything is torn down. Then,
    with every process gated, refuse if an import's exported object no
    longer exists in a process that runs the shim (nvproxy records each
    import's source object): nobody would re-export it after a restore,
2.  `cuda-checkpoint --action lock` on all ranks in parallel (drains GPU
    work; gating mid-collective can starve peers, which fails the lock and
    with it the checkpoint),
3.  `--action unlock` (the teardown must issue CUDA calls, which a locked
    process cannot),
4.  create `suspend`, wait for `suspended.<pid>`,
5.  verify the checkpoint-blocker set is empty (trust but verify),
6.  re-lock, `--action checkpoint` (one process at a time in job mode), save
    the sandbox;
7.  on restore: `--toggle`, or `--action restore --device-map` then
    `--action unlock` when the GPUs changed,
8.  remove `suspend`, wait for `resumed.<pid>`,
9.  remove `gate`, which releases the application. Not earlier: a bind waits
    for every device to be added, not for every rank's memory to be bound, so
    a rank released at its own resume could reach a group a peer is still
    binding.

Every `cuda-checkpoint` invocation is bounded by `--cuda-checkpoint-timeout`
(10 minutes by default): one still running is killed, and the checkpoint or
restore fails.

## Environment variables

| Variable                 | Default         | Effect                                                            |
| :----------------------- | :-------------- | :---------------------------------------------------------------- |
| `MCSHIM_LOG`             | `/tmp/mcshim/mcshim.log` | append the log to this path; `stderr` for stderr. The shim never prints otherwise: every process in the container loads it |
| `MCSHIM_DISABLE`         | unset           | silent: no control thread, acks, or gate (interposition/tracking stay active) |
| `MCSHIM_ALLOW_FABRIC`    | unset           | keep fabric handle types; the gate refuses while a fabric-capable allocation is alive (see below) |
| `MCSHIM_HOST_BUILD`      | unset           | build.sh: build with the host toolchain instead of docker         |
| `MCSHIM_BUILD_IMAGE`     | pinned 22.04    | build.sh: alternative base image                                  |

## Restoring onto different GPUs

The interposer needs nothing special. runsc creates `/dev/nvidia#` files for
the GPUs the restored container is given, rebinds open device FDs to them,
and passes `cuda-checkpoint --device-map` so CUDA state is restored onto the
new devices. Device ordinals inside the sandbox do not change, so the
interposer's rebuild runs against the same ordinals it recorded.

## Design summary

*   **Lookup redirection.** torch, NCCL and ctypes resolve driver entry
    points through `dlsym`, `cuGetProcAddress` or cudart's
    `cudaGetDriverEntryPoint*`, bypassing symbol interposition. The shim
    interposes those resolvers and redirects by the **address** the lookup
    returned: that identifies the exact exported symbol, and so its ABI (for
    example `cuMulticastBindMem` resolves to the 7-argument
    `cuMulticastBindMem_v2` at CUDA 13.1+, and PTDS lookups to `_ptsz` /
    `_ptds`). Every exported variant has a wrapper with its own prototype. A
    lookup of a tracked entry point, or of a resolver, that returns a symbol
    the shim has no wrapper for (a future `_v3`) refuses checkpoints. A cudart
    resolver answers with the ABI of its own runtime, and libraries can load
    different runtimes `RTLD_LOCAL`, so each call goes to the runtime the
    caller's own dependencies resolve the resolver to, and a resolver found
    by `dlsym` stays bound to that handle's runtime. A caller the shim cannot
    place (e.g. through libffi) gets the only `libcudart` loaded; with
    several loaded, checkpoints are refused.
*   **Tracking tables.** Fixed-size (`MAXN` = 4096 each): objects (`g_alloc`),
    mappings (`g_map`), multicast binds (`g_bind`). This is live state: an
    object is forgotten once it has no application reference, mapping or
    bind. Slots fill first-free, and scans stop at each table's high-water
    mark. Any overflow is loud and sticky (`g_untracked`): arming the gate
    refuses thereafter, failing the checkpoint before anything is torn down.
*   **Handles.** The application only ever sees the handle values it was
    given. A rebuild gives objects new driver handles, and every
    handle-taking entry point translates (`xlate`), including
    `cuMemRetainAllocationHandle`'s result. The driver reuses values a
    rebuild freed, so a new object whose driver handle equals a live object's
    application value is given a synthetic value instead.
*   **References.** The shim counts the application's references (1 per
    create or import, +1 per retain, -1 per release). Suspend drops them;
    resume holds one while it rebuilds, then returns the count to the
    application's (releasing it if the application held none, or retaining
    more from a mapping).
*   **Contexts.** Calls are issued from each recorded device's primary
    context, retained for the duration of a transition. The shim never
    retains a primary context the application does not hold (that would
    create one, which fails in exclusive-process mode), so the gate refuses
    if one is inactive (a device only added to a group needs none: the
    group's own device re-adds it). Freed exports are copied through a device
    that has read-write access to the mapping.
*   **Binds.** A bind by address over tracked memory is recorded as a bind of
    that allocation at the matching offset, so its replay depends neither on
    the VA nor on the order of remaps; only memory the shim does not track,
    which stays resident, is rebound by address. A v1 bind applies to the
    device holding the memory, whatever the current device; for untracked
    memory the shim asks the driver. The driver refuses to bind imported
    memory (measured on R610).
*   **Locking.** No call that can wait on the GPU or on a peer runs under
    the shim's lock: binds and multicast maps wait for every device to join,
    and `cuMemUnmap` and `cuMemSetAccess` can wait for GPU work, which may
    wait for a host thread that needs the lock. These look up under the lock,
    call unlocked, and commit only if the objects they looked up are
    unchanged (each table slot has a generation, and lookups skip an object
    being unmapped, whose handle the driver may already be reissuing). Other
    mutators hold the lock across the real call.
*   **Identical-VA guarantee.** Suspend unmaps with `cuMemUnmap` only, never
    `cuMemAddressFree`, so the VA reservations survive the checkpoint; resume
    maps back into them and replays each mapping's access set, merged by
    location across `cuMemSetAccess` calls.
*   **Freed UC exporters.** A multicast-bound exporter allocation left
    resident across the checkpoint fails its next export after restore with
    OBJECT_NOT_FOUND (measured on R610: vLLM TP=4, torch `_symmetric_memory`).
    Suspend saves its contents into process memory (carried by the
    checkpoint) and releases it; resume recreates it, re-maps at the identical
    VAs, restores contents, and re-exports. Costs a device-host-device copy
    and checkpoint growth of the same size.
*   **Cross-rank resume.** (1) every exporter re-creates its object,
    re-exports it and publishes `<pid> <fd>` in `/tmp/mcshim` under the
    original export's identity (nvproxy's fdinfo line); (2) importers copy
    the fd with `pidfd_getfd(2)` and re-import (the identity is per object,
    so every export of an object shares it); (3a) every group gets its
    devices back, (3b) then every bind, which blocks until all devices have
    joined, (3c) then every mapping; (4) references are restored. Publishing
    before fetching, and adding devices before binding, keep ranks that hold
    objects in different orders from deadlocking. `pidfd_getfd` needs ptrace
    access, which YAMA denies between sibling processes, so an exporter sets
    `PR_SET_PTRACER_ANY` while it has fds published: until the sentry removes
    the `gate` marker after every process has resumed.
*   **The gate.** While armed, entry points that submit GPU work (launches,
    copies, memsets, stream memory operations, batches) and every tracked
    mutator block. Calls are counted from entry to return, and arming waits
    for the count to drain, so once `gated.<pid>` is written no application
    thread is inside a tracked call and none can complete one over the
    teardown. Stream synchronization blocks but is not counted: it may wait
    on a peer that is already gated. The shim's own teardown/rebuild calls the
    real entry points and is never gated. The release follows state, not
    marker edges: the application runs again once the `gate` marker is gone
    and the process is neither torn down nor broken, so a gate removed and
    re-created while it is torn down neither releases it early nor strands
    it. After a failed teardown or rebuild, the shim refuses to release it.
*   **Fabric handles.** Fabric handles are shared through IMEX, which the shim
    cannot rebuild after a restore. Unless `MCSHIM_ALLOW_FABRIC=1`, the shim
    strips them from `cuMemCreate` and `cuMulticastCreate`, refuses fabric
    exports and reports `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED` as
    0; on a single node POSIX fds are equivalent. With it, the gate refuses
    while a fabric-capable allocation is alive. Measured on R610:
    `cuda-checkpoint` checkpoints and restores one process's fabric-capable
    allocation, exported or not; creating one needs an IMEX channel
    (`NOT_PERMITTED` without).

## What refuses a checkpoint

The gate refuses, before anything is torn down, when a process has:

*   overflowed a tracking table, or resolved an unknown ABI of a tracked
    entry point;
*   exported or imported a memory pool (`cuMemPool{Export,Import}*`), created,
    imported or bound a logical endpoint, exported or imported a handle type
    other than POSIX fds, mapped a sparse array (`cuMemMapArrayAsync`), or
    allocated managed memory (which `cuda-checkpoint` cannot carry; frees are
    not tracked, so this refuses for the process's lifetime);
*   an export without nvproxy's identity (not running under nvproxy);
*   bound memory the shim does not track by handle, or bound one range by
    address across several mappings;
*   set access on part of a mapping, or on more than 16 locations;
*   several references to an object it must tear down but has not mapped,
    multicast-bound memory that is not mapped or has no read-write mapping, or
    such an object on an unknown device;
*   released a device's primary context while holding tracked state;
*   tracked calls still in flight after 30 s.

The sentry also refuses up front when a process that does not run the shim
holds multicast objects or imported memory, or when an earlier attempt's
teardown failed; and, once every process is gated, when an import's exported
object no longer exists in a process that runs the shim (its exporter freed
it or exited, or runs in another container).

## Testing

`test/run.sh [-r RUNSC] [-i ROOTFS -e ENVFILE] [test...]` runs the tests on a
host with two NVLS-capable GPUs (`MCSHIM_TEST_GPUS`, default `0,1`). Each test
but `orphan` plays the sentry's side of the protocol, with a suspend and
resume in place of a checkpoint and restore:

*   `abi`: every lookup returns the wrapper of the symbol the driver returned;
*   `gate`: submissions block; arming waits for a synchronous call in flight;
    the application stays gated after its own resume until the gate goes,
    and through a gate removed and re-created while torn down;
*   `mc`: v1 and v2 binds by handle and by address, including over memory
    the shim never saw; merged access; retained and colliding handles;
*   `refcount`: objects held only by mappings;
*   `refuse`: each refusal above that can be provoked, in a fresh process;
*   `mapwait`: multicast maps that wait for a peer's device do not deadlock
    two processes, and a map stuck in flight makes the gate refuse;
*   `ipc`: two processes importing exported groups in opposite order (under
    runsc, `-r`);
*   `orphan`: `runsc checkpoint` refuses an import whose exporter freed the
    object, and the application keeps running (under runsc, `-r`);
*   `torch-kernel`, `torch-symm`: the gate stops kernels that PyTorch submits,
    including a multimem all-reduce on symmetric memory across two ranks
    (under runsc, in a rootfs with PyTorch: `-i`, with its env file `-e`).

## Threat model

The control directory is **container-writable by design**; the shim runs
inside the container's trust domain, not gVisor's:

*   Any container process can forge markers or acks. The consequence is
    self-harm only: it can hang or fail *its own container's* checkpoint
    (e.g. suspending the app spuriously, or acking a teardown that did not
    happen and failing the checkpoint at the blocker check). It gains nothing
    it could not already do to the app directly, being the same trust domain.
*   The shim never trusts marker *content*, only existence, so nothing parses
    attacker-controlled bytes out of the control directory.
*   While fds are published after a restore, any process in the container
    may ptrace the exporters (`PR_SET_PTRACER_ANY`). Do **not** share
    `/tmp/mcshim` across jobs (e.g. through a volume): rendezvous keys could
    collide.
*   The sentry side (`state_cuda_shim.go`) treats acks as liveness signals
    with timeouts, never as data.

## Known limitations

*   **Quiescing is not a cut.** The gate stops submissions at slightly
    different times on each rank (the control thread polls every 5 ms), so a
    collective can straddle it; the lock then times out and the checkpoint
    fails cleanly. Checkpoint idle engines (sleeping or paused).
*   **One checkpoint per restore.** An object re-exported after a restore
    keeps its original rendezvous identity, so checkpointing a restored
    process whose importers re-imported a *new* export is not supported.
*   **Fork.** A child forked after `cuInit` cannot use CUDA, so the shim
    stays inactive in it; inherited published fds are closed.
*   **Unmapped export contents.** A multicast-bound export that is torn down
    is saved through its mappings, so contents outside every mapping are not
    preserved (the gate refuses one with no mapping at all).
*   **A failed teardown or rebuild leaves the application gated for good.**
    Deliberately: better blocked than corrupt. Peers may already have
    released state the failed process needs, so nothing is rolled back; the
    sentry fails the checkpoint and the workload must be restarted.
*   **Left to `cuda-checkpoint`:** in job mode it carries legacy IPC memory,
    IPC events and dma-buf exports (`cuMemGetHandleForAddressRange`); measured
    on R610. It cannot carry managed memory, which the gate refuses.
*   **Binds by address across mappings** are refused. On H100 the multicast
    and allocation granularities are both 2 MiB, so such a bind could be split
    per mapping; none of the tested engines makes one.
*   **Multicast slot contention on restore** manifests as a re-bind timeout
    (binds are the cross-rank barrier, so one rank failing to join blocks
    the rest until the deadline).
*   **`cuMemAddressReserve`/`cuMemAddressFree` are not interposed** (they
    create no checkpoint blockers), so an application thread freeing its own
    VA reservation during the suspend window would go unnoticed; the
    identical-VA rebuild would then fail loudly at re-map time.
