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
carries it when the processes share a job, which runsc arranges with
`--cuda-checkpoint-path`.

Build with `./build.sh` (toolkit-free; runs in a pinned ubuntu:22.04 container
by default so the result loads under older glibc). The Bazel targets
`//tools/mcshim:mcshim` builds the same artifact, and `//runsc` embeds it (through `//runsc/mcshimbin`) so that a
stock runsc binary can inject the interposer into containers whose images do
not carry it (`--cuda-multicast-shim-source=EMBEDDED`).

## How it gets into a container

With `runsc --cuda-multicast-shim-path=/path/to/mcshim.so` (plus nvproxy and
an R610+ driver) -- or with `--cuda-multicast-shim-source=EMBEDDED`, in which case
runsc first writes its embedded copy of `mcshim.so` into the container
filesystem at that path (default
`/usr/local/lib/mcshim.so`) -- `Loader.setupCudaMulticastShim`
(`runsc/boot/loader.go`):

*   prepends the shim to the container's `LD_PRELOAD` **and** appends it to
    `/etc/ld.so.preload` through the container's VFS (launchers like SGLang's
    `torch_memory_saver` rewrite `LD_PRELOAD` for exactly the worker processes
    that matter; `ld.so.preload` is immune),
*   sets `NCCL_CUMEM_ENABLE=1` unless the container sets it.

The shim and the sentry (`pkg/sentry/control/state_cuda_shim.go`)
rendezvous in `/tmp/mcshim`, which must be part of the checkpoint image (not
a host mount). The sentry drives the shim in every process that announced
itself there (`present.<pid>`).

The shim is inert until a process resolves a tracked CUDA entry point and
calls `cuInit`; only then does it start its control thread and participate in
the protocol. Helpers that merely inherit the preload (shells,
`cuda-checkpoint` itself) never ack anything.

## Control protocol

Everything goes through `/tmp/mcshim`. Markers are **existence-based and
edge-triggered**: the shim reacts to a marker appearing or disappearing, never
to its content, which keeps the protocol race-free for any number of rank
processes sharing one directory.

| File                  | Written by | Meaning                                                        |
| :-------------------- | :--------- | :------------------------------------------------------------- |
| `gate`                | sentry     | created: block GPU submission (refused with `error.<pid>` if state is untracked); removed: unblock (refused after a failed transition) |
| `suspend`             | sentry     | created: tear down tracked state; removed: rebuild it          |
| `present.<pid>`       | shim       | this pid runs a control thread and will ack transitions        |
| `gated.<pid>`         | shim       | gate armed                                                     |
| `suspended.<pid>` / `resumed.<pid>` | shim | teardown / rebuild finished                      |
| `error.<pid>`         | shim       | the transition in flight failed (sentry fails fast on this)    |

The sentry waits up to 5 minutes for acks. On startup the control thread
removes any stale acks a dead predecessor with the same (reused) pid left
behind, then writes `present.<pid>`.

The `suspend` marker lives in the container filesystem, so it is **part of the
checkpoint image**: after a restore it still exists, the shim stays suspended
(and the gate stays armed), until the sentry removes it to trigger the
rebuild.

## The sentry's sequence

From `pkg/sentry/control/state_cuda.go` / `state_cuda_shim.go`:

1.  create `gate`, wait for `gated.<pid>` (no CUDA calls involved, so this is
    safe at any point; it stops *new* submissions),
2.  `cuda-checkpoint --action lock` on all ranks in parallel (drains in-flight
    work; gating mid-collective can starve peers, which fails the lock and
    with it the checkpoint),
3.  `--action unlock` (the teardown must issue CUDA calls, which a locked
    process cannot),
4.  create `suspend`, wait for `suspended.<pid>`,
5.  verify the checkpoint-blocker set is empty (trust but verify),
6.  re-lock, `--action checkpoint` (one process at a time in job mode), save
    the sandbox;
7.  on restore: `--toggle` each process (`--action restore --device-map`
    then `--action unlock` when the GPUs changed),
8.  remove `suspend`, wait for `resumed.<pid>`,
9.  remove `gate` (the shim also releases the gate itself on a successful
    resume).

## Environment variables

| Variable                 | Default         | Effect                                                            |
| :----------------------- | :-------------- | :---------------------------------------------------------------- |
| `MCSHIM_LOG`             | stderr          | append log to this path instead of stderr                         |
| `MCSHIM_DISABLE`         | unset           | silent: no control thread, acks, or gate (interposition/tracking stay active) |
| `MCSHIM_HOST_BUILD`      | unset           | build.sh: build with the host toolchain instead of docker         |
| `MCSHIM_BUILD_IMAGE`     | pinned 22.04    | build.sh: alternative base image                                  |

## Restoring onto different GPUs

The interposer needs nothing special. runsc creates `/dev/nvidia#` files for
the GPUs the restored container is given, rebinds open device FDs to them,
and passes `cuda-checkpoint --device-map` so CUDA state is restored onto the
new devices; the interposer's rebuild then runs against them.

## Design summary

*   **Tracking tables.** Fixed-size (`MAXN` = 4096 each): allocations/groups
    (`g_alloc`, with per-object `torn_down`), mappings (`g_map`, per-mapping
    `suspended`), multicast binds (`g_bind`, per-bind `unbound`). This is
    live state: app-initiated frees drop entries out of the replay set. Any
    overflow is loud and sticky (`g_untracked`): arming the gate refuses
    thereafter, failing the checkpoint before anything is torn down instead
    of corrupting the restore.
*   **Identical-VA guarantee.** Suspend unmaps with `cuMemUnmap` only, never
    `cuMemAddressFree`, so the VA reservations survive the checkpoint; resume
    maps back into them.
*   **Freed UC exporters.** A multicast-bound exporter allocation left
    resident across the checkpoint fails its next export after restore with
    OBJECT_NOT_FOUND (measured on R610: vLLM TP=4, torch `_symmetric_memory`).
    Suspend saves its contents into process memory (carried by the
    checkpoint) and releases it; resume recreates it, re-maps at the identical
    VAs, restores contents, and re-exports. Costs a device-host-device copy
    and checkpoint growth of the same size.
*   **Three-phase cross-rank resume.** (1) every exporter re-creates its
    object, re-exports it and publishes `<pid> <fd>` in `/tmp/mcshim` under
    the original export's identity (nvproxy's fdinfo line); (2) importers
    copy the fd with `pidfd_getfd(2)` and re-import; (3) binds and mappings
    are rebuilt. `cuMulticastBindMem` blocks until every device has joined
    the group, so the binds are the cross-rank barrier. Publishing strictly
    before fetching prevents rank-pair deadlock. `pidfd_getfd` needs ptrace
    access, which YAMA denies between sibling processes, so an exporter sets
    `PR_SET_PTRACER_ANY` while it has fds published: until the sentry removes
    the `gate` marker after every process has resumed.
*   **The gate.** While suspended, interposed submission entry points
    (launch/memcpy/memset/stream) block on a condvar instead of touching
    unmapped VAs and faulting the context. All tracked mutators
    (`cuMemCreate`, `cuMulticast*`, export/import, map/unmap, release) are
    gated too: a thread reaching one in the unlocked teardown
    window would create or free shared state after the strict blocker gate
    verified there was none. The shim's own teardown/rebuild calls the real
    entry points and is never gated. After a failed teardown or rebuild, the
    shim refuses to release the gate even if the `gate` marker is removed.
*   **Handle translation.** Rebuilds rotate opaque handles; each object keeps
    the original handle the application holds, and calls using it are
    translated (`xlate_mc`). Because the driver reuses handle values, an
    object whose original handle is issued anew stops translating it, so the
    value can never be misrouted to a dead object.

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

*   **PTDS apps.** For `cuGetProcAddress` lookups with the per-thread-default-
    stream flag, the shim declines to redirect stream-semantics-sensitive
    entries (its wrappers forward to the legacy-stream reals). Such apps are
    not gated on those entries.
*   **Fork.** A child forked after `cuInit` cannot use CUDA, so the shim
    stays inactive in it; inherited published fds are closed.
*   **A failed teardown or rebuild leaves the application gated for good.**
    Deliberately: better blocked than corrupt. Peers may already have
    released state the failed process needs, so nothing is rolled back; the
    sentry fails the checkpoint and the workload must be restarted.
*   **Fixed table sizes.** `MAXN` = 4096 entries per table, with loud +
    sticky refusal on overflow rather than silent partial tracking.
*   **Legacy CUDA IPC needs job mode.** Without `--cuda-checkpoint-path`,
    `cuda-checkpoint --action checkpoint` hangs on processes holding legacy
    IPC imports (measured: vLLM TP=4 with custom all-reduce). runsc logs a
    warning at container creation when the interposer is enabled without it.
*   **Multicast slot contention on restore** manifests as a re-bind timeout
    (binds are the cross-rank barrier, so one rank failing to join blocks
    the rest until the deadline).
*   **`cuMemAddressReserve`/`cuMemAddressFree` are not interposed** (they
    create no checkpoint blockers), so an application thread freeing its own
    VA reservation during the suspend window would go unnoticed; the
    identical-VA rebuild would then fail loudly at re-map time.
