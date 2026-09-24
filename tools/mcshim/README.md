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
*   **Legacy CUDA IPC** (`cuIpcGetMemHandle` / `cuIpcOpenMemHandle`, used
    by the engines' custom all-reduce for `cudaMalloc`'d buffers) is
    **promoted to VMM IPC** at export time: the exporter's allocation is
    replaced in place by a `cuMemCreate`'d one at the same VA (contents
    preserved) and exported as a POSIX fd; the blob handed to the
    application names that fd, and the importer's `cuIpcOpenMemHandle`
    imports and maps it. From then on it is the VMM path above, which the
    shim checkpoints exactly. `MCSHIM_IPC_PROMOTE=0` disables promotion, in
    which case imports are closed before the checkpoint and *replayed*
    heuristically after it (see below) -- which cannot bring packed imports
    back.

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
*   sets `MCSHIM_DIR` in the container environment (default `/tmp/mcshim`)
    unless the container chose its own,
*   records `GVISOR_CUDA_MULTICAST_SHIM_DIR` in the container *spec*, which is
    how the sentry (`pkg/sentry/control/state_cuda_shim.go`) later discovers
    that it owns an interposer in this container.

The shim is inert until a process resolves a tracked CUDA entry point and
calls `cuInit`; only then does it start its control thread and participate in
the protocol. Helpers that merely inherit the preload (shells,
`cuda-checkpoint` itself) never ack anything.

## Control protocol

Everything goes through `$MCSHIM_DIR`. Markers are **existence-based and
edge-triggered**: the shim reacts to a marker appearing or disappearing, never
to its content, which keeps the protocol race-free for any number of rank
processes sharing one directory.

| File                  | Written by | Meaning                                                        |
| :-------------------- | :--------- | :------------------------------------------------------------- |
| `gate`                | sentry     | created: block GPU submission; removed: unblock (refused while teardown state is outstanding) |
| `suspend`             | sentry     | created: tear down tracked state; removed: rebuild it          |
| `present.<pid>`       | shim       | this pid runs a control thread and will ack transitions        |
| `gated.<pid>` / `ungated.<pid>`     | shim | gate armed / released                            |
| `suspended.<pid>` / `resumed.<pid>` | shim | teardown / rebuild finished                      |
| `error.<pid>`         | shim       | the transition in flight failed (sentry fails fast on this)    |

The sentry waits up to 5 minutes for acks; the shim caps its own resume at
240s (`RESUME_DEADLINE_MS`) so the two sides cannot time out disagreeing. On
startup the control thread removes any stale acks a dead predecessor with the
same (reused) pid left behind, then writes `present.<pid>`.

The `suspend` marker lives in the container filesystem, so it is **part of the
checkpoint image**: after a restore it still exists, the shim stays suspended
(and the gate stays armed), until the sentry removes it to trigger the
rebuild.

## The sentry's sequence

From `pkg/sentry/control/state_cuda.go` / `state_cuda_shim.go`:

1.  create `gate`, wait for `gated.<pid>` (no CUDA calls involved, so this is
    safe at any point; it stops *new* submissions),
2.  `cuda-checkpoint --action lock` on all ranks in parallel (drains in-flight
    work; on failure the gate is released and the pair retried, since gating
    mid-collective can starve peers),
3.  `--action unlock` (the teardown must issue CUDA calls, which a locked
    process cannot),
4.  create `suspend`, wait for `suspended.<pid>`,
5.  verify the checkpoint-blocker set is empty (trust but verify),
6.  re-lock, `--action checkpoint`, save the sandbox;
7.  on restore: `--toggle` each process, wait until all report "running",
8.  remove `suspend`, wait for `resumed.<pid>`,
9.  remove `gate` (the shim also releases the gate itself on a successful
    resume).

## Environment variables

| Variable                 | Default         | Effect                                                            |
| :----------------------- | :-------------- | :---------------------------------------------------------------- |
| `MCSHIM_DIR`             | `/tmp/mcshim`   | control/rendezvous directory (the sentry normally sets it)        |
| `MCSHIM_LOG`             | stderr          | append log to this path instead of stderr                         |
| `MCSHIM_VERBOSE`         | unset           | per-entry suspend/resume diagnostics (chatty across ranks)        |
| `MCSHIM_DISABLE`         | unset           | silent: no control thread, acks, or gate (interposition/tracking stay active) |
| `MCSHIM_IPC_PROMOTE`     | `1`             | promote legacy IPC exports to VMM IPC at export time (0 = legacy close+replay only) |
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
    `suspended`), multicast binds (`g_bind`, per-bind `unbound`), legacy IPC
    participation (`g_ipc`). This is live state: app-initiated frees drop
    entries out of the replay set. Any overflow is loud and sticky
    (`g_track_overflow`): suspend refuses thereafter, failing the checkpoint
    up front instead of corrupting the restore. The per-entry flags clear as
    each entry is torn down or rebuilt, making both directions re-enterable
    after partial failures: a retried edge does exactly the remaining work.
*   **Identical-VA guarantee.** Suspend unmaps with `cuMemUnmap` only, never
    `cuMemAddressFree`, so the VA reservations survive the checkpoint; resume
    maps back into them (re-reserving at the fixed address if a reservation
    did not survive).
*   **Freed UC exporters.** A multicast-bound exporter allocation left
    resident across the checkpoint fails its next export after restore with
    OBJECT_NOT_FOUND (measured on R610: vLLM TP=4, torch `_symmetric_memory`).
    Suspend saves its contents into process memory (carried by the
    checkpoint) and releases it; resume recreates it, re-maps at the identical
    VAs, restores contents, and re-exports. Costs a device-host-device copy
    and checkpoint growth of the same size.
*   **Three-phase cross-rank resume.** (1) every exporter re-creates its
    object, re-exports it and serves the fd on a unix socket keyed by the
    original export identity (nvproxy's fdinfo oracle, else `st_dev:st_ino`
    plus a creation ordinal); (2) importers connect and re-import; (3) binds
    and mappings are rebuilt. `cuMulticastBindMem` blocks until every device
    has joined the group, so the binds are the cross-rank barrier. Serving
    strictly before fetching prevents rank-pair deadlock.
*   **Legacy IPC promotion.** Why not replay: `cuIpcOpenMemHandle` takes no
    address hint and the driver's legacy VA allocator is top-down first-fit
    and refuses an exact-fit hole (measured, `gpu_mem_snapshots/probes/
    ipc_reopen_probe.py`), so an import whose neighbours abut it never
    reopens at its VA. Promotion sidesteps the allocator entirely. The
    promoted range must stay indistinguishable from `cudaMalloc` memory to
    the application: `cuMemRetainAllocationHandle` reports it as non-VMM
    (SGLang's custom all-reduce probes that to pick its registration path)
    and `cuMemFree` of the base releases the VMM object. Buffers whose size
    is not a 2 MiB multiple (none observed) fall back to the legacy path.
*   **Legacy IPC replay** (`MCSHIM_IPC_PROMOTE=0`). The rendezvous key is the *original* blob (a
    re-export produces different bytes, but both sides know the original).
    Exporters serve the new blob under the old key; importers reopen in
    ascending original-open order, and since `cuIpcOpenMemHandle` takes no
    address hint, the shim holds reservations over the closed ranges across
    the checkpoint and walks stray placements back by plugging lower arena
    holes one reservation at a time. A reopen that still lands elsewhere
    fails the resume loudly -- a moved import is silent corruption.
*   **The gate.** While suspended, interposed submission entry points
    (launch/memcpy/memset/stream, plus the `cuMemAlloc*` family, whose
    allocations would shift the IPC reopen walk) block on a condvar instead
    of touching unmapped VAs and faulting the context. All tracked mutators
    (`cuMemCreate`, `cuMulticast*`, export/import, `cuIpc*`, map/unmap,
    release) are gated too: a thread reaching one in the unlocked teardown
    window would create or free shared state after the strict blocker gate
    verified there was none. The shim's own teardown/rebuild calls the real
    entry points and is never gated. While teardown state is outstanding (a
    resume failed partway), the shim refuses to release the gate even if
    the `gate` marker is removed.
*   **Handle aliasing.** Rebuilds rotate opaque handles; every handle an
    object ever had is kept in an alias list and stale references are
    translated (`xlate`). Because the driver reuses handle values, a newly
    issued value is first purged from every alias list (`aka_purge`) so it
    can never be misrouted to a dead object. The list holds 16 values
    (`MAX_AKA`; one rotation per rebuild); on overflow the oldest rotated
    value is evicted, the original handle is always kept, and the process
    logs the eviction and refuses further suspends.

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
*   The rendezvous sockets carry fds/blobs between ranks of one job. Do
    **not** share one `MCSHIM_DIR` volume across jobs: rendezvous keys could
    collide and cross-connect unrelated processes.
*   The sentry side (`state_cuda_shim.go`) treats acks as liveness signals
    with timeouts, never as data.

## Known limitations

*   **PTDS apps.** For `cuGetProcAddress` lookups with the per-thread-default-
    stream flag, the shim declines to redirect stream-semantics-sensitive
    entries (its wrappers forward to the legacy-stream reals). Such apps are
    not gated on those entries.
*   **Fork.** A child forked after `cuInit` starts with fresh, empty tracking
    (correct: CUDA contexts are unusable across fork) and its own gate/locks;
    inherited rendezvous fds are closed in the child (a serve thread's own
    dup is unreachable and may linger until exec); it participates in the
    protocol only if it initializes CUDA itself.
*   **A failed resume leaves the application gated.** Deliberately: better
    blocked than corrupt. The shim refuses to release the gate while
    teardown state is outstanding -- the sentry's unwind removing the `gate`
    marker does not override it. Per-entry flags clear as each object is
    rebuilt, so a retried suspend/resume edge converges once the underlying
    cause clears; until then the orchestrator must treat the workload as
    unhealthy.
*   **Fixed table sizes.** `MAXN` = 4096 entries per table, with loud +
    sticky refusal on overflow rather than silent partial tracking.
*   **Low-arena legacy imports** (e.g. custom all-reduce signal pads at
    TP=8) sit in driver-owned regions where no replay can place them, so
    they cannot cross a checkpoint on these drivers at all; run such
    workloads with the engine's custom all-reduce disabled.
*   **Multicast slot contention on restore** manifests as a re-bind timeout
    (binds are the cross-rank barrier, so one rank failing to join blocks
    the rest until the deadline).
*   **`cuMemAddressReserve`/`cuMemAddressFree` are not interposed** (they
    create no checkpoint blockers), so an application thread freeing its own
    VA reservation during the suspend window would go unnoticed; the
    identical-VA rebuild would then fail loudly at re-map time.
