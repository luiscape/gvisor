# Multi-GPU CUDA checkpoint/restore: status and reviewer reply

Status as of 2026-10-08. Branch `luis/multicast-xgpu-upstream`, merged with
upstream master at `c66c2773d`. This is a dev-only file: drop it before
opening upstream PRs.

## Summary

-   **Third interposer review:** every item and every optional item is
    addressed. The reply is at the end of this file. Before the engineering
    brief work started, the branch passed every test on H100 ×8 with driver
    610.57.04: the shim suite, all 15 engine cells and the 3 cells without
    `CAP_SYS_PTRACE`.
-   **Engineering brief, P0 tier:** all seven tasks are done and pushed. The
    full engine gate and a large-model check (Qwen2.5-72B on all 8 GPUs) are
    running on the final binary.
-   **Scope for the first merge:** R610 only. R615 comes after the first
    version is merged.

## P0 status

| Task | Status | Commit | Validation |
| :--- | :--- | :--- | :--- |
| P0.1 cudart resolvers per caller | Done | `ea443eb56` | `rtres` (cudart 12.8 and 13.0 in one process): each caller gets its own runtime's `cuMemcpyBatchAsync` ABI, by call and through `dlsym`; the old shim gave the 12.8 caller `cuMemcpyBatchAsync_v2`. An unplaced caller with two runtimes refuses checkpoints. A runtime renamed `libcudart.so.99` resolves. `torch-kernel` and `torch-symm` still pass. |
| P0.2 track the resolver hooks | Done | `5fd7d9c25` | `abi`: an unknown `cuGetProcAddress_v3` refuses the gate; the old shim accepted. 1,132 lookups up to CUDA 13.4 still redirect. |
| P0.3 silent preload | Done | `b86d14b83` | `silent`: `sh -c true` under the preload prints nothing; logs go to `/tmp/mcshim/mcshim.log`. `reason`: a refusal's cause and last log lines reach `runsc checkpoint`'s error. `TestContainerFileExists` covers the IMAGE-mode check. |
| P0.4 bound every cuda-checkpoint call | Done | `3592a60f1` | `deadline`: a hung lock, and a hung checkpoint, are killed after the timeout (5 s in the test); the checkpoint fails with the reason, and the application keeps running. |
| P0.5 confine job mode to opt-in | Done | `41a023ea2` | `optout`: without the interposer, no `--launch-job`, and exactly `--get-state` then `--toggle`. `TestCudaMulticastShimEnabled`. |
| P0.6 `MCSHIM_ALLOW_FABRIC` and 00f8 | Done | `9815084a2` | Measured on R610 (below). `refuse fabric`: refused while a fabric-capable allocation is alive, accepted once it is released. |
| P0.7 stale comments | Done | `8a0fb2db0` | Includes a fourth stale `do_suspend` reference the brief missed. |

## Decisions taken

### P0.5: one opt-in

The interposer flag (`--cuda-multicast-shim-path` or
`--cuda-multicast-shim-source=EMBEDDED`, which already requires
`--cuda-checkpoint-path`, on R610+) is the only switch. It turns on all of
the following together:

-   the `cuda-checkpoint --launch-job` wrap;
-   sequential `cuda-checkpoint`, as jobs require;
-   two-phase lock/checkpoint;
-   the interposer protocol (gate, suspend, resume);
-   the blocker inventory;
-   bounded invocations (`--cuda-checkpoint-timeout`, default 10 minutes).

Without the flag, CUDA checkpoints are exactly as upstream's: one unbounded
`--toggle` per process, parallel unless `--cuda-checkpoint-sequential`, toggled
back if any fails. There is no job and no blocker inventory. The runtime
`--cuda-checkpoint-path` then only sets the default binary for `runsc
checkpoint` (upstream has no runtime flag, so no existing configuration
changes).

Why:

-   **Simple to review and canary:** nothing changes unless you opt in.
-   **No exception needed to decision 5.**
-   **Avoids new refusals for existing configurations.** The blocker
    inventory now also counts fd imports. For configurations that don't opt
    in, it would turn today's "checkpoint succeeds, restore fails" for VMM
    IPC into a refusal at checkpoint time. That's better, but it is visible,
    and the import tracking has only been validated on the 15 engine cells.

Shipping the inventory and the deadlines to everyone is a small follow-up
once the canary shows no false positives.

### P0.6: what `cuda-checkpoint` does with fabric allocations (R610)

Measured on 610.57.04, H100, single process, `cuda-checkpoint --toggle`
twice, contents verified after restore:

| Allocation | Result |
| :--- | :--- |
| `POSIX_FD` | checkpoint and restore OK |
| `POSIX_FD`, live POSIX export | OK |
| `FABRIC\|POSIX_FD` | OK |
| `FABRIC\|POSIX_FD`, live FABRIC export | OK |
| `FABRIC\|POSIX_FD`, live POSIX export | OK |
| `FABRIC\|POSIX_FD` without an IMEX channel | `cuMemCreate` fails with `NOT_PERMITTED` (800) |

So the interposer's comment ("00f8 cannot be serialized") was wrong, and the
blocker inventory's comment was right. Both comments now record the
measurement. FABRIC stays stripped by default, since the shim cannot rebuild
IMEX sharing. Under `MCSHIM_ALLOW_FABRIC`, the gate refuses while a
fabric-capable allocation is alive, because other drivers are unmeasured.

### P0.2

R610's `libcuda` exports only `cuGetProcAddress` and `cuGetProcAddress_v2`.
There is no `_v2_ptsz`, so no new wrapper is needed.

## Open questions

-   **The blocker inventory for configurations that don't opt in.** I
    recommend leaving it out of the first merge (see P0.5).
-   **The fdinfo line.** nvproxy now adds an `nvproxy_exported_object` line to
    `/proc/<pid>/fdinfo` for exported `/dev/nvidiactl` fds, for every nvproxy
    user. Applications can see it, but it is harmless (dmabuf has the same
    pattern); this is open question 4 in the design doc. Gating it on the flag
    means plumbing the flag into nvproxy's options.
-   **Correction to an earlier status.** Upstream master has #14850 (FLA) but
    not #14817 (`--device-map`): master's `state_cuda.go` still has the FIXME
    for it.

## Validation on this node (H100 ×8, 610.57.04)

-   **Go unit tests and nogo:** `nvproxy`, `control`, `state`, `boot`,
    `config` all pass.
-   **Native interposer tests:** `abi gate mc refcount refuse mapwait silent`
    all pass. `refuse fabric` needs an IMEX channel, which I created only for
    that run.
-   **Under runsc:** `ipc orphan deadline reason optout` all pass. `deadline`,
    `reason` and `optout` use a stub `cuda-checkpoint`
    (`tools/mcshim/test/ckpt_stub.c`) that logs its calls and hangs or fails
    on request.
-   **Engine cells:** a quick check on the P0.5 binary passed: `vllm_tp2`
    (44x faster to first inference than a cold boot) and `sglang_tp4_symm`
    (19x). The full gate (15 + 3 cells) is running on the final binary
    (`ea443eb56`).
-   **Shim suite on the final binary:** all 15 tests pass (abi, gate, mc,
    refcount, refuse, mapwait, silent, rtres, ipc, orphan, deadline, reason,
    optout, torch-kernel, torch-symm).

### Large model on all 8 GPUs

vLLM 0.29.0 serving Qwen2.5-72B-Instruct (bf16, 145 GB) at TP=8 with the
final binary: **PASS**, identical output at temperature 0, restored onto all 8
GPUs.

| Step | Time |
| :--- | :--- |
| Cold boot | 386 s |
| Checkpoint (engine asleep) | 61.8 s, image 172 GB |
| `runsc restore` returns | 8.3 s |
| First inference after restore | 35.5 s (10.8x faster than a cold boot) |

Where the time goes:

-   **Interposer:** 10 processes managed; 0.1 s to arm the gate, 0.8 s for
    the teardown, 1.2 s for the rebuild. The run created 10 multicast
    (NVLS) objects.
-   **`cuda-checkpoint`:** lock and checkpoint took 31.8 s, and the restore
    toggle 23.7 s, one process at a time as job mode requires. That is most
    of the checkpoint and of the time to first inference. Running it in
    parallel (which job mode forbids) is the main remaining lever (P2.5).

## Next

1.  The full shim suite and the 15 + 3 engine cells on the final binary
    (running).
2.  A large workload on all 8 GPUs: vLLM with Qwen2.5-72B at TP=8 passed
    (above); SGLang next if time allows.
3.  After the first merge: R615 (P2.4), then P1.

---

## Reply to the third interposer review

Thanks again. All five items and all the optional ones are addressed.
Branch: `luis/multicast-xgpu-upstream`.

**1. Release while torn down.** Done as suggested.

-   The release is now derived from state: the gate marker is gone, the gate
    is armed, and the process is neither torn down nor broken. The `armed`
    flag is gone.
-   The preflight disarms only when the process is neither torn down nor
    broken.
-   `on_gate_up` removes a stale `error.<pid>` before acking.
-   A suspend that arrives without the gate now refuses, so that preflight
    path is deleted.

Test: `gate` removes and re-creates the gate while the process is torn down,
once accepted and once refused, and checks that the application is neither
released early nor stranded.

**2. Multicast `cuMemMap` under `g_lock`.** Agreed: the rule is "no call that
can wait on the GPU or a peer runs under the lock", not "calls the header
marks synchronous".

-   `cuMemMap` now translates under the lock, calls the driver without it,
    and commits only if the handle's slot still holds the same object (each
    slot has a generation).
-   The preflight returns before taking the lock when the drain fails.
-   I also took the optional unmap handle-reuse fix: `cuMemUnmap` marks the
    objects it frees as dying, and lookups skip them.

Test: `mapwait`. With the old shim, two processes deadlock: each maps its own
group on one thread while another thread adds its device to the peer's group.
The test also checks that a map stuck in flight makes the gate refuse after
the drain timeout.

**3. Import check in the sentry.** Conceded, and done as suggested.

-   nvproxy now records objects imported through `IMPORT_OBJECT(S)_FROM_FD`,
    together with the exported object they came from.
-   Once every process is gated, the sentry checks that each import's
    exported object still exists in a process that runs the interposer. This
    happens before anything is locked or torn down.
-   The shim's `exp-*` announcements and its `kill(pid, 0)` probe are gone.

The end-to-end test caught two things that reading the code would not have:

-   libcuda reuses a freed RM handle immediately: the exporter's next
    allocation took it. A handle-based check therefore passed an orphaned
    import, so the check now compares nvproxy's object records instead of
    handles.
-   libcuda exports some allocations with a companion object in a later slot
    of the fd, and the importer imports both. Only slot 0 has a recorded
    identity, so imports from any slot are attributed to slot 0, the
    allocation the interposer re-exports.

Writing the import handlers also turned up a latent bug:
`NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM` was 0, while the driver defines it
as 1.

New test, `orphan`: one process maps memory that another process exported and
then freed. `run.sh` runs a real `runsc checkpoint`, expects the refusal
("imports that could not be rebuilt after a restore: ... which will not be
re-exported"), and checks that both processes keep running.

**4. Post-suspend backstop.** It already existed: `CheckpointBlockers` runs
after every process acks `suspended`, before `cuda-checkpoint`. But nvproxy
had no record of fd imports, so it could not see a leaked one. With item 3,
imports are blockers too. The up-front check refuses them in processes
without the interposer, and the backstop catches any the interposer left
behind.

**5. `mem_dev`.** Measured on R610: the driver refuses every multicast bind of
imported memory (`CUDA_ERROR_INVALID_VALUE`), by handle or by address, v1 or
v2, while the same binds of local memory succeed. So an import is never a
bind target, and the only wrong case was a v1 bind by address of memory the
shim never saw.

-   That case now asks the driver (`CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL`).
-   Tracked memory uses its allocation's location.

`mc` covers it: a group bound by address, with device 0 current, over memory
on both devices that the shim never saw. With the old shim, suspend unbinds
the wrong device (device 0) and fails.

**Optional items.**

-   **Unmap handle-reuse window:** done (see 2).
-   **Bind flags:** recorded and replayed.
-   **Unmapped export contents:** documented in the README's known
    limitations.
-   **`devs[]` in `devs_in_use`:** dropped. A transition re-adds a group's
    devices from the group's own device, so a device only added to a group
    needs no context. Test: a `refuse` case that adds a device from another
    device's context is accepted and survives a cycle.
-   **Suspend without a gate:** now refuses (see 1).
-   **Table scans:** done. I measured before deciding: a miss scanned all
    4096 slots in about 2.5 µs, since a slot is 232 to 240 bytes, and VMM
    calls do several scans. Slots fill first-free from 0, so a grow-only
    high-water mark per table bounds scans by the peak number of live
    entries: 0.18 µs at 300 entries. Launches never scan; the gate only
    checks a flag.

**Also hardened on the sentry side.** A checkpoint attempt after a failed
teardown used to wait out the 5-minute ack timeout for acks no process would
send. It is now refused at once. Before each request, the sentry also clears
every stale ack of the kind it waits for, not only error acks.

**Validation (H100 ×8, R610.57.04).**

-   Unit tests pass, and so does the shim suite (abi, gate, mc, refcount,
    refuse, mapwait, ipc, orphan, torch-kernel, torch-symm).
-   All 15 engine cells pass a checkpoint/restore with identical output:
    vLLM TP=2/4/8, SGLang TP=4/8, NVLS, symmetric memory, FlashInfer fusion,
    and restores onto other GPUs. So do the 3 cells without
    `CAP_SYS_PTRACE`.
-   Timings are unchanged from the last round (design doc updated). The
    interposer's share is about 0.1 s to arm the gate, 0.1 to 0.7 s for the
    teardown, and 0.4 to 1.1 s for the rebuild.

The first gate run of the import check refused every vLLM cell and the
symmetric-memory cells with "imported from an unknown export". That was the
companion object from item 3; it is fixed, and those cells were re-run.

### Changes since this reply

These came from our own hardening pass, not from the review, but you may want
to look at them:

-   **Silent preload.** The shim prints nothing (it is preloaded into every
    process in the container). It logs to `/tmp/mcshim/mcshim.log`, and the
    sentry quotes a refusal's reason and the shim's last log lines in its
    error. In IMAGE mode, a missing interposer file is not added to
    `/etc/ld.so.preload`.
-   **Bounded `cuda-checkpoint`.** With the interposer, every invocation has a
    deadline (`--cuda-checkpoint-timeout`); one still running is killed, and
    the operation fails.
-   **Opt-in only.** Without the interposer, checkpoints are exactly as
    upstream's: no job, one `--toggle` per process, no blocker inventory.
-   **Resolvers tracked.** A lookup returning an unknown `cuGetProcAddress`
    ABI refuses checkpoints.
-   **Fabric allocations.** Measured on R610: `cuda-checkpoint` carries a
    process's fabric-capable allocations. Under `MCSHIM_ALLOW_FABRIC`, the
    gate still refuses while one is alive.
