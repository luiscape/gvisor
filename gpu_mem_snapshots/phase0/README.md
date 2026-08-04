# Phase 0: measurements before implementing multicast suspend/restore

Companion to `/TASK.md` (multicast object suspend/restore in nvproxy) and the
implementation outline. Three measurements, each answering one design
question. **No replay code gets written until these are answered.**

> **Later results built on these measurements:**
> - `NCCL_SUSPEND_RESULTS.md` — NCCL-level NVLS suspend/resume (patched NCCL),
>   PASS native + gVisor, single- and multi-process.
> - `mcshim/README.md` — **generic libcuda-level interposer (Idea D)**: same
>   round-trip with no NCCL fork and no app hooks, PASS native + gVisor,
>   single-process (`run_mcshim_native.py` / `run_mcshim_gvisor.sh`) **and**
>   multi-process, one rank per GPU at WORLD=2/4 (`run_mcshim_mp_native.py` /
>   `run_mcshim_mp_gvisor.sh`) — the shim brokers the cross-rank fd
>   rendezvous itself. Stock-NCCL NVLS validation
>   (`run_nccl_mcshim_native.sh`): multicast suspend works, but restore hits
>   cuda-checkpoint's live-UC-import limitation (minimal proof:
>   `ipc_taint.py --mode hold` under `--launch-job`) — see
>   `mcshim/README.md` for the isolated gap + remediation shape.

All of this runs **natively** (no gVisor) except measurement 3, which is
gVisor instrumentation.

## Prerequisites

- Hopper GPUs (sm_90+) with NVSwitch, fabric manager running (multicast).
- Bare driver install is enough: the tests use `ctypes` on `libcuda.so.1`
  directly — no CUDA toolkit, no torch.
- `cuda-checkpoint` binary for measurement 1
  (https://github.com/NVIDIA/cuda-checkpoint, `bin/x86_64_Linux/`); pass via
  `CUDA_CHECKPOINT=...` or have it on `PATH`.

```
sudo CUDA_CHECKPOINT=/path/to/cuda-checkpoint bash gpu_mem_snapshots/phase0/run_phase0.sh
```

Individual legs: `ONLY=attach`, `ONLY=hold`, `ONLY=taint`. GPUs: `GPUS="0 1"`.

## Measurement 1: IPC taint (`ipc_taint.py`) — the go/no-go

**Question** (TASK.md "Measure before implementing"): does a
`cuMemCreate(POSIX_FILE_DESCRIPTOR)` allocation become checkpointable again
once every export FD is closed and every peer import is released — or is it
permanently IPC-tainted?

Two processes, one GPU each, no NCCL/PyTorch. Exporter creates + maps + writes
a pattern; peer imports, verifies readback, then **fully releases** (unmap,
address-free, `cuMemRelease`, close fd); exporter closes its export fd; parent
drives `cuda-checkpoint` lock → checkpoint → restore → unlock on the exporter;
exporter verifies the pattern.

- `--mode hold` (run first, automatically): sensitivity control. Peer keeps
  its import mapped; the checkpoint is **expected to be refused/blocked**.
  If it succeeds, the harness (or the driver's semantics) don't match TASK.md's
  premise — stop and investigate before trusting the taint leg.
- `--mode taint`: the real measurement.
  - **PASS** ⇒ work items 1–4 suffice; unicast allocations stay resident and
    device memory remains cuda-checkpoint's responsibility.
  - **FAIL** ⇒ nvproxy would also have to tear down/replay unicast
    allocations (device-memory contents become nvproxy's problem).
    **Stop and escalate** — materially larger design.

## Measurement 2: attach blocking (`attach_blocking.py`) — work item 4

**Question** (TASK.md work item 4): does `cuMulticastAddDevice` (RM
`NV00FD_CTRL_CMD_ATTACH_GPU`) block until **all** participating GPUs have
joined? If yes, nvproxy `afterLoad`'s serial replay across clients deadlocks
and batched attach is mandatory.

Rank A creates a 2-device multicast object, exports it, hands the FD to rank
B, and immediately attaches its GPU + binds local memory (both timed). Rank B
deliberately lags by `DELAY` (default 8s) before importing/attaching/binding.

| rank A timing                  | meaning                                      |
|--------------------------------|----------------------------------------------|
| addDevice ≈ 0s, bind ≈ 0s      | serial replay safe; WI4 stays a fallback     |
| addDevice ≈ DELAY              | attach blocks on all-join ⇒ **WI4 required** |
| addDevice ≈ 0s, bind ≈ DELAY   | batch `ATTACH_MEM` replay instead            |
| watchdog exit (rc=3)           | indefinite hang ⇒ WI4 + timeouts everywhere  |

## Measurement 3: object-graph census across cuda-checkpoint (gVisor)

**Question**: at sentry save time (i.e. *after* `cuda-checkpoint --action
checkpoint` released GPU state), which RM objects still live in nvproxy's
object graph? Specifically: do the physical memory objects that multicast
`ATTACH_MEM` references survive, or does libcuda free them during checkpoint
(to be recreated only by the post-restore toggle)?

This decides **where multicast replay can live**:
- memory objects **survive** ⇒ replay can ride `nvproxy.afterLoad()`'s
  topological sort (TASK.md's stated plan);
- memory objects are **freed** ⇒ replay must be deferred to
  `postResumeCuda`, after the restore toggle, while sandbox tasks are still
  frozen.

Implementation (in-tree, this branch):
- `pkg/sentry/devices/nvproxy/fabric.go`: `ObjectGraphCensus()` — per-client
  class→count histogram of the live object graph; fabric/multicast classes
  are marked `!` in the log line.
- `pkg/sentry/control/state_cuda.go`: logs the census `[pre-lock]` and
  `[post-checkpoint]` plus a per-client diff during `checkpointCudaProcs`, and
  `[post-load/pre-toggle]` / `[post-toggle]` plus diff around the restore
  toggle in `postResumeCuda`.

To run on a box with docker: build runsc from this branch (`sg docker -c
'make build TARGETS=//runsc'`), then run any multi-GPU checkpoint the usual
way (e.g. `repros/run_repro.sh tp` or the `symmem_nccl_ckpt.py` harness) and
grep the sandbox log:

```
grep 'nvproxy object census\|nvproxy census diff' /var/log/runsc/*.log
```

To run WITHOUT docker (self-contained, single GPU):

```
sudo nvidia-smi -pm 1   # required: see gotchas below
sudo bash gpu_mem_snapshots/phase0/run_census.sh
```

`run_census.sh` builds a bare OCI bundle (host rootfs read-only, GKE-style
nvidia device injection), runs `census_workload.py` (cuMemAlloc +
cuMemCreate/cuMemMap patterns, fabric-free), and drives
`runsc checkpoint --cuda-checkpoint-path` + `runsc restore`.

Gotchas discovered while bringing this up (all encoded in the script):
- **Persistence mode must be on** (`nvidia-smi -pm 1`): with the sandbox dead
  between checkpoint and restore, the GPU deinitializes and the restore-side
  device-remap probe fails (`NV0000_CTRL_CMD_GPU_GET_ID_INFO ... status=0x1f`).
- **/sys/module/{nvidia,nvidia_uvm}/initstate shims**: with a host rootfs,
  `nvidia-modprobe` exists, so libcuda verifies module initstate via sysfs,
  which gVisor doesn't synthesize → `cuInit` = `CUDA_ERROR_NO_DEVICE`. Docker
  GPU images lack nvidia-modprobe and skip this check. (Possible upstream
  nvproxy improvement: synthesize these sysfs entries.)
- gVisor restores sandbox clocks, so restore detection needs an explicit
  marker (`runsc exec ... touch /tmp/restored`), not a wall-clock jump.
- The deprivileged gofer can't read `$HOME`; the workload is staged in
  world-readable `/opt/phase0`.

## Results

| # | measurement        | date       | driver     | verdict |
|---|--------------------|------------|------------|---------|
| 1 | ipc_taint (hold)   | 2026-08-08 | 580.173.02 | SENSITIVE: full cycle refused with live import — **importer's** `--action restore` fails `"operation not supported"` (exporter side passes alone; the refusal lives on the import side) |
| 1 | ipc_taint (taint)  | 2026-08-08 | 580.173.02 | **NOT TAINTED**: after full release (peer unmap+addressFree+memRelease+close, exporter closes export fd), lock/checkpoint/restore/unlock passes on both pids, pattern intact, peer context alive ⇒ **work items 1–4 sufficient; unicast stays resident** |
| 2 | attach_blocking    | 2026-08-08 | 580.173.02 | Q1 `cuMulticastAddDevice`: **non-blocking** (0.000s vs 8s peer lag) ⇒ serial ATTACH_GPU replay safe. Q2 `cuMulticastBindMem` issued pre-all-join: **BLOCKS until all devices join** (8.09s ≈ delay), then succeeds ⇒ replay must complete ALL `ATTACH_GPU` across clients for a multicast object **before** any `ATTACH_MEM`; WI4 batching applies at the GPU→MEM boundary, with timeouts |
| 3 | census (save side) | 2026-08-08 | 580.173.02 | **libcuda frees the ENTIRE root client during `--action checkpoint`** — pre-lock census: 1 client, 124 objects; post-checkpoint census: **0 clients** (`RELEASED during checkpoint`). Nothing of the app's RM state survives to sentry-save. |
| 3 | census (restore)   | 2026-08-08 | 580.173.02 | The restore `--toggle` recreates 120 objects under a **NEW client handle** (`0xc1d00922` → `0xc1d00937`); workload passes post-restore with patterns intact at identical VAs. |
| 3b| handle stability   | 2026-08-08 | 580.173.02 | From `run_gate_test.sh` (census now logs handle values): all 36× `0x40` NV01_MEMORY_LOCAL_USER (vidmem) **object handles are byte-identical across the restore toggle**; only the client handle changes. ⇒ recorded `ATTACH_MEM.hMemory` references remain valid at replay; replay needs only old→new **client** handle remapping. |

### Slice 3 result: multicast suspend/replay hits the R580 cuda-checkpoint boundary

Implemented nvproxy multicast suspend (`SuspendMulticastObjects`) + replay
(`ReplayMulticastObjects`), gated behind `runsc checkpoint
--cuda-multicast-suspend` (default off). Verified with `run_suspend_test.sh`
(a live 2-GPU 00FD object, no app teardown):

| stage | result |
|---|---|
| checkpoint save | **now succeeds** (was a native hang / "out of memory") |
| — mechanism | nvproxy frees the 00FD object host-side between the lock and checkpoint phases, and plants a same-handle, same-size plain-vidmem **substitute** so libcuda's checkpoint content-save pass (which UVM-maps every handle) doesn't fault on the freed handle |
| restore toggle | **FAILS**: `cuda-checkpoint --toggle` → `"unknown error"`, *before* nvproxy replay runs |

**Root cause / boundary — now confirmed on R610 610.57.04 under the
`--launch-job` protocol** (harness `run_suspend_test.sh JOB=1`, the R610
mechanism PROGRESS.md targets; job wrapping verified engaged in the sandbox
log):

1. **Native R610 with the job still hangs** on a live multicast object
   (`native_mc_610.py`: `--action checkpoint` times out at 90s) — even with
   just `cuMulticastCreate + cuMulticastAddDevice` and no bound memory / no
   mapped VA. So nvproxy suspend IS required; the job mode does not lift the
   checkpoint hang.
2. **nvproxy suspend makes the SAVE succeed on R610** (checkpoint rc=0, was a
   hang) via the freed-00FD + same-handle vidmem substitute.
3. **The restore toggle proactively refuses** (`"unknown error"`), on BOTH
   R580 and R610: during the toggle cuda-checkpoint recreates the ordinary
   allocations, then — when it reaches the multicast object in libcuda's
   CRIU-preserved state — returns the error **without issuing any
   00FD/ATTACH_GPU/ATTACH_MEM ioctl** (grep the restore log: zero `0xfd*`
   controls, zero class-`0x000000fd` allocs). There is no interceptable
   operation for nvproxy to satisfy.

Controls that pin it to multicast specifically:
- `MODE=no-mc JOB=1` restores cleanly (the process survives) ⇒ the
  memfd/job-file restore and the toggle itself are fine; multicast is the
  trigger.
- `MAP_MC_VA=after-restore` (multicast object live but MC VA not mapped at
  checkpoint) still fails ⇒ it is the 00FD object in libcuda's state, not the
  VA mapping.

**Conclusion:** cuda-checkpoint's restore cannot reconstruct a process whose
libcuda userspace state includes a multicast object, independent of the
driver-side RM state nvproxy controls, on 580 *and* 610. This is not fixable
from nvproxy: the refusal is internal to cuda-checkpoint with no ioctl to
intercept. A fix requires an NVIDIA cuda-checkpoint change (skip/omit
multicast from libcuda's restore, or let a plugin supply it) or a libcuda-level
teardown (which needs app/NCCL/PyTorch changes — a TASK.md non-goal).
`native_mc_610.py` is a clean, gVisor-free reproducer to file with NVIDIA
(checkpoint of a live multicast job hangs).

**Convergence (shippable now):**
- `--cuda-multicast-suspend` **defaults off**: multicast stays a hard
  checkpoint blocker (slice 1), so users get a clean, attributed error rather
  than a checkpoint that restores broken — on every driver.
- With the flag on, nvproxy makes the *save* half work and the harness
  reports the cuda-checkpoint restore boundary cleanly. The suspend/replay
  machinery is complete and unit-tested; it will become end-to-end functional
  only once cuda-checkpoint stops refusing multicast at restore (NVIDIA-side).

**Design consequences of measurement 3** (this is the decisive one):

1. TASK.md's plan to replay multicast objects from `nvproxy.afterLoad()`'s
   topological sort **cannot work as stated on this driver**: at sentry-save
   time the app's object graph is empty (the whole client was released during
   the checkpoint action), so there is nothing for the topo sort to order the
   multicast objects against — and their parent client doesn't exist yet.
2. Multicast replay must instead run in `postResumeCuda`, **after** the
   cuda-checkpoint restore toggle recreates the client, while sandbox tasks
   are still frozen (the dual-site `Restore` design from the outline).
3. The replay must **remap the parent client handle** (old→new), since the
   toggle-created client gets a different handle. Object handles under it can
   still be requested at their old values (RM handles are client-chosen), so
   the identical-handle invariant for what libcuda's restored bookkeeping
   references remains achievable — but suspended-object records must be keyed
   to survive the client-handle swap.

**Re-run all three on the R610 target driver** (the box PROGRESS.md used)
before locking the design:
- measurement 1, because the refusal semantics evidently vary by driver (on
  580, an idle exporter with a live export fd checkpoints fine; only the
  importer refuses);
- measurement 3, because R610's `--launch-job` job mechanism and the R610
  restore path may change what survives the checkpoint action and whether the
  client handle is preserved.

Notes:
- This host currently has driver **580.173.02** and no `cuda-checkpoint` in
  PATH; the R610 feature branch work in `PROGRESS.md` used driver 610.57.04.
  Run measurements on the same driver you intend to ship against — IPC-taint
  semantics may differ between driver majors, so ideally record both.
- `attach_blocking` needs no cuda-checkpoint and can run anywhere multicast
  works (`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED=1`).
