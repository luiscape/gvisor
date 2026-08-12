# Test suite: NCCL NVLS suspend/resume, end to end under gVisor

Tests the NCCL fork (`nccl/`, branch `luis/nvls-suspend-resume`) as the
mechanism that makes an NVLS multicast workload checkpointable, rather than
the generic `mcshim` interposer. Both paths solve the same problem in the
same place (in-process, through libcuda); this one is NCCL-native and needs no
interposer, at the cost of only covering multicast that **NCCL** owns.

Two commits on that branch:

1. `Add NVLS suspend API` — extends `ncclCommSuspend`/`Resume` to release and
   rebuild the NVLS multicast layer. Upstreamable; no trigger, no policy.
2. `Add an optional checkpoint control thread` — the trigger. Env-gated on
   `NCCL_CKPT_CTRL_DIR`; inert otherwise.

## Why a trigger was needed

`ncclCommSuspend` must be called by the application, and real stacks cannot:
**PyTorch never exposes `ncclComm_t` to Python**, so a torch or vLLM worker
cannot reach its own communicators. Every previous NCCL-path result in this
tree used a ctypes harness that held the comm pointer itself, which is why the
patch had never been tested on a real stack.

The control thread closes that gap inside NCCL, where the communicators are
already known, and speaks the **same marker protocol gVisor already drives**
for the multicast interposer (`pkg/sentry/control/state_cuda_shim.go`):

    present.<pid>        a process with communicators announces itself
    gate      appears -> block new collectives, ack gated.<pid>
    suspend   appears -> ncclCommSuspend(NCCL_SUSPEND_MEM) on all, ack suspended.<pid>
    suspend   removed -> ncclCommResume on all, ack resumed.<pid>

So gVisor's existing orchestration drives either mechanism unchanged, and the
two can be A/B'd with the same harnesses.

## Two ordering rules, both established by measurement

1. **The app must be quiesced before suspend.** `ncclCommSuspend` copies UC
   contents to a CPU backup; under a live collective that faults the context
   with `illegal memory access` (observed simultaneously on all four ranks:
   `ncclCommMemSuspend: Failed to copy to CPU backup`). The control thread
   therefore arms its gate and drains the device before suspending.
2. **The gate does not stop a captured CUDA graph.** It is enforced in
   `ncclEnqueueCheck`, so it stops collectives submitted through NCCL's API,
   but a captured graph containing an NCCL kernel is replayed by the driver
   without re-entering NCCL. Graph-replaying workloads must additionally be
   quiesced from outside — cuda-checkpoint's lock (which gVisor takes before
   asking for the suspend) or an engine-level pause. This is the same class of
   gap as the interposer's "rule 4".

## Tiers

### Tier 0 — ctypes NCCL (pre-existing, revalidated)

`run_nccl_suspend_mp_native.sh`, `run_nccl_suspend_mp_gvisor.sh`. The workload
holds the comm pointer and calls suspend/resume itself, so it does not
exercise the trigger. Fastest check that the patch itself is sound.

Revalidated against the reworked patch (the `ncclNvlsMcMem` refactor):
**WORLD=4 native PASS**, all ranks `post-restore pass failures=0`.

### Tier 1 — PyTorch only (fast)

`torch_nccl_ckpt.py` + `torch_nccl_launcher.py`: one process per GPU
(tensor-parallel topology, no engine), `torch.distributed` NCCL with buffers
large enough that NCCL selects NVLS, a warmup allreduce, and a **captured CUDA
graph** of an allreduce replayed and verified every iteration. Each rank
contributes a fixed value so the expected sum is independent of loop skew, and
the buffer is pre-filled with a sentinel so a no-op collective cannot pass.

**The application never calls the NCCL API.** That is the point of this tier.

| Runner | Legs |
| --- | --- |
| `run_torch_nccl_native.sh` | `LEG=control` (no control thread) vs `LEG=main` |
| `run_torch_nccl_gvisor.sh` | full `runsc checkpoint` / `restore` |

Knobs: `WORLD` (2/4/8), `NO_GRAPH=1` (eager only), `SYMM_MEM=1` (see
rejection paths).

Results, 8×H100 NVSwitch, driver 610.57.04, WORLD=4:

| Leg | Result |
| --- | --- |
| native CONTROL — live NVLS, no suspend | `cuda-checkpoint --action checkpoint` **HANGS**, rc=124 on all 4 pids |
| native MAIN — control thread suspends | checkpoint rc=0, restore rc=0, all ranks `failures=0` |
| **gVisor, full C/R, CUDA graph** | suspend 4/4 acks, **checkpoint rc=0 (12s)**, **restore rc=0 (2s)**, resume 4/4 acks, all ranks `post-restore pass failures=0`, `data_ptr` unchanged |
| **gVisor, full C/R, `NO_GRAPH=1`** | same, PASS |
| **gVisor, `SYMM_MEM=1`** | checkpoint **refused** with `task 5 (client 0x…): 1 multicast` — expected, see below |

The CONTROL leg is what makes the MAIN result meaningful: it proves NVLS
multicast was genuinely engaged, and that the patch is what makes the process
checkpointable.

### Tier 2 — inference engine, sleep workflow (next)

vLLM already has the lifecycle hooks this needs: `/sleep` quiesces the engine
and releases weights, `/wake_up` brings it back, and `cr-bench/bench_4_vllm_multi.sh`
already drives them around a checkpoint. Two ways to wire the NCCL path in:

* **Transparent** — set `NCCL_CKPT_CTRL_DIR` and let gVisor drive the markers,
  exactly as tier 1 does. The engine needs no change, but rule 2 above means
  the engine must be idle: `/sleep` already provides that.
* **Explicit** — have the engine's sleep/wake hooks call `ncclCommSuspend`/
  `Resume`. vLLM's pynccl communicator is reachable from Python via ctypes
  (torch's is not), so this is feasible for vLLM's own comms but does not
  cover torch's `ProcessGroupNCCL` ones — which is precisely why the control
  thread exists.

Requires staging the patched `libnccl.so.2` into the image (LD_PRELOAD over
the torch-bundled 2.29.7, which has upstream `ncclCommSuspend` but no NVLS
extension — a useful built-in A/B).

## Rejection paths (new in the reworked patch)

`ncclNvlsSuspendCheck` runs **before any destructive work**, so a rejection
leaves the communicator fully intact. It refuses three cases, each of which a
real stack can hit:

| Case | How to exercise |
| --- | --- |
| shared NVLS resources (`refCount > 1`) | a split communicator |
| NVLS-registered user buffers | NCCL user-buffer registration |
| NCCL symmetric-memory teams (`ncclDevrHasMulticastTeam`) | NCCL window registration / `nccl_device` |

### Measured: torch symmetric memory is caught by gVisor, not by NCCL

`SYMM_MEM=1` was expected to trip `ncclDevrHasMulticastTeam`. It does not, and
the reason is worth recording: **torch `_symmetric_memory` creates its
multicast through the CUDA driver directly, not through NCCL's device-runtime
teams**, so NCCL has no knowledge of it and correctly does not reject.

What happens instead is the right outcome anyway. NCCL suspends its own NVLS
layer (4/4 acks), the torch multicast object remains, and gVisor's blocker
gate refuses the checkpoint naming the owning rank:

    cuda-checkpoint cannot proceed: 1 resource(s) it cannot serialize are
    still live after 10s: task 5 (client 0xc1d06def): 1 multicast

So the failure is loud and attributable rather than a hang or a snapshot that
misbehaves after restore, and the two mechanisms are shown to be exactly
complementary: **NCCL can only release multicast that NCCL owns**, and a
workload with a non-NCCL owner needs `mcshim` for it. The runner asserts this
refusal, so `SYMM_MEM=1` is a passing test of the boundary.

## Environment

* Patched NCCL built in a CUDA container and staged:
  `nccl/build/lib/libnccl.so.2.30.7` → `/opt/phase0/nccl-patched/libnccl.so.2`
  (host glibc + CUDA 13 device compile is incompatible; see `NCCL_PATCH.md`).
* Tier 1 runs inside the benchmark image, which carries PyTorch; the gVisor
  runner uses its extracted rootfs (`/data/cr-bench/rootfs-cr-bench-vllm`).
* `nvidia-smi -pm 1` is required, as everywhere else in this tree: without
  persistence the GPU deinitialises between checkpoint and restore.
