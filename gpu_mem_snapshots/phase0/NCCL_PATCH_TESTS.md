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

### Tier 2 — inference engine, sleep workflow (vLLM) — WORKING

Enabled with `NCCL_CKPT_PATCH=1` on the existing bench. The engine is **not
modified**: `cr-bench/common.sh` stages the patched `libnccl.so.2` into the
rootfs and `_bench_vllm_impl.sh` adds three environment variables.

The third of those is the interesting one. NCCL's control thread speaks the
same marker protocol as the interposer, so setting
`GVISOR_CUDA_MULTICAST_SHIM_DIR` to the same directory as
`NCCL_CKPT_CTRL_DIR` makes **gVisor's existing orchestration drive NCCL** —
gate and suspend before the checkpoint, rebuild after the restore toggle —
with no change to gVisor at all. The two mechanisms are interchangeable behind
one driver.

vLLM's `/sleep` provides the quiesce that rule 1 requires and, being an engine
pause, also covers rule 2 (its CUDA graphs are not replaying).

Results, stock vLLM 0.27 + torch.compile + CUDA graphs, NVLS on, TP=2,
`--gpus 0,1`:

| Leg | Result |
| --- | --- |
| `NCCL_CKPT_PATCH=0` (control, NVLS live) | checkpoint **refused**: `task 680 (client 0xc1d07081): 1 multicast` |
| `NCCL_CKPT_PATCH=1` | **PASS** — checkpoint 13.8s, restore 4.0s, wake_up ok, answer **EXACT MATCH**, 14.9x faster than cold boot (215s) |

Sentry log for the passing run, showing gVisor driving NCCL:

    Multicast interposer gated     2 of 4 CUDA process(es) in 100ms
    Multicast interposer suspended 2 of 4 CUDA process(es) in 2.11s
    Multicast interposer resumed   2 of 4 CUDA process(es) in 1.06s

"2 of 4" is correct and worth noting: only the two TP workers hold NCCL
communicators. The API server and engine-core processes hold NVIDIA device FDs
— so gVisor lists them as CUDA processes — but never register a communicator,
never write `present.<pid>`, and are therefore not waited on. Waiting on them
would hang every checkpoint.

Run it with:

    sudo RUNSC=/usr/local/bin/runsc-phase0 \
         CUDA_CKPT_JOB_FILE=1 CUDA_CKPT_SEQUENTIAL=1 \
         NCCL_CKPT_PATCH=1 NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1 \
         DISABLE_CUSTOM_ALL_REDUCE=1 VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
         bash cr-bench/bench_4_vllm_multi.sh --gpus 0,1 --tp 2

`DISABLE_CUSTOM_ALL_REDUCE=1` and `VLLM_ALLREDUCE_USE_SYMM_MEM=0` are part of
the scope, not incidental: both allocate multicast that NCCL does not own, and
the `SYMM_MEM=1` result below shows what happens if such an owner is left live.
Covering those needs `mcshim`; this path covers NCCL's NVLS.

**Explicit alternative (not needed, recorded for completeness):** having
`/sleep` and `/wake_up` call `ncclCommSuspend`/`Resume` directly. vLLM's pynccl
communicator is reachable from Python via ctypes, but torch's
`ProcessGroupNCCL` is not, so an engine hook cannot cover every communicator —
which is exactly why the control thread exists. The transparent path above
covers both and needs no engine change.

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
