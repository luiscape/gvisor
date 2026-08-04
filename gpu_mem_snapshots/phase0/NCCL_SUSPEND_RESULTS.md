# NCCL suspend/resume checkpoint model — working e2e (native + gVisor)

> Patch documentation: **`NCCL_PATCH.md`** (design, invariants, limitations)
> and **`nccl-nvls-suspend.patch`** (exact diff, base commit `5067397c`,
> NCCL 2.30.7) — the patch itself is uncommitted working-tree state in
> `nccl/`, preserved by those two files.

This exploration checkpoints/restores an NVLS (multicast) NCCL workload —
including captured CUDA graphs — by having **NCCL itself** release its
multicast layer through the CUDA API before `cuda-checkpoint`, and re-create
it after restore. No changes to the application, PyTorch, or nvproxy are
required for the core path.

## Why this works where nvproxy-only did not

Earlier we proved that freeing the multicast RM objects from **nvproxy** makes
the checkpoint SAVE succeed but the restore toggle proactively refuses
(`"unknown error"`), because libcuda's userspace bookkeeping still lists the
multicast allocation and cuda-checkpoint owns its restore. The refusal happens
above the ioctl boundary nvproxy mediates, so it is unfixable from nvproxy.

The insight established here: the teardown must run **in-process through
libcuda**, so all three layers stay consistent — NCCL's structs, libcuda's
tables, and the kernel RM state. `ncclCommSuspend`/`ncclCommResume` (upstream
NCCL >= 2.29.7) does exactly this for NCCL's dynamic buffers; we extended it
to also cover the **NVLS multicast** layer.

## What was built

Upstream NCCL already ships the memory-manager suspend/resume
(`ncclCommSuspend(comm, NCCL_SUSPEND_MEM)` / `ncclCommResume`), which unmaps
UC buffers keeping their VA reservations, offloads contents to CPU, releases
physical handles, and re-creates everything at identical VAs on resume. But
its `mem_manager.cc` path does **not** touch the NVLS multicast groups
(`ncclNvlsSharedRes`), which are tracked as `ncclMemPersist`. Those live 00FD
objects are exactly what hangs `cuda-checkpoint`.

Patch (in `nccl/`, host code only):

- `src/transport/nvls.cc`: new `ncclNvlsSuspend` / `ncclNvlsResume`.
  - Suspend: copy UC buff/credit contents to CPU backups; `cuMemUnmap` the MC
    VAs (retain reservations); `cuMulticastUnbind` + `cuMemRelease` the group
    handles; `cuMemUnmap` + `cuMemRelease` the UC physical memory (retain
    reservations). Untrack the UC allocations from the mem manager.
  - Resume: re-create the multicast group via the **same rendezvous as setup**
    (rank 0 `cuMulticastCreate` + export, peers import via the NCCL proxy);
    `cuMulticastAddDevice`; re-create UC memory and `cuMemMap` at the
    **identical VA**; restore contents from CPU backup; intra-node barrier;
    `cuMulticastBindMem`; `cuMemMap` the MC VA at its **identical** address.
    Re-track the UC allocations.
- `src/mem_manager.cc`: call `ncclNvlsSuspend` at the end of
  `ncclCommMemSuspend` and `ncclNvlsResume` at the end of `ncclCommMemResume`.
- `src/include/transport.h`: `ncclNvlsSharedRes` gains `mcSuspended` +
  `buffCpuBackup` / `creditCpuBackup`; declares the two new functions.

All VAs (UC unicast, MC multicast) are byte-identical across
suspend→checkpoint→restore→resume, so captured CUDA graphs and NCCL conn
structs (which reference those VAs) remain valid. Only the CUmem/multicast
**handles** change, which only teardown paths observe.

Build (host glibc 2.41 + CUDA 13 device compile is incompatible, so build in a
CUDA container):

```
cd nccl && rm -rf build
sudo docker run --rm -v "$PWD":/nccl -w /nccl nvidia/cuda:13.0.1-devel-ubuntu24.04 \
  bash -c 'apt-get update -qq && apt-get install -y -qq git python3 && \
           make src.build -j64 NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"'
cp build/lib/libnccl.so.2.30.7 /opt/phase0/nccl/nvidia/nccl/lib/libnccl.so.2
```

## Results (driver 610.57.04, 4x H100 NVSwitch, NVLS engaged)

Workload `nccl_suspend_workload.py`: 4-GPU `ncclCommInitAll` clique,
`NCCL_NVLS_ENABLE=1`, per-iteration verified allreduce, and a captured CUDA
**graph** of the allreduce replayed and verified each iteration. NVLS reports
~503 MB suspendable.

### Native (`run_nccl_suspend_native.sh`) — PASS

| leg | result |
|-----|--------|
| CONTROL: live NVLS, no suspend | `cuda-checkpoint checkpoint` **HANGS** (rc=124, 60s) |
| MAIN: `ncclCommSuspend` → lock → checkpoint → restore → unlock → `ncclCommResume` | checkpoint **rc=0 (4s)**, restore **rc=0 (5s)**, `post-restore pass failures=0` |

### gVisor (`run_nccl_suspend_gvisor.sh MODE=plain`) — PASS

4-GPU workload under `runsc` (job-wrapped via `--cuda-checkpoint-path`,
`--network=none` for NCCL bootstrap loopback):

```
ncclCommSuspend      -> SUSPENDED (memstats suspended=1)
runsc checkpoint     -> rc=0 (7s)
runsc restore        -> rc=0
ncclCommResume       -> RESUMED (memstats suspended=0)
post-restore         -> pass failures=0   (eager allreduce + CUDA graph replay)
==== RESULT: PASS ====
```

No `--cuda-multicast-suspend` needed: NCCL already released the multicast layer
before `cuda-checkpoint` runs, so nvproxy sees zero multicast blockers.
(`MODE=combo` additionally enables nvproxy multicast-suspend as a belt-and-
suspenders for any stray objects; `MODE=gate` is a diagnostic that enumerates
what survives `ncclCommSuspend`.)

## Multi-process ranks (vLLM/SGLang tensor-parallel topology)

The single-process `ncclCommInitAll` clique above proves the mechanism, but
real engines run **one process per GPU**. `nccl_suspend_mp.py` +
`nccl_mp_launcher.py` reproduce that: a launcher forks one rank process per
GPU, each `ncclCommInitRank`s into one communicator (bootstrap id shared via a
file), runs verified allreduce + a captured CUDA graph, and calls
`ncclCommSuspend`/`Resume` on file markers. Under `cuda-checkpoint
--launch-job`, the launcher and all rank children are one checkpoint job.

This is the faithful equivalent of the repo's own vLLM/SGLang reduction
(`gpu_mem_snapshots/repros/repro_tp_nccl.py` "graph" mode: "mimics the
captured-graph + coupled-NCCL state of a vLLM/SGLang TP worker without loading
a model"), with the NCCL suspend/resume calls added.

### Native (`run_nccl_suspend_mp_native.sh`, WORLD=4) — PASS

| leg | result |
|-----|--------|
| CONTROL: 4 live-NVLS ranks, no suspend | per-pid `cuda-checkpoint checkpoint` **HANGS** (60s timeout) |
| MAIN: 4 ranks `ncclCommSuspend` → per-pid lock/checkpoint/restore/unlock → `ncclCommResume` | all pids checkpoint+restore **rc=0**; all 4 ranks `post-restore pass failures=0` |

### gVisor (`run_nccl_suspend_mp_gvisor.sh`, WORLD=4) — PASS

```
launcher (cuda-checkpoint --launch-job) forks 4 ranks; NVLS engaged (503MB)
ncclCommSuspend (all ranks)  -> memstats suspended=1
runsc checkpoint             -> rc=0 (9s)
runsc restore                -> rc=0 (1s)
ncclCommResume (all ranks)   -> memstats suspended=0
all 4 ranks                  -> post-restore pass failures=0  (eager + CUDA graph)
==== RESULT: PASS ====
```

Note on the verification harness: independent per-process rank loops are not
lock-step, so each rank contributes a **fixed** value (rank+1) and checks the
constant expected sum every iteration (recv pre-filled with a sentinel to
catch a stale/no-op collective). An earlier iteration-derived expected value
spuriously "failed" only because rank counters drifted; the collective output
was always correct.

## Testing the actual vLLM / SGLang cases in this repo — status & requirement

The repo's vLLM/SGLang benches (`cr-bench/bench_4_vllm_multi.sh`,
`bench_6_sglang_multi.sh`) are Docker images (`cr-bench/images/Dockerfile.{vllm,sglang}`)
bundling PyTorch + the engine + models. On this host they are **not runnable
as-is** for two independent reasons:

1. **Infra not provisioned**: no built images, no HF model cache, and no host
   PyTorch (python 3.14, no pip/torch wheels). Building requires multi-GB
   torch+vLLM+model downloads.
2. **Engine integration is required and absent**: stock vLLM/SGLang (and the
   PyTorch they bundle) **do not call `ncclCommSuspend`/`ncclCommResume`**, and
   they bundle their own (older) NCCL without the NVLS suspend extension. So
   even fully built, they would hang exactly as the CONTROL legs show — the
   suspend/resume is precisely the integration this work adds.

To run the real engines end-to-end, two changes are needed (both engine-side,
out of scope for nvproxy/gVisor):

- **Ship the patched NCCL** into the image (replace the torch-bundled
  `libnccl.so.2`, or `LD_PRELOAD` `/opt/phase0/nccl/.../libnccl.so.2` — it is
  ABI `libnccl.so.2`, forward-compatible for torch's public-API use).
- **Call suspend/resume around the checkpoint**: reach each rank's NCCL
  communicator (via torch `ProcessGroupNCCL`) and call `ncclCommSuspend`
  before `runsc checkpoint`, `ncclCommResume` after `runsc restore`. In vLLM
  this fits the existing `/sleep` + `/wake_up` lifecycle hooks the benches
  already use (`SLEEP_LEVEL` before checkpoint); in SGLang the equivalent
  release/resume hook.

Until that engine hook exists, `nccl_suspend_mp.py` (native + gVisor, PASS
above) is the faithful, runnable stand-in for the vLLM/SGLang TP case: same
topology (process-per-GPU), same stack elements (NCCL NVLS multicast + a
captured CUDA graph of a collective), driven through the same
`cuda-checkpoint --launch-job` + `runsc checkpoint/restore` path.

## Orchestration (answers the (b)-before-or-after-(c) question)

Order is: (a) pause app so it issues no new collectives/comms → (b)
`ncclCommSuspend` on every comm **before** (c) `cuda-checkpoint`; then restore,
then `ncclCommResume`, then unpause. Suspend strictly precedes checkpoint
(measured: even a bare multicast object hangs the checkpoint). Under gVisor
the app-level suspend/resume is driven here via `runsc exec` markers; in
production the engine (vLLM/SGLang) or a coordinating shim calls
`ncclCommSuspend`/`ncclCommResume` around the `runsc checkpoint`/`restore`.

## Files

- `nccl/` — patched NCCL (branch/tree as cloned; do not commit — user manages git).
- `NCCL_PATCH.md` / `nccl-nvls-suspend.patch` — patch design doc + preserved diff.
- `nccl_suspend_workload.py` — single-process NVLS + CUDA-graph clique.
- `nccl_suspend_mp.py` — per-rank (one process per GPU) NVLS + CUDA-graph.
- `nccl_mp_launcher.py` — forks one rank per GPU (the cuda-checkpoint job root).
- `_nccl.py` — ctypes bindings (init/allreduce/suspend/resume/memstats,
  single- and multi-process).
- `run_nccl_suspend_native.sh` — single-process native e2e (CONTROL + MAIN).
- `run_nccl_suspend_gvisor.sh` — single-process gVisor e2e (MODE=plain|combo|gate).
- `run_nccl_suspend_mp_native.sh` — multi-process native e2e (CONTROL + MAIN).
- `run_nccl_suspend_mp_gvisor.sh` — multi-process gVisor e2e (vLLM/SGLang topology).

## Next / open

- **DONE — multi-process ranks**: validated native + gVisor with one process
  per GPU (`nccl_suspend_mp.py`), the vLLM/SGLang TP topology. The NVLS resume
  rendezvous (rank-0 create+export, peers import via the NCCL proxy) works
  across separate processes as expected.
- **vLLM/SGLang engine integration** (the remaining gap to run the actual
  benches): ship the patched NCCL into the image + call
  `ncclCommSuspend`/`Resume` around the checkpoint from the engine (vLLM
  `/sleep`+`/wake_up`, SGLang equivalent). See the status section above.
- P2P (non-NVLS) fabric paths and symmetric-memory (`torch.distributed.
  _symmetric_memory`) multicast are separate owners of 00FD objects; NCCL
  suspend only covers NCCL's. `MODE=combo` (nvproxy multicast-suspend) is the
  fallback for non-NCCL multicast, but it hits the cuda-checkpoint restore
  boundary — those owners would each need their own in-process suspend.
- Upstreaming: the NVLS suspend/resume is a natural extension of NCCL's
  existing `ncclCommSuspend`; worth proposing to NVIDIA.
