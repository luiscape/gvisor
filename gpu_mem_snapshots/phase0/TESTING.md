# Testing index: GPU multicast checkpoint/restore

There are two working mechanisms and about twenty runners in this directory,
accumulated across several lines of investigation. This is the map. Start
here.

## What is actually broken (read this first)

`IPC_CHECKPOINT_BISECT.md` isolates, natively and deterministically, the two
driver defects that cap every mechanism here. Multicast is not one of them --
it is a *consequence* of the first:

1. **A live imported VMM handle breaks `restore`** (`cuMemImportFromShareableHandle`). `cuMemRelease` of the import is necessary and sufficient; unmapping is not enough, and mapping is not required. Reversible, which is what makes suspend/replay viable at all.
2. **`cuIpcGetMemHandle` breaks `checkpoint`** on the exporter, and only freeing the allocation clears it. Not reversible, so no interposer can cover legacy CUDA IPC -- this is why vLLM's custom all-reduce must stay off.

Both reproduce with no gVisor, no NCCL, no PyTorch and no multicast, in
`ipc_scale_probe.py`.

## Which mechanism should I use?

Multicast objects (`NV_MEMORY_MULTICAST_FABRIC`, class `0x00fd`) make a
process un-checkpointable, and something has to release them in-process
through libcuda before `cuda-checkpoint` runs, then rebuild them at identical
virtual addresses afterwards. Two things can do that:

| | **mcshim** (interposer) | **NCCL fork** |
| --- | --- | --- |
| Covers | **every** multicast owner | only multicast NCCL owns |
| Needs | `LD_PRELOAD`, `dlsym` interception (~1.9k lines C) | a patched libnccl |
| Engine changes | none | none (control thread) or a sleep hook |
| Quiesces the app itself | **yes** (gates libcuda entry points) | no (graph replay bypasses it) |
| vLLM with custom all-reduce / symmetric memory | works | **refused** -- cannot reach those owners |

**Default to mcshim.** It is the only one that covers every owner, and it is
the only one that can quiesce an application that does not pause itself. The
NCCL fork is the simpler stack and is a reasonable choice if you control the
engine configuration and can keep the other multicast owners off.

They compose: `MECH=both` is verified, and the interposer correctly sees only
what NCCL did not already release.

## Canonical commands

Preconditions everywhere: `sudo nvidia-smi -pm 1` (without persistence the GPU
deinitialises between checkpoint and restore), fabric manager `Completed`, and
all GPUs reading 0 MiB before you start.

    # 1. Synthetic acceptance matrix -- the fastest full check of the interposer
    sudo bash run_matrix.sh

    # 2. PyTorch tier: torch.distributed NCCL + NVLS + a captured CUDA graph
    sudo WORLD=4 MECH=mcshim bash run_torch_nccl_gvisor.sh
    sudo WORLD=4 MECH=mcshim NO_PAUSE=1 bash run_torch_nccl_gvisor.sh   # app never pauses
    sudo WORLD=4 MECH=nccl   bash run_torch_nccl_gvisor.sh              # NCCL fork instead
    sudo WORLD=4 MECH=nccl   SYMM_MEM=1 bash run_torch_nccl_gvisor.sh   # boundary: expect refusal

    # 3. vLLM end to end
    cd ../../cr-bench
    sudo RUNSC=/usr/local/bin/runsc-phase0 CUDA_CKPT_JOB_FILE=1 CUDA_CKPT_SEQUENTIAL=1 \
         CUDA_MULTICAST_SHIM=1 bash bench_4_vllm_multi.sh --gpus 0,1 --tp 2

    # 4. Pass rates (the restore-toggle bug is intermittent; one run proves little)
    sudo TRIALS=5 TP=2 GPUS=0,1 MECH=mcshim bash vllm_trials.sh

    # 5. Parameter sweep: which vLLM configurations are covered
    sudo TP=2 GPUS=0,1 bash mcshim_sweep.sh

## Runner index

**Current -- use these**

| Runner | What it covers |
| --- | --- |
| `run_matrix.sh` | interposer acceptance matrix, TP=4/8, NVLS on/off, paused/running |
| `run_torch_nccl_gvisor.sh` | PyTorch tier under gVisor. `MECH=nccl\|mcshim\|both`, `WORLD`, `NO_PAUSE`, `SYMM_MEM`, `NO_GRAPH` |
| `run_torch_nccl_native.sh` | same workload natively, `LEG=control\|main` (control proves NVLS was live) |
| `../../cr-bench/bench_4_vllm_multi.sh` | vLLM multi-GPU lifecycle: boot, sleep, checkpoint, restore, wake, verify |
| `../../cr-bench/vllm_trials.sh` | repeats a vLLM cell and classifies failures. `MECH=mcshim\|nccl` |
| `../../cr-bench/mcshim_sweep.sh` | one run per vLLM parameter cell, to find breakage |

**Mechanism-specific e2e** (narrower, still valid)

| Runner | What it covers |
| --- | --- |
| `run_mcshim_gvisor.sh`, `run_mcshim_mp_gvisor.sh` | interposer on raw-multicast workloads, single/multi process |
| `run_nccl_mcshim_gvisor.sh`, `run_nccl_mcshim_native.sh` | interposer against **stock** NCCL NVLS |
| `run_nccl_shim_gvisor_driven.sh` | the gVisor-driven interposer acceptance run |
| `run_nccl_suspend_*.sh` (4) | the NCCL fork driven by a ctypes harness that holds the comm itself |
| `run_app_suspend_test.sh` | in-process suspend/resume validation |

**Diagnostics and probes** (answer one question; not pass/fail suites)

`ipc_scale_probe.py` (**which sharing step breaks cuda-checkpoint** -- native,
deterministic, `--stage everything`; see `IPC_CHECKPOINT_BISECT.md`),
`run_fd_identity_gvisor.sh` (rendezvous identity oracle),
`run_p2p_reexport_gvisor.sh` (re-export fidelity),
`run_census.sh` (nvproxy object-graph census across a checkpoint).

**Superseded -- kept for the record, do not build on**

| Runner | Why |
| --- | --- |
| `run_phase0.sh` | Phase 0 measurements; their conclusions are in `README.md` |
| `run_gate_test.sh`, `run_suspend_test.sh` | the nvproxy-only suspend/replay path, which cannot work: freeing the objects from nvproxy makes the checkpoint save succeed but the restore toggle refuses, because libcuda's userspace bookkeeping still lists the allocation |

## Documents

| Doc | Contents |
| --- | --- |
| `IPC_CHECKPOINT_BISECT.md` | **what is actually broken in the driver**: a stage-by-stage native bisect of both the VMM and the legacy `cuIpc*` sharing paths, and what each implies for the mechanisms here |
| `GVISOR_MULTICAST_CR.md` | **the interposer's production doc**: how it works, the ordering rules and why each exists, results, parameter sweep, known limitations |
| `NCCL_PATCH_TESTS.md` | the NCCL fork: test tiers, results, the two ordering rules, rejection paths |
| `NCCL_PATCH.md` | the NCCL patch design and invariants |
| `NCCL_SUSPEND_RESULTS.md` | earlier NCCL-path results (ctypes harnesses) |
| `mcshim/README.md` | interposer internals, implementation findings, debugging |
| `README.md` | Phase 0 measurements that set the direction |
| `../PROGRESS.md` | the overall investigation narrative |

## Known-failing, and not our code

Two things fail independently of any mechanism here. Do not spend time on them
without reading the writeups first:

1. **cuda-checkpoint's restore toggle** returns `"unknown error"` on some
   workers, before any rebuild runs. Intermittent, dominant at TP>=4, hit
   SGLang too, and reproducible natively under `runc`. `NCCL_CUMEM_ENABLE=1`
   reproduces it at TP=2, which is the cheapest known repro -- see the sweep
   section of `GVISOR_MULTICAST_CR.md`. A fix belongs in cuda-checkpoint.
2. **FlashInfer all-reduce** never boots in the benchmark image: its bundled
   cccl headers are incompatible with the image's CUDA 13.3 compiler. This is
   the one multicast owner with **no coverage at all**.
