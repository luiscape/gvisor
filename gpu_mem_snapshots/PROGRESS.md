# PROGRESS: multi-GPU GPU checkpoint/restore under gVisor (H100 + R610)

Companion to `HANDOFF.md`. Records the state of the investigation into
checkpoint/restore of multi-GPU (tensor-parallel) GPU containers under gVisor.

## Goal

Enable `runsc checkpoint` / `restore` of multi-GPU tensor-parallel GPU
containers (vLLM / SGLang) on **H100 + NVSwitch, driver 610.57.04**, ideally
*preserving CUDA graphs + torch.compile state* (that captured state is the main
value-add of snapshotting a warmed-up engine).

## Environment / build

- 8× H100 80GB + NVSwitch, driver `610.57.04`, Amazon Linux 2023.
- Branch `luis/start-job-for-cuda-checkpoint`.
- Build: `sg docker -c 'make build TARGETS=//runsc'` then
  `sudo cp bazel-bin/runsc/runsc_/runsc /usr/local/bin/runsc-cr610-job`.
  (Never host `bazel`; always `make` via the build container.)
- All test commands pin `RUNSC=/usr/local/bin/runsc-cr610-job`.
- Harnesses: `gpu_mem_snapshots/repros/` (`run_repro.sh` + `repro_*.py`) and
  `cr-bench/` (`bench_*.sh`). Native (no-gVisor) A/B: `gpu_mem_snapshots/native_ab.sh`.

## Feature under test

Global flag `--cuda-checkpoint-path=<path>`: for a GPU container on R610+ with
nvproxy on, gVisor wraps the container command in `cuda-checkpoint --launch-job`
so all its CUDA processes share a cuda-checkpoint *job*. Per container.
Checkpoints must add `runsc checkpoint --cuda-checkpoint-sequential`.
Harness knobs: `CUDA_CKPT_JOB_FILE=1` (adds the flag), `CUDA_CKPT_SEQUENTIAL=1`.

## TL;DR

| Workload | Checkpointable? | Condition |
| --- | --- | --- |
| Single-GPU (all classes A/B/C/inductor) | ✅ | works as-is |
| Multi-GPU **pure NCCL** (no engine) | ✅ | fabric-free **and** quiesced |
| Multi-GPU **NCCL + a CUDA graph** of a collective | ✅ | fabric-free, quiesced |
| Multi-GPU **vLLM TP, default** (torch.compile + piecewise CUDA graphs) | ❌ | **cuda-checkpoint cannot restore it — NOT a gVisor bug** |
| Multi-GPU **vLLM TP, `--enforce-eager`** | ✅ | fabric-free + eager (loses compile/graph perf) |

**Headline:** the multi-GPU vLLM restore failure is a **cuda-checkpoint (CUDA
driver R610) limitation, not a gVisor gap** — proven by a native `runc` A/B that
fails identically with no gVisor in the loop. A checkpointable multi-GPU vLLM
config exists today: fabric-free + `--enforce-eager`.

## Findings

### 1. Fabric/multicast memory is the NCCL multi-GPU blocker
- `cuda-checkpoint` cannot serialize cross-process **shared fabric/multicast**
  memory (`NV_MEMORY_FABRIC` 0xf8, `NV_MEMORY_MULTICAST_FABRIC` 0xfd) created via
  `cuMemExportToShareableHandle` / NVLS multicast → checkpoint **hangs**.
- `NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0` → **0 fabric allocations** →
  checkpointable. (The var is `NCCL_NVLS_ENABLE`, no trailing `D`.)
- Verified TP=4 pure-NCCL (`repro tp`, mode=idle): NVLS on → HANG; NVLS off →
  checkpoint 12.8 s / restore 1.3 s / **PASS** (post-restore all-reduce matches).
- Single-GPU forms no multicast group → no fabric → these flags are **no-ops**
  on single-GPU (no correctness or perf impact).

### 2. Two independent conditions for a multi-GPU NCCL snapshot
1. **Fabric-free** (above).
2. **Quiesced at checkpoint.** Busy ranks (mid-collective) make
   `cuda-checkpoint --action lock` return `"device not ready"`
   (`cudaErrorNotReady`); checkpoint then fails after the 30 s lock timeout.
   Verified: `tp` mode=idle → PASS; mode=spin (continuous all-reduce) →
   CHECKPOINT-BLOCKED. The phased-lock timeout turns the old indefinite *hang*
   into a clean, GPU-recovering failure (GPUs return to 0 MiB).

### 3. vLLM TP restore failure — root-caused
- Config to even reach checkpoint (fabric-free): `NCCL_NVLS_ENABLE=0
  NCCL_CUMEM_ENABLE=0 --disable-custom-all-reduce VLLM_ALLREDUCE_USE_SYMM_MEM=0`
  (vLLM custom all-reduce and torch symmetric-memory both allocate fabric).
- Symptom: **checkpoint OK, restore fails inside `cuda-checkpoint --action
  restore` / `--toggle` on a TP worker**: `Could not restore on process ID N:
  "invalid argument"` / `"unknown error"`.
- The `HANDOFF.md` "GT200_DEBUGGER / `NV_ERR_OBJECT_NOT_FOUND` 0x57 /
  add-an-ioctl" theory was **wrong**: `0x57` is an ioctl escape number, not a
  status; debugger objects alloc/free cleanly; the `0x56` (`NV_ERR_NOT_SUPPORTED`)
  controls seen at restore are benign driver-sourced capability probes (present
  on the *original* boot too); `UVM_MM_INITIALIZE`'s `0x10006` occurs on 100% of
  calls (benign). **nvproxy forwards every ioctl faithfully — no nvproxy gap.**

### 4. DECISIVE: native `runc` A/B proves cuda-checkpoint is the blocker
`gpu_mem_snapshots/native_ab.sh` runs the same fabric-free vLLM TP=2 image
**natively** (`docker --runtime=nvidia`, no gVisor/CRIU/nvproxy) and drives
`cuda-checkpoint` lock→checkpoint→restore→unlock on every CUDA pid:

| Native config | lock | checkpoint | restore | post-restore infer |
| --- | --- | --- | --- | --- |
| default (compile + piecewise cudagraphs) | ✓ | ✓ | ✗ worker `"invalid argument"` | broken |
| `--enforce-eager` | ✓ | ✓ | ✓ | ✓ (matches reference) |

⇒ The failure reproduces with **no gVisor in the loop** → it is a
cuda-checkpoint limitation. The specific trigger is vLLM's **torch.compile +
piecewise-CUDA-graph** state in the TP workers.

### 5. Fast repros isolate the trigger
`repro_tp_nccl.py` (`run_repro.sh tp`) modes: `idle`, `spin`, `graph`, `compile`.
- `graph` (NCCL + a hand-captured CUDA graph of a collective, fabric-free) →
  **PASS** C/R, collective replays correctly after restore. So NCCL + *plain*
  CUDA graphs round-trip fine; it is vLLM's compile/piecewise-graph machinery
  specifically that cuda-checkpoint cannot restore.
- `compile` mode added; note the `gms-repro` image needs `gcc/g++/python3-dev`
  (added) for Triton/Inductor; full Inductor in that minimal image still needs
  a `libcuda.so` link fix (not pursued — the native vLLM A/B already answered
  the question).

## Actionable conclusions

1. **Checkpointable multi-GPU vLLM today:** fabric-free
   (`NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0 --disable-custom-all-reduce
   VLLM_ALLREDUCE_USE_SYMM_MEM=0`) **+ `--enforce-eager`**. Trade-off: loses
   torch.compile + CUDA-graph perf.
2. **Preserving compile/graphs needs an NVIDIA cuda-checkpoint fix** — not
   addressable in gVisor. `native_ab.sh` is a clean gVisor-free reproducer to
   file with NVIDIA (default = fails, `EAGER=1` = passes).
3. **Pure-NCCL multi-GPU:** fabric-free + quiesce → works.
4. **nvproxy "fabric drain"** approach = dead end (libcuda keeps its own
   userspace fabric bookkeeping).

## Code state on this branch (WIP)

- `pkg/sentry/control/state_cuda.go` — **KEEP.** Phased cuda-checkpoint toggle
  (lock-all-in-parallel → checkpoint → restore). Fixed the original quiesce
  deadlock and bounds the busy-lock case with a clean timeout.
- `pkg/abi/nvgpu` + `pkg/sentry/devices/nvproxy/version.go` —
  `NV2080_CTRL_CMD_FLA_GET_FABRIC_MEM_STATS` (0x20803504) handler (**committed**,
  a real fix; workers couldn't reach checkpoint without it). Parity + differ pass.
- `pkg/sentry/devices/nvproxy/{fabric.go,fabric_unsafe.go,uvm.go,nvproxy.go,version.go}`
  — fabric-drain PoC. **INERT / dead-end** (`DrainFabricMemory` call disabled).
  Kept for reference only; do not pursue (finding 4/§drain).

## Open items / next steps

1. **Confirm** fabric-free + `--enforce-eager` end-to-end **under gVisor**
   (`bench_4_vllm_multi.sh EAGER=1`, TP=2 and TP=4); record cold-boot / checkpoint
   / restore times + post-restore inference match. (Native proves cuda-checkpoint
   can do it; this proves gVisor's checkpoint/restore + nvproxy replay handle it.)
   ```
   sudo RUNSC=/usr/local/bin/runsc-cr610-job CUDA_CKPT_JOB_FILE=1 CUDA_CKPT_SEQUENTIAL=1 \
     NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0 DISABLE_CUSTOM_ALL_REDUCE=1 VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
     EAGER=1 REBUILD_ROOTFS=1 bash cr-bench/bench_4_vllm_multi.sh --gpus 0,1 --tp 2
   ```
2. Quantify perf cost of the checkpointable config (eager, classic P2P) vs
   full-fast (compile + graphs + NVLS).
3. File the cuda-checkpoint restore bug with NVIDIA (repro: `native_ab.sh`).
4. Separate blocker (out of scope): SGLang TP init hang on 610 — new-in-610
   nvproxy ioctl gaps (`frontend status=0x56` / `uvm status=0x10006`).

## Gotchas (carried from HANDOFF + observed)

- `RUNSC` must be the 610 binary (`runsc-cr610-job`).
- After a driver/image change, first run per image needs `REBUILD_ROOTFS=1`;
  repro image edits need `--rebuild` (repros are baked into the image).
- `pkill -f <name>` self-matches and kills the shell. Kill leaked sandboxes by
  PID (`nvidia-smi --query-compute-apps=pid ...`), then verify GPUs read 0 MiB.
- NVLS only engages at TP≥4 on NVSwitch (2-GPU NCCL uses direct P2P).
