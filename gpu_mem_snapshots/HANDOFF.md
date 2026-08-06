# Handoff: validate `--cuda-checkpoint-path` (CUDA IPC C/R) on H100

You are on an **H100 node** (multi-GPU, NVLink). The goal is to validate the
gVisor `--cuda-checkpoint-path` feature — which wraps a GPU container's command
in `cuda-checkpoint --launch-job` so that its CUDA processes share a
cuda-checkpoint *job* — on real **multi-GPU** workloads that the previous A10G
box could not exercise.

## What changed (the feature under test)

A new top-level config flag `--cuda-checkpoint-path=<path-in-container>`.
When it is set for a GPU container **and** nvproxy is enabled **and** the driver
is **R610+**, gVisor prepends `cuda-checkpoint --launch-job` to that container's
command. `cuda-checkpoint` initializes a memfd job file, exports
`CUDA_CHECKPOINT_JOB_FILE`, and `exec`s the original command (so the app stays
PID 1). All of the app's CUDA processes (and any they fork) join the job, which
is what makes CUDA IPC (`cuIpcGetMemHandle`) state checkpointable/restorable on
R610+.

- The flag is **per container** (each container in a multi-container sandbox
  carries its own config). A non-GPU sidecar is unaffected.
- Job members must be toggled **sequentially**, so checkpoints must add
  `runsc checkpoint --cuda-checkpoint-sequential`.
- Files changed: `runsc/config/config.go`, `runsc/config/flags.go`,
  `runsc/boot/loader.go` (function `setupCudaCheckpointJob`). Doc in
  `g3doc/user_guide/checkpoint_restore.md`.

Branch: `luis/multi-snapshot-repros`. The flag change may be uncommitted in the
working tree — make sure it is present here (`grep -rn cuda-checkpoint-path
runsc/config` should show the global flag).

## What A10G already proved (don't re-litigate)

Single-GPU classes on driver 610.57.04 all pass with the fix:

| Class | What it holds | Result on A10G |
| --- | --- | --- |
| A   | 5 normal CUDA contexts        | PASS |
| B1  | NVML-only `/dev/nvidiactl`     | PASS |
| B2  | inherited fork `/dev/nvidia*`  | PASS |
| C1  | **live CUDA IPC mapping**      | **PASS with the flag** (hung without it) |
| C2  | cuMem/VMM allocation           | PASS |
| inductor | compile-worker inherited FDs | PASS |

C1 is the one the flag fixes: without `--cuda-checkpoint-path` the sequential
toggle of the IPC producer/consumer **hangs at checkpoint**; with it, C1
checkpoints/restores cleanly.

**H100 is needed to test what A10G could not: genuine cross-GPU IPC and real
tensor-parallel engine snapshots.**

## Prerequisites (verify first)

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv,noheader
```

- Driver must be **610.57.04** (the ABI gVisor supports here). If not, install it
  with `gms/install_nvidia.sh` (runfile method; its apt/reboot branches are
  broken on Amazon Linux — do the runfile install manually if needed) and
  confirm with `runsc nvproxy list-supported-drivers`.
- Docker + nvidia-container-toolkit configured.
- All GPUs should read 0 MiB before a run.

## Build & install the test runsc

The default benchmark binary (`runsc-crbench`) does **not** support driver 610.
Build a 610-capable binary from this tree:

```bash
make build TARGETS=//runsc
sudo cp bazel-bin/runsc/runsc_/runsc /usr/local/bin/runsc-cr610-job
# (make build prints the exact binary path at the end)
```

Confirm the flag is present:

```bash
/usr/local/bin/runsc-cr610-job flags | grep cuda-checkpoint-path
```

All test commands below pin `RUNSC=/usr/local/bin/runsc-cr610-job`.

## Test harness knobs

The repro/bench harnesses read two env vars:

- `CUDA_CKPT_JOB_FILE=1` → adds `--cuda-checkpoint-path=<cuda-checkpoint>` to the
  container (enables the job wrap).
- `CUDA_CKPT_SEQUENTIAL=1` → adds `--cuda-checkpoint-sequential` at checkpoint.

Use **both together** for anything with CUDA IPC. The first run per Docker image
after a driver change needs `REBUILD_ROOTFS=1` once (stale libcuda otherwise →
CUDA error 803).

---

## Priority 1 — regression: single-GPU repro matrix

Confirm the A10G results hold on H100 (catches driver/ABI regressions).

```bash
for id in a b1 b2 c1 c2 inductor; do
  sudo RUNSC=/usr/local/bin/runsc-cr610-job CUDA_CKPT_JOB_FILE=1 CUDA_CKPT_SEQUENTIAL=1 \
    timeout 500 bash gms/repros/run_repro.sh "$id"
done
```

Expected: all six `RESULT = PASS`. C1 specifically must NOT hang.

Sanity of the gate: run C1 **without** the flag and confirm it still hangs
(proves the flag is what fixes it):

```bash
sudo RUNSC=/usr/local/bin/runsc-cr610-job CUDA_CKPT_SEQUENTIAL=1 \
  timeout 240 bash gms/repros/run_repro.sh c1   # expect: hang / timeout
```

---

## Priority 2 — the reason for H100: multi-GPU CUDA IPC + tensor parallel

These are the cases the A10G box could not meaningfully exercise. This is the
main deliverable.

### 2a. vLLM tensor-parallel snapshot

```bash
sudo RUNSC=/usr/local/bin/runsc-cr610-job CUDA_CKPT_JOB_FILE=1 CUDA_CKPT_SEQUENTIAL=1 \
  REBUILD_ROOTFS=1 bash cr-bench/bench_4_vllm_multi.sh --gpus 0,1
# then try TP=4: --gpus 0,1,2,3
```

TP workers share GPU memory via CUDA IPC and communicate over NCCL. The question
is whether the `--launch-job` wrap lets the whole TP group checkpoint/restore
coherently. Record checkpoint/restore times and whether inference matches after
restore.

### 2b. SGLang tensor-parallel snapshot

```bash
sudo RUNSC=/usr/local/bin/runsc-cr610-job CUDA_CKPT_JOB_FILE=1 CUDA_CKPT_SEQUENTIAL=1 \
  REBUILD_ROOTFS=1 HEALTH_TIMEOUT=900 bash cr-bench/bench_6_sglang_multi.sh --tp 2
```

> KNOWN RISK: on 610 the A10G box saw SGLang TP=2 **hang at init** (never
> healthy in 25 min), traced to nvproxy `frontend ioctl status=0x56`
> (`NV_ERR_NOT_SUPPORTED`) / `uvm ioctl status=0x10006` handler gaps that are
> new-in-610 and invisible to the struct-parity test. If it hangs, this is
> likely the **same nvproxy gap, not the `--launch-job` feature**. Confirm by:
> ```bash
> grep -hE 'frontend ioctl failed|uvm ioctl failed|status=0x56|status=0x10006' \
>   /data/cr-bench/cr-bench-sglang-multi-*/logs/runsc.log.*boot.txt | sort | uniq -c
> ```
> Try `EAGER=1` and `--enable-memory-saver`-off variants to bisect engine vs
> nvproxy. If it is the ioctl gap, note the exact cmd numbers — those need
> handlers added to the v610 ABI in `pkg/sentry/devices/nvproxy/version.go`
> (out of scope for this PR, but the blocker for multi-GPU engines).

### 2c. Cross-GPU CUDA IPC (peer) — the pure mechanism

`repro_c1` currently shares IPC within one GPU. On H100 you can extend it to a
true cross-GPU peer mapping (producer on GPU 0, consumer opening the handle with
peer access on GPU 1). If you add that variant, run it with the flag and confirm
it checkpoints/restores. This isolates cross-GPU IPC from all engine noise.

---

## Priority 3 — NCCL and single-GPU engine baselines

```bash
# NCCL snapshot matrix (multi-GPU collectives).
sudo RUNSC=/usr/local/bin/runsc-cr610-job CUDA_CKPT_JOB_FILE=1 CUDA_CKPT_SEQUENTIAL=1 \
  bash cr-bench/bench_7_nccl.sh

# Single-GPU engine baselines (should already pass; quick confidence checks).
sudo RUNSC=/usr/local/bin/runsc-cr610-job CUDA_CKPT_JOB_FILE=1 CUDA_CKPT_SEQUENTIAL=1 \
  REBUILD_ROOTFS=1 bash cr-bench/bench_3_vllm_single.sh
sudo RUNSC=/usr/local/bin/runsc-cr610-job CUDA_CKPT_JOB_FILE=1 CUDA_CKPT_SEQUENTIAL=1 \
  REBUILD_ROOTFS=1 bash cr-bench/bench_5_sglang_single.sh
```

> NOTE on NCCL: ranks coupled through NCCL must be toggled **in parallel**, which
> is the opposite of the job requirement (sequential). With `--cuda-checkpoint-
> sequential` a pure-NCCL rank pair may stall. The IPC job feature and NCCL's
> parallel requirement are in tension for a container that does both; the A10G
> NCCL cases only passed with `NCCL_CUMEM_ENABLE=0`. Record what you observe —
> the combined NCCL+IPC multi-GPU case is the open hard problem.

---

## Expected outcomes to record (per run)

1. Did the wrap engage? `grep 'Wrapped container' <logdir>/runsc.log.*boot.txt`.
2. cold boot time, checkpoint time, restore time.
3. Did it checkpoint (no hang)? Did it restore healthy? Does post-restore
   inference match the reference?
4. On failure: the exact stage (checkpoint hang / restore "operation not
   supported" / NCCL init error / nvproxy ioctl status) and the verbatim log
   line.

## Gotchas (from the A10G box — don't repeat)

- **`RUNSC` must be the 610 binary.** `runsc-crbench` (release build) can't run
  610. Always pass `RUNSC=/usr/local/bin/runsc-cr610-job`.
- **Stale rootfs.** After a driver install/upgrade, the first run per Docker
  image needs `REBUILD_ROOTFS=1` (else CUDA error 803 from 580-era libcuda in
  the cached rootfs).
- **`pkill -f <name>` self-matches** its own command line and kills the invoking
  shell (exit 255, swallows the rest of the command). Kill leaked sandboxes by
  PID instead: `nvidia-smi --query-compute-apps=pid --format=csv,noheader` then
  `sudo kill -9 <pid>`. Between runs, verify all GPUs read 0 MiB.
- **SGLang cold boot is slow** (single ~6 min; TP needs `HEALTH_TIMEOUT=900`+).
  Don't blind-retry a hang — diagnose it (engine init vs nvproxy ioctl gap).
- **`make build`/`make test`, not host `bazel`** (host bazel fails on
  zstd/vdso/aarch64 issues; `make` uses the build container).

## Deliverable

A short results table: for each Priority-1/2/3 case — wrap engaged? stage
reached (checkpoint/restore/inference)? PASS or the verbatim failure signature?
plus driver + runsc versions. The headline question to answer:

> Does `--cuda-checkpoint-path` (the `--launch-job` wrap) make **multi-GPU**
> CUDA IPC workloads (vLLM/SGLang tensor-parallel) checkpoint/restore on H100 +
> R610, and if not, is the blocker the feature or the separate nvproxy-610 ioctl
> handler gaps?
