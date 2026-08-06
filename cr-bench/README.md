# cr-bench: gVisor Checkpoint/Restore Benchmarks (native cuda-checkpoint)

Four progressively harder checkpoint/restore benchmarks for gVisor, driven
entirely by pure `runsc` (no Docker run/checkpoint, no CRIU), exercising the
**sentry-native cuda-checkpoint integration**:

```
runsc checkpoint --cuda-checkpoint-path=/usr/local/bin/cuda-checkpoint ...
```

At checkpoint time the sentry:

1. Discovers every CUDA process in the sandbox (any thread group holding an
   open nvproxy device FD), and filters real CUDA processes with
   `cuda-checkpoint --get-state` (driver >= R555) or by CUDA thread name.
2. Execs `cuda-checkpoint --toggle --pid N` **inside the container** on every
   CUDA process — in parallel, which is required when processes share GPU
   memory via CUDA IPC (multi-GPU tensor parallelism). This drains the GPUs,
   copies device memory into host memory, and releases the GPUs.
3. Serializes the sandbox as usual (GPU state travels inside `pages.img`).

At restore/resume time the sentry automatically re-execs
`cuda-checkpoint --toggle` on the recorded processes — no extra flags, no
`LD_PRELOAD`, no helper daemon (this is what distinguishes these benchmarks
from the older `gcr/` libgcr-based flow).

## The benchmarks

| # | Script | What it proves | Verification |
|---|--------|----------------|--------------|
| 1 | `bench_1_cpu.sh` | Plain CPU memory C/R (baseline) | sha256 of every buffer pre vs post; post-restore write/read |
| 2 | `bench_2_gpu.sh` | GPU memory C/R via native cuda-checkpoint; supports multiple GPUs *without* NCCL (`--gpus 0,1,2,3`) | per-GPU tensor checksums pre vs post; live matmul on every GPU |
| 3 | `bench_3_vllm_single.sh` | vLLM single-GPU C/R with `/sleep` + `/wake_up` lifecycle | deterministic inference pre vs post; multi-query functional check |
| 4 | `bench_4_vllm_multi.sh` | **vLLM multi-GPU (tensor parallel) C/R** — engine core + N TP workers sharing memory via CUDA IPC, communicating via NCCL | same as 3 |
| 7 | `bench_7_nccl.sh` | **Minimal NCCL snapshot repro** — bare NCCL process group (no serving stack), knobs for transport/allocator/activity; `--native` runs the same toggle cycle under docker/runc to bisect driver-side vs gVisor-side failures. Failures are *results* (exit 2 = REPRO), not harness errors | persisted-tensor checksum + fresh deterministic allreduce pre vs post; hang detection |

Benchmark 4 is the interesting one. Recommended escalation path when
debugging it: 1 → 2 (`--gpus 0,1`: multi-GPU, multi-context, but a single
process and no NCCL/IPC) → 3 (multi-process, single GPU) → 4 (multi-process +
IPC + NCCL).

## Prerequisites

- `runsc` built from a tree that includes BOTH `--cuda-checkpoint-path`
  support AND commit `1e693aa6e` ("Add application-driven checkpoint/restore
  support", which sets the kernel Saver in `Loader.New`) — older builds
  nil-panic in `invokeCudaCheckpoint`. Build and install:

  ```bash
  make build TARGETS=//runsc
  sudo cp bazel-bin/runsc/runsc_/runsc /usr/local/bin/runsc-crbench
  ```

  The benchmarks default to `/usr/local/bin/runsc-crbench` (override with
  `RUNSC=...`).
- NVIDIA driver >= R550; **>= R570 strongly recommended** for benchmark 4
  (CUDA IPC support in cuda-checkpoint). Note: the scripts auto-select the
  nvproxy driver ABI — if the exact host driver version is not in
  `runsc nvproxy list-supported-drivers`, the closest same-major version is
  used (`--nvproxy-driver-version=latest` does NOT work when the host driver
  is older than the newest supported one: CUDA init fails with
  `invalid argument`).
- Docker (used once per image to build/export the rootfs), `nvidia-container-cli`
  (for injecting host driver userspace into the rootfs).
- >= 2 GPUs for benchmark 4.
- Everything runs as root (`sudo`).

The `cuda-checkpoint` binary is downloaded from
[NVIDIA/cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint) at image
build time and baked into the GPU images at `/usr/local/bin/cuda-checkpoint`
(the sentry execs it *inside* the container, so it must be in the image).

## Usage

```bash
# 1. CPU memory snapshot/restore (simple case)
sudo bash cr-bench/bench_1_cpu.sh
sudo bash cr-bench/bench_1_cpu.sh --mem-mb 4096

# 2. GPU memory snapshot/restore (native cuda-checkpoint)
sudo bash cr-bench/bench_2_gpu.sh                       # 1 GPU
sudo bash cr-bench/bench_2_gpu.sh --gpus 0,1,2,3 --gpu-mem-mb 2048

# 3. vLLM single-GPU (uses /sleep + /wake_up around the snapshot)
sudo bash cr-bench/bench_3_vllm_single.sh
sudo bash cr-bench/bench_3_vllm_single.sh --sleep-level 1   # offload weights to CPU first

# 4. vLLM multi-GPU tensor parallel  ← the valuable one
sudo bash cr-bench/bench_4_vllm_multi.sh                    # TP=2 on GPUs 0,1
sudo bash cr-bench/bench_4_vllm_multi.sh --gpus 0,1,2,3 --tp 4
```

Common flags: `--restore-gpus LIST` (cross-GPU restore, see below),
`--compression none|flate-best-speed`, `--no-exclude-zero`,
`--rebuild-rootfs`, `--sequential` (run cuda-checkpoint serially — useful for
isolating which process fails, but will deadlock IPC-connected processes).

Common environment overrides: `RUNSC`, `DATA_ROOT` (default `/data/cr-bench`),
`NVPROXY_DRIVER_VER` (default `latest`), `CUDA_CHECKPOINT_PATH`,
`HEALTH_TIMEOUT`.

## vLLM lifecycle management (benchmarks 3 & 4)

The vLLM image runs `apps/vllm_sleep_server.py`, a wrapper around
`vllm.entrypoints.openai.api_server` that injects three endpoints
(same approach as the proven `gcr/test/vllm_sleep_patch.py`):

- `POST /sleep?level=0` — pause the scheduler; CUDA stays active but idle.
  Used **before** checkpoint so no inference is in flight and NCCL is quiet.
- `POST /sleep?level=1` — additionally offload weights to CPU and drop the
  KV cache (smaller GPU footprint for cuda-checkpoint to stage; try it with
  `--sleep-level 1`).
- `POST /wake_up` — resume; used **after** restore.
- `GET /is_sleeping` — state check.

The full flow for 3 & 4:

```
runsc run → wait /health → reference inference
→ POST /sleep?level=0
→ runsc checkpoint --cuda-checkpoint-path=/usr/local/bin/cuda-checkpoint
→ runsc restore → wait /health
→ POST /wake_up
→ first inference (timed) → verification queries
```

## Cross-GPU restore (device remapping)

All GPU benchmarks accept `--restore-gpus LIST` to checkpoint on one GPU set
and restore on a **different** one:

```bash
sudo bash cr-bench/bench_2_gpu.sh --gpus 0 --restore-gpus 1
sudo bash cr-bench/bench_2_gpu.sh --gpus 0,1 --restore-gpus 2,3
sudo bash cr-bench/bench_3_vllm_single.sh --gpus 0 --restore-gpus 2
sudo bash cr-bench/bench_4_vllm_multi.sh  --gpus 0,1 --restore-gpus 2,3
```

How it works: the harness generates a second OCI bundle whose
`linux.devices` list contains the restore GPUs and passes it to
`runsc restore --bundle`.  At save, the sentry records the device set
(minor + UUID + PCI IDs) in checkpoint metadata
(`runsc/boot/nvproxy.go:setNvproxyDeviceRemapMetadata`); at restore it
derives the new set from the restore bundle's spec and nvproxy remaps every
device FD positionally (both sets sorted by minor).  The benchmark verifies
placement from the host: memory must be resident on the restore GPUs and
absent from the originals (`cb_verify_gpu_placement`), and the sentry log
summary prints the remapping (old UUID => new UUID).

Constraints (from `nvproxy.MakeDeviceRemapping`):
- Homogeneous GPUs only (same PCI vendor/device ID).
- `#saved devices <= #restored devices`.
- **Do not use `--nvproxy-docker` (hook mode)** for cross-GPU restores: in
  hook mode the restore-side device set is derived from a dev-gofer
  directory listing of ALL host GPUs, which collapses the remapping into an
  identity map — cuda-checkpoint then fails with "no CUDA-capable device is
  detected".  These benchmarks therefore use explicit `spec.Linux.Devices`
  (GKE-style) with `--nvproxy` only, and bake the driver userspace into the
  rootfs.

Verified on this machine (all PASS, checksums/answers exact, host placement
confirmed): GPU 0→1, GPUs 0,1→2,3 (plain CUDA), vLLM single GPU 0→2, and
vLLM TP=2 GPUs 0,1→2,3.

## Multi-GPU specifics (benchmark 4)

- vLLM is launched with `--tensor-parallel-size N
  --distributed-executor-backend mp --enforce-eager`; workers are separate
  processes, so the sentry must toggle several CUDA processes at once.
  **Parallel toggling is the default and is required** — IPC-connected
  processes cannot be suspended one at a time.
- `NCCL_CUMEM_ENABLE=0` (default here) makes NCCL use classic allocations;
  cuda-checkpoint's coverage of VMM (`cuMemCreate`/`cuMemMap`) allocations is
  driver-dependent. Test the VMM path with `NCCL_CUMEM_ENABLE=1`.
- `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200` prevents the NCCL watchdog from
  killing workers across the suspend window.
- Default model is `Qwen/Qwen2.5-1.5B-Instruct` (works with TP=2 and TP=4).
  `Qwen/Qwen2.5-0.5B-Instruct` (14 heads) only supports TP=2. Both are
  pre-downloaded into the image (`--build-arg MODELS=...` to change).

## Layout

```
cr-bench/
├── README.md
├── common.sh                 # shared machinery (rootfs, bundle, runsc C/R)
├── _bench_vllm_impl.sh       # shared vLLM lifecycle (benchmarks 3 & 4)
├── bench_1_cpu.sh
├── bench_2_gpu.sh
├── bench_3_vllm_single.sh
├── bench_4_vllm_multi.sh
├── apps/
│   ├── cpu_mem_server.py     # stdlib-only memory server (benchmark 1)
│   ├── gpu_mem_server.py     # torch multi-GPU tensor server (benchmark 2)
│   └── vllm_sleep_server.py  # vLLM + /sleep /wake_up /is_sleeping
└── images/
    ├── Dockerfile.cpu
    ├── Dockerfile.gpu        # torch + cuda-checkpoint
    └── Dockerfile.vllm       # vllm + models + cuda-checkpoint
```

All artifacts (rootfs cache, bundles, checkpoint images, runsc debug logs)
live under `$DATA_ROOT` (default `/data/cr-bench`); each run's directory is
printed at exit. The rootfs export is cached per image; use
`--rebuild-rootfs` after rebuilding an image.

## Reference results

Measured on 4× A10G (24 GB, no NVLink), host driver 580.95.05 (nvproxy ABI
580.65.06), runsc `release-20260330.0-695-ge52d7a0faa4d`, systrap, no
compression, `--exclude-committed-zero-pages`. LLM benchmarks use the
default configuration: **CUDA graphs + torch.compile enabled** (this is the
main C/R use-case — restore skips the expensive warmup) and **deep quiesce**
(vLLM `sleep?level=1` / SGLang `release_memory_occupation`: weights → CPU,
idle KV cache discarded, so cuda-checkpoint stages minimal GPU memory):

| Benchmark | Cold boot | Checkpoint | Restore → first inference | Speedup | Verified |
|---|---|---|---|---|---|
| 1 CPU (512 MiB) | 3.5 s | 373 ms | 439 ms (health) | — | checksums exact ✓ |
| 2 GPU 1× (512 MiB) | 3.6 s | 957 ms | 49 ms (matmul) | — | checksums exact ✓ |
| 3 vLLM 1 GPU (Qwen2.5-0.5B) | 95 s | 3.9 s | 4.6 s | **20.5×** | answers exact ✓ |
| 4 vLLM TP=2 (Qwen2.5-1.5B) | 105 s | 9.3 s | 7.2 s | **14.5×** | answers exact ✓ |
| 5 SGLang 1 GPU (Qwen2.5-0.5B) | 374 s | 10.8 s | 6.2 s | **60.2×** | answers exact ✓ |
| 6 SGLang TP=2 (Qwen2.5-1.5B) | 461 s | 25.7 s | 10.4 s | **44.5×** | answers exact ✓ |

Notes:

- The speedups are dominated by warmup avoidance: CUDA graph capture and
  torch.compile make cold boot expensive (SGLang's `--enable-torch-compile`
  especially), while restore recovers the fully-warmed state. Use `--eager`
  (vLLM) / `--eager --no-torch-compile` (SGLang) for the old eager
  configuration (speedups drop to ~4–10×).
- Deep quiesce shrinks GPU staging dramatically: SGLang GPU memory in use
  drops from ~16 GB to ~0.8–1.4 GB per GPU before checkpoint. Use
  `--sleep-level 0` (vLLM) / `--quiesce pause` (SGLang) to keep GPU memory
  resident instead; first-inference-after-restore then avoids the weight
  reload but checkpointing stages all GPU memory.
- CUDA graphs + vLLM's cumem (VMM) allocator + SGLang's torch_memory_saver
  all survive cuda-checkpoint under nvproxy on driver R580.
- `cuda-checkpoint --toggle` suspend time is variable driver-side (3–20 s
  observed for identical workloads); checkpoint totals vary accordingly.

### NCCL snapshot failure matrix (bench_7_nccl.sh)

Minimal repro for "NCCL sessions fail multi-GPU snapshots", same host
(2× A10G used; PCIe P2P is **disabled at the topology level** on this
platform — NCCL logs `P2P is disabled between connected GPUs` — so NCCL
uses SHM (default) or NET/Socket transport; the CUDA-IPC/P2P transport
cannot be exercised here and needs P2P-capable hardware):

| Config (2 ranks, allreduce) | Native (docker + parallel toggle) | gVisor C/R |
|---|---|---|
| defaults: cuMem ON + SHM transport | **REPRO** — suspend OK, resume fails `operation not supported` | **REPRO** — identical error at post-restore toggle; sandbox killed |
| defaults + collectives in flight (`--active`) | **REPRO** — same signature | **REPRO** — same signature |
| defaults, raw NCCL 2.30.7 (`--mode ncclraw`) | (not run) | **REPRO** — library upgrade alone does not fix it |
| defaults + `ncclCommSuspend(NCCL_SUSPEND_MEM)` before toggle (`--suspend`, NCCL ≥ 2.30) | **REPRO** — suspend API succeeds on all ranks, toggle-resume still fails identically | **REPRO** — same (sentry post-restore toggle fails before ncclCommResume can run) |
| `NCCL_CUMEM_HOST_ENABLE=0` (device cuMem stays ON, SHM transport) | **PASS** | **PASS** — values exact |
| `NCCL_CUMEM_HOST_ENABLE=0` + `--active` | (not run) | **PASS** — allreduce loop continues across snapshot |
| `NCCL_CUMEM_ENABLE=0` (SHM transport) | PASS | PASS — values exact |
| `NCCL_CUMEM_ENABLE=0` + `--active` | (not run) | PASS — allreduce loop continues across snapshot |
| cuMem ON + `NCCL_SHM_DISABLE=1` (NET/Socket) | PASS | PASS — values exact |
| `--mode p2p1proc` (multi-GPU, no NCCL) | (not run) | PASS |
| cuMem ON + `--lifecycle` (ncclCommDestroy pre-checkpoint, re-init post-restore) | PASS (toggle cycle) | **PASS** — values exact; cuMem stays enabled the whole time |
| cuMem ON + in-process `setenv NCCL_CUMEM_ENABLE=0` + comm re-init (no restart) | **REPRO** — param is cached per process; flipping env mid-process is ignored | — |

Findings:

- **Root cause is driver-side, not gVisor**: the native (no-gVisor) control
  fails identically. gVisor faithfully propagates the toggle failure and
  kills the sandbox (`restore.go: post restore work failed`).
- The failure requires **both** NCCL's cuMem allocator **and** the SHM
  transport — and the precise culprit is **host-side cuMem**
  (`NCCL_CUMEM_HOST_ENABLE`, default-on with recent NCCL + drivers): the
  FD-exported `cuMemCreate(HOST)` handles backing SHM-transport buffers
  cannot be re-created at restore on driver R580. Granular actions show
  suspend/checkpoint succeed and `--action restore` fails with `OS call
  failed or operation not supported on this OS`, leaving processes stuck
  in `checkpointed`. Disabling only host cuMem
  (`NCCL_CUMEM_HOST_ENABLE=0`) fixes it while keeping the device-side
  cuMem allocator.
- **The `ncclCommSuspend`/`ncclCommResume` API (NCCL ≥ 2.30) does not
  help**: its only flag, `NCCL_SUSPEND_MEM`, releases the communicator's
  dynamic *device* scratch allocations, but the host-cuMem SHM transport
  registrations survive suspension — the toggle-restore fails identically
  with every rank suspended (verified natively; ~575 MiB/GPU still
  resident after suspend). A transport-level suspend flag would be needed
  and does not exist. The API pairs naturally with the sentry lifecycle
  (suspend → checkpoint → restore → resume, like vLLM `/sleep`+`/wake_up`)
  via `bench_7_nccl.sh --suspend`, so it is ready to re-test on newer
  drivers/NCCL.
- This is exactly the class of issue the cuda-checkpoint README's
  "610 driver features" advertise fixing (shareable-handle/IPC
  restoration). Verifying that needs a driver ≥ 610 (plus nvproxy ABI
  support) — software, not hardware.
- **The failing feature is NCCL default behavior, not something the
  engines opt into.** Whether an engine workload hits it depends on who
  disables cuMem:
    - **SGLang protects itself**: `sglang/srt/entrypoints/engine.py` sets
      `NCCL_CUMEM_ENABLE=0` by default (unless `--enable-symm-mem`).
    - **vLLM 0.26.0 does NOT** (older vLLMs had an env override; it has
      been removed). Stock vLLM TP>1 fails: bench_4 re-run with
      `NCCL_CUMEM_ENABLE=1` reproduces the identical toggle failure on
      the TP worker PIDs and the sandbox is killed. Bench 4 passes only
      because `_bench_vllm_impl.sh` sets `NCCL_CUMEM_ENABLE=0` in the
      container env by default (as does `_bench_sglang_impl.sh`,
      redundantly with SGLang's own default).
    - Both engines' stacks bundle NCCL 2.28.9, which fails with defaults
      just like 2.29/2.30 (verified by running the bare repro app inside
      the vLLM image).
- Practical mitigation today: set `NCCL_CUMEM_HOST_ENABLE=0` (narrowest;
  keeps device cuMem) or `NCCL_CUMEM_ENABLE=0` in the workload environment
  before initializing NCCL.
- **Flipping `NCCL_CUMEM_ENABLE` around a snapshot does not work.** NCCL
  reads each `NCCL_*` param once per process and caches it
  (`ncclLoadParam`), so an in-process `setenv` + communicator re-init
  still allocates with the original setting (verified: control and
  env-flip cycles fail identically). A gVisor restore also resurrects the
  process image including its environment, so nothing external can inject
  new env either. What DOES work is the **communicator lifecycle**:
  `ncclCommDestroy` on all ranks before checkpoint and a full re-init
  (new uniqueId) after restore — with cuMem enabled the entire time —
  because no NCCL allocations exist at toggle time
  (`bench_7_nccl.sh --lifecycle`, PASS end-to-end under gVisor). This is
  the pattern an engine would need for snapshot-compatible symm-mem.
  Beware lazy connection setup when testing: a re-init'd communicator
  allocates transport buffers on first collective, not at init — a toggle
  cycle before any collective ran will spuriously "pass".
- Bonus finding: ranks coupled through NCCL **must be toggled in
  parallel**. A sequential suspend of rank 0 while rank 1 has collectives
  in flight deadlocks until a driver timeout (~11 min observed) and then
  fails. The sentry already does this (parallel toggle is the default;
  `--cuda-checkpoint-sequential` would hit exactly this); the bench's
  `--native` path mirrors it.
- Not reproducible on this hardware: NVLink/P2P-transport failure modes,
  fabric handles (IMEX/MNNVL), multi-node NET — need SXM/NVSwitch systems
  or a second node respectively.

### Cold start vs restore vs in-memory restore

Knobs: `CKPT_TMPFS=1` puts the checkpoint image dir on tmpfs (equivalent to
memfd — both shmem), `RESTORE_BACKGROUND=1` passes `--background` to runsc
restore (requires `COMPRESSION=none`), `DROP_CACHES=1` drops page caches
before restore for honest cold-storage numbers. Same workloads as above:

| | vLLM 1 GPU (4.8 GB pages.img) | SGLang TP=2 (23.7 GB pages.img) |
|---|---|---|
| **Cold start** (run → serving) | 95.9 s | 445–472 s |
| **Restore, cold NVMe** | cmd 1.9 s · inference 4.6 s · load 3.5 GB/s | cmd 8.8 s · inference 13.0 s · load 2.9 GB/s (8.2 s) |
| **Restore, in-memory + --background** | cmd 1.3 s · inference 5.3 s · load 6.4 GB/s (0.74 s) | cmd 6.4 s · inference 10.5 s · load 6.0 GB/s (4.0 s) |
| **Restore, in-memory + adopt (zero-copy)** | cmd 0.64 s · inference 4.6 s · adopt **1.5 ms** | cmd 2.9 s · inference 7.8 s · adopt **3.7 ms** |

Observations:

- Async page loading is used in both modes; `--background` only makes the
  restore command return before loading completes. In every configuration
  tested the loader reported **"0 waiters waited 0s for 0 bytes"** — no
  task ever faulted on a not-yet-loaded page, even for the 23.7 GB image,
  because driver-side GPU restore and engine resume fully overlap the load.
- In the first two rows gVisor does not map pages.img directly as
  application memory: it mmaps its own MemoryFile and copies pages.img
  into it (pgalloc/save_restore.go:LoadFrom). tmpfs makes that copy run
  at memory speed (~6 GB/s here) instead of storage speed.
- The **adopt** row eliminates the copy entirely: checkpoint with
  `--pages-layout=identity` (`PAGES_IDENTITY=1`) writes pages.img sparsely
  with page contents at file offset == MemoryFile offset, and restore with
  `--adopt-pages-file` (`ADOPT_PAGES=1`) opens pages.img O_RDWR and adopts
  it as the sentry's memory file, just mmapping its pages in place
  (23.8 GB "loaded" in 3.7 ms). This also halves transient memory use — no
  2× residency while both the image and the memfd copy are live. The
  trade-off: the restored sandbox now owns and mutates pages.img, so the
  image is consumed and can only be restored once (runsc unlinks
  pages.img after a successful adopting restore, so a second attempt
  fails cleanly with "no such file"); requires
  `COMPRESSION=none`, a tmpfs-resident image for the speed win, and
  incompatible with `--direct` and the checkpoint gofer. Private
  MemoryFiles (a few MB) are saved inline in pages_meta.img and still
  copied. Identity checkpoints can *only* be restored with
  `--adopt-pages-file` (the sentry enforces this).
- Time-to-first-inference is dominated by cuda-checkpoint's GPU restore
  (highly variable, 0.9–16.5 s observed), which is why the vLLM in-memory
  run shows a slightly *higher* inference time than its cold-NVMe run —
  storage is not the bottleneck at NVMe speeds. In-memory images pay off
  for the restore-command latency, for larger images, and for slower
  (EBS/network) storage.
- Both stacks need torch's pip `nvidia/cu13/lib` on `LD_LIBRARY_PATH`
  (libnvrtc for vLLM's cumem allocator, libcudart for torch_memory_saver);
  the harness wires this automatically.

NOTE: the SGLang benchmarks require a runsc with the CUDA process
enumeration race fix made in this tree
(pkg/sentry/control/state_cuda.go): preSaveCuda re-enumerates and
suspends until no new CUDA processes appear, then verifies under kernel
pause. Without it, a process completing cuInit() between enumeration and
save (e.g. SGLang/vLLM JIT compile workers) escapes cuda-checkpoint
suspension; its leftover device mappings panic the save ("Can't save pma
with non-MemoryFile") or its live RM objects fail restore ("failed to
restore object handle ... NvStatus 31"). Optional defense-in-depth
patches for the device-mapping panic (ablated, not load-bearing with the
race fix in place) are kept in patches/.

The sentry toggled 3 CUDA processes for single-GPU vLLM (API server, engine
core, worker), 4 for TP=2, and 6 for TP=4 — all in parallel. The
`cuda-ckpt` timeline (from the sentry debug logs, printed by each benchmark)
shows GPU restore dominating post-restore latency (~13–19 s for vLLM),
consistent with driver-side context reconstruction cost.

## How this differs from `gcr/`

The `gcr/` tree is the earlier LD_PRELOAD (`libgcr.so`) + trigger-daemon
based GPU C/R prototype. These benchmarks instead rely on the in-tree
sentry support (`pkg/sentry/control/state_cuda.go`), which requires zero
in-container cooperation beyond having the `cuda-checkpoint` binary present
— the container app does not need to be signal-aware, preloaded, or
patched (the vLLM sleep endpoints are an app-level *lifecycle* nicety, not
part of the snapshot mechanism).
