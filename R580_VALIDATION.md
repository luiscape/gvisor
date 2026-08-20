# R580 validation log

Running record of multi-GPU CUDA checkpoint/restore validation for the
multicast interposer on **driver 580.173.02** (8x H100 + NVSwitch).

Branch: `luis/r580-multicast-snapshots`. Harnesses live on
`luis/experiment-nccl-state` (`cr-bench/`, `gpu_mem_snapshots/`).

## Environment

| Component | Version |
| --- | --- |
| Driver | 580.173.02 (sha256 matches `nvproxy/version.go:1099`, natively supported) |
| Fabric manager | 580.173.02, 8/8 GPUs `Completed`/`Success` |
| cuda-checkpoint | 580.173.02 -- **no `--launch-job`** (R610 feature), so benches pass `CUDA_CKPT_JOB_FILE=0` |
| OS / kernel | Ubuntu 26.04, 7.0.0-1006-aws, gcc 15 |
| runsc | `/usr/local/bin/runsc-r580`, built from this branch |
| Interposer | `tools/mcshim/mcshim.so` + `mcshim-helper` (containerized ubuntu:22.04 build) |

Bring-up notes: the driver builds stock on kernel 7.0 (580.126.20+ ships
NVIDIA's kernel-7.0 compat; **do not** try 580.95.05, which cannot build).
Fabric manager must use its own packaged `fabricmanager.cfg` -- a 610-era config
makes it exit 255.

## The R580 driver wall (unchanged on .173.02)

A cuda-checkpoint-**restored** process cannot admit fresh devices:
`cuMulticastCreate` succeeds but `cuMulticastAddDevice` returns
`CUDA_ERROR_INVALID_DEVICE(101)`, and `cuCtxCreate` returns OOM. Both devices
still report `MULTICAST_SUPPORTED=1` before and after, so it is not a
capability loss.

Measured with `gpu_mem_snapshots/phase0/native_mc_after_restore.py`:

| Run | Result |
| --- | --- |
| `--no-cr` (control) | `ALL=PASS` |
| with checkpoint/restore | `create=OK`, then `add_device_0=INVALID_DEVICE(101)` -> `ALL=FAIL` |

Identical to 580.126.20, so **NVIDIA did not fix this between .126.20 and
.173.02** and the helper-proxy rebuild remains required. The interposer works
around it by exec'ing `mcshim-helper` (never checkpointed) to perform
create+attach on the rank's behalf; the rank then imports the group fd.

## `--device-map` is non-functional on R580

`cuda-checkpoint --help` advertises `--device-map oldUuid=newUuid`, so it looked
like the missing piece for cross-GPU restore. It is not
(`gpu_mem_snapshots/phase0/native_device_map.py`):

| Restore invocation | Result |
| --- | --- |
| `--action restore` (no flag) | rc=0, pattern intact, context usable |
| `--device-map <old>=<new>` | rc=1 `"invalid argument"` |
| same, UUID without `GPU-` prefix | rc=1 `"invalid argument"` |
| **identity** map `<old>=<old>` | rc=1 `"invalid argument"` |

The identity row settles it: mapping a GPU to itself is a semantic no-op and
still fails, so the flag is rejected, not the move. This does not matter for
gVisor, which remaps below libcuda -- see next section.

## Fix: device namespace stability across restore (`5842cc18d`)

Restoring a sandbox onto a different GPU set **silently kept using the original
GPUs**. The remapping was computed correctly and never reached the host device
files:

* `frontendFD.load()` reopens the host device named after `fd.dev`, but only
  `nvproxy.afterLoad()` remapped `fd.dev`, and stateify does not order one
  object's `load()` against another object's `afterLoad()`. The reopen won.
* `frontendDevice.Open()` derives the host name from a static minor-to-name
  table built at `Register()` time, so even with FDs fixed, anything opening a
  device by name after restore (libcuda re-initializing during the
  cuda-checkpoint restore toggle, or the freshly exec'd helper) returned to the
  original GPU. This was the dominant effect.

Nothing reported an error: the workload kept verifying correctly on GPUs the
sandbox was no longer entitled to, which may belong to another sandbox.

Fixed by translating sandbox-visible minors to host minors in
`frontendDevice.basename()` from a map recorded once per restore. Sandbox-visible
minors deliberately do not change, so the application's device namespace is
identical to before the checkpoint and both existing FDs and later opens resolve
to the GPUs the sandbox now owns.

Evidence, checkpoint on GPUs 0,1 restored onto 6,7:

| Signal | Before fix | After fix |
| --- | --- | --- |
| `/proc/<sentry>/fd` | 17x `nvidia0`, 17x `nvidia1`, none 6/7 | 17x `nvidia6`, 17x `nvidia7`, none 0/1 |
| `nvidia-smi` attribution | GPUs 0,1 | GPUs 6,7 |
| in-sandbox `cuDeviceGetUuid` | old UUIDs | new UUIDs |

## Supported configurations (summary)

What is known to work on this driver, from the runs recorded below. Everything
assumes: the interposer (`--cuda-multicast-shim-path` + `mcshim-helper`),
`MCSHIM_IPC_SUSPEND=1`, `NCCL_CUMEM_ENABLE=1`, `CUDA_CKPT_SEQUENTIAL=1`, the
engine quiesced via its sleep/release API before checkpoint, and NO eager mode:
torch.compile and CUDA graphs are preserved in every supported cell.

| TP | Engine | Custom all-reduce | NVLS | Symmetric memory | Same-GPU C/R | Cross-GPU C/R |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | vLLM | ON | inactive at TP=2 (NCCL uses direct P2P) | ON | supported | supported |
| 2 | SGLang | ON (default) | -- | -- | supported | supported |
| 4 | vLLM | ON | ON | ON | supported | supported |
| 4 | SGLang | ON (default) | see row below | see row below | supported | supported |
| 4 | SGLang `--enable-nccl-nvls` | ON | ON | -- | supported | supported |
| 4 | SGLang `--enable-torch-symm-mem` | -- | -- | torch 2.11 | **NOT checkpointable** (see below) | -- |
| 8 | vLLM | **OFF (required)** | ON | ON | supported | n/a (uses all GPUs) |
| 8 | SGLang | **OFF (required)** | -- | -- | supported | n/a (uses all GPUs) |

The two TP=8 restrictions, precisely:

*   **vLLM TP=8 + custom AR**: custom AR shares its buffers over legacy CUDA
    IPC, whose imports scale as (TP-1) x 4 per rank: 4 at TP=2, 12 at TP=4, 28
    at TP=8. `cuIpcOpenMemHandle` takes no address hint, so the interposer
    reconstructs placement by reopening in original order and plugging arena
    holes; that handles 4 and 12 with margin and collapses at 28 (the driver
    abandons the low arena). The shim refuses to resume rather than let CUDA
    graphs run against moved pointers. Custom AR's advantage (small-message
    latency at small TP) is weakest at TP=8, where NVLS + symmetric memory
    carry the collectives.
*   **SGLang TP=8 + custom AR**: SGLang's own JIT kernel
    (`kernels/jit/csrc/distributed/ipc.cuh`) fails to compile at world=8 in
    this image -- an application-level bug unrelated to checkpoint/restore.

### SGLang `--enable-torch-symm-mem`: FIXED (runtime-resolver interposition)

torch 2.11's symmetric memory prefers **fabric handles** wherever the device
advertises `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED`, and every such
allocation creates an `NV_MEMORY_FABRIC` (00f8) object at allocation time that
cuda-checkpoint cannot serialize. The blocker gate refuses the checkpoint (one
fabric object per rank), exactly as designed.

**Root cause of the shim invisibility (found, fixed).** The earlier diagnosis
("statically linked runtime, unfixable") was wrong. torch 2.11 resolves its
entire c10 `DriverAPI` table -- `cuMemCreate`, `cuMulticast*`,
`cuMemExport/ImportFromShareableHandle`, `cuDeviceGetAttribute`, ... --
through `cudaGetDriverEntryPointByVersion`, a **public export of its bundled
`libcudart.so.13`** (`torch/include/c10/cuda/driver_api.h` pins per-symbol
versions, e.g. `(cuMemCreate, 12000)`, `(cuMulticastCreate, 12030)`).
libcudart then reaches libcuda through an internal dlvsym/`cuGetExportTable`
bootstrap that neither the `dlsym` hook nor the `cuGetProcAddress` wrappers
ever see -- hence 680 invisible fabric allocs while the strip fired only on
NCCL's path. But the torch->libcudart hop is an ordinary cross-DSO binding,
which LD_PRELOAD wins: the shim now interposes all four resolver exports
(`cudaGetDriverEntryPoint[ByVersion][_ptsz]`), forwards to the real runtime,
and post-processes results through the same `gpa_redirect` as
`cuGetProcAddress` (runtime stream-flag values coincide with the driver's
bits; in the `_ptsz` variants `cudaEnableDefault` itself means PTDS).

Better still, disassembly of `libc10_cuda.so` shows torch's
`isFabricSupported()` is an **empirical probe**: `cuMemCreate` +
`cuMemExportToShareableHandle(type=FABRIC)` on a scratch allocation. With the
resolver interposed, the existing `cuMemCreate` fabric strip makes that probe
fail cleanly, so torch itself concludes fabric is unavailable and falls back
to POSIX-FD symm-mem -- the exact path the shim already suspends/resumes.

Native verification (cr-bench-sglang image, torch 2.11, 2 ranks, bf16
`symm_mem.empty` + `rendezvous` + `multimem_all_reduce_`): every tracked
entry point resolves to a shim wrapper via
`cudaGetDriverEntryPointByVersion`, the fabric strip fires on torch's probe,
the multicast group is tracked (`track MC group`), and the multimem
all-reduce still produces correct results with a live `multicast_ptr`.

**gVisor e2e acceptance: PASS** (commit `06fbdce69`). SGLang TP=4
`--enable-torch-symm-mem --dtype bfloat16 --disable-custom-all-reduce`,
no torch.compile, `CB_IMEX=0`: checkpoint 45 s, restore + first inference
37.7 s (**3.6x**), post-restore answer EXACT MATCH, **0 fabric allocations**
in the sentry log, reproduced twice (plus once on the final binary).

Getting there surfaced a second, deeper blocker beyond shim invisibility:
on fabric-attached GPUs libcuda lazily creates **driver-internal
NV_MEMORY_FABRIC (00f8) FLA registrations** over VMM allocations shared
between processes (torch's symm-mem workspace; `map.hVidMem` identifies the
covered vidmem) even when the app requested plain POSIX-FD handles.
cuda-checkpoint *checkpoints* these but cannot *restore* them: the restored
process dies with `NV_ERR_OBJECT_NOT_FOUND` storms (measured; the strict
post-suspend gate now exists precisely to refuse such snapshots). The
trigger is `NV2080_CTRL_CMD_GET_GPU_FABRIC_PROBE_INFO` succeeding, so
nvproxy now gates that control (and the `nvidia-caps-imex-channels`
devices) on `fabric-imex-mgmt`. Denying the 00f8 class instead does NOT
work: after a successful probe, libcuda treats FLA-registration failure
during FD export as fatal (`CUDA driver error: unknown error`, measured).
Also fixed on the way: torch's `isFabricSupported` TORCH_CHECKs the
`nvmlDeviceGetGpuFabricInfoV` status and crashes on NVML error instead of
falling back -- the shim now smooths that into SUCCESS +
`state=NOT_SUPPORTED`, which torch handles.

**SUPERSEDED (2026-08-19 evening): IMEX is out of the picture entirely.**
The fabric-probe gating described below was reverted: the probe is back to
compUtil (upstream behavior), so **single-node NVLS, multimem, and engaged
FlashInfer fusion all work in DEFAULT sandboxes and all checkpoint/restore**
(FLA registrations are suspended by nvproxy at checkpoint,
capability-independent). `fabric-imex-mgmt` now gates only real IMEX
(multi-node channel devices) -- out of scope. `CB_IMEX` no longer affects
any validated behavior. Validated under default caps, TP=4, full C/R:
forced `--enable-nccl-nvls` PASS; fusion `trtllm` ENGAGED PASS; torch
symm-mem two-shot via the new opt-in `MCSHIM_HIDE_MULTICAST=1` PASS (the
opt-in exists because torch *multimem* remains the one un-checkpointable
multicast path -- the NVIDIA-side FLA-caching gap).

Historical record of the interim capability-knob design (superseded):

| Workload | `CB_IMEX=0` (no fabric cap) | `CB_IMEX=1` (fabric cap granted) |
| --- | --- | --- |
| SGLang `--enable-torch-symm-mem` TP=4 (bf16, no compile) | **PASS 3.6x**, 3 same-GPU runs + 1 cross-GPU (0-3 -> 4-7) | blocked by gate (1 persistent FLA/rank, by design) |
| SGLang `--enable-torch-symm-mem` TP=8, Qwen2.5-3B (the A/B-winning config) | **PASS 1.9x** | blocked by gate |
| vLLM TP=4 `VLLM_ALLREDUCE_USE_SYMM_MEM=1` | **PASS 9.8x** | untested |
| SGLang `--enable-nccl-nvls` TP=4 | fails at boot: NCCL treats forced NVLS + denied fabric probe as fatal (`unhandled cuda error`) | **PASS 13.1x** (672 transient 00f8 bind companions, all shim-released) |
| Everything else in the matrix (no fabric users) | PASS (vLLM TP=2 re-run: 14.5x) | PASS |

**Honest accounting of what `CB_IMEX=0` costs:** libcuda couples
`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` to the fabric probe, so denying
the probe disables ALL multicast -- not just fabric handles. Measured: the
raw 2-process multicast harness fails its attr-132 gate without the
capability (the earlier "NVLS multicast unaffected" claim was wrong; the
phase0 harness grants the capability). Framework fallbacks are graceful:
torch symm-mem selects its two-shot kernel (still symm-mem P2P, still most
of the win over pynccl), NCCL auto mode falls back to non-NVLS algorithms.
Only *forced* NVLS crashes.

Rules of thumb: leave `CB_IMEX=0` for checkpointable torch-symm-mem
workloads and do not force `--enable-nccl-nvls` there; grant the capability
for max-perf NVLS/multimem workloads, which remain checkpointable as long
as fabric objects stay transient or shim-released (NCCL NVLS yes; torch
symm-mem no -- its workspace keeps a persistent FLA registration).

Known flake (1 occurrence in ~12 e2e runs, pre-existing save-path race,
not introduced by these changes -- none of them touch mm/pma): checkpoint
fails with `Can't save pma with non-MemoryFile of type
*nvproxy.frontendFDMemmapFile`; the container unwinds cleanly and a retry
passes. Signature recorded here for recognition.

Remaining known-notworking: `--enable-torch-compile` + symm-mem is an
app-level SGLang/torch inductor bug (its fused kernel passes
`multicast_ptr` as a plain int, dynamo cannot trace it: `'int' object has
no attribute 'to'` -> inductor `xreplace` fatal during CUDA graph capture;
**reproduced natively without gVisor/shim**). The validated symm-mem
recipe matches the TP=8 A/B: no torch.compile.

**Measured cost of losing it** (A/B on this box, plain docker/runc since the
delta is GPU-side; SGLang TP=8, Qwen2.5-3B, `--disable-custom-all-reduce` in
both arms, bench_serving random 128/256, seed 42):

| Load | TPOT off | TPOT on | delta | throughput off -> on |
| --- | --- | --- | --- | --- |
| concurrency 1 | 2.60 ms | 2.19 ms | **-15.8%** | 370 -> 438 tok/s |
| concurrency 8 | 2.88 ms | 2.31 ms | **-19.8%** | 2610 -> 3197 tok/s |

Per-op that is ~6-8 us saved per all-reduce (72 per token on this model),
matching the multimem-vs-NCCL-LL estimate. Caveats: a 3B model at TP=8 is
maximally all-reduce-bound, so this is the upper end; the absolute ~0.4-0.6 ms
per token transfers to larger models, where it lands at roughly 4-7% of a
15-30 ms TPOT. Also note the eligibility conditions found while validating the
A/B: this SGLang's symm-mem path engages only for **bfloat16** models (an
earlier fp16 A/B produced identical arms because every tensor failed the dtype
check) and only within the (sm, world) support table {sm90/sm100 x 2,4,6,8}.

Alternatives tried: `TORCH_SYMMMEM=NCCL` (route symm-mem through NCCL windows,
which resolve dynamically and use fds) -- rejected by SGLang's own usage:
`NCCLSymmetricMemoryAllocator::alloc must not be called with a group_name`,
scheduler crash at startup. So on this SGLang+torch combination the option is
simply unsupported for C/R; use `--enable-nccl-nvls` (validated, including
cross-GPU) without `--enable-torch-symm-mem`. Note vLLM's symmetric-memory
all-reduce (`VLLM_ALLREDUCE_USE_SYMM_MEM=1`) remains fully supported -- its
torch build allocates through interposable paths with fd handles.

### FlashInfer validation ladder (2026-08-19)

All runs: stock SGLang flags, no torch.compile, `MCSHIM_IPC_SUSPEND=1`.

| Stage | Config | Verdict |
| --- | --- | --- |
| S1 | TP=4 same-GPU, `--attention-backend flashinfer --sampling-backend flashinfer`, `CB_IMEX=0` | **PASS 7.0x** (cold boot 299.6s, checkpoint 61.7s, first inference 42.5s) |
| S2 | S1 + cross-GPU restore 0-3 -> 4-7 | **PASS** |
| S3 | S1 + `--enable-flashinfer-allreduce-fusion` | **PASS, but fusion vacuous** -- flag is a deprecated alias; the workspace preflight (`cuMulticastGetGranularity`) gets NOT_SUPPORTED under `CB_IMEX=0` and SGLang logs "Skipping allreduce fusion" and falls back cleanly |
| S4 | S3 at TP=8 (Qwen2.5-3B, custom-AR off) | **PASS, fusion vacuous** (same graceful skip) |
| S5 | TP=4, `--flashinfer-allreduce-fusion-backend trtllm`, `CB_IMEX=1` -- fusion **engaged** (workspace MC group tracked) | **checkpoint refused**: 1 fabric object per rank survives shim suspend (252 f8 created, 248 released, 4 live) |
| S6 | S5 at TP=8 | **checkpoint refused**: same signature (632 created, 624 released, 8 live -- exactly 1/rank) |

Key findings:

1. **FlashInfer attention + sampling are fully supported** under C/R,
   same-GPU and cross-GPU. The old "needs nvcc at runtime" concern is moot
   on this image (flashinfer 0.6.15, sm90 AOT kernels).
2. **This SGLang lineage's all-reduce-fusion workspace is multicast/VMM,
   not legacy IPC** (`cuMulticastGetGranularity` preflight; the shim tracks
   its MC group like any other). The feared TP=8 legacy-IPC reopen collapse
   does not apply to it.
3. Fusion under `CB_IMEX=0` **declines gracefully** -- correct fallback, no
   crash (unlike forced NVLS). A checkpointable deployment can simply leave
   fusion requested; it self-disables.
4. Engaged fusion is blocked by **exactly the torch-multimem blocker**: one
   persistent owner-side NV_MEMORY_FABRIC FLA registration per rank over
   the fusion workspace, surviving the shim's group/bind/import teardown.
   Both remaining gaps are the same object class with the same fix surface.

### FLA registration suspend/replay (implemented same day)

nvproxy now host-frees FLA registrations inside the cuda-checkpoint suspend
window and replays them where the process's driver state survives (unwind /
resume-after-save); after a true restore libcuda lazily re-registers on the
next export (measured), so afterLoad drops the record. Iteration history
that matters for future work -- each variant was run e2e and failed for a
measured reason: replay bookkeeping attached to graph objects is cascaded
away by cuda-checkpoint's own frees before state.Save; foreign-fd RM calls
fail INVALID_CLIENT; app-closed fds fail ENOTTY/EBADF; post-toggle the
rebuilt graph uses FRESH handles, so identity replay after true restore is
impossible (and unnecessary).

| Config (all `CB_IMEX=1`, fabric granted) | Before | After |
| --- | --- | --- |
| SGLang TP=4 `--flashinfer-allreduce-fusion-backend trtllm` | checkpoint refused (1 fabric/rank) | **PASS end-to-end** (fusion re-engages post-restore) |
| SGLang TP=4 `--enable-torch-symm-mem` (multimem) | checkpoint refused | checkpoint + FLA suspend pass; **restore fails, root-caused as NVIDIA-side**: torch's userspace caches the FLA registration handle across exports, so its post-restore re-export retries EXPORT_OBJECT_TO_FD against the stale handle into OBJECT_NOT_FOUND. Sentry-side recreation is impossible (toggle rebuilds the client via debugger paths invisible to nvproxy; sentry RM_ALLOC into it fails NV_ERR_INSUFFICIENT_PERMISSIONS regardless of carrier fd -- probed exhaustively). Fix requires cuda-checkpoint to serialize or reconcile FLA registrations. Workaround unchanged: run torch symm-mem under `CB_IMEX=0` (two-shot kernel, PASS) |
| phase0 harness + no-fabric matrix | PASS | PASS (unaffected) |

Not validated (no claims): SGLang mscclpp, pipeline parallelism, MoE/EP,
quantized paths, models beyond Qwen2.5 1.5B/3B.
Out of scope by design: chained cross-GPU restores (panics loudly), R610
(deferred; its cuda-checkpoint job would remove the legacy-IPC teardown
entirely and with it the TP=8 custom-AR restriction).

## Results

### Interposer-level (phase0 harnesses)

| Test | World | Move | Result |
| --- | --- | --- | --- |
| `run_mcshim_mp_native.py` | 2 | -- | PASS |
| `run_mcshim_mp_native.py` | 4 | -- | PASS |
| `run_mcshim_mp_native.py` | 8 | -- | PASS |
| `run_mcshim_mp_gvisor.sh` | 2 | same-GPU | PASS |
| `run_mcshim_mp_gvisor.sh` | 2 | 0,1 -> 6,7 | PASS (placement asserted) |
| `run_mcshim_mp_gvisor.sh` | 4 | 0-3 -> 4-7 | PASS (placement asserted) |

All runs: helper proxy exercised on every rank, multicast VA re-mapped
`IDENTICAL`, 0 verification failures.

### Engine-level (cr-bench)

Config for every run: NVLS + symmetric memory + custom all-reduce **enabled**,
`torch.compile` + CUDA graphs (no `--enforce-eager`), sleep/checkpoint/restore/
wake_up lifecycle.

| Workload | TP | Move | Cold boot | Restore | 1st inference | Speedup | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vLLM | 2 | same | 219.4 s | 4.09 s | 14.9 s | **14.7x** | PASS |
| vLLM | 4 | same | 320.6 s | 6.05 s | 25.9 s | **12.4x** | PASS |
| vLLM trials (n=3) | 4 | same | -- | -- | -- | -- | **3/3 PASS, 0 toggle failures** |
| SGLang | 2 | same | 560.2 s | 9.51 s | 22.5 s | **24.9x** | PASS |
| SGLang | 4 | same | 627.5 s | 17.03 s | 45.2 s | **13.9x** | PASS |
| SGLang | 2 | 0,1 -> 6,7 | -- | -- | 22.5 s | -- | **PASS** |
| SGLang `--enable-nccl-nvls` | 4 | same | 631.0 s | -- | 16.55 s | 47.5 s | **PASS** |
| SGLang `--enable-nccl-nvls` | 4 | 0-3 -> 4-7 | -- | -- | 17.33 s | 48.7 s | **PASS**, placement verified |
| SGLang `--enable-torch-symm-mem` | 4 | same | 625.7 s | -- | -- | -- | **BLOCKED by design** (fabric objects; see below) |
| vLLM trials (n=3) | 2 | 0,1 -> 6,7 | -- | -- | -- | -- | **3/3 PASS, 0 toggle failures** (reviewed code) |
| SGLang | 4 | 0-3 -> 4-7 | 628.4 s | 60.6 s | 16.84 s | 46.4 s | **PASS**, placement verified |
| vLLM (full-cycle record) | 2 | 0,1 -> 6,7 | 216.4 s | 13.4 s / 9.5 G | 3.82 s | 14.8 s | **PASS**, 14.6x, placement verified |
| vLLM (Qwen2.5-3B, no custom AR) | 8 | 0-7 -> same | 325.6 s | 30 G | 11.89 s | 69.8 s | **PASS**, 4.7x |
| SGLang (Qwen2.5-3B, no custom AR) | 8 | 0-7 -> same | 687.4 s | 160.9 s | 19.6 s | 84.7 s | **PASS**, 8.1x |
| phase0 gVisor harness | W=8 | 0-7 -> same | -- | -- | -- | -- | **PASS**, all 8 UUIDs asserted |

TP=8 notes: Qwen2.5-1.5B (12 attention heads) cannot shard 8 ways, so TP=8 uses
Qwen2.5-3B-Instruct (16 q / 2 kv heads) baked into rebuilt images. Two findings
at TP=8, neither a C/R regression:

1. **vLLM TP=8 with custom all-reduce enabled fails to resume**: each rank
   holds 28 legacy cuIpcOpenMemHandle imports and 21 reopened at the wrong VA
   (the API takes no address hint; the shim's hole-plug walk-back copes at
   TP=2/4 scale but not 8x28). The shim refused loudly rather than corrupt
   CUDA graphs. With `--disable-custom-all-reduce` (NVLS carries the
   collectives) TP=8 passes. Known limitation: custom AR + TP=8.
2. **SGLang TP=8 needed three environment accommodations**: its own JIT kernel
   (`kernels/jit/csrc/distributed/ipc.cuh`, a tvm::ffi compile error at
   world=8) is avoided by `--disable-custom-all-reduce`; its 8-rank
   compile/autotune cold boot exceeds the bench's 600 s health window
   (`HEALTH_TIMEOUT=1500`); and one TP=8 rank tracks >512 shim objects, which
   tripped the interposer's sticky overflow guard exactly as designed --
   fixed by raising MAXN to 4096.
| vLLM | 2 | 0,1 -> 6,7 | -- | 3.70 s | 14.9 s | -- | **PASS** (after `72267d698`) |
| vLLM | 4 | 0-3 -> 4-7 | -- | -- | 26.0 s | -- | **PASS** |
| vLLM | 2 | same (regression) | -- | -- | 14.9 s | 14.6x | PASS |

Checkpoint times / image sizes: vLLM TP=2 13.3 s / 9.5 G, TP=4 15 G, SGLang TP=2
24.8 s / 24 G, SGLang TP=4 60.6 s / 44 G. All passing runs answered all 3
verification queries correctly.

The vLLM TP=4 trial count matches the 580.126.20 reference (3/3, 0 toggle
failures), and SGLang's 24.9x is the largest speedup measured, because its cold
boot is the longest (560 s).

### Cross-GPU at the engine level: fixed (`72267d698`)

This was blocked, then root-caused and fixed. The original symptom and the
diagnosis are kept below because the shape of the bug is instructive.

**Fix:** `NV0000_CTRL_CMD_OS_UNIX_GET_EXPORT_OBJECT_INFO` returns a
`DeviceInstance` output. RM truthfully reports the device instance now backing
the exported object (6 or 7 after a move), while a restored process's libcuda
only knows the instances it enumerated before the checkpoint (0 and 1), so its
lookup fails and it returns `CUDA_ERROR_INVALID_DEVICE` -- without any ioctl or
RM status ever failing. nvproxy now translates that output back to the instance
the application saw, exactly as it already remaps `DeviceInstance` in
`NV01_DEVICE_0`'s allocation parameters.

Result on vLLM TP=2, GPUs 0,1 -> 6,7:

```
nvproxy: GET_EXPORT_OBJECT_INFO: reporting device instance 6 as 0   (x100)
nvproxy: GET_EXPORT_OBJECT_INFO: reporting device instance 7 as 1   (x100)
vLLM wake_up -> {"status":"ok","sleeping":false}
First inference: 14889 ms after restore -> "Paris"
Restored on GPUs: 6,7 (placement verified: YES)
RESULT: PASS
```

How it was found, in case a similar bug appears: the failing import was
surrounded by controls that all succeeded, so the first useful step was proving
the driver never rejected anything. `rmControlInvoke` now logs a nonzero RM
`Status` at Debug -- a control can succeed at the ioctl level and still be
rejected by RM, and that gap is what made this look like a driver limitation.
Once the log showed no rejection anywhere, the cause had to be userspace-side
validation of a value RM had reported, which pointed straight at the
`DeviceInstance` output.

### Original symptom and diagnosis (pre-fix)

Cross-GPU restore works for the **multicast** layer (phase0 harness, W=2 and
W=4, placement asserted) but vLLM TP=2 restored onto GPUs 6,7 fails:

```
[mcshim] RESUME: phase1 done (3 MC creators, 50 UC exporters served, 10 IPC exporters served)
[mcshim] track IMPORT idx=0 ... import idx=0 classified as MC group    <- multicast OK
[mcshim] RESUME: re-import idx=164 rc=101 after 100 attempts           <- unicast FAILS
```

`rc=101` is `CUDA_ERROR_INVALID_DEVICE`. The multicast group re-imports fine
through the helper, then a **unicast VMM peer buffer** re-import fails, wake_up
returns FAILED and the container exits.

This is consistent with the R580 device-admission wall: a restored process
cannot admit devices it did not hold at checkpoint, and after a cross-GPU
restore its physical GPUs are exactly that. Multicast escapes it because
create+attach can be proxied through a never-checkpointed helper; a unicast
import cannot, because the resulting handle must live in the rank.

The phase0 multicast harness does not exercise this path -- its resume logs
show `imports=0` -- which is why it passes cross-GPU while vLLM does not.

Prediction to test on R610: cross-GPU should work there, since R610 has no
device-admission wall. If so `tools/mcshim/README.md`'s "pre-R610 gap" is
correct after all, though for a narrower reason than stated (unicast VMM peer
imports specifically, not imports in general).

Reproduced twice, before and after the deadlock fix below, failing at the same
allocation index (164) on both ranks -- so it is a property of the workload's
unicast peer sharing, not a timing artifact.

And it is the whole path, not one allocation: rank 682's 50 imports occupy
tracking indices 164-213, so **164 is the first import attempted**. The rebuild
aborts on the very first unicast peer import rather than failing on some
particular buffer, which is what the device-admission explanation predicts.

**Status: on R580, both same-GPU restore and GPU relocation work for vLLM at
TP=2/TP=4 and SGLang at TP=2/TP=4, with NVLS, symmetric memory and custom
all-reduce enabled and compile + CUDA graphs preserved.** The earlier conclusion
that relocation needed R610 was wrong; it needed two nvproxy fixes.

## Caution for reviewers: a restore-path deadlock, now fixed

The first version of the device-namespace fix recorded the minor translation
under `fdsMu`. `afterLoad()` calls `frontendFD.load()` for every FD *while
holding* `fdsMu`, so that self-deadlocked the restore -- it hung after
`All MemoryFile pages have been loaded` with no error, no timeout and no output.

It did not reproduce on vLLM TP=2 or TP=4 (five successful restores) because
whether it deadlocks depends on the FD count and which path records the
translation first. SGLang TP=2 hung for 46 minutes, and only a `SIGQUIT` stack
dump identified it. Fixed in `1a6cfb2d9` with a dedicated mutex.

Two lessons worth carrying: a hang in this path produces no diagnostic at all,
and passing vLLM runs are not sufficient evidence that a restore-path change is
safe.

## Verification hygiene

The phase0 gVisor harness originally reported PASS purely on collective
correctness, which a restore that never moved satisfies trivially -- that is how
the device-namespace bug stayed hidden. It now asserts that every rank's
`cuDeviceGetUuid` is one of the restore-target GPUs.

A correction to an earlier claim in this log: I also believed `cr-bench` never
verified placement, having grepped only `_bench_vllm_impl.sh` for
`$PLACEMENT_OK` and found it printed and gating the verdict but never assigned.
That was wrong -- `common.sh:843` already compares memory in use on the restore
set against the original set and clears `PLACEMENT_OK`, and it correctly failed
the cross-GPU vLLM run above. A redundant second check was added and then
reverted.

That also weakens the suspicion that `HANDOFF.md`'s R610 cross-GPU result was an
illusion: the existing check would have caught memory staying on the original
GPUs. It should still be re-run on R610 with the current binaries, but it is no
longer presumed wrong.

## Reproduce

```sh
# interposer-level, fast
sudo RUNSC=/usr/local/bin/runsc-r580 WORLD=4 GPUS=0,1,2,3 RESTORE_GPUS=4,5,6,7 \
  bash gpu_mem_snapshots/phase0/run_mcshim_mp_gvisor.sh

# engine-level
sudo RUNSC=/usr/local/bin/runsc-r580 CUDA_MULTICAST_SHIM=1 \
  CUDA_MULTICAST_SHIM_SRC=$PWD/tools/mcshim/mcshim.so \
  CUDA_CKPT_JOB_FILE=0 CUDA_CKPT_SEQUENTIAL=1 NCCL_CUMEM_ENABLE=1 \
  MCSHIM_IPC_SUSPEND=1 REBUILD_ROOTFS=1 \
  bash cr-bench/bench_4_vllm_multi.sh --gpus 0,1 --tp 2
```

As of 2026-08-18 the harnesses request `fabric-imex-mgmt` by name
(`--nvproxy-allowed-driver-capabilities=all,fabric-imex-mgmt`; opt out with
`CB_IMEX=0`). The capability is privileged and excluded from `all`. Verified:
runsc boots with it on this host, phase0 cross-GPU C/R passes, vLLM TP=2 passes
at 14.5x. It does not change checkpointability either way: single-node
fabric-handle allocation already worked without it, and fabric objects remain
refused by the blocker gate.

`REBUILD_ROOTFS=1` is needed only on the first run per image. Images
(`cr-bench-vllm`, `cr-bench-sglang`) are built from `cr-bench/images/` and
pre-download the Qwen models, so `HF_HUB_OFFLINE=1` at runtime is satisfied.
