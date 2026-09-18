# PROGRESS: multi-GPU CUDA checkpoint/restore under gVisor (R580)

Successor to `gpu_mem_snapshots/PROGRESS.md` (the H100 + R610 investigation).
Records the state of the R580 effort as of 2026-08-24 (last active session);
written 2026-09-14 after an environment reset. Branch of record:
**`luis/r580-multicast-snapshots`** (origin = github.com/luiscape/gvisor).

## 2026-09-14 session: NEW_UPDATES.md execution (UNCOMMITTED working tree)

All work below is in the working tree only (standing rule: no commits/pushes);
the single commit made is the merge of `origin/master` (`e606870d3`).

### Environment (fresh node, kernel 7.0.0-aws)
- Driver 580.173.02 + FM 580.173.02 (from redist tarball), CTK 1.20, Docker 29.
- **runsc install layout changed by upstream**: sidecars are mandatory
  (`gvisor_sentry`, `checkpointgofer`, prewarmer, fd-parking under
  `gvisor-bin/` next to runsc). Build `make build TARGETS=//:release`, install
  `bazel-bin/release/{runsc,gvisor-bin/}` to `/usr/local/lib/runsc-r580/`,
  symlink `/usr/local/bin/runsc-r580`. A bare `runsc` copy no longer boots.
- Harness rebuilt: `cr-bench/{common.sh,_bench_*_impl.sh,bench_4_vllm_multi.sh,
  bench_6_sglang_multi.sh,build_images.sh,images/}` (git-excluded). Rootfs
  trees at `/data/cr-bench/rootfs-cr-bench-{vllm,sglang}` + `.env` (image
  env captured by build_images.sh: SGLang lives in a venv at `/opt/sglang`).
  Persistent compile caches at `/data/cr-bench/cache-<image>` (mounted at
  `/cache`; SGLang TP=4 torch.compile autotune alone exceeds 600 s cold).
  `--network=none` is required (a plain private netns leaves `lo` DOWN ->
  Gloo "Cannot resolve 127.0.0.1"). `date +%s%3N` is broken on uutils.
- Trial driver: `/tmp/trial.sh <name> <vllm|sglang> [ENV=V..] -- <args>`
  (logs `/tmp/trials/<name>.log`); gate: `e2e_gate.sh`.

### NEW_UPDATES.md status
| # | Status |
| --- | --- |
| U1 | done (loader pins `NCCL_CUMEM_ENABLE=1`; shim `ipc.<pid>` + notice; sentry logs per-rank counts) |
| U2 | done (default on; `MCSHIM_KEEP_UC_EXPORTS=1` opt-out) |
| U3 | done differently: pma panic fixed earlier; residual root cause = late CUDA init -> **admission gate** (the post-action straggler guard was removed as redundant); all cudaProcs FD tables walked |
| U4 | done: pause geometry correct (vLLM/SGLang TP=2 PASS); fast variant = SGLang `--quiesce kv`; carry cost measured ~1 s/GB |
| U5 | measured, closed (~70 ms/GB restore) |
| U6 | measured, closed (1.5 s) |
| U7 | code done (MAX_AKA 16, eviction logged + suspend refused); churn stress not run |
| U8 | dropped: promotion made the legacy-import budget moot (removed) |
| U9 | DP=2/TP=2 (same + cross-GPU), PP=2/TP=2, TP=8, FP8 all PASS; MoE/EP and 30B not run (need model downloads / long boots) |
| U10 | done (cuMemSetAccess desc[], MAXN table named, cudaProcs, export slots, ld.so.preload on restore, --device-map note) |
| U11 | not started |

### Next
1. MoE/EP cell (Qwen1.5-MoE-A2.7B or Mixtral class; ~30 GB download) and a
   30B-class TP=4 cell -- both need `build_images.sh` model additions and a
   longer `HEALTH_TIMEOUT`.
2. Decide whether the fusion regression (SGLang 0.5.19 / flashinfer 0.6.18)
   deserves a pinned-image cell in the gate or is simply documented.
3. U7 churn stress (sleep/wake x N, then one C/R) if sleep/wake cycling is a
   deployment pattern.
4. U11 spike (tmpfs-backed host weights) only if release-quiesce images of
   large models become the bottleneck.

### Upstream merge accommodations
- `--cuda-multicast-shim-embedded` (bool) -> `--cuda-multicast-shim-source=
  IMAGE|EMBEDDED` (upstream now forbids new bool flags; `config_test` passes).
- `//pkg/usermem` re-added to `runsc/boot/BUILD`.

### Items done (each validated by the trials below)
- **U3' straggler guard**: `nvproxy.LiveObjects` (management classes
  excluded) checked after the checkpoint action -> unwind + retryable error;
  gate matcher extended. Unit tests in `checkpoint_blockers_test.go`.
- **U1**: loader pins `NCCL_CUMEM_ENABLE=1`; shim publishes `ipc.<pid>` (live
  legacy-import count) + one-time notice; sentry logs per-rank counts.
- **U8 (partial)**: `MCSHIM_IPC_MAX_IMPORTS` budget -> `warn.<pid>` -> refusal
  in preSaveCuda before any lock. **Opt-in (default off)**: measured counts
  are a weak proxy (vLLM TP=4: 2/rank; SGLang TP=4: 30/rank live, 25 peak
  during boot -- all replay fine). Marker tracks the LIVE count, not the peak.
- **U2**: free-UC-exports default on; `MCSHIM_KEEP_UC_EXPORTS=1` opts out.
- **U10**: cuMemSetAccess records all desc[] over range-covered mappings and
  replays all; per-slot export accounting (`exportedObjs map[slot]`); cudaProcs
  walks all tasks' FD tables; overflow refusal names the table; shim files +
  `/etc/ld.so.preload` reinstated after restore (`reinstateCudaMulticastShim`,
  under `l.mu` -- do NOT call `GetContainerSpecs` there); `--device-map` FIXME
  -> explanatory comment.
- **U7 (code)**: `MAX_AKA` 16; overflow keeps the original handle, evicts the
  oldest rotated one, logs, sets sticky `g_alias_overflow` -> suspend refused.

### Trials on the merged build (580.173.02, Qwen2.5-1.5B)
| Cell | Result | ckpt / restore / first-infer | Notes |
| --- | --- | --- | --- |
| vLLM TP=2 same-GPU (release) | PASS 22.4x | 14.1 s (11G) / 4.4 s / 10.2 s | uc_freed=3/rank by default |
| vLLM TP=4 same-GPU | PASS | gate | |
| vLLM TP=2 -> GPUs 4,5 | PASS | gate | |
| SGLang TP=4 `--enable-nccl-nvls` | PASS 16.2x | 46 s (20G) / 7.7 s / 23 s | 30 legacy imports/rank; SGLang 0.5.19 enables symm-mem multicast all-reduce by default |
| **vLLM TP=2 pause quiesce** (weights+KV resident) | **PASS** 3.0x | 137 s (16G) / 6.2 s / 105 s | first time this geometry works; 10 FLA regs host-freed, lazy re-register OK |
| SGLang TP=4 fusion `trtllm` (compile on or off) | **FAIL at resume** | 31 s (19G) / 7.3 s / - | 12 of 21 legacy IPC imports/rank do not return to their VA (see below) |
| SGLang TP=4 fusion + `--disable-custom-all-reduce` | PASS 6.1x (fusion vacuous: gated on custom AR) | 32 s (19G) / 7.2 s / 23 s | 0 legacy imports |
| SGLang TP=4 stock (`--no-torch-compile`) | PASS | | 30 legacy imports/rank, all replayed |
| SGLang TP=2 pause quiesce (GPUs 2,3) | PASS 5.2x | 144 s (13G) / 4.9 s / 108 s | same KV-copy cost profile as vLLM pause |
| vLLM TP=2 + deterministic straggler (`CB_STRAGGLER=1`) | PASS | 14.4 s | admission gate held the straggler 21.8 s; it initialized after restore |
| **Gate re-baseline** (vllm_tp4, vllm_tp2_xgpu, sglang_tp4_nvls, sglang_tp4_stock, vllm_tp2_straggler) | **5/5 PASS** | | `/data/e2e_gate_20260915_161628` |
| U9: vLLM DP=2 TP=2 (2 engine cores, 4 ranks) | PASS 12.6x | 29 s (21G) / 8.1 s / 21 s | no code change needed |
| U9: vLLM DP=2 TP=2 -> GPUs 4-7 | PASS | 29 s / - / 22 s | placement 4,5,6,7 |
| U9: vLLM PP=2 TP=2 | PASS | 33 s (16G) / - / 27 s | P2P send/recv comms |
| vLLM TP=8 (3B, CAR off), all 8 GPUs | PASS | 53 s (32G) / - / 45 s | |
| U4: SGLang TP=4 `--quiesce kv` (KV freed, weights resident) | PASS | 33 s (18G) / 6.8 s / 23.5 s | 4.5 GB resident/GPU; FLA suspend n>0, lazy re-register OK |
| U4: SGLang TP=4 `--quiesce kv` -> GPUs 4-7 | PASS | 33 s / 6.8 s / 23.8 s | |
| U9: vLLM TP=2 `--quantization fp8` (dynamic FP8) -> GPUs 2,3 | PASS | 16 s (9.8G) / - / 11.6 s | FP8 kernels + workspaces |
| U4 reference: SGLang TP=4 `--quiesce release` | PASS | 30 s (18G) / 6.9 s / 23.3 s | at 1.5B the two geometries tie; kv wins only when weights are large enough that D2H+H2D (release) exceeds cuda-checkpoint's ~1 s/GB carry |

### U5 / U6 measured -> closed as not material
- **U5 pinned host memory** (native, GPU 7, 2 GB device + N GB `pin_memory`
  host): checkpoint 5.39 / 5.60 / 5.77 s and restore 3.60 / 4.38 / 4.75 s for
  0 / 8 / 16 GB pinned -> **~25 ms/GB checkpoint, ~70 ms/GB restore**. For
  the validated configs (1.5-3 GB offloaded per rank) that is <0.3 s per
  rank; a 30B TP=4 model (~15 GB/rank) would pay ~1 s/rank at restore. No
  host-registration interposition warranted.
- **cuda-checkpoint device copy rate** (native, 60 GB resident): checkpoint
  60.5 s, restore 31.8 s -> ~1 GB/s out, ~2 GB/s in. Under gVisor the
  pause-quiesce vLLM TP=2 (2 x 60 GB, sequential) did 1.26 / 1.17 GB/s. So
  keeping the KV pool resident costs ~1 s/GB each way: the pause geometry
  can only be fast if the engine frees KV (SGLang `tags=["kv_cache"]`,
  harness `--quiesce kv`).
- **U6 shim resume**: measured 1.4-1.5 s end-to-end at 99-117 imports/rank
  (vLLM TP=2, SGLang TP=4 NVLS), vs 8-20 s for the restore toggle and 10-20 s
  for the engine's wake-up H2D. Not worth the concurrency; skipped.

### New: CUDA admission gate (`nvproxy/cuda_admission.go`)
The straggler guard fired for real (SGLang fusion: 4 late `mcshim-helper`-like
processes with 33 RM objects each, initializing CUDA 1.5 s after the process
set was collected). Rather than only refusing, `preSaveCuda` now closes an
admission gate before collecting the set: a process's first RM root-client
allocation blocks (interruptible, `ERESTARTNOINTR`) until the sequence ends;
sentry-exec'd processes and the collected set are exempt. A held process has
no GPU state, saves as a sleeping syscall, and initializes against the
restored devices afterwards. Validated with `CB_STRAGGLER=1` (harness spawns
a process that calls `cuInit` the instant the shim `gate` marker appears).
The post-action guard stays as the backstop.

### 2026-09-17: simplification pass + driver-version finding
Removed (redundant or dead after promotion): the post-action straggler
guard (`LiveObjects`, `managementClasses`; the admission gate covers the
race), the U8 legacy-import budget and `ipc.<pid>`/`warn.<pid>` markers
(`checkCudaShimLegacyIPC`), the `MCSHIM_IPC_PROMOTE_FLOOR` diagnostic, the
`reinstateCudaMulticastShim` restore hook and the loader refactor around it
(the overlay carries the files; the hook only mattered for a rootfs that
lost them), and `undoCheckpoint` (single caller). Loader delta is now the
`NCCL_CUMEM_ENABLE=1` pin and the flag rename only.

**Cross-GPU restore on R610 exposed a driver-dependent behaviour**: the
`GET_EXPORT_OBJECT_INFO` device-instance translation (needed on 580, where
libcuda resolves the instance against its pre-restore table) breaks 610
(libcuda now resolves against current state; translating hands it a stale
instance -> `CUDA_ERROR_INVALID_DEVICE` on re-import). Measured all four
cells (580 on/off = PASS/FAIL, 610 on/off = FAIL/PASS); the translation is
now gated to `Major() < 610`. 590/595 untested.

Validation on the simplified tree (610, `shmem_enabled=always`): gate 6/6
after the fix (same-GPU cells passed before it), SGLang MoE TP=4 PASS,
vLLM TP=8 custom-AR PASS, failed-save recovery PASS.

### BREAKTHROUGH (2026-09-16): legacy CUDA IPC promoted to VMM IPC
Root cause of every legacy-IPC restore failure, measured with
`gpu_mem_snapshots/probes/ipc_reopen_probe.py` (native, 2 procs): the
driver's legacy VA allocator is **top-down first-fit and refuses an
exact-fit hole**, so an import whose neighbours abut it (32 MiB at a 32 MiB
stride; 2 MiB runs) can never be reopened at its VA. Per-import holds bring
back only imports with slack below; a "release all holds, plug new holes"
replay (tried as `MCSHIM_IPC_REPLAY=replay`, since removed) works in the
probe but not in the engines (transient allocations at original-open time).
No replay strategy can work in general.

So the shim now **sidesteps the allocator**: at `cuIpcGetMemHandle` the
exporter's `cudaMalloc`'d buffer is replaced in place by a `cuMemCreate`'d
one at the same VA (D2H, free, `cuMemAddressReserve(base)`, create, map,
H2D -- measured to be exact and cuda-checkpoint-safe with
`ipc_migrate_probe.py` / `promote_ckpt_probe.py`), exported as a POSIX fd
and served on the rendezvous socket; the blob returned to the app names the
fd (`MCSHIMP1` magic). The importer's `cuIpcOpenMemHandle` recognises it,
fetches, imports, maps. Everything downstream is the VMM path the shim
already checkpoints exactly (unmap keeping the reservation; remap at resume).
Three details that mattered: (1) `cuMemRetainAllocationHandle` must report
promoted ranges as non-VMM (SGLang's custom AR v2 probes it per buffer set
to choose its registration path; a mixed answer crashed boot); (2) the shim's
serve threads must be joinable -- `shutdown()` does not wake `accept()` under
gVisor, so the threads' dup of the export fd stayed open and blocked the
checkpoint; now `poll()` + stop pipe + `pthread_join`; (3) sub-2 MiB buffers
(vLLM custom-AR metadata, 0x41300 bytes, in the low arena) are
granule-rounded rather than left legacy. The sentry no longer gates on
exported fds when the shim is present (it closes them all at suspend; the
strict post-suspend gate still catches app-held ones). Default ON
(`MCSHIM_IPC_PROMOTE=0` restores the old behaviour).

| With promotion | Result | ckpt / first-infer |
| --- | --- | --- |
| SGLang TP=4 stock (38 promoted exports/rank, 0 legacy) | PASS | 30 s / 23.6 s |
| **SGLang MoE TP=4** (was FAIL) | **PASS** | 56 s (49G) / 28 s |
| **SGLang TP=4 FlashInfer trtllm fusion** (was FAIL) | **PASS** | 31 s / 24 s |
| vLLM TP=4 | PASS | 24 s / 19 s |
| **vLLM TP=8, custom all-reduce ON** (was "unfixable by construction") | **PASS** | 54 s (32G) / 44 s |
| vLLM TP=2, SGLang TP=2 | PASS (no legacy IPC at TP=2) | |
| **Gate re-run with promotion** (vllm_tp4, vllm_tp2_xgpu, sglang_tp4_nvls, sglang_tp4_stock, **sglang_tp4_fusion -> GPUs 4-7**, straggler) | **6/6 PASS** | `/data/e2e_gate_20260916_*` |

With this, **custom all-reduce is supported in every tested configuration**
(vLLM TP=2/4/8, SGLang TP=2/4 incl. MoE and FlashInfer fusion, same-GPU and
cross-GPU). The `--disable-custom-all-reduce` recipe and the U8 import budget
are no longer needed for supportability (budget remains opt-in diagnostic).
The legacy close/replay code path remains only for `MCSHIM_IPC_PROMOTE=0`
and for the (unobserved) case where promotion falls back. **Decision
(2026-09-16): keep it as a safety net** until promotion has more mileage;
revisit removal in a later cleanup. The `MCSHIM_IPC_REPLAY` experiment was
removed (regressed the engines); fallback regated PASS after removal.

### Regression: SGLang 0.5.19 + FlashInfer 0.6.18 trtllm all-reduce fusion -- RESOLVED by promotion above; history kept:
The August PASS was on flashinfer 0.6.15. Now, with fusion engaged, each
rank holds 21 legacy IPC imports (vs 30 in the stock config, which all
replay); 12 of them reopen ~15-32 GiB ABOVE their original VA, in a fresh
descending arena, deterministically, regardless of: torch.compile on/off,
`MCSHIM_KEEP_UC_EXPORTS`, holding the full power-of-two slot instead of the
exact range (made it worse: 16 moved), or freeing all holds up front (all 21
moved -- confirms the per-import hold IS what brings the other 9 back).
The 12 = 3 peers x 4 buffers of the fusion configuration. Hole-plugging
cannot help (it only handles imports that land BELOW their target). This is
the known `cuIpcOpenMemHandle`-has-no-address-hint limit surfacing in a new
configuration; the mitigation is the same as TP=8 custom AR: run fusion
with `--disable-custom-all-reduce` (then fusion is vacuous) or without the
trtllm backend. Documented as unsupported for this image lineage.

### Host incident: fabric manager restart
`apt-daily-upgrade` restarted `nvidia-fabricmanager` at 06:24 (systemd
re-exec), orphaning GPU registrations: every `cuMulticastBindMem` then
failed with CUDA 401 natively (no gVisor). FM log: "failed to find the member
GPU handle ... All GPUs in the partition need to be reset". Fix: stop FM,
`nvidia-smi -r`, start FM. `apt-daily{,-upgrade}.timer` are now disabled.
Symptom to recognise: NCCL `Failed to bind NVLink SHARP (NVLS) Multicast
memory ... 401` at boot, in every container, native included.

**U4 finding**: pause quiesce is now *correct* but not *fast*: cuda-checkpoint
moves the resident KV pool (60 GB/GPU at util 0.7) at ~1.2 GB/s in both
directions (95 s checkpoint action, 103 s restore toggle); the image stays
small (16G) only because `--exclude-committed-zero-pages` drops the zero KV
pages. The fast geometry needs the engine to free the KV pool but keep
weights (SGLang `release_memory_occupation(tags=["kv_cache"])`; vLLM has no
such level) -- and/or the cuda-checkpoint copy rate under gVisor investigated.

## Goal

`runsc checkpoint` / `runsc restore` of single-node multi-GPU (TP=2-8)
inference containers (vLLM, SGLang) on 8x H100 + NVSwitch, driver
**580.173.02**, preserving CUDA graphs / torch.compile state, including
restore onto DIFFERENT GPUs. Workflow: boot -> warm -> `/sleep` ->
checkpoint -> restore -> `/wake_up` -> serve.

## Architecture (what ships on the branch)

Two cooperating layers. The sentry orchestrates; a container-side
`LD_PRELOAD` interposer executes the parts that must run through libcuda.

| Layer | Piece | Job |
| --- | --- | --- |
| runsc/loader | `--cuda-multicast-shim-path` | LD_PRELOAD + `/etc/ld.so.preload` injection (launchers rewrite LD_PRELOAD); sets `MCSHIM_MC_PROXY`, `MCSHIM_IPC_SUSPEND`, `MCSHIM_IPC_REPLAY_FLOOR=0`, `MCSHIM_HELPER` |
| sentry | `control/state_cuda.go` | gate -> parallel lock -> unlock -> shim suspend -> FLA suspend -> strict blocker gate -> re-lock -> checkpoint; inverse on restore (shim resume strictly after the toggle finishes everywhere) |
| sentry | `control/state_cuda_shim.go` | marker-file protocol with the interposer (existence-based, edge-triggered, per-pid acks, 5-min timeouts) |
| sentry | `nvproxy/checkpoint_blockers.go` | blocker inventory (00fd/00f8/00fb/exported-fd) with per-rank attribution; failed-export logging; fdinfo identity oracle (`nvproxy_exported_object` line) |
| sentry | `nvproxy/fla_registration.go` | host-free FLA registrations (00f8) in the suspend window; unwind replay via client-identity check; true restore relies on lazy re-registration |
| sentry | `nvproxy` device remap | sandbox-visible minors stable across cross-GPU restore; `DeviceInstance` translated back in RM outputs |
| sentry | `nvproxy/frontend_mmap.go`, `uvm_mmap.go` | mappings tracked; `InvalidateUnsavable` drops all translations over the unsavable device memmap.Files before save (was `MappableNoTrackMappings` = no-op -> save panic) |
| container | `tools/mcshim/mcshim.c` (+`mcshim_helper.c`) | tracks multicast groups / VMM exports+imports / legacy IPC / mappings at the libcuda layer; SUSPEND tears down, RESUME rebuilds at byte-identical VAs; helper proxies create/addDevice (blocked in restored processes); submission + mutator gate |

Interposition covers all three resolution paths (dlsym, `cuGetProcAddress`
v1/v2, cudart `cudaGetDriverEntryPoint*` — torch >= 2.11) and strips/refuses
`CU_MEM_HANDLE_TYPE_FABRIC` (IMEX out of scope).

## Mechanism inventory -> resolution

1. **Multicast (0x00fd)** — NCCL NVLS, torch symm-mem, FlashInfer fusion:
   shim teardown/rebuild + helper proxy. ✅
2. **Shared VMM P2P** (NCCL `cuMem`): close/re-import; exporter re-serves fd
   over unix socket keyed by the nvproxy fdinfo oracle. ✅
3. **Legacy CUDA IPC** (custom all-reduce): close + replay with
   reservation-holds and hole-plugging walk-back (no address hint exists).
   ✅ TP<=4; TP=8 requires custom-AR off (low-arena signal pads are
   unreplayable by construction).
4. **FLA registrations (0x00f8)** — libcuda-internal, one per peer-shared
   allocation on fabric-attached systems, no userspace API frees them,
   checkpointable-but-not-restorable: sentry host-frees in the suspend
   window; libcuda re-registers lazily post-restore. ✅
5. **FLA stale-cache on RESIDENT allocations** (the torch **multimem**
   killer, probed to exact mechanism: libcuda caches the registration
   handle per allocation; a resident allocation restored after its
   registration was freed presents the dead pair `[hVidMem, hFabricReg]`
   on its next export -> `OBJECT_NOT_FOUND` 0x57; no layer above libcuda
   can repair it): **`MCSHIM_FREE_UC_EXPORTS=1`** — suspend saves
   multicast-bound UC exporter contents into process memory, releases the
   allocation through libcuda (bookkeeping torn down consistently; FLA
   suspend then finds zero), resume recreates fresh at identical VAs +
   restores contents + re-exports. ✅
6. **Quiescence**: parallel lock + gate + bounded retries; teardown
   sandwiched inside lock/unlock/re-lock; ALL tracked mutators gated
   (TOCTOU closed in audit). ✅
7. **Cross-GPU restore**: device-view stability. ✅ (chained re-restores
   panic loudly by design)
8. **Save-side device mappings**: frontendFD/uvmFD now track mappings and
   invalidate before save (root cause of the historical
   `frontendFDMemmapFile` pma panic). ✅ pushed; see "straggler" below.
9. **Single-pass scope**: resume-after-successful-save with fabric users
   fails loudly (`cudaSaveFailedKey` marker distinguishes failed-save
   recovery, which replays FLAs via client-identity or drops staled ones).

## Validation matrix (all on the shipping binary+shim, full C/R cycle)

| Workload | Same-GPU | Cross-GPU (restore 0-3 -> 4-7) |
| --- | --- | --- |
| vLLM TP=2/4/8 (TP=8 CAR-off) | PASS | PASS (TP=2/4) |
| SGLang TP=2/4/8 (TP=8 CAR-off) | PASS | PASS (TP=2/4) |
| Forced `--enable-nccl-nvls` TP=4 | PASS | PASS 13.4x |
| vLLM symm-mem (`VLLM_ALLREDUCE_USE_SYMM_MEM=1`) TP=4 | PASS | PASS 9.2x |
| FlashInfer attn/sampling; fusion `trtllm` ENGAGED | PASS | PASS |
| torch symm-mem two-shot (bf16) TP=4/8 | PASS | PASS |
| torch symm-mem **multimem** TP=4 (`MCSHIM_FREE_UC_EXPORTS=1`) | PASS 5.4x (ckpt 40.2s) | PASS 5.4x |
| torch symm-mem **multimem** TP=8 (3B, CAR-off) | PASS 2.2x (uc_freed=2 on all 8 ranks) | n/a (all GPUs) |
| Failed-save recovery (`--leave-running`, FLAs live) | PASS | n/a |
| phase0 2-proc multicast harness | PASS (re-run after every sentry/shim change) | PASS |

Gate tooling: `e2e_gate.sh` (4-trial regression gate + engagement checks +
flake retry), `e2e_gate_t5..t8`, `e2e_probe_multimem*.sh`,
`e2e_close_cells.sh`. Trials must verify ENGAGEMENT (fusion self-disables
silently; fp16 disables symm-mem; `uc_freed`/`host-freed FLA` counts).

## Known-not-working / out of scope

- **TP=8 + custom all-reduce**: low-arena legacy imports unreplayable ->
  run CAR-off. (Driver placement property, not fixable here.)
- **Pause-quiesce (fully resident) checkpoints**: save now succeeds (after
  the pma fix) but restore dies replaying the resident object graph
  (`class 0x50a0`, NvStatus 31) — the separate "resident GPU state in the
  nvproxy graph" feature; sleep-workflow unaffected. Out of scope.
- **symm-mem + torch.compile**: upstream SGLang/torch inductor bug
  (multicast_ptr passed as plain int), reproduced natively.
- Multi-node / IMEX / MNNVL; R610 job mode (removed from branch, deferred);
  chained cross-GPU restores.

## The "pma flake" story (root-caused end-to-end)

Historical `Can't save pma with non-MemoryFile of type
*nvproxy.frontendFDMemmapFile` failures (~1/12, worse on symm-mem configs,
deterministic under pause-quiesce) had TWO stacked causes:

1. `frontendFD`/`uvmFD` used `MappableNoTrackMappings`, whose
   `InvalidateUnsavable` is a no-op — any translation over the unsavable
   device memmap.File at encoding time panicked the save. **Fixed**
   (vfio `pciDeviceFD` pattern), pushed in `e87baefdc`.
2. The reason translations existed at all in sleep-mode saves: a
   **straggler CUDA process** that initialized CUDA after the cuda-checkpoint
   process set was collected — its RM objects survive the checkpoint action
   (healthy saves have an EMPTY nvproxy object graph; measured), its GPU
   state is NOT in the snapshot, and post-fix the failure moves to restore
   (`failed to restore object ... class 0x50a0 ... NvStatus 31`). Frequency
   rises under host CPU load (bazel builds racing engine boot widen the
   late-init window).

## ⚠️ LOST WORK (uncommitted at environment reset) — re-implement

The **straggler guard** was implemented, built, unit-tested, phase0-PASSed,
and mid-gate (2/2 PASS at the break) but never committed; the working tree
was reset. Re-implementation sketch (small, ~70 lines):

- `nvproxy/checkpoint_blockers.go`: `LiveObjectsByClient(vfsObj)` — walk
  `nvp.clients[*].resources` (skip the client object itself), return
  `[]CheckpointBlocker` with `Kind:"live-object"`, sorted.
- `control/state_cuda.go`, end of `checkpointCudaProcs` (after the
  checkpoint action, before the success return): if
  `LiveObjectsByClient(...)` is non-empty -> unwind exactly like the
  checkpoint-phase-failure path (restore toggle, `ReplayFLARegistrations`,
  unlock) and fail with a retryable error naming the per-rank leftovers
  ("RM objects survived the cuda-checkpoint action ... retry the
  checkpoint"). Retry is self-healing: the straggler has CUDA state by
  then and joins the next process set.
- `e2e_gate.sh`: add "RM objects survived the cuda-checkpoint action" to
  the known-transient retry matcher (alongside "non-MemoryFile of type"),
  checking both the bench log and the run dir's `runsc-checkpoint.log`.
- Validate: phase0 + full gate + fusion double-run (the straggler-prone
  config).

## Branch state (origin/luis/r580-multicast-snapshots)

- All session work through `e87baefdc` ("nvproxy: invalidate device-mapping
  translations before save") is pushed and present.
- Post-session commits by owner: "Add makefile command to build shim",
  "Embed binary", plus merges of origin/master.
- Companion docs on the branch: `R580_VALIDATION.md` (full trial history +
  measured dead ends), `FLA_REPLAY_DESIGN.md`, `tools/mcshim/README.md`.
- The retired clean-history PR branch `luis/multicast-gpu-snapshots` is
  stale; the r580 branch is the single source of truth.

## Standard invocation

```
sudo env RUNSC=<runsc> CUDA_MULTICAST_SHIM=1 \
  CUDA_MULTICAST_SHIM_SRC=$PWD/tools/mcshim/mcshim.so \
  CUDA_CKPT_JOB_FILE=0 CUDA_CKPT_SEQUENTIAL=1 NCCL_CUMEM_ENABLE=1 \
  MCSHIM_IPC_SUSPEND=1 [MCSHIM_FREE_UC_EXPORTS=1] [MCSHIM_HIDE_MULTICAST=1] \
  bash cr-bench/bench_6_sglang_multi.sh --gpus 0,1,2,3 --tp 4 \
  [--no-torch-compile] [--restore-gpus 4,5,6,7]
```

Build: `sg docker -c 'make build TARGETS=//runsc'`;
shim: `sg docker -c 'bash tools/mcshim/build.sh'`. Never host bazel.

## Hard-won pitfalls (operational)

- Never background `&&` chains; nohup only standalone scripts. A launcher
  session ending mid-run also kills wrapper bookkeeping (bench itself may
  survive as root).
- Run dirs eat the disk (~750 GB observed twice): lazy-umount
  `mount | grep cr-bench` targets, `rm -rf` run dirs, keep `rootfs-*`.
- Root-owned stale logs in /tmp silently produce false verdicts; `sudo rm`
  before reruns.
- `pgrep/pkill -f` self-match; verify GPUs via
  `nvidia-smi --query-compute-apps` = 0.
- afterLoad "restored object" logs land in the restored sandbox's
  *boot.txt*; a healthy sleep-mode image replays ZERO nvproxy objects.
- The phase0 harness hard-fails if `tools/mcshim/mcshim.so` is missing
  (a stale-shim fallback once misdirected a full debugging session).
- C block comments cannot contain `*/` (cuMulticast*/... in a comment
  broke the build once).

## Next steps

1. **Re-implement + validate the straggler guard** (sketch above) — the
   only regression from the reset.
2. Re-run the full gate on the branch head after its master merges
   (binary was lost with /usr/local/bin; rebuild + reinstall).
3. Optional: multimem-vs-two-shot perf A/B; decide whether the loader
   should set `MCSHIM_FREE_UC_EXPORTS` by default; upstream PR carve-out
   (drop dev-only docs/harness files, ~6.8k PR-relevant insertions).
