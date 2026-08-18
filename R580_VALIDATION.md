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

`REBUILD_ROOTFS=1` is needed only on the first run per image. Images
(`cr-bench-vllm`, `cr-bench-sglang`) are built from `cr-bench/images/` and
pre-download the Qwen models, so `HF_HUB_OFFLINE=1` at runtime is satisfied.
