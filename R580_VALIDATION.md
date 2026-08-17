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
| vLLM | 4 | same | | | | | in progress |
| vLLM | 2 | 0,1 -> 6,7 | | | | | queued |
| vLLM | 4 | 0-3 -> 4-7 | | | | | queued |
| vLLM trials (n=3) | 4 | same | | | | | queued |
| SGLang | 2 | same | | | | | queued |

vLLM TP=2 checkpoint: 13.3 s, image 9.5 G, all 3 verification queries correct.

## Verification hygiene

Two layers of false confidence were found and removed; both had made a
cross-GPU restore that **never moved** report PASS:

1. `cb_assert_gpu_placement()` did not exist, yet `_bench_vllm_impl.sh` and
   `_bench_sglang_impl.sh` already printed `placement verified: YES` and gated
   PASS on `${PLACEMENT_OK:-1}` -- a variable nothing ever assigned.
2. The phase0 gVisor harness reported PASS purely on collective correctness,
   which a non-moved restore satisfies trivially.

Both now assert placement. The check reads `/dev/nvidia*` FDs from the **sentry**
process, not `nvidia-smi --query-compute-apps`: under the sleep workflow the
engine has released its device memory by restore time, so nvidia-smi lists no
processes at all, while the sentry still holds every device it opened. Scoping
to the sentry also excludes `nvidia-persistenced`, which holds all GPUs.

Consequence: **`HANDOFF.md`'s R610 cross-GPU claim is not trustworthy** -- it was
measured with the broken code path and with both checks dormant. It needs
re-running on R610 before any such claim ships.

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
