# The engine sleep/checkpoint/restore/wake workflow: what works

The target workflow, and what `bench_4_vllm_multi.sh` implements:

1. engine container starts
2. loads weights, runs `torch.compile`, captures CUDA graphs (**eager off**)
3. `POST /sleep?level=1` — weights offloaded to CPU, KV cache dropped
4. `runsc checkpoint` of the whole container
5. `runsc restore`
6. `POST /wake_up`
7. requests verified against a pre-checkpoint reference answer

## It passes, at TP=2 and TP=4

H100 + NVSwitch, driver 610.57.04, `runsc-phase0`, interposer enabled:

```
sudo CUDA_MULTICAST_SHIM=1 CUDA_CKPT_JOB_FILE=1 CUDA_CKPT_SEQUENTIAL=1 \
     EAGER=0 DISABLE_CUSTOM_ALL_REDUCE=1 NCCL_CUMEM_ENABLE=1 \
     RUNSC=/usr/local/bin/runsc-phase0 \
     bash cr-bench/bench_4_vllm_multi.sh --gpus 0,1,2,3 --tp 4
```

| step | TP=2 | TP=4 |
| --- | --- | --- |
| cold boot (run -> health) | 215 s | 251 s |
| checkpoint | 12.3 s, 9.6 GB | 23.9 s, 16 GB |
| GPU memory after checkpoint | 0 MiB | 0 MiB (all four) |
| `runsc restore` | 4.0 s | 6.2 s |
| health after restore | 4.1 s | 6.3 s |
| `POST /wake_up` | **ok** | **ok** |
| first inference after restore | 14.8 s | 25.7 s |
| answer vs. pre-checkpoint reference | **EXACT MATCH** | **EXACT MATCH** |

`torch.compile` and CUDA graphs are **on** (`EAGER=0`). Interposer accounting
per worker, TP=4: `groups=2 imports=51 released=53 remapped=53 ipc_closed=0`
-- every mapping back at an identical VA, no legacy IPC anywhere.

**TP=4 exercises NVLS**, which TP=2 does not: 2-GPU NCCL uses direct P2P, so
the multicast path is only really under test from 4 ranks up. Confirmed rather
than assumed, with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NVLS,REG`:

```
NVLS multicast support is available on dev 0..3 (NVLS_NCHANNELS 16)
NVLS rank 0..3 (dev 0..3) alloc done
NVLS importing shareableHandle ... from rank 0
4 nvls channels
```

Worth noting the tracked multicast group count is `groups=2` at both TP=2 and
TP=4, so that number alone is not evidence that NVLS engaged -- the NCCL debug
output is.

This supersedes `gpu_mem_snapshots/PROGRESS.md`'s conclusion that multi-GPU
vLLM with compile + piecewise CUDA graphs cannot be restored. That was true of
`cuda-checkpoint` alone; it is not true with the interposer releasing and
replaying the cross-process imports.

## The two settings that matter, and why

### `NCCL_CUMEM_ENABLE=1` — required

This is the one that decides whether the workflow works at all. It selects
which API NCCL uses for intra-node P2P buffers:

| | NCCL P2P uses | live legacy imports/worker | restore |
| --- | --- | --- | --- |
| `NCCL_CUMEM_ENABLE=0` | legacy CUDA IPC | ~48 | **fails** |
| `NCCL_CUMEM_ENABLE=1` | VMM (`cuMemCreate`/`cuMemMap`) | 0 | **passes** |

The interposer restores VMM mappings at their original addresses by retaining
the address reservation across the checkpoint. Legacy IPC has no equivalent:
`cuIpcOpenMemHandle` takes no address hint and the driver packs from the low
end of its region, so reopened imports land somewhere else (measured: 0 of 48
returned to their original address). The interposer detects this and fails the
resume loudly rather than leaving the engine with stale pointers.

The harness used to default this to `0` as "the safe path", reasoning about
`cuda-checkpoint`'s VMM coverage. That reasoning predates the interposer,
which is what restores these mappings now. The default is now `1`.

### `DISABLE_CUSTOM_ALL_REDUCE=1` — still required

vLLM's custom all-reduce uses legacy CUDA IPC directly, and unlike NCCL it has
no VMM mode to switch to. Measured with everything else already correct
(`NCCL_CUMEM_ENABLE=1`, TP=2):

| | legacy imports/worker | result |
| --- | --- | --- |
| `DISABLE_CUSTOM_ALL_REDUCE=1` | 0 | **PASS**, exact match |
| `DISABLE_CUSTOM_ALL_REDUCE=0` | 10 | `10 MOVED`, resume fails, `wake_up` fails |

**`/sleep` does not help here.** Sleep level 1 offloads weights and drops the
KV cache; the custom all-reduce buffers are registered once at init and are
outside that scope, so they are still live and still imported at checkpoint
time. Sleep solves quiescing and frees the large allocations — which is what
makes checkpointing tractable — but it does not release the IPC registrations
that break restore.

Closing that last gap needs vLLM to re-run its own IPC exchange after restore.
It owns those pointers and can update them; the interposer cannot.

## Symmetric memory

`VLLM_ALLREDUCE_USE_SYMM_MEM` can stay on: torch symmetric memory creates
multicast through the VMM/multicast API, which the interposer handles (it is
covered by the `groups=` count above and by the PyTorch tier's `SYMM_MEM=1`
case).

## Diagnosing a failure

The interposer's per-worker accounting in `applog/vllm.log` says which layer
is at fault:

- `ipc_closed=N` with `N > 0` — legacy CUDA IPC is in use. Find the owner;
  it is either `NCCL_CUMEM_ENABLE=0` (~48 at TP=2) or custom all-reduce (~10).
- `M MOVED` — legacy imports did not come back to their addresses. Expected
  whenever `ipc_closed > 0` at engine scale; not fixable by configuration
  other than removing the owner.
- `remapped=N` / `re-mapped IDENTICAL (retained-reservation)` — the VMM path
  working as intended.
- `GATE: app thread blocked until resume` followed by `shm_broadcast` timeouts
  means a resume failed and left the application gated; the `wake_up` hang is
  a consequence, not the cause.
