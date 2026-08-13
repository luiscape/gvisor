# The engine sleep/checkpoint/restore/wake workflow: what works

The target workflow, and what `bench_4_vllm_multi.sh` implements:

1. engine container starts
2. loads weights, runs `torch.compile`, captures CUDA graphs (**eager off**)
3. `POST /sleep?level=1` — weights offloaded to CPU, KV cache dropped
4. `runsc checkpoint` of the whole container
5. `runsc restore`
6. `POST /wake_up`
7. requests verified against a pre-checkpoint reference answer

## It passes, at TP=2, TP=4 and TP=8

H100 + NVSwitch, driver 610.57.04, `runsc-phase0`, interposer enabled:

```
sudo CUDA_MULTICAST_SHIM=1 CUDA_CKPT_JOB_FILE=1 CUDA_CKPT_SEQUENTIAL=1 \
     EAGER=0 DISABLE_CUSTOM_ALL_REDUCE=1 NCCL_CUMEM_ENABLE=1 \
     RUNSC=/usr/local/bin/runsc-phase0 \
     bash cr-bench/bench_4_vllm_multi.sh --gpus 0,1,2,3 --tp 4
```

| step | TP=2 | TP=4 | TP=8 |
| --- | --- | --- | --- |
| cold boot (run -> health) | 215 s | 251 s | 318 s |
| checkpoint | 12.3 s, 9.6 GB | 23.9 s, 16 GB | 57.9 s, 30 GB |
| GPU memory after checkpoint | 0 MiB | 0 MiB | 0 MiB (all eight) |
| `runsc restore` | 4.0 s | 6.2 s | 12.4 s |
| health after restore | 4.1 s | 6.3 s | 12.5 s |
| `POST /wake_up` | **ok** | **ok** | **ok** |
| first inference after restore | 14.8 s | 25.7 s | 60.2 s |
| answer vs. pre-checkpoint reference | **EXACT MATCH** | **EXACT MATCH** | **EXACT MATCH** |

TP=2 and TP=4 above are `Qwen2.5-1.5B-Instruct`; TP=8 needs
`heads % 8 == 0`, which that model does not satisfy (12 heads), so TP=8 uses
`Qwen2.5-3B-Instruct` (16 heads, intermediate 11008 -- clean at 2, 4 and 8).
That model was first run at TP=4 as a control, so a TP=8 result could not be
confounded by the model change; it passed with an exact match.

TP=8 accounting was uniform across all eight workers --
`groups=3 imports=55 released=58 ipc_closed=0 ipc_left_live=0` -- and NCCL
reported NVLS available and allocated on all eight devices.

Caching a TP=8-capable model, since the sandbox runs `--network=none`:

```
sudo docker run --rm --network=host --entrypoint python3 \
  -v /data/cr-bench/rootfs-cr-bench-vllm/app/hf_cache:/hf \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=0 cr-bench-vllm -c \
  "from huggingface_hub import snapshot_download; \
   snapshot_download('Qwen/Qwen2.5-3B-Instruct', \
     allow_patterns=['*.json','*.safetensors','*.txt'])"
```

The image bakes `HF_HUB_OFFLINE=1` and its entrypoint is the server, so both
have to be overridden.

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

## Compatibility

Two different mechanisms carry shared memory across the checkpoint, and the
split is not the one an earlier revision of this document claimed:

- **VMM** (`cuMemCreate` / `cuMemMap` / multicast) -- the **interposer**
  releases these before the checkpoint and remaps them at identical addresses,
  which it can do because `cuMemAddressReserve` takes an address hint and the
  reservation is retained.
- **Legacy CUDA IPC** (`cuIpcGetMemHandle` / `cuIpcOpenMemHandle`) -- the
  **driver** carries these, via R610's job mode (`--launch-job`), which
  documents CUDA IPC support. Measured: vLLM TP=2 with 58 live legacy imports
  per worker checkpoints and restores to an exact match with the interposer not
  touching them at all.

So the interposer must **leave legacy IPC alone** (`MCSHIM_IPC_SUSPEND` unset,
the default). It has the machinery to close and reopen legacy imports, and
using it is actively harmful here: `cuIpcOpenMemHandle` has no address hint, so
what it closes it cannot put back (measured: 0 of 58 returned to their original
address). That teardown is only for environments where the driver does *not*
cover IPC -- standalone/non-job mode, or pre-R610.

An earlier revision concluded legacy IPC was categorically unrestorable and
that custom all-reduce therefore had to stay off. That was wrong twice over: it
measured the consequences of the interposer's own teardown rather than a driver
limitation, and it drew categorical conclusions from single runs of an
intermittent workload. What is actually true is a **reliability** difference,
not a compatibility one (below).

### By feature

| Feature | Shares memory via | Carried by | Status |
| --- | --- | --- | --- |
| `torch.compile` + CUDA graphs (`EAGER=0`) | n/a | n/a | **works**, TP=2 and TP=4 |
| NCCL P2P, `NCCL_CUMEM_ENABLE=1` | VMM | interposer | **works** |
| NCCL P2P, `NCCL_CUMEM_ENABLE=0` | legacy IPC | driver (job mode) | **works at TP=2**; see scale limit |
| NCCL NVLS multicast (TP>=4) | VMM multicast | interposer | **works**, NVLS confirmed via `NCCL_DEBUG` |
| torch symmetric memory | VMM multicast | interposer | **works** in the PyTorch tier; **not** confirmed inside vLLM |
| vLLM custom all-reduce | legacy IPC | driver (job mode) | **works at TP=2**; see scale limit |
| FlashInfer all-reduce | not established | -- | **unknown**, no coverage |

### By configuration

All measured end to end under gVisor with `EAGER=0` (compile + CUDA graphs on)
and `MCSHIM_IPC_SUSPEND` unset. "legacy live" is per worker.

Repeated trials (`vllm_trials.sh`), `CUMEM=1` throughout:

| TP | custom AR | trial set | + earlier singles | total |
| --- | --- | --- | --- | --- |
| 4 | **off** | **5/5** | 2/2 | **7/7** |
| 4 | on | 3/5 | -- | 3/5 |
| 2 | on | 3/3 | 0/1 | 3/4 |
| 2 | off | -- | 1/1 | 1/1 |

Every failure was classified as a restore-toggle failure, with **zero** in the
"other" bucket, so none of these were disk or infrastructure noise.

**Custom all-reduce costs reliability, measurably.** Off: 8/8 across both TP.
On: 6/9. It is not a compatibility boundary -- it passes, repeatedly, and
produces exact matches when it does -- but roughly a third of restores fail
with it enabled, against none observed without it.

Other cells, single runs only, for the record:

| TP | `CUMEM` | custom AR | legacy live (driver) | runs |
| --- | --- | --- | --- | --- |
| 2 | 0 | on | 58 | 1/1 |
| 4 | 0 | on | 102-126 | 1/2 |

Use `vllm_trials.sh` on the cell you intend to depend on. Single runs in this
workload have repeatedly proven misleading in both directions.

### Does NVLS let custom all-reduce work?

NVLS is irrelevant to the question -- they are independent layers. NVLS is
NCCL's collective algorithm and its multicast objects go through the VMM API
(interposer); custom all-reduce bypasses NCCL entirely and exchanges peer
buffers over legacy CUDA IPC (driver). Enabling one does not remove the other.

Custom all-reduce costs **live legacy imports for the driver to carry** --
`buffers x (world - 1)` of them, additive with whatever NCCL contributes at
`NCCL_CUMEM_ENABLE=0`. The driver does carry them, and does so correctly when
it works. What it costs is reliability: 6/9 with it on versus 8/8 with it off.

No *threshold* in import count has held up -- 126 live imports passed, 10 live
imports failed once then passed three times -- so this is not a capacity limit
to tune around. But the presence of live legacy imports does track the failure
rate, which is the practical form of the same observation.

## The settings that matter, and why

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

### `DISABLE_CUSTOM_ALL_REDUCE=1` — recommended (costs ~1/3 of restores if not)

vLLM's custom all-reduce uses legacy CUDA IPC, which the driver carries in job
mode, so it is not incompatible -- it passes repeatedly and produces exact
matches when it does. It is a reliability trade, measured at TP=4 with
everything else held fixed:

| | trials |
| --- | --- |
| `DISABLE_CUSTOM_ALL_REDUCE=1` | **5/5** |
| `DISABLE_CUSTOM_ALL_REDUCE=0` | 3/5 |

Across both TP values: 8/8 with it off, 6/9 with it on. So enabling it costs
roughly a third of your restores, and every failure lands on the restore
toggle. Turn it off unless its inference speedup is worth retrying a checkpoint
restore that often on your workload.

**`/sleep` does not change this.** Sleep level 1 offloads weights and drops the
KV cache; custom all-reduce registers its buffers once at init, outside that
scope, so they are still live and imported at checkpoint time. Sleep gives
quiesce and frees the large allocations — which is what makes checkpointing
tractable — but it does not reduce the number of IPC registrations the driver
has to carry.

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
