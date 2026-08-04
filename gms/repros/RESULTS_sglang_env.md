# SGLang env-var snapshot repro (local adaptation of gcr/qwen_repro.py)

`repro_sglang_env.py` + `run_sglang_env.sh` run a real SGLang server under
`runsc` (small text model, Qwen2.5-0.5B-Instruct), warm it up, release GPU
memory (`/release_memory_occupation`), dump every `/dev/nvidia*` FD holder's
environment (`dump_gpu_process_env`, ported verbatim from qwen_repro.py), then
snapshot → restore → `/wake` → `/verify`. The **outcome is decided by the
env-var `--profile`**, exactly as in the original.

All Modal machinery removed; lifecycle driven by the local runner. The wrapper
serves `/health` (always responsive), `/wake` (retries resume + continue like
cr-bench), `/verify` (chat, compares to the reference answer captured while
awake).

## Environment

Same box as `RESULTS.md`: A10G, driver 580.95.05, nvproxy ABI 580.65.06.
**fix** runsc `g6bd82d0a1502` (race fix + nvproxy C/R); **stub** runsc
`gfcd95d5fac0f` (prod-equivalent). SGLang 0.5.16 image `cr-bench-sglang`.

## Profiles and results

| profile | env delta | runsc | result | signature |
|---|---|---|---|---|
| `good` | `NCCL_CUMEM_ENABLE=0`, `NCCL_CUMEM_HOST_ENABLE=0`, `TORCHINDUCTOR_COMPILE_THREADS=1`, IPC off | fix, TP=1 | **PASS** | ref `Paris` → checkpoint 3.7s → restore 1.2s → wake → verify `Paris` (match); dump: `OK env`, `OK fds` |
| `bad-inductor` | torch.compile ON, `TORCHINDUCTOR_COMPILE_THREADS` unset | **stub**, TP=1 | **CHECKPOINT-BLOCKED** | `encoding error: nvproxy.frontendFD is not saveable`; dump: `BLOCKER: 32 holder(s) carry inherited-fork /dev/nvidia* FDs` |
| `bad-inductor` | same | fix, TP=1 | PASS | dump still shows the 32 inherited-fork holders, but nvproxy C/R serializes them |
| `bad-nccl` | `NCCL_CUMEM_ENABLE=1`, `NCCL_CUMEM_HOST_ENABLE=1` | fix, **TP=2** | **RESTORE-BLOCKED** | boot log: `cuda-checkpoint --toggle for PID 128 ... "operation not supported"` → `Killing the sandbox after post restore work failed`; dump predicted it: `WARNING: 4/4 GPU FD-holder(s) missing ...=0; these will likely fail restore` |

Run: `sudo bash gms/repros/run_sglang_env.sh --profile <good|bad-nccl|bad-inductor> [--tp N] [--gpus 0,1]`
(`RUNSC=/usr/local/bin/runsc CUDA_CKPT=0` selects the stub for the
checkpoint-blocking case.)

## What each profile demonstrates

### good — snapshot-safe (control)
Dump shows 3 CUDA-session holders (launch_server, scheduler, detokenizer), all
carrying `NCCL_CUMEM_ENABLE=0 NCCL_CUMEM_HOST_ENABLE=0`, no inherited-fork FDs.
`OK env` + `OK fds`. Full round-trip succeeds, answer identical pre/post.

### bad-inductor — checkpoint-side failure (inherited fork FDs)
With `--enable-torch-compile` and no `TORCHINDUCTOR_COMPILE_THREADS=1`,
PyTorch Inductor's `SubprocPool` spawns ~20 persistent
`torch/_inductor/compile_worker/__main__.py` subprocesses that inherit the
scheduler's `/dev/nvidia*` FDs but have **no CUDA context of their own**
(`kind=INHERITED-fork-fd`, no `cudaXXXXXXXXXXX` thread). cuda-checkpoint cannot
toggle them, so on the **stub** their frontend FDs abort the checkpoint with
the exact prod panic. On the **fix**, nvproxy C/R serializes them and it
passes. This is the mechanism the qwen_repro `TORCHINDUCTOR_COMPILE_THREADS=1`
env var exists to prevent (compile in-process, no pool).

### bad-nccl — restore-side failure (NCCL host cuMem), TP=2
With `NCCL_CUMEM_*=1`, NCCL's host-side cuMem allocations back the TP
communicator with FD-exported handles the driver cannot recreate on restore.
The dump flags every TP holder `<-- BAD env` and warns of the outcome; on
restore the sentry's `cuda-checkpoint --toggle` of the TP scheduler PIDs fails
`operation not supported` and the sandbox is killed. Needs TP≥2 (no NCCL
communicator at TP=1, so the env var has no effect there — the good/bad
distinction only appears once NCCL initializes).

## Takeaway

The `dump_gpu_process_env` diagnostic is the actionable artifact: pre-snapshot
it labels each `/dev/nvidia*` holder as `CUDA-session` vs `INHERITED-fork-fd`,
flags `NCCL_CUMEM_*` != 0, and prints the two verdicts (`OK/ BLOCKER fds`,
`OK/ WARNING env`) that predict checkpoint-side and restore-side failure
respectively — both confirmed against the actual runsc outcome here.
