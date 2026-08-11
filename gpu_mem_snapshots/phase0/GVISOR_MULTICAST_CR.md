# Multicast checkpoint/restore, driven by gVisor

Multi-GPU workloads holding **multicast** objects (`NV_MEMORY_MULTICAST_FABRIC`,
class `0x00fd`) could not be checkpointed at all: `cuda-checkpoint` refuses, so
NVLS had to be disabled. This makes them checkpointable with **no NCCL fork, no
engine patch, and no application change**.

## How it works

`runsc --cuda-multicast-shim-path=<mcshim.so>` makes the sentry `LD_PRELOAD` a
libcuda interposer into a GPU container and export `MCSHIM_DIR`
(`runsc/boot/loader.go`). The interposer tracks every multicast group and CUDA
IPC import; gVisor tells it when to release them and when to rebuild
(`pkg/sentry/control/state_cuda_shim.go`). Everything comes back at
**byte-identical virtual addresses**, so application pointers and captured CUDA
graphs stay valid — only the opaque handles change, and the interposer rewrites
stale handle values for later calls.

Enable with (per container):

    runsc --cuda-checkpoint-path=/usr/local/bin/cuda-checkpoint \
          --cuda-multicast-shim-path=/usr/local/lib/mcshim.so ...
    runsc checkpoint --cuda-checkpoint-path=... --cuda-checkpoint-sequential ...

Build the interposer with `gpu_mem_snapshots/phase0/mcshim/build.sh`. It builds
inside `ubuntu:22.04` on purpose: a host build links against the host glibc, and
glibc 2.38+ redirects `sscanf` to `__isoc23_sscanf`, so the library then fails
to load in older images — which shows up only as the container exiting
immediately with an empty log.

## The sequence, and why each step is where it is

    gate ──► lock ──► unlock ──► tear down ──► lock ──► checkpoint ──► save
                                                                        │
    resume ◄── rebuild ◄── toggle (restore+unlock) ◄── restore ◄─────────┘

1. **Gate before lock.** The gate stops the application submitting *new* GPU
   work; the lock drains and preempts work already *in flight*. Neither
   suffices: gating alone deadlocks (see 2), and locking alone cannot keep up
   with a workload that never idles, reporting `device not ready`.
2. **Retry the (gate, lock) pair.** A rank gated just before submitting
   collective N starves peers already spinning in N, and the lock cannot
   quiesce those peers either. Releasing the gate lets that collective finish,
   so the pair is retried up to `cudaLockGateAttempts` times, each attempt a
   fresh chance to catch every rank between collectives. Failure is clean, with
   the application still running.
3. **Tear down inside the lock, after an unlock.** The interposer needs libcuda,
   which a locked process cannot use — but the application must not be running
   either. Arming the gate while locked is safe (it only flips a flag), so
   unlocking then leaves an already-drained GPU that the application is barred
   from touching.
4. **Verify, do not trust.** After the teardown the blocker gate is re-run with
   nothing exempt, so anything left unreleased fails here rather than becoming a
   snapshot that only misbehaves after restore.
5. **Rebuild after the toggle, and hold the application until it finishes.**
   `--toggle` restores *and unlocks*, so the application becomes runnable while
   the rebuild is still in flight; the gate is what keeps it off half-rebuilt
   multicast VAs. It is released only on success.

Only processes that announced `present.<pid>` are expected to acknowledge.
gVisor picks CUDA processes by open NVIDIA device FDs, which is deliberately
broad — a vLLM API server holds them without ever touching multicast, and
waiting on it would hang every checkpoint.

## Results

8× H100 + NVSwitch, driver 610.57.04.

Synthetic acceptance (`run_matrix.sh`), stock NCCL with a captured CUDA graph,
all with **0 context faults**:

| Case | Result |
| --- | --- |
| TP=4 NVLS, paused | PASS |
| TP=4 NVLS, running | PASS |
| TP=4 NVLS, running, no idle gap | PASS |
| TP=4 no NVLS, running, no idle gap | PASS |
| TP=8 NVLS, paused | PASS |
| TP=8 NVLS, running, no idle gap | PASS |

Stock vLLM 0.27, stock NCCL, `cr-bench/bench_4_vllm_multi.sh` with
`CUDA_MULTICAST_SHIM=1`:

| Case | Result |
| --- | --- |
| TP=2, multicast live, torch.compile + CUDA graphs | PASS — ckpt 12.6s, restore 4.1s, answer EXACT MATCH |
| TP=4, NCCL NVLS on, `--enforce-eager` | PASS — ckpt 24.3s, restore 5.7s, answer EXACT MATCH |
| TP=4, NCCL NVLS on, torch.compile + CUDA graphs | FAIL, not ours — see below |

The TP=4 row is the new capability: it previously required NVLS off, because
live multicast made the process un-checkpointable outright.

## Known limitation, outside this work

vLLM TP=4 with torch.compile + piecewise CUDA graphs still fails, in
`cuda-checkpoint --toggle`, which returns `unknown error` restoring three of
four workers — before gVisor reaches the rebuild. This is PROGRESS.md finding
\#3/\#4, reproduced natively under `runc` with no gVisor in the loop
(`gpu_mem_snapshots/native_ab.sh`).

It was bisected here to confirm it is not multicast related: with vLLM's own
custom all-reduce and symmetric memory disabled, the interposer has only NCCL's
two NVLS groups to handle and suspends them cleanly (`groups=2 imports=0`), and
the toggle still fails identically.

## Debugging

* `MCSHIM_LOG=<path>` sends interposer logging to a file instead of stderr.
* `MCSHIM_VERBOSE=1` adds per-symbol resolution tracing (very chatty).
* `MCSHIM_DISABLE=1` disables the interposer in a process.
* `CTXPROBE` lines report a `cuCtxSynchronize` result at suspend entry, suspend
  exit and rebuild entry. CUDA latches an unrecoverable fault into the context,
  so the first probe reporting non-zero brackets exactly when it died. This is
  what the matrix asserts on, and it is the single most useful signal here: a
  rebuild can report complete success on an already-dead context, because VMM
  calls never touch the device.

A caution recorded from experience: **environment variables do not reach the
sentry.** `runsc` clears the sandbox environment
(`cmd.Env = []string{}` in `runsc/sandbox/sandbox.go`), so an env-gated knob
read by sentry code silently never fires — which once produced a completely
misleading bisect.
