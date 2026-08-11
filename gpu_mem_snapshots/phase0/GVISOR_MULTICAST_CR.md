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

## Rebuilding the environment from scratch

Nothing here depends on a warm machine, but four artifacts are not in git and
have to be recreated. Recorded because doing it blind costs an hour.

1. **Driver + fabric manager.** `instal_nvidia.sh` installs 610.57.04. On a
   fresh Ubuntu 26.04 image it needs two things it does not do itself:
   `build-essential` plus `linux-headers-$(uname -r)` (the runfile aborts with
   "Unable to find the development tool `cc`"), and correct apt version pins --
   NVIDIA publishes `nvidia-fabricmanager=610.57.04-1ubuntu1`, not the
   `610.57.04-1` the script asks for, and the container toolkit pin has moved
   too. Fabric manager **must** match the driver exactly or NVSwitch never
   initialises and there is no NVLS to test.
2. **Persistence.** The runfile ships no systemd unit, so `nvidia-smi -pm 1`
   alone is lost on reboot. Install a `nvidia-persistenced.service` and enable
   it.
3. **Stock NCCL** at `/opt/phase0/nccl-stock/libnccl.so.2`: extract
   `libnccl.so.2` from the `nvidia-nccl-cu13` wheel. A wheel build is exactly
   what "stock" means here, and it avoids a source build.
4. **cuda-checkpoint** at `/usr/local/bin/cuda-checkpoint`, from the NVIDIA
   cuda-checkpoint repo.

Then `make build TARGETS=//runsc` (containerised, needs Docker),
`mcshim/build.sh`, and `run_matrix.sh` to confirm.

Sanity checks before trusting any result: `nvidia-smi -q | grep -A3 Fabric`
must say `Completed` / `Success`, and every GPU must read 0 MiB.

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
| TP=2, multicast live, torch.compile + CUDA graphs | **8/8 PASS** — ckpt 12.6s, restore 4.1s, answer EXACT MATCH |
| TP=4, NCCL NVLS on, `--enforce-eager` | **3/4 PASS** — ckpt 24.3s, restore 5.7s, answer EXACT MATCH |
| TP=4, NCCL NVLS on, torch.compile + CUDA graphs | **5/8 PASS** — answer EXACT MATCH |

Every failure in those runs was cuda-checkpoint's own restore toggle (below).
**None was attributable to the interposer**, which is the distinction
`cr-bench/vllm_trials.sh` exists to make: it reports a pass rate and classifies
each failure, reading the sentry log rather than the benchmark's stdout, because
that is where the toggle failure is reported.

The last row is worth stating plainly, because earlier notes in this tree
(including an earlier version of this file) call it impossible. It is the
configuration PROGRESS.md's TL;DR marks ❌ "cuda-checkpoint cannot restore it".
It now succeeds most of the time. Two separate things had to be fixed to get
there, and the second was found only by repeating the run: multicast had to stop
being a checkpoint blocker at all, and the interposer had to stop mistranslating
handle aliases (see below). Before the alias fix this configuration failed every
time, which is why it was written off.

The TP=4 row is the new capability: it previously required NVLS off, because
live multicast made the process un-checkpointable outright.

## Known limitation, outside this work

At TP=4, roughly a third of runs fail inside `cuda-checkpoint --toggle`, which
returns `unknown error` restoring some of the workers — before gVisor reaches
the rebuild, and regardless of whether CUDA graphs are enabled. This is the
error signature of PROGRESS.md finding \#3, and that finding was reproduced
natively under `runc` with no gVisor in the loop
(`gpu_mem_snapshots/native_ab.sh`).

It is not multicast related. Bisected: with vLLM's own custom all-reduce and
symmetric memory disabled, the interposer has only NCCL's two NVLS groups to
handle and suspends them cleanly (`groups=2 imports=0`), and the toggle still
fails the same way. It is also independent of the interposer's own health — the
suspend, the blocker check and the rebuild all report success in the runs that
hit it.

Note this is weaker than PROGRESS.md's conclusion, which treats the
compile + CUDA-graph case as a hard restore failure. Under this path it is
intermittent rather than deterministic, plausibly because the workers now carry
much less live IPC and fabric state into the checkpoint. Retrying the checkpoint
is a viable mitigation today; a fix belongs in cuda-checkpoint.

## A bug worth remembering: handle aliases

After a rebuild, an object's handle changes, so the interposer keeps the previous
values as aliases and rewrites later calls that still use one (`xlate_mc`). The
driver, however, **reuses handle values**. If a value being issued for a new
allocation still appears in some other object's alias list, that alias is stale,
and rewriting through it silently redirects a legitimate reference to an
unrelated object.

The symptom was nothing like the cause: an unrelated `cuMem*` call failing with
`operation not supported` inside vLLM's own sleep/wake allocator, intermittently
(1/4 runs passed) and only after a rebuild had created aliases. vLLM's
`/sleep` releases and re-creates every weight allocation, which makes value
collisions routine; the NCCL harness barely churns allocations and never showed
it.

Aliases are now dropped from every other object whenever the driver reissues
that value, and cleared when an object is forgotten. That took vLLM TP=2 from
1/4 to 4/4.

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
