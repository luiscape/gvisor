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

vLLM is not covered at TP=8: both models baked into the benchmark image have an
attention-head count that is not divisible by 8 (Qwen2.5-1.5B has 12), so vLLM
refuses at startup, before anything under test runs. Testing it needs a model
with 8 | heads added to the image. The multicast machinery itself is covered at
8 ranks by the synthetic matrix above.

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

## Parameter sweep (2026-08-12): which vLLM configurations are covered

vLLM 0.27 can create multicast from four places -- NCCL NVLS, custom
all-reduce, torch symmetric memory, and the FlashInfer all-reduce backend --
and the benchmark's defaults only exercise the first two. `mcshim_sweep.sh`
runs one checkpoint/restore per cell at TP=2 to find breakage.

| Cell | Result | Interposer saw |
| --- | --- | --- |
| baseline (custom all-reduce on, cuMem off) | PASS | `groups=4 imports=2` |
| `VLLM_ALLREDUCE_USE_SYMM_MEM=1` | PASS | `groups=4 imports=2` |
| `SLEEP_LEVEL=2` (discard all GPU memory) | PASS | `groups=4 imports=2` |
| `NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=1` | 1/3 PASS | `groups=3 imports=50` |
| all of the above together | PASS | `groups=3 imports=50` |
| `VLLM_ALLREDUCE_USE_FLASHINFER=1` | n/a | never boots |

Three things worth recording.

**The interposer did not fail in any cell.** Every failure was
cuda-checkpoint's restore toggle, and in each of those the suspend, the
blocker check and the rebuild all reported success first.

**`NCCL_CUMEM_ENABLE=1` reproduces the toggle bug at TP=2**, which is new and
useful. The bug was previously only reachable at TP>=4, where a run costs four
GPUs and several minutes. Turning on cuMem takes the interposer from 2 live
peer imports to **50** -- NCCL switches to VMM allocations -- and the toggle
starts failing (1/3) in a configuration that is otherwise 8/8. That is direct
evidence for the standing theory that the toggle failure is about live CUDA
IPC/VMM imports rather than about multicast, and it gives NVIDIA a much
cheaper reproducer than a TP=4 engine run.

**Two cells prove less than they appear to.** `symmmem` reports the same
`groups=4` as the baseline, so vLLM's symmetric-memory path evidently did not
engage at this model size (it is size-gated by `should_use_symm_mem`); torch
symmetric memory is covered instead by the PyTorch tier, where enabling it
demonstrably adds a group (`groups=1` -> `groups=2`, see NCCL_PATCH_TESTS.md).
And `flashinfer` never reaches a checkpoint at all. It JIT-compiles its
kernels at startup and the engine dies during cold boot. Two blockers, one
fixed and one not:

* It looks for `nvcc` under `CUDA_HOME` and the image has no
  `/usr/local/cuda`. It does, however, ship a complete CUDA 13.3 toolchain in
  torch's pip tree (`nvidia/cu13`: `bin/nvcc`, `include`, `nvvm`), so the
  bench now points `CUDA_HOME` there -- no CUDA toolkit needs adding to the
  image.
* With that fixed it compiles and then fails on a version mismatch:
  FlashInfer 0.6.16 bundles its own cccl/libcudacxx headers, which reject the
  13.3 compiler with `"CUDA compiler and CUDA toolkit headers are
  incompatible, please check your include paths"`.

That second one is a packaging problem in the image, unrelated to
checkpoint/restore, and it needs a matching flashinfer/CUDA pair to resolve.
So **this multicast owner remains untested** -- the one real coverage gap
left. Nothing is known about whether the interposer handles it.

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

What it looks like, and two things that do **not** fix it, so they are not
retried:

* The shape is always the same: some number of workers toggle successfully and
  then every remaining one fails, i.e. the first failure cascades. That looks
  like a restore-order dependency — a member that imported CUDA IPC memory can
  only come back once its exporter has, and the interposer releases the
  VMM-based imports but not the legacy `cuIpcGetMemHandle` ones, which is what
  job mode exists to carry.
* **Toggling members in parallel: 0/4.** The sequential requirement is real and
  still holds even with multicast and peer imports released beforehand.
* **Sorting members by PID to approximate exporter-before-importer: 6/9,
  against 5/8 unsorted.** No effect, and the ordering is not even stable enough
  to make the outcome reproducible, so the code deliberately does not sort.
* **Retrying just the failed members, after every member that could be restored
  has been: does not help.** The retry fired once across five runs and reported
  `still failing after a retry`.

That last one matters: if this were purely a restore-order problem, a second
pass would find the missing exporter already running. It does not, so the first
failure is corrupting shared job state rather than merely arriving too early —
which is why none of the orchestration-level fixes above work, and why a fix has
to come from cuda-checkpoint. The mitigation available today is to retry the
whole checkpoint/restore.

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
