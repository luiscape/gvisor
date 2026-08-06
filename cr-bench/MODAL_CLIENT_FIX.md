# Fixing `nvproxy.frontendFD is not saveable` in modal-client GPU memory snapshots

## The bug, in one paragraph

`gpu_memory_snapshot.py` toggles **CUDA sessions** via `cuda-checkpoint`, then
runs a plain gVisor checkpoint. This works only under a hidden invariant:
*driver-checkpointed processes close their `/dev/nvidia*` FDs, so nothing
GPU-related survives to the sentry's state encoder*. The runsc in production
has stub nvproxy checkpoint support that **unconditionally panics on any open
frontend FD** (`save_restore_impl.go: panic("nvproxy.frontendFD is not
saveable")`). Any process holding a `/dev/nvidiactl` / `/dev/nvidiaN` FD that
is **not a toggleable CUDA session** breaks the invariant:

| FD holder | `cuda-checkpoint --get-state` | Result at snapshot |
|---|---|---|
| CUDA process (toggled) | `running` → toggled | OK — driver closes its FDs |
| NVML-only session (pynvml, torch utilization queries) | fails (`initialization error`) → invisible to the scan | panic (`can't save with live nvproxy clients`) |
| Process mid-`cuInit`, or bare/inherited nvidia FD | fails → invisible | panic (`nvproxy.frontendFD is not saveable`) ← **the prod error** |

SGLang violates the invariant structurally (NVML sessions + JIT/compile
subprocesses that init CUDA at arbitrary times). The failing FD in the prod
trace had `clients: nil, mmapLength: 0` — a bare FD, not a CUDA context.

Reproduction (1 process, `open("/dev/nvidiactl")`, zero ioctls):
`cr-bench/bench_8_cuda_churn.sh` with `MODAL_FLOW=1 RAWFD_WORKERS=1` against a
stub-era runsc reproduces the exact panic; the same run passes on a runsc with
nvproxy C/R support.

## The real fix (platform)

Upgrade the sandbox runtime to a gVisor with **nvproxy checkpoint/restore
support** (frontend/uvm FD serialization + RM object replay; upstream master
has it) and prefer its **native integration**
(`runsc checkpoint --cuda-checkpoint-path=...`) over the client-side toggle:
the sentry enumerates FD holders authoritatively, toggles with the kernel
paused (no scan-to-checkpoint race), and untoggles before user code resumes.
Everything below is mitigation until that lands.

## Client-side mitigations (ordered)

### 1. Pre-flight check: fail fast with a useful error

Before calling the snapshot RPC, verify the invariant instead of letting the
encoder panic:

```python
def nvidia_fd_holders() -> dict[int, list[str]]:
    """pid -> open /dev/nvidia* paths, for all processes."""
    holders = {}
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        links = []
        try:
            for fd in (pid_dir / "fd").iterdir():
                target = os.readlink(fd)
                if target.startswith("/dev/nvidia"):
                    links.append(target)
        except OSError:
            continue  # process exited or permission
        if links:
            holders[int(pid_dir.name)] = links
    return holders
```

After `CudaCheckpointSession.checkpoint()` succeeds, require
`nvidia_fd_holders()` to be **empty**. If not, raise listing pid, comm
(`/proc/<pid>/comm`) and the FD paths. This converts an opaque sentry panic
into "PID 4211 (sglang::scheduler) still holds /dev/nvidiactl (NVML session?)".

### 2. Close the enumeration race (mid-init processes)

Port the stabilization loop from gVisor's native path: after toggling, rescan
`/proc` for **new** CUDA sessions and toggle them too; repeat until a scan
finds nothing new (bounded, e.g. 5 passes with 100ms sleep). A process that
was mid-`cuInit` during the first scan becomes toggleable one pass later.
This also needs the check in (1) as the final gate — a process can still be
pre-`cuInit` yet already hold the FD.

### 3. Eliminate NVML sessions before snapshot (workload contract)

NVML FDs can only be closed by their owning process, so this is a documented
requirement on the workload's pre-snapshot hook:

- Call `pynvml.nvmlShutdown()` (and stop anything polling GPU stats) in the
  snapshot-enter hook.
- torch: avoid `torch.cuda.utilization()` / memory-stats pollers around
  snapshot time; they hold an NVML session for process lifetime.
- Sidecars (`nvidia-smi` loops, DCGM exporters) must not run inside the
  snapshotted container.

For engines that can't comply (SGLang), only the platform fix removes this
class.

### 4. Keep the retry/backoff — and add liveness re-check

The existing `toggle()` retry loop is good (the sentry's native path is
currently *less* forgiving here). One improvement: on toggle failure, re-check
`/proc/<pid>` existence — a process that exited between scan and toggle should
be dropped, not retried to timeout.

## Related pitfall you will hit next (multi-GPU)

Once frontendFDs are handled, TP>1 workloads on driver R580 fail at
**restore** with `operation not supported`: NCCL's host-side cuMem allocations
(`NCCL_CUMEM_HOST_ENABLE=1`, default) back SHM-transport buffers with
FD-exported handles the driver cannot re-create. Mitigation: set
`NCCL_CUMEM_HOST_ENABLE=0` (or `NCCL_CUMEM_ENABLE=0`) in the container env
before NCCL init. SGLang sets `NCCL_CUMEM_ENABLE=0` itself by default, but
`--enable-symm-mem` re-enables it; stock vLLM ≥0.26 sets nothing and fails.
NVIDIA's 610 driver notes claim a fix (unverified here). Flipping the env var
at runtime does not work (NCCL caches params per process); only
pre-initialization env or full communicator teardown/re-init changes behavior.
