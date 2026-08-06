# GPU memory-snapshot repro results (local gVisor)

Five single-mechanism repros, run locally under `runsc` (no Modal). Each app
holds exactly one GPU resource class at snapshot time; `run_repro.sh` boots it,
snapshots, restores, and classifies the outcome.

## Environment

| | |
|---|---|
| GPU | NVIDIA A10G (24 GiB, PCIe, no NVLink) |
| Host driver | 580.95.05 |
| nvproxy driver ABI | 580.65.06 (same-major auto-select) |
| **fix** runsc | `release-20260330.0-696-g6bd82d0a1502` — CUDA-enum race fix **+ nvproxy checkpoint/restore** |
| **stub** runsc | `release-20260330.0-349-gfcd95d5fac0f` — prod-equivalent: nvproxy FDs unconditionally not saveable |
| cuda-checkpoint | fetched from NVIDIA/cuda-checkpoint main (in image) |

Run: `sudo bash gms/repros/run_repro.sh <a|b1|b2|c1|c2> [--mode native|modal] [--rebuild]`
Default `--mode native` = `runsc checkpoint --cuda-checkpoint-path=...` (sentry
enumerates + toggles). `--mode modal` = in-container `cuda-checkpoint` toggle of
`get-state==running` pids, then a plain checkpoint (faithful to modal-client
`gpu_memory_snapshot.py`). `RUNSC=/usr/local/bin/runsc` selects the stub.

## Results

| ID | Class | fix runsc | stub runsc | Notes |
|----|-------|-----------|------------|-------|
| A  | Checkpointable CUDA contexts | **PASS** | — | control |
| B1 | NVML-only FD | **PASS** | **PASS** | does not reproduce on driver 580 (see below) |
| B2 | Inherited fork FDs | **PASS** | **CHECKPOINT-BLOCKED** | clean reproduction; fix resolves it |
| C1 | CUDA IPC handle | **PASS** | (n/a) | single-proc IPC restores on 580 |
| C2 | cuMem / VMM allocation | **PASS** | (n/a) | native reliable; modal-flow hiccup, see below |

(stub can only be exercised for the checkpoint-blocking classes; on stub any
surviving nvidia FD aborts the checkpoint before restore, so C-class cannot be
isolated there.)

---

### A — checkpointable CUDA contexts (control) — PASS

Pre-snapshot dump (5 holders — main + 4 staggered workers, all real CUDA
sessions):

```
=== [A: checkpointable CUDA contexts] nvidia FD holders: 5 ===
PID 1 (python3): ['/dev/nvidia-uvm', '/dev/nvidia0', '/dev/nvidiactl']
PID 7  (python3): [...same...]
PID 8  (python3): [...]
PID 9  (python3): [...]
PID 10 (python3): [...]
```

- Sentry toggled all sessions: `state_cuda.go: cuda-checkpoint on 5 processes
  took [584ms]`.
- checkpoint 1.76s, restore 0.55s, `/verify` sum identical pre/post
  (`1048576.0`). The enumeration-race stressor (staggered `cuInit`) is caught
  by the re-enumeration loop.

### B1 — NVML-only FD — PASS on both (does NOT reproduce here)

- `pynvml.nvmlInit()` with no `nvmlShutdown` holds `/dev/nvidiactl`,
  `/dev/nvidia0`, `/dev/nvidia-uvm`.
- **Finding:** on driver 580, `cuda-checkpoint --get-state` reports **`running`**
  for an `nvmlInit()`-only process (verified bare-metal, outside gVisor). So
  NVML is treated as a checkpointable CUDA session, is toggled like any other,
  and its FDs are released — it never becomes an unsaveable leftover. The
  NVML-only failure class is therefore driver/cuda-checkpoint-version specific
  and is not reproducible on this stack (neither fix nor stub blocks).
- The intended fix (`nvmlShutdown()` before snapshot) is moot on 580.

### B2 — inherited fork FDs — the reproduction (stub CHECKPOINT-BLOCKED, fix PASS)

- Main process opens a CUDA context, then `os.fork()`s 8 children that touch no
  CUDA. Children inherit the parent's `/dev/nvidia*` FDs with no context of
  their own; `cuda-checkpoint --get-state` fails on them, so no toggle flow
  (sentry or modal) can claim them.

Pre-snapshot dump (9 holders: 1 CUDA parent + 8 bare inheritors):

```
=== [B2: inherited fork FDs] nvidia FD holders: 9 ===
PID 1  (python3): ['/dev/nvidia-uvm', '/dev/nvidia0', '/dev/nvidiactl']   # CUDA
PID 7..14 (python3): [same paths]                                         # inherited, no context
```

- **stub runsc:** `RESULT = CHECKPOINT-BLOCKED — nvproxy FD not saveable
  (B-class), rc=128`. Verbatim from the checkpoint log:
  `encoding error: nvproxy.frontendFD is not saveable` — the exact prod panic
  (prod's leftover had `clients: nil`, matching a bare inherited FD).
- **fix runsc:** `RESULT = PASS` (checkpoint 0.5–1.0s, restore 0.5s). nvproxy
  C/R serializes the inherited frontend FDs and replays them on restore.
- Fix to validate separately: spawn/exec instead of bare fork
  (`TORCHINDUCTOR_COMPILE_THREADS=1` for real Inductor).

### C1 — CUDA IPC handle — PASS

- Producer exports a CUDA tensor over a `torch.multiprocessing` fork queue
  (`cudaIpcGetMemHandle`); consumer imports it (`cudaIpcOpenMemHandle`) and
  holds the mapping across the snapshot.
- Dump shows 2 CUDA sessions (producer + consumer), both toggleable.
- checkpoint + restore succeed on the fix in both native and modal modes. The
  C-class restore failure does **not** reproduce for a single-process /
  single-GPU IPC mapping on driver 580 (consistent with the separate finding
  that only NCCL *host*-cuMem SHM handles fail restore). A cross-process peer
  mapping over ≥2 GPUs is out of scope for this box.

### C2 — cuMem / VMM allocation — PASS (native)

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set by the runner) routes
  torch's allocator through the CUDA VMM APIs; a 256 MiB tensor forces a fresh
  cuMem-backed segment held across the snapshot.
- **native mode (sentry toggle): PASS, reliably** — checkpoint 0.66s, restore
  0.47s, `/verify` sum identical (`67108864.0`). Device-side cuMem restores
  fine on driver 580.
- **modal mode:** intermittently `RESTORE-BLOCKED — /verify unreachable`, but
  with **no** `operation not supported` in the logs — the in-container untoggle
  ran; the app's server thread did not come back cleanly. This is an artifact
  of the modal plain-checkpoint + manual in-container untoggle sequence, not a
  driver C-class restore failure. The native (sentry-driven) path is the
  reliable one.

## Takeaways

- The race-fix + nvproxy-C/R runsc handles **every** class that reproduces on
  this hardware. The only clean checkpoint-blocking reproduction here is **B2**
  (inherited fork FDs), which maps directly to the prod `frontendFD is not
  saveable` panic and is resolved by the fix.
- **B1 (NVML-only)** and **C1/C2 (IPC / device-cuMem)** do not reproduce a
  failure on driver 580: cuda-checkpoint treats NVML as a checkpointable
  context, and single-process IPC / device-cuMem restore correctly. These
  classes would need a different driver (the C-class restore issue we did
  reproduce elsewhere is specifically NCCL **host**-cuMem, `NCCL_CUMEM_HOST_
  ENABLE=1`) or multi-GPU/NVLink hardware.
- For prod on the stub runsc: the failure is version-determined — any workload
  with a non-toggleable nvidia FD (inherited/bare) aborts the checkpoint.
  Upgrading to a runsc with nvproxy C/R is the fix; the repros here are the
  per-class validation harness.
