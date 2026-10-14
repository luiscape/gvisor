# GPU validation on B300: gVisor checkpoint hook + mcshim

Instructions for an agent on an 8x B300 host. Dev-only; not part of any PR.

## Goal

Validate, on real GPUs, the new design for multi-GPU CUDA checkpoint/restore
in gVisor. Until now it has only been tested on CPU. The previous design (the
sentry drove the interposer itself) passed this matrix on B300 with driver
610.57.04; this run is for the new design:

- **gVisor** (`github.com/luiscape/gvisor`, branch
  `luis/multicast-xgpu-upstream`, at `702e62ba6` or later):
  - `--cuda-checkpoint-job=launch` wraps GPU containers in
    `cuda-checkpoint --launch-job`.
  - `--checkpoint-hook=PATH` runs a program inside each GPU container at
    `checkpoint`, `restore [--device-map MAP]` and `resume`, instead of
    gVisor's built-in `cuda-checkpoint` handling. Page faults of GPU
    processes are held from before `checkpoint` until after `restore`.
  - nvproxy reports `nvproxy_exported_object: client=... object=...` in
    `/proc/[pid]/fdinfo/[fd]` for exported fds.
  - Design: `mcshim-plan.md`; proposal:
    `g3doc/proposals/cuda_multigpu_checkpoint.md`.
- **mcshim** (`github.com/modal-projects/mcshim`, `main` at `f6b2dfa` or
  later): the interposer `mcshim.so` (C, `LD_PRELOAD`) and the hook
  `mcshim-ckpt` (static Go). README and `docs/design.md` describe both.

## Rules

- Do not push to either repository. If you find a bug, write it up (section
  9) with a proposed patch as a diff in the report; apply it locally only to
  continue testing, and say so.
- Record exact versions, commands and timings. A result without its command
  line is not useful.
- Stop and report at the first failure of section 3 (the smoke test): later
  sections depend on it.
- Use `make` in gVisor (`make build TARGETS=//runsc`), not bare `bazel`.

## 1. Environment check (record everything)

```sh
uname -m                     # must be x86_64: gVisor GPU checkpoint does not support arm64
nvidia-smi                   # 8x B300, driver version
nvidia-smi -q | grep -i -A3 fabric        # fabric state: Completed / Success
systemctl status nvidia-fabricmanager     # running, same version as the driver
nvidia-smi topo -m           # NVLink between all GPUs
cuda-checkpoint --help | grep -E -- '--launch-job|--action|--device-map'
cat /sys/kernel/mm/transparent_hugepage/shmem_enabled
docker info >/dev/null && go version && python3 --version
```

Requirements: driver R610 or newer (R580+ for `--device-map`), fabric manager
running, `cuda-checkpoint` with `--launch-job`. If the driver is not R610+,
stop: the job wrap is disabled below R610. Set
`transparent_hugepage/shmem_enabled=always` if you are allowed to (it affects
restore time, not correctness), and note whether you did.

## 2. Build

```sh
# gVisor
cd gvisor && git checkout luis/multicast-xgpu-upstream && git log --oneline -1
make build TARGETS=//runsc
RUNSC=$(realpath bazel-out/k8-fastbuild/bin/runsc/runsc_/runsc)
$RUNSC --cuda-checkpoint-path=/x --cuda-checkpoint-job=launch --checkpoint-hook=/h flags >/dev/null && echo flags ok

# mcshim
cd ../mcshim && git log --oneline -1
make            # mcshim.so (glibc floor checked) and mcshim-ckpt
make test       # CPU: glibc floor, preload, noeffect, tables, go test
make test-docker
file mcshim-ckpt   # must say "statically linked"
```

All must pass before continuing.

## 3. Smoke test: one checkpoint and restore through the hook

This checks what CPU tests cannot. Do it before anything else.

Use mcshim's runner, which builds an OCI bundle and runs runsc with the
deployment the design expects (read `sandboxed()`, `runsc()` and `use_hook()`
in `test/run.sh`):

```sh
cd mcshim
MCSHIM_TEST_GPUS=0,1 test/run.sh -r "$RUNSC" ipc torch-kernel torch-symm \
  -i <rootfs with PyTorch> -e <its env file>
```

(`-i`/`-e`: a rootfs with PyTorch and the host driver's userspace, e.g. an
exported `nvcr.io/nvidia/pytorch` image; see the header of `test/run.sh`.)

If the runner's tests do not exercise a full checkpoint/restore through the
hook, do one by hand with a two-rank PyTorch program that uses symmetric
memory or NCCL with NVLS, under `runsc --nvproxy --cuda-checkpoint-path=...
--cuda-checkpoint-job=launch --checkpoint-hook=/mnt/mcshim-ckpt`, with
`LD_PRELOAD=/mnt/mcshim.so` in the spec and `/etc/ld.so.preload` bind-mounted
read-only from a file containing `/mnt/mcshim.so`:
`runsc checkpoint --image-path=DIR <id>`, then `runsc restore
--image-path=DIR --bundle=... <id2>`, and check the program's results after
restore.

Run runsc with `--debug --debug-log=/tmp/runsc/ --strace=false` and collect,
for the report:

1. **Process discovery.** While the workload runs:
   `runsc exec <id> sh -c 'for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep nvidia && echo "^ $p"; done'`.
   Record exactly what the `/dev/nvidia*` links look like. `mcshim-ckpt` finds
   CUDA processes by a link starting with `/dev/nvidia` or a character device
   of major 195.
2. **Preload.** `runsc exec <id> cat /etc/ld.so.preload`, and
   `grep mcshim /proc/<pid>/maps` for every rank. Note whether the bind mount
   needed a mount point in the rootfs (the runner copies `/etc` for that).
3. **Hook phases.** In the debug log: `checkpoint hook checkpoint in container
   ... took [...]; status ...`, then after restore `checkpoint hook restore`
   and `checkpoint hook resume`, in that order. The hook's own lines start
   with `mcshim-ckpt: `.
4. **Page-fault ordering.** The interposer's rebuild runs in `resume`, after
   page faults are released. If the ordering were wrong, ranks fail after
   restore with `CUDA_ERROR_ILLEGAL_ADDRESS` (700). Pass = identical results
   after restore and no CUDA error.
5. **Job wrap.** `Wrapped container ... in a cuda-checkpoint job` in the log.
6. **fdinfo.** For an exported fd in a rank:
   `runsc exec <id> cat /proc/<pid>/fdinfo/<fd>` shows
   `nvproxy_exported_object:	client=0x... object=0x...`.

## 4. mcshim test suite on GPUs

```sh
cd mcshim
MCSHIM_TEST_GPUS=0,1 test/run.sh abi gate mc refcount refuse mapwait silent
MCSHIM_TEST_GPUS=0,1 test/run.sh -r "$RUNSC" ipc deadline reason undo optout
MCSHIM_TEST_GPUS=0,1 test/run.sh -r "$RUNSC" -i <rootfs> -e <env> \
  torch-kernel torch-symm torch-expandable
```

`rtres` needs `MCSHIM_TEST_CUDART` (two `libcudart.so` of different
majors); run it if you can, plus a renamed copy
(`patchelf --set-soname libcudart-1a2b3c4d.so.12`). `orphan` is expected to
SKIP (a known TODO in `mcshim-ckpt`). `deadline` takes about 2 minutes per
case by design.

## 5. Failure paths through runsc

Use a fake hook (a shell script) in a container with a real CUDA process, so
that runsc runs it:

```sh
#!/bin/sh
echo "$(date +%s.%N) $*" >> /mnt/hook.log
[ "$1" = checkpoint ] && exit "${HOOK_CKPT_STATUS:-1}"
exit 0
```

| Case | Setup | Expect |
| :--- | :--- | :--- |
| Recoverable failure | `HOOK_CKPT_STATUS=1` in the spec env | `runsc checkpoint` fails with the hook's output; `hook.log` shows `checkpoint` then `resume`; the application keeps running |
| Unrecoverable failure | `HOOK_CKPT_STATUS=2` | fails, message says it could not be undone; `hook.log` shows `checkpoint` only |
| Hook missing | `--checkpoint-hook=/nonexistent` | fails with "could not start the checkpoint hook"; application keeps running |
| Docker-style pause | `runsc pause <id>` then checkpoint with the real `mcshim-ckpt` and `--leave-running` (or the equivalent resume path) | succeeds; application runs after resume |
| Failed save with resume | if you can make the save itself fail after the hook succeeded (e.g. image path on a full or read-only filesystem) with `--leave-running` | `restore` and `resume` run; application keeps running |

## 6. Without the hook (must behave like upstream)

| Case | Flags | Expect |
| :--- | :--- | :--- |
| Built-in, parallel | `--cuda-checkpoint-path` only | one `cuda-checkpoint --toggle` per process (debug log), checkpoint and restore succeed for a single-process CUDA program |
| Built-in, job | `--cuda-checkpoint-path --cuda-checkpoint-job=launch`, no hook | wrapped in a job; toggles run sequentially; TP=2 vLLM without NVLS or symmetric memory checkpoints and restores |
| Missing binary | `--cuda-checkpoint-job=launch` with a `--cuda-checkpoint-path` absent from the image | container starts with a warning; `runsc checkpoint` fails naming the container ("GPU containers that do not run in a cuda-checkpoint job") |
| Old driver gate | n/a on R610+ | skip |

## 7. Engine matrix

Serve Qwen2.5-1.5B-Instruct (TP=8: Qwen2.5-3B-Instruct, whose heads divide
by 8) with vLLM 0.29.0 and SGLang 0.5.20 (or the versions available; record
them), in containers that contain `mcshim.so`, `mcshim-ckpt` and
`cuda-checkpoint`, with the flags and preload of section 3.

Each run: boot; record a completion at temperature 0; put the engine to sleep
(vLLM `POST /sleep?level=1`; SGLang `release_memory_occupation`, started with
`--enable-memory-saver --enable-weights-cpu-backup`); checkpoint; restore
(onto other GPUs where listed: give the restored container different
devices); wake (vLLM `/wake_up`; SGLang `resume_memory_occupation`); repeat
the completion. Pass = identical output, on the requested GPUs.

| Workload | GPUs |
| :--- | :--- |
| vLLM TP=2 | 0,1 |
| vLLM TP=4 | 0-3 |
| vLLM TP=8 | 0-7 |
| vLLM TP=2 | 0,1 -> 4,5 |
| vLLM TP=4 | 0-3 -> 4-7 |
| vLLM TP=2 | 0,1 -> 1,2 |
| SGLang TP=4 | 0-3 |
| SGLang TP=4 | 0-3 -> 4-7 |
| SGLang TP=4 `--enable-nccl-nvls` | 0-3, and 0-3 -> 4-7 |
| SGLang TP=4 `--enable-torch-symm-mem` | 0-3, and 0-3 -> 4-7 |
| SGLang TP=4 FlashInfer all-reduce fusion | 0-3, and 0-3 -> 4-7 |
| SGLang TP=8 | 0-7 |

Record per run: checkpoint time and image size, restore time, time to first
inference after restore, and cold-boot time to first inference. The previous
design's B300 numbers are in the proposal's validation table, for comparison.

Then:
- vLLM TP=4, SGLang TP=4 `--enable-torch-symm-mem`, and vLLM TP=2
  0,1 -> 4,5 again **without `CAP_SYS_PTRACE`** in the container.
- Qwen2.5-72B-Instruct at TP=8 on vLLM and SGLang, and restored onto other
  GPUs, if the weights fit and are available.
- Tokens/s with and without the interposer on one configuration that does not
  use CUDA graphs (vLLM `--enforce-eager`), same prompt set, 3 runs each.

## 8. Things to watch for

- A checkpoint under load may fail at the lock step; the application must
  keep running. Retry after putting the engine to sleep.
- `mcshim-ckpt` refuses a CUDA process that does not run the interposer
  (no `present.<pid>` in `/tmp/mcshim`): check that SGLang's worker processes
  map `mcshim.so` (its launcher rewrites `LD_PRELOAD`, which is why
  `/etc/ld.so.preload` is needed).
- `/tmp/mcshim` must be in the container's own filesystem (not a host
  mount): its markers must survive the restore.
- Keep the same `mcshim.so` bytes at the same path for restore.

## 9. Report

Write `REPORT.md` (in the working directory, not committed) with:

1. Environment (section 1 output) and both repositories' commits.
2. One line per test: name, command, PASS/FAIL/SKIP, timings.
3. The section 3 observations (process links, preload, hook phase lines,
   fdinfo line).
4. The engine table with timings.
5. For each failure: the command, the error, the last 50 relevant lines of
   the runsc debug log and of `/tmp/mcshim/mcshim.log`, your diagnosis, and a
   proposed patch (diff) if you have one.
6. Anything in this document that turned out to be wrong.
