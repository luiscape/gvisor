# Plan: vLLM TP=4 + NVLS + compile/CUDA-graphs + patched NCCL, e2e C/R under gVisor

Goal: prove the NCCL-fork checkpoint path on a **real engine**: Qwen on 4
GPUs, NVLS multicast enabled, torch.compile + piecewise CUDA graphs enabled
(vLLM defaults), `runsc checkpoint`/`restore`, post-restore inference matches.
This is the config PROGRESS.md's TL;DR marked ❌ ("cuda-checkpoint cannot
restore it") — the fork + job mode is the bet that overturns it.

Existing assets (nothing needs rebuilding from scratch):
- `cr-bench/bench_4_vllm_multi.sh` + `_bench_vllm_impl.sh` — full lifecycle
  (boot → reference inference → /sleep → checkpoint → restore → /wake_up →
  verify), with knobs: `NCCL_NVLS_ENABLE`, `NCCL_CUMEM_ENABLE`,
  `DISABLE_CUSTOM_ALL_REDUCE`, `VLLM_ALLREDUCE_USE_SYMM_MEM`, `EAGER`,
  `IMAGE`, `CUDA_CKPT_JOB_FILE=1` (job wrapping), `CUDA_CKPT_SEQUENTIAL=1`.
- `cr-bench/images/Dockerfile.vllm` — torch cu126 + vLLM + Qwen2.5-0.5B/1.5B
  pre-downloaded + cuda-checkpoint + sleep-server.
- Patched NCCL 2.30.7 (`nccl/` working tree; `NCCL_PATCH.md`,
  `nccl-nvls-suspend.patch`): `ncclCommSuspend(MEM)` now covers NVLS; the
  upstream part already covers cross-rank UC imports.
- runsc built from this tree (`runsc-phase0`), driver 610.57.04, 8×H100.

## The one missing piece: a suspend trigger

Stock vLLM/torch never call `ncclCommSuspend`. Torch's ProcessGroupNCCL does
not expose `ncclComm_t` to Python, so an engine-side hook can't reach every
comm (vLLM's pynccl comm is reachable via ctypes, torch's is not — and both
exist per worker). ⇒ Put the trigger **in the fork**, mirroring mcshim's
proven control protocol:

**Fork addition (test-scoped, env-gated `NCCL_CKPT_CTRL_DIR`):**
- a global registry of live comms (append at init-rank completion, remove at
  destroy/abort; NCCL has no public enumeration),
- a control thread (started when the first comm registers) polling the dir:
  - `suspend` appears → `ncclCommSuspend(comm, NCCL_SUSPEND_MEM)` on every
    registered comm → ack `nccl-suspended.<pid>`,
  - `suspend` removed → `ncclCommResume` on all → ack `nccl-resumed.<pid>`,
  - failures → `nccl-error.<pid>` + a WARN, no retry storm (edge-triggered).
- This is the `ncclCommsSuspendAll` shape. Resume runs concurrently in every
  worker process by construction (one control thread each), satisfying the
  collective-resume requirement (`cuMulticastBindMem` blocks until all join).

**Bench addition (env-gated in `_bench_vllm_impl.sh`, e.g.
`NCCL_SUSPEND_HOOK=1`):** between `/sleep` and `runsc checkpoint`:
`runsc exec touch $CTRL_DIR/suspend` + wait for TP-many acks; after restore,
`runsc exec rm` + wait for resumed-acks, then `/wake_up`. (Ack counting =
`runsc exec ls`, same as run_mcshim_mp_gvisor.sh.)

## Steps

0. **Preflight** (minutes): persistence mode on (systemd unit, done), fabric
   manager active, `runsc-phase0` present, ~50GB disk free (940GB free ✓),
   Docker ✓. Host network needed once for image build.
1. **Build the base image** (~30–60 min, multi-GB downloads):
   `cd cr-bench && sudo docker build -t cr-bench-vllm -f images/Dockerfile.vllm .`
2. **Extend the fork** with the `NCCL_CKPT_CTRL_DIR` control thread
   (~100 lines: registry + thread + marker protocol); rebuild in the CUDA-13
   container (recipe in `NCCL_PATCH.md`); refresh
   `nccl-nvls-suspend.patch` afterwards. *(implementation item #1)*
3. **Overlay image** `cr-bench-vllm-ncclfork` (small Dockerfile,
   `FROM cr-bench-vllm`):
   - `COPY libnccl.so.2 /opt/ncclfork/libnccl.so.2`
   - `ENV LD_PRELOAD=/opt/ncclfork/libnccl.so.2` (torch links libnccl
     dynamically → preload wins for the soname)
   - `ENV VLLM_NCCL_SO_PATH=/opt/ncclfork/libnccl.so.2` (vLLM's pynccl
     dlopens explicitly and honors this env)
   - `ENV NCCL_CKPT_CTRL_DIR=/tmp/ncclckpt NCCL_DEBUG=INFO
     NCCL_DEBUG_SUBSYS=INIT,NVLS`
4. **Bench hook** from above in `_bench_vllm_impl.sh`. *(implementation #2)*
5. **Boot-only smoke** (no checkpoint): boot TP=4 under runsc, confirm in the
   app log (a) NCCL version banner says **2.30.7** (fork loaded, not the
   torch-bundled one), (b) NVLS engaged (`NCCL INFO NVLS` lines), (c) a
   manual marker touch produces TP-many `nccl-suspended.*` acks and the log
   shows the patch's `NVLS Suspend ... released` lines, and resume works.
   This isolates fork-loading problems from checkpoint problems.
6. **Test matrix** (all: TP=4, GPUs 0–3, Qwen2.5-1.5B-Instruct, job mode):

   ```
   COMMON: sudo RUNSC=/usr/local/bin/runsc-phase0 CUDA_CKPT_JOB_FILE=1 \
     CUDA_CKPT_SEQUENTIAL=1 IMAGE=cr-bench-vllm-ncclfork REBUILD_ROOTFS=1 \
     NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=1 DISABLE_CUSTOM_ALL_REDUCE=1 \
     VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
     bash cr-bench/bench_4_vllm_multi.sh --gpus 0,1,2,3 --tp 4
   ```
   - **Leg A (control)**: hook OFF → checkpoint must FAIL/hang on live NVLS
     (proves NVLS is really engaged and really the blocker).
   - **Leg B (main)**: hook ON, `EAGER=0` (compile + piecewise cudagraphs,
     the ❌ config) → want full PASS: checkpoint rc=0, restore rc=0,
     post-restore inference matches reference.
   - **Leg C (bisect, only if B fails)**: hook ON, `EAGER=1` — separates
     "NCCL/multicast layer still broken" (C fails too) from "compile/graph
     state still unrestorable" (C passes, B fails).
   - Optional: TP=2 (no NVLS at 2 GPUs — direct P2P) as a cheaper sanity leg.

## Pass criteria
- Leg A blocked; Leg B: rc=0/rc=0, `nccl-suspended/resumed` acks = TP count,
  patch's `NVLS Suspend/Resume` INFO lines present, post-restore completion
  matches the pre-checkpoint reference, no worker crash in applog.

## Known risks / what to watch
1. **The old ❌ finding.** PROGRESS.md's native A/B failed restore
   (`"invalid argument"`) even fabric-free with compile+cudagraphs on. Our
   new evidence reframes it: that signature is exactly the
   live-VMM-UC-import failure (`ipc_taint --mode hold` under job mode), and
   that A/B predates job-wrapping + suspend. `ncclCommSuspend` releases
   NCCL's VMM imports; job mode covers legacy cuIpc. If Leg B still fails
   with the same signature, the census tooling (`run_census.sh`, blocker
   inventory) identifies which process/object class remains — that result
   is valuable either way.
2. **`refCount > 1` NVLS sharing.** torch/vLLM may `ncclCommSplit` subgroups
   that share NVLS resources; the patch refuses suspend then
   (`ncclInvalidUsage`, see NCCL_PATCH.md limitations). Watch for its WARN
   in the applog; if hit, that's a fork gap to close (suspend once per
   shared resource, resume once).
3. **NVLS user-buffer registration** (`cuMulticastBindAddr` regRecords) is
   not covered by the patch; vLLM piecewise-cudagraph capture may register
   buffers (`NCCL_LOCAL_REGISTER`). If suspend leaves 00FD objects alive,
   the checkpoint blocker inventory will name them; mitigation for the test:
   `NCCL_LOCAL_REGISTER=0` / `NCCL_GRAPH_REGISTER=0`, noted as a fork gap.
4. **Version mix**: fork is NCCL 2.30.7 built with CUDA 13 (static cudart,
   sm_90) preloaded into a torch-cu126 stack — public-ABI compatible
   (`libnccl.so.2`), verified in step 5 before any checkpoint is attempted.
5. vLLM sleep-mode allocator is VMM-based (process-local, no IPC) —
   single-GPU benches already proved cuda-checkpoint 610 handles it.

## Not in scope (deliberately)
- vLLM custom allreduce and torch symm-mem allreduce stay OFF: their fabric
  IPC is outside NCCL and needs the mcshim path (or engine changes). NVLS —
  the headline blocker — is what this test proves.
- No gVisor/nvproxy code changes; no git operations (fork changes stay as
  working-tree state; refresh the preserved .patch file).
