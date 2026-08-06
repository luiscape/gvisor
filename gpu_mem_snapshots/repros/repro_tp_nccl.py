# repro_tp_nccl.py — Class C3: multi-GPU NCCL tensor-parallel busy-spin.
#
# Minimal, fast reproduction of the vLLM/SGLang tensor-parallel checkpoint
# hang WITHOUT any inference engine (no model load, no torch.compile), so it
# boots in seconds instead of ~150s.
#
# What it reproduces
# ------------------
# It launches one worker process per GPU (like a TP worker group). Each worker
# joins an NCCL communicator (torch.distributed, backend="nccl") and runs a
# warmup all-reduce, which establishes both the NCCL communicator AND the
# intra-node CUDA IPC mappings the ranks use to exchange data. The workers then
# stay COUPLED and BUSY-SPINNING on continuous all-reduce (REPRO_MODE=spin,
# default) — the same state the vLLM TP workers are in at checkpoint time
# (pinned at ~99% CPU on their NCCL/IPC peer wait).
#
# Under `runsc checkpoint --cuda-checkpoint-path=...` (with the --launch-job
# wrap, i.e. CUDA_CKPT_JOB_FILE=1), the sentry runs `cuda-checkpoint --toggle`
# on every CUDA process. cuda-checkpoint cannot quiesce a rank that is
# spin-waiting on its still-running peer, so the toggle of the coupled workers
# never completes and the checkpoint hangs — reproducing the multi-GPU blocker
# that the single-GPU repros (a/b1/b2/c1/c2) do not exhibit.
#
# Modes (REPRO_MODE env):
#   spin (default) — workers run continuous all-reduce (unambiguously coupled +
#                    busy). Most reliable reproduction of the hang.
#   idle           — workers do one warmup all-reduce then wait on a barrier and
#                    sleep. NCCL stays initialized (its proxy/watchdog threads
#                    may still spin). Use this to bisect "engine busy" vs "NCCL
#                    initialized" as the trigger.
#   graph          — workers capture a CUDA graph CONTAINING a cross-GPU NCCL
#                    all-reduce, then stay idle (lockable). This mimics the
#                    captured-graph + coupled-NCCL state of a vLLM/SGLang TP
#                    worker without loading a model, to test whether
#                    cuda-checkpoint can round-trip that state on RESTORE (the
#                    pure-NCCL idle/spin modes cannot reproduce that). /verify
#                    replays the captured collective once and checks the result.
#   compile        — workers torch.compile (Inductor) a compute region so the
#                    CUDA context holds JIT-compiled/loaded modules (cubins),
#                    combined with a cross-GPU NCCL all-reduce, then stay idle.
#                    Isolates torch.compile x multi-GPU as the restore trigger
#                    (single-GPU `inductor` already round-trips).
#
# Requires >= 2 GPUs exposed to the container. The worker count is auto-detected
# from torch.cuda.device_count(), so `run_repro.sh tp --gpus 0,1` gives TP=2 and
# `--gpus 0,1,2,3` gives TP=4.
#
# App contract (via _harness.serve): GET /health once workers are ready, GET
# /verify returns liveness + the all-reduce result (which must equal the world
# size, since every rank contributes a tensor of ones).

import ctypes
import os
import time
import multiprocessing as mp

MASTER_PORT = int(os.environ.get("TP_MASTER_PORT", "29500"))
TENSOR_ELEMS = int(os.environ.get("TP_TENSOR_ELEMS", str(4 * 1024 * 1024)))  # 16 MiB f32
REPRO_MODE = os.environ.get("REPRO_MODE", "spin").strip().lower()


def _worker(rank, world_size, ready_q, stop_evt, iters, elem, replay_gen):
    """One TP worker: bind to a GPU, join NCCL, then stay coupled + busy."""
    # Heavy/GPU imports belong in the worker, not at module import time.
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # Inside the container the exposed GPUs are always cuda:0..cuda:(N-1),
    # regardless of the host indices passed via --gpus.
    torch.cuda.set_device(rank)

    # Match the vLLM/SGLang benchmark default: classic allocations, not VMM, so
    # cuda-checkpoint's coverage is comparable.
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    x = torch.ones(TENSOR_ELEMS, device="cuda")

    # Warmup all-reduce: builds the NCCL communicator + intra-node CUDA IPC
    # mappings between the ranks. After this the ranks are coupled.
    dist.all_reduce(x)
    torch.cuda.synchronize()

    print(f"[rank={rank} pid={os.getpid()}] NCCL ready on cuda:{rank}, "
          f"warmup all-reduce elem={x[0].item():.0f} (expect {world_size})",
          flush=True)

    if REPRO_MODE == "graph":
        # vLLM/SGLang-like: capture a CUDA graph that CONTAINS a cross-GPU NCCL
        # collective, then stay idle. This is the captured-graph + coupled-NCCL
        # state that the pure-NCCL idle/spin modes do NOT exercise, and the
        # thing cuda-checkpoint must round-trip on restore. The worker stays
        # idle (lockable) at checkpoint and replays the graph on demand so
        # /verify can confirm the collective still runs after restore.
        static_in = torch.ones(TENSOR_ELEMS, device="cuda")
        static_out = torch.empty_like(static_in)

        # Warm up the op (incl. NCCL) on a side stream before capture, per the
        # torch CUDA-graph-with-collectives recipe.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                static_out.copy_(static_in)
                static_out.mul_(2.0)
                dist.all_reduce(static_out)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        dist.barrier()  # all ranks capture in lockstep

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out.copy_(static_in)
            static_out.mul_(2.0)
            dist.all_reduce(static_out)

        for _ in range(3):  # establish steady state
            g.replay()
        torch.cuda.synchronize()
        if rank == 0:
            elem.value = float(static_out[0].item())  # expect 2 * world_size
        print(f"[rank={rank} pid={os.getpid()}] CUDA graph captured, replay "
              f"elem={static_out[0].item():.0f} (expect {2 * world_size})",
              flush=True)
        ready_q.put(rank)

        # Keep the graph + its static buffers alive across the snapshot. Stay
        # idle so cuda-checkpoint can lock at checkpoint time; replay only when
        # /verify bumps replay_gen (all ranks replay the collective together).
        _keep = (g, static_in, static_out)  # noqa: F841
        dist.barrier()
        last_gen = 0
        while not stop_evt.is_set():
            gen = replay_gen.value
            if gen != last_gen:
                last_gen = gen
                g.replay()
                torch.cuda.synchronize()
                if rank == 0:
                    elem.value = float(static_out[0].item())
                with iters.get_lock():
                    iters.value += 1
            else:
                time.sleep(0.1)
        return

    if REPRO_MODE == "compile":
        # vLLM-like: torch.compile (Inductor) a compute region so the CUDA
        # context holds dynamically JIT-compiled + loaded modules (cubins),
        # combined with a cross-GPU NCCL all-reduce. Single-GPU `inductor`
        # already round-trips, so this isolates torch.compile x multi-GPU as
        # the restore trigger. Stays idle (lockable) at checkpoint; replays on
        # demand for /verify.
        dim = int(os.environ.get("TP_COMPILE_DIM", "1024"))
        w = torch.randn(dim, dim, device="cuda")
        inp = torch.ones(dim, dim, device="cuda")

        def step(x):
            y = torch.relu(x @ w)
            dist.all_reduce(y)
            return y

        cstep = torch.compile(step)
        for _ in range(3):  # trigger compilation + steady state
            out = cstep(inp)
        torch.cuda.synchronize()
        if rank == 0:
            elem.value = float(out[0, 0].item())
        print(f"[rank={rank} pid={os.getpid()}] torch.compile ready, "
              f"elem={out[0, 0].item():.3f}", flush=True)
        ready_q.put(rank)

        _keep = (w, inp)  # noqa: F841
        dist.barrier()
        last_gen = 0
        while not stop_evt.is_set():
            gen = replay_gen.value
            if gen != last_gen:
                last_gen = gen
                out = cstep(inp)
                torch.cuda.synchronize()
                if rank == 0:
                    elem.value = float(out[0, 0].item())
                with iters.get_lock():
                    iters.value += 1
            else:
                time.sleep(0.1)
        return

    ready_q.put(rank)

    if REPRO_MODE == "idle":
        # NCCL stays initialized; proxy/watchdog threads may still spin.
        dist.barrier()
        while not stop_evt.is_set():
            time.sleep(0.5)
        return

    # REPRO_MODE == "spin": stay coupled + busy on continuous collectives.
    n = 0
    while not stop_evt.is_set():
        x.fill_(1.0)
        dist.all_reduce(x)
        torch.cuda.synchronize()
        n += 1
        if rank == 0 and (n & 0x3FF) == 0:
            with iters.get_lock():
                iters.value = n
            elem.value = float(x[0].item())


def setup():
    import torch

    world = torch.cuda.device_count()
    if world < 2:
        raise RuntimeError(
            f"repro_tp_nccl needs >= 2 GPUs but the container sees {world}; "
            "run with e.g. `run_repro.sh tp --gpus 0,1`")

    ctx = mp.get_context("spawn")
    ready_q = ctx.Queue()
    stop_evt = ctx.Event()
    iters = ctx.Value(ctypes.c_ulonglong, 0)
    elem = ctx.Value(ctypes.c_double, 0.0)
    # Generation counter that /verify bumps to request one on-demand graph
    # replay from every worker (REPRO_MODE=graph).
    replay_gen = ctx.Value(ctypes.c_ulonglong, 0)

    procs = []
    for rank in range(world):
        p = ctx.Process(
            target=_worker,
            args=(rank, world, ready_q, stop_evt, iters, elem, replay_gen),
            daemon=True,
        )
        p.start()
        procs.append(p)

    # Wait for every worker to finish its warmup all-reduce (i.e. NCCL + IPC
    # established) before reporting /health.
    ready = set()
    deadline = time.time() + 300
    while len(ready) < world and time.time() < deadline:
        try:
            ready.add(ready_q.get(timeout=5))
        except Exception:  # noqa: BLE001
            for p in procs:
                if not p.is_alive():
                    raise RuntimeError(
                        f"TP worker exited during setup (exitcode={p.exitcode})")
    if len(ready) < world:
        raise RuntimeError(f"only {len(ready)}/{world} TP workers became ready")

    print(f"[pid={os.getpid()}] all {world} TP workers ready "
          f"(mode={REPRO_MODE})", flush=True)
    return {"procs": procs, "stop": stop_evt, "world": world,
            "iters": iters, "elem": elem, "replay_gen": replay_gen}


def verify(state):
    world = state["world"]
    alive = sum(1 for p in state["procs"] if p.is_alive())

    if REPRO_MODE in ("graph", "compile"):
        # Request one fresh replay from all ranks and confirm the captured/
        # compiled cross-GPU collective still runs after restore (this is what
        # fails if cuda-checkpoint could not round-trip the graph/module +
        # NCCL state). All ranks must replay in lockstep (the all-reduce syncs
        # them), so iters must advance by world_size.
        prev = int(state["iters"].value)
        state["replay_gen"].value += 1
        deadline = time.time() + 30
        while int(state["iters"].value) < prev + world and time.time() < deadline:
            time.sleep(0.1)
        replayed = int(state["iters"].value) >= prev + world
        ok = bool(replayed and alive == world)
        if REPRO_MODE == "graph":
            # graph doubles a tensor of ones then all-reduces => 2 * world.
            ok = ok and abs(state["elem"].value - 2 * world) < 1e-3
        return {
            "ok": ok,
            "workers": world,
            "alive": alive,
            "replayed_post": replayed,
            "replay_elem": state["elem"].value,
            "mode": REPRO_MODE,
        }

    # Report worker liveness + the all-reduce result. Every rank contributes a
    # tensor of ones, so the reduced per-element value must equal world size.
    return {
        "ok": alive == world,
        "workers": world,
        "alive": alive,
        "iters": int(state["iters"].value),
        "allreduce_elem": state["elem"].value,
        "expected_elem": world,
        "mode": REPRO_MODE,
    }


if __name__ == "__main__":
    from _harness import serve

    serve("C3: NCCL TP busy-spin (multi-GPU)", setup, verify)
