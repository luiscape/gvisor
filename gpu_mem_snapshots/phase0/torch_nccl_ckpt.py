#!/usr/bin/env python3
"""PyTorch tier: torch.distributed NCCL (NVLS) checkpoint/restore workload.

One process per GPU, exactly the tensor-parallel topology an inference engine
uses, but with no engine: just torch.distributed + NCCL. This is the fast
workload for exercising the NCCL NVLS suspend/resume patch end-to-end under
gVisor.

The point of this tier is that the application does **nothing** for the
checkpoint. It never calls ncclCommSuspend -- it cannot, since torch never
exposes ncclComm_t to Python. NCCL's own checkpoint control thread
(NCCL_CKPT_CTRL_DIR, src/misc/ckpt_ctrl.cc) suspends and resumes every
communicator, and its gate blocks collectives while the memory is released.
So this workload deliberately keeps issuing collectives right across the
checkpoint, with no pause: if the gate did not work, the run would fault.

Each rank:
  * init_process_group("nccl"), large buffers so NCCL selects NVLS,
  * a warmup allreduce, then a captured CUDA graph of an allreduce,
  * a verification loop replaying the graph and checking the exact sum.

Every rank contributes a fixed value (rank+1) so the expected sum is constant
and independent of loop skew between ranks; recv is pre-filled with a sentinel
each iteration so a no-op collective is caught rather than passing silently.

Status is a per-rank line in <dir>/status.<rank>; the runner polls it.
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def write_status(d, rank, line):
    tmp = os.path.join(d, f"status.{rank}.tmp")
    with open(tmp, "w") as f:
        f.write(line + "\n")
    os.replace(tmp, os.path.join(d, f"status.{rank}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/torchnccl")
    ap.add_argument("--numel", type=int, default=32 * 1024 * 1024,
                    help="elements per buffer (128MB fp32; large enough that "
                         "NCCL picks NVLS on NVSwitch)")
    ap.add_argument("--seconds", type=int, default=600)
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("--no-graph", action="store_true",
                    help="skip CUDA graph capture (eager collectives only)")
    ap.add_argument("--symm-mem", action="store_true",
                    help="also allocate a torch symmetric-memory tensor. This "
                         "creates a multicast owner NCCL does not know about, "
                         "so ncclNvlsSuspendCheck must REJECT the suspend "
                         "(rejection-path test)")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")

    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    log(rank, f"init_process_group done (world={world}, dev={dev})")

    # Optional: a second, non-NCCL multicast owner. ncclNvlsSuspendCheck is
    # expected to refuse to suspend while this exists.
    symm_t = None
    if args.symm_mem:
        import torch.distributed._symmetric_memory as symm_mem
        group_name = dist.group.WORLD.group_name
        if hasattr(symm_mem, "enable_symm_mem_for_group"):
            symm_mem.enable_symm_mem_for_group(group_name)
        symm_t = symm_mem.empty(1024 * 1024, dtype=torch.float32, device=dev)
        symm_mem.rendezvous(symm_t, group=group_name)
        log(rank, "symmetric memory rendezvous done (expect suspend REJECT)")

    val = float(rank + 1)
    expected = float(sum(r + 1 for r in range(world)))
    SENTINEL = -12345.0

    buf = torch.empty(args.numel, dtype=torch.float32, device=dev)

    def collective():
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)

    # Warm up: NCCL establishes NVLS connections and registers buffers lazily,
    # and none of that can happen during graph capture.
    for _ in range(3):
        buf.fill_(val)
        collective()
    torch.cuda.synchronize()
    dist.barrier()

    graph = None
    if not args.no_graph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            buf.fill_(val)
            collective()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            collective()
        torch.cuda.synchronize()
        log(rank, "CUDA graph captured")

    ptr = buf.data_ptr()
    write_status(args.dir, rank, f"READY pid={os.getpid()} data_ptr=0x{ptr:x} "
                                 f"graph={graph is not None} expected={expected}")
    log(rank, f"READY data_ptr=0x{ptr:x} expected={expected}")

    t0 = time.monotonic()
    it = 0
    failures = 0
    paused = False
    restored = os.path.exists(os.path.join(args.dir, "restored"))
    while time.monotonic() - t0 < args.seconds:
        it += 1
        if not restored and os.path.exists(os.path.join(args.dir, "restored")):
            restored = True
            log(rank, f"RESTORE-DETECTED iter={it}")

        # Application-level quiesce. NCCL's own gate stops collectives
        # submitted through its API, but a captured CUDA graph is replayed by
        # the driver without re-entering NCCL, so a graph-replaying workload
        # must stop itself. A real engine does this in its sleep hook; here a
        # marker stands in for it.
        if os.path.exists(os.path.join(args.dir, "pause")):
            if not paused:
                torch.cuda.synchronize()
                paused = True
                write_status(args.dir, rank, f"PAUSED iter={it} failures={failures}")
                log(rank, f"paused at iter={it}")
            time.sleep(0.1)
            continue
        if paused:
            paused = False
            log(rank, f"unpaused at iter={it}")

        # Pre-fill with a sentinel so a collective that silently does nothing
        # is caught instead of passing on stale data.
        buf.fill_(SENTINEL)
        buf.fill_(val)
        if graph is not None:
            graph.replay()
        else:
            collective()
        torch.cuda.synchronize()

        got_min = buf.min().item()
        got_max = buf.max().item()
        ok = (got_min == expected and got_max == expected)
        if buf.data_ptr() != ptr:
            ok = False
        if not ok:
            failures += 1

        tag = "post-restore" if restored else "pre-checkpoint"
        write_status(args.dir, rank,
                     f"iter={it} {tag} {'pass' if ok else 'FAIL'} "
                     f"min={got_min} max={got_max} want={expected} "
                     f"failures={failures} data_ptr=0x{buf.data_ptr():x}")
        time.sleep(args.interval)

    write_status(args.dir, rank, f"DONE iters={it} failures={failures} "
                                 f"restored={restored}")
    dist.barrier()
    dist.destroy_process_group()
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
