#!/usr/bin/env python3
"""
Checkpoint/restore verification harness for NCCL NVLS + PyTorch symmetric memory
inside captured CUDA graphs.

Single node, 4-8 GPUs, Hopper (multicast required).

    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NVLS,REG,ALLOC \
    TORCH_SYMMMEM=CUDA \
    torchrun --nproc_per_node=8 symmem_nccl_ckpt_test.py --seconds 600

Checkpoint and restore the sandbox at any point while it is looping. The
harness verifies continuously and reports:

  * VA inventory (symmetric buffer ptrs, multicast ptr, signal pad ptrs, NCCL
    collective buffer) -- must be byte-identical before and after restore
  * graph replay correctness for a graph containing BOTH an NCCL collective and
    a multimem (multicast) symmetric-memory collective
  * a RESTORE-DETECTED marker when wall clock jumps relative to monotonic clock

Exit code 0 only if every iteration passed and no VA changed.
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------

def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def require_multicast(rank, dev):
    major, _ = torch.cuda.get_device_capability(dev)
    if major < 9:
        raise RuntimeError(
            f"multicast requires Hopper (sm_90+); device reports sm_{major}x. "
            "This harness is Hopper-only by design."
        )


def init_dist():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl")
    return rank, world, local


# --------------------------------------------------------------------------
# allocation
# --------------------------------------------------------------------------

def alloc_symm(numel, dtype, device, group_name):
    """Allocate a symmetric-memory tensor and rendezvous it.

    rendezvous() is what triggers cuMemExportToShareableHandle on the local
    allocation, cuMemImportFromShareableHandle on peers, and (on Hopper)
    cuMulticastCreate + cuMulticastBindMem. This is the call that makes the
    process un-checkpointable today.
    """
    t = symm_mem.empty(numel, dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(t, group=group_name)
    return t, hdl


def va_inventory(rank, symm_t, hdl, nccl_buf):
    """Every pointer that must survive checkpoint/restore unchanged."""
    inv = {
        "symm_tensor_data_ptr": symm_t.data_ptr(),
        "nccl_buf_data_ptr": nccl_buf.data_ptr(),
    }
    for attr in ("multicast_ptr", "buffer_ptrs", "signal_pad_ptrs"):
        val = getattr(hdl, attr, None)
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            for i, p in enumerate(val):
                inv[f"{attr}[{i}]"] = int(p)
        else:
            inv[attr] = int(val)
    return inv


def fmt_inventory(inv):
    return "\n".join(f"    {k:28s} = 0x{v:016x}" for k, v in sorted(inv.items()))


# --------------------------------------------------------------------------
# graph capture
# --------------------------------------------------------------------------

def build_graph(rank, world, symm_t, group_name, nccl_buf, use_multimem):
    """Capture one CUDA graph containing an NCCL collective AND a symmetric
    memory multicast collective.

    Both must be warmed up on a side stream first: NCCL lazily establishes
    connections and registers buffers on first use, and that setup cannot
    happen during capture.
    """
    static_in = torch.empty_like(nccl_buf)
    static_symm_src = torch.empty_like(symm_t)

    def payload():
        # NCCL path. Large enough to select the NVLS algorithm.
        nccl_buf.copy_(static_in)
        dist.all_reduce(nccl_buf, op=dist.ReduceOp.SUM)
        # Symmetric memory path (multicast / multimem instructions).
        symm_t.copy_(static_symm_src)
        if use_multimem:
            torch.ops.symm_mem.multimem_all_reduce_(symm_t, "sum", group_name)
        else:
            torch.ops.symm_mem.one_shot_all_reduce(symm_t, "sum", group_name)

    # Warmup on a side stream.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        static_in.fill_(float(rank + 1))
        static_symm_src.fill_(float(rank + 1))
        for _ in range(3):
            payload()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        payload()
    torch.cuda.synchronize()
    log(rank, "graph captured")
    return g, static_in, static_symm_src


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

def verify(rank, world, g, static_in, static_symm_src, symm_t, nccl_buf, it):
    """Fill inputs with integer-valued floats so equality is exact.

    Each rank contributes (rank+1+it); sum over world is deterministic.
    """
    val = float(rank + 1 + (it % 7))
    static_in.fill_(val)
    static_symm_src.fill_(val)

    expected = float(sum(r + 1 + (it % 7) for r in range(world)))

    g.replay()
    torch.cuda.synchronize()

    errs = []
    if not torch.all(nccl_buf == expected):
        errs.append(
            f"nccl: expected {expected}, got min={nccl_buf.min().item()} "
            f"max={nccl_buf.max().item()}"
        )
    if not torch.all(symm_t == expected):
        errs.append(
            f"symm: expected {expected}, got min={symm_t.min().item()} "
            f"max={symm_t.max().item()}"
        )
    return errs


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=600,
                    help="how long to loop; checkpoint/restore during this")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="seconds between verification iterations")
    ap.add_argument("--numel", type=int, default=32 * 1024 * 1024,
                    help="elements per buffer (128MB fp32; large enough for NVLS)")
    ap.add_argument("--no-multimem", action="store_true",
                    help="use one_shot instead of multimem_all_reduce_ "
                         "(bisect: isolates multicast from plain IPC)")
    args = ap.parse_args()

    rank, world, local = init_dist()
    dev = torch.device(f"cuda:{local}")
    require_multicast(rank, dev)

    group_name = dist.group.WORLD.group_name
    if hasattr(symm_mem, "enable_symm_mem_for_group"):
        symm_mem.enable_symm_mem_for_group(group_name)

    nccl_buf = torch.zeros(args.numel, dtype=torch.float32, device=dev)
    symm_t, hdl = alloc_symm(args.numel, torch.float32, dev, group_name)

    mc = getattr(hdl, "multicast_ptr", 0)
    if not args.no_multimem and not mc:
        raise RuntimeError(
            "symmetric memory rendezvous returned no multicast_ptr. "
            "Multicast is required for this harness; check NVSwitch/fabric "
            "manager, or run with --no-multimem to bisect."
        )
    log(rank, f"multicast_ptr = 0x{int(mc):x}")

    g, static_in, static_symm_src = build_graph(
        rank, world, symm_t, group_name, nccl_buf,
        use_multimem=not args.no_multimem,
    )

    inv_before = va_inventory(rank, symm_t, hdl, nccl_buf)
    log(rank, "VA inventory BEFORE:\n" + fmt_inventory(inv_before))

    errs = verify(rank, world, g, static_in, static_symm_src,
                  symm_t, nccl_buf, 0)
    if errs:
        log(rank, "FAIL before checkpoint: " + "; ".join(errs))
        return 1
    log(rank, "baseline verification PASS -- safe to checkpoint now")

    # ------------------------------------------------------------------
    # loop: checkpoint/restore the sandbox at any point during this
    # ------------------------------------------------------------------
    t_mono0, t_wall0 = time.monotonic(), time.time()
    failures = 0
    va_changed = False
    restored = False
    it = 0

    while time.monotonic() - t_mono0 < args.seconds:
        it += 1
        d_mono = time.monotonic() - t_mono0
        d_wall = time.time() - t_wall0
        # A wall-clock jump with no matching monotonic advance means the
        # process was frozen -- i.e. we just came back from a restore.
        if not restored and (d_wall - d_mono) > 5.0:
            restored = True
            log(rank, f"*** RESTORE DETECTED (froze ~{d_wall - d_mono:.1f}s) ***")
            inv_after = va_inventory(rank, symm_t, hdl, nccl_buf)
            log(rank, "VA inventory AFTER:\n" + fmt_inventory(inv_after))
            for k, v in sorted(inv_before.items()):
                if inv_after.get(k) != v:
                    va_changed = True
                    log(rank, f"VA CHANGED {k}: "
                              f"0x{v:016x} -> 0x{inv_after.get(k, 0):016x}")
            if not va_changed:
                log(rank, "VA inventory identical across restore")

        errs = verify(rank, world, g, static_in, static_symm_src,
                      symm_t, nccl_buf, it)
        tag = "post-restore" if restored else "pre-checkpoint"
        if errs:
            failures += 1
            log(rank, f"iter {it} ({tag}) FAIL: " + "; ".join(errs))
        else:
            log(rank, f"iter {it} ({tag}) pass")
        time.sleep(args.interval)

    ok = (failures == 0) and not va_changed
    log(rank, f"DONE iters={it} failures={failures} "
              f"va_changed={va_changed} restore_seen={restored} "
              f"result={'PASS' if ok else 'FAIL'}")
    if not restored:
        log(rank, "NOTE: no restore was detected -- this run did not test "
                  "checkpoint/restore, only steady-state correctness.")

    dist.barrier()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
