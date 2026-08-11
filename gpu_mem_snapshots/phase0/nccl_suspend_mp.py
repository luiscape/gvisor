#!/usr/bin/env python3
"""Multi-process NCCL suspend/resume checkpoint workload (one process per GPU,
like real vLLM/SGLang tensor-parallel ranks).

Rank 0 generates an ncclUniqueId and writes it to --dir/uniqueid; all ranks
ncclCommInitRank into one communicator, run a verified allreduce (eager + a
captured CUDA graph), and on file markers call ncclCommSuspend / ncclComm
Resume. This exercises the same NVLS multicast suspend/resume as the
single-process clique, but with the per-process topology the checkpoint job
actually sees in production (cuda-checkpoint --launch-job wraps a launcher
that forks one child per rank).

Coordination is via the shared --dir (a tmpfs bind), so no torchrun / network
is needed:
  <dir>/uniqueid            rank 0 -> peers (NCCL bootstrap id)
  <dir>/ready.<rank>        each rank -> launcher
  <dir>/status.<rank>       per-rank status line
  <dir>/suspend, resume, restored   launcher -> all ranks (broadcast markers)

Run one process per rank:
  RANK=0 WORLD=4 python3 nccl_suspend_mp.py --dir /tmp/mp [--graph]
  ...
  RANK=3 WORLD=4 python3 nccl_suspend_mp.py --dir /tmp/mp [--graph]
(run_nccl_suspend_mp.sh / the gVisor launcher spawns these).
"""

import argparse
import ctypes
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cuda as cu
import _nccl as nc


def f32bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _ctl_pages():
    """Count r--s /dev/nvidiactl control pages currently mapped."""
    try:
        return sum(1 for ln in open("/proc/self/maps")
                   if "r--s" in ln and "nvidiactl" in ln)
    except OSError:
        return -1


def _dump_maps(d, rank, tag):
    """Snapshot GPU/UVM address-space mappings (for diffing across the shim's
    suspend/resume: a range present pre- but missing/changed post- is a
    mapping the shim failed to replay -> the 719 culprit)."""
    try:
        lines = []
        with open("/proc/self/maps") as f:
            for ln in f:
                if any(k in ln for k in ("nvidia", "uvm", "/dev/nvidia")):
                    lines.append(ln.rstrip())
        with open(os.path.join(d, f"maps.{tag}.{rank}"), "w") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/mp")
    ap.add_argument("--count", type=int, default=32 * 1024 * 1024)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--pause-only", action="store_true",
                    help="never call ncclCommSuspend/Resume; instead quiesce "
                         "while a `pause` marker exists (existence-based). "
                         "For validating the mcshim LD_PRELOAD interposer "
                         "against STOCK NCCL: the shim owns the multicast "
                         "suspend, NCCL is never patched or called.")
    ap.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    ap.add_argument("--world", type=int, default=int(os.environ.get("WORLD", "1")))
    args = ap.parse_args()
    rank, world = args.rank, args.world
    count = args.count
    nbytes = count * 4
    os.makedirs(args.dir, exist_ok=True)

    def status(line):
        line = f"[rank{rank}] {line}"
        tmp = os.path.join(args.dir, f"status.{rank}.tmp")
        with open(tmp, "w") as f:
            f.write(line + "\n")
        os.replace(tmp, os.path.join(args.dir, f"status.{rank}"))
        print(line, flush=True)

    def wait_file(name, timeout=120):
        p = os.path.join(args.dir, name)
        for _ in range(int(timeout / 0.1)):
            if os.path.exists(p):
                return True
            time.sleep(0.1)
        return False

    # Each rank drives exactly one GPU (its own primary context).
    cu.call("cuInit", 0)
    dev = ctypes.c_int()
    cu.call("cuDeviceGet", ctypes.byref(dev), rank)
    ctx = ctypes.c_void_p()
    cu.call("cuDevicePrimaryCtxRetain", ctypes.byref(ctx), dev.value)
    cu.call("cuCtxSetCurrent", ctx)
    stream = ctypes.c_void_p()
    cu.call("cuStreamCreate", ctypes.byref(stream), 0)
    sendp = ctypes.c_ulonglong()
    recvp = ctypes.c_ulonglong()
    cu.call("cuMemAlloc_v2", ctypes.byref(sendp), nbytes)
    cu.call("cuMemAlloc_v2", ctypes.byref(recvp), nbytes)
    sendb, recvb = sendp.value, recvp.value

    # Bootstrap id exchange over the shared dir.
    uid = nc.ncclUniqueId()
    uidpath = os.path.join(args.dir, "uniqueid")
    if rank == 0:
        nc.call("ncclGetUniqueId", ctypes.byref(uid))
        # ncclUniqueId is opaque binary with embedded NULs; copy raw bytes
        # (do NOT use bytes(uid.internal) -- c_char arrays truncate at NUL).
        raw = ctypes.string_at(ctypes.byref(uid), nc.NCCL_UNIQUE_ID_BYTES)
        tmp = uidpath + ".tmp"
        with open(tmp, "wb") as f:
            f.write(raw)
        os.replace(tmp, uidpath)
    else:
        if not wait_file("uniqueid"):
            status("FATAL no uniqueid from rank 0")
            return 1
        time.sleep(0.2)
        with open(uidpath, "rb") as f:
            raw = f.read(nc.NCCL_UNIQUE_ID_BYTES)
        ctypes.memmove(ctypes.byref(uid), raw, len(raw))

    _dump_maps(args.dir, rank, "phase0_preinit")  # ctx + stream + cuMemAlloc
    comm = nc.ncclComm_t()
    print(f"[rank{rank}] NCCL {nc.version()} InitRank world={world} dev={rank}",
          flush=True)
    nc.call("ncclCommInitRank", ctypes.byref(comm), world, uid, rank)
    _dump_maps(args.dir, rank, "phase1_comminit")  # after ncclCommInitRank

    # Each rank contributes a FIXED value (rank+1). Independent per-process
    # loops drift (ranks are not lock-step without a barrier), so the expected
    # sum must not depend on a per-rank iteration counter. recv is pre-filled
    # with a sentinel so a stale/no-op collective is caught.
    RANK_CONTRIB = float(rank + 1)
    EXPECTED = float(sum(r + 1 for r in range(world)))
    SENTINEL = 0x7E577E57

    def fill_send(_base=None):
        cu.call("cuMemsetD32_v2", sendb, f32bits(RANK_CONTRIB), count)
        cu.call("cuMemsetD32_v2", recvb, SENTINEL, count)
        cu.call("cuCtxSynchronize")

    def allreduce_eager():
        nc.call("ncclAllReduce", sendb, recvb, count, nc.ncclFloat32,
                nc.ncclSum, comm, stream)
        cu.call("cuStreamSynchronize", stream)

    def check(expected, what):
        head = cu.read_u32(recvb, 8)
        tail = cu.read_u32(recvb + nbytes - 32, 8)
        want = f32bits(expected)
        if any(g != want for g in head + tail):
            got = struct.unpack("<f", struct.pack("<I", head[0]))[0]
            return [f"{what}: want {expected} got {got}"]
        return []

    # Warmup (NVLS lazy init).
    fill_send()
    allreduce_eager()
    errs = check(EXPECTED, "warmup")
    if errs:
        status("FATAL warmup: " + "; ".join(errs))
        return 1
    _dump_maps(args.dir, rank, "phase2_warmup")  # after first allreduce (lazy conns)
    suspendable, susp, persist = nc.mem_stats(comm)
    status(f"WARMUP-OK expected={EXPECTED} suspendable={suspendable} "
           f"suspended={susp} persist={persist}")

    # Optional CUDA graph of the allreduce.
    gexec = None
    if args.graph:
        cu.call("cuStreamBeginCapture_v2", stream,
                cu.CU_STREAM_CAPTURE_MODE_RELAXED)
        nc.call("ncclAllReduce", sendb, recvb, count, nc.ncclFloat32,
                nc.ncclSum, comm, stream)
        g = ctypes.c_void_p()
        cu.call("cuStreamEndCapture", stream, ctypes.byref(g))
        ge = ctypes.c_void_p()
        cu.call("cuGraphInstantiateWithFlags", ctypes.byref(ge), g, 0)
        cu.call("cuGraphDestroy", g)
        gexec = ge
        status("GRAPH-CAPTURED")

    # Signal ready to the launcher.
    open(os.path.join(args.dir, f"ready.{rank}"), "w").close()
    status(f"READY pid={os.getpid()} graph={bool(gexec)}")

    suspended = False
    restored = False
    paused = False
    it, failures = 0, 0
    while True:
        it += 1
        if not restored and os.path.exists(os.path.join(args.dir, "restored")):
            restored = True
            status(f"RESTORE-DETECTED iter={it}")

        if args.pause_only:
            want = os.path.exists(os.path.join(args.dir, "pause"))
            if want and not paused:
                paused = True
                _dump_maps(args.dir, rank, "pre")
                status(f"PAUSED iter={it} (quiesced; shim may suspend)")
            elif not want and paused:
                paused = False
                restored = True
                _dump_maps(args.dir, rank, "post")
                status(f"UNPAUSED iter={it}")
            if paused:
                time.sleep(args.interval)
                continue
        elif not suspended and os.path.exists(os.path.join(args.dir, "suspend")):
            try:
                _dump_maps(args.dir, rank, "presuspend")
                t0 = time.monotonic()
                nc.call("ncclCommSuspend", comm, nc.NCCL_SUSPEND_MEM)
                dt = time.monotonic() - t0
                suspended = True
                _dump_maps(args.dir, rank, "suspended")
                status(f"SUSPENDED iter={it} ({dt:.2f}s) memstats={nc.mem_stats(comm)}")
            except nc.NcclError as e:
                failures += 1
                status(f"iter={it} SUSPEND FAIL: {e} failures={failures}")
        if not args.pause_only and suspended and \
                os.path.exists(os.path.join(args.dir, "resume")):
            try:
                t0 = time.monotonic()
                nc.call("ncclCommResume", comm)
                dt = time.monotonic() - t0
                suspended = False
                _dump_maps(args.dir, rank, "postresume")
                status(f"RESUMED iter={it} ({dt:.2f}s) memstats={nc.mem_stats(comm)}")
            except nc.NcclError as e:
                failures += 1
                status(f"iter={it} RESUME FAIL: {e} failures={failures}")

        if suspended:
            status(f"iter={it} suspended (idle) failures={failures}")
            time.sleep(args.interval)
            continue

        errs = []
        try:
            fill_send()
            allreduce_eager()
            errs += check(EXPECTED, "eager")
            if gexec is not None:
                fill_send()
                cu.call("cuGraphLaunch", gexec, stream)
                cu.call("cuStreamSynchronize", stream)
                errs += check(EXPECTED, "graph")
        except (nc.NcclError, cu.CudaError) as e:
            errs.append(f"collective failed: {e}")

        tag = "post-restore" if restored else "pre-checkpoint"
        if errs:
            failures += 1
            status(f"iter={it} {tag} FAIL: {'; '.join(errs)} failures={failures}")
        else:
            status(f"iter={it} {tag} pass failures={failures} ctlpages={_ctl_pages()}")
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
