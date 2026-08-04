#!/usr/bin/env python3
"""E2E workload for the NCCL suspend/resume checkpoint model (upstream API).

Single process, N GPUs (default 4 => NVLS engages on NVSwitch):

  * ncclCommInitAll clique, NCCL_NVLS_ENABLE=1
  * per-iteration verified allreduce (eager), optionally also replayed from a
    captured CUDA GRAPH (--graph) to exercise restore-after-graph-capture
  * on `suspend` marker: ncclCommSuspend(comm, NCCL_SUSPEND_MEM) on every comm
    -- NCCL releases its dynamic GPU allocations INCLUDING the NVLS multicast
    objects, through libcuda, so libcuda's bookkeeping stays consistent for
    cuda-checkpoint
  * on `resume` marker: ncclCommResume on every comm; NCCL recreates the NVLS
    buffers (VA-stable, so captured graphs stay valid); verification continues

Status protocol (--dir/status): READY / iter=N <tag> pass|FAIL, plus
SUSPENDED/RESUMED markers with ncclCommMemStats numbers.

Env: NCCL_LIB (libnccl.so.2 path), NCCL_NVLS_ENABLE=1 recommended.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngpus", type=int, default=4)
    ap.add_argument("--count", type=int, default=32 * 1024 * 1024,
                    help="floats per buffer (128MB @ 32M: large enough for NVLS)")
    ap.add_argument("--dir", default="/tmp")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--graph", action="store_true",
                    help="also capture the allreduce into CUDA graphs and "
                         "verify graph replay across suspend/resume")
    args = ap.parse_args()
    n = args.ngpus
    count = args.count
    nbytes = count * 4

    def status(line):
        tmp = os.path.join(args.dir, "status.tmp")
        with open(tmp, "w") as f:
            f.write(line + "\n")
        os.replace(tmp, os.path.join(args.dir, "status"))
        print(line, flush=True)

    print(f"NCCL version {nc.version()}", flush=True)
    if not nc.has("ncclCommSuspend"):
        status("FATAL libnccl has no ncclCommSuspend")
        return 1

    # --- CUDA setup: primary ctx + stream + buffers per device -------------
    cu.call("cuInit", 0)
    ctxs, streams, sendb, recvb = [], [], [], []
    for i in range(n):
        d = ctypes.c_int()
        cu.call("cuDeviceGet", ctypes.byref(d), i)
        c = ctypes.c_void_p()
        cu.call("cuDevicePrimaryCtxRetain", ctypes.byref(c), d.value)
        ctxs.append(c)
        cu.call("cuCtxSetCurrent", c)
        s = ctypes.c_void_p()
        cu.call("cuStreamCreate", ctypes.byref(s), 0)
        streams.append(s)
        for lst in (sendb, recvb):
            p = ctypes.c_ulonglong()
            cu.call("cuMemAlloc_v2", ctypes.byref(p), nbytes)
            lst.append(p.value)

    # --- NCCL clique --------------------------------------------------------
    comms = (nc.ncclComm_t * n)()
    devlist = (ctypes.c_int * n)(*range(n))
    nc.call("ncclCommInitAll", comms, n, devlist)
    status(f"NCCL-INIT version={nc.version()} ngpus={n}")

    def fill_inputs(val_of_rank):
        for i in range(n):
            cu.call("cuCtxSetCurrent", ctxs[i])
            cu.call("cuMemsetD32_v2", sendb[i], f32bits(val_of_rank(i)), count)
        for i in range(n):
            cu.call("cuCtxSetCurrent", ctxs[i])
            cu.call("cuCtxSynchronize")

    def allreduce_eager():
        nc.call("ncclGroupStart")
        for i in range(n):
            cu.call("cuCtxSetCurrent", ctxs[i])
            nc.call("ncclAllReduce", sendb[i], recvb[i], count,
                    nc.ncclFloat32, nc.ncclSum, comms[i], streams[i])
        nc.call("ncclGroupEnd")
        for i in range(n):
            cu.call("cuCtxSetCurrent", ctxs[i])
            cu.call("cuStreamSynchronize", streams[i])

    def check_result(expected, what):
        errs = []
        for i in range(n):
            cu.call("cuCtxSetCurrent", ctxs[i])
            head = cu.read_u32(recvb[i], 8)
            tail = cu.read_u32(recvb[i] + nbytes - 32, 8)
            want = f32bits(expected)
            if any(g != want for g in head + tail):
                got = struct.unpack("<f", struct.pack("<I", head[0]))[0]
                errs.append(f"gpu{i} {what}: want {expected} got {got}")
        return errs

    # --- warmup (NVLS lazy init happens here) -------------------------------
    fill_inputs(lambda r: r + 1)
    expected0 = float(sum(r + 1 for r in range(n)))
    allreduce_eager()
    errs = check_result(expected0, "warmup")
    if errs:
        status("FATAL warmup failed: " + "; ".join(errs))
        return 1

    suspendable, suspended_flag, persist = nc.mem_stats(comms[0])
    status(f"WARMUP-OK expected={expected0} "
           f"memstats: suspendable={suspendable} suspended={suspended_flag} "
           f"persistent={persist}")
    if suspendable == 0:
        print("WARNING: comm reports 0 suspendable bytes -- NVLS may not "
              "have engaged; blockers/gate will confirm", flush=True)

    # --- optional CUDA graph capture ----------------------------------------
    gexecs = []
    if args.graph:
        # Capture the grouped allreduce into one graph per device stream.
        for i in range(n):
            cu.call("cuCtxSetCurrent", ctxs[i])
            cu.call("cuStreamBeginCapture_v2", streams[i],
                    cu.CU_STREAM_CAPTURE_MODE_RELAXED)
        nc.call("ncclGroupStart")
        for i in range(n):
            cu.call("cuCtxSetCurrent", ctxs[i])
            nc.call("ncclAllReduce", sendb[i], recvb[i], count,
                    nc.ncclFloat32, nc.ncclSum, comms[i], streams[i])
        nc.call("ncclGroupEnd")
        for i in range(n):
            cu.call("cuCtxSetCurrent", ctxs[i])
            g = ctypes.c_void_p()
            cu.call("cuStreamEndCapture", streams[i], ctypes.byref(g))
            ge = ctypes.c_void_p()
            cu.call("cuGraphInstantiateWithFlags", ctypes.byref(ge), g, 0)
            cu.call("cuGraphDestroy", g)
            gexecs.append(ge)
        status("GRAPH-CAPTURED")

    def allreduce_graph():
        for i in range(n):
            cu.call("cuCtxSetCurrent", ctxs[i])
            cu.call("cuGraphLaunch", gexecs[i], streams[i])
        for i in range(n):
            cu.call("cuCtxSetCurrent", ctxs[i])
            cu.call("cuStreamSynchronize", streams[i])

    status(f"READY pid={os.getpid()} ngpus={n} graph={bool(gexecs)} "
           f"suspendable={suspendable}")

    # --- main loop -----------------------------------------------------------
    suspended = False
    restored = False
    it, failures = 0, 0
    while True:
        it += 1
        if not restored and os.path.exists(os.path.join(args.dir, "restored")):
            restored = True
            status(f"RESTORE-DETECTED iter={it}")
        if not suspended and os.path.exists(os.path.join(args.dir, "suspend")):
            os.remove(os.path.join(args.dir, "suspend"))
            try:
                t0 = time.monotonic()
                # Single-process multi-comm clique: per-comm calls that
                # synchronize across ranks must be grouped, or comm[0]'s
                # suspend blocks waiting for its peers (measured).
                nc.call("ncclGroupStart")
                for i in range(n):
                    nc.call("ncclCommSuspend", comms[i], nc.NCCL_SUSPEND_MEM)
                nc.call("ncclGroupEnd")
                dt = time.monotonic() - t0
                st = nc.mem_stats(comms[0])
                suspended = True
                status(f"SUSPENDED iter={it} ({dt:.2f}s) memstats={st}")
            except nc.NcclError as e:
                failures += 1
                status(f"iter={it} SUSPEND FAIL: {e} failures={failures}")
        if suspended and os.path.exists(os.path.join(args.dir, "resume")):
            os.remove(os.path.join(args.dir, "resume"))
            try:
                t0 = time.monotonic()
                nc.call("ncclGroupStart")
                for i in range(n):
                    nc.call("ncclCommResume", comms[i])
                nc.call("ncclGroupEnd")
                dt = time.monotonic() - t0
                st = nc.mem_stats(comms[0])
                suspended = False
                status(f"RESUMED iter={it} ({dt:.2f}s) memstats={st}")
            except nc.NcclError as e:
                failures += 1
                status(f"iter={it} RESUME FAIL: {e} failures={failures}")

        if suspended:
            status(f"iter={it} suspended (idle) failures={failures}")
            time.sleep(args.interval)
            continue

        # Verified collective(s): eager, and graph replay if captured.
        val = it % 5
        fill_inputs(lambda r: r + 1 + val)
        expected = float(sum(r + 1 + val for r in range(n)))
        errs = []
        try:
            allreduce_eager()
            errs += check_result(expected, "eager")
            if gexecs:
                fill_inputs(lambda r: r + 2 + val)
                gexpected = float(sum(r + 2 + val for r in range(n)))
                allreduce_graph()
                errs += check_result(gexpected, "graph")
        except (nc.NcclError, cu.CudaError) as e:
            errs.append(f"collective failed: {e}")

        tag = "post-restore" if restored else "pre-checkpoint"
        if errs:
            failures += 1
            status(f"iter={it} {tag} FAIL: {'; '.join(errs)} failures={failures}")
        else:
            status(f"iter={it} {tag} pass failures={failures}")
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
