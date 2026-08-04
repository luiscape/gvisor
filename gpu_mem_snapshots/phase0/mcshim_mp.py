#!/usr/bin/env python3
"""Multi-process (one process per GPU) transparent multicast workload for the
mcshim interposer -- the vLLM/SGLang tensor-parallel topology.

A launcher forks one rank process per GPU. Rank 0 creates the multicast group
and exports it (cuMemExportToShareableHandle); peers import it over the app's
own socketpair channel (standing in for NCCL bootstrap / torch rendezvous).
Every rank adds its device, binds local shareable vidmem, maps a unicast VA +
the multicast VA, writes a per-rank pattern, and loops verifying.

NO suspend/resume logic anywhere: the LD_PRELOAD shim tracks the
export/import graph (fd st_dev:st_ino identity) and on resume the creator
rank's shim re-exports and serves the new fd on a unix socket, peer shims
re-import, and all ranks re-add + re-bind (cuMulticastBindMem blocking until
all devices join is the cross-rank barrier) and re-map at IDENTICAL VAs.

Markers under --dir (existence-based, shared by all ranks):
  pause     while present, all ranks quiesce (idle engine); removal unpauses
  (the shim owns suspend / suspended.<pid> / resumed.<pid>)

Per-rank outputs: wl.status.rank<r> (status), wl.pid.rank<r> (host pid).
"""

import argparse
import ctypes
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cuda as cu

PATTERN_BASE = 0xAB000000


def rank_main(rank, world, sock, args):
    d = args.dir

    def status(line):
        tmp = os.path.join(d, f"wl.status.rank{rank}.tmp")
        with open(tmp, "w") as f:
            f.write(line + "\n")
        os.replace(tmp, os.path.join(d, f"wl.status.rank{rank}"))
        print(f"[rank {rank}] {line}", flush=True)

    with open(os.path.join(d, f"wl.pid.rank{rank}"), "w") as f:
        f.write(str(os.getpid()))

    cu.call("cuInit", 0)
    dev = ctypes.c_int()
    cu.call("cuDeviceGet", ctypes.byref(dev), rank)
    if cu.device_attr(cu.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev.value) != 1:
        status("FATAL no multicast support")
        return 1
    ctx = ctypes.c_void_p()
    cu.call("cuDevicePrimaryCtxRetain", ctypes.byref(ctx), dev.value)
    cu.call("cuCtxSetCurrent", ctx)

    prop = cu.CUmulticastObjectProp()
    prop.numDevices = world
    prop.handleTypes = cu.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    gran = ctypes.c_size_t()
    cu.call("cuMulticastGetGranularity", ctypes.byref(gran), ctypes.byref(prop),
            cu.CU_MULTICAST_GRANULARITY_RECOMMENDED)
    size = (max(args.size, gran.value) + gran.value - 1) // gran.value * gran.value
    prop.size = size

    # Group setup: rank 0 creates + exports; peers import. The fd travels over
    # the app's own channel (socketpair via launcher), like NCCL's bootstrap.
    mc = ctypes.c_ulonglong()
    if rank == 0:
        cu.call("cuMulticastCreate", ctypes.byref(mc), ctypes.byref(prop))
        fd = cu.export_posix_fd(mc.value)
        cu.send_msg(sock, "mcfd", fds=[fd])  # launcher fans out to peers
        os.close(fd)
    else:
        _, fds = cu.recv_msg(sock, expect="mcfd")
        mc.value = cu.import_posix_fd(fds[0])
        os.close(fds[0])

    with cu.watchdog(f"rank{rank} AddDevice", 60):
        cu.call("cuMulticastAddDevice", mc.value, dev.value)

    # Local shareable vidmem bound into the group. BindMem blocks until every
    # device has joined -- the setup barrier.
    p = cu.alloc_prop(dev.value)
    memh = cu.mem_create(size, p)
    with cu.watchdog(f"rank{rank} BindMem", 120):
        cu.call("cuMulticastBindMem", mc.value, 0, memh, 0, size, 0)

    uni_va = cu.reserve_map_rw(memh, size, dev.value)
    pattern = PATTERN_BASE + rank
    cu.memset_u32(uni_va, pattern, size // 4)
    mc_va = cu.reserve_map_rw(mc.value, size, dev.value)
    cu.call("cuCtxSynchronize")

    status(f"READY pid={os.getpid()} mc=0x{mc.value:x} mc_va=0x{mc_va:x} "
           f"uni_va=0x{uni_va:x} size=0x{size:x}")

    def verify():
        """Read the pattern back at the ORIGINAL VA: validates content
        survival and unicast VA stability. MC-VA identity is enforced by the
        shim (resume fails unless the re-map lands at the same address)."""
        errs = []
        head = cu.read_u32(uni_va, 64)
        tail = cu.read_u32(uni_va + size - 256, 64)
        if any(g != pattern for g in head + tail):
            errs.append(f"unicast pattern lost (got {head[0]:#x})")
        return errs

    errs = verify()
    if errs:
        status("FAIL baseline: " + "; ".join(errs))
        return 1
    status("baseline mc-live pass")

    it, failures = 0, 0
    restored = False
    paused = False
    pause_file = os.path.join(d, "pause")
    while True:
        it += 1
        want_pause = os.path.exists(pause_file)
        if want_pause and not paused:
            paused = True
            status(f"PAUSED iter={it}")
        elif not want_pause and paused:
            paused = False
            restored = True
            status(f"UNPAUSED iter={it}")
        if paused:
            time.sleep(args.interval)
            continue
        errs = verify()
        tag = "post-restore" if restored else "pre-checkpoint"
        if errs:
            failures += 1
            status(f"iter={it} {tag}+mc-live FAIL: {'; '.join(errs)} "
                   f"failures={failures}")
        else:
            status(f"iter={it} {tag}+mc-live pass failures={failures}")
        time.sleep(args.interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--size", type=lambda s: int(s, 0), default=32 << 20)
    ap.add_argument("--dir", default="/tmp/mcshim")
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    # Socketpairs: rank0 <-> each peer, for the app-level fd handoff.
    # The launcher relays rank0's export fd to every peer.
    pairs = [socket.socketpair() for _ in range(args.world)]
    kids = []
    for r in range(args.world):
        pid = os.fork()
        if pid == 0:
            # Child: keep only my end of my pair.
            mine = pairs[r][1]
            for i, (a, b) in enumerate(pairs):
                a.close()
                if i != r:
                    b.close()
            os._exit(rank_main(r, args.world, mine, args) or 0)
        kids.append(pid)
    for _, b in pairs:
        b.close()

    # Relay: receive the fd once from rank 0, fan out to peers.
    if args.world > 1:
        msg, fds, _, _ = socket.recv_fds(pairs[0][0], 4096, 1)
        assert msg.startswith(b"mcfd"), msg
        for r in range(1, args.world):
            socket.send_fds(pairs[r][0], [b"mcfd\n"], fds)
        os.close(fds[0])

    rc = 0
    try:
        for pid in kids:
            _, st = os.waitpid(pid, 0)
            rc |= os.waitstatus_to_exitcode(st)
    except KeyboardInterrupt:
        pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
