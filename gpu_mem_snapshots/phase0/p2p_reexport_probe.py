#!/usr/bin/env python3
"""Does peer (NVLink) access to RE-IMPORTED VMM allocations survive
cuda-checkpoint / runsc checkpoint? The decisive test for the mcshim UC-import
"719" fault, with NO NCCL and NO shim -- just two processes, one GPU each, a
real SM kernel that reads peer buffers over NVLink, in the bidirectional,
many-buffer topology NCCL actually uses.

Each rank (GPU r):
  - creates N local buffers (cuMemCreate POSIX_FD) + maps + writes a distinct
    pattern per (rank,buffer);
  - exports all N to the peer and imports the peer's N (fd exchange over a
    socketpair, which survives checkpoint: only GPU state is toggled);
  - runs a PTX kernel that loads *peer_va -> local, for every peer buffer, and
    verifies the pattern (baseline peer read works).

Then, mirroring the mcshim flow across the checkpoint:
  - RELEASE: unmap each import (keep the VA reservation) + cuMemRelease; the
    exporter keeps its own buffers resident.
  - parent (native) or host (gVisor) drives lock/checkpoint/restore/unlock.
  - RESUME: re-export all local buffers (new fds over the same socketpair),
    re-import the peer's, re-map at IDENTICAL VAs + cuMemSetAccess, and re-run
    the peer-read kernel for every buffer.

VERDICT (post == OK ?):
  OK   -> peer access survives; the mcshim 719 is a shim/NCCL-specific issue.
  FAIL -> cuda-checkpoint/gVisor does not re-establish peer access to a
          re-imported allocation at this topology; exporter must recreate
          (not keep resident). Bare repro for NVIDIA.

Usage: sudo [P2P_NBUF=48] python3 p2p_reexport_probe.py [--gpus 0,1]
       (gVisor: run_p2p_reexport_gvisor.sh)
"""

import argparse
import ctypes
import os
import socket
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cuda as cu

DIR = "/tmp/p2p_reexport_probe"
PATTERN = 0xC0FFEE00
N_BUF = int(os.environ.get("P2P_NBUF", "48"))
# Replicate the mcshim exactly: do suspend (unmap+release) and resume
# (re-import+re-map) on a BACKGROUND thread, while the peer-read kernels run on
# the main thread -- the one structural difference left between this probe and
# the shim.
THREADED = os.environ.get("P2P_THREADED", "0") == "1"

# PTX: peek(src,dst){ u32 v=*src; *src=v; *dst=v; } -- a real SM global LOAD
# AND STORE to the peer VA, since NCCL collectives both read and write peer
# buffers; a broken peer mapping (read or write) faults like an NCCL kernel.
PTX = b"""
.version 7.8
.target sm_90
.address_size 64
.visible .entry peek(.param .u64 p_src, .param .u64 p_dst)
{
    .reg .b64 %rd<3>;
    .reg .b32 %r<2>;
    ld.param.u64 %rd1, [p_src];
    ld.param.u64 %rd2, [p_dst];
    ld.global.u32 %r1, [%rd1];
    st.global.u32 [%rd1], %r1;
    st.global.u32 [%rd2], %r1;
    ret;
}
"""


def mark(name):
    return os.path.join(DIR, name)


def wait_mark(name, timeout=120):
    for _ in range(int(timeout / 0.1)):
        if os.path.exists(mark(name)):
            return True
        time.sleep(0.1)
    return False


def write_mark(name, body=""):
    with open(mark(name), "w") as f:
        f.write(body + "\n")


def bufval(rank, i):
    return (PATTERN + rank * 100003 + i) & 0xFFFFFFFF


def dump_maps(rank, tag):
    try:
        lines = [ln.rstrip() for ln in open("/proc/self/maps")
                 if "nvidia" in ln or "uvm" in ln]
        with open(mark(f"maps.{tag}.{rank}"), "w") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass


def peek_fn():
    mod = ctypes.c_void_p()
    cu.call("cuModuleLoadData", ctypes.byref(mod), PTX)
    fn = ctypes.c_void_p()
    cu.call("cuModuleGetFunction", ctypes.byref(fn), mod, b"peek")
    return fn


def peer_read(fn, peer_va, dst_va):
    src = ctypes.c_ulonglong(peer_va)
    dst = ctypes.c_ulonglong(dst_va)
    params = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(src), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(dst), ctypes.c_void_p),
    )
    cu.call("cuLaunchKernel", fn, 1, 1, 1, 1, 1, 1, 0, None, params, None)
    cu.call("cuCtxSynchronize")
    return cu.read_u32(dst_va, 1)[0]


def set_access(va, size, dev):
    d = cu.CUmemAccessDesc()
    d.location.type = cu.CU_MEM_LOCATION_TYPE_DEVICE
    d.location.id = dev
    d.flags = cu.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    cu.call("cuMemSetAccess", va, size, ctypes.byref(d), 1)


def rank_main(rank, sock, ordinals):
    dev = cu.init_device(ordinals[rank])
    fn = peek_fn()
    dst = ctypes.c_ulonglong()
    cu.call("cuMemAlloc_v2", ctypes.byref(dst), 4)
    peer_rank = 1 - rank

    # Local buffers, exported.
    p = cu.alloc_prop(dev)
    size = max(2 << 20, cu.alloc_granularity(p))
    mine = []  # handles of my own buffers (kept resident across checkpoint)
    for i in range(N_BUF):
        h = cu.mem_create(size, p)
        va = cu.reserve_map_rw(h, size, dev)
        cu.memset_u32(va, bufval(rank, i), size // 4)
        mine.append(h)

    def send_mine():
        for i in range(N_BUF):
            fd = cu.export_posix_fd(mine[i])
            cu.send_msg(sock, f"fd {i}", fds=[fd])
            os.close(fd)

    peer_h = [None] * N_BUF  # imported handles (released at suspend)

    def recv_peer(va_hint):
        """Import peer's N buffers. va_hint[i] (or None) forces the mapping VA
        (used on resume to re-map at the identical address)."""
        vas = [None] * N_BUF
        for _ in range(N_BUF):
            msg, fds = cu.recv_msg(sock, expect="fd")
            i = int(msg.split()[1])
            h = cu.import_posix_fd(fds[0])
            os.close(fds[0])
            peer_h[i] = h
            if va_hint[i] is None:
                vas[i] = cu.reserve_map_rw(h, size, dev)
            else:
                cu.call("cuMemMap", va_hint[i], size, 0, h, 0)
                set_access(va_hint[i], size, dev)
                vas[i] = va_hint[i]
        return vas

    ctx = ctypes.c_void_p()
    cu.call("cuCtxGetCurrent", ctypes.byref(ctx))

    # Avoid a fd-exchange deadlock: rank 0 sends first, rank 1 receives first.
    peer_va = [None] * N_BUF
    if rank == 0:
        send_mine(); peer_va = recv_peer(peer_va)
    else:
        peer_va = recv_peer(peer_va); send_mine()

    def verify():
        bad = 0
        for i in range(N_BUF):
            got = peer_read(fn, peer_va[i], dst.value)
            if got != bufval(peer_rank, i):
                bad += 1
        return "OK" if bad == 0 else f"{bad}/{N_BUF}-bad"

    try:
        base = verify()
    except cu.CudaError as e:
        base = f"FAULT({e.name})"
    with open(mark(f"pid.{rank}"), "w") as f:
        f.write(str(os.getpid()))
    dump_maps(rank, "pre")
    write_mark(f"ready.{rank}", f"baseline={base}")

    def suspend():
        for i in range(N_BUF):
            cu.call("cuMemUnmap", peer_va[i], size)  # keep the reservation
            cu.call("cuMemRelease", peer_h[i])       # drop the live import

    def resume():
        # Re-establish, avoiding deadlock in the same send/recv order.
        if rank == 0:
            send_mine(); recv_peer(peer_va)
        else:
            recv_peer(peer_va); send_mine()

    if THREADED:
        # Exactly like the mcshim: a background thread performs the teardown
        # and rebuild (setting the context current itself); the main thread
        # only runs the peer-read kernels.
        def control():
            cu.call("cuCtxSetCurrent", ctx)
            wait_mark("release"); suspend(); write_mark(f"released.{rank}")
            wait_mark("resume"); resume(); write_mark(f"ctlresumed.{rank}")
        t = threading.Thread(target=control, daemon=True)
        t.start()
        wait_mark(f"ctlresumed.{rank}")
    else:
        wait_mark("release"); suspend(); write_mark(f"released.{rank}")
        wait_mark("resume"); resume()

    try:
        post = verify()
    except cu.CudaError as e:
        post = f"FAULT({e.name})"
    dump_maps(rank, "post")
    write_mark(f"resumed.{rank}", f"post={post}")
    wait_mark("stop")


def _run_rank(rank, sock, ordinals):
    try:
        rank_main(rank, sock, ordinals)
    except BaseException:
        import traceback
        with open(mark(f"err.{rank}"), "w") as f:
            f.write(traceback.format_exc())
        write_mark(f"ready.{rank}", "CRASHED")
        write_mark(f"released.{rank}")
        write_mark(f"resumed.{rank}", "post=CRASH")
        os._exit(1)
    os._exit(0)


def launcher(ordinals):
    os.makedirs(DIR, exist_ok=True)
    a, b = socket.socketpair()
    p0 = os.fork()
    if p0 == 0:
        b.close(); _run_rank(0, a, ordinals)
    p1 = os.fork()
    if p1 == 0:
        a.close(); _run_rank(1, b, ordinals)
    a.close(); b.close()
    os.waitpid(p0, 0); os.waitpid(p1, 0)


def cc(binary, pid, action, extra=()):
    p = subprocess.run([binary, "--action", action, "--pid", str(pid), *extra],
                       capture_output=True, text=True, timeout=90)
    print(f"[parent] {action} pid={pid}: rc={p.returncode} "
          f"{(p.stdout + p.stderr).strip()}", flush=True)
    return p.returncode


def parent(args):
    os.makedirs(DIR, exist_ok=True)
    for f in os.listdir(DIR):
        os.remove(os.path.join(DIR, f))
    job = subprocess.Popen([args.cuda_checkpoint, "--launch-job", sys.executable,
                            os.path.abspath(__file__), "--launcher",
                            "--gpus", args.gpus])
    if not (wait_mark("ready.0") and wait_mark("ready.1")):
        print("[parent] ranks never ready"); job.kill(); return 1
    print(f"[parent] rank0 {open(mark('ready.0')).read().strip()}", flush=True)
    print(f"[parent] rank1 {open(mark('ready.1')).read().strip()}", flush=True)
    pids = [int(open(mark("pid.0")).read()), int(open(mark("pid.1")).read())]

    write_mark("release")
    if not (wait_mark("released.0") and wait_mark("released.1")):
        print("[parent] release stalled"); job.kill(); return 1
    for pid in pids:
        cc(args.cuda_checkpoint, pid, "lock", ("--timeout", "30000"))
    for pid in pids:
        cc(args.cuda_checkpoint, pid, "checkpoint")
    for pid in pids:
        cc(args.cuda_checkpoint, pid, "restore")
    for pid in pids:
        cc(args.cuda_checkpoint, pid, "unlock")
    write_mark("resume")
    if not (wait_mark("resumed.0", 60) and wait_mark("resumed.1", 60)):
        print("[parent] resume stalled"); job.kill(); return 1
    post0 = open(mark("resumed.0")).read().strip()
    post1 = open(mark("resumed.1")).read().strip()
    write_mark("stop")
    try:
        job.wait(timeout=15)
    except subprocess.TimeoutExpired:
        job.kill()
    print(f"\n[parent] rank0 {post0}\n[parent] rank1 {post1}")
    ok = post0 == "post=OK" and post1 == "post=OK"
    print(f"[parent] VERDICT: bidirectional {N_BUF}-buffer peer access after "
          f"re-import {'SURVIVES' if ok else 'is BROKEN'}", flush=True)
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launcher", action="store_true")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--cuda-checkpoint", default="/usr/local/bin/cuda-checkpoint")
    args = ap.parse_args()
    ordinals = [int(g) for g in args.gpus.split(",")]
    if args.launcher:
        launcher(ordinals)
        return 0
    return parent(args)


if __name__ == "__main__":
    sys.exit(main())
