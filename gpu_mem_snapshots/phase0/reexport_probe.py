#!/usr/bin/env python3
"""Does a VMM POSIX-FD allocation survive cuda-checkpoint well enough to be
RE-EXPORTED and RE-USED afterward? Decides the mcshim UC-import strategy:

  If re-export after restore SUCCEEDS -> the exporter keeps its allocation
    resident (cuda-checkpoint's job) and only re-exports on resume.
  If it FAILS -> the exporter must release + recreate its allocation on
    suspend/resume (like the NCCL patch does for NVLS UC buffers), preserving
    content via CPU backup.

Also tests double-export WITHOUT any checkpoint, to separate "can't export
twice" from "checkpoint invalidated the handle".

child: cuMemCreate(POSIX_FD)+map+write pattern; export->fd0; close fd0; READY.
parent: cuda-checkpoint lock/checkpoint/restore/unlock on child; then STEP.
child on STEP: (1) read back pattern (content survived?),
               (2) cuMemExportToShareableHandle again (re-export works?).

Usage: sudo python3 reexport_probe.py [--cuda-checkpoint PATH]
"""

import argparse
import ctypes
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cuda as cu

READY = "/tmp/reexport_probe.ready"
STEP = "/tmp/reexport_probe.step"
DONE = "/tmp/reexport_probe.done"
SIZE = 2 << 20


def child():
    for f in (READY, STEP, DONE):
        if os.path.exists(f):
            os.remove(f)
    dev = cu.init_device(0)
    p = cu.alloc_prop(dev)
    size = max(SIZE, cu.alloc_granularity(p))
    h = cu.mem_create(size, p)
    va = cu.reserve_map_rw(h, size, dev)
    cu.memset_u32(va, 0x1234ABCD, size // 4)

    # Double-export WITHOUT checkpoint (baseline: is re-export ever allowed?).
    fd0 = cu.export_posix_fd(h)
    os.close(fd0)
    try:
        fd1 = cu.export_posix_fd(h)
        os.close(fd1)
        dbl = "OK"
    except cu.CudaError as e:
        dbl = e.name
    print(f"[child] double-export (no checkpoint): {dbl}", flush=True)

    with open(READY, "w") as f:
        f.write(f"{os.getpid()} va=0x{va:x}\n")

    while not os.path.exists(STEP):
        time.sleep(0.2)

    # After restore: content intact?
    try:
        got = cu.read_u32(va, 4)
        content = "intact" if all(g == 0x1234ABCD for g in got) else f"LOST({got[0]:#x})"
    except cu.CudaError as e:
        content = f"read-FAIL({e.name})"
    # After restore: can we re-export the same allocation?
    try:
        fd2 = cu.export_posix_fd(h)
        os.close(fd2)
        reexport = "OK"
    except cu.CudaError as e:
        reexport = e.name
    with open(DONE, "w") as f:
        f.write(f"content={content} reexport={reexport}\n")
    print(f"[child] post-restore content={content} reexport={reexport}",
          flush=True)
    time.sleep(2)


def cc(binary, pid, action, extra=()):
    p = subprocess.run([binary, "--action", action, "--pid", str(pid), *extra],
                       capture_output=True, text=True, timeout=90)
    print(f"[parent] {action}: rc={p.returncode} {(p.stdout + p.stderr).strip()}",
          flush=True)
    return p.returncode


def parent(args):
    for f in (READY, STEP, DONE):
        if os.path.exists(f):
            os.remove(f)
    job = subprocess.Popen([args.cuda_checkpoint, "--launch-job", sys.executable,
                            os.path.abspath(__file__), "--child"])
    for _ in range(120):
        if os.path.exists(READY):
            break
        if job.poll() is not None:
            print("[parent] child died early"); return 1
        time.sleep(1)
    else:
        print("[parent] child never ready"); job.kill(); return 1
    kids = subprocess.run(["pgrep", "-P", str(job.pid)], capture_output=True,
                          text=True).stdout.split()
    pid = int(kids[0]) if kids else job.pid
    print(f"[parent] child ready pid={pid}: {open(READY).read().strip()}",
          flush=True)

    cc(args.cuda_checkpoint, pid, "lock", ("--timeout", "30000"))
    cc(args.cuda_checkpoint, pid, "checkpoint")
    cc(args.cuda_checkpoint, pid, "restore")
    cc(args.cuda_checkpoint, pid, "unlock")

    open(STEP, "w").close()
    for _ in range(30):
        if os.path.exists(DONE):
            break
        time.sleep(1)
    print(f"[parent] VERDICT: {open(DONE).read().strip() if os.path.exists(DONE) else 'no result'}",
          flush=True)
    job.wait(timeout=15)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--cuda-checkpoint", default="/usr/local/bin/cuda-checkpoint")
    args = ap.parse_args()
    return child() if args.child else parent(args)


if __name__ == "__main__":
    sys.exit(main())
