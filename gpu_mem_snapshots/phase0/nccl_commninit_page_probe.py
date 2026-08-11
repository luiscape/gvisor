#!/usr/bin/env python3
"""Minimal, shim-free repro: does a bare cuda-checkpoint cycle drop the
`r--s /dev/nvidiactl` control page that ncclCommInitRank creates?

Single process, single GPU, WORLD=1 NCCL communicator -> NO peer imports, NO
multicast, NO mcshim, NO gVisor. Just:
  ncclCommInitRank(world=1) -> one r--s /dev/nvidiactl page appears
  cuda-checkpoint lock -> checkpoint -> restore -> unlock (job mode)
  -> is the page still mapped?

If the count drops, the loss is 100% cuda-checkpoint's (an RM control mapping
it fails to re-establish on restore), independent of everything this project
built. This is the repro to file with NVIDIA; it is the root cause of the
post-restore CUDA_ERROR_LAUNCH_FAILED (719) in the full stock-NCCL run.

Usage: sudo NCCL_LIB=/opt/phase0/nccl-stock/libnccl.so.2 \
            python3 nccl_commninit_page_probe.py
"""

import argparse
import ctypes
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cuda as cu
import _nccl as nc

READY = "/tmp/nccl_commninit_probe.ready"
PID = "/tmp/nccl_commninit_probe.pid"
STOP = "/tmp/nccl_commninit_probe.stop"


def ctl_pages(pid):
    n = 0
    try:
        for ln in open(f"/proc/{pid}/maps"):
            if "r--s" in ln and "nvidiactl" in ln:
                n += 1
    except OSError:
        return -1
    return n


def child():
    for f in (READY, PID, STOP):
        if os.path.exists(f):
            os.remove(f)
    cu.init_device(0)
    uid = nc.ncclUniqueId()
    nc.call("ncclGetUniqueId", ctypes.byref(uid))
    comm = nc.ncclComm_t()
    nc.call("ncclCommInitRank", ctypes.byref(comm), 1, uid, 0)
    with open(PID, "w") as f:
        f.write(str(os.getpid()))
    with open(READY, "w") as f:
        f.write("ready\n")
    while not os.path.exists(STOP):
        time.sleep(0.2)


def cc(binary, pid, action, extra=()):
    p = subprocess.run([binary, "--action", action, "--pid", str(pid), *extra],
                       capture_output=True, text=True, timeout=90)
    print(f"[parent] {action}: rc={p.returncode} {(p.stdout + p.stderr).strip()}",
          flush=True)
    return p.returncode


def parent(args):
    for f in (READY, PID, STOP):
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
    pid = int(open(PID).read())
    print(f"[parent] child pid={pid}", flush=True)

    pre = ctl_pages(pid)
    print(f"[parent] r--s /dev/nvidiactl pages PRE  = {pre}", flush=True)
    cc(args.cuda_checkpoint, pid, "lock", ("--timeout", "30000"))
    cc(args.cuda_checkpoint, pid, "checkpoint")
    cc(args.cuda_checkpoint, pid, "restore")
    cc(args.cuda_checkpoint, pid, "unlock")
    post = ctl_pages(pid)
    print(f"[parent] r--s /dev/nvidiactl pages POST = {post}", flush=True)

    open(STOP, "w").close()
    try:
        job.wait(timeout=15)
    except subprocess.TimeoutExpired:
        job.kill()

    print(f"\n[parent] VERDICT: cuda-checkpoint "
          f"{'DROPS' if post < pre else 'preserves'} the ncclCommInitRank "
          f"control page ({pre} -> {post}) -- "
          f"{'root cause of the 719, an NVIDIA cuda-checkpoint gap' if post < pre else 'not the culprit'}",
          flush=True)
    return 0 if post == pre else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--cuda-checkpoint", default="/usr/local/bin/cuda-checkpoint")
    args = ap.parse_args()
    return child() if args.child else parent(args)


if __name__ == "__main__":
    sys.exit(main())
