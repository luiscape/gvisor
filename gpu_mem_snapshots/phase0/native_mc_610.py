#!/usr/bin/env python3
"""Native (no gVisor) probe: can cuda-checkpoint's job mechanism on R610
checkpoint/restore a process holding a LIVE multicast (00FD) object by itself?

This decides the R610 design: if the job-mode checkpoint/restore works
natively, nvproxy should NOT suspend multicast (just forward ioctls); if it
hangs/fails, nvproxy suspend/replay is required.

  child = cuda-checkpoint --launch-job python3 <this --child>
    creates a 2-GPU multicast object (create + AddDevice + BindMem), maps the
    MC VA, writes READY, then idles.
  parent drives cuda-checkpoint lock -> checkpoint -> restore -> unlock on the
    child's python pid.

Usage:
  sudo python3 native_mc_610.py [--cuda-checkpoint /usr/local/bin/cuda-checkpoint]
"""

import argparse
import ctypes
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cuda as cu

READY = "/tmp/native_mc_610.ready"
STOP = "/tmp/native_mc_610.stop"


def child():
    # MODE env controls how much multicast state is live at checkpoint:
    #   full   (default): create+AddDevice+BindMem+map MC VA  (the real case)
    #   nobind:           create+AddDevice only (no memory bound, no MC VA)
    #   nomap:            create+AddDevice+BindMem, but MC VA NOT mapped
    # Isolates what makes cuda-checkpoint hang: the 00FD object itself, the
    # bound fabric memory, or the mapped multicast VA.
    mode = os.environ.get("MC_CHILD_MODE", "full")
    if os.path.exists(READY):
        os.remove(READY)
    cu.call("cuInit", 0)
    ctxs, devs = [], []
    for o in (0, 1):
        d = ctypes.c_int()
        cu.call("cuDeviceGet", ctypes.byref(d), o)
        c = ctypes.c_void_p()
        cu.call("cuDevicePrimaryCtxRetain", ctypes.byref(c), d.value)
        devs.append(d.value)
        ctxs.append(c)
    cu.call("cuCtxSetCurrent", ctxs[0])
    prop = cu.CUmulticastObjectProp()
    prop.numDevices = 2
    prop.handleTypes = cu.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    g = ctypes.c_size_t()
    cu.call("cuMulticastGetGranularity", ctypes.byref(g), ctypes.byref(prop), 1)
    size = g.value
    prop.size = size
    mc = ctypes.c_ulonglong()
    cu.call("cuMulticastCreate", ctypes.byref(mc), ctypes.byref(prop))
    fd = cu.export_posix_fd(mc.value)
    for d in devs:
        cu.call("cuMulticastAddDevice", mc.value, d)
    os.close(fd)
    mc_va = 0
    for i, d in enumerate(devs):
        cu.call("cuCtxSetCurrent", ctxs[i])
        p = cu.alloc_prop(d)
        memh = cu.mem_create(size, p)
        if mode != "nobind":
            cu.call("cuMulticastBindMem", mc.value, 0, memh, 0, size, 0)
        cu.reserve_map_rw(memh, size, d)
    if mode == "full":
        cu.call("cuCtxSetCurrent", ctxs[0])
        mc_va = cu.reserve_map_rw(mc.value, size, devs[0])
    cu.call("cuCtxSynchronize")
    with open(READY, "w") as f:
        f.write(f"{os.getpid()} mode={mode} mc=0x{mc.value:x} mc_va=0x{mc_va:x}\n")
    print(f"[child] ready pid={os.getpid()} mode={mode} mc_va=0x{mc_va:x}", flush=True)
    while not os.path.exists(STOP):
        time.sleep(0.5)
    print("[child] stopping", flush=True)


def find_cuda_pid(cc, job_pid):
    """The python child is a descendant of the launch-job process."""
    out = subprocess.run(["pgrep", "-P", str(job_pid)], capture_output=True,
                         text=True)
    kids = [int(x) for x in out.stdout.split()]
    # Prefer a pid that cuda-checkpoint reports as 'running'.
    for pid in [job_pid] + kids:
        st = subprocess.run([cc, "--get-state", "--pid", str(pid)],
                            capture_output=True, text=True)
        if "running" in (st.stdout + st.stderr):
            return pid
    return kids[0] if kids else job_pid


def action(cc, pid, act, timeout, extra=()):
    t0 = time.monotonic()
    try:
        p = subprocess.run([cc, "--action", act, "--pid", str(pid), *extra],
                           capture_output=True, text=True, timeout=timeout)
        rc, out, to = p.returncode, (p.stdout + p.stderr).strip(), False
    except subprocess.TimeoutExpired as e:
        rc, to = -1, True
        out = ((e.stdout or b"").decode() + (e.stderr or b"").decode()).strip()
    dt = time.monotonic() - t0
    print(f"[parent] {act}: rc={rc} {'TIMEOUT ' if to else ''}({dt:.1f}s) "
          f"out=[{out}]", flush=True)
    return rc, to


def parent(args):
    cc = args.cuda_checkpoint
    for f in (READY, STOP):
        if os.path.exists(f):
            os.remove(f)
    job = subprocess.Popen([cc, "--launch-job", sys.executable,
                            os.path.abspath(__file__), "--child"])
    for _ in range(120):
        if os.path.exists(READY):
            break
        if job.poll() is not None:
            print(f"[parent] job exited early rc={job.returncode}")
            return 1
        time.sleep(1)
    else:
        print("[parent] child never became ready")
        job.kill()
        return 1
    print(f"[parent] child ready: {open(READY).read().strip()}", flush=True)
    pid = find_cuda_pid(cc, job.pid)
    print(f"[parent] driving cuda-checkpoint on cuda pid {pid} "
          f"(job pid {job.pid})", flush=True)

    results = {}
    rc, to = action(cc, pid, "lock", 60, ("--timeout", "30000"))
    results["lock"] = rc
    if rc == 0:
        rc, to = action(cc, pid, "checkpoint", 90)
        results["checkpoint"] = "TIMEOUT/HANG" if to else rc
        if rc == 0:
            rc, _ = action(cc, pid, "restore", 90)
            results["restore"] = rc
        action(cc, pid, "unlock", 30)

    open(STOP, "w").close()
    job.wait(timeout=30)

    print("\n[parent] ==== NATIVE R610 MULTICAST RESULT ====", flush=True)
    print(f"[parent] {results}", flush=True)
    ok = results.get("checkpoint") == 0 and results.get("restore") == 0
    if ok:
        print("[parent] VERDICT: cuda-checkpoint job mode checkpoints+restores "
              "LIVE multicast natively => nvproxy should NOT suspend on R610; "
              "just forward ioctls.", flush=True)
    else:
        print("[parent] VERDICT: cuda-checkpoint job mode CANNOT handle live "
              "multicast natively (see results) => nvproxy suspend/replay is "
              "required; the toggle failure is intrinsic.", flush=True)
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--cuda-checkpoint", default="/usr/local/bin/cuda-checkpoint")
    args = ap.parse_args()
    if args.child:
        child()
        return 0
    return parent(args)


if __name__ == "__main__":
    sys.exit(main())
