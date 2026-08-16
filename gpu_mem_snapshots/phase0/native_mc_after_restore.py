#!/usr/bin/env python3
"""Native (no gVisor) probe: can a cuda-checkpoint-restored process create and
attach a NEW multicast (00FD) object on this driver?

This is the R580 question. The interposer's rebuild does exactly this sequence
after every restore (cuMulticastCreate -> AddDevice -> BindMem -> map); on
610.57.04 it works. On 580.126.20 the e2e run fails at AddDevice, so this
probe isolates whether that is a driver limitation (fails natively too) or
something in the gVisor/nvproxy/interposer stack (passes natively).

  child: init 2 GPUs, create a plain VMM allocation per GPU, write READY,
         idle. When RESUME appears: cuMulticastCreate + AddDevice(x2) +
         BindMem(x2) + map, write each step's result to RESULT, idle to STOP.
  parent: waits READY, drives cuda-checkpoint lock -> checkpoint -> restore ->
         unlock on the child, touches RESUME, prints RESULT.

Modes:
  --no-cr     skip the checkpoint cycle (control: proves the sequence works
              on this driver in a never-checkpointed process)
  --job       launch the child under `cuda-checkpoint --launch-job`

Usage:
  sudo python3 native_mc_after_restore.py [--no-cr] [--job] \
      [--cuda-checkpoint /usr/local/bin/cuda-checkpoint]
"""

import argparse
import ctypes
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cuda as cu

READY = "/tmp/native_mc_ar.ready"
RESUME = "/tmp/native_mc_ar.resume"
RESULT = "/tmp/native_mc_ar.result"
STOP = "/tmp/native_mc_ar.stop"


def _step(results, name, fn):
    try:
        out = fn()
        results.append(f"{name}=OK")
        return out
    except cu.CudaError as e:
        results.append(f"{name}=CUresult({e})")
        raise


def child():
    for f in (READY, RESULT):
        if os.path.exists(f):
            os.remove(f)
    cu.call("cuInit", 0)
    ctxs, devs = [], []
    for o in (0, 1):
        d = ctypes.c_int()
        cu.call("cuDeviceGet", ctypes.byref(d), o)
        c = ctypes.c_void_p()
        cu.call("cuDevicePrimaryCtxRetain", ctypes.byref(c), d.value)
        devs.append(d.value)
        ctxs.append(c)
    # Realistic pre-checkpoint state: one plain VMM allocation per GPU
    # (POSIX_FD-capable, like NCCL_CUMEM_ENABLE=1 allocations), never exported.
    prop0 = cu.alloc_prop(devs[0])
    gran = cu.alloc_granularity(prop0)
    size = gran
    memhs = []
    for i, d in enumerate(devs):
        cu.call("cuCtxSetCurrent", ctxs[i])
        p = cu.alloc_prop(d)
        memh = cu.mem_create(size, p)
        cu.reserve_map_rw(memh, size, d)
        memhs.append(memh)
    cu.call("cuCtxSetCurrent", ctxs[0])
    cu.call("cuCtxSynchronize")
    with open(READY, "w") as f:
        f.write(f"{os.getpid()}\n")
    print(f"[child] ready pid={os.getpid()} size=0x{size:x}", flush=True)

    while not os.path.exists(RESUME):
        time.sleep(0.2)
    print("[child] RESUME seen; building multicast", flush=True)

    results = []
    try:
        cu.call("cuCtxSetCurrent", ctxs[0])
        _step(results, "ctx_sync_after_restore", lambda: cu.call("cuCtxSynchronize"))
        mprop = cu.CUmulticastObjectProp()
        mprop.numDevices = 2
        mprop.handleTypes = cu.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        g = ctypes.c_size_t()
        _step(results, "granularity", lambda: cu.call(
            "cuMulticastGetGranularity", ctypes.byref(g), ctypes.byref(mprop), 1))
        mcsize = g.value
        mprop.size = mcsize
        mc = ctypes.c_ulonglong()
        _step(results, "create", lambda: cu.call(
            "cuMulticastCreate", ctypes.byref(mc), ctypes.byref(mprop)))
        for d in devs:
            _step(results, f"add_device_{d}", lambda d=d: cu.call(
                "cuMulticastAddDevice", mc.value, d))
        # Bind fresh allocations of multicast granularity (the pre-restore
        # ones may be smaller than the MC granularity).
        for i, d in enumerate(devs):
            cu.call("cuCtxSetCurrent", ctxs[i])
            p = cu.alloc_prop(d)
            bmemh = cu.mem_create(mcsize, p)
            _step(results, f"bind_mem_{d}", lambda m=bmemh: cu.call(
                "cuMulticastBindMem", mc.value, 0, m, 0, mcsize, 0))
        cu.call("cuCtxSetCurrent", ctxs[0])
        _step(results, "map_mc_va", lambda: cu.reserve_map_rw(
            mc.value, mcsize, devs[0]))
        results.append("ALL=PASS")
    except cu.CudaError:
        results.append("ALL=FAIL")
    with open(RESULT, "w") as f:
        f.write("\n".join(results) + "\n")
    print("[child] result: " + " ".join(results), flush=True)

    while not os.path.exists(STOP):
        time.sleep(0.5)
    print("[child] stopping", flush=True)


def cc_action(cc, pid, *args):
    r = subprocess.run([cc, *args, "--pid", str(pid)],
                       capture_output=True, text=True)
    tag = " ".join(args)
    print(f"[parent] cuda-checkpoint {tag}: rc={r.returncode} "
          f"out={r.stdout.strip()!r} err={r.stderr.strip()!r}", flush=True)
    return r.returncode


def find_cuda_pid(job_pid):
    out = subprocess.run(["pgrep", "-P", str(job_pid)], capture_output=True,
                         text=True)
    kids = [int(x) for x in out.stdout.split()]
    return kids[0] if kids else job_pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--no-cr", action="store_true")
    ap.add_argument("--job", action="store_true")
    ap.add_argument("--cuda-checkpoint",
                    default="/usr/local/bin/cuda-checkpoint")
    args = ap.parse_args()
    if args.child:
        child()
        return 0

    for f in (READY, RESUME, RESULT, STOP):
        if os.path.exists(f):
            os.remove(f)
    argv = [sys.executable, os.path.abspath(__file__), "--child"]
    if args.job:
        argv = [args.cuda_checkpoint, "--launch-job"] + argv
    proc = subprocess.Popen(argv)
    try:
        while not os.path.exists(READY):
            if proc.poll() is not None:
                print("[parent] child died before READY", flush=True)
                return 1
            time.sleep(0.2)
        with open(READY) as f:
            pid = int(f.read().split()[0])
        cuda_pid = find_cuda_pid(proc.pid) if args.job else pid
        print(f"[parent] child ready, cuda pid={cuda_pid}", flush=True)

        if not args.no_cr:
            if cc_action(args.cuda_checkpoint, cuda_pid, "--action", "lock",
                         "--timeout", "10000"):
                return 1
            if cc_action(args.cuda_checkpoint, cuda_pid, "--action",
                         "checkpoint"):
                return 1
            print("[parent] checkpointed; GPU should be empty; restoring",
                  flush=True)
            if cc_action(args.cuda_checkpoint, cuda_pid, "--action",
                         "restore"):
                return 1
            if cc_action(args.cuda_checkpoint, cuda_pid, "--action",
                         "unlock"):
                return 1
        else:
            print("[parent] --no-cr: skipping checkpoint cycle", flush=True)

        open(RESUME, "w").close()
        deadline = time.time() + 120
        while not os.path.exists(RESULT) and time.time() < deadline:
            if proc.poll() is not None:
                print("[parent] child died before RESULT", flush=True)
                return 1
            time.sleep(0.2)
        if not os.path.exists(RESULT):
            print("[parent] TIMEOUT waiting for RESULT", flush=True)
            return 1
        with open(RESULT) as f:
            body = f.read()
        print("[parent] ===== RESULT =====\n" + body, flush=True)
        return 0 if "ALL=PASS" in body else 2
    finally:
        open(STOP, "w").close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
