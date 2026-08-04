#!/usr/bin/env python3
"""Transparent multicast workload for the mcshim (Idea D) interposer.

Unlike multicast_workload.py, this workload contains NO suspend/resume/teardown
CUDA logic: it just creates a multicast group (cuMulticastCreate -> AddDevice ->
BindMem -> map MC VA + unicast VAs), writes known unicast patterns, and loops
verifying. The multicast suspend/resume is performed transparently by mcshim.so
(LD_PRELOAD), exactly as a generic gVisor-injected interposer would do for a
stock NCCL/PyTorch process that never calls ncclCommSuspend.

The only "cooperation" is the (a) quiesce step of the (a)->(b)->(c) checkpoint
orchestration: on a `pause` marker the workload stops issuing CUDA work (an
idle inference engine), so the shim can tear the multicast layer down without
racing concurrent kernels. That is orchestration, not a shim hook.

Markers under --dir (driven by the runner via `runsc exec` / touch):
  pause             existence-based: quiesce while the file exists (step (a));
                    removing it un-quiesces. Race-free for N rank processes.
  (the shim owns suspend / suspended.<pid> / resumed.<pid> in the same dir)

Verification (R610-safe; plain load/store through an MC VA is rejected on R610,
only multimem PTX works, which this ctypes harness can't emit): unicast
bound-vidmem patterns must survive checkpoint/restore, read at the VAs
recorded before the checkpoint. Multicast-VA identity is enforced by the shim
itself: its resume fails loudly unless the MC VA re-maps at the identical
address (see the RESUME lines in mcshim.log).
"""

import argparse
import ctypes
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cuda as cu

PATTERNS = (0xCAFE0000, 0xBEEF0000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--size", type=lambda s: int(s, 0), default=32 << 20)
    ap.add_argument("--dir", default="/tmp/mcshim")
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()
    ordinals = [int(g) for g in args.gpus.split(",")]
    assert len(ordinals) == 2, "exactly 2 GPUs"
    os.makedirs(args.dir, exist_ok=True)

    def status(line):
        tmp = os.path.join(args.dir, "wl.status.tmp")
        with open(tmp, "w") as f:
            f.write(line + "\n")
        os.replace(tmp, os.path.join(args.dir, "wl.status"))
        print(line, flush=True)

    def marker(name):
        return os.path.join(args.dir, name)

    cu.call("cuInit", 0)
    devs, ctxs = [], []
    for o in ordinals:
        dev = ctypes.c_int()
        cu.call("cuDeviceGet", ctypes.byref(dev), o)
        if cu.device_attr(cu.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev.value) != 1:
            status(f"FATAL no multicast support on device {o}")
            return 1
        ctx = ctypes.c_void_p()
        cu.call("cuDevicePrimaryCtxRetain", ctypes.byref(ctx), dev.value)
        devs.append(dev.value)
        ctxs.append(ctx)
    cu.call("cuCtxSetCurrent", ctxs[0])

    prop = cu.CUmulticastObjectProp()
    prop.numDevices = 2
    prop.handleTypes = cu.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    gran = ctypes.c_size_t()
    cu.call("cuMulticastGetGranularity", ctypes.byref(gran), ctypes.byref(prop),
            cu.CU_MULTICAST_GRANULARITY_RECOMMENDED)
    size = (max(args.size, gran.value) + gran.value - 1) // gran.value * gran.value
    prop.size = size

    # Create the multicast group + add devices. The shim tracks all of this.
    mc = ctypes.c_ulonglong()
    cu.call("cuMulticastCreate", ctypes.byref(mc), ctypes.byref(prop))
    export_fd = cu.export_posix_fd(mc.value)
    for d in devs:
        cu.call("cuMulticastAddDevice", mc.value, d)
    # Single-process: the export fd is not needed after AddDevice.
    os.close(export_fd)

    # Per-GPU shareable vidmem, bound into the group, plus unicast RW mappings.
    mems, uni_vas = [], []
    for i, d in enumerate(devs):
        cu.call("cuCtxSetCurrent", ctxs[i])
        p = cu.alloc_prop(d)
        memh = cu.mem_create(size, p)
        cu.call("cuMulticastBindMem", mc.value, 0, memh, 0, size, 0)
        va = cu.reserve_map_rw(memh, size, d)
        cu.memset_u32(va, PATTERNS[i], size // 4)
        mems.append(memh)
        uni_vas.append(va)

    # Map the multicast VA on GPU 0 (shim will unmap/re-map at this same VA).
    cu.call("cuCtxSetCurrent", ctxs[0])
    mc_va = cu.reserve_map_rw(mc.value, size, devs[0])
    cu.call("cuCtxSynchronize")

    status(f"READY pid={os.getpid()} mc=0x{mc.value:x} "
           f"mc_va=0x{mc_va:x} uni_vas={[hex(v) for v in uni_vas]} "
           f"size=0x{size:x}")

    def verify():
        """Read the patterns back at the ORIGINAL VAs: this validates both
        content survival and unicast VA stability in one step."""
        errs = []
        for i in range(2):
            cu.call("cuCtxSetCurrent", ctxs[i])
            head = cu.read_u32(uni_vas[i], 64)
            tail = cu.read_u32(uni_vas[i] + size - 256, 64)
            if any(g != PATTERNS[i] for g in head + tail):
                errs.append(f"gpu{i} unicast pattern lost (got {head[0]:#x})")
        return errs

    # Baseline.
    errs = verify()
    if errs:
        status("FAIL baseline: " + "; ".join(errs))
        return 1
    status("baseline mc-live pass -- safe to (a) pause then (b) suspend")

    it, failures = 0, 0
    restored = False
    paused = False
    while True:
        it += 1
        want_pause = os.path.exists(marker("pause"))
        if want_pause and not paused:
            paused = True
            status(f"PAUSED iter={it} (quiesced; no CUDA work -- shim may suspend)")
        elif not want_pause and paused:
            paused = False
            # A restore happened while we were paused iff the shim resumed.
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


if __name__ == "__main__":
    sys.exit(main())
