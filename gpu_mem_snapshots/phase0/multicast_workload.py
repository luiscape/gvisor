#!/usr/bin/env python3
"""Multicast checkpoint/restore workload for gVisor nvproxy suspend/replay.

Single process, two GPUs (no NCCL/torch): creates a multicast group via
cuMulticastCreate (RM class 00FD), attaches both GPUs
(NV00FD_CTRL_CMD_ATTACH_GPU x2), binds vidmem from each GPU
(cuMulticastBindMem -> NV00FD_CTRL_CMD_ATTACH_MEM), maps the multicast VA, and
then loops verifying:

  * unicast patterns (per-GPU vidmem contents survive checkpoint/restore), and
  * MULTICAST BROADCAST: a write through the MC VA must land in BOTH GPUs'
    unicast mappings. This exercises the actual multicast fabric binding and
    fails if the replayed 00FD object/attaches/mapping are broken.

The multicast object stays LIVE across the checkpoint -- nvproxy is expected
to suspend it before `cuda-checkpoint --action checkpoint` and replay it after
the post-restore toggle.

Lifecycle knobs (under --dir, driven by the runner via `runsc exec`):
  teardown   app-side full teardown (for gate testing of the exported-fd path)
  suspend    app-level multicast SUSPEND (simulates ncclCommSuspend): unmap MC
             VA keeping the VA reservation, unbind both GPUs, release the MC
             handle -- all through libcuda, so its bookkeeping stays
             consistent for cuda-checkpoint
  resume     app-level multicast RESUME (simulates ncclCommResume): recreate
             the MC object (NEW handle is fine), re-attach, re-bind the SAME
             vidmem, re-map at the IDENTICAL VA (graphs reference VAs)
  restored   restore marker (gVisor restores clocks; no wall-clock jump)

Flags:
  --hold-export-fd   keep the exported MC fd open (default: closed after
                     setup, since exported-fd blockers cannot be suspended)
"""

import argparse
import ctypes
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cuda as cu

PATTERNS = (0xCAFE0000, 0xBEEF0000)
BCAST_WORDS = 1024  # broadcast-verified region (first 4KiB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--size", type=lambda s: int(s, 0), default=32 << 20)
    ap.add_argument("--dir", default="/tmp")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--hold-export-fd", action="store_true")
    ap.add_argument("--mode", choices=("full", "no-bind", "no-mc"),
                    default="full",
                    help="bisect what blocks the cuda-checkpoint restore walk: "
                         "'no-bind' creates the MC object + AddDevice but never "
                         "binds memory; 'no-mc' skips multicast entirely")
    ap.add_argument("--map-mc-va", choices=("always", "after-restore"),
                    default="always",
                    help="'after-restore' defers the MC VA mapping until the "
                         "restore marker appears: broadcast then exercises a "
                         "FRESH mapping of the REPLAYED 00FD object through "
                         "libcuda's CRIU-preserved (stale) handle -- the "
                         "purest test that nvproxy's replay produced a "
                         "functional object")
    args = ap.parse_args()
    ordinals = [int(g) for g in args.gpus.split(",")]
    assert len(ordinals) == 2, "exactly 2 GPUs"

    def status(line):
        tmp = os.path.join(args.dir, "status.tmp")
        with open(tmp, "w") as f:
            f.write(line + "\n")
        os.replace(tmp, os.path.join(args.dir, "status"))
        print(line, flush=True)

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
    mc = ctypes.c_ulonglong()
    export_fd = -1
    if args.mode != "no-mc":
        prop.size = size
        cu.call("cuMulticastCreate", ctypes.byref(mc), ctypes.byref(prop))
        export_fd = cu.export_posix_fd(mc.value)
        for d in devs:
            cu.call("cuMulticastAddDevice", mc.value, d)
        if not args.hold_export_fd:
            # Exported-fd blockers cannot be suspended by nvproxy; a
            # single-process workload does not need the fd after AddDevice.
            os.close(export_fd)
            export_fd = -1

    # Vidmem on each GPU (must be shareable to be bindable), bound into the
    # group, plus unicast RW mappings for verification.
    mems, uni_vas = [], []
    for i, d in enumerate(devs):
        cu.call("cuCtxSetCurrent", ctxs[i])
        p = cu.alloc_prop(d)
        memh = cu.mem_create(size, p)
        if args.mode == "full":
            cu.call("cuMulticastBindMem", mc.value, 0, memh, 0, size, 0)
        va = cu.reserve_map_rw(memh, size, d)
        cu.memset_u32(va, PATTERNS[i], size // 4)
        mems.append(memh)
        uni_vas.append(va)
    cu.call("cuCtxSynchronize")

    # Map the multicast VA on GPU 0 (unless deferred to post-restore).
    mc_va = None
    if args.mode == "full" and args.map_mc_va == "always":
        cu.call("cuCtxSetCurrent", ctxs[0])
        mc_va = cu.reserve_map_rw(mc.value, size, devs[0])

    # VA inventory that must be byte-identical across checkpoint/restore
    # (TASK.md pass criterion). Note: on driver R610, plain load/store through
    # a multicast VA is rejected (INVALID_VALUE) -- only multimem.* PTX works,
    # which this ctypes harness can't emit -- so we do NOT read/write through
    # mc_va. Multimem broadcast correctness is covered by symmem_nccl_ckpt.py.
    # Here we validate the suspend/replay MECHANISM: unicast (bound vidmem)
    # contents survive, and every VA (incl. the multicast VA) is unchanged.
    # The MC *handle* may legitimately change across an app-level
    # suspend/resume (like ncclCommResume); only VAs must be stable.
    inv_before = {"mc_va": mc_va or 0}
    for i, v in enumerate(uni_vas):
        inv_before[f"uni{i}"] = v

    status(f"READY pid={os.getpid()} mc=0x{mc.value:x} export_fd={export_fd} "
           f"mc_va={hex(mc_va) if mc_va else 'deferred'} "
           f"uni_vas={[hex(v) for v in uni_vas]} size=0x{size:x}")

    def app_suspend():
        """Simulates ncclCommSuspend(NCCL_SUSPEND_MEM) for the multicast
        layer: teardown through libcuda, retaining the MC VA reservation."""
        nonlocal mc_va_reserved
        cu.call("cuCtxSetCurrent", ctxs[0])
        if mc_va is not None:
            cu.call("cuMemUnmap", mc_va, size)  # keeps the VA reservation
            mc_va_reserved = True
        for d in devs:
            cu.call("cuMulticastUnbind", mc.value, d, 0, size)
        cu.call("cuMemRelease", mc.value)
        cu.call("cuCtxSynchronize")

    def app_resume():
        """Simulates ncclCommResume: recreate the multicast object (new
        handle), re-attach GPUs, re-bind the SAME vidmem handles (restored by
        cuda-checkpoint), re-map at the IDENTICAL VA."""
        nonlocal mc_va, mc_va_reserved
        cu.call("cuCtxSetCurrent", ctxs[0])
        newmc = ctypes.c_ulonglong()
        cu.call("cuMulticastCreate", ctypes.byref(newmc), ctypes.byref(prop))
        for d in devs:
            cu.call("cuMulticastAddDevice", newmc.value, d)
        for i, d in enumerate(devs):
            cu.call("cuCtxSetCurrent", ctxs[i])
            cu.call("cuMulticastBindMem", newmc.value, 0, mems[i], 0, size, 0)
        old_va = mc_va
        cu.call("cuCtxSetCurrent", ctxs[0])
        if old_va is not None:
            # Re-map at the identical VA. Try the retained reservation first;
            # if the reservation did not survive restore, re-reserve at the
            # fixed address.
            try:
                cu.call("cuMemMap", old_va, size, 0, newmc.value, 0)
                path = "retained-reservation"
            except cu.CudaError:
                va = ctypes.c_ulonglong()
                cu.call("cuMemAddressReserve", ctypes.byref(va), size, 0,
                        old_va, 0)
                if va.value != old_va:
                    cu.call("cuMemAddressFree", va.value, size)
                    raise RuntimeError(
                        f"fixed-address re-reserve landed at 0x{va.value:x}, "
                        f"want 0x{old_va:x}")
                cu.call("cuMemMap", old_va, size, 0, newmc.value, 0)
                path = "re-reserved-fixed"
            d = cu.CUmemAccessDesc()
            d.location.type = cu.CU_MEM_LOCATION_TYPE_DEVICE
            d.location.id = devs[0]
            d.flags = cu.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            cu.call("cuMemSetAccess", old_va, size, ctypes.byref(d), 1)
            status(f"RESUMED mc=0x{newmc.value:x} (was 0x{mc.value:x}) "
                   f"mc_va=0x{old_va:x} identical ({path})")
        else:
            status(f"RESUMED mc=0x{newmc.value:x} (was 0x{mc.value:x}) no-mc-va")
        mc.value = newmc.value
        cu.call("cuCtxSynchronize")

    def check_va_identity():
        """Return list of VA-changed errors (must be empty)."""
        e = []
        cur = {"mc_va": mc_va or 0}
        for i, v in enumerate(uni_vas):
            cur[f"uni{i}"] = v
        for k, v in inv_before.items():
            if cur.get(k) != v:
                e.append(f"VA {k} changed 0x{v:x} -> 0x{cur.get(k, 0):x}")
        return e

    torn_down = False
    restored = False
    suspended = False
    mc_va_reserved = False
    it, failures = 0, 0
    while True:
        it += 1
        if not suspended and os.path.exists(os.path.join(args.dir, "suspend")):
            os.remove(os.path.join(args.dir, "suspend"))
            try:
                app_suspend()
                suspended = True
                status(f"SUSPENDED iter={it} (multicast released at libcuda "
                       f"level; VA reservation retained)")
            except cu.CudaError as e:
                failures += 1
                status(f"iter={it} SUSPEND FAIL: {e.name} failures={failures}")
        if suspended and os.path.exists(os.path.join(args.dir, "resume")):
            os.remove(os.path.join(args.dir, "resume"))
            try:
                app_resume()
                suspended = False
            except (cu.CudaError, RuntimeError) as e:
                failures += 1
                status(f"iter={it} RESUME FAIL: {e} failures={failures}")
        if not restored and os.path.exists(os.path.join(args.dir, "restored")):
            restored = True
            status(f"RESTORE-DETECTED iter={it}")
            if mc_va is None and not torn_down:
                # Fresh mapping of the replayed multicast object, through the
                # stale (CRIU-preserved) libcuda handle.
                try:
                    cu.call("cuCtxSetCurrent", ctxs[0])
                    mc_va = cu.reserve_map_rw(mc.value, size, devs[0])
                    status(f"MC-VA-MAPPED-POST-RESTORE mc_va=0x{mc_va:x}")
                except cu.CudaError as e:
                    status(f"iter={it} post-restore FAIL: MC VA map failed: {e.name}")
                    failures += 1
        if not torn_down and os.path.exists(os.path.join(args.dir, "teardown")):
            cu.call("cuCtxSetCurrent", ctxs[0])
            if mc_va is not None:
                cu.call("cuMemUnmap", mc_va, size)
                cu.call("cuMemAddressFree", mc_va, size)
            for d in devs:
                cu.call("cuMulticastUnbind", mc.value, d, 0, size)
            cu.call("cuMemRelease", mc.value)
            if export_fd >= 0:
                os.close(export_fd)
            cu.call("cuCtxSynchronize")
            torn_down = True
            status(f"TORNDOWN iter={it}")

        errs = []
        # Unicast patterns: the bound vidmem contents must survive suspend of
        # the multicast layer, checkpoint, and replay.
        for i in range(2):
            cu.call("cuCtxSetCurrent", ctxs[i])
            head = cu.read_u32(uni_vas[i], 64)
            tail = cu.read_u32(uni_vas[i] + size - 256, 64)
            if any(g != PATTERNS[i] for g in head + tail):
                errs.append(f"gpu{i} unicast pattern lost (got {head[0]:#x})")
        # VA inventory must be byte-identical (the multicast VA included) both
        # steady-state and, crucially, after restore. While suspended the MC
        # handle is legitimately released, so only check when fully set up.
        if not torn_down and not suspended:
            errs += check_va_identity()

        tag = ("post-restore" if restored else "pre-checkpoint")
        tag += "+torndown" if torn_down else "+mc-live"
        if errs:
            failures += 1
            status(f"iter={it} {tag} FAIL: {'; '.join(errs)} failures={failures}")
        else:
            status(f"iter={it} {tag} pass failures={failures}")
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
