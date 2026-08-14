#!/usr/bin/env python3
"""Does cuIpcOpenMemHandle give back the SAME address after a close/reopen?

`IPC_CHECKPOINT_BISECT.md` shows that in job mode a live legacy CUDA IPC
import is the only thing blocking restore, and that closing it before the
checkpoint is sufficient. That suggests extending mcshim to close legacy
imports at suspend and reopen them at resume, exactly as it already does for
VMM imports.

There is one reason that might not work, and it needs measuring before any of
it is built. The whole design rests on the invariant that every GPU virtual
address is identical after restore -- CRIU restores libcuda's structures and
the application's pointers verbatim, so a moved buffer is silent corruption,
not an error. For VMM imports mcshim can *force* that: cuMemAddressReserve
takes an address hint, so the replayed mapping is placed exactly where it was.

    cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle, uint flags)

has no such parameter. The driver picks the address. So either it happens to
return the same one, or the approach is not viable in this form.

This probe answers that, and nothing else:

  1. in-process close/reopen -- is the address stable at all?
  2. close/reopen with an intervening allocation -- is it *deterministic*, or
     merely "whatever was next free"?
  3. across a real cuda-checkpoint checkpoint/restore, which is the case that
     actually matters.

Run it in job mode, since that is the only mode where legacy IPC is
checkpointable at all:

    cuda-checkpoint --launch-job python3 legacy_va_probe.py

Exit code 0 only if the address is identical in every case.
"""

import argparse
import multiprocessing
import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cuda as cu
from ipc_scale_probe import Chan, ckpt, parallel

BUF_BYTES = 4 << 20
PATTERN = 0xC0DE0000


def exporter(chan, args):
    tag = "[exporter]"
    dev = cu.init_device(args.gpu)
    ptr = cu.mem_alloc(BUF_BYTES)
    cu.memset_u32(ptr, PATTERN, 64)
    cu.call("cuCtxSynchronize")
    chan.send(f"BLOB pid={os.getpid()} blob={cu.ipc_get_handle(ptr).hex()}")

    # Re-export on demand: after a restore the exporter's own state has been
    # rebuilt, so the importer must not assume its old blob is still valid.
    while True:
        msg, _ = chan.recv()
        if msg.startswith("EXIT"):
            break
        if msg.startswith("REEXPORT"):
            cu.call("cuCtxSynchronize")
            chan.send(f"BLOB blob={cu.ipc_get_handle(ptr).hex()}")
    os._exit(0)


def importer(chan, args):
    """Open the peer's handle, then probe how stable the returned VA is."""
    tag = "[importer]"
    dev = cu.init_device(args.peer_gpu)
    msg, _ = chan.recv(expect="BLOB")
    blob = bytes.fromhex(msg.split("blob=")[1])

    va1 = cu.ipc_open_handle(blob)
    got = cu.read_u32(va1, 1)[0]
    if got != PATTERN:
        raise RuntimeError(f"import readback 0x{got:08x} != 0x{PATTERN:08x}")
    cu.log(tag, f"open #1                       va=0x{va1:016x}")

    # (1) plain close/reopen.
    cu.ipc_close_handle(va1)
    va2 = cu.ipc_open_handle(blob)
    cu.log(tag, f"open #2 (after close)         va=0x{va2:016x} "
                f"{'SAME' if va2 == va1 else 'CHANGED'}")

    # (2) close, allocate something else, reopen. If the driver just hands
    # out the next free slot, this perturbation moves the import.
    cu.ipc_close_handle(va2)
    filler = cu.mem_alloc(BUF_BYTES * 4)
    va3 = cu.ipc_open_handle(blob)
    cu.log(tag, f"open #3 (after interposed alloc) va=0x{va3:016x} "
                f"{'SAME' if va3 == va1 else 'CHANGED'}")
    cu.call("cuMemFree_v2", filler)

    # va2 == va1 is a requirement: a plain close/reopen must be transparent.
    # va3 is a *diagnostic*, not a requirement -- if an interposed allocation
    # moves the import, the driver is handing out the next free slot rather
    # than a stable address, which tells the interposer it must reopen in the
    # original order with nothing allocated in between.
    reopen_stable = (va2 == va1)
    order_sensitive = (va3 != va1)

    # Can placement be FORCED back? This decides whether legacy IPC imports
    # can be replayed at all.
    #
    # cuIpcOpenMemHandle takes no address hint, but it appears to hand out the
    # lowest free address in its region -- which is exactly why a reopen after
    # the address space has changed lands BELOW the original (measured in
    # vLLM: every import moved down, into one dense run). If that is the rule,
    # then reserving the gap underneath the target should push the driver back
    # up to the original address.
    #
    # va3 is deliberately a moved address (an allocation was interposed above),
    # so this reproduces the failing case and then tries to repair it.
    # Can a moved import be forced back to an EXACT address?
    #
    # The steering probe below shows a reservation is respected, but "lands
    # somewhere else" is not the requirement -- "lands exactly on the recorded
    # address" is. This reproduces the vLLM failure direction first (the import
    # comes back BELOW where it was) and then tries to repair it.
    #
    # To get a downward move, mimic what the engine does: something occupies
    # the low region when the import is first opened, and is gone by the time
    # it is reopened (in vLLM, the weights and KV cache that /sleep released).
    exact = None
    cu.ipc_close_handle(va3)
    filler = cu.mem_alloc(BUF_BYTES)          # occupy the low slot
    va_hi = cu.ipc_open_handle(blob)          # import lands above it
    cu.ipc_close_handle(va_hi)
    cu.call("cuMemFree_v2", filler)           # low slot free again
    va_lo = cu.ipc_open_handle(blob)          # ... so this packs low
    cu.log(tag, f"reopen after freeing the filler  va=0x{va_lo:016x} "
                f"(target 0x{va_hi:016x}, "
                f"{'MOVED DOWN' if va_lo < va_hi else 'did not move down'})")
    if va_lo < va_hi:
        cu.ipc_close_handle(va_lo)
        gap = va_hi - va_lo
        try:
            resv = cu.reserve_va(gap, va_lo)  # fence off everything below
            va_fix = cu.ipc_open_handle(blob)
            exact = (va_fix == va_hi)
            cu.log(tag, f"reopen with gap 0x{gap:x} fenced at 0x{va_lo:016x} "
                        f"va=0x{va_fix:016x} "
                        + ("EXACT -- placement is controllable"
                           if exact else
                           f"off by {va_fix - va_hi:+d} bytes"))
            cu.ipc_close_handle(va_fix)
            cu.call("cuMemAddressFree", resv, gap)
        except Exception as e:  # noqa: BLE001 - a failure here is the answer
            cu.log(tag, f"exact-placement attempt failed: {e}")
    va3 = cu.ipc_open_handle(blob)

    # Does a VA reservation count as "occupied" for cuIpcOpenMemHandle's
    # placement? If it does, the driver's choice is steerable: fence off the
    # region it would otherwise pick and it must land elsewhere. If it does
    # not -- if the import lands on top of a reservation, or the reservation
    # is simply ignored -- then placement cannot be controlled at all and
    # replaying legacy IPC imports at fixed addresses is impossible.
    steerable = None
    cu.ipc_close_handle(va3)
    va1 = va1  # (target for the steering probe below is the original address)
    try:
        resv = cu.reserve_va(BUF_BYTES, va1)  # fence off the natural slot
        va_f = cu.ipc_open_handle(blob)
        steerable = (va_f != va1)
        cu.log(tag, f"open #3b (natural slot 0x{va1:016x} reserved) "
                    f"va=0x{va_f:016x} "
                    + ("AVOIDED the reservation -> steerable"
                       if steerable else
                       "IGNORED the reservation -> NOT steerable"))
        cu.ipc_close_handle(va_f)
        cu.call("cuMemAddressFree", resv, BUF_BYTES)
    except Exception as e:  # noqa: BLE001 - a failure here is itself the answer
        cu.log(tag, f"open #3b steering probe failed: {e}")
    va3 = cu.ipc_open_handle(blob)

    # (3) the case that matters: close the import, let the parent drive a real
    # checkpoint/restore, then reopen and compare.
    cu.ipc_close_handle(va3)
    chan.send(f"READY pid={os.getpid()} va=0x{va1:016x} "
              f"reopen_stable={int(reopen_stable)}")

    chan.recv(expect="REOPEN")
    chan.send("REEXPORT")  # ask the parent to refresh the blob
    msg, _ = chan.recv(expect="BLOB")
    blob2 = bytes.fromhex(msg.split("blob=")[1])
    blob_same = blob2 == blob

    va4 = cu.ipc_open_handle(blob2)
    got = cu.read_u32(va4, 1)[0]
    cu.log(tag, f"open #4 (after C/R)           va=0x{va4:016x} "
                f"{'SAME' if va4 == va1 else 'CHANGED'}")
    ok = (va4 == va1) and (got == PATTERN) and reopen_stable
    chan.send(f"RESULT {'PASS' if ok else 'FAIL'} va4=0x{va4:016x} "
              f"va_survives_cr={int(va4 == va1)} content=0x{got:08x} "
              f"blob_changed={int(not blob_same)} "
              f"reopen_stable={int(reopen_stable)} "
              f"order_sensitive={int(order_sensitive)} "
              f"placement_steerable={steerable} exact_placement={exact}")
    chan.recv(expect="EXIT")
    chan.send("EXIT")  # release the exporter rather than EOF-ing it
    os._exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--peer-gpu", type=int, default=1)
    ap.add_argument("--cuda-checkpoint", default=os.environ.get(
        "CUDA_CHECKPOINT", "/usr/local/bin/cuda-checkpoint"))
    args = ap.parse_args()

    job = os.environ.get("CUDA_CHECKPOINT_JOB_FILE")
    print(f"job mode: {job if job else 'OFF -- legacy IPC is not '
                                      'checkpointable at all this way'}",
          flush=True)

    mp = multiprocessing.get_context("fork")
    pe, ce = socket.socketpair()   # parent <-> exporter
    pi, ci = socket.socketpair()   # parent <-> importer
    # And a direct exporter<->importer link for the blob handoff.
    xe, xi = socket.socketpair()

    def run_exp():
        cu.close_all(pe, pi, ci, xi)
        exporter(Chan(xe), args)

    def run_imp():
        cu.close_all(pe, pi, ce, xe)
        importer_with_parent(Chan(xi), Chan(ci), args)

    procs = [mp.Process(target=run_exp), mp.Process(target=run_imp)]
    for p in procs:
        p.start()
    cu.close_all(ce, ci, xe, xi)
    c_imp = Chan(pi)

    msg, _ = c_imp.recv(expect="READY")
    imp_pid = int(msg.split("pid=")[1].split()[0])
    va_before = msg.split("va=")[1].split()[0]
    exp_pid = procs[0].pid
    print(f"importer pid={imp_pid} exporter pid={exp_pid} "
          f"va_before={va_before}", flush=True)

    pids = [exp_pid, imp_pid]
    phases = {}
    for action, extra in (("lock", ("--timeout", "30000")),
                          ("checkpoint", ()), ("restore", ()), ("unlock", ())):
        res = parallel(
            lambda p, a=action, e=extra: ckpt(args.cuda_checkpoint, p, a, extra=e),
            pids)
        phases[action] = res
        bad = [r for r in res if r["rc"] != 0]
        print(f"  {action:10s} {'ok' if not bad else 'FAIL'}"
              + ("".join(f"\n      {r['out']}" for r in bad)), flush=True)
        if bad and action in ("lock", "checkpoint"):
            print("cannot proceed; the C/R leg of this probe did not run.",
                  flush=True)
            for p in procs:
                p.kill()
            return 2

    c_imp.send("REOPEN")
    c_imp.sock.settimeout(120)
    try:
        msg, _ = c_imp.recv(expect="RESULT")
    except (socket.timeout, EOFError, RuntimeError) as e:
        print(f"importer did not report back: {e}", flush=True)
        for p in procs:
            p.kill()
        return 1
    print(f"\n{msg}", flush=True)
    c_imp.send("EXIT")
    for p in procs:
        p.join(timeout=15)
        if p.is_alive():
            p.kill()
    return 0 if msg.startswith("RESULT PASS") else 1


def importer_with_parent(chan_peer, chan_parent, args):
    """The importer talks to the exporter for blobs and to the parent for
    checkpoint sequencing; bridge the two so importer() stays readable."""
    class Bridge:
        def send(self, msg, fds=()):
            (chan_peer if msg.startswith(("REEXPORT", "EXIT"))
             else chan_parent).send(msg, fds)

        def recv(self, expect=None, maxfds=64):
            src = chan_peer if (expect == "BLOB" or expect is None) \
                else chan_parent
            return src.recv(expect=expect, maxfds=maxfds)

        @property
        def sock(self):
            return chan_parent.sock

    importer(Bridge(), args)


if __name__ == "__main__":
    sys.exit(main())
