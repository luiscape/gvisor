#!/usr/bin/env python3
"""Phase 0 measurement #2: cuMulticastAddDevice blocking semantics (TASK.md
work item 4, "Measure this with 2 ranks before implementing").

Questions, in replay-scheduler terms:
  Q1. Does cuMulticastAddDevice (RM NV00FD_CTRL_CMD_ATTACH_GPU) block until
      ALL participating GPUs have joined? If yes, nvproxy's serial afterLoad
      replay across clients would deadlock => work item 4 is mandatory.
  Q2. What happens to cuMulticastBindMem (RM NV00FD_CTRL_CMD_ATTACH_MEM)
      issued BEFORE all devices have joined -- fast error, success, or hang?
      This constrains replay ordering: all ATTACH_GPU across clients must
      complete before any ATTACH_MEM replays iff early bind misbehaves.

Native (no gVisor). Two processes, one GPU each:

  rank A (GPU 0)                          rank B (GPU 1)
  --------------                          --------------
  t=0  cuMulticastCreate(numDevices=2)
       export fd  ------------fd------->  (received, sleeps DELAY secs)
       cuMulticastAddDevice(gpu0) <TIMED=Q1>
       cuMulticastBindMem       <TIMED=Q2, expected to fail fast>
                                  t=DELAY import fd
                                          cuMulticastAddDevice(gpu1) <TIMED>
       -- parent barrier: both ranks have joined --
       cuMulticastBindMem <TIMED>         cuMulticastBindMem <TIMED>

Interpretation of rank A's Q1 timing (DELAY defaults to 8s):
  addDevice ~0s     -> NON-BLOCKING: serial ATTACH_GPU replay is safe
  addDevice ~DELAY  -> BLOCKING on all-join: work item 4 is REQUIRED
  watchdog (rc=3)   -> indefinite hang: work item 4 + timeouts everywhere

Exit codes: 0 = measured (see VERDICT lines), 1 = error, 3 = watchdog hang.

Usage:
  sudo python3 attach_blocking.py [--delay 8] [--gpu 0 --peer-gpu 1]
"""

import argparse
import ctypes
import json
import multiprocessing
import os
import socket
import sys
import time

import _cuda as cu


def check_multicast_support(tag, dev):
    if not cu.has("cuMulticastCreate"):
        cu.log(tag, "driver lacks cuMulticastCreate entry point")
        sys.exit(1)
    if cu.device_attr(cu.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev) != 1:
        cu.log(tag, "CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED=0 -- no NVSwitch "
                    "multicast on this device (fabric manager down?)")
        sys.exit(1)


def make_local_mem(dev, size):
    """A shareable physical allocation to bind into the multicast group,
    sized to cover the whole group (NCCL binds shareable memory too)."""
    prop = cu.alloc_prop(dev)  # POSIX_FD handle type
    gran = cu.alloc_granularity(prop)
    size = (size + gran - 1) // gran * gran
    return cu.mem_create(size, prop), size


def bind_local(tag, mc, dev, size, what, timeout):
    """Timed cuMulticastBindMem of a fresh local allocation. Returns a dict
    {secs, ok, err} -- errors are captured, not raised (Q2 expects one)."""
    memh, msize = make_local_mem(dev, size)
    t0 = time.monotonic()
    err = None
    with cu.watchdog(what, timeout, tag):
        try:
            cu.call("cuMulticastBindMem", mc, 0, memh, 0, min(size, msize), 0)
        except cu.CudaError as e:
            err = e.name
    dt = time.monotonic() - t0
    cu.log(tag, f"{what}: {dt:.3f}s {'OK' if err is None else 'err=' + err}")
    if err is not None:
        cu.call("cuMemRelease", memh)
    return {"secs": round(dt, 3), "ok": err is None, "err": err}


def rank_a(sock, parent, args):
    tag = "[rankA]"
    dev = cu.init_device(args.gpu)
    check_multicast_support(tag, dev)

    prop = cu.CUmulticastObjectProp()
    prop.numDevices = 2
    prop.handleTypes = cu.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    gran = ctypes.c_size_t()
    cu.call("cuMulticastGetGranularity", ctypes.byref(gran), ctypes.byref(prop),
            cu.CU_MULTICAST_GRANULARITY_RECOMMENDED)
    size = (max(args.size, gran.value) + gran.value - 1) // gran.value * gran.value
    prop.size = size

    mc = ctypes.c_ulonglong()
    t_create, _ = cu.timed("cuMulticastCreate", lambda: cu.call(
        "cuMulticastCreate", ctypes.byref(mc), ctypes.byref(prop)), tag)
    fd = cu.export_posix_fd(mc.value)
    cu.log(tag, f"pid={os.getpid()} gpu={args.gpu} mc={mc.value:#x} "
                f"size={size:#x} exported fd={fd}")
    # Hand B the fd BEFORE we attach, so B's DELAY strictly precedes B's join.
    cu.send_msg(sock, f"MC size={size:#x}", fds=[fd])

    # Q1: does this return before B (still sleeping DELAY secs) joins?
    with cu.watchdog("cuMulticastAddDevice(rank A)", args.delay + 60, tag):
        t_add, _ = cu.timed(f"cuMulticastAddDevice(gpu{args.gpu})", lambda:
                            cu.call("cuMulticastAddDevice", mc.value, dev), tag)

    # Q2: bind BEFORE all devices joined -- fast error, success, or block?
    early = bind_local(tag, mc.value, dev, size,
                       "cuMulticastBindMem(EARLY, pre-all-join)",
                       args.delay + 60)

    cu.send_msg(parent, "ADDED")
    with cu.watchdog("rank A waiting for BIND barrier", 300, tag):
        cu.recv_msg(parent, expect="BIND")
    if early["ok"]:
        # The early bind already bound [0, size) for this device; binding the
        # same offset range again would be a double-bind (INVALID_VALUE).
        cu.log(tag, "cuMulticastBindMem(post-all-join): skipped "
                    "(early bind already bound this device)")
        final = {"secs": 0.0, "ok": True, "err": None, "skipped": True}
    else:
        final = bind_local(tag, mc.value, dev, size,
                           "cuMulticastBindMem(post-all-join)", 120)

    cu.send_msg(parent, json.dumps({"rank": "A", "create": t_create,
                                    "add": t_add, "early_bind": early,
                                    "bind": final}))
    with cu.watchdog("rank A waiting for EXIT", 300, tag):
        cu.recv_msg(parent, expect="EXIT")
    sys.exit(0)


def rank_b(sock, parent, args):
    tag = "[rankB]"
    dev = cu.init_device(args.peer_gpu)
    check_multicast_support(tag, dev)

    msg, fds = cu.recv_msg(sock, expect="MC")
    size = int(msg.split("size=")[1], 16)
    cu.log(tag, f"pid={os.getpid()} gpu={args.peer_gpu} got fd={fds[0]}; "
                f"sleeping {args.delay}s before joining (this is the window "
                f"in which rank A must not block)")
    time.sleep(args.delay)

    mc = ctypes.c_ulonglong()
    t_import, _ = cu.timed("cuMemImportFromShareableHandle", lambda: cu.call(
        "cuMemImportFromShareableHandle", ctypes.byref(mc),
        ctypes.c_void_p(fds[0]), cu.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR), tag)
    with cu.watchdog("cuMulticastAddDevice(rank B)", 120, tag):
        t_add, _ = cu.timed(f"cuMulticastAddDevice(gpu{args.peer_gpu})", lambda:
                            cu.call("cuMulticastAddDevice", mc.value, dev), tag)

    cu.send_msg(parent, "ADDED")
    with cu.watchdog("rank B waiting for BIND barrier", 300, tag):
        cu.recv_msg(parent, expect="BIND")
    final = bind_local(tag, mc.value, dev, size,
                       "cuMulticastBindMem(post-all-join)", 120)

    cu.send_msg(parent, json.dumps({"rank": "B", "import": t_import,
                                    "add": t_add, "bind": final}))
    with cu.watchdog("rank B waiting for EXIT", 300, tag):
        cu.recv_msg(parent, expect="EXIT")
    sys.exit(0)


def classify(secs, delay):
    if secs < min(1.0, delay * 0.2):
        return "non-blocking"
    if secs > delay * 0.8:
        return "BLOCKING(all-join)"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--peer-gpu", type=int, default=1)
    ap.add_argument("--delay", type=float, default=8.0,
                    help="seconds rank B lags behind rank A's attach")
    ap.add_argument("--size", type=lambda s: int(s, 0), default=32 << 20)
    args = ap.parse_args()

    mp = multiprocessing.get_context("fork")
    s_ab_a, s_ab_b = socket.socketpair()
    s_pa_p, s_pa_c = socket.socketpair()
    s_pb_p, s_pb_c = socket.socketpair()

    # fork inherits every socketpair end into every child; each process must
    # close the ends it doesn't own or peers never see EOF on crash.
    def run_a():
        cu.close_all(s_ab_b, s_pa_p, s_pb_p, s_pb_c)
        rank_a(s_ab_a, s_pa_c, args)

    def run_b():
        cu.close_all(s_ab_a, s_pa_p, s_pa_c, s_pb_p)
        rank_b(s_ab_b, s_pb_c, args)

    pa = mp.Process(target=run_a)
    pb = mp.Process(target=run_b)
    pa.start()
    pb.start()
    cu.close_all(s_ab_a, s_ab_b, s_pa_c, s_pb_c)

    def recv_from(sock, proc, name, expect=None, timeout=600):
        """recv with EOF->None on child death (e.g. watchdog exit 3)."""
        sock.settimeout(timeout)
        try:
            msg, _ = cu.recv_msg(sock, expect=expect)
            return msg
        except (EOFError, TimeoutError):
            proc.join(timeout=10)
            print(f"[parent] rank {name} died/stalled before reporting "
                  f"(exitcode={proc.exitcode})", flush=True)
            return None

    # Barrier: both ranks joined -> release binds.
    a_added = recv_from(s_pa_p, pa, "A", expect="ADDED")
    b_added = recv_from(s_pb_p, pb, "B", expect="ADDED")
    results = {}
    if a_added and b_added:
        cu.send_msg(s_pa_p, "BIND")
        cu.send_msg(s_pb_p, "BIND")
        for sock, proc, name in ((s_pa_p, pa, "A"), (s_pb_p, pb, "B")):
            msg = recv_from(sock, proc, name)
            results[name] = json.loads(msg) if msg else \
                {"died": True, "exitcode": proc.exitcode}
    else:
        results["A"] = {"died": not a_added, "exitcode": pa.exitcode}
        results["B"] = {"died": not b_added, "exitcode": pb.exitcode}

    for sock, proc in ((s_pa_p, pa), (s_pb_p, pb)):
        if proc.is_alive():
            try:
                cu.send_msg(sock, "EXIT")
            except OSError:
                pass
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()

    print("\n[parent] ==== SUMMARY (delay=%.1fs) ====" % args.delay, flush=True)
    print(json.dumps({"delay": args.delay, "results": results}, indent=2),
          flush=True)

    ra = results.get("A", {})
    if ra.get("died"):
        if ra.get("exitcode") == cu.WATCHDOG_EXIT_CODE:
            print("[parent] VERDICT Q1: rank A HUNG past the watchdog => "
                  "attach blocks indefinitely on all-join. Work item 4 "
                  "(batched attach) is REQUIRED, with timeouts.", flush=True)
            return 0
        return 1

    add_cls = classify(ra["add"], args.delay)
    print(f"[parent] VERDICT Q1 (ATTACH_GPU): rank A addDevice="
          f"{ra['add']:.3f}s [{add_cls}]", flush=True)
    if add_cls == "non-blocking":
        print("[parent]   => serial ATTACH_GPU replay across clients is safe; "
              "work item 4 stays a timeout-guarded fallback.", flush=True)
    elif "BLOCKING" in add_cls:
        print("[parent]   => ATTACH_GPU blocks until all ranks join: batched "
              "attach across clients (work item 4) is REQUIRED.", flush=True)
    else:
        print("[parent]   => inconclusive; re-run with a larger --delay.",
              flush=True)

    eb = ra.get("early_bind", {})
    if eb:
        eb_cls = classify(eb.get("secs", 0), args.delay)
        if eb.get("ok") and eb_cls == "non-blocking":
            kind = "succeeds immediately"
            note = "no ordering constraint between ATTACH_GPU and ATTACH_MEM."
        elif eb.get("ok"):
            kind = (f"BLOCKS until all-join ({eb['secs']:.3f}s ~ delay), "
                    "then succeeds")
            note = ("ATTACH_MEM inherits the all-join blocking semantics: a "
                    "serial replay that binds one client's memory before all "
                    "clients' ATTACH_GPU have replayed will stall (and "
                    "deadlock if the joins can never proceed). Replay must "
                    "complete ALL ATTACH_GPU across clients for a multicast "
                    "object before replaying any ATTACH_MEM -- work item 4 "
                    "applies to the ATTACH_GPU/ATTACH_MEM boundary, with "
                    "timeouts.")
        elif eb.get("secs", 0) < 1.0:
            kind = f"fails fast ({eb.get('err')})"
            note = ("replay must complete ALL ATTACH_GPU (all clients) for a "
                    "multicast object before replaying any ATTACH_MEM; "
                    "failure mode is at least a clean error, not a hang.")
        else:
            kind = f"slow error ({eb.get('secs')}s, {eb.get('err')})"
            note = ("bind stalls pre-all-join: batch binds after a global "
                    "attach barrier, with timeouts.")
        print(f"[parent] VERDICT Q2 (early ATTACH_MEM): {kind}", flush=True)
        print(f"[parent]   => {note}", flush=True)

    binds_ok = all(results.get(r, {}).get("bind", {}).get("ok")
                   for r in ("A", "B"))
    print(f"[parent] post-all-join binds: "
          f"{'both OK' if binds_ok else 'FAILED -- investigate'}", flush=True)
    return 0 if binds_ok else 1


if __name__ == "__main__":
    sys.exit(main())
