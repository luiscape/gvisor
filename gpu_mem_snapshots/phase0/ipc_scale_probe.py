#!/usr/bin/env python3
"""Which CUDA VMM/IPC step makes cuda-checkpoint refuse to restore?

Every checkpoint mechanism in this tree is capped by the same failure: after a
checkpoint, `cuda-checkpoint --action restore` (or --toggle) returns
"invalid argument" / "unknown error". It is not multicast (bisected), not
gVisor (reproduces under runc), and not fixable by reordering the calls.

This probe isolates it with nothing else in the picture -- W processes, one GPU
each, no NCCL, no PyTorch, no multicast, no engine, no gVisor -- and walks the
CUDA VMM sharing sequence one step at a time (--stage), so the exact step that
arms the failure is named rather than inferred:

    plain     cuMemCreate with NO exportable handle type, mapped.
    alloc     cuMemCreate with CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
              mapped, never exported.
    export    + cuMemExportToShareableHandle (this process holds the FD).
    share     + the FD is passed to every peer, which holds it WITHOUT
              importing (isolates FD possession from CUDA-level import).
    import    + every peer cuMemImportFromShareableHandle + maps it.
    teardown  = import, then every mapping, handle and FD released everywhere
              before the checkpoint (answers TASK.md's "is an allocation
              permanently IPC-tainted?").

Each stage is a superset of the one above, so the first failing stage is the
culprit. Sweep the whole ladder:

    python3 ipc_scale_probe.py --world 2 --stage all --trials 3

Then drive the same phased sequence gVisor uses (control/state_cuda.go): lock
every process in parallel, checkpoint, restore, unlock.

Exit code 0 only if every trial round-trips.
"""

import argparse
import multiprocessing
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cuda as cu

BUF_BYTES = 2 << 20  # one granule; size is not the variable under test

# Each stage is a strict superset of the one before it, so the first stage that
# fails names the exact step that arms the failure.
STAGES = {
    "plain": "VMM alloc, not exportable",
    "alloc": "exportable handle type, never exported",
    "export": "cuMemExportToShareableHandle, FD held locally",
    "share": "FD passed to peers, not imported by them",
    "nomap": "peers cuMemImportFromShareableHandle, never mapped",
    "import": "peers cuMemImportFromShareableHandle + map",
    "unmap": "imported+mapped, then unmapped but handle NOT released",
    "release": "imported, then cuMemRelease'd; export FDs still open",
    "teardown": "imported, then fully released everywhere",
}

# The legacy (pre-VMM) CUDA IPC ladder. cuIpcOpenMemHandle is a completely
# separate API from the VMM one above, and is what vLLM's custom all-reduce
# and torch's cudaIpc paths use, so it needs its own bisect.
LEGACY_STAGES = {
    "legacy-alloc": "cuMemAlloc, never shared",
    "legacy-export": "cuIpcGetMemHandle, handle kept local",
    "legacy-import": "peers cuIpcOpenMemHandle",
    "legacy-close": "peers open then cuIpcCloseMemHandle before checkpoint",
    "legacy-free": "peers close AND the exporter cuMemFrees the allocation",
}
ALL_STAGES = {**STAGES, **LEGACY_STAGES}

# Stages at which a rank hands its export FDs to its peers.
SHARING_STAGES = ("share", "nomap", "import", "unmap", "release", "teardown")

# Stages that release what they imported before the checkpoint, and so need a
# barrier ensuring no peer is still holding an import when it starts.
DRAINING_STAGES = ("unmap", "release", "teardown", "legacy-close",
                   "legacy-free")


class Chan:
    """Line-framed messages, with optional SCM_RIGHTS, over a socketpair.

    A socketpair is a *stream*, so a single recv can return several messages
    at once. Reading one message per recv and discarding the remainder loses
    whatever followed it, which shows up as a deadlock in whichever peer waits
    for the swallowed message next. (That is a race, so it hides at small
    world sizes and appears at large ones.) Buffer the remainder instead.

    FDs are only ever attached to a message that its receiver is waiting on
    with an empty buffer, so attributing every FD in a read to the first
    message in that read is unambiguous here.
    """

    def __init__(self, sock):
        self.sock = sock
        self.buf = b""

    def send(self, msg, fds=()):
        socket.send_fds(self.sock, [msg.encode() + b"\n"], list(fds))

    def recv(self, expect=None, maxfds=64):
        fds = []
        while b"\n" not in self.buf:
            data, f, _, _ = socket.recv_fds(self.sock, 8192, maxfds)
            fds += f
            if not data:
                raise EOFError("peer closed the socket (it probably died)")
            self.buf += data
        line, self.buf = self.buf.split(b"\n", 1)
        msg = line.decode().strip()
        if expect is not None and not msg.startswith(expect):
            raise RuntimeError(f"expected {expect!r}, got {msg!r}")
        return msg, fds

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def legacy_rank_proc(rank, chan, args, tag, v):
    """The legacy cuIpc* ladder: cuMemAlloc + cuIpcGetMemHandle/OpenMemHandle.

    Legacy IPC handles are opaque 64-byte blobs, not OS handles, so they travel
    as plain bytes; no SCM_RIGHTS is involved anywhere in this path.
    """
    stage = args.stage
    with cu.watchdog("context init", 120, tag):
        dev = cu.init_device(rank % args.gpus)

    my_ptrs = []
    with cu.watchdog("legacy alloc+export", 120, tag):
        for _ in range(args.allocs):
            p = cu.mem_alloc(BUF_BYTES)
            cu.memset_u32(p, 0xB5000000 | rank, 16)
            my_ptrs.append(p)
        blobs = ([cu.ipc_get_handle(p) for p in my_ptrs]
                 if stage != "legacy-alloc" else [])
    v(f"{len(my_ptrs)} legacy buffers, {len(blobs)} IPC handles")

    # Only publish from legacy-import onward; legacy-export keeps them local.
    publish = blobs if stage in ("legacy-import", "legacy-close",
                                 "legacy-free") else []
    chan.send("FDS rank=%d blobs=%s" % (
        rank, b"".join(publish).hex() if publish else ""))

    opened = []
    while True:
        with cu.watchdog("awaiting peer IPC handles", 180, tag):
            msg, _ = chan.recv()
        if msg.startswith("GO"):
            break
        hexed = msg.split("blobs=", 1)[1]
        raw = bytes.fromhex(hexed)
        with cu.watchdog("cuIpcOpenMemHandle", 180, tag):
            for i in range(0, len(raw), cu.CU_IPC_HANDLE_SIZE):
                opened.append(cu.ipc_open_handle(
                    raw[i:i + cu.CU_IPC_HANDLE_SIZE]))
    v(f"opened {len(opened)} peer IPC handles")

    if stage in ("legacy-close", "legacy-free"):
        with cu.watchdog("cuIpcCloseMemHandle", 120, tag):
            for p in opened:
                cu.ipc_close_handle(p)
            opened = []
        if stage == "legacy-free":
            # Destroy the exported allocation itself. An application cannot
            # normally do this (it is its data), but it decides whether the
            # taint lives on the allocation or on the process.
            for p in my_ptrs:
                cu.call("cuMemFree_v2", p)
            my_ptrs = []
        chan.send("DRAINED")
        chan.recv(expect="GO")

    cu.call("cuCtxSynchronize")
    chan.send(f"READY pid={os.getpid()} imports={len(opened)} held_fds=0")

    chan.recv(expect="VERIFY")
    cu.call("cuCtxSynchronize")
    for i, p in enumerate(my_ptrs):
        got = cu.read_u32(p, 1)[0]
        want = 0xB5000000 | rank
        if got != want:
            raise RuntimeError(f"buffer {i} content changed: "
                               f"0x{got:08x} != 0x{want:08x}")
    chan.send("RESULT PASS")
    chan.recv(expect="EXIT")


def rank_proc(rank, chan, args):
    """Walk the sharing ladder up to args.stage, then idle for the checkpoint."""
    tag = f"[rank{rank}]"
    v = (lambda m: cu.log(tag, m)) if args.verbose else (lambda m: None)
    try:
        stage = args.stage
        if stage in LEGACY_STAGES:
            legacy_rank_proc(rank, chan, args, tag, v)
            os._exit(0)
        with cu.watchdog("context init", 120, tag):
            dev = cu.init_device(rank % args.gpus)
        # "plain" allocates ordinary VMM memory that cannot be shared at all;
        # every other stage requests an exportable handle type.
        prop = cu.alloc_prop(dev, handle_types=(
            0 if stage == "plain" else cu.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR))
        gran = cu.alloc_granularity(prop)
        size = max(BUF_BYTES, gran)

        my_fds, my_vas, my_handles = [], [], []
        with cu.watchdog("local alloc+export", 120, tag):
            for _ in range(args.allocs):
                h = cu.mem_create(size, prop)
                my_handles.append(h)
                my_vas.append(cu.reserve_map_rw(h, size, dev))
                if stage not in ("plain", "alloc"):
                    my_fds.append(cu.export_posix_fd(h))
            # Own each buffer with a known pattern, so a silently broken
            # restore is caught rather than assumed away.
            for va in my_vas:
                cu.memset_u32(va, 0xA5000000 | rank, 16)
        v(f"{len(my_vas)} buffers of {size:#x}, {len(my_fds)} exported")

        # Publish ours, then take the peers' descriptors from the parent.
        # Before the "share" stage the FDs stay local -- publishing them would
        # make peers import, which is the very step being isolated.
        publish = my_fds if stage in SHARING_STAGES else []
        chan.send(f"FDS rank={rank}", fds=publish)
        if args.close_exports:
            for fd in my_fds:
                os.close(fd)
            my_fds = []
        imported, held_fds, peer_maps = 0, [], []
        while True:
            with cu.watchdog(f"awaiting peer FDs (imported {imported})", 180, tag):
                msg, fds = chan.recv()
            if msg.startswith("GO"):
                break
            with cu.watchdog(f"importing {len(fds)} peer FDs", 180, tag):
                for fd in fds:
                    if stage == "share":
                        # Hold the FD, but never tell CUDA about it.
                        held_fds.append(fd)
                        continue
                    h = cu.import_posix_fd(fd)
                    if stage == "nomap":
                        peer_maps.append((h, None))
                    else:
                        peer_maps.append((h, cu.reserve_map_rw(h, size, dev)))
                    os.close(fd)
                    imported += 1
            v(f"imported {imported}")

        if stage in DRAINING_STAGES:
            # Give back what was imported and only then checkpoint. "unmap"
            # gives back only the mapping; "teardown" gives back everything,
            # everywhere. If even teardown fails, an allocation that was ever
            # exportable is permanently tainted and no amount of well-behaved
            # teardown by the application can make it checkpointable.
            with cu.watchdog(stage, 120, tag):
                for h, va in peer_maps:
                    if va is not None:
                        cu.call("cuMemUnmap", va, size)
                        cu.call("cuMemAddressFree", va, size)
                    if stage != "unmap":
                        cu.call("cuMemRelease", h)
                if stage != "unmap":
                    peer_maps = []
                    imported = 0
                if stage == "teardown":
                    for fd in my_fds:
                        os.close(fd)
                    my_fds = []
            # Everyone must finish releasing before anyone reports READY,
            # or a peer's import can still be live at checkpoint time.
            chan.send("DRAINED")
            chan.recv(expect="GO")
            v("teardown complete; no IPC handles live anywhere")

        cu.call("cuCtxSynchronize")
        chan.send(f"READY pid={os.getpid()} imports={imported} "
                  f"held_fds={len(held_fds)}")

        # Idle with the imports live while the parent drives cuda-checkpoint.
        chan.recv(expect="VERIFY")
        cu.call("cuCtxSynchronize")  # first GPU work after restore
        for i, va in enumerate(my_vas):
            got = cu.read_u32(va, 1)[0]
            want = 0xA5000000 | rank
            if got != want:
                raise RuntimeError(f"buffer {i} content changed: "
                                   f"0x{got:08x} != 0x{want:08x}")
        chan.send("RESULT PASS")
        chan.recv(expect="EXIT")
    except Exception as e:  # noqa: BLE001 - report, don't crash silently
        cu.log(tag, f"FAILED: {type(e).__name__}: {e}")
        try:
            chan.send(f"RESULT FAIL {e}")
        except OSError:
            pass
        os._exit(1)
    os._exit(0)


VERBOSE = False


def ckpt(cc, pid, action, timeout_s=120, extra=()):
    cmd = [cc, "--action", action, "--pid", str(pid), *extra]
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        rc, out, to = p.returncode, (p.stdout + p.stderr).strip(), False
    except subprocess.TimeoutExpired as e:
        rc, to = -1, True
        out = ((e.stdout or b"").decode(errors="replace")
               + (e.stderr or b"").decode(errors="replace")).strip()
    r = {"pid": pid, "action": action, "rc": rc, "out": out,
         "secs": round(time.monotonic() - t0, 2), "timeout": to}
    if VERBOSE:
        print(f"    {action:10s} pid={pid} rc={rc} {r['secs']}s"
              + (" TIMEOUT" if to else "") + (f" {out!r}" if out else ""),
              flush=True)
    return r


def parallel(fn, items):
    """gVisor issues each phase across all processes at once, not serially."""
    with ThreadPoolExecutor(max_workers=max(len(items), 1)) as ex:
        return list(ex.map(fn, items))


def one_trial(args, trial):
    mp = multiprocessing.get_context("fork")
    parent_socks, child_socks, procs = [], [], []
    for rank in range(args.world):
        a, b = socket.socketpair()
        parent_socks.append(a)
        child_socks.append(b)
    for rank in range(args.world):
        def run(rank=rank):
            cu.close_all(*parent_socks,
                         *[s for i, s in enumerate(child_socks) if i != rank])
            rank_proc(rank, Chan(child_socks[rank]), args)
        p = mp.Process(target=run)
        p.start()
        procs.append(p)
    cu.close_all(*child_socks)
    chans = [Chan(s) for s in parent_socks]

    # Collect what every rank published, then fan it out to the others. VMM
    # sharing passes OS file descriptors (SCM_RIGHTS); legacy CUDA IPC passes
    # opaque 64-byte blobs, which are just bytes.
    legacy = args.stage in LEGACY_STAGES
    by_rank = {}
    for rank in range(args.world):
        msg, fds = chans[rank].recv(expect="FDS")
        by_rank[rank] = msg.split("blobs=", 1)[1] if legacy else fds
    for rank in range(args.world):
        if legacy:
            peers = "".join(b for r, b in by_rank.items() if r != rank)
            if peers:
                chans[rank].send(f"PEERS blobs={peers}")
        else:
            peers = [fd for r, fds in by_rank.items() if r != rank for fd in fds]
            for i in range(0, len(peers), 16):
                chans[rank].send("PEERS", fds=peers[i:i + 16])
        chans[rank].send("GO")
    if not legacy:
        for fds in by_rank.values():
            for fd in fds:
                os.close(fd)

    # Barrier: no rank reports READY until every rank has finished releasing,
    # so the checkpoint cannot race a peer that still holds an import.
    if args.stage in DRAINING_STAGES:
        for rank in range(args.world):
            chans[rank].recv(expect="DRAINED")
        for rank in range(args.world):
            chans[rank].send("GO")

    pids, imports = [], 0
    for rank in range(args.world):
        msg, _ = chans[rank].recv(expect="READY")
        pids.append(int(msg.split("pid=")[1].split()[0]))
        imports = int(msg.split("imports=")[1].split()[0])

    # Same phased sequence gVisor uses (control/state_cuda.go): lock every
    # process in parallel so coupled ranks quiesce together, then checkpoint,
    # then restore. gVisor restores serially under --cuda-checkpoint-sequential.
    phases = {}

    def act(action, targets, **kw):
        fn = lambda p: ckpt(args.cuda_checkpoint, p, action, **kw)
        return parallel(fn, targets) if args.parallel_restore or action != "restore" \
            else [fn(p) for p in targets]

    lock = parallel(
        lambda p: ckpt(args.cuda_checkpoint, p, "lock", extra=("--timeout", "30000")),
        pids)
    phases["lock"] = lock
    ok = all(r["rc"] == 0 for r in lock)

    if ok:
        chk = parallel(lambda p: ckpt(args.cuda_checkpoint, p, "checkpoint"), pids)
        phases["checkpoint"] = chk
        ok = all(r["rc"] == 0 for r in chk)

    restore = []
    if ok:
        restore = act("restore", pids)
        phases["restore"] = restore
        ok = all(r["rc"] == 0 for r in restore)

    if any(r["rc"] == 0 for r in lock):
        phases["unlock"] = parallel(
            lambda p: ckpt(args.cuda_checkpoint, p, "unlock"), pids)

    # A rank whose CUDA context did not come back can block indefinitely on
    # its first post-restore call, so never wait on it without a deadline.
    verified = 0
    for rank in range(args.world):
        try:
            chans[rank].send("VERIFY")
            chans[rank].sock.settimeout(60)
            msg, _ = chans[rank].recv(expect="RESULT")
            verified += 1 if msg.split()[1] == "PASS" else 0
        except (EOFError, OSError, RuntimeError, socket.timeout):
            pass
    for rank in range(args.world):
        try:
            chans[rank].send("EXIT")
        except OSError:
            pass
    for p in procs:
        p.join(timeout=15)
        if p.is_alive():
            p.kill()
            p.join(timeout=15)
    for c in chans:
        c.close()

    # Which restore calls failed, and how far in?
    failed_at = None
    n_failed = 0
    if restore:
        for i, r in enumerate(restore):
            if r["rc"] != 0:
                n_failed += 1
                if failed_at is None:
                    failed_at = i
    errs = sorted({r["out"] for r in restore if r["rc"] != 0})
    if any(r["rc"] != 0 for r in phases.get("checkpoint", [])):
        errs += sorted({"checkpoint: " + r["out"]
                        for r in phases["checkpoint"] if r["rc"] != 0})
    if any(r["rc"] != 0 for r in lock):
        errs += sorted({"lock: " + r["out"] for r in lock if r["rc"] != 0})
    print(f"  trial {trial}: imports/proc={imports} verified={verified}/{args.world} "
          f"lock={'ok' if all(r['rc']==0 for r in lock) else 'FAIL'} "
          f"checkpoint={'ok' if phases.get('checkpoint') and all(r['rc']==0 for r in phases['checkpoint']) else 'FAIL/skip'} "
          f"restore_failed={n_failed}/{len(restore) if restore else 0}"
          + (f" first_fail_index={failed_at}" if failed_at is not None else "")
          + (f" errs={errs}" if errs else ""), flush=True)
    return {"imports": imports, "ok": ok and verified == args.world,
            "restore_failed": n_failed, "first_fail_index": failed_at,
            "errs": errs}


def run_stage(args, stage):
    args.stage = stage
    print(f"\n=== stage={stage} world={args.world} allocs/rank={args.allocs} "
          f"trials={args.trials} ({ALL_STAGES[stage]})", flush=True)
    results = [one_trial(args, t + 1) for t in range(args.trials)]
    passed = sum(1 for r in results if r["ok"])
    errs = sorted({e for r in results for e in r["errs"]})
    print(f"RESULT stage={stage}: {passed}/{len(results)} pass", flush=True)
    return {"stage": stage, "passed": passed, "total": len(results),
            "errs": errs}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--gpus", type=int, default=None,
                    help="distinct GPUs to spread ranks over (default: --world)")
    ap.add_argument("--allocs", type=int, default=1,
                    help="buffers per rank; at stage=import each rank ends up "
                         "with allocs*(world-1) live imports")
    ap.add_argument("--stage", default="all",
                    choices=["all", "legacy", "everything", *ALL_STAGES],
                    help="how far along the sharing sequence to go; "
                         "'all' = the VMM ladder, 'legacy' = the cuIpc* "
                         "ladder, 'everything' = both")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--close-exports", action="store_true",
                    help="close local export FDs once published, leaving only "
                         "the peers' live imports (isolates import from export)")
    ap.add_argument("--parallel-restore", action="store_true",
                    help="restore all at once instead of serially")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--cuda-checkpoint", default=os.environ.get(
        "CUDA_CHECKPOINT", "/usr/local/bin/cuda-checkpoint"))
    args = ap.parse_args()
    if args.gpus is None:
        args.gpus = args.world
    global VERBOSE
    VERBOSE = args.verbose

    stages = {"all": list(STAGES), "legacy": list(LEGACY_STAGES),
              "everything": list(ALL_STAGES)}.get(args.stage, [args.stage])
    summary = [run_stage(args, s) for s in stages]

    print("\n" + "=" * 72)
    print(f"SUMMARY world={args.world} allocs/rank={args.allocs}")
    for s in summary:
        verdict = "pass" if s["passed"] == s["total"] else "FAIL"
        print(f"  {s['stage']:9s} {s['passed']}/{s['total']} {verdict}")
        for e in s["errs"]:
            print(f"            {e}")
    first_bad = next((s["stage"] for s in summary if s["passed"] != s["total"]),
                     None)
    if first_bad:
        print(f"\nFirst failing stage: {first_bad} -- {ALL_STAGES[first_bad]}")
    return 0 if first_bad is None else 1


if __name__ == "__main__":
    sys.exit(main())
