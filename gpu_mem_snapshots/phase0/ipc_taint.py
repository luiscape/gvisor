#!/usr/bin/env python3
"""Phase 0 measurement #1: IPC taint (TASK.md "Measure before implementing").

Question: does a cuMemCreate allocation with requestedHandleTypes =
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR become checkpointable again once
every export FD is closed and every peer import is released -- or is it
permanently IPC-tainted?

Native (no gVisor, no NCCL, no PyTorch). Two processes, one GPU each:

  exporter (GPU A)                      peer (GPU B)
  ----------------                      ------------
  cuMemCreate(POSIX_FD) + map + write
  cuMemExportToShareableHandle --fd-->  cuMemImportFromShareableHandle + map
                                        read + verify pattern
                                        unmap + cuMemRelease + close fd (taint)
  close export fd            <--done--       ... or keep it mapped (hold)
  [ cuda-checkpoint phased cycle on BOTH pids, mirroring gVisor's
    state_cuda.go: lock all -> checkpoint all -> restore all -> unlock all ]
  verify pattern                        verify (pattern in hold, liveness in taint)

Modes:
  --mode taint  the real measurement: peer releases + all FDs closed before
                the checkpoint. PASS => TASK.md work items 1-4 are sufficient.
                FAIL => unicast teardown/replay is also needed: STOP, escalate.
  --mode hold   sensitivity control: peer KEEPS its import mapped and FDs stay
                open. The cycle is expected to refuse/block somewhere (the
                premise this test measures the release of). If the full cycle
                passes even here, this driver does not refuse live
                POSIX-FD-IPC memory at all -- re-check TASK.md's premise on
                the target driver.

Exit codes: 0 = expected outcome for the mode, 1 = unexpected, 3 = watchdog.

Usage (see run_phase0.sh):
  sudo python3 ipc_taint.py --mode taint --cuda-checkpoint /path/to/cuda-checkpoint
"""

import argparse
import json
import multiprocessing
import os
import socket
import subprocess
import sys
import time

import _cuda as cu

CKPT_ACTION_TIMEOUT_S = 120
HOLD_CKPT_TIMEOUT_S = 30  # in hold mode, "blocked" is an expected outcome
PATTERN_CHUNKS = 16


def pattern_val(chunk):
    return (0xA5 << 24) | chunk


def verify_pattern(tag, va, size, expected_of_chunk):
    """Read back the chunked pattern; returns (errs, first_bad)."""
    n_u32 = size // 4
    chunk_u32 = n_u32 // PATTERN_CHUNKS
    errs, first_bad = 0, None
    for c in range(PATTERN_CHUNKS):
        got = cu.read_u32(va + c * chunk_u32 * 4, min(chunk_u32, 1024))
        want = expected_of_chunk(c)
        bad = sum(1 for g in got if g != want)
        if bad and first_bad is None:
            first_bad = (c, want, next(g for g in got if g != want))
        errs += bad
    return errs, first_bad


# ---------------------------------------------------------------------------
# exporter child
# ---------------------------------------------------------------------------

def exporter_proc(sock_peer, sock_parent, args):
    tag = "[exporter]"
    with cu.watchdog("exporter setup+export", 120, tag):
        dev = cu.init_device(args.gpu)
        prop = cu.alloc_prop(dev)
        gran = cu.alloc_granularity(prop)
        size = max(gran, args.size - args.size % gran or gran)
        handle = cu.mem_create(size, prop)
        va = cu.reserve_map_rw(handle, size, dev)
        cu.log(tag, f"pid={os.getpid()} gpu={args.gpu} size={size:#x} "
                    f"va={va:#x} handle={handle:#x}")

        # Write a chunked pattern (memset per chunk; no kernels needed).
        n_u32 = size // 4
        chunk_u32 = n_u32 // PATTERN_CHUNKS
        for c in range(PATTERN_CHUNKS):
            cu.memset_u32(va + c * chunk_u32 * 4, pattern_val(c), chunk_u32)
        cu.call("cuCtxSynchronize")

        export_fd = cu.export_posix_fd(handle)
        cu.log(tag, f"exported POSIX fd={export_fd}")
        cu.send_msg(sock_peer, f"IMPORT size={size:#x}", fds=[export_fd])

    if args.mode == "taint":
        with cu.watchdog("waiting for peer RELEASED", 120, tag):
            cu.recv_msg(sock_peer, expect="RELEASED")
        os.close(export_fd)
        cu.log(tag, "peer released its import; export fd closed. "
                    "No IPC handles remain open anywhere.")
    else:  # hold
        with cu.watchdog("waiting for peer HELD", 120, tag):
            cu.recv_msg(sock_peer, expect="HELD")
        cu.log(tag, f"peer HOLDS its import; export fd={export_fd} stays open.")

    cu.call("cuCtxSynchronize")
    cu.send_msg(sock_parent, f"READY pid={os.getpid()} va={va:#x} size={size:#x}")

    # Idle (no CUDA calls) while the parent drives cuda-checkpoint on us.
    with cu.watchdog("waiting for parent VERIFY", 600, tag):
        cu.recv_msg(sock_parent, expect="VERIFY")

    # First CUDA call after restore: read back and verify the pattern.
    with cu.watchdog("post-restore verification (cuMemcpyDtoH)", 120, tag):
        errs, first_bad = verify_pattern(tag, va, size, pattern_val)
    if errs == 0:
        cu.log(tag, f"VERIFY PASS: pattern intact at va={va:#x}")
    else:
        c, want, got = first_bad if first_bad is not None else (-1, 0, 0)
        cu.log(tag, f"VERIFY FAIL: {errs} bad words; first bad chunk "
                    f"{c}: want {want:#x} got {got:#x}")
    cu.send_msg(sock_parent, f"RESULT {'PASS' if errs == 0 else 'FAIL'}")
    cu.send_msg(sock_peer, "EXIT")
    sys.exit(0 if errs == 0 else 1)


# ---------------------------------------------------------------------------
# peer child
# ---------------------------------------------------------------------------

def peer_proc(sock_peer, sock_parent, args):
    tag = "[peer]"
    with cu.watchdog("peer import+verify", 120, tag):
        dev = cu.init_device(args.peer_gpu)
        msg, fds = cu.recv_msg(sock_peer, expect="IMPORT")
        size = int(msg.split("size=")[1], 16)
        fd = fds[0]
        handle = cu.import_posix_fd(fd)
        va = cu.reserve_map_rw(handle, size, dev)
        # Prove the import is real: read the exporter's pattern through it.
        got = cu.read_u32(va, 8)
        ok = all(g == pattern_val(0) for g in got)
        cu.log(tag, f"pid={os.getpid()} gpu={args.peer_gpu} imported fd={fd} "
                    f"va={va:#x} readback={'ok' if ok else 'MISMATCH ' + hex(got[0])}")
        if not ok:
            cu.send_msg(sock_parent, "RESULT FAIL import readback mismatch")
            sys.exit(1)

    if args.mode == "taint":
        # Full teardown: unmap, free VA reservation, release imported handle,
        # close the received fd. After this the peer holds NOTHING of the
        # exporter's allocation (it remains an otherwise-live CUDA process).
        with cu.watchdog("peer release", 120, tag):
            cu.call("cuMemUnmap", va, size)
            cu.call("cuMemAddressFree", va, size)
            cu.call("cuMemRelease", handle)
            os.close(fd)
            cu.call("cuCtxSynchronize")
        cu.log(tag, "released import (unmap + addressFree + memRelease + close fd)")
        cu.send_msg(sock_peer, "RELEASED")
    else:  # hold
        cu.log(tag, "HOLDING import mapped for the duration (sensitivity control)")
        cu.send_msg(sock_peer, "HELD")

    cu.send_msg(sock_parent, f"READY pid={os.getpid()}")

    # Idle while the parent drives cuda-checkpoint on us too.
    with cu.watchdog("peer waiting for parent VERIFY", 600, tag):
        cu.recv_msg(sock_parent, expect="VERIFY")
    with cu.watchdog("peer post-restore verification", 120, tag):
        if args.mode == "hold":
            # The import is still mapped: it must still read correctly.
            got = cu.read_u32(va, 8)
            ok = all(g == pattern_val(0) for g in got)
            cu.log(tag, f"VERIFY {'PASS' if ok else 'FAIL'}: import readback "
                        f"{'intact' if ok else hex(got[0])}")
        else:
            # Import fully released: liveness of the CUDA context is the test.
            try:
                cu.call("cuCtxSynchronize")
                ok = True
                cu.log(tag, "VERIFY PASS: context alive after restore")
            except cu.CudaError as e:
                ok = False
                cu.log(tag, f"VERIFY FAIL: context dead after restore: {e}")
    cu.send_msg(sock_parent, f"RESULT {'PASS' if ok else 'FAIL'}")

    with cu.watchdog("peer waiting for EXIT", 900, tag):
        cu.recv_msg(sock_peer, expect="EXIT")
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------
# parent: drives the phased cuda-checkpoint cycle on BOTH pids
# ---------------------------------------------------------------------------

def ckpt(cc, pid, action, timeout_s, extra=()):
    """Run one cuda-checkpoint action; returns dict(rc, out, secs, timeout)."""
    cmd = [cc, "--action", action, "--pid", str(pid), *extra]
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        rc, out, timed_out = p.returncode, (p.stdout + p.stderr).strip(), False
    except subprocess.TimeoutExpired as e:
        rc, timed_out = -1, True
        out = ((e.stdout or b"").decode(errors="replace") +
               (e.stderr or b"").decode(errors="replace")).strip()
    secs = time.monotonic() - t0
    print(f"[parent] cuda-checkpoint --action {action} --pid {pid}: rc={rc} "
          f"{'TIMEOUT ' if timed_out else ''}({secs:.1f}s) out=[{out}]", flush=True)
    return {"action": action, "pid": pid, "rc": rc, "out": out,
            "secs": round(secs, 2), "timeout": timed_out}


def phase(cc, pids, action, timeout_s, extra=()):
    """Run one action on every pid (gVisor's phased flow: complete each phase
    across all processes before starting the next). Returns (results, all_ok)."""
    results = [ckpt(cc, pid, action, timeout_s, extra) for pid in pids]
    return results, all(r["rc"] == 0 for r in results)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("taint", "hold"), default="taint")
    ap.add_argument("--gpu", type=int, default=0, help="exporter GPU ordinal")
    ap.add_argument("--peer-gpu", type=int, default=1, help="peer GPU ordinal")
    ap.add_argument("--size", type=lambda s: int(s, 0), default=64 << 20,
                    help="allocation size (rounded to granularity)")
    ap.add_argument("--cuda-checkpoint", default=os.environ.get("CUDA_CHECKPOINT"),
                    required=os.environ.get("CUDA_CHECKPOINT") is None,
                    help="path to the cuda-checkpoint binary")
    args = ap.parse_args()
    cc = args.cuda_checkpoint

    mp = multiprocessing.get_context("fork")  # children init CUDA post-fork
    s_peer_a, s_peer_b = socket.socketpair()
    s_par_exp_a, s_par_exp_b = socket.socketpair()
    s_par_peer_a, s_par_peer_b = socket.socketpair()

    # fork inherits every socketpair end into every child; each process must
    # close the ends it doesn't own or peers never see EOF on crash.
    def run_exporter():
        cu.close_all(s_peer_b, s_par_exp_a, s_par_peer_a, s_par_peer_b)
        exporter_proc(s_peer_a, s_par_exp_b, args)

    def run_peer():
        cu.close_all(s_peer_a, s_par_exp_a, s_par_exp_b, s_par_peer_a)
        peer_proc(s_peer_b, s_par_peer_b, args)

    exp = mp.Process(target=run_exporter)
    peer = mp.Process(target=run_peer)
    exp.start()
    peer.start()
    cu.close_all(s_peer_a, s_peer_b, s_par_exp_b, s_par_peer_b)

    def ready_pid(sock, proc, name):
        try:
            msg, _ = cu.recv_msg(sock, expect="READY")
        except EOFError:
            proc.join(timeout=10)
            print(f"[parent] {name} died during setup "
                  f"(exitcode={proc.exitcode}); see its log above", flush=True)
            return None
        print(f"[parent] {name} ready: {msg}", flush=True)
        return int(msg.split("pid=")[1].split()[0])

    exp_pid = ready_pid(s_par_exp_a, exp, "exporter")
    peer_pid = ready_pid(s_par_peer_a, peer, "peer")
    if exp_pid is None or peer_pid is None:
        for p in (exp, peer):
            if p.is_alive():
                p.terminate()
        return 1
    pids = [exp_pid, peer_pid]  # exporter first, like gVisor's process walk

    results = []
    r, lock_ok = phase(cc, pids, "lock", CKPT_ACTION_TIMEOUT_S,
                       ("--timeout", "30000"))
    results += r

    ckpt_ok = False
    if lock_ok:
        ckpt_timeout = (HOLD_CKPT_TIMEOUT_S if args.mode == "hold"
                        else CKPT_ACTION_TIMEOUT_S)
        r, ckpt_ok = phase(cc, pids, "checkpoint", ckpt_timeout)
        results += r

    restore_ok = False
    if ckpt_ok:
        r, restore_ok = phase(cc, pids, "restore", CKPT_ACTION_TIMEOUT_S)
        results += r
    if lock_ok:
        # Unlock whatever can be unlocked, even after a failed phase.
        r, _ = phase(cc, pids, "unlock", CKPT_ACTION_TIMEOUT_S)
        results += r

    # Ask both children to verify.
    def collect_result(sock, proc, name):
        try:
            cu.send_msg(sock, "VERIFY")
            msg, _ = cu.recv_msg(sock, expect="RESULT")
            return msg.split()[1] == "PASS"
        except (EOFError, OSError) as e:
            print(f"[parent] {name} died before verifying: {e} "
                  f"(a dead process after restore is itself a FAIL datum)",
                  flush=True)
            return False

    exp_pass = collect_result(s_par_exp_a, exp, "exporter")
    peer_pass = collect_result(s_par_peer_a, peer, "peer")
    verify_pass = exp_pass and peer_pass

    exp.join(timeout=60)
    peer.join(timeout=60)
    for p in (exp, peer):
        if p.is_alive():
            p.terminate()

    # ------------------------------------------------------------------
    # verdict
    # ------------------------------------------------------------------
    cycle_ok = lock_ok and ckpt_ok and restore_ok
    print("\n[parent] ==== SUMMARY (mode=%s) ====" % args.mode, flush=True)
    print(json.dumps({"mode": args.mode, "actions": results,
                      "exporter_verify": exp_pass, "peer_verify": peer_pass},
                     indent=2), flush=True)

    if args.mode == "hold":
        if not cycle_ok:
            print("[parent] VERDICT: cycle BLOCKED/refused while import held "
                  "(expected) -- harness is sensitive to live IPC. "
                  "Proceed to --mode taint.", flush=True)
            return 0
        print("[parent] VERDICT: ANOMALY -- full lock/checkpoint/restore/"
              "unlock cycle SUCCEEDED on both pids with a live peer import"
              f" (verify: exporter={exp_pass} peer={peer_pass}).\n"
              "  This driver+cuda-checkpoint does not refuse live POSIX-FD "
              "IPC memory; TASK.md's premise does not reproduce here. "
              "Re-check on the target (R610) driver before trusting the "
              "taint leg as a discriminator.", flush=True)
        return 1

    # taint mode
    if cycle_ok and verify_pass:
        print("[parent] VERDICT: NOT TAINTED -- allocation is checkpointable "
              "again after all exports/imports are released.\n"
              "  => TASK.md work items 1-4 are sufficient; unicast "
              "allocations stay resident.", flush=True)
        return 0
    print("[parent] VERDICT: IPC-TAINTED (or restore/verify failed) -- "
          f"lock_ok={lock_ok} checkpoint_ok={ckpt_ok} restore_ok={restore_ok} "
          f"exporter_verify={exp_pass} peer_verify={peer_pass}.\n"
          "  => STOP AND ESCALATE per TASK.md: nvproxy would also need to "
          "tear down/replay unicast allocations (materially larger design).",
          flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
