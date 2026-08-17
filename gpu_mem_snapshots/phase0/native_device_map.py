#!/usr/bin/env python3
"""Native (no gVisor) probe: can a cuda-checkpoint-restored process be moved to
a DIFFERENT physical GPU with --device-map, and can it then import peer memory?

This is the cross-GPU question. Restoring a checkpoint onto a different GPU set
(e.g. a container that had GPUs 0,1 landing on 6,7) works on R610 but fails on
pre-R610, where the interposer's rebuild re-imports return
CUDA_ERROR_INVALID_DEVICE. R580's cuda-checkpoint does accept --device-map,
which nothing in our stack passes yet (see the FIXME in
pkg/sentry/control/state_cuda.go), so the open question is whether it repairs
the restored process's device identity well enough for the rebuild to proceed.

  child:  retain a primary context on --src-ordinal, make a VMM allocation,
          fill it with a known pattern, write READY (pid + device UUID), idle.
          On RESUME: re-read the pattern, write a fresh pattern (proving the
          context still executes), and -- in the "import" phase -- import a
          shareable allocation from a never-checkpointed peer on the NEW GPU.
  parent: lock -> checkpoint -> restore [--device-map src=dst] -> unlock,
          then RESUME and report. Also reports where the driver actually thinks
          the process lives, via nvidia-smi's compute-apps GPU UUID.

Phases (--phase):
  plain   pattern survival + context usable after a device-mapped restore
  import  additionally import a peer's POSIX-FD allocation on the new GPU
          (the operation that fails cross-GPU on pre-R610)

A/B controls:
  --no-map  restore WITHOUT --device-map (isolates the flag from the move)
  --no-cr   skip the checkpoint cycle entirely (baseline sanity)

Usage:
  sudo python3 native_device_map.py [--phase plain|import]
      [--src-ordinal 0] [--dst-index 6] [--no-map] [--no-cr]
"""

import argparse
import ctypes
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cuda as cu

READY = "/tmp/native_devmap.ready"
RESUME = "/tmp/native_devmap.resume"
RESULT = "/tmp/native_devmap.result"
STOP = "/tmp/native_devmap.stop"

PATTERN_A = 0xA5A5A5A5
PATTERN_B = 0x5A5A5A5A
NWORDS = 1024


def _step(results, name, fn):
    try:
        out = fn()
        results.append(f"{name}=OK")
        return out
    except cu.CudaError as e:
        results.append(f"{name}=CUresult({e})")
        raise


def smi_uuids():
    """[(index, uuid)] as the driver reports them, in nvidia-smi order."""
    p = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid",
                        "--format=csv,noheader"],
                       capture_output=True, text=True, check=True)
    out = []
    for line in p.stdout.strip().splitlines():
        idx, uuid = (f.strip() for f in line.split(","))
        out.append((int(idx), uuid))
    return out


def smi_placement(pid):
    """The GPU UUID(s) the driver currently attributes to pid, or []."""
    p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                        "--format=csv,noheader"],
                       capture_output=True, text=True)
    found = []
    for line in p.stdout.strip().splitlines():
        if not line.strip():
            continue
        fields = [f.strip() for f in line.split(",")]
        if len(fields) >= 2 and fields[0] == str(pid):
            found.append(fields[1])
    return found


# ---------------------------------------------------------------------------
# peer: a never-checkpointed process that exports an allocation on one GPU
# ---------------------------------------------------------------------------

def peer(ordinal, sock_fd):
    import socket
    sock = socket.socket(fileno=sock_fd)
    dev = cu.init_device(ordinal)
    prop = cu.alloc_prop(dev)
    size = cu.alloc_granularity(prop)
    handle = cu.mem_create(size, prop)
    va = cu.reserve_map_rw(handle, size, dev)
    cu.memset_u32(va, PATTERN_B, NWORDS)
    fd = cu.export_posix_fd(handle)
    cu.send_msg(sock, f"FD {size}", fds=(fd,))
    # Hold the export open until told to go: closing it early would free the
    # object out from under the importer.
    cu.recv_msg(sock)
    return 0


# ---------------------------------------------------------------------------
# child: the process that gets checkpointed and moved
# ---------------------------------------------------------------------------

def child():
    phase = os.environ.get("DEVMAP_PHASE", "plain")
    src_ordinal = int(os.environ.get("DEVMAP_SRC_ORDINAL", "0"))
    for f in (READY, RESULT):
        if os.path.exists(f):
            os.remove(f)

    dev = cu.init_device(src_ordinal)
    uuid_before = cu.device_uuid(dev)
    prop = cu.alloc_prop(dev)
    size = cu.alloc_granularity(prop)
    handle = cu.mem_create(size, prop)
    va = cu.reserve_map_rw(handle, size, dev)
    cu.memset_u32(va, PATTERN_A, NWORDS)
    if cu.read_u32(va, 4) != [PATTERN_A] * 4:
        print("[child] pre-checkpoint pattern readback failed", flush=True)
        return 1
    with open(READY, "w") as f:
        f.write(f"{os.getpid()} {uuid_before}\n")
    print(f"[child] ready pid={os.getpid()} ordinal={src_ordinal} "
          f"uuid={uuid_before} va=0x{va:x}", flush=True)

    while not os.path.exists(RESUME):
        time.sleep(0.2)
    dst_ordinal = int(open(RESUME).read().strip() or src_ordinal)
    print(f"[child] RESUME seen (dst_ordinal={dst_ordinal})", flush=True)

    results = [f"phase={phase}", f"uuid_before={uuid_before}",
               f"va=0x{va:x}"]
    try:
        # What does the process think its device is now?
        try:
            results.append(f"uuid_after={cu.device_uuid(dev)}")
        except cu.CudaError as e:
            results.append(f"uuid_after=CUresult({e})")

        _step(results, "ctx_sync", lambda: cu.call("cuCtxSynchronize"))
        got = _step(results, "pattern_read", lambda: cu.read_u32(va, 4))
        results.append("pattern_intact="
                       + str(got == [PATTERN_A] * 4))
        # A fresh write proves the context still executes work, not just that
        # the mapping is readable.
        _step(results, "memset_new",
              lambda: cu.memset_u32(va, PATTERN_B, NWORDS))
        got2 = _step(results, "reread", lambda: cu.read_u32(va, 4))
        results.append("write_works=" + str(got2 == [PATTERN_B] * 4))

        if phase == "import":
            import socket
            parent_sock, child_sock = socket.socketpair()
            argv = [sys.executable, os.path.abspath(__file__), "--peer",
                    "--src-ordinal", str(dst_ordinal)]
            pr = subprocess.Popen(argv, pass_fds=(child_sock.fileno(),),
                                  env=dict(os.environ,
                                           DEVMAP_PEER_FD=str(
                                               child_sock.fileno())))
            child_sock.close()
            msg, fds = cu.recv_msg(parent_sock, expect="FD")
            psize = int(msg.split()[1])
            imported = _step(results, "import_peer_fd",
                             lambda: cu.import_posix_fd(fds[0]))
            os.close(fds[0])
            iva = _step(results, "map_peer",
                        lambda: cu.reserve_map_rw(imported, psize, dev))
            peer_got = _step(results, "peer_read",
                             lambda: cu.read_u32(iva, 4))
            results.append("peer_pattern="
                           + str(peer_got == [PATTERN_B] * 4))
            cu.send_msg(parent_sock, "DONE")
            pr.wait(timeout=30)

        results.append("ALL=PASS")
    except cu.CudaError:
        results.append("ALL=FAIL")
    except Exception as e:  # noqa: BLE001 - report, do not crash the probe
        results.append(f"harness_error={e!r}")
        results.append("ALL=FAIL")

    with open(RESULT, "w") as f:
        f.write("\n".join(results) + "\n")
    print("[child] result: " + " ".join(results), flush=True)

    while not os.path.exists(STOP):
        time.sleep(0.5)
    print("[child] stopping", flush=True)
    return 0


# ---------------------------------------------------------------------------
# parent
# ---------------------------------------------------------------------------

def cc_action(cc, pid, *args):
    r = subprocess.run([cc, *args, "--pid", str(pid)],
                       capture_output=True, text=True)
    print(f"[parent] cuda-checkpoint {' '.join(args)}: rc={r.returncode} "
          f"out={r.stdout.strip()!r} err={r.stderr.strip()!r}", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--peer", action="store_true")
    ap.add_argument("--phase", default="plain", choices=["plain", "import"])
    ap.add_argument("--src-ordinal", type=int, default=0,
                    help="CUDA ordinal the child runs on before checkpoint")
    ap.add_argument("--dst-index", type=int, default=6,
                    help="nvidia-smi index to move the process to")
    ap.add_argument("--no-map", action="store_true",
                    help="restore without --device-map (A/B control)")
    ap.add_argument("--map-format", default="uuid",
                    choices=["uuid", "nopfx"],
                    help="UUID spelling in --device-map: as nvidia-smi prints "
                         "it, or with the \"GPU-\" prefix stripped")
    ap.add_argument("--no-cr", action="store_true")
    ap.add_argument("--cuda-checkpoint",
                    default="/usr/local/bin/cuda-checkpoint")
    args = ap.parse_args()

    if args.peer:
        return peer(args.src_ordinal, int(os.environ["DEVMAP_PEER_FD"]))
    if args.child:
        return child()

    for f in (READY, RESUME, RESULT, STOP):
        if os.path.exists(f):
            os.remove(f)

    uuids = dict(smi_uuids())
    dst_uuid = uuids.get(args.dst_index)
    if dst_uuid is None:
        print(f"[parent] no GPU at index {args.dst_index}", flush=True)
        return 1

    env = dict(os.environ, DEVMAP_PHASE=args.phase,
               DEVMAP_SRC_ORDINAL=str(args.src_ordinal))
    argv = [sys.executable, os.path.abspath(__file__), "--child"]
    proc = subprocess.Popen(argv, env=env)
    try:
        while not os.path.exists(READY):
            if proc.poll() is not None:
                print("[parent] child died before READY", flush=True)
                return 1
            time.sleep(0.2)
        pid_s, src_uuid = open(READY).read().split()
        pid = int(pid_s)
        print(f"[parent] child ready pid={pid}", flush=True)
        print(f"[parent] placement before: {smi_placement(pid)}", flush=True)
        print(f"[parent] moving {src_uuid} -> {dst_uuid} "
              f"(index {args.dst_index})", flush=True)

        if not args.no_cr:
            if cc_action(args.cuda_checkpoint, pid, "--action", "lock",
                         "--timeout", "10000"):
                return 1
            if cc_action(args.cuda_checkpoint, pid, "--action", "checkpoint"):
                return 1
            restore = ["--action", "restore"]
            if not args.no_map:
                src, dst = src_uuid, dst_uuid
                if args.map_format == "nopfx":
                    src = src.removeprefix("GPU-")
                    dst = dst.removeprefix("GPU-")
                restore += ["--device-map", f"{src}={dst}"]
            if cc_action(args.cuda_checkpoint, pid, *restore):
                print("[parent] VERDICT: device-mapped RESTORE ITSELF FAILED",
                      flush=True)
                return 2
            if cc_action(args.cuda_checkpoint, pid, "--action", "unlock"):
                return 1
        else:
            print("[parent] --no-cr: skipping checkpoint cycle", flush=True)

        print(f"[parent] placement after: {smi_placement(pid)}", flush=True)
        with open(RESUME, "w") as f:
            f.write(f"{args.dst_index}\n")

        deadline = time.time() + 180
        while not os.path.exists(RESULT) and time.time() < deadline:
            if proc.poll() is not None:
                print("[parent] child died before RESULT", flush=True)
                return 1
            time.sleep(0.2)
        if not os.path.exists(RESULT):
            print("[parent] TIMEOUT waiting for RESULT", flush=True)
            return 1
        body = open(RESULT).read()
        print("[parent] ===== RESULT =====\n" + body, flush=True)
        print(f"[parent] placement final: {smi_placement(pid)}", flush=True)
        ok = "ALL=PASS" in body
        print(f"[parent] VERDICT: {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 2
    finally:
        open(STOP, "w").close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
