#!/usr/bin/env python3
"""Fork one torch_nccl_ckpt.py rank per GPU under a single parent.

The parent is the process gVisor wraps in `cuda-checkpoint --launch-job`, so
the launcher and every rank belong to one checkpoint job. This mirrors how an
inference engine spawns tensor-parallel workers, without torchrun (which adds
its own supervisor processes and complicates the process set gVisor picks up).
"""

import argparse
import os
import signal
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/torchnccl")
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--master-port", default="29555")
    ap.add_argument("--workload", default=None,
                    help="path to torch_nccl_ckpt.py (default: alongside this file)")
    args, passthrough = ap.parse_known_args()

    workload = args.workload or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "torch_nccl_ckpt.py")
    os.makedirs(args.dir, exist_ok=True)

    procs = []
    for rank in range(args.world):
        env = dict(os.environ)
        env.update({
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(args.world),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": args.master_port,
            # Each rank drives exactly one GPU.
            "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(args.world)),
        })
        cmd = [sys.executable, workload, "--dir", args.dir] + passthrough
        procs.append(subprocess.Popen(cmd, env=env))
        print(f"[launcher] rank {rank} pid={procs[-1].pid}", flush=True)

    # Wait for every rank to report READY, then announce it once so the runner
    # has a single line to poll for.
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        ready = 0
        for rank in range(args.world):
            p = os.path.join(args.dir, f"status.{rank}")
            try:
                with open(p) as f:
                    if f.read().startswith("READY") or "iter=" in open(p).read():
                        ready += 1
            except OSError:
                pass
        if ready == args.world:
            print("ALL-READY", flush=True)
            break
        if any(p.poll() is not None for p in procs):
            print("[launcher] a rank exited during startup", flush=True)
            break
        time.sleep(1)
    else:
        print("[launcher] timed out waiting for ranks", flush=True)

    def terminate(signum, frame):
        for p in procs:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    rc = 0
    for p in procs:
        rc |= p.wait()
    return rc


if __name__ == "__main__":
    sys.exit(main())
