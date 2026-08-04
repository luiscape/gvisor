#!/usr/bin/env python3
"""Launcher that forks one nccl_suspend_mp.py rank per GPU (like a vLLM/SGLang
tensor-parallel launcher). This is the PID that cuda-checkpoint --launch-job
wraps: all rank children become members of the same checkpoint job.

The launcher itself does no CUDA work; it just spawns ranks, waits for all
`ready.<rank>` files, writes a combined `status` line, and reaps children.

  WORLD=4 python3 nccl_mp_launcher.py --dir /tmp/mp [--graph]
"""

import argparse
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/mp")
    ap.add_argument("--world", type=int, default=int(os.environ.get("WORLD", "4")))
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--pause-only", action="store_true",
                    help="forwarded to ranks (mcshim/stock-NCCL mode)")
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))

    def status(line):
        tmp = os.path.join(args.dir, "status.tmp")
        with open(tmp, "w") as f:
            f.write(line + "\n")
        os.replace(tmp, os.path.join(args.dir, "status"))
        print(line, flush=True)

    procs = []
    for rank in range(args.world):
        env = dict(os.environ, RANK=str(rank), WORLD=str(args.world))
        cmd = [sys.executable, os.path.join(here, "nccl_suspend_mp.py"),
               "--dir", args.dir, "--rank", str(rank), "--world", str(args.world),
               "--interval", str(args.interval)]
        if args.graph:
            cmd.append("--graph")
        if args.pause_only:
            cmd.append("--pause-only")
        procs.append(subprocess.Popen(cmd, env=env))
    status(f"LAUNCHED world={args.world} pids={[p.pid for p in procs]}")

    # Wait for all ranks READY.
    for _ in range(3000):
        if all(os.path.exists(os.path.join(args.dir, f"ready.{r}"))
               for r in range(args.world)):
            status(f"ALL-READY world={args.world}")
            break
        for p in procs:
            if p.poll() is not None:
                status(f"RANK-DIED pid={p.pid} rc={p.returncode}")
                for q in procs:
                    if q.poll() is None:
                        q.kill()
                return 1
        time.sleep(0.1)
    else:
        status("TIMEOUT waiting for ranks READY")
        return 1

    # Idle; the ranks self-verify. Reap on exit / signal.
    try:
        while True:
            time.sleep(1)
            dead = [p for p in procs if p.poll() is not None]
            if dead:
                status(f"RANK-EXITED rc={[p.returncode for p in dead]}")
                break
    except KeyboardInterrupt:
        pass
    for p in procs:
        if p.poll() is None:
            p.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
