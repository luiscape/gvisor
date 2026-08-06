# Repro A (positive control): checkpointable CUDA contexts only.
#
# Main process holds a live CUDA context; N_WORKERS subprocesses each create
# their own context after a staggered delay (a mild enumeration-race
# stressor). Every /dev/nvidia* FD holder at snapshot time is a real,
# checkpointable CUDA session, so checkpoint AND restore must succeed.
#
# Run:  sudo bash gms/repros/run_repro.sh a
# Expect: PASS — dump shows N_WORKERS+1 CUDA sessions, /verify returns sum.

import subprocess
import sys

N_WORKERS = 4

WORKER_SCRIPT = """
import sys, time
time.sleep(float(sys.argv[1]))
import torch
t = torch.ones(1 << 20, device="cuda")
torch.cuda.synchronize()
print("worker ready", flush=True)
while True:
    time.sleep(60)
"""


def setup():
    import torch
    from _diag import wait_for_nvidia_fd

    t = torch.ones(1 << 20, device="cuda")
    torch.cuda.synchronize()

    procs = [
        subprocess.Popen([sys.executable, "-c", WORKER_SCRIPT, str(i * 0.5)])
        for i in range(N_WORKERS)
    ]
    for p in procs:
        wait_for_nvidia_fd(p.pid)
    return {"t": t, "procs": procs}


def verify(state):
    import torch

    torch.cuda.synchronize()
    return {"ok": True, "sum": float(state["t"].sum().item())}


if __name__ == "__main__":
    from _harness import serve

    serve("A: checkpointable CUDA contexts", setup, verify)
