# Repro B1: NVML-only /dev/nvidiactl FD, no CUDA context.
#
# A child process calls nvmlInit() and never nvmlShutdown(); it holds an open
# /dev/nvidiactl with zero CUDA state. cuda-checkpoint --get-state fails on
# it, so it is invisible to the toggle flow and its frontend FD reaches the
# sentry's state encoder.
#
# Run:  sudo bash gms/repros/run_repro.sh b1
# Expect on stub runsc: checkpoint blocked ("nvproxy.frontendFD is not
# saveable" / "can't save with live nvproxy clients"). On a runsc with
# nvproxy C/R: checkpoint succeeds (the FD is serialized).
# Fix to validate separately: nvmlShutdown() before snapshot.

import subprocess
import sys

NVML_SCRIPT = """
import time
import pynvml
pynvml.nvmlInit()
print("nvml ready", flush=True)
# Deliberately no nvmlShutdown(): the RM session and /dev/nvidiactl FD stay
# open until process exit.
while True:
    time.sleep(60)
"""


def setup():
    from _diag import wait_for_nvidia_fd

    child = subprocess.Popen([sys.executable, "-c", NVML_SCRIPT])
    wait_for_nvidia_fd(child.pid, timeout=60)
    return {"child": child}


if __name__ == "__main__":
    from _harness import serve

    serve("B1: NVML-only FD", setup)
