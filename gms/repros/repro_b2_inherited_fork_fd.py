# Repro B2: /dev/nvidia* FDs inherited via fork() without exec.
#
# The main process creates a CUDA context (opening /dev/nvidia* FDs), then
# fork()s children that never touch CUDA. Each child holds inherited copies
# of the parent's nvidia FDs with no CUDA context of its own — the distilled
# PyTorch Inductor compile-worker case. cuda-checkpoint can toggle the parent
# (closing the PARENT's FDs) but the children's inherited FDs remain open and
# unclaimable.
#
# Run:  sudo bash gms/repros/run_repro.sh b2
# Expect on stub runsc: checkpoint blocked (B-class); dump shows the children
# holding /dev/nvidiactl, /dev/nvidiaN, /dev/nvidia-uvm.
# Fix to validate separately: spawn/exec instead of bare fork
# (e.g. TORCHINDUCTOR_COMPILE_THREADS=1 for the real Inductor case).

import os
import signal
import time

N_CHILDREN = 8


def setup():
    import torch

    # FIRST: open /dev/nvidia* FDs in this process via a CUDA context.
    t = torch.ones(1 << 20, device="cuda")
    torch.cuda.synchronize()

    # THEN: fork children that inherit those FDs and do nothing GPU-related.
    children = []
    for _ in range(N_CHILDREN):
        pid = os.fork()
        if pid == 0:
            try:
                while True:
                    time.sleep(60)
            finally:
                os._exit(0)
        children.append(pid)
    return {"t": t, "children": children}


def _cleanup(children):
    for pid in children:
        try:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        except OSError:
            pass


if __name__ == "__main__":
    from _harness import serve

    serve("B2: inherited fork FDs", setup)
