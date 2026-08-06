# Minimal repro of the PyTorch Inductor compile-worker FD-inheritance case.
#
# Mechanism (distilled from SGLang + torch.compile): a process opens a CUDA
# context (which opens /dev/nvidia* FDs), then PyTorch Inductor's async-compile
# SubprocPool spawns persistent compile-worker subprocesses. Those workers
# inherit the parent's /dev/nvidia* FDs but have no CUDA context of their own
# (clientless holders) — the thing cuda-checkpoint cannot toggle and that the
# stub runsc refuses to save ("nvproxy.frontendFD is not saveable").
#
# The whole trigger is:
#     torch.ones(..., device="cuda")            # open /dev/nvidia*
#     AsyncCompile.warm_pool()                  # spawn compile workers
# warm_pool() alone spawns the pool without needing a C compiler; if a real
# toolchain is present, a trivial torch.compile() is also attempted to fan out
# the leaf worker children (the SGLang shape). Either way the workers inherit
# the FDs and are held across the snapshot.
#
# Run:  sudo bash gms/repros/run_repro.sh inductor
# On a runsc with nvproxy C/R: PASS (the inherited FDs are serialized +
# reopened). On the stub: CHECKPOINT-BLOCKED.

import os
import time


def setup():
    import torch

    # 1. Open a CUDA context in this process -> /dev/nvidia* FDs.
    t = torch.ones(1 << 20, device="cuda")
    torch.cuda.synchronize()

    # 2. Spawn the Inductor compile-worker SubprocPool. These subprocesses
    #    inherit the /dev/nvidia* FDs opened above.
    from torch._inductor.async_compile import AsyncCompile

    AsyncCompile.warm_pool()

    # 3. Best-effort real compile to fan out leaf workers (needs a C
    #    compiler + triton; harmless if it fails on a -base image).
    try:

        @torch.compile
        def f(x):
            return (x * 2 + 1).relu().sum()

        f(torch.randn(1024, device="cuda"))
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(f"torch.compile fan-out skipped (no toolchain?): {exc!r}", flush=True)

    # Wait until at least one compile-worker subprocess is holding nvidia FDs.
    deadline = time.time() + 60
    while time.time() < deadline:
        workers = _compile_worker_holders()
        if workers:
            print(f"compile-worker nvidia-FD holders: {sorted(workers)}", flush=True)
            break
        time.sleep(0.5)
    else:
        print("WARNING: no compile-worker FD holder appeared", flush=True)

    return {"t": t}


def _compile_worker_holders() -> set[int]:
    from _diag import nvidia_fd_paths

    out = set()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{entry}/cmdline") as f:
                cmd = f.read()
        except OSError:
            continue
        if "compile_worker" in cmd and nvidia_fd_paths(pid):
            out.add(pid)
    return out


def verify(state):
    import torch

    torch.cuda.synchronize()
    return {"ok": True, "sum": float(state["t"].sum().item())}


if __name__ == "__main__":
    from _harness import serve

    serve("INDUCTOR: compile-worker inherited FDs", setup, verify)
