# Repro C1: live CUDA IPC mapping held across the snapshot.
#
# A producer process exports a CUDA tensor over a torch.multiprocessing queue
# (cudaIpcGetMemHandle); a consumer imports it (cudaIpcOpenMemHandle) and
# HOLDS the mapping across the snapshot. Both are checkpointable CUDA
# sessions, so the checkpoint itself succeeds — but at restore,
# cuda-checkpoint cannot re-create the imported IPC mapping.
#
# Run:  sudo bash gms/repros/run_repro.sh c1
# Expect: checkpoint succeeds; RESTORE fails with
#   Error toggling CUDA in process ID <pid>: "OS call failed or operation
#   not supported on this OS" (sometimes preceded by "invalid argument").
# Fix to validate separately: don't hold a live IPC mapping at snapshot.


def _producer(queue, stop):
    import torch

    t = torch.ones(1 << 22, device="cuda")
    torch.cuda.synchronize()
    queue.put(t)  # exports a cudaIpcMemHandle for t's storage
    print("C1: producer exported tensor", flush=True)
    stop.wait()  # keep the exporting allocation alive across the snapshot


def _consumer(queue, ready, stop):
    t = queue.get()  # cudaIpcOpenMemHandle happens here
    val = float(t.sum().item())
    print(f"C1: consumer imported tensor, sum={val}", flush=True)
    ready.set()
    stop.wait()  # HOLD the imported mapping across the snapshot


def setup():
    # NB: do NOT initialize CUDA in this (parent) process before forking.
    import torch.multiprocessing as mp

    ctx = mp.get_context("fork")
    stop = ctx.Event()
    ready = ctx.Event()
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=_producer, args=(queue, stop), daemon=True),
        ctx.Process(target=_consumer, args=(queue, ready, stop), daemon=True),
    ]
    for p in procs:
        p.start()
    if not ready.wait(120):
        raise RuntimeError("consumer never imported the IPC handle")
    return {"stop": stop, "procs": procs}


if __name__ == "__main__":
    from _harness import serve

    serve("C1: CUDA IPC handle", setup)
