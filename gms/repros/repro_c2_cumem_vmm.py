# Repro C2: cuMem / VMM-backed allocation held across the snapshot.
#
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True routes torch's caching
# allocator through the CUDA VMM APIs (cuMemCreate + cuMemAddressReserve +
# cuMemMap). A large tensor forces a fresh expandable (cuMem-backed) segment
# that stays mapped across the snapshot. Checkpoint succeeds (one
# self-contained CUDA context) but restore must re-create the cuMem handles.
#
# The runner (run_repro.sh c2) sets PYTORCH_CUDA_ALLOC_CONF in the container.
#
# Run:  sudo bash gms/repros/run_repro.sh c2
# Expect: checkpoint succeeds; RESTORE fails with
#   Error toggling CUDA in process ID <pid>: "OS call failed or operation
#   not supported on this OS".
# If this torch build does not route expandable_segments through cuMem (this
# unexpectedly passes), note it in RESULTS.md and fall back to raw cuMem*.
# Fix to validate separately: release VMM allocations before snapshot (or
# NCCL_CUMEM_*=0 when the cuMem source is NCCL).

import os


def setup():
    assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True", (
        "PYTORCH_CUDA_ALLOC_CONF not set; repro would not use cuMem"
    )
    import torch

    # 256 MiB: forces a fresh expandable (cuMem-backed) segment.
    t = torch.ones(64 << 20, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    print(
        f"C2: allocated {t.numel() * 4 >> 20} MiB "
        f"(reserved {torch.cuda.memory_reserved() >> 20} MiB)",
        flush=True,
    )
    return {"t": t}


def verify(state):
    import torch

    torch.cuda.synchronize()
    return {"ok": True, "sum": float(state["t"].sum().item())}


if __name__ == "__main__":
    from _harness import serve

    serve("C2: cuMem/VMM allocation", setup, verify)
