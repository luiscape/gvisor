"""
Pageable GPU memory bandwidth benchmark — Modal edition.

Measures pageable vs pinned transfer bandwidth to isolate nvproxy ioctl
overhead. Pageable cudaMemcpy is the path affected by nvproxy (3 ioctls
per 4 MB chunk); pinned bypasses it entirely and serves as the control.

Usage:
    # gVisor (default Modal runtime):
    GPU=a10g modal run reproduction_modal.py

    # runc baseline:
    GPU=a10g MODAL_FUNCTION_RUNTIME=runc modal run reproduction_modal.py

    # Pin to a specific worker (same instance for apples-to-apples):
    MODAL_WORKER_ID=wo-zjsq96q14b3lp7hbyrjoeba5q GPU=a10g modal run reproduction_modal.py

    # Full comparison (run both, same worker):
    MODAL_WORKER_ID=wo-zjsq96q14b3lp7hbyrjoeba5q GPU=a10g MODAL_FUNCTION_RUNTIME=runc modal run reproduction_modal.py
    MODAL_WORKER_ID=wo-zjsq96q14b3lp7hbyrjoeba5q GPU=a10g modal run reproduction_modal.py

    # Custom sizes:
    GPU=a10g modal run reproduction_modal.py --sizes-mb 64,256,1024
"""

import os
import resource
import subprocess

import modal

GPU = os.environ.get("GPU", "a10g")
CPU = float(os.environ.get("CPU", "10"))
MEMORY = 1024 * 10  # MB

app = modal.App("pageable-bw-bench")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", add_python="3.11"
    )
    .uv_pip_install("torch==2.11.0", "numpy")
    .entrypoint([])
)

GB = 1024**3
MB = 1024**2


def benchmark_transfer(
    size_bytes: int,
    direction: str,
    pinned: bool,
    warmup: int = 5,
    repeats: int = 20,
) -> float:
    """Return bandwidth in GB/s."""
    import torch

    device = torch.device("cuda:0")
    n = size_bytes // 4

    if direction == "h2d":
        cpu = torch.empty(n, dtype=torch.float32, pin_memory=pinned)
        gpu = torch.empty(n, dtype=torch.float32, device=device)
        xfer = lambda: gpu.copy_(cpu)
    else:
        gpu = torch.empty(n, dtype=torch.float32, device=device)
        cpu = torch.empty(n, dtype=torch.float32, pin_memory=pinned)
        xfer = lambda: cpu.copy_(gpu)

    for _ in range(warmup):
        xfer()
    torch.cuda.synchronize()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(repeats):
        xfer()
    t1.record()
    torch.cuda.synchronize()

    elapsed = t0.elapsed_time(t1) / 1000.0  # ms -> s
    return (size_bytes * repeats) / elapsed / GB


def print_env():
    """Print runtime environment info for comparison across runs."""
    import torch

    props = torch.cuda.get_device_properties(0)

    # Detect runtime
    runtime = "native"
    if os.path.exists("/proc/self/gvisor"):
        runtime = "gVisor"
    else:
        try:
            with open("/proc/self/cgroup") as f:
                if any(k in f.read() for k in ("docker", "containerd")):
                    runtime = "container (runc)"
        except OSError:
            pass

    # RLIMIT_MEMLOCK
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    memlock_str = f"soft={soft}, hard={hard}"
    if soft == resource.RLIM_INFINITY:
        memlock_str = "unlimited"
    elif soft < 1024 * 1024:
        memlock_str = f"{soft // 1024} KB"

    # PCIe info (best effort)
    pcie = "unknown"
    try:
        out = subprocess.check_output(
            "lspci -vvv 2>/dev/null | grep -A2 'NVIDIA' | grep LnkSta | head -1",
            shell=True,
            text=True,
        ).strip()
        if "LnkSta:" in out:
            pcie = out.split("LnkSta:")[1].strip().split(",")[0]
    except Exception:
        pass

    # Memory cgroup limit
    mem_limit = "unlimited"
    for path in [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ]:
        try:
            with open(path) as f:
                val = f.read().strip()
                if val != "max" and val != "9223372036854771712":
                    mem_limit = f"{int(val) / GB:.1f} GB"
                break
        except (OSError, ValueError):
            pass

    # Swap limit
    swap_limit = "unknown"
    for path in [
        "/sys/fs/cgroup/memory.swap.max",
        "/sys/fs/cgroup/memory/memory.memsw.limit_in_bytes",
    ]:
        try:
            with open(path) as f:
                val = f.read().strip()
                swap_limit = "disabled" if val == "0" else val
                break
        except OSError:
            pass

    # Worker ID (set via MODAL_WORKER_ID env var for pinning)
    worker_id = os.environ.get("MODAL_WORKER_ID", "unknown")
    hostname = os.uname().nodename

    print("=" * 64)
    print("  Pageable GPU Memory Bandwidth Benchmark")
    print(f"  GPU            : {props.name}")
    print(f"  VRAM           : {props.total_memory / GB:.1f} GB")
    print(f"  CPUs           : {os.cpu_count()}")
    print(f"  PyTorch        : {torch.__version__}")
    print(f"  Runtime        : {runtime}")
    print(f"  Worker ID      : {worker_id}")
    print(f"  Hostname       : {hostname}")
    print(f"  RLIMIT_MEMLOCK : {memlock_str}")
    print(f"  Memory limit   : {mem_limit}")
    print(f"  Swap limit     : {swap_limit}")
    print(f"  PCIe           : {pcie}")
    print("=" * 64)
    print()


def run_sweep(sizes_mb: list[int], warmup: int = 5, repeats: int = 20):
    dir_label = {"h2d": "CPU -> GPU", "d2h": "GPU -> CPU"}

    header = f"{'Size':>10}  {'Direction':>10}  {'Mode':>10}  {'BW':>12}"
    print(header)
    print("-" * len(header))

    prev = None
    for sz in sizes_mb:
        if prev is not None and sz != prev:
            print()
        prev = sz
        nbytes = sz * MB
        for direction in ("h2d", "d2h"):
            for mode, pin in (("pageable", False), ("pinned", True)):
                bw = benchmark_transfer(nbytes, direction, pin, warmup, repeats)
                print(
                    f"{sz:>8} MB  {dir_label[direction]:>10}  "
                    f"{mode:>10}  {bw:>9.2f} GB/s"
                )


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY, timeout=600)
def benchmark(sizes_mb: list[int], warmup: int, repeats: int):
    print_env()
    run_sweep(sizes_mb=sizes_mb, warmup=warmup, repeats=repeats)


@app.local_entrypoint()
def main(
    sizes_mb: str = "1,4,16,64,256,512,1024",
    warmup: int = 5,
    repeats: int = 20,
):
    benchmark.remote(
        sizes_mb=[int(s) for s in sizes_mb.split(",")],
        warmup=warmup,
        repeats=repeats,
    )
