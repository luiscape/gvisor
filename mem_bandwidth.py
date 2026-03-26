#!/usr/bin/env python3
"""
CPU ⇔ GPU Memory Bandwidth Benchmark (PyTorch, standalone).

Measures host-to-device and device-to-host transfer bandwidth across
five memory modes:

  pinned      – cudaHostAlloc / pin_memory()
  pageable    – plain malloc (the nvproxy hot path: 3 ioctls per 4 MB chunk)
  prefaulted  – pageable with every page touched before timing
  hugepage    – prefaulted + madvise(MADV_HUGEPAGE)
  registered  – cudaHostRegister on pre-faulted pageable memory

Usage — single run:
    python3 mem_bandwidth.py
    python3 mem_bandwidth.py --sizes-mb 64,256,1024

Usage — automatic Docker comparison (runc vs gVisor):
    python3 mem_bandwidth.py --docker-compare
    python3 mem_bandwidth.py --docker-compare --runtimes runc,runsc,runsc-fastpath

The --docker-compare flag builds a tiny container image, runs the benchmark
inside each requested runtime, collects JSON results, and prints a
side-by-side table with deltas.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path

GB = 1024**3
MB = 1024**2

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prefault(tensor):
    """Touch every page so faults happen outside the timed region."""
    tensor.fill_(0)


def _try_madvise_hugepage(tensor):
    """Hint to back this region with transparent huge pages."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    MADV_HUGEPAGE = 14
    ptr = tensor.data_ptr()
    nbytes = tensor.nelement() * tensor.element_size()
    page = 2 * MB
    aligned = (ptr + page - 1) & ~(page - 1)
    length = nbytes - (aligned - ptr)
    if length > 0:
        libc.madvise(
            ctypes.c_void_p(aligned), ctypes.c_size_t(length), MADV_HUGEPAGE
        )


def _detect_runtime() -> str:
    """Best-effort detection of the container runtime."""
    if os.path.exists("/proc/self/gvisor"):
        return "gVisor"
    try:
        with open("/proc/self/cgroup") as f:
            data = f.read()
        if "docker" in data or "containerd" in data:
            return "container"
    except OSError:
        pass
    return "native"

# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

@dataclass
class Result:
    size_mb: int
    direction: str      # "h2d" or "d2h"
    mode: str           # pinned / pageable / prefaulted / hugepage / registered
    bw_gbs: float       # GB/s
    repeats: int
    runtime: str = ""


def benchmark_transfer(
    size_bytes: int,
    direction: str,
    mode: str,
    warmup: int = 5,
    repeats: int = 20,
) -> float:
    """Returns bandwidth in GB/s."""
    import torch

    device = torch.device("cuda:0")
    n_floats = size_bytes // 4
    pinned = mode == "pinned"

    if direction == "h2d":
        cpu_t = torch.empty(n_floats, dtype=torch.float32)
        if pinned:
            cpu_t = cpu_t.pin_memory()
        elif mode == "prefaulted":
            _prefault(cpu_t)
        elif mode == "hugepage":
            _try_madvise_hugepage(cpu_t)
            _prefault(cpu_t)
        elif mode == "registered":
            _prefault(cpu_t)
            torch.cuda.cudart().cudaHostRegister(cpu_t.data_ptr(), size_bytes, 0)
        gpu_t = torch.empty(n_floats, dtype=torch.float32, device=device)
        transfer = lambda: gpu_t.copy_(cpu_t)
    else:
        gpu_t = torch.empty(n_floats, dtype=torch.float32, device=device)
        cpu_t = torch.empty(n_floats, dtype=torch.float32)
        if pinned:
            cpu_t = cpu_t.pin_memory()
        elif mode == "prefaulted":
            _prefault(cpu_t)
        elif mode == "hugepage":
            _try_madvise_hugepage(cpu_t)
            _prefault(cpu_t)
        elif mode == "registered":
            _prefault(cpu_t)
            torch.cuda.cudart().cudaHostRegister(cpu_t.data_ptr(), size_bytes, 0)
        transfer = lambda: cpu_t.copy_(gpu_t)

    for _ in range(warmup):
        transfer()
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeats):
        transfer()
    end.record()
    torch.cuda.synchronize(device)

    elapsed_s = start.elapsed_time(end) / 1000.0

    if mode == "registered":
        torch.cuda.cudart().cudaHostUnregister(cpu_t.data_ptr())

    return (size_bytes * repeats) / elapsed_s / GB


DIRECTIONS = [("h2d", "CPU → GPU"), ("d2h", "GPU → CPU")]
MODES = [
    ("pinned",      "pinned"),
    ("pageable",    "pageable"),
    ("prefaulted",  "pre-fault"),
    ("hugepage",    "huge+pflt"),
    ("registered",  "hostReg"),
]


def run_sweep(
    sizes_mb: list[int],
    warmup: int = 5,
    repeats: int = 20,
    runtime_label: str = "",
) -> list[Result]:
    results: list[Result] = []
    for size_mb in sizes_mb:
        size_bytes = size_mb * MB
        for dir_key, _ in DIRECTIONS:
            for mode_key, _ in MODES:
                bw = benchmark_transfer(size_bytes, dir_key, mode_key, warmup, repeats)
                results.append(Result(
                    size_mb=size_mb,
                    direction=dir_key,
                    mode=mode_key,
                    bw_gbs=round(bw, 3),
                    repeats=repeats,
                    runtime=runtime_label,
                ))
    return results


def print_results(results: list[Result]):
    dir_labels = dict(DIRECTIONS)
    mode_labels = dict(MODES)
    hdr = f"{'Size':>10}  {'Direction':>12}  {'Memory':>10}  {'Bandwidth':>12}  {'Repeat':>7}"
    print(hdr)
    print("-" * len(hdr))
    prev_size = None
    for r in results:
        if prev_size is not None and r.size_mb != prev_size:
            print()
        prev_size = r.size_mb
        print(
            f"{r.size_mb:>8} MB  {dir_labels.get(r.direction, r.direction):>12}  "
            f"{mode_labels.get(r.mode, r.mode):>10}  "
            f"{r.bw_gbs:>9.2f} GB/s  {r.repeats:>7}x"
        )


def print_header():
    import torch
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    rt = _detect_runtime()
    print("=" * 64)
    print("  PyTorch CPU ⇔ GPU Memory Bandwidth Benchmark")
    print(f"  GPU     : {props.name}")
    print(f"  VRAM    : {props.total_memory / GB:.1f} GB")
    print(f"  CPUs    : {os.cpu_count()}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  Runtime : {rt}")
    print("=" * 64)
    print()


# ---------------------------------------------------------------------------
# Docker comparison mode
# ---------------------------------------------------------------------------

DOCKERFILE_TEMPLATE = textwrap.dedent("""\
    FROM {base_image}
    ENV DEBIAN_FRONTEND=noninteractive
    RUN apt-get update -qq && \\
        apt-get install -y -qq python3-pip > /dev/null 2>&1 ; \\
        pip3 install --break-system-packages --quiet \\
            torch --index-url https://download.pytorch.org/whl/cu126 || \\
        pip3 install --quiet torch
    COPY mem_bandwidth.py /bench/mem_bandwidth.py
    WORKDIR /bench
    ENTRYPOINT ["python3", "/bench/mem_bandwidth.py"]
""")


def docker_compare(
    sizes_mb: list[int],
    warmup: int,
    repeats: int,
    runtimes: list[str],
    base_image: str,
):
    """Build a container, run the benchmark under each runtime, compare."""

    script_path = Path(__file__).resolve()
    tmp_dir = Path("/tmp/nvproxy_bw_bench")
    tmp_dir.mkdir(exist_ok=True)

    # Write Dockerfile
    dockerfile = tmp_dir / "Dockerfile"
    dockerfile.write_text(DOCKERFILE_TEMPLATE.format(base_image=base_image))

    # Copy this script into the build context
    import shutil
    shutil.copy2(script_path, tmp_dir / "mem_bandwidth.py")

    # Build the image
    image_tag = "nvproxy-bw-bench:latest"
    print(f"Building container image ({image_tag}) …")
    subprocess.run(
        ["docker", "build", "-t", image_tag, "."],
        cwd=tmp_dir,
        check=True,
        capture_output=True,
    )
    print("Build complete.\n")

    sizes_arg = ",".join(str(s) for s in sizes_mb)
    all_results: dict[str, list[Result]] = {}

    for rt in runtimes:
        print(f"{'─'*64}")
        print(f"  Running under: {rt}")
        print(f"{'─'*64}")
        # For runc, use the "nvidia" runtime which wraps runc and injects
        # GPU devices via nvidia-container-runtime.  gVisor runtimes
        # (runsc, runsc-fastpath, …) handle GPU access through nvproxy
        # and work with --gpus all directly.
        docker_runtime = "nvidia" if rt == "runc" else rt
        cmd = [
            "docker", "run", "--rm",
            "--runtime", docker_runtime,
            "--gpus", "all",
            image_tag,
            "--sizes-mb", sizes_arg,
            "--warmup", str(warmup),
            "--repeats", str(repeats),
            "--json",
            "--runtime-label", rt,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  ✗ FAILED (exit {proc.returncode})")
            if proc.stderr:
                for line in proc.stderr.strip().splitlines()[-10:]:
                    print(f"    {line}")
            print()
            continue

        # Parse JSON results from the last line
        json_line = None
        for line in reversed(proc.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("["):
                json_line = line
                break
        if json_line is None:
            print(f"  ✗ No JSON output found")
            print(f"  stdout: {proc.stdout[:500]}")
            continue

        rows = json.loads(json_line)
        results = [Result(**r) for r in rows]
        all_results[rt] = results
        print(f"  ✓ {len(results)} measurements collected\n")

    if len(all_results) < 1:
        print("No results to compare.")
        return

    # -----------------------------------------------------------------------
    # Print comparison table
    # -----------------------------------------------------------------------
    print()
    print("=" * 100)
    print("  COMPARISON TABLE")
    print("=" * 100)

    dir_labels = dict(DIRECTIONS)
    mode_labels = dict(MODES)
    rt_names = list(all_results.keys())

    # Build lookup: (size, dir, mode) -> {runtime: bw}
    lookup: dict[tuple, dict[str, float]] = {}
    for rt, results in all_results.items():
        for r in results:
            key = (r.size_mb, r.direction, r.mode)
            lookup.setdefault(key, {})[rt] = r.bw_gbs

    # Header
    bw_cols = "  ".join(f"{rt:>12}" for rt in rt_names)
    delta_cols = ""
    if len(rt_names) == 2:
        delta_cols = f"  {'Δ%':>8}"
    elif len(rt_names) > 2:
        # Show delta of each subsequent runtime vs the first
        delta_cols = "  ".join(f"{'Δ vs ' + rt_names[0]:>12}" for _ in rt_names[1:])
        delta_cols = "  " + delta_cols

    hdr = f"{'Size':>8}  {'Dir':>10}  {'Mode':>10}  {bw_cols}{delta_cols}"
    print(hdr)
    print("─" * len(hdr))

    prev_size = None
    for key in sorted(lookup.keys()):
        size_mb, direction, mode = key
        if prev_size is not None and size_mb != prev_size:
            print()
        prev_size = size_mb

        bws = [lookup[key].get(rt, float("nan")) for rt in rt_names]
        bw_strs = "  ".join(f"{bw:>9.2f} GB" for bw in bws)

        delta_strs = ""
        if len(rt_names) >= 2:
            base = bws[0]
            deltas = []
            for bw in bws[1:]:
                if base > 0 and bw == bw:  # not nan
                    pct = (bw - base) / base * 100
                    sign = "+" if pct >= 0 else ""
                    deltas.append(f"{sign}{pct:>6.1f}%")
                else:
                    deltas.append(f"{'—':>8}")
            delta_strs = "  " + "  ".join(f"{d:>12}" for d in deltas)
            if len(rt_names) == 2:
                delta_strs = "  " + "  ".join(f"{d:>8}" for d in deltas)

        dir_label = dir_labels.get(direction, direction)
        mode_label = mode_labels.get(mode, mode)
        print(f"{size_mb:>6} MB  {dir_label:>10}  {mode_label:>10}  {bw_strs}{delta_strs}")

    print()

    # Summary: average pageable bandwidth per runtime
    print("─" * 64)
    print("  Pageable transfer averages (the nvproxy-sensitive path):")
    for rt in rt_names:
        pageable_bws = [
            lookup[k][rt]
            for k in lookup
            if k[2] == "pageable" and rt in lookup[k]
        ]
        if pageable_bws:
            avg = sum(pageable_bws) / len(pageable_bws)
            print(f"    {rt:>20s}:  {avg:>8.2f} GB/s average")
    print("─" * 64)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CPU ⇔ GPU memory bandwidth benchmark (PyTorch)"
    )
    parser.add_argument(
        "--sizes-mb", type=str, default="1,4,16,64,256,512,1024",
        help="Comma-separated transfer sizes in MB (default: 1,4,16,64,256,512,1024)",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--json", action="store_true",
        help="Append JSON array of results to stdout (for machine parsing)",
    )
    parser.add_argument(
        "--runtime-label", type=str, default="",
        help="Label to tag results with (e.g. 'runc', 'runsc')",
    )
    parser.add_argument(
        "--docker-compare", action="store_true",
        help="Run benchmark inside Docker under multiple runtimes and compare",
    )
    parser.add_argument(
        "--runtimes", type=str, default="runc,runsc",
        help="Comma-separated Docker runtimes for --docker-compare "
             "(default: runc,runsc). 'runc' is automatically mapped to "
             "the 'nvidia' Docker runtime for GPU passthrough.",
    )
    parser.add_argument(
        "--base-image", type=str,
        default="nvidia/cuda:12.6.3-devel-ubuntu22.04",
        help="Base Docker image for --docker-compare",
    )
    args = parser.parse_args()

    sizes_mb = [int(s) for s in args.sizes_mb.split(",")]

    if args.docker_compare:
        runtimes = [r.strip() for r in args.runtimes.split(",")]
        docker_compare(
            sizes_mb=sizes_mb,
            warmup=args.warmup,
            repeats=args.repeats,
            runtimes=runtimes,
            base_image=args.base_image,
        )
        return

    # Direct execution (native or inside container)
    runtime_label = args.runtime_label or _detect_runtime()

    print_header()
    results = run_sweep(
        sizes_mb=sizes_mb,
        warmup=args.warmup,
        repeats=args.repeats,
        runtime_label=runtime_label,
    )
    print_results(results)

    if args.json:
        print()
        print(json.dumps([asdict(r) for r in results]))


if __name__ == "__main__":
    main()
