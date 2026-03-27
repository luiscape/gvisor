#!/usr/bin/env python3
"""
Pageable GPU memory bandwidth benchmark — measures the nvproxy-sensitive path.

Pageable cudaMemcpy is the transfer mode affected by nvproxy ioctl overhead:
the CUDA driver issues ~3 ioctls per 4 MB chunk, and each round-trips through
the gVisor Sentry. Pinned transfers bypass this path entirely (zero-copy DMA)
and serve as the control — if pinned is identical but pageable differs, the
gap is pure ioctl overhead.

Usage:
    # Single run (native or inside a container):
    python3 mem_bandwidth.py

    # Compare runtimes via Docker:
    python3 mem_bandwidth.py --docker-compare
    python3 mem_bandwidth.py --docker-compare --runtimes runc,runsc,runsc-fastpath

    # Quick smoke test:
    python3 mem_bandwidth.py --docker-compare --sizes-mb 256,1024 --repeats 10
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

GB = 1024**3
MB = 1024**2


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------


def _detect_runtime() -> str:
    if os.path.exists("/proc/self/gvisor"):
        return "gVisor"
    try:
        with open("/proc/self/cgroup") as f:
            if any(k in f.read() for k in ("docker", "containerd")):
                return "container"
    except OSError:
        pass
    return "native"


@dataclass
class Result:
    size_mb: int
    direction: str  # h2d | d2h
    mode: str  # pageable | pinned
    bw_gbs: float
    repeats: int
    runtime: str = ""


def benchmark_transfer(
    size_bytes: int, direction: str, pinned: bool, warmup: int = 5, repeats: int = 20
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

    elapsed = t0.elapsed_time(t1) / 1000.0  # ms → s
    return (size_bytes * repeats) / elapsed / GB


def run_sweep(
    sizes_mb: list[int], warmup: int, repeats: int, runtime_label: str = ""
) -> list[Result]:
    results = []
    for sz in sizes_mb:
        nbytes = sz * MB
        for direction in ("h2d", "d2h"):
            for mode, pin in (("pageable", False), ("pinned", True)):
                bw = benchmark_transfer(nbytes, direction, pin, warmup, repeats)
                results.append(
                    Result(sz, direction, mode, round(bw, 3), repeats, runtime_label)
                )
    return results


def print_table(results: list[Result]):
    dir_label = {"h2d": "CPU → GPU", "d2h": "GPU → CPU"}
    hdr = f"{'Size':>10}  {'Direction':>10}  {'Mode':>10}  {'BW':>12}"
    print(hdr)
    print("─" * len(hdr))
    prev = None
    for r in results:
        if prev and r.size_mb != prev:
            print()
        prev = r.size_mb
        print(
            f"{r.size_mb:>8} MB  {dir_label[r.direction]:>10}  "
            f"{r.mode:>10}  {r.bw_gbs:>9.2f} GB/s"
        )


def print_header():
    import torch

    props = torch.cuda.get_device_properties(0)
    print("=" * 60)
    print("  Pageable GPU Memory Bandwidth Benchmark")
    print(f"  GPU     : {props.name}")
    print(f"  VRAM    : {props.total_memory / GB:.1f} GB")
    print(f"  CPUs    : {os.cpu_count()}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  Runtime : {_detect_runtime()}")

    # PCIe link info (best effort)
    try:
        out = subprocess.check_output(
            "lspci -vvv 2>/dev/null | grep -A2 'NVIDIA' | grep LnkSta | head -1",
            shell=True,
            text=True,
        ).strip()
        if out:
            print(f"  PCIe    : {out.split('LnkSta:')[1].strip().split(',')[0]}")
    except Exception:
        pass
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Docker comparison
# ---------------------------------------------------------------------------

DOCKERFILE = textwrap.dedent("""\
    FROM {base_image}
    ENV DEBIAN_FRONTEND=noninteractive
    RUN apt-get update -qq && apt-get install -y -qq python3-pip pciutils > /dev/null 2>&1 && \
        pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
    COPY mem_bandwidth.py /bench/mem_bandwidth.py
    WORKDIR /bench
    ENTRYPOINT ["python3", "/bench/mem_bandwidth.py"]
""")


def docker_compare(
    sizes_mb: list[int], warmup: int, repeats: int, runtimes: list[str], base_image: str
):
    tmp = Path("/tmp/nvproxy_bw_bench")
    tmp.mkdir(exist_ok=True, mode=0o777)
    df = tmp / "Dockerfile"
    df.write_text(DOCKERFILE.format(base_image=base_image))
    os.chmod(df, 0o644)
    dst = tmp / "mem_bandwidth.py"
    shutil.copy2(Path(__file__).resolve(), dst)
    os.chmod(dst, 0o644)

    tag = "nvproxy-bw-bench:latest"
    # Skip build if image already exists
    check = subprocess.run(
        ["docker", "image", "inspect", tag],
        capture_output=True,
    )
    if check.returncode == 0:
        print(
            f"Image {tag} already exists, skipping build. (delete with: docker rmi {tag})"
        )
    else:
        print(f"Building image ({tag})…")
        proc = subprocess.run(
            ["docker", "build", "--no-cache", "-t", tag, "."],
            cwd=tmp,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print("BUILD FAILED:")
            for ln in proc.stdout.splitlines()[-20:]:
                print(f"  {ln}")
            for ln in proc.stderr.splitlines()[-10:]:
                print(f"  {ln}")
            sys.exit(1)
        print("Done.")
    print()

    sizes_arg = ",".join(str(s) for s in sizes_mb)
    all_results: dict[str, list[Result]] = {}

    for rt in runtimes:
        print(f"{'─' * 60}")
        print(f"  Running: {rt}")
        print(f"{'─' * 60}")
        docker_rt = "nvidia" if rt == "runc" else rt
        cmd = [
            "docker",
            "run",
            "--rm",
            "--runtime",
            docker_rt,
            "--gpus",
            "all",
            tag,
            "--sizes-mb",
            sizes_arg,
            "--warmup",
            str(warmup),
            "--repeats",
            str(repeats),
            "--json",
            "--runtime-label",
            rt,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  FAILED (exit {proc.returncode})")
            for ln in proc.stderr.strip().splitlines()[-8:]:
                print(f"    {ln}")
            print()
            continue
        # Extract JSON from last matching line
        json_line = next(
            (
                l
                for l in reversed(proc.stdout.splitlines())
                if l.strip().startswith("[")
            ),
            None,
        )
        if not json_line:
            print(f"  No JSON output.\n  stdout: {proc.stdout[:300]}")
            continue
        results = [Result(**r) for r in json.loads(json_line)]
        all_results[rt] = results
        print(f"  OK — {len(results)} measurements\n")

    if not all_results:
        print("No results.")
        return

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    rt_names = list(all_results.keys())
    dir_label = {"h2d": "CPU → GPU", "d2h": "GPU → CPU"}

    # build lookup: (size, dir, mode) → {rt: bw}
    lookup: dict[tuple, dict[str, float]] = {}
    for rt, res in all_results.items():
        for r in res:
            lookup.setdefault((r.size_mb, r.direction, r.mode), {})[rt] = r.bw_gbs

    print()
    print("=" * 90)
    print("  COMPARISON")
    print("=" * 90)

    bw_hdr = "  ".join(f"{rt:>14}" for rt in rt_names)
    # If 2+ runtimes, show deltas vs first
    delta_hdr = ""
    if len(rt_names) >= 2:
        delta_hdr = "  ".join(f"{'Δ' + rt_names[0]:>10}" for _ in rt_names[1:])
        delta_hdr = "  " + delta_hdr
    hdr = f"{'Size':>8}  {'Dir':>10}  {'Mode':>10}  {bw_hdr}{delta_hdr}"
    print(hdr)
    print("─" * len(hdr))

    prev_sz = None
    for key in sorted(lookup.keys()):
        sz, d, mode = key
        if prev_sz is not None and sz != prev_sz:
            print()
        prev_sz = sz
        bws = [lookup[key].get(rt, float("nan")) for rt in rt_names]
        bw_str = "  ".join(f"{bw:>11.2f} GB" for bw in bws)
        delta_str = ""
        if len(rt_names) >= 2:
            base = bws[0]
            parts = []
            for bw in bws[1:]:
                if base > 0 and bw == bw:
                    pct = (bw - base) / base * 100
                    parts.append(f"{pct:>+8.1f}%")
                else:
                    parts.append(f"{'—':>10}")
            delta_str = "  " + "  ".join(f"{p:>10}" for p in parts)
        marker = dir_label.get(d, d)
        print(f"{sz:>6} MB  {marker:>10}  {mode:>10}  {bw_str}{delta_str}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("─" * 60)
    for mode in ("pageable", "pinned"):
        print(f"  {mode.upper()} averages:")
        for rt in rt_names:
            vals = [lookup[k][rt] for k in lookup if k[2] == mode and rt in lookup[k]]
            if vals:
                avg = sum(vals) / len(vals)
                print(f"    {rt:>20s}:  {avg:>8.2f} GB/s")
        # Show gap between first and rest for pageable
        if mode == "pageable" and len(rt_names) >= 2:
            base_vals = [
                lookup[k][rt_names[0]]
                for k in lookup
                if k[2] == "pageable" and rt_names[0] in lookup[k]
            ]
            base_avg = sum(base_vals) / len(base_vals) if base_vals else 0
            for rt in rt_names[1:]:
                vals = [
                    lookup[k][rt]
                    for k in lookup
                    if k[2] == "pageable" and rt in lookup[k]
                ]
                if vals and base_avg:
                    avg = sum(vals) / len(vals)
                    pct = (avg - base_avg) / base_avg * 100
                    ioctls_1g = (1024 // 4) * 3  # 768
                    # Estimate per-ioctl overhead from 1 GB d2h delta
                    d2h_base = [
                        lookup[k][rt_names[0]]
                        for k in lookup
                        if k[2] == "pageable"
                        and k[1] == "d2h"
                        and k[0] >= 256
                        and rt_names[0] in lookup[k]
                    ]
                    d2h_this = [
                        lookup[k][rt]
                        for k in lookup
                        if k[2] == "pageable"
                        and k[1] == "d2h"
                        and k[0] >= 256
                        and rt in lookup[k]
                    ]
                    if d2h_base and d2h_this:
                        db = sum(d2h_base) / len(d2h_base)
                        dt = sum(d2h_this) / len(d2h_this)
                        t_base = 1.0 / db * 1000  # ms for 1 GB
                        t_this = 1.0 / dt * 1000
                        delta_ms = t_this - t_base
                        if delta_ms > 0:
                            per_ioctl = delta_ms / ioctls_1g * 1000  # µs
                            print(
                                f"      {rt} vs {rt_names[0]} d2h gap: "
                                f"{pct:+.1f}% → ~{per_ioctl:.0f} µs/ioctl overhead "
                                f"({delta_ms:.1f} ms over {ioctls_1g} ioctls)"
                            )
    print("─" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Pageable GPU memory bandwidth benchmark")
    p.add_argument("--sizes-mb", default="1,4,16,64,256,512,1024")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--json", action="store_true")
    p.add_argument("--runtime-label", default="")
    p.add_argument("--docker-compare", action="store_true")
    p.add_argument("--runtimes", default="runc,runsc")
    p.add_argument("--base-image", default="nvidia/cuda:12.6.3-base-ubuntu22.04")
    args = p.parse_args()
    sizes = [int(s) for s in args.sizes_mb.split(",")]

    if args.docker_compare:
        rts = [r.strip() for r in args.runtimes.split(",")]
        docker_compare(sizes, args.warmup, args.repeats, rts, args.base_image)
        return

    runtime_label = args.runtime_label or _detect_runtime()
    print_header()
    results = run_sweep(sizes, args.warmup, args.repeats, runtime_label)
    print_table(results)
    if args.json:
        print()
        print(json.dumps([asdict(r) for r in results]))


if __name__ == "__main__":
    main()
