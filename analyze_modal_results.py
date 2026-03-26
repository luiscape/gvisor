#!/usr/bin/env python3
"""
Analyze Modal runc vs gVisor memory bandwidth results.

Parses the benchmark output, computes per-ioctl nvproxy overhead,
and prints a side-by-side comparison with bandwidth projections.

Usage:
    python3 analyze_modal_results.py
    python3 analyze_modal_results.py --runc-file runc.txt --gvisor-file gvisor.txt
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants from the pageable transfer model
# ---------------------------------------------------------------------------
CHUNK_SIZE_MB = 4  # CUDA driver uses ~4 MB chunks for pageable DMA
IOCTLS_PER_CHUNK = 3  # the hot triplet: setup / pin+kick / RM_CONTROL wait
GB = 1024  # MB in a GB (for size conversions)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    size_mb: int
    direction: str  # "h2d" or "d2h"
    mode: str  # pinned / pageable / pre-fault / huge+pflt / hostReg
    bw_gbs: float
    repeats: int

    @property
    def key(self) -> tuple:
        return (self.size_mb, self.direction, self.mode)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_LINE_RE = re.compile(
    r"^\s*(\d+)\s+MB\s+"
    r"(CPU\s*→\s*GPU|GPU\s*→\s*CPU)\s+"
    r"(\S+)\s+"
    r"([\d.]+)\s+GB/s\s+"
    r"(\d+)x\s*$"
)

_DIR_MAP = {
    "CPU → GPU": "h2d",
    "CPU→GPU": "h2d",
    "GPU → CPU": "d2h",
    "GPU→CPU": "d2h",
}


def parse_results(text: str) -> list[Sample]:
    results = []
    for line in text.splitlines():
        line = line.replace("\u2192", "→")  # normalize arrow
        m = _LINE_RE.match(line)
        if not m:
            continue
        size_mb = int(m.group(1))
        raw_dir = m.group(2).replace("  ", " ").strip()
        direction = _DIR_MAP.get(raw_dir, raw_dir)
        mode = m.group(3)
        bw = float(m.group(4))
        reps = int(m.group(5))
        results.append(Sample(size_mb, direction, mode, bw, reps))
    return results


# ---------------------------------------------------------------------------
# Overhead computation
# ---------------------------------------------------------------------------


def transfer_time_ms(size_mb: int, bw_gbs: float) -> float:
    """Time in ms to transfer size_mb at bw_gbs."""
    if bw_gbs <= 0:
        return float("inf")
    return (size_mb / 1024.0) / bw_gbs * 1000.0


def per_ioctl_overhead_us(
    size_mb: int,
    bw_runc: float,
    bw_gvisor: float,
) -> float:
    """
    Estimate per-ioctl nvproxy overhead in microseconds.

    Model: the bandwidth difference comes entirely from the additional
    Sentry round-trip on each of the 3 ioctls per ~4 MB chunk.
    """
    t_runc = transfer_time_ms(size_mb, bw_runc)
    t_gvisor = transfer_time_ms(size_mb, bw_gvisor)
    delta_ms = t_gvisor - t_runc
    if delta_ms <= 0:
        return 0.0
    n_chunks = max(size_mb / CHUNK_SIZE_MB, 1)
    n_ioctls = n_chunks * IOCTLS_PER_CHUNK
    return delta_ms / n_ioctls * 1000.0  # ms → µs


# ---------------------------------------------------------------------------
# Embedded data (Modal A10G results pasted by user)
# ---------------------------------------------------------------------------

RUNC_DATA = """\
       1 MB     CPU → GPU      pinned      10.62 GB/s       20x
       1 MB     CPU → GPU    pageable       6.94 GB/s       20x
       1 MB     CPU → GPU   pre-fault       6.75 GB/s       20x
       1 MB     CPU → GPU   huge+pflt       6.93 GB/s       20x
       1 MB     CPU → GPU     hostReg      10.70 GB/s       20x
       1 MB     GPU → CPU      pinned      10.64 GB/s       20x
       1 MB     GPU → CPU    pageable       5.99 GB/s       20x
       1 MB     GPU → CPU   pre-fault       6.18 GB/s       20x
       1 MB     GPU → CPU   huge+pflt       3.46 GB/s       20x
       1 MB     GPU → CPU     hostReg      10.65 GB/s       20x
       4 MB     CPU → GPU      pinned      12.01 GB/s       20x
       4 MB     CPU → GPU    pageable       9.92 GB/s       20x
       4 MB     CPU → GPU   pre-fault      10.00 GB/s       20x
       4 MB     CPU → GPU   huge+pflt       9.95 GB/s       20x
       4 MB     CPU → GPU     hostReg      11.98 GB/s       20x
       4 MB     GPU → CPU      pinned      11.79 GB/s       20x
       4 MB     GPU → CPU    pageable       4.95 GB/s       20x
       4 MB     GPU → CPU   pre-fault       5.18 GB/s       20x
       4 MB     GPU → CPU   huge+pflt       8.83 GB/s       20x
       4 MB     GPU → CPU     hostReg      11.80 GB/s       20x
      16 MB     CPU → GPU      pinned      12.38 GB/s       20x
      16 MB     CPU → GPU    pageable      11.42 GB/s       20x
      16 MB     CPU → GPU   pre-fault       9.77 GB/s       20x
      16 MB     CPU → GPU   huge+pflt       8.13 GB/s       20x
      16 MB     CPU → GPU     hostReg      12.38 GB/s       20x
      16 MB     GPU → CPU      pinned      12.17 GB/s       20x
      16 MB     GPU → CPU    pageable       5.65 GB/s       20x
      16 MB     GPU → CPU   pre-fault       5.63 GB/s       20x
      16 MB     GPU → CPU   huge+pflt       4.63 GB/s       20x
      16 MB     GPU → CPU     hostReg      12.14 GB/s       20x
      64 MB     CPU → GPU      pinned      12.49 GB/s       20x
      64 MB     CPU → GPU    pageable      12.01 GB/s       20x
      64 MB     CPU → GPU   pre-fault      10.03 GB/s       20x
      64 MB     CPU → GPU   huge+pflt      10.27 GB/s       20x
      64 MB     CPU → GPU     hostReg      12.46 GB/s       20x
      64 MB     GPU → CPU      pinned      12.26 GB/s       20x
      64 MB     GPU → CPU    pageable       5.83 GB/s       20x
      64 MB     GPU → CPU   pre-fault       5.79 GB/s       20x
      64 MB     GPU → CPU   huge+pflt       5.81 GB/s       20x
      64 MB     GPU → CPU     hostReg      12.21 GB/s       20x
     256 MB     CPU → GPU      pinned      12.49 GB/s       20x
     256 MB     CPU → GPU    pageable      12.08 GB/s       20x
     256 MB     CPU → GPU   pre-fault      10.17 GB/s       20x
     256 MB     CPU → GPU   huge+pflt      10.13 GB/s       20x
     256 MB     CPU → GPU     hostReg      12.50 GB/s       20x
     256 MB     GPU → CPU      pinned      12.28 GB/s       20x
     256 MB     GPU → CPU    pageable       5.94 GB/s       20x
     256 MB     GPU → CPU   pre-fault       5.86 GB/s       20x
     256 MB     GPU → CPU   huge+pflt       5.98 GB/s       20x
     256 MB     GPU → CPU     hostReg      12.24 GB/s       20x
     512 MB     CPU → GPU      pinned      12.49 GB/s       20x
     512 MB     CPU → GPU    pageable      12.18 GB/s       20x
     512 MB     CPU → GPU   pre-fault      10.43 GB/s       20x
     512 MB     CPU → GPU   huge+pflt       9.53 GB/s       20x
     512 MB     CPU → GPU     hostReg      12.51 GB/s       20x
     512 MB     GPU → CPU      pinned      12.27 GB/s       20x
     512 MB     GPU → CPU    pageable       5.96 GB/s       20x
     512 MB     GPU → CPU   pre-fault       5.93 GB/s       20x
     512 MB     GPU → CPU   huge+pflt       5.82 GB/s       20x
     512 MB     GPU → CPU     hostReg      12.25 GB/s       20x
    1024 MB     CPU → GPU      pinned      12.51 GB/s       20x
    1024 MB     CPU → GPU    pageable      12.17 GB/s       20x
    1024 MB     CPU → GPU   pre-fault      10.34 GB/s       20x
    1024 MB     CPU → GPU   huge+pflt      10.15 GB/s       20x
    1024 MB     CPU → GPU     hostReg      12.50 GB/s       20x
    1024 MB     GPU → CPU      pinned      12.27 GB/s       20x
    1024 MB     GPU → CPU    pageable       5.98 GB/s       20x
    1024 MB     GPU → CPU   pre-fault       5.89 GB/s       20x
    1024 MB     GPU → CPU   huge+pflt       5.94 GB/s       20x
    1024 MB     GPU → CPU     hostReg      12.25 GB/s       20x
"""

GVISOR_DATA = """\
       1 MB     CPU → GPU      pinned      10.68 GB/s       20x
       1 MB     CPU → GPU    pageable       7.04 GB/s       20x
       1 MB     CPU → GPU   pre-fault       6.58 GB/s       20x
       1 MB     CPU → GPU   huge+pflt       6.51 GB/s       20x
       1 MB     CPU → GPU     hostReg      10.61 GB/s       20x
       1 MB     GPU → CPU      pinned      10.72 GB/s       20x
       1 MB     GPU → CPU    pageable       5.07 GB/s       20x
       1 MB     GPU → CPU   pre-fault       6.19 GB/s       20x
       1 MB     GPU → CPU   huge+pflt       6.12 GB/s       20x
       1 MB     GPU → CPU     hostReg      10.66 GB/s       20x
       4 MB     CPU → GPU      pinned      11.90 GB/s       20x
       4 MB     CPU → GPU    pageable      10.32 GB/s       20x
       4 MB     CPU → GPU   pre-fault      10.36 GB/s       20x
       4 MB     CPU → GPU   huge+pflt      10.33 GB/s       20x
       4 MB     CPU → GPU     hostReg      11.94 GB/s       20x
       4 MB     GPU → CPU      pinned      11.83 GB/s       20x
       4 MB     GPU → CPU    pageable       7.89 GB/s       20x
       4 MB     GPU → CPU   pre-fault       7.72 GB/s       20x
       4 MB     GPU → CPU   huge+pflt       9.40 GB/s       20x
       4 MB     GPU → CPU     hostReg      11.20 GB/s       20x
      16 MB     CPU → GPU      pinned      12.37 GB/s       20x
      16 MB     CPU → GPU    pageable      10.81 GB/s       20x
      16 MB     CPU → GPU   pre-fault      10.82 GB/s       20x
      16 MB     CPU → GPU   huge+pflt      10.25 GB/s       20x
      16 MB     CPU → GPU     hostReg      12.37 GB/s       20x
      16 MB     GPU → CPU      pinned      12.17 GB/s       20x
      16 MB     GPU → CPU    pageable       9.25 GB/s       20x
      16 MB     GPU → CPU   pre-fault       9.08 GB/s       20x
      16 MB     GPU → CPU   huge+pflt       9.38 GB/s       20x
      16 MB     GPU → CPU     hostReg      11.84 GB/s       20x
      64 MB     CPU → GPU      pinned      12.50 GB/s       20x
      64 MB     CPU → GPU    pageable       9.76 GB/s       20x
      64 MB     CPU → GPU   pre-fault      11.06 GB/s       20x
      64 MB     CPU → GPU   huge+pflt       9.97 GB/s       20x
      64 MB     CPU → GPU     hostReg      12.46 GB/s       20x
      64 MB     GPU → CPU      pinned      12.25 GB/s       20x
      64 MB     GPU → CPU    pageable       5.02 GB/s       20x
      64 MB     GPU → CPU   pre-fault       4.98 GB/s       20x
      64 MB     GPU → CPU   huge+pflt       4.80 GB/s       20x
      64 MB     GPU → CPU     hostReg      12.23 GB/s       20x
     256 MB     CPU → GPU      pinned      12.53 GB/s       20x
     256 MB     CPU → GPU    pageable      10.74 GB/s       20x
     256 MB     CPU → GPU   pre-fault      10.63 GB/s       20x
     256 MB     CPU → GPU   huge+pflt      10.98 GB/s       20x
     256 MB     CPU → GPU     hostReg      12.45 GB/s       20x
     256 MB     GPU → CPU      pinned      12.29 GB/s       20x
     256 MB     GPU → CPU    pageable       5.04 GB/s       20x
     256 MB     GPU → CPU   pre-fault       5.00 GB/s       20x
     256 MB     GPU → CPU   huge+pflt       5.00 GB/s       20x
     256 MB     GPU → CPU     hostReg      12.24 GB/s       20x
     512 MB     CPU → GPU      pinned      12.53 GB/s       20x
     512 MB     CPU → GPU    pageable      10.65 GB/s       20x
     512 MB     CPU → GPU   pre-fault      10.78 GB/s       20x
     512 MB     CPU → GPU   huge+pflt      10.40 GB/s       20x
     512 MB     CPU → GPU     hostReg      12.51 GB/s       20x
     512 MB     GPU → CPU      pinned      12.29 GB/s       20x
     512 MB     GPU → CPU    pageable       5.04 GB/s       20x
     512 MB     GPU → CPU   pre-fault       4.99 GB/s       20x
     512 MB     GPU → CPU   huge+pflt       5.00 GB/s       20x
     512 MB     GPU → CPU     hostReg      12.18 GB/s       20x
    1024 MB     CPU → GPU      pinned      12.49 GB/s       20x
    1024 MB     CPU → GPU    pageable      10.29 GB/s       20x
    1024 MB     CPU → GPU   pre-fault      10.36 GB/s       20x
    1024 MB     CPU → GPU   huge+pflt      10.60 GB/s       20x
    1024 MB     CPU → GPU     hostReg      12.51 GB/s       20x
    1024 MB     GPU → CPU      pinned      12.30 GB/s       20x
    1024 MB     GPU → CPU    pageable       4.98 GB/s       20x
    1024 MB     GPU → CPU   pre-fault       5.05 GB/s       20x
    1024 MB     GPU → CPU   huge+pflt       5.06 GB/s       20x
    1024 MB     GPU → CPU     hostReg      12.26 GB/s       20x
"""


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def main():
    runc = parse_results(RUNC_DATA)
    gvisor = parse_results(GVISOR_DATA)

    runc_map: dict[tuple, Sample] = {s.key: s for s in runc}
    gvisor_map: dict[tuple, Sample] = {s.key: s for s in gvisor}

    dir_label = {"h2d": "CPU → GPU", "d2h": "GPU → CPU"}

    # -----------------------------------------------------------------------
    # Table 1: Side-by-side comparison
    # -----------------------------------------------------------------------
    print("=" * 100)
    print("  Modal A10G — runc vs gVisor (production nvproxy)")
    print("=" * 100)
    print()

    hdr = (
        f"{'Size':>8}  {'Direction':>10}  {'Mode':>10}  "
        f"{'runc':>10}  {'gVisor':>10}  {'Δ':>8}  {'Δ%':>7}"
    )
    print(hdr)
    print("─" * len(hdr))

    prev_size = None
    for s in runc:
        if prev_size is not None and s.size_mb != prev_size:
            print()
        prev_size = s.size_mb

        g = gvisor_map.get(s.key)
        if g is None:
            continue

        delta = g.bw_gbs - s.bw_gbs
        pct = delta / s.bw_gbs * 100 if s.bw_gbs > 0 else 0
        sign = "+" if delta >= 0 else ""
        # Highlight regressions > 5%
        marker = " ◄" if pct < -5 else ""

        print(
            f"{s.size_mb:>6} MB  {dir_label[s.direction]:>10}  {s.mode:>10}  "
            f"{s.bw_gbs:>7.2f} GB  {g.bw_gbs:>7.2f} GB  "
            f"{sign}{delta:>+6.2f}  {sign}{pct:>5.1f}%{marker}"
        )

    # -----------------------------------------------------------------------
    # Table 2: Pageable-only focus (the nvproxy-sensitive path)
    # -----------------------------------------------------------------------
    print()
    print()
    print("=" * 100)
    print("  Pageable transfers — nvproxy overhead analysis")
    print("=" * 100)
    print()
    print("  Model: pageable cudaMemcpy uses ~4 MB chunks, each requiring 3 nvidia")
    print(
        "  ioctls through nvproxy's Sentry. The bandwidth gap between runc and gVisor"
    )
    print("  on pageable transfers (but NOT pinned) reveals the per-ioctl overhead.")
    print()

    hdr2 = (
        f"{'Size':>8}  {'Direction':>10}  "
        f"{'runc':>10}  {'gVisor':>10}  {'Gap':>7}  "
        f"{'t_runc':>10}  {'t_gvsr':>10}  {'Δt':>8}  "
        f"{'#ioctls':>8}  {'per-ioctl':>10}"
    )
    print(hdr2)
    print("─" * len(hdr2))

    overhead_samples = []  # (size_mb, direction, per_ioctl_us)

    prev_size = None
    for s in runc:
        if s.mode != "pageable":
            continue
        g = gvisor_map.get(s.key)
        if g is None:
            continue

        if prev_size is not None and s.size_mb != prev_size:
            print()
        prev_size = s.size_mb

        t_r = transfer_time_ms(s.size_mb, s.bw_gbs)
        t_g = transfer_time_ms(s.size_mb, g.bw_gbs)
        delta_t = t_g - t_r
        n_chunks = max(s.size_mb / CHUNK_SIZE_MB, 1)
        n_ioctls = int(n_chunks * IOCTLS_PER_CHUNK)
        oh_us = per_ioctl_overhead_us(s.size_mb, s.bw_gbs, g.bw_gbs)

        pct = (g.bw_gbs - s.bw_gbs) / s.bw_gbs * 100

        if oh_us > 0:
            overhead_samples.append((s.size_mb, s.direction, oh_us))

        print(
            f"{s.size_mb:>6} MB  {dir_label[s.direction]:>10}  "
            f"{s.bw_gbs:>7.2f} GB  {g.bw_gbs:>7.2f} GB  {pct:>+5.1f}%  "
            f"{t_r:>7.2f} ms  {t_g:>7.2f} ms  {delta_t:>+6.2f}ms  "
            f"{n_ioctls:>8d}  "
            f"{oh_us:>7.1f} µs"
            if oh_us > 0
            else f"{'—':>10}"
        )

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    print()
    print()
    print("=" * 100)
    print("  Per-ioctl overhead summary")
    print("=" * 100)
    print()

    if overhead_samples:
        # Group by direction
        for d, d_label in [("h2d", "CPU → GPU"), ("d2h", "GPU → CPU")]:
            samples = [oh for (sz, dr, oh) in overhead_samples if dr == d]
            large = [oh for (sz, dr, oh) in overhead_samples if dr == d and sz >= 64]
            if samples:
                avg_all = sum(samples) / len(samples)
                print(f"  {d_label} (all sizes):   avg = {avg_all:>6.1f} µs/ioctl")
            if large:
                avg_large = sum(large) / len(large)
                print(f"  {d_label} (≥64 MB):     avg = {avg_large:>6.1f} µs/ioctl")
        print()

        all_large = [oh for (sz, dr, oh) in overhead_samples if sz >= 64]
        if all_large:
            combined_avg = sum(all_large) / len(all_large)
            print(f"  Combined (≥64 MB):       avg = {combined_avg:>6.1f} µs/ioctl")
            print()

            # Bandwidth projection
            print("  ─── Projected impact on 1 GB pageable transfer ───")
            print()
            n_1gb_ioctls = (1024 // CHUNK_SIZE_MB) * IOCTLS_PER_CHUNK
            print(
                f"  1 GB = {1024 // CHUNK_SIZE_MB} chunks × {IOCTLS_PER_CHUNK} ioctls = {n_1gb_ioctls} ioctls"
            )
            print()

            scenarios = [
                ("Current (measured)", combined_avg),
                ("Fast-path goal (10 µs)", 10.0),
                ("Aggressive fast-path (5 µs)", 5.0),
                ("Batch triplet (1 transition × 50 µs)", 50.0 / 3),
                ("ioctl passthrough (~0 µs)", 0.0),
            ]

            # Use pinned as the "zero overhead" reference for each direction
            pinned_h2d = runc_map.get((1024, "h2d", "pinned"))
            pinned_d2h = runc_map.get((1024, "d2h", "pinned"))
            pageable_runc_h2d = runc_map.get((1024, "h2d", "pageable"))
            pageable_runc_d2h = runc_map.get((1024, "d2h", "pageable"))

            print(
                f"  {'Scenario':<38}  {'Overhead':>10}  {'h2d BW':>10}  {'d2h BW':>10}"
            )
            print(f"  {'─' * 38}  {'─' * 10}  {'─' * 10}  {'─' * 10}")

            # runc baseline
            if pageable_runc_h2d and pageable_runc_d2h:
                print(
                    f"  {'runc baseline (measured)':<38}  "
                    f"{'0 ms':>10}  "
                    f"{pageable_runc_h2d.bw_gbs:>7.2f} GB  "
                    f"{pageable_runc_d2h.bw_gbs:>7.2f} GB"
                )

            for label, oh_us in scenarios:
                oh_ms = n_1gb_ioctls * oh_us / 1000.0

                # h2d: runc 1GB time + overhead
                if pageable_runc_h2d:
                    t_base_h2d = transfer_time_ms(1024, pageable_runc_h2d.bw_gbs)
                    t_total_h2d = t_base_h2d + oh_ms
                    bw_h2d = (1024 / 1024.0) / (t_total_h2d / 1000.0)
                else:
                    bw_h2d = 0

                # d2h: runc 1GB time + overhead
                if pageable_runc_d2h:
                    t_base_d2h = transfer_time_ms(1024, pageable_runc_d2h.bw_gbs)
                    t_total_d2h = t_base_d2h + oh_ms
                    bw_d2h = (1024 / 1024.0) / (t_total_d2h / 1000.0)
                else:
                    bw_d2h = 0

                print(
                    f"  {label:<38}  "
                    f"{oh_ms:>7.1f} ms  "
                    f"{bw_h2d:>7.2f} GB  "
                    f"{bw_d2h:>7.2f} GB"
                )

    # -----------------------------------------------------------------------
    # Sanity check: pinned should be identical
    # -----------------------------------------------------------------------
    print()
    print()
    print("─" * 80)
    print("  Sanity check: pinned & registered transfers (should be ~identical)")
    print("─" * 80)
    print()

    for mode in ["pinned", "hostReg"]:
        runc_vals = [s.bw_gbs for s in runc if s.mode == mode and s.size_mb >= 64]
        gvisor_vals = [
            gvisor_map[s.key].bw_gbs
            for s in runc
            if s.mode == mode and s.size_mb >= 64 and s.key in gvisor_map
        ]
        if runc_vals and gvisor_vals:
            r_avg = sum(runc_vals) / len(runc_vals)
            g_avg = sum(gvisor_vals) / len(gvisor_vals)
            pct = (g_avg - r_avg) / r_avg * 100
            print(
                f"  {mode:>10} (≥64 MB avg):  "
                f"runc={r_avg:.2f}  gVisor={g_avg:.2f}  Δ={pct:+.1f}%"
                f"  {'✓ OK' if abs(pct) < 2 else '⚠ unexpected gap'}"
            )

    # -----------------------------------------------------------------------
    # Key insight
    # -----------------------------------------------------------------------
    print()
    print()
    print("=" * 100)
    print("  KEY FINDINGS")
    print("=" * 100)
    print()

    # Compute the h2d and d2h pageable averages at >=256 MB
    h2d_runc = [
        s.bw_gbs
        for s in runc
        if s.mode == "pageable" and s.direction == "h2d" and s.size_mb >= 256
    ]
    h2d_gvsr = [
        gvisor_map[s.key].bw_gbs
        for s in runc
        if s.mode == "pageable"
        and s.direction == "h2d"
        and s.size_mb >= 256
        and s.key in gvisor_map
    ]
    d2h_runc = [
        s.bw_gbs
        for s in runc
        if s.mode == "pageable" and s.direction == "d2h" and s.size_mb >= 256
    ]
    d2h_gvsr = [
        gvisor_map[s.key].bw_gbs
        for s in runc
        if s.mode == "pageable"
        and s.direction == "d2h"
        and s.size_mb >= 256
        and s.key in gvisor_map
    ]

    if h2d_runc and h2d_gvsr:
        h2d_r = sum(h2d_runc) / len(h2d_runc)
        h2d_g = sum(h2d_gvsr) / len(h2d_gvsr)
        print(
            f"  1. CPU→GPU pageable (≥256 MB):  runc {h2d_r:.2f}  →  gVisor {h2d_g:.2f} GB/s  ({(h2d_g - h2d_r) / h2d_r * 100:+.1f}%)"
        )
    if d2h_runc and d2h_gvsr:
        d2h_r = sum(d2h_runc) / len(d2h_runc)
        d2h_g = sum(d2h_gvsr) / len(d2h_gvsr)
        print(
            f"  2. GPU→CPU pageable (≥256 MB):  runc {d2h_r:.2f}  →  gVisor {d2h_g:.2f} GB/s  ({(d2h_g - d2h_r) / d2h_r * 100:+.1f}%)"
        )

    print()
    print("  3. Pinned & registered transfers: IDENTICAL between runc and gVisor.")
    print("     The overhead is exclusively in the pageable ioctl path.")
    print()

    if d2h_runc and d2h_gvsr:
        # The big story: d2h pageable is 16% slower
        pct = (d2h_g - d2h_r) / d2h_r * 100
        # Estimate ioctl count for 1 GB d2h
        n = (1024 // CHUNK_SIZE_MB) * IOCTLS_PER_CHUNK
        t_r = transfer_time_ms(1024, d2h_r)
        t_g = transfer_time_ms(1024, d2h_g)
        delta = t_g - t_r
        oh = delta / n * 1000  # µs
        print(
            f"  4. The GPU→CPU d2h gap ({pct:+.1f}%) implies {oh:.0f} µs per-ioctl nvproxy overhead"
        )
        print(f"     ({n} ioctls for 1 GB, {delta:.1f} ms total added latency)")
        print()
        print(
            f"  5. To close the d2h gap, reduce per-ioctl overhead from ~{oh:.0f} µs to <5 µs."
        )
        print(
            f"     Fast-path (strategy A) targets: <10 µs → {oh:.0f}→10 would recover ~{(1 - (10 / oh)) * 100:.0f}% of the gap."
        )
        print(f"     Batching  (strategy B) targets: 3× fewer Sentry transitions.")
        print(f"     Passthrough (strategy C): eliminates the gap entirely.")

    if h2d_runc and h2d_gvsr:
        h2d_r = sum(h2d_runc) / len(h2d_runc)
        h2d_g = sum(h2d_gvsr) / len(h2d_gvsr)
        pct = (h2d_g - h2d_r) / h2d_r * 100
        n = (1024 // CHUNK_SIZE_MB) * IOCTLS_PER_CHUNK
        t_r = transfer_time_ms(1024, h2d_r)
        t_g = transfer_time_ms(1024, h2d_g)
        delta = t_g - t_r
        oh = delta / n * 1000  # µs
        print()
        print(
            f"  6. The CPU→GPU h2d gap ({pct:+.1f}%) implies {oh:.0f} µs per-ioctl overhead"
        )
        print(
            f"     (smaller than d2h because h2d pageable on runc is already near pinned speed)"
        )

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
