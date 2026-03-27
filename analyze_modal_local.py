#!/usr/bin/env python3
"""
Analyze Modal vs local gVisor pageable transfer results.

Compares runc and gVisor bandwidth across Modal production and local EC2,
computes per-ioctl overhead estimates, and identifies the root cause pattern.

Usage:
    python3 analyze_modal_local.py
"""

# ---------------------------------------------------------------------------
# Embedded data from benchmark runs
# ---------------------------------------------------------------------------

# Modal g5.12xlarge (24 runc CPUs, 21 gVisor CPUs)
# Instance type confirmed same as local dev instance
MODAL_SMALL = {
    "label": "Modal g5.12xlarge",
    "runc": {
        "h2d": {
            "pageable": {
                1: 7.14,
                4: 10.34,
                16: 11.68,
                64: 12.10,
                256: 12.19,
                512: 12.22,
                1024: 12.23,
            },
            "pinned": {
                1: 10.68,
                4: 12.04,
                16: 12.41,
                64: 12.51,
                256: 12.53,
                512: 12.54,
                1024: 12.54,
            },
        },
        "d2h": {
            "pageable": {
                1: 6.17,
                4: 9.91,
                16: 11.42,
                64: 10.61,
                256: 10.71,
                512: 10.72,
                1024: 10.74,
            },
            "pinned": {
                1: 10.71,
                4: 11.84,
                16: 12.19,
                64: 12.27,
                256: 12.30,
                512: 12.30,
                1024: 12.30,
            },
        },
    },
    "gvisor": {
        "h2d": {
            "pageable": {
                1: 7.05,
                4: 10.31,
                16: 11.40,
                64: 11.76,
                256: 12.09,
                512: 11.92,
                1024: 12.19,
            },
            "pinned": {
                1: 10.74,
                4: 12.03,
                16: 12.40,
                64: 12.51,
                256: 12.52,
                512: 12.54,
                1024: 12.54,
            },
        },
        "d2h": {
            "pageable": {
                1: 6.39,
                4: 9.91,
                16: 11.38,
                64: 5.04,
                256: 8.47,
                512: 5.08,
                1024: 10.57,
            },
            "pinned": {
                1: 10.77,
                4: 11.84,
                16: 12.18,
                64: 12.49,
                256: 12.47,
                512: 12.30,
                1024: 12.32,
            },
        },
    },
}

# Modal g5.48xlarge (96 runc CPUs, 80 gVisor CPUs)
MODAL_LARGE = {
    "label": "Modal g5.48xlarge",
    "runc": {
        "h2d": {
            "pageable": {
                1: 7.04,
                4: 10.03,
                16: 11.54,
                64: 12.03,
                256: 12.09,
                512: 12.17,
                1024: 11.90,
            },
            "pinned": {
                1: 10.58,
                4: 12.02,
                16: 12.22,
                64: 12.45,
                256: 12.50,
                512: 12.50,
                1024: 12.51,
            },
        },
        "d2h": {
            "pageable": {
                1: 5.89,
                4: 4.98,
                16: 4.70,
                64: 5.75,
                256: 5.74,
                512: 5.78,
                1024: 5.79,
            },
            "pinned": {
                1: 10.66,
                4: 11.82,
                16: 12.09,
                64: 12.24,
                256: 12.28,
                512: 12.27,
                1024: 12.26,
            },
        },
    },
    "gvisor": {
        "h2d": {
            "pageable": {
                1: 6.90,
                4: 10.29,
                16: 9.32,
                64: 9.24,
                256: 9.63,
                512: 10.46,
                1024: 10.67,
            },
            "pinned": {
                1: 10.52,
                4: 11.97,
                16: 12.38,
                64: 12.50,
                256: 12.49,
                512: 12.53,
                1024: 12.51,
            },
        },
        "d2h": {
            "pageable": {
                1: 3.48,
                4: 4.99,
                16: 10.28,
                64: 4.94,
                256: 5.04,
                512: 5.02,
                1024: 5.01,
            },
            "pinned": {
                1: 10.70,
                4: 11.72,
                16: 12.18,
                64: 12.24,
                256: 12.25,
                512: 12.29,
                1024: 12.30,
            },
        },
    },
}

# Local EC2 g5.12xlarge (48 CPUs, 1 NUMA node, PCIe Gen1 x8)
# runsc with Modal flags (--platform=systrap --directfs --overlay2=root:self etc.)
LOCAL = {
    "label": "Local g5.12xlarge (Modal flags)",
    "runc": {
        "h2d": {
            "pageable": {64: 12.14, 256: 12.22, 1024: 12.25},
            "pinned": {64: 12.51, 256: 12.54, 1024: 12.54},
        },
        "d2h": {
            "pageable": {64: 10.63, 256: 10.72, 1024: 10.80},
            "pinned": {64: 12.28, 256: 12.30, 1024: 12.31},
        },
    },
    "gvisor": {
        "h2d": {
            "pageable": {64: 11.91, 256: 12.15, 1024: 12.21},
            "pinned": {64: 12.51, 256: 12.54, 1024: 12.54},
        },
        "d2h": {
            "pageable": {64: 10.64, 256: 10.76, 1024: 10.78},
            "pinned": {64: 12.27, 256: 12.30, 1024: 12.31},
        },
    },
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_MB = 4
IOCTLS_PER_CHUNK = 3


def per_ioctl_us(size_mb, bw_base, bw_test):
    """Estimate per-ioctl overhead in microseconds from bandwidth gap."""
    if bw_base <= 0 or bw_test <= 0 or bw_test >= bw_base:
        return 0.0
    t_base = (size_mb / 1024.0) / bw_base * 1000.0  # ms
    t_test = (size_mb / 1024.0) / bw_test * 1000.0
    n_ioctls = max(size_mb / CHUNK_MB, 1) * IOCTLS_PER_CHUNK
    return (t_test - t_base) / n_ioctls * 1000.0  # us


def avg(vals):
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def print_section(title):
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)
    print()


def analyze_dataset(data):
    label = data["label"]
    sizes = sorted(data["runc"]["d2h"]["pageable"].keys())

    print(
        f"  {'Size':>6}  {'runc d2h':>10}  {'gV d2h':>10}  {'d2h gap':>8}  "
        f"{'runc h2d':>10}  {'gV h2d':>10}  {'h2d gap':>8}  "
        f"{'pinned ok?':>10}"
    )
    print(
        f"  {'----':>6}  {'--------':>10}  {'------':>10}  {'-------':>8}  "
        f"{'--------':>10}  {'------':>10}  {'-------':>8}  "
        f"{'----------':>10}"
    )

    for sz in sizes:
        r_d2h = data["runc"]["d2h"]["pageable"].get(sz)
        g_d2h = data["gvisor"]["d2h"]["pageable"].get(sz)
        r_h2d = data["runc"]["h2d"]["pageable"].get(sz)
        g_h2d = data["gvisor"]["h2d"]["pageable"].get(sz)
        r_pin = data["runc"]["d2h"]["pinned"].get(sz, 0)
        g_pin = data["gvisor"]["d2h"]["pinned"].get(sz, 0)

        if r_d2h and g_d2h:
            d2h_gap = (g_d2h - r_d2h) / r_d2h * 100
        else:
            d2h_gap = 0

        if r_h2d and g_h2d:
            h2d_gap = (g_h2d - r_h2d) / r_h2d * 100
        else:
            h2d_gap = 0

        if r_pin and g_pin:
            pin_gap = abs(g_pin - r_pin) / r_pin * 100
            pin_ok = "yes" if pin_gap < 2 else f"NO ({pin_gap:+.1f}%)"
        else:
            pin_ok = "--"

        marker = " <<<" if d2h_gap < -10 else ""

        r_d2h_s = f"{r_d2h:>7.2f} GB" if r_d2h else f"{'--':>10}"
        g_d2h_s = f"{g_d2h:>7.2f} GB" if g_d2h else f"{'--':>10}"
        r_h2d_s = f"{r_h2d:>7.2f} GB" if r_h2d else f"{'--':>10}"
        g_h2d_s = f"{g_h2d:>7.2f} GB" if g_h2d else f"{'--':>10}"

        print(
            f"  {sz:>4} MB  {r_d2h_s}  {g_d2h_s}  {d2h_gap:>+6.1f}%{marker}"
            f"  {r_h2d_s}  {g_h2d_s}  {h2d_gap:>+6.1f}%"
            f"  {pin_ok:>10}"
        )


def main():
    print("=" * 78)
    print("  MODAL vs LOCAL — Pageable Memory Bandwidth Analysis")
    print("  Same GPU (A10G), Same driver (580.95.05), Same runsc binary")
    print("=" * 78)

    # -----------------------------------------------------------------------
    # Table 1: Modal g5.12xlarge (same instance type as local)
    # -----------------------------------------------------------------------
    print_section("Modal g5.12xlarge — runc vs gVisor")
    analyze_dataset(MODAL_SMALL)

    # -----------------------------------------------------------------------
    # Table 2: Modal g5.48xlarge
    # -----------------------------------------------------------------------
    print_section("Modal g5.48xlarge — runc vs gVisor")
    analyze_dataset(MODAL_LARGE)

    # -----------------------------------------------------------------------
    # Table 3: Local g5.12xlarge with Modal flags
    # -----------------------------------------------------------------------
    print_section("Local g5.12xlarge (Modal flags) — runc vs gVisor")
    analyze_dataset(LOCAL)

    # -----------------------------------------------------------------------
    # Table 4: runc baseline comparison (is the hardware the same?)
    # -----------------------------------------------------------------------
    print_section("runc d2h pageable — Modal vs Local (hardware fingerprint)")

    sizes = [64, 256, 1024]
    print(
        f"  {'Size':>6}  {'Modal g5.12xl':>14}  {'Modal g5.48xl':>14}  {'Local g5.12xl':>14}"
    )
    print(f"  {'----':>6}  {'-' * 14}  {'-' * 14}  {'-' * 14}")
    for sz in sizes:
        ms = MODAL_SMALL["runc"]["d2h"]["pageable"].get(sz)
        ml = MODAL_LARGE["runc"]["d2h"]["pageable"].get(sz)
        lo = LOCAL["runc"]["d2h"]["pageable"].get(sz)
        ms_s = f"{ms:>11.2f} GB" if ms else f"{'--':>14}"
        ml_s = f"{ml:>11.2f} GB" if ml else f"{'--':>14}"
        lo_s = f"{lo:>11.2f} GB" if lo else f"{'--':>14}"
        print(f"  {sz:>4} MB  {ms_s}  {ml_s}  {lo_s}")

    print()
    print("  Modal g5.12xl runc matches Local g5.12xl runc -> SAME HARDWARE")
    print("  Modal g5.48xl runc is ~2x slower d2h -> DIFFERENT (NUMA/PCIe)")

    # -----------------------------------------------------------------------
    # Table 5: The bimodal pattern on Modal g5.12xlarge
    # -----------------------------------------------------------------------
    print_section("THE BIMODAL PATTERN — Modal g5.12xlarge gVisor d2h")

    sizes_all = sorted(MODAL_SMALL["runc"]["d2h"]["pageable"].keys())
    print(
        f"  {'Size':>6}  {'runc d2h':>10}  {'gVisor d2h':>12}  {'gap':>8}  {'pattern':>20}"
    )
    print(
        f"  {'----':>6}  {'--------':>10}  {'----------':>12}  {'---':>8}  {'-------':>20}"
    )

    for sz in sizes_all:
        r = MODAL_SMALL["runc"]["d2h"]["pageable"][sz]
        g = MODAL_SMALL["gvisor"]["d2h"]["pageable"][sz]
        gap = (g - r) / r * 100

        if gap > -5:
            pattern = "OK (matches runc)"
        elif g < 6:
            pattern = "SLOW PATH (~5 GB/s)"
        else:
            pattern = f"degraded"

        print(f"  {sz:>4} MB  {r:>7.2f} GB  {g:>9.2f} GB  {gap:>+6.1f}%  {pattern:>20}")

    # -----------------------------------------------------------------------
    # Key findings
    # -----------------------------------------------------------------------
    print_section("KEY FINDINGS")

    print("""\
  1. HARDWARE IS IDENTICAL for g5.12xlarge (runc proves it).
     Modal runc d2h pageable matches local runc d2h pageable within 1%.

  2. gVisor d2h pageable on Modal shows a BIMODAL pattern:
     - Sizes <= 16 MB: gVisor MATCHES runc (no overhead)
     - Sizes 64 and 512 MB: gVisor drops to ~5 GB/s (52% slower!)
     - Size 256 MB: intermediate (8.47 GB/s, 21% slower)
     - Size 1024 MB: gVisor RECOVERS to match runc (10.57 vs 10.74)

  3. This is NOT per-ioctl overhead. If it were:
     - The gap would scale linearly with transfer size
     - 1024 MB would have the LARGEST gap (most ioctls)
     - Instead 1024 MB has the SMALLEST gap (2%)

  4. This IS a memory-management or page-pinning issue:
     - Only d2h (GPU->CPU) is affected, not h2d
     - Pinned transfers are unaffected (proves PCIe is fine)
     - The bimodal pattern suggests a threshold where gVisor's
       Sentry memory layout causes the nvidia driver to use a
       different (slower) DMA page-pinning strategy

  5. We CANNOT reproduce it locally with identical flags because:
     - The trigger is not in runsc flags
     - It's in Modal's host environment: cgroup memory limits,
       host kernel /proc/sys/vm/* settings, or the way Modal's
       orchestration layer sets up the Sentry process memory

  6. The g5.48xlarge shows a DIFFERENT pattern:
     - runc itself is 2x slower d2h (5.79 vs 10.74 GB/s)
     - This is a NUMA/PCIe topology issue on the larger instance
     - The gVisor gap on top of that is a separate compounding effect""")

    # -----------------------------------------------------------------------
    # Hypotheses for the bimodal pattern
    # -----------------------------------------------------------------------
    print_section("HYPOTHESES FOR THE 64/512 MB d2h SLOW PATH")

    print("""\
  The nvidia driver pins host pages differently depending on the transfer
  size and the host memory layout. For d2h (GPU writes to host), the
  driver calls get_user_pages() to pin the destination buffer, then
  programs the GPU DMA engine to write to those physical pages.

  Hypothesis A: Sentry address space layout
    gVisor's Sentry maps guest memory differently than a normal process.
    At certain allocation sizes, the virtual-to-physical mapping may
    cross boundaries that cause get_user_pages() to take a slow path
    (e.g., falling back from huge pages to small pages, or triggering
    IOMMU scatter-gather instead of contiguous DMA).

  Hypothesis B: cgroup memory pressure
    Modal likely runs containers with memory cgroup limits. Under
    memory pressure, the kernel's page reclaim can interfere with
    page pinning. The bimodal pattern (64 MB slow, 1024 MB fast)
    could reflect different reclaim behavior at different working
    set sizes relative to the cgroup limit.

  Hypothesis C: Transparent Huge Page (THP) interaction
    gVisor's memory allocator may produce different THP coverage
    at different sizes. If 64 MB allocations land in a region
    without THP backing, the nvidia driver pins 16K small pages
    instead of 32 huge pages, which is dramatically slower.
    The 1024 MB allocation may trigger THP collapse, recovering
    performance.

  Hypothesis D: CUDA driver staging buffer
    The CUDA driver may use an internal staging buffer for d2h
    pageable transfers. If gVisor's Sentry causes this staging
    buffer to be allocated in a suboptimal location (wrong NUMA
    node, non-huge-page-backed), certain transfer sizes that
    exceed the staging buffer trigger repeated re-pinning.

  TO INVESTIGATE:
    - Run `perf record -g` on the Sentry process during a d2h
      pageable transfer on Modal to see where time is spent
    - Check /proc/<sentry_pid>/smaps for THP coverage of the
      guest memory region used by the transfer
    - Compare /proc/sys/vm/* settings between Modal host and
      local EC2 (especially transparent_hugepage, zone_reclaim_mode,
      dirty_ratio)
    - Check cgroup memory.limit_in_bytes for the gVisor container
      on Modal vs locally (we run with unlimited memory locally)""")

    # -----------------------------------------------------------------------
    # Ioctl overhead estimate (for the g5.48xlarge where it's cleaner)
    # -----------------------------------------------------------------------
    print_section("PER-IOCTL OVERHEAD ESTIMATE (g5.48xlarge, cleaner signal)")

    sizes_large = [64, 256, 512, 1024]
    print(
        f"  {'Size':>6}  {'runc d2h':>10}  {'gVisor d2h':>12}  {'delta_ms':>10}  "
        f"{'#ioctls':>8}  {'per-ioctl':>10}"
    )
    print(
        f"  {'----':>6}  {'--------':>10}  {'----------':>12}  {'--------':>10}  "
        f"{'-------':>8}  {'---------':>10}"
    )

    overheads = []
    for sz in sizes_large:
        r = MODAL_LARGE["runc"]["d2h"]["pageable"][sz]
        g = MODAL_LARGE["gvisor"]["d2h"]["pageable"][sz]
        t_r = (sz / 1024.0) / r * 1000.0
        t_g = (sz / 1024.0) / g * 1000.0
        delta = t_g - t_r
        n = max(sz / CHUNK_MB, 1) * IOCTLS_PER_CHUNK
        oh = delta / n * 1000.0 if delta > 0 else 0
        overheads.append(oh)
        print(
            f"  {sz:>4} MB  {r:>7.2f} GB  {g:>9.2f} GB  {delta:>+7.1f} ms  "
            f"{int(n):>8}  {oh:>7.1f} us"
        )

    avg_oh = avg([o for o in overheads if o > 0])
    print(f"\n  Average per-ioctl overhead: {avg_oh:.1f} us")
    print(f"  (Note: on g5.48xlarge the runc baseline is already slow due")
    print(f"   to NUMA, so this conflates NUMA + nvproxy overhead)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print_section("SUMMARY")
    print("""\
  The pageable transfer performance gap between runc and gVisor on Modal
  has TWO distinct root causes depending on instance size:

  g5.12xlarge (same hardware as local dev):
    - runc baselines MATCH between Modal and local
    - gVisor shows a BIMODAL d2h slow path at 64/512 MB sizes
    - NOT reproducible locally -> caused by Modal host environment
    - NOT per-ioctl overhead (wrong scaling pattern)
    - Likely a page-pinning / memory-management interaction

  g5.48xlarge (large multi-NUMA instance):
    - runc itself is 2x slower d2h due to NUMA topology
    - gVisor adds ~35 us/ioctl on top (per-ioctl + NUMA for Sentry)
    - Compounds with the already-slow NUMA baseline

  The nvproxy fast-path optimization (sync.Pool, rmControlFast,
  feHandlerFast) saves ~1-3 us per ioctl in microbenchmarks but
  cannot address the primary bottleneck, which is in the
  page-pinning layer below nvproxy's ioctl dispatch.""")
    print()


if __name__ == "__main__":
    main()
