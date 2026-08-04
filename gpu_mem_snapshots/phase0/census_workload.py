#!/usr/bin/env python3
"""Phase 0 measurement #3 workload: a minimal, fabric-free, single-GPU CUDA
process to checkpoint/restore under gVisor, so the nvproxy object-graph census
(logged by state_cuda.go around the cuda-checkpoint phases) can be collected.

Allocates device memory through BOTH allocators whose RM objects matter to the
multicast design:
  * cuMemAlloc            -- classic allocation
  * cuMemCreate + cuMemMap -- VMM physical allocation, the same RM object kind
                              (NV01_MEMORY_LOCAL_USER) that cuMulticastBindMem
                              (NV00FD_CTRL_CMD_ATTACH_MEM) references.

Then loops forever verifying both patterns, writing one status line per
iteration to --status-file. Restore detection: gVisor restores the sandbox's
clocks, so a wall-vs-monotonic jump does NOT occur (unlike native CRIU);
instead the runner marks the restore explicitly by creating --restored-file
via `runsc exec` after `runsc restore` returns.

Run under runsc (see run_census.sh). Exit code: only on failure.
"""

import argparse
import ctypes
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cuda as cu

PATTERN_A = 0xA110CA7E  # cuMemAlloc region
PATTERN_B = 0xB1D0FEED  # cuMemCreate/cuMemMap region


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--size", type=lambda s: int(s, 0), default=64 << 20)
    ap.add_argument("--status-file", default="/tmp/status")
    ap.add_argument("--restored-file", default="/tmp/restored")
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    def status(line):
        tmp = args.status_file + ".tmp"
        with open(tmp, "w") as f:
            f.write(line + "\n")
        os.replace(tmp, args.status_file)
        print(line, flush=True)

    dev = cu.init_device(args.gpu)

    # cuMemAlloc region.
    va_a = ctypes.c_ulonglong()
    cu.call("cuMemAlloc_v2", ctypes.byref(va_a), args.size)
    n_u32_a = args.size // 4
    cu.memset_u32(va_a.value, PATTERN_A, n_u32_a)

    # cuMemCreate/cuMemMap (VMM) region, no export handle types: fabric-free.
    prop = cu.alloc_prop(dev, handle_types=0)
    gran = cu.alloc_granularity(prop)
    vmm_size = max(gran, args.size - args.size % gran or gran)
    handle = cu.mem_create(vmm_size, prop)
    va_b = cu.reserve_map_rw(handle, vmm_size, dev)
    n_u32_b = vmm_size // 4
    cu.memset_u32(va_b, PATTERN_B, n_u32_b)
    cu.call("cuCtxSynchronize")

    status(f"READY pid={os.getpid()} gpu={args.gpu} "
           f"va_a={va_a.value:#x} va_b={va_b:#x} vmm_size={vmm_size:#x}")

    it, failures, restored = 0, 0, False
    while True:
        it += 1
        if not restored and os.path.exists(args.restored_file):
            restored = True
            status(f"RESTORE-DETECTED iter={it} (marker file)")
            time.sleep(0.5)

        errs = []
        got_a = cu.read_u32(va_a.value, 256)
        if any(g != PATTERN_A for g in got_a):
            errs.append(f"cuMemAlloc pattern lost (got {got_a[0]:#x})")
        got_b = cu.read_u32(va_b, 256)
        if any(g != PATTERN_B for g in got_b):
            errs.append(f"cuMemCreate pattern lost (got {got_b[0]:#x})")
        # Tail of both regions too.
        if cu.read_u32(va_a.value + (n_u32_a - 4) * 4, 4) != [PATTERN_A] * 4:
            errs.append("cuMemAlloc tail lost")
        if cu.read_u32(va_b + (n_u32_b - 4) * 4, 4) != [PATTERN_B] * 4:
            errs.append("cuMemCreate tail lost")

        tag = "post-restore" if restored else "pre-checkpoint"
        if errs:
            failures += 1
            status(f"iter={it} {tag} FAIL: " + "; ".join(errs) +
                   f" failures={failures}")
        else:
            status(f"iter={it} {tag} pass failures={failures} "
                   f"restored={restored}")
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
