#!/usr/bin/env python3
"""Step-0 probe (mcshim/README.md): do distinct exported objects get distinct
fd identities (st_dev:st_ino), natively and under gVisor?

Creates two multicast groups + one shareable UC allocation, exports each via
cuMemExportToShareableHandle, and fstats the three fds. If the keys are
distinct, the mcshim's existing fd-identity rendezvous scales and no nvproxy
identity oracle is needed (in that environment). Natively all NVIDIA export
fds were observed to share one inode (7:55e).

Also prints readlink(/proc/self/fd/N) to show what the fd actually is, and
reads /proc/self/fdinfo/N: under gVisor with the nvproxy identity oracle,
exported-object fds carry an `nvproxy_exported_object: client=... object=...`
line whose (client, object) pair is the scalable identity. When present, the
probe keys on it instead of fstat.
"""

import ctypes
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cuda as cu


def oracle_of(fd):
    """Returns the nvproxy_exported_object fdinfo line's value, or None."""
    try:
        with open(f"/proc/self/fdinfo/{fd}") as f:
            for line in f:
                if line.startswith("nvproxy_exported_object:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def key_of(fd, label):
    st = os.fstat(fd)
    try:
        link = os.readlink(f"/proc/self/fd/{fd}")
    except OSError:
        link = "?"
    oracle = oracle_of(fd)
    print(f"  {label:10s} fd={fd:<3d} st_dev={st.st_dev:#x} "
          f"st_ino={st.st_ino:#x} -> {link} oracle=[{oracle}]", flush=True)
    return oracle if oracle is not None else (st.st_dev, st.st_ino)


def main():
    cu.call("cuInit", 0)
    dev = ctypes.c_int()
    cu.call("cuDeviceGet", ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p()
    cu.call("cuDevicePrimaryCtxRetain", ctypes.byref(ctx), dev.value)
    cu.call("cuCtxSetCurrent", ctx)

    prop = cu.CUmulticastObjectProp()
    prop.numDevices = 2
    prop.handleTypes = cu.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    gran = ctypes.c_size_t()
    cu.call("cuMulticastGetGranularity", ctypes.byref(gran), ctypes.byref(prop),
            cu.CU_MULTICAST_GRANULARITY_RECOMMENDED)
    prop.size = gran.value

    print(f"[probe] pid={os.getpid()} exporting 2 MC groups + 1 UC alloc:",
          flush=True)
    keys = []

    for i in range(2):
        mc = ctypes.c_ulonglong()
        cu.call("cuMulticastCreate", ctypes.byref(mc), ctypes.byref(prop))
        fd = cu.export_posix_fd(mc.value)
        keys.append(key_of(fd, f"mc-group-{i}"))

    p = cu.alloc_prop(dev.value)
    ucsize = max(2 << 20, cu.alloc_granularity(p))
    uch = cu.mem_create(ucsize, p)
    fd = cu.export_posix_fd(uch)
    keys.append(key_of(fd, "uc-alloc"))

    distinct = len(set(keys)) == len(keys)
    print(f"[probe] keys distinct: {distinct} "
          f"({len(set(keys))}/{len(keys)} unique)", flush=True)

    # Cross-process leg: an SCM_RIGHTS recipient must observe the SAME
    # identity for a received fd (same FileDescription -> same fdinfo line).
    # This is what lets an importer match a received fd to the exporter's
    # object with no import-side bookkeeping anywhere.
    mc = ctypes.c_ulonglong()
    cu.call("cuMulticastCreate", ctypes.byref(mc), ctypes.byref(prop))
    xfd = cu.export_posix_fd(mc.value)
    parent_key = oracle_of(xfd)
    a, b = socket.socketpair()
    pid = os.fork()
    if pid == 0:
        a.close()
        _, fds, _, _ = socket.recv_fds(b, 16, 1)
        child_key = oracle_of(fds[0])
        match = child_key == parent_key and child_key is not None
        print(f"  child      fd={fds[0]:<3d} oracle=[{child_key}] "
              f"match={match}", flush=True)
        os._exit(0 if match else 1)
    b.close()
    socket.send_fds(a, [b"x"], [xfd])
    _, st = os.waitpid(pid, 0)
    xproc = os.waitstatus_to_exitcode(st) == 0
    print(f"[probe] cross-process identity match: {xproc} "
          f"(parent=[{parent_key}])", flush=True)
    distinct = distinct and (parent_key is None or xproc)
    print(f"[probe] VERDICT: {'DISTINCT — fd identity scales here' if distinct else 'COLLIDING — identity oracle (or alternative) required here'}",
          flush=True)
    return 0 if distinct else 2


if __name__ == "__main__":
    sys.exit(main())
