"""Minimal ctypes bindings for the CUDA driver API (libcuda.so.1).

Used by the Phase 0 measurement programs (ipc_taint.py, attach_blocking.py).
Deliberately toolkit-free: no cuda.h, no nvcc, no torch -- just libcuda, so it
runs on a bare driver install.
"""

import ctypes
import os
import sys
import threading
import time

_lib = ctypes.CDLL("libcuda.so.1")

# ---------------------------------------------------------------------------
# constants (from cuda.h)
# ---------------------------------------------------------------------------

CUDA_SUCCESS = 0

CU_MEM_ALLOCATION_TYPE_PINNED = 0x1
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1
CU_MEM_LOCATION_TYPE_DEVICE = 0x1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3

CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0x0
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 0x1
CU_MULTICAST_GRANULARITY_MINIMUM = 0x0
CU_MULTICAST_GRANULARITY_RECOMMENDED = 0x1

CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103
CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132


# ---------------------------------------------------------------------------
# structs (layouts from cuda.h, x86_64)
# ---------------------------------------------------------------------------

class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CUmemAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _CUmemAllocFlags),
    ]


class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", CUmemLocation), ("flags", ctypes.c_int)]


class CUmulticastObjectProp(ctypes.Structure):
    _fields_ = [
        ("numDevices", ctypes.c_uint),
        ("size", ctypes.c_size_t),
        ("handleTypes", ctypes.c_ulonglong),
        ("flags", ctypes.c_ulonglong),
    ]


# ---------------------------------------------------------------------------
# prototypes
# ---------------------------------------------------------------------------

_c_int_p = ctypes.POINTER(ctypes.c_int)
_c_size_p = ctypes.POINTER(ctypes.c_size_t)
_c_ull = ctypes.c_ulonglong
_c_ull_p = ctypes.POINTER(_c_ull)

_protos = {
    "cuInit": (ctypes.c_uint,),
    "cuGetErrorName": (ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)),
    "cuGetErrorString": (ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)),
    "cuDeviceGet": (_c_int_p, ctypes.c_int),
    "cuDeviceGetAttribute": (_c_int_p, ctypes.c_int, ctypes.c_int),
    "cuDevicePrimaryCtxRetain": (ctypes.POINTER(ctypes.c_void_p), ctypes.c_int),
    "cuCtxSetCurrent": (ctypes.c_void_p,),
    "cuCtxSynchronize": (),
    "cuMemAlloc_v2": (_c_ull_p, ctypes.c_size_t),
    "cuMemGetAllocationGranularity": (
        _c_size_p, ctypes.POINTER(CUmemAllocationProp), ctypes.c_int),
    "cuMemCreate": (
        _c_ull_p, ctypes.c_size_t, ctypes.POINTER(CUmemAllocationProp), _c_ull),
    "cuMemRelease": (_c_ull,),
    "cuMemAddressReserve": (_c_ull_p, ctypes.c_size_t, ctypes.c_size_t, _c_ull, _c_ull),
    "cuMemAddressFree": (_c_ull, ctypes.c_size_t),
    "cuMemMap": (_c_ull, ctypes.c_size_t, ctypes.c_size_t, _c_ull, _c_ull),
    "cuMemUnmap": (_c_ull, ctypes.c_size_t),
    "cuMemSetAccess": (_c_ull, ctypes.c_size_t, ctypes.POINTER(CUmemAccessDesc), ctypes.c_size_t),
    "cuMemExportToShareableHandle": (ctypes.c_void_p, _c_ull, ctypes.c_int, _c_ull),
    "cuMemImportFromShareableHandle": (_c_ull_p, ctypes.c_void_p, ctypes.c_int),
    "cuMemsetD32_v2": (_c_ull, ctypes.c_uint, ctypes.c_size_t),
    "cuMemcpyDtoH_v2": (ctypes.c_void_p, _c_ull, ctypes.c_size_t),
    "cuMulticastCreate": (_c_ull_p, ctypes.POINTER(CUmulticastObjectProp)),
    "cuMulticastAddDevice": (_c_ull, ctypes.c_int),
    "cuMulticastBindMem": (_c_ull, ctypes.c_size_t, _c_ull, ctypes.c_size_t, ctypes.c_size_t, _c_ull),
    "cuMulticastUnbind": (_c_ull, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t),
    "cuMulticastGetGranularity": (
        _c_size_p, ctypes.POINTER(CUmulticastObjectProp), ctypes.c_int),
    # Streams + CUDA graphs (for graph-capture checkpoint/restore validation).
    "cuStreamCreate": (ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint),
    "cuStreamSynchronize": (ctypes.c_void_p,),
    "cuStreamBeginCapture_v2": (ctypes.c_void_p, ctypes.c_int),
    "cuStreamEndCapture": (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)),
    "cuGraphInstantiateWithFlags": (
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, _c_ull),
    "cuGraphLaunch": (ctypes.c_void_p, ctypes.c_void_p),
    "cuGraphDestroy": (ctypes.c_void_p,),
    "cuMemsetD32Async": (_c_ull, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p),
    # Module + kernel launch (for a real SM peer-read over NVLink, loaded
    # from PTX so no nvcc is required).
    "cuModuleLoadData": (ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p),
    "cuModuleGetFunction": (
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p),
    "cuLaunchKernel": (
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p)),
}

# CUstreamCaptureMode
CU_STREAM_CAPTURE_MODE_GLOBAL = 0
CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1
CU_STREAM_CAPTURE_MODE_RELAXED = 2

for _name, _args in _protos.items():
    try:
        _fn = getattr(_lib, _name)
    except AttributeError:
        continue  # older drivers may lack multicast entry points
    _fn.restype = ctypes.c_int
    _fn.argtypes = list(_args)


class CudaError(Exception):
    def __init__(self, fn, code):
        name = ctypes.c_char_p()
        desc = ctypes.c_char_p()
        _lib.cuGetErrorName(code, ctypes.byref(name))
        _lib.cuGetErrorString(code, ctypes.byref(desc))
        self.code = code
        self.fn = fn
        self.name = (name.value or b"?").decode()
        super().__init__(
            f"{fn} failed: {self.name} ({code}): {(desc.value or b'?').decode()}")


def call(fn, *args):
    """Invoke a driver API entry point, raising CudaError on failure."""
    f = getattr(_lib, fn, None)
    if f is None:
        raise CudaError(fn, 3)  # CUDA_ERROR_NOT_INITIALIZED stand-in
    rc = f(*args)
    if rc != CUDA_SUCCESS:
        raise CudaError(fn, rc)
    return rc


def has(fn):
    return getattr(_lib, fn, None) is not None


# ---------------------------------------------------------------------------
# convenience wrappers
# ---------------------------------------------------------------------------

def init_device(ordinal):
    """cuInit + primary context on device `ordinal`. Returns CUdevice."""
    call("cuInit", 0)
    dev = ctypes.c_int()
    call("cuDeviceGet", ctypes.byref(dev), ordinal)
    ctx = ctypes.c_void_p()
    call("cuDevicePrimaryCtxRetain", ctypes.byref(ctx), dev.value)
    call("cuCtxSetCurrent", ctx)
    return dev.value


def device_attr(attr, dev):
    v = ctypes.c_int()
    call("cuDeviceGetAttribute", ctypes.byref(v), attr, dev)
    return v.value


def alloc_prop(dev, handle_types=CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR):
    p = CUmemAllocationProp()
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED
    p.requestedHandleTypes = handle_types
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    p.location.id = dev
    return p


def alloc_granularity(prop):
    g = ctypes.c_size_t()
    call("cuMemGetAllocationGranularity", ctypes.byref(g), ctypes.byref(prop),
         CU_MEM_ALLOC_GRANULARITY_MINIMUM)
    return g.value


def mem_create(size, prop):
    h = _c_ull()
    call("cuMemCreate", ctypes.byref(h), size, ctypes.byref(prop), 0)
    return h.value


def reserve_map_rw(handle, size, dev, va_hint=0):
    """cuMemAddressReserve + cuMemMap + cuMemSetAccess(RW on dev). Returns VA."""
    va = _c_ull()
    call("cuMemAddressReserve", ctypes.byref(va), size, 0, va_hint, 0)
    call("cuMemMap", va.value, size, 0, handle, 0)
    d = CUmemAccessDesc()
    d.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    d.location.id = dev
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    call("cuMemSetAccess", va.value, size, ctypes.byref(d), 1)
    return va.value


def export_posix_fd(handle):
    fd = ctypes.c_int(-1)
    call("cuMemExportToShareableHandle", ctypes.byref(fd), handle,
         CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0)
    return fd.value


def import_posix_fd(fd):
    h = _c_ull()
    # For POSIX FD imports, osHandle is the fd *value* cast to void*.
    call("cuMemImportFromShareableHandle", ctypes.byref(h),
         ctypes.c_void_p(fd), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
    return h.value


def memset_u32(va, value, count):
    call("cuMemsetD32_v2", va, value, count)
    call("cuCtxSynchronize")


def read_u32(va, count):
    buf = (ctypes.c_uint * count)()
    call("cuMemcpyDtoH_v2", buf, va, count * 4)
    return list(buf)


# ---------------------------------------------------------------------------
# watchdog: turn indefinite hangs into clean, attributable failures.
# TASK.md: multicast attach failure manifests as a hang, not an error.
# ---------------------------------------------------------------------------

WATCHDOG_EXIT_CODE = 3


class watchdog:
    """Context manager: if the body doesn't finish within `secs`, print a loud
    message and hard-exit the process with WATCHDOG_EXIT_CODE (3)."""

    def __init__(self, what, secs, tag=""):
        self.what = what
        self.secs = secs
        self.tag = tag

    def _fire(self):
        print(f"{self.tag} WATCHDOG: {self.what!r} still blocked after "
              f"{self.secs}s -- HANG (exit {WATCHDOG_EXIT_CODE})", flush=True)
        os._exit(WATCHDOG_EXIT_CODE)

    def __enter__(self):
        self.timer = threading.Timer(self.secs, self._fire)
        self.timer.daemon = True
        self.timer.start()
        return self

    def __exit__(self, *exc):
        self.timer.cancel()
        return False


def timed(what, fn, tag=""):
    """Run fn(), print and return (elapsed_seconds, result)."""
    t0 = time.monotonic()
    result = fn()
    dt = time.monotonic() - t0
    print(f"{tag} {what}: {dt:.3f}s", flush=True)
    return dt, result


# ---------------------------------------------------------------------------
# tiny line-oriented message protocol over a socketpair (also carries FDs)
# ---------------------------------------------------------------------------

def close_all(*socks):
    """Close inherited socketpair ends a process doesn't own, so that a
    crashing process produces EOF (not a hang) on its peers' sockets."""
    for s in socks:
        try:
            s.close()
        except OSError:
            pass


def send_msg(sock, msg, fds=()):
    import socket as _s
    _s.send_fds(sock, [msg.encode() + b"\n"], list(fds))


def recv_msg(sock, expect=None, maxfds=1):
    msg, fds, _, _ = __import__("socket").recv_fds(sock, 4096, maxfds)
    if not msg:
        raise EOFError("peer closed socket (it probably crashed; see its log)")
    msg = msg.decode().strip()
    if expect is not None and not msg.startswith(expect):
        raise RuntimeError(f"expected message {expect!r}, got {msg!r}")
    return msg, fds


def log(tag, msg):
    print(f"{tag} {msg}", flush=True)
    sys.stdout.flush()
