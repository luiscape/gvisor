"""Minimal ctypes bindings for NCCL (libnccl.so.2), including the
suspend/resume API (NCCL >= 2.29.7):

    ncclCommSuspend(comm, NCCL_SUSPEND_MEM)  # release dynamic GPU allocations
    ncclCommResume(comm)                     # bring them back
    ncclCommMemStats(comm, stat, &value)     # incl. suspend-related stats

Used by nccl_suspend_workload.py to validate the checkpoint/restore model:
suspend releases NVLS multicast objects at the libcuda level (keeping
libcuda's bookkeeping consistent for cuda-checkpoint), resume recreates them.

Library path: NCCL_LIB env var, else /opt/phase0/nccl/nvidia/nccl/lib/libnccl.so.2.
"""

import ctypes
import os

_DEFAULT_LIB = "/opt/phase0/nccl/nvidia/nccl/lib/libnccl.so.2"
_lib = ctypes.CDLL(os.environ.get("NCCL_LIB", _DEFAULT_LIB))

# ---------------------------------------------------------------------------
# constants (from nccl.h)
# ---------------------------------------------------------------------------

ncclSuccess = 0
ncclInProgress = 7

ncclFloat32 = 7
ncclSum = 0

NCCL_SUSPEND_MEM = 0x01

# ncclCommMemStat_t
ncclStatGpuMemSuspend = 0    # allocated GPU memory that can be suspended (bytes)
ncclStatGpuMemSuspended = 1  # suspended? (0=active, 1=suspended)
ncclStatGpuMemPersist = 2    # allocated GPU memory that cannot be suspended

ncclComm_t = ctypes.c_void_p
cudaStream_t = ctypes.c_void_p  # interchangeable with CUstream

NCCL_UNIQUE_ID_BYTES = 128


class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * NCCL_UNIQUE_ID_BYTES)]


_protos = {
    "ncclGetVersion": (ctypes.POINTER(ctypes.c_int),),
    "ncclGetErrorString": None,  # special: returns char*
    "ncclCommInitAll": (ctypes.POINTER(ncclComm_t), ctypes.c_int,
                        ctypes.POINTER(ctypes.c_int)),
    "ncclGetUniqueId": (ctypes.POINTER(ncclUniqueId),),
    "ncclCommInitRank": (ctypes.POINTER(ncclComm_t), ctypes.c_int,
                         ncclUniqueId, ctypes.c_int),
    "ncclCommDestroy": (ncclComm_t,),
    "ncclCommCount": (ncclComm_t, ctypes.POINTER(ctypes.c_int)),
    "ncclGroupStart": (),
    "ncclGroupEnd": (),
    "ncclAllReduce": (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                      ctypes.c_int, ctypes.c_int, ncclComm_t, cudaStream_t),
    "ncclCommSuspend": (ncclComm_t, ctypes.c_int),
    "ncclCommResume": (ncclComm_t,),
    "ncclCommMemStats": (ncclComm_t, ctypes.c_int,
                         ctypes.POINTER(ctypes.c_uint64)),
    "ncclCommGetAsyncError": (ncclComm_t, ctypes.POINTER(ctypes.c_int)),
}

for _name, _args in _protos.items():
    _fn = getattr(_lib, _name, None)
    if _fn is None:
        continue
    if _name == "ncclGetErrorString":
        _fn.restype = ctypes.c_char_p
        _fn.argtypes = [ctypes.c_int]
    else:
        _fn.restype = ctypes.c_int
        _fn.argtypes = list(_args)


class NcclError(Exception):
    def __init__(self, fn, code):
        msg = _lib.ncclGetErrorString(code).decode()
        self.code = code
        super().__init__(f"{fn} failed: {msg} ({code})")


def call(fn, *args):
    """Invoke an NCCL API, raising NcclError on failure."""
    f = getattr(_lib, fn, None)
    if f is None:
        raise NcclError(fn, 3)  # ncclInternalError stand-in for missing symbol
    rc = f(*args)
    if rc not in (ncclSuccess, ncclInProgress):
        raise NcclError(fn, rc)
    return rc


def has(fn):
    return getattr(_lib, fn, None) is not None


def version():
    v = ctypes.c_int()
    call("ncclGetVersion", ctypes.byref(v))
    return v.value


def mem_stats(comm):
    """Returns (suspendable_bytes, suspended_flag, persistent_bytes)."""
    out = []
    for stat in (ncclStatGpuMemSuspend, ncclStatGpuMemSuspended,
                 ncclStatGpuMemPersist):
        v = ctypes.c_uint64()
        call("ncclCommMemStats", comm, stat, ctypes.byref(v))
        out.append(v.value)
    return tuple(out)
