# repro_fabric_vmm.py — Class C5: single-process CUDA fabric (VMM multicast).
#
# NOTE (finding): this is a NEGATIVE CONTROL, not a reproduction of the hang.
# A single process that creates an NVSwitch multicast object
# (cuMulticastCreate -> NV_MEMORY_MULTICAST_FABRIC, class 0x000000fd) binding
# physical memory from 2 GPUs, and maps it, CHECKPOINTS AND RESTORES FINE
# (verified: `cuda-checkpoint --action checkpoint` succeeds, RESULT = PASS).
#
# So fabric/multicast memory *by itself* is NOT what cuda-checkpoint chokes on.
# The hang requires the multicast to be SHARED ACROSS PROCESSES (a
# cuda-checkpoint job with >1 member all referencing the same multicast), as in
# vLLM SYMM_MEM / NCCL NVLS. Use the `symm` repro (2-process torch symmetric
# memory) for the positive reproduction. Triangulation:
#     tp   (NCCL only, idle)                 -> PASS
#     fabric (this: 1-proc multicast)        -> PASS
#     symm (NCCL + shared multicast, idle)   -> HANG
# i.e. the blocker is cross-process *shared* fabric/multicast, not fabric per se.
#
# It drives the CUDA VMM + Multicast API directly via ctypes (no NCCL, no
# torch.distributed) and is useful for isolating single-process fabric behavior.
#
# Requires >= 2 NVSwitch/NVLink GPUs (multicast binds >= 2 devices). Boots in a
# few seconds. PyTorch is used only to create the CUDA primary contexts; the
# fabric API is driven via ctypes against the injected libcuda.so.1 (matches the
# host driver ABI, R610).
#
# Notes:
#  - The pure cuMemExportToShareableHandle(FABRIC) path (MODE=export) needs an
#    IMEX domain (/dev/nvidia-caps-imex-channels), which is not provisioned in
#    this container, so it returns CUDA_ERROR_NOT_PERMITTED. The multicast path
#    (MODE=multicast, default) is the fabric memory that actually occurs on this
#    single-node NVSwitch box.
#
# Knobs (env):
#   FABRIC_MODE      = multicast (default) | export
#   FABRIC_GPUS      = number of GPUs to bind for multicast (default 2)
#   FABRIC_ALLOC_MIB = allocation size, rounded up to granularity (default 32)
#
# App contract (via _harness.serve): GET /health once mapped, GET /verify.

import ctypes
import os

CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_HANDLE_TYPE_NONE = 0
CU_MEM_HANDLE_TYPE_FABRIC = 8
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
CU_MULTICAST_GRANULARITY_RECOMMENDED = 1

FABRIC_MODE = os.environ.get("FABRIC_MODE", "multicast").lower()
FABRIC_GPUS = int(os.environ.get("FABRIC_GPUS", "2"))
ALLOC_BYTES = int(os.environ.get("FABRIC_ALLOC_MIB", "32")) * 1024 * 1024


class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
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
        ("allocFlags", _AllocFlags),
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


def _load_cuda():
    lib = ctypes.CDLL("libcuda.so.1")
    sigs = {
        "cuInit": [ctypes.c_uint],
        "cuGetErrorName": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
        "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        "cuMemGetAllocationGranularity": [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(CUmemAllocationProp), ctypes.c_int],
        "cuMemCreate": [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t, ctypes.POINTER(CUmemAllocationProp), ctypes.c_ulonglong],
        "cuMemExportToShareableHandle": [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_int, ctypes.c_ulonglong],
        "cuMemAddressReserve": [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ulonglong, ctypes.c_ulonglong],
        "cuMemMap": [ctypes.c_ulonglong, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ulonglong, ctypes.c_ulonglong],
        "cuMemSetAccess": [ctypes.c_ulonglong, ctypes.c_size_t, ctypes.POINTER(CUmemAccessDesc), ctypes.c_size_t],
        "cuMulticastGetGranularity": [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(CUmulticastObjectProp), ctypes.c_int],
        "cuMulticastCreate": [ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(CUmulticastObjectProp)],
        "cuMulticastAddDevice": [ctypes.c_ulonglong, ctypes.c_int],
        "cuMulticastBindMem": [ctypes.c_ulonglong, ctypes.c_size_t, ctypes.c_ulonglong, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ulonglong],
    }
    for name, argtypes in sigs.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
    return lib


_cuda = None


def _check(rc, what):
    if rc != 0:
        name = ctypes.c_char_p()
        _cuda.cuGetErrorName(rc, ctypes.byref(name))
        s = name.value.decode() if name.value else "?"
        raise RuntimeError(f"{what} failed: CUresult={rc} ({s})")


def _round_up(n, g):
    return ((n + g - 1) // g) * g


def _device_mem(dev, size, handle_type):
    prop = CUmemAllocationProp()
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = handle_type
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = dev
    h = ctypes.c_ulonglong(0)
    _check(_cuda.cuMemCreate(ctypes.byref(h), size, ctypes.byref(prop), 0),
           f"cuMemCreate(dev={dev},handleType={handle_type})")
    return h


def _map(handle, size, devices):
    ptr = ctypes.c_ulonglong(0)
    _check(_cuda.cuMemAddressReserve(ctypes.byref(ptr), size, 0, 0, 0), "cuMemAddressReserve")
    _check(_cuda.cuMemMap(ptr.value, size, 0, handle, 0), "cuMemMap")
    descs = (CUmemAccessDesc * len(devices))()
    for i, d in enumerate(devices):
        descs[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE
        descs[i].location.id = d
        descs[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _check(_cuda.cuMemSetAccess(ptr.value, size, descs, len(devices)), "cuMemSetAccess")
    return ptr


def _setup_multicast():
    import torch

    devices = list(range(FABRIC_GPUS))
    if torch.cuda.device_count() < FABRIC_GPUS:
        raise RuntimeError(f"multicast needs {FABRIC_GPUS} GPUs, container sees {torch.cuda.device_count()}")
    ctx = []
    for d in devices:
        torch.cuda.set_device(d)
        ctx.append(torch.zeros(1, device=f"cuda:{d}"))  # create primary context

    _check(_cuda.cuInit(0), "cuInit")

    mcprop = CUmulticastObjectProp()
    mcprop.numDevices = FABRIC_GPUS
    mcprop.size = ALLOC_BYTES
    mcprop.handleTypes = 0
    mcprop.flags = 0
    gran = ctypes.c_size_t(0)
    _check(_cuda.cuMulticastGetGranularity(ctypes.byref(gran), ctypes.byref(mcprop),
                                           CU_MULTICAST_GRANULARITY_RECOMMENDED),
           "cuMulticastGetGranularity")
    g = gran.value or (2 * 1024 * 1024)
    size = _round_up(ALLOC_BYTES, g)
    mcprop.size = size

    mc = ctypes.c_ulonglong(0)
    _check(_cuda.cuMulticastCreate(ctypes.byref(mc), ctypes.byref(mcprop)), "cuMulticastCreate")
    for d in devices:
        cudev = ctypes.c_int(0)
        _check(_cuda.cuDeviceGet(ctypes.byref(cudev), d), f"cuDeviceGet({d})")
        _check(_cuda.cuMulticastAddDevice(mc, cudev.value), f"cuMulticastAddDevice({d})")

    phys = []
    for d in devices:
        h = _device_mem(d, size, CU_MEM_HANDLE_TYPE_NONE)
        _check(_cuda.cuMulticastBindMem(mc, 0, h, 0, size, 0), f"cuMulticastBindMem(dev={d})")
        phys.append(h)

    ptr = _map(mc, size, devices)
    print(f"[pid={os.getpid()}] multicast fabric memory mapped: size={size} devices={devices} "
          f"mc_va={hex(ptr.value)}", flush=True)
    return {"mode": "multicast", "mc": mc, "phys": phys, "ptr": ptr, "size": size,
            "ctx": ctx, "devices": devices}


def _setup_export():
    import torch

    torch.cuda.set_device(0)
    ctx = torch.zeros(1, device="cuda:0")
    _check(_cuda.cuInit(0), "cuInit")
    prop = CUmemAllocationProp()
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = 0
    gran = ctypes.c_size_t(0)
    _check(_cuda.cuMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop),
                                               CU_MEM_ALLOC_GRANULARITY_MINIMUM),
           "cuMemGetAllocationGranularity")
    size = _round_up(ALLOC_BYTES, gran.value or (2 * 1024 * 1024))
    h = ctypes.c_ulonglong(0)
    _check(_cuda.cuMemCreate(ctypes.byref(h), size, ctypes.byref(prop), 0), "cuMemCreate(FABRIC)")
    shareable = (ctypes.c_ubyte * 64)()
    _check(_cuda.cuMemExportToShareableHandle(ctypes.byref(shareable), h, CU_MEM_HANDLE_TYPE_FABRIC, 0),
           "cuMemExportToShareableHandle")
    ptr = _map(h, size, [0])
    print(f"[pid={os.getpid()}] exported FABRIC memory mapped: size={size} va={hex(ptr.value)}", flush=True)
    return {"mode": "export", "handle": h, "ptr": ptr, "size": size, "ctx": ctx, "devices": [0]}


def setup():
    global _cuda
    _cuda = _load_cuda()
    if FABRIC_MODE == "export":
        return _setup_export()
    return _setup_multicast()


def verify(state):
    return {
        "ok": True,
        "mode": state["mode"],
        "size": state["size"],
        "devices": state["devices"],
        "va": hex(state["ptr"].value),
    }


if __name__ == "__main__":
    from _harness import serve

    serve("C5: CUDA fabric VMM/multicast memory", setup, verify)
