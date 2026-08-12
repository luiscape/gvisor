"""Engine-side NCCL suspend/resume hook for vLLM (prototype of an engine fork).

Staged into the vLLM image and called from vllm/v1/worker/gpu_worker.py's
Worker.sleep()/wake_up(), so it runs once per tensor-parallel worker process.

This is the alternative to NCCL's own control thread: instead of a marker
protocol driven by gVisor, the *engine* calls the NCCL API directly as part of
its existing sleep lifecycle. It needs the patched libnccl, whose
ncclCommSuspend/Resume also release and rebuild the NVLS multicast layer.

Reachability is the whole question this answers. vLLM's PyNcclCommunicator
holds its ncclComm_t in Python (self.comm) so it can be suspended from here;
torch's ProcessGroupNCCL does not expose its communicator, so any multicast
owned by *that* comm cannot be reached this way. Whatever is left live is
reported by gVisor's checkpoint blocker gate, which names the owning rank.

Enabled only when VLLM_NCCL_SUSPEND_HOOK=1; otherwise every entry point is a
no-op, so the image stays usable for the other benches.
"""

import ctypes
import os
import sys

NCCL_SUSPEND_MEM = 0x01

_ENABLED = os.environ.get("VLLM_NCCL_SUSPEND_HOOK") == "1"
_suspended = False


def _log(msg):
    print(f"[vllm-nccl-ckpt pid={os.getpid()}] {msg}", file=sys.stderr, flush=True)


def _iter_pynccl_comms():
    """Yield (label, PyNcclCommunicator) for every group vLLM has built.

    vLLM keeps one GroupCoordinator per parallel axis; each may or may not have
    a pynccl communicator depending on world size and configuration.
    """
    try:
        from vllm.distributed import parallel_state as ps
    except Exception as e:  # pragma: no cover
        _log(f"cannot import vllm parallel_state: {e}")
        return

    getters = [
        ("world", "get_world_group"),
        ("tp", "get_tp_group"),
        ("pp", "get_pp_group"),
        ("dp", "get_dp_group"),
        ("ep", "get_ep_group"),
    ]
    seen = set()
    for label, name in getters:
        fn = getattr(ps, name, None)
        if fn is None:
            continue
        try:
            group = fn()
        except Exception:
            # Not initialized for this configuration (e.g. no DP/EP).
            continue
        for comm in _comms_of_group(group):
            key = id(comm)
            if key in seen:
                continue
            seen.add(key)
            yield label, comm


def _comms_of_group(group):
    """A GroupCoordinator reaches pynccl either directly or via its device
    communicator, depending on the vLLM version."""
    out = []
    for attr in ("pynccl_comm",):
        c = getattr(group, attr, None)
        if c is not None:
            out.append(c)
    dc = getattr(group, "device_communicator", None)
    if dc is not None:
        c = getattr(dc, "pynccl_comm", None)
        if c is not None:
            out.append(c)
    return out


def _comm_handle(comm_obj):
    """The raw ncclComm_t to pass to the C API.

    vLLM types it as ncclComm_t = ctypes.c_void_p, so self.comm is already a
    ctypes instance -- re-wrapping it raises "cannot be converted to pointer".
    Accept either form so this survives a vLLM change to a plain int.
    """
    c = comm_obj.comm
    return c if isinstance(c, ctypes.c_void_p) else ctypes.c_void_p(c)


def _comm_repr(comm_obj):
    c = comm_obj.comm
    v = c.value if isinstance(c, ctypes.c_void_p) else c
    return f"{(v or 0):#x}"


def _nccl_call(comm_obj, fname, *args):
    """Call fname on the same libnccl instance pynccl loaded.

    ncclCommSuspend/Resume are not in vLLM's NCCLLibrary function table, so
    resolve them from the underlying ctypes handle directly.
    """
    lib = comm_obj.nccl.lib
    fn = getattr(lib, fname, None)
    if fn is None:
        raise RuntimeError(
            f"{fname} not found in libnccl -- the patched NCCL is not loaded "
            "(check LD_PRELOAD / VLLM_NCCL_SO_PATH)")
    fn.restype = ctypes.c_int
    rc = fn(*args)
    if rc != 0:
        raise RuntimeError(f"{fname} failed with ncclResult={rc}")


def suspend():
    """Release NCCL's dynamic memory and NVLS multicast layer on this rank."""
    global _suspended
    if not _ENABLED or _suspended:
        return
    n = 0
    for label, comm_obj in _iter_pynccl_comms():
        if getattr(comm_obj, "disabled", False):
            continue
        _nccl_call(comm_obj, "ncclCommSuspend",
                   _comm_handle(comm_obj), ctypes.c_int(NCCL_SUSPEND_MEM))
        n += 1
        _log(f"suspended {label} comm {_comm_repr(comm_obj)}")
    _suspended = True
    _log(f"suspend complete ({n} communicator(s))")


def resume():
    """Rebuild what suspend() released. Collective across the comm's ranks:
    cuMulticastBindMem blocks until every device has re-joined, so every rank
    must be in here concurrently -- which vLLM's collective_rpc guarantees."""
    global _suspended
    if not _ENABLED or not _suspended:
        return
    n = 0
    for label, comm_obj in _iter_pynccl_comms():
        if getattr(comm_obj, "disabled", False):
            continue
        _nccl_call(comm_obj, "ncclCommResume", _comm_handle(comm_obj))
        n += 1
        _log(f"resumed {label} comm {_comm_repr(comm_obj)}")
    _suspended = False
    _log(f"resume complete ({n} communicator(s))")
