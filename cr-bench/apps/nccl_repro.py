#!/usr/bin/env python3
"""
nccl_repro.py — minimal NCCL multi-GPU checkpoint/restore reproduction app.

A CUDA-free parent process spawns WORLD_SIZE worker processes (one per
GPU, mirroring vLLM/SGLang tensor-parallel workers). Each worker joins a
NCCL process group over localhost TCP, allreduces a deterministic
"persisted" tensor once at init, and then either idles (default) or runs
an unsynchronized allreduce loop (ACTIVE=1) so that collectives are in
flight when cuda-checkpoint toggles the process.

The parent serves HTTP (it holds no CUDA state, so the sentry never
toggles it and the endpoints stay responsive even if workers wedge):

GET /health  → 200 once every rank finished its initial allreduce
GET /info    → worker pids, mode, world size, NCCL-related env
GET /verify  → asks every rank to report:
                 persisted  checksum of the reduced tensor kept on GPU
                            since init (GPU memory integrity)
                 fresh      result of a brand-new deterministic
                            allreduce (does NCCL still work at all)
                 iters      allreduce-loop iteration count (ACTIVE=1)
               Ranks that fail to answer within VERIFY_TIMEOUT seconds
               are reported as "hang" — that is a reproduction signal,
               not a harness error.

Modes (MODE env)
----------------
nccl      multi-process NCCL process group via torch.distributed (default)
ncclraw   multi-process NCCL via ctypes on libnccl directly. Exposes the
          raw ncclComm_t, enabling the ncclCommSuspend/ncclCommResume
          (NCCL >= 2.30) lifecycle: GET /suspend releases the
          communicator's dynamic GPU allocations (cuMem shareable
          handles, SHM-transport buffers) BEFORE cuda-checkpoint runs,
          GET /resume re-creates them after restore — mirroring vLLM's
          /sleep + /wake_up pattern.
p2p1proc  single CUDA process using every visible GPU with direct
          cross-device copies — peer access without NCCL and without
          CUDA IPC. Control case that separates "IPC problem" from
          "P2P problem".

Environment
-----------
WORLD_SIZE      number of ranks / GPUs (default 2)
MODE            nccl | p2p1proc (default nccl)
ACTIVE          1 = keep allreduces in flight continuously (default 0)
TENSOR_MB       size of persisted and work tensors per rank (default 64)
PORT            HTTP listen port (default 8199)
VERIFY_TIMEOUT  seconds to wait for each /verify round (default 30)
NCCL_*          forwarded to workers untouched (set NCCL_DEBUG=INFO to
                log transport selection into this process's stdout)
"""

import json
import multiprocessing as mp
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "2"))
MODE = os.environ.get("MODE", "nccl")
ACTIVE = os.environ.get("ACTIVE", "0") == "1"
TENSOR_MB = int(os.environ.get("TENSOR_MB", "64"))
PORT = int(os.environ.get("PORT", "8199"))
VERIFY_TIMEOUT = float(os.environ.get("VERIFY_TIMEOUT", "30"))

STORE_URL = "tcp://127.0.0.1:29500"


def _log(msg):
    print(f"[pid={os.getpid()}] {msg}", flush=True)


# ── raw NCCL via ctypes (for ncclCommSuspend/Resume) ─────────────────────
NCCL_SUSPEND_MEM = 0x01
NCCL_FLOAT32 = 7
NCCL_SUM = 0


def _load_nccl():
    import ctypes
    import glob

    cands = glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/nccl/lib/libnccl.so.2") or [
        "libnccl.so.2"
    ]
    lib = ctypes.CDLL(cands[0], mode=ctypes.RTLD_GLOBAL)

    # NB: use c_uint8, not c_char — ctypes reads c_char arrays as
    # NUL-terminated strings, which would truncate the uniqueId.
    class UniqueId(ctypes.Structure):
        _fields_ = [("internal", ctypes.c_uint8 * 128)]

    lib.ncclGetErrorString.restype = ctypes.c_char_p
    return lib, UniqueId


def _nccl_check(lib, rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed: rc={rc} ({lib.ncclGetErrorString(rc).decode()})")


def ncclraw_worker(rank, world_size, tensor_mb, active, cmd_q, res_q, uid_q):
    import ctypes

    import torch

    lib, UniqueId = _load_nccl()
    ver = ctypes.c_int()
    lib.ncclGetVersion(ctypes.byref(ver))
    torch.cuda.set_device(rank)
    torch.cuda.init()
    dev = f"cuda:{rank}"
    _log(f"rank {rank}: raw NCCL version {ver.value}")

    def init_comm():
        uid = UniqueId()
        if rank == 0:
            _nccl_check(lib, lib.ncclGetUniqueId(ctypes.byref(uid)), "ncclGetUniqueId")
            blob = bytes(uid.internal)
            for _ in range(world_size - 1):
                uid_q.put(blob)
        else:
            blob = uid_q.get(timeout=120)
            ctypes.memmove(ctypes.byref(uid), blob, len(blob))
        c = ctypes.c_void_p()
        _nccl_check(
            lib,
            lib.ncclCommInitRank(ctypes.byref(c), world_size, uid, rank),
            "ncclCommInitRank",
        )
        return c

    comm = init_comm()

    elems = tensor_mb * 1024 * 1024 // 4
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    def allreduce(t):
        _nccl_check(
            lib,
            lib.ncclAllReduce(
                ctypes.c_void_p(t.data_ptr()), ctypes.c_void_p(t.data_ptr()),
                ctypes.c_size_t(t.numel()), NCCL_FLOAT32, NCCL_SUM, comm, stream,
            ),
            "ncclAllReduce",
        )
        torch.cuda.synchronize()

    gen = torch.Generator(device=dev)
    gen.manual_seed(100 + rank)
    persisted = torch.randn(elems, device=dev, dtype=torch.float32, generator=gen)
    allreduce(persisted)
    work = torch.ones(elems, device=dev, dtype=torch.float32)

    res_q.put(("ready", rank, os.getpid(), float(persisted.sum().item())))
    _log(f"rank {rank}: ready (raw NCCL, persisted sum {float(persisted.sum().item()):.6g})")

    iters = 0
    suspended = False
    while True:
        if active and not suspended and comm is not None:
            allreduce(work)
            iters += 1
            try:
                cmd = cmd_q.get_nowait()
            except Exception:
                continue
        else:
            try:
                cmd = cmd_q.get(timeout=1.0)
            except Exception:
                continue
        op, gen_id = cmd[0], cmd[1]
        try:
            if op == "verify":
                if comm is None:
                    raise RuntimeError("communicator is torn down")
                p = float(persisted.sum().item())
                g2 = torch.Generator(device=dev)
                g2.manual_seed(7 + rank)
                fresh = torch.randn(elems, device=dev, dtype=torch.float32, generator=g2)
                allreduce(fresh)
                res_q.put(("verify", rank, gen_id,
                           {"persisted": p, "fresh": float(fresh.sum().item()),
                            "iters": iters, "suspended": suspended}))
            elif op == "suspend":
                torch.cuda.synchronize()
                _nccl_check(lib, lib.ncclCommSuspend(comm, NCCL_SUSPEND_MEM), "ncclCommSuspend")
                suspended = True
                _log(f"rank {rank}: ncclCommSuspend(NCCL_SUSPEND_MEM) OK")
                res_q.put(("suspend", rank, gen_id, {"ok": True}))
            elif op == "resume":
                _nccl_check(lib, lib.ncclCommResume(comm), "ncclCommResume")
                suspended = False
                _log(f"rank {rank}: ncclCommResume OK")
                res_q.put(("resume", rank, gen_id, {"ok": True}))
            elif op == "teardown":
                # Collective: all ranks must call ncclCommDestroy concurrently.
                torch.cuda.synchronize()
                _nccl_check(lib, lib.ncclCommDestroy(comm), "ncclCommDestroy")
                comm = None
                _log(f"rank {rank}: ncclCommDestroy OK")
                res_q.put(("teardown", rank, gen_id, {"ok": True}))
            elif op == "reinit":
                comm = init_comm()
                _log(f"rank {rank}: reinit OK")
                res_q.put(("reinit", rank, gen_id, {"ok": True}))
            elif op == "setenv":
                key, val = cmd[2], cmd[3]
                os.environ[key] = val
                _log(f"rank {rank}: setenv {key}={val} (in-process)")
                res_q.put(("setenv", rank, gen_id, {"ok": True, key: val}))
        except Exception as exc:  # noqa: BLE001
            _log(f"rank {rank}: {op} FAILED: {exc!r}")
            res_q.put((op, rank, gen_id, {"error": repr(exc), "iters": iters}))


# ── worker: one NCCL rank ─────────────────────────────────────────────
def _rank_verify(torch, dist, rank, persisted, elems, dev):
    """Returns (persisted_sum, fresh_sum). Deterministic across runs."""
    persisted_sum = float(persisted.sum().item())
    gen = torch.Generator(device=dev)
    gen.manual_seed(7 + rank)
    fresh = torch.randn(elems, device=dev, dtype=torch.float32, generator=gen)
    dist.all_reduce(fresh)
    torch.cuda.synchronize()
    fresh_sum = float(fresh.sum().item())
    return persisted_sum, fresh_sum


def nccl_worker(rank, world_size, tensor_mb, active, cmd_q, res_q):
    import torch
    import torch.distributed as dist

    dev = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    _log(f"rank {rank}: init_process_group nccl world_size={world_size}")
    dist.init_process_group(
        "nccl", init_method=STORE_URL, rank=rank, world_size=world_size
    )

    elems = tensor_mb * 1024 * 1024 // 4
    gen = torch.Generator(device=dev)
    gen.manual_seed(100 + rank)
    persisted = torch.randn(elems, device=dev, dtype=torch.float32, generator=gen)
    dist.all_reduce(persisted)  # first collective establishes transports
    torch.cuda.synchronize()
    work = torch.ones(elems, device=dev, dtype=torch.float32)

    res_q.put(("ready", rank, os.getpid(), float(persisted.sum().item())))
    _log(f"rank {rank}: ready (persisted sum {float(persisted.sum().item()):.6g})")

    iters = 0
    while True:
        if active:
            # Keep collectives in flight; sync only every 10 iterations so
            # the stream always has pending NCCL work at checkpoint time.
            dist.all_reduce(work)
            iters += 1
            if iters % 10 == 0:
                torch.cuda.synchronize()
            try:
                cmd = cmd_q.get_nowait()
            except Exception:
                continue
        else:
            try:
                cmd = cmd_q.get(timeout=1.0)
            except Exception:
                continue
        if cmd[0] == "verify":
            gen_id = cmd[1]
            try:
                p, f = _rank_verify(torch, dist, rank, persisted, elems, dev)
                res_q.put(("verify", rank, gen_id, {"persisted": p, "fresh": f, "iters": iters}))
            except Exception as exc:  # noqa: BLE001
                res_q.put(("verify", rank, gen_id, {"error": repr(exc), "iters": iters}))


# ── worker: single process, all GPUs, no NCCL/IPC (control) ──────────────
def p2p1proc_worker(rank, world_size, tensor_mb, active, cmd_q, res_q):
    import torch

    n = torch.cuda.device_count()
    elems = tensor_mb * 1024 * 1024 // 4
    persisted = []
    for d in range(n):
        gen = torch.Generator(device=f"cuda:{d}")
        gen.manual_seed(100 + d)
        persisted.append(
            torch.randn(elems, device=f"cuda:{d}", dtype=torch.float32, generator=gen)
        )
    # Sum across devices with direct cross-device copies (peer access when
    # available, staged copies otherwise) — no IPC involved.
    acc = persisted[0].clone()
    for d in range(1, n):
        acc += persisted[d].to("cuda:0")
    for d in range(n):
        torch.cuda.synchronize(d)
    res_q.put(("ready", 0, os.getpid(), float(acc.sum().item())))
    _log(f"p2p1proc: ready across {n} GPUs (sum {float(acc.sum().item()):.6g})")

    iters = 0
    while True:
        try:
            cmd = cmd_q.get(timeout=1.0)
        except Exception:
            if active and n > 1:
                acc.copy_(persisted[1].to("cuda:0"))
                iters += 1
            continue
        if cmd[0] == "verify":
            gen_id = cmd[1]
            try:
                fresh = persisted[0].clone()
                for d in range(1, n):
                    fresh += persisted[d].to("cuda:0")
                torch.cuda.synchronize()
                res_q.put(("verify", 0, gen_id, {
                    "persisted": float(sum(float(t.sum().item()) for t in persisted)),
                    "fresh": float(fresh.sum().item()),
                    "iters": iters,
                }))
            except Exception as exc:  # noqa: BLE001
                res_q.put(("verify", 0, gen_id, {"error": repr(exc), "iters": iters}))


# ── parent: HTTP + result collector ───────────────────────────────────────
class State:
    def __init__(self, nranks):
        self.nranks = nranks
        self.lock = threading.Condition()
        self.ready = {}    # rank -> pid
        self.init_sums = {}
        self.results = {}  # (op, gen_id) -> {rank: dict}


STATE = None
CMD_QS = []
GEN = [0]


def _collector(res_q):
    while True:
        msg = res_q.get()
        with STATE.lock:
            if msg[0] == "ready":
                _, rank, pid, s = msg
                STATE.ready[rank] = pid
                STATE.init_sums[rank] = s
            else:
                op, rank, gen_id, payload = msg
                STATE.results.setdefault((op, gen_id), {})[rank] = payload
            STATE.lock.notify_all()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        try:
            if path == "/health":
                with STATE.lock:
                    ok = len(STATE.ready) == STATE.nranks
                self._json(200 if ok else 503, {"status": "ok" if ok else "starting",
                                                "ready": len(STATE.ready)})
            elif path == "/info":
                with STATE.lock:
                    info = {
                        "mode": MODE,
                        "world_size": WORLD_SIZE,
                        "active": ACTIVE,
                        "tensor_mb": TENSOR_MB,
                        "parent_pid": os.getpid(),
                        "worker_pids": dict(sorted(STATE.ready.items())),
                        "init_sums": STATE.init_sums,
                        "nccl_env": {k: v for k, v in os.environ.items()
                                     if k.startswith("NCCL_")},
                    }
                self._json(200, info)
            elif path == "/verify":
                self._roundtrip("verify")
            elif path == "/suspend":
                self._roundtrip("suspend")
            elif path == "/resume":
                self._roundtrip("resume")
            elif path == "/teardown":
                self._roundtrip("teardown")
            elif path == "/reinit":
                self._roundtrip("reinit")
            elif path == "/setenv":
                # /setenv?key=NCCL_CUMEM_ENABLE&value=0
                from urllib.parse import parse_qs, urlparse
                qs = parse_qs(urlparse(self.path).query)
                key = qs.get("key", ["NCCL_CUMEM_ENABLE"])[0]
                val = qs.get("value", [""])[0]
                self._roundtrip("setenv", extra=(key, val))
            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": repr(exc)})

    def _roundtrip(self, op, extra=()):
        if op in ("suspend", "resume", "teardown", "reinit", "setenv") and MODE != "ncclraw":
            self._json(400, {"error": f"{op} requires MODE=ncclraw"})
            return
        GEN[0] += 1
        gen_id = GEN[0]
        for q in CMD_QS:
            q.put((op, gen_id) + extra)
        deadline = time.time() + VERIFY_TIMEOUT
        nranks = 1 if MODE == "p2p1proc" else WORLD_SIZE
        key = (op, gen_id)
        with STATE.lock:
            while (len(STATE.results.get(key, {})) < nranks
                   and time.time() < deadline):
                STATE.lock.wait(timeout=1.0)
            got = dict(STATE.results.get(key, {}))
        hung = [r for r in range(nranks) if r not in got]
        errors = {str(r): v["error"] for r, v in got.items() if "error" in v}
        self._json(200, {
            "op": op,
            "gen": gen_id,
            "ranks": {str(r): got[r] for r in sorted(got)},
            "hung_ranks": hung,
            "errors": errors,
            "complete": not hung and not errors,
        })

    def _json(self, code, obj):
        body = json.dumps(obj, sort_keys=True).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    global STATE
    _log(f"nccl_repro starting: mode={MODE} world_size={WORLD_SIZE} "
         f"active={ACTIVE} tensor_mb={TENSOR_MB}")
    _log(f"NCCL env: { {k: v for k, v in os.environ.items() if k.startswith('NCCL_')} }")

    ctx = mp.get_context("spawn")
    res_q = ctx.Queue()
    nworkers = 1 if MODE == "p2p1proc" else WORLD_SIZE
    STATE = State(nworkers)
    targets = {"nccl": nccl_worker, "ncclraw": ncclraw_worker, "p2p1proc": p2p1proc_worker}
    target = targets[MODE]
    uid_q = ctx.Queue() if MODE == "ncclraw" else None

    procs = []
    for rank in range(nworkers):
        cmd_q = ctx.Queue()
        CMD_QS.append(cmd_q)
        args = (rank, WORLD_SIZE, TENSOR_MB, ACTIVE, cmd_q, res_q)
        if uid_q is not None:
            args += (uid_q,)
        p = ctx.Process(target=target, args=args, daemon=True)
        p.start()
        procs.append(p)
        _log(f"spawned rank {rank} pid={p.pid}")

    threading.Thread(target=_collector, args=(res_q,), daemon=True).start()

    server = HTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _log(f"HTTP server on 0.0.0.0:{PORT}")

    while True:
        time.sleep(5)
        for p in procs:
            if not p.is_alive():
                _log(f"FATAL: worker pid={p.pid} exited with {p.exitcode}")
                os._exit(1)


if __name__ == "__main__":
    main()
