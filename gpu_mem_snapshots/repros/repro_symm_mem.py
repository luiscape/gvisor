# repro_symm_mem.py — Class C4: multi-GPU PyTorch symmetric-memory (fabric/multicast).
#
# Fast, model-free reproduction of the *fabric memory* that vLLM's SYMM_MEM
# all-reduce backend (and NCCL NVLS) allocates on H100/NVSwitch, which
# cuda-checkpoint cannot checkpoint ("IPC memory created with
# cuMemExportToShareableHandle()"). Boots in seconds instead of ~150s, so the
# fabric checkpoint/restore path can be studied without re-running a vLLM
# compile.
#
# What it allocates
# -----------------
# Each worker joins the NCCL process group and uses PyTorch's symmetric-memory
# API (torch.distributed._symmetric_memory) — the same API vLLM's SYMM_MEM
# backend uses:
#     t   = symm_mem.empty(N, ...)          # symmetric allocator (CUDA VMM)
#     hdl = symm_mem.rendezvous(t, group)   # exports fabric handles + binds a
#                                           # multicast object across ranks
# This produces exactly the driver objects that hang the multi-GPU vLLM
# checkpoint:
#     NV_MEMORY_FABRIC           (class 0x000000f8)  — per-rank fabric export
#     NV_MEMORY_MULTICAST_FABRIC (class 0x000000fd)  — multicast object
#
# Drain / rebuild (the "stable solution" under study)
# ---------------------------------------------------
# cuda-checkpoint spins forever inside its own worker thread when it tries to
# serialize this fabric memory. The fix is to remove the fabric memory before
# `runsc checkpoint` and re-establish it after restore, WITHOUT tearing down
# the NCCL process group (so a real engine's CUDA graphs / compiled state are
# preserved). This repro exposes that as app-cooperative HTTP hooks:
#     POST /drain    -> every worker frees its symmetric tensor + handle
#                       (no fabric memory remains; NCCL group stays alive)
#     POST /rebuild  -> every worker re-empty()+rendezvous() (collective)
# The /drain and /rebuild responses report each rank's buffer virtual address
# before and after, so we can see whether the symmetric allocator hands back
# the SAME VA (which a captured CUDA graph would require).
#
# Endpoints: GET /health, GET /verify, POST /drain, POST /rebuild.
# Env: REPRO_MODE=idle (default) keeps fabric resident; SYMM_ELEMS sizes it.
# Requires >= 2 GPUs with multicast support (H100 + NVSwitch).

import ctypes
import json
import os
import threading
import time
import multiprocessing as mp
from http.server import BaseHTTPRequestHandler, HTTPServer

_PORT = int(os.environ.get("PORT", "8199"))
MASTER_PORT = int(os.environ.get("TP_MASTER_PORT", "29500"))
SYMM_ELEMS = int(os.environ.get("SYMM_ELEMS", str(1024 * 1024)))  # 4 MiB f32
REPRO_MODE = os.environ.get("REPRO_MODE", "idle").strip().lower()


def _worker(rank, world_size, cmd_q, ack_q, stop_evt):
    import torch
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "1")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "1")
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD
    gname = group.group_name
    try:
        symm_mem.enable_symm_mem_for_group(gname)
    except Exception as e:  # noqa: BLE001
        print(f"[rank={rank}] enable_symm_mem_for_group: {e!r}", flush=True)

    state = {"t": None, "hdl": None}

    def _rendezvous():
        t = symm_mem.empty(SYMM_ELEMS, dtype=torch.float32, device=f"cuda:{rank}")
        t.fill_(1.0)
        hdl = symm_mem.rendezvous(t, group)
        # multicast all-reduce forces the fabric/multicast path; validates data.
        try:
            torch.ops.symm_mem.multimem_all_reduce_(t, "sum", gname)
        except Exception:  # noqa: BLE001
            torch.ops.symm_mem.one_shot_all_reduce(t, "sum", gname)
        torch.cuda.synchronize()
        state["t"], state["hdl"] = t, hdl
        mc = 0
        try:
            mc = int(getattr(hdl, "multicast_ptr", 0) or 0)
        except Exception:  # noqa: BLE001
            mc = 0
        return {"va": int(t.data_ptr()), "elem": float(t[0].item()),
                "multicast": mc != 0}

    def _drain():
        state["hdl"] = None
        state["t"] = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    info = _rendezvous()
    ack_q.put({"rank": rank, "event": "ready", **info})

    while not stop_evt.is_set():
        try:
            cmd = cmd_q.get(timeout=0.5)
        except Exception:  # noqa: BLE001
            continue
        try:
            if cmd == "drain":
                old_va = int(state["t"].data_ptr()) if state["t"] is not None else 0
                _drain()
                ack_q.put({"rank": rank, "event": "drained", "old_va": old_va})
            elif cmd == "rebuild":
                info = _rendezvous()
                ack_q.put({"rank": rank, "event": "rebuilt", **info})
            elif cmd == "stop":
                ack_q.put({"rank": rank, "event": "stopped"})
                break
        except Exception as e:  # noqa: BLE001
            ack_q.put({"rank": rank, "event": "error", "error": repr(e)})


class _Cluster:
    def __init__(self):
        self.ctx = mp.get_context("spawn")
        self.ack_q = self.ctx.Queue()
        self.stop_evt = self.ctx.Event()
        self.cmd_qs = []
        self.procs = []
        self.world = 0
        self.va = {}          # rank -> current buffer VA
        self.multicast = {}   # rank -> bool
        self.elem = {}        # rank -> last all-reduce element

    def _collect(self, event, n, timeout=300):
        got = []
        deadline = time.time() + timeout
        while len(got) < n and time.time() < deadline:
            try:
                msg = self.ack_q.get(timeout=5)
            except Exception:  # noqa: BLE001
                if any(not p.is_alive() for p in self.procs):
                    raise RuntimeError("a symm_mem worker exited unexpectedly")
                continue
            if msg.get("event") == "error":
                raise RuntimeError(f"worker rank {msg['rank']}: {msg['error']}")
            if msg.get("event") == event:
                got.append(msg)
        if len(got) < n:
            raise RuntimeError(f"only {len(got)}/{n} workers acked {event!r}")
        return got

    def start(self):
        import torch
        self.world = torch.cuda.device_count()
        if self.world < 2:
            raise RuntimeError(
                f"repro_symm_mem needs >= 2 GPUs but the container sees {self.world}")
        for rank in range(self.world):
            cq = self.ctx.Queue()
            p = self.ctx.Process(target=_worker,
                                 args=(rank, self.world, cq, self.ack_q, self.stop_evt),
                                 daemon=True)
            p.start()
            self.cmd_qs.append(cq)
            self.procs.append(p)
        for m in self._collect("ready", self.world):
            self._record(m)

    def _record(self, m):
        self.va[m["rank"]] = m.get("va", 0)
        if "multicast" in m:
            self.multicast[m["rank"]] = m["multicast"]
        if "elem" in m:
            self.elem[m["rank"]] = m["elem"]

    def broadcast(self, cmd, ack_event):
        old_va = dict(self.va)
        for cq in self.cmd_qs:
            cq.put(cmd)
        msgs = self._collect(ack_event, self.world)
        for m in msgs:
            self._record(m)
        return old_va, {m["rank"]: m for m in msgs}


def _mk_handler(cluster):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj, sort_keys=True).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _summary(self):
            alive = sum(1 for p in cluster.procs if p.is_alive())
            return {
                "ok": alive == cluster.world,
                "workers": cluster.world,
                "alive": alive,
                "multicast": all(cluster.multicast.get(r, False)
                                 for r in range(cluster.world)),
                "allreduce_elem": cluster.elem.get(0, 0.0),
                "expected_elem": cluster.world,
                "mode": REPRO_MODE,
            }

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            try:
                if path == "/health":
                    self._json(200, {"status": "ok"})
                elif path == "/verify":
                    self._json(200, self._summary())
                else:
                    self._json(404, {"error": "not found"})
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": repr(exc)})

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            try:
                if path == "/drain":
                    old_va, acks = cluster.broadcast("drain", "drained")
                    self._json(200, {"status": "drained",
                                     "freed_va": {r: acks[r]["old_va"]
                                                  for r in acks}})
                elif path == "/rebuild":
                    old_va, acks = cluster.broadcast("rebuild", "rebuilt")
                    new_va = {r: acks[r]["va"] for r in acks}
                    preserved = all(old_va.get(r) == new_va.get(r)
                                    for r in range(cluster.world))
                    self._json(200, {"status": "rebuilt",
                                     "old_va": old_va, "new_va": new_va,
                                     "va_preserved": preserved,
                                     **self._summary()})
                else:
                    self._json(404, {"error": "not found"})
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": repr(exc)})

    return Handler


def main():
    print(f"[pid={os.getpid()}] repro starting: C4 symmetric-memory fabric", flush=True)
    cluster = _Cluster()
    cluster.start()
    try:
        from _diag import dump_pre_snapshot_state
        dump_pre_snapshot_state("C4: symmetric-memory fabric/multicast")
    except Exception:  # noqa: BLE001
        pass
    print(f"[pid={os.getpid()}] all {cluster.world} workers ready "
          f"(mode={REPRO_MODE}, multicast={cluster.multicast}, va={cluster.va})",
          flush=True)

    server = HTTPServer(("0.0.0.0", _PORT), _mk_handler(cluster))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[pid={os.getpid()}] READY, serving on :{_PORT}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
