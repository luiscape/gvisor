#!/usr/bin/env python3
"""
gpu_mem_server.py — GPU memory checkpoint/restore benchmark app.

Allocates tensors with known contents on one or more GPUs, then serves
HTTP endpoints so a harness can verify that GPU memory survived a
checkpoint/restore cycle driven by gVisor's native cuda-checkpoint
integration (runsc checkpoint --cuda-checkpoint-path=...).

Unlike the GCR/libgcr test server, this app needs NO LD_PRELOAD and no
in-container cooperation: the sentry invokes cuda-checkpoint --toggle on
this process transparently around save/restore.

Endpoints
---------
GET /health     → 200 {"status": "ok"}
GET /checksums  → 200 {"checksums": {"cuda:0": [...], ...}, "num_gpus": N}
GET /matmul     → 200 {"results": {"cuda:0": checksum, ...}}   (live compute on every GPU)
GET /info       → 200 {pid, per-GPU memory info}

Environment
-----------
GPU_MEM_MB   memory to allocate PER GPU (default 512)
NUM_TENSORS  number of tensors per GPU (default 4)
PORT         HTTP listen port (default 8199)

The set of GPUs used is every device torch can see (shaped by
NVIDIA_VISIBLE_DEVICES / the OCI spec).
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

GPU_MEM_MB = int(os.environ.get("GPU_MEM_MB", "512"))
NUM_TENSORS = int(os.environ.get("NUM_TENSORS", "4"))
PORT = int(os.environ.get("PORT", "8199"))

_torch = None
_tensors = {}  # device str -> [tensor, ...]


def _init_gpu():
    global _torch
    pid = os.getpid()
    print(f"[pid={pid}] importing torch", flush=True)
    import torch

    _torch = torch
    torch.cuda.init()

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("no CUDA devices visible")

    elems = (GPU_MEM_MB * 1024 * 1024) // (4 * NUM_TENSORS)  # fp32
    for d in range(n_gpus):
        dev = f"cuda:{d}"
        name = torch.cuda.get_device_name(d)
        gen = torch.Generator(device=dev)
        gen.manual_seed(42 + d)
        _tensors[dev] = [
            torch.randn(elems, device=dev, dtype=torch.float32, generator=gen)
            for _ in range(NUM_TENSORS)
        ]
        torch.cuda.synchronize(d)
        alloc_mb = torch.cuda.memory_allocated(d) / 1e6
        print(
            f"[pid={pid}] {dev} ({name}): {NUM_TENSORS} tensors, {alloc_mb:.0f} MB",
            flush=True,
        )
    print(f"[pid={pid}] initialized {n_gpus} GPU(s)", flush=True)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        routes = {
            "/health": self._health,
            "/checksums": self._checksums,
            "/matmul": self._matmul,
            "/info": self._info,
        }
        handler = routes.get(path)
        if handler is None:
            self._json(404, {"error": "not found", "path": self.path})
            return
        try:
            handler()
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": str(exc)})

    def _health(self):
        self._json(200, {"status": "ok"})

    def _checksums(self):
        sums = {
            dev: [float(t.sum().item()) for t in ts] for dev, ts in _tensors.items()
        }
        self._json(200, {"checksums": sums, "num_gpus": len(_tensors)})

    def _matmul(self):
        torch = _torch
        results = {}
        for dev in _tensors:
            gen = torch.Generator(device=dev)
            gen.manual_seed(7)
            a = torch.randn(256, 256, device=dev, dtype=torch.float32, generator=gen)
            b = torch.randn(256, 256, device=dev, dtype=torch.float32, generator=gen)
            c = a @ b
            torch.cuda.synchronize()
            results[dev] = float(c.sum().item())
        self._json(200, {"results": results})

    def _info(self):
        torch = _torch
        gpus = {}
        for d in range(torch.cuda.device_count()):
            gpus[f"cuda:{d}"] = {
                "name": torch.cuda.get_device_name(d),
                "allocated_mb": round(torch.cuda.memory_allocated(d) / 1e6, 1),
                "reserved_mb": round(torch.cuda.memory_reserved(d) / 1e6, 1),
            }
        self._json(
            200,
            {
                "pid": os.getpid(),
                "num_tensors_per_gpu": NUM_TENSORS,
                "gpu_mem_mb_per_gpu": GPU_MEM_MB,
                "gpus": gpus,
            },
        )

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    pid = os.getpid()
    print(f"[pid={pid}] starting gpu_mem_server", flush=True)
    _init_gpu()

    server = HTTPServer(("0.0.0.0", PORT), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[pid={pid}] HTTP server listening on 0.0.0.0:{PORT}", flush=True)
    print("READY", flush=True)

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
