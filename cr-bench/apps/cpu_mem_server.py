#!/usr/bin/env python3
"""
cpu_mem_server.py — CPU memory checkpoint/restore benchmark app.

Allocates a configurable amount of anonymous memory filled with
incompressible random data, then serves HTTP endpoints so a harness can
verify that the memory contents survived a gVisor checkpoint/restore
cycle bit-exactly.

Endpoints
---------
GET /health     → 200 {"status": "ok"}
GET /checksums  → 200 {"checksums": [hex, ...], "num_buffers": N, "total_mb": M}
GET /touch      → 200 {"ok": true, ...}   (writes + re-reads a buffer: memory is live)
GET /info       → 200 {pid, mem info}

Environment
-----------
MEM_MB       total memory to allocate (default 1024)
NUM_BUFFERS  number of buffers to split it into (default 8)
PORT         HTTP listen port (default 8199)

Stdlib only — no third-party dependencies.
"""

import hashlib
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

MEM_MB = int(os.environ.get("MEM_MB", "1024"))
NUM_BUFFERS = int(os.environ.get("NUM_BUFFERS", "8"))
PORT = int(os.environ.get("PORT", "8199"))

_buffers = []


def _init_mem():
    """Allocate NUM_BUFFERS buffers of incompressible random data."""
    pid = os.getpid()
    per_buf = (MEM_MB * 1024 * 1024) // NUM_BUFFERS
    t0 = time.time()
    for i in range(NUM_BUFFERS):
        _buffers.append(bytearray(os.urandom(per_buf)))
        print(
            f"[pid={pid}] buffer {i}: {per_buf // (1024 * 1024)} MiB allocated",
            flush=True,
        )
    dt = time.time() - t0
    print(
        f"[pid={pid}] allocated {MEM_MB} MiB in {NUM_BUFFERS} buffers ({dt:.1f}s)",
        flush=True,
    )


def _checksums():
    return [hashlib.sha256(bytes(b)).hexdigest()[:16] for b in _buffers]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        routes = {
            "/health": self._health,
            "/checksums": self._route_checksums,
            "/touch": self._touch,
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

    def _route_checksums(self):
        self._json(
            200,
            {
                "checksums": _checksums(),
                "num_buffers": len(_buffers),
                "total_mb": MEM_MB,
            },
        )

    def _touch(self):
        # Prove memory is writable + readable after restore: overwrite a
        # 1 MiB scratch region of buffer 0 and verify the round-trip.
        scratch = os.urandom(1024 * 1024)
        _buffers[0][: len(scratch)] = scratch
        ok = bytes(_buffers[0][: len(scratch)]) == scratch
        self._json(200, {"ok": ok, "bytes": len(scratch)})

    def _info(self):
        rss_kb = 0
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
        except OSError:
            pass
        self._json(
            200,
            {
                "pid": os.getpid(),
                "num_buffers": len(_buffers),
                "total_mb": MEM_MB,
                "rss_mb": rss_kb // 1024,
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
    print(f"[pid={pid}] starting cpu_mem_server", flush=True)
    _init_mem()

    server = HTTPServer(("0.0.0.0", PORT), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[pid={pid}] HTTP server listening on 0.0.0.0:{PORT}", flush=True)
    print("READY", flush=True)

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
