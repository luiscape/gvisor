# Container-side harness shared by all local gVisor GPU-snapshot repros.
#
# Each repro calls serve(label, setup, verify):
#   - setup() sets up EXACTLY this repro's one resource and returns any state
#     that must be kept alive across the snapshot (kept referenced here).
#   - dump_pre_snapshot_state(label) is then printed as the closest observable
#     point to the checkpoint (the run_repro.sh runner checkpoints while this
#     process is idle in the server loop).
#   - an HTTP server exposes:
#       GET /health  -> 200 once setup finished (runner waits on this)
#       GET /verify  -> verify(state) as JSON, or {"ok": true}
#
# Dependency-free (stdlib only). Heavy/GPU imports belong in each repro's
# setup(), not at import time.

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from _diag import dump_pre_snapshot_state

_PORT = int(os.environ.get("PORT", "8199"))


def serve(label, setup, verify=None):
    print(f"[pid={os.getpid()}] repro starting: {label}", flush=True)
    state = setup()
    # Closest observable point to the snapshot.
    dump_pre_snapshot_state(label)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            try:
                if path == "/health":
                    self._json(200, {"status": "ok"})
                elif path == "/verify":
                    out = verify(state) if verify else {"ok": True}
                    self._json(200, out)
                else:
                    self._json(404, {"error": "not found"})
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": repr(exc)})

        def _json(self, code, obj):
            body = json.dumps(obj, sort_keys=True).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("0.0.0.0", _PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[pid={os.getpid()}] READY, serving on :{_PORT}", flush=True)
    while True:
        time.sleep(3600)
