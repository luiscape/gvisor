#!/usr/bin/env python3
"""
cuda_churn.py — continuous CUDA process/thread churn stress for snapshot.

Stresses the sentry's CUDA process enumeration during checkpoint
(pkg/sentry/control/state_cuda.go): at any instant some processes are
mid-cuInit (late initializers the re-enumeration loop must catch), some
are exiting (toggle targets may die mid-pass), and every CUDA process
runs collectives-free multi-threaded GPU work on multiple streams.

Topology (all knobs via env):
  - LONG_WORKERS long-lived CUDA processes (round-robin across GPUs),
    each holding a deterministic persisted tensor and THREADS threads
    doing continuous matmuls on private streams. Used to verify GPU
    memory integrity and liveness across the snapshot.
  - CHURN_SLOTS spawner threads in the (CUDA-free) parent, each running
    an endless loop: spawn a churn process -> it sleeps a random jitter
    (0..JITTER_MS), inits CUDA, runs THREADS matmul threads for a random
    lifetime (MIN_LIFE_MS..MAX_LIFE_MS), self-checks a deterministic
    result, exits 0 -> parent reaps it and immediately spawns the next.

Endpoints (parent, CUDA-free, never toggled):
  GET /health   200 once all long-lived workers are ready and >=1 churn
                process has completed successfully
  GET /stats    births / clean deaths / failures / churn rate; use a
                pre/post snapshot delta to prove churn resumed
  GET /verify   long-lived workers: persisted-tensor checksum + fresh
                deterministic matmul (hang/timeout detection built in)

Besides CUDA processes (toggled by cuda-checkpoint), this also runs
processes that hold nvproxy device FDs but are deliberately NOT toggled
(state_cuda.go filters them out via --get-state), so gVisor must
serialize their nvproxy state raw and replay it on restore:
  - NVML workers (ctypes on libnvidia-ml.so.1): open frontendFD with
    live RM client objects, continuous control-ioctl polling; churn
    variants exit WITHOUT nvmlShutdown half the time (dangling session).
  - Raw-FD workers: open /dev/nvidiactl, /dev/nvidiaN, /dev/nvidia-uvm
    and do nothing — bare frontendFD/uvmFD serialization.
Both exist long-lived (untoggled nvproxy state is GUARANTEED present at
every checkpoint) and churning (FDs appear/vanish mid-save).

Env: LONG_WORKERS=2 CHURN_SLOTS=4 THREADS=4 TENSOR_MB=32 JITTER_MS=500
     MIN_LIFE_MS=500 MAX_LIFE_MS=3000 PORT=8199 VERIFY_TIMEOUT=30
     NVML_WORKERS=1 NVML_CHURN_SLOTS=2 RAWFD_WORKERS=1 RAWFD_CHURN_SLOTS=2
"""

import json
import multiprocessing as mp
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

LONG_WORKERS = int(os.environ.get("LONG_WORKERS", "2"))
CHURN_SLOTS = int(os.environ.get("CHURN_SLOTS", "4"))
THREADS = int(os.environ.get("THREADS", "4"))
TENSOR_MB = int(os.environ.get("TENSOR_MB", "32"))
JITTER_MS = int(os.environ.get("JITTER_MS", "500"))
MIN_LIFE_MS = int(os.environ.get("MIN_LIFE_MS", "500"))
MAX_LIFE_MS = int(os.environ.get("MAX_LIFE_MS", "3000"))
PORT = int(os.environ.get("PORT", "8199"))
VERIFY_TIMEOUT = float(os.environ.get("VERIFY_TIMEOUT", "30"))
NVML_WORKERS = int(os.environ.get("NVML_WORKERS", "1"))
NVML_CHURN_SLOTS = int(os.environ.get("NVML_CHURN_SLOTS", "2"))
RAWFD_WORKERS = int(os.environ.get("RAWFD_WORKERS", "1"))
RAWFD_CHURN_SLOTS = int(os.environ.get("RAWFD_CHURN_SLOTS", "2"))


def _log(msg):
    print(f"[pid={os.getpid()}] {msg}", flush=True)


def _threaded_matmuls(torch, dev, nthreads, stop_evt, err_list):
    """Run continuous matmuls on nthreads private streams until stop_evt."""
    def loop(tid):
        try:
            gen = torch.Generator(device=dev)
            gen.manual_seed(1000 + tid)
            a = torch.randn(256, 256, device=dev, generator=gen)
            stream = torch.cuda.Stream(device=dev)
            with torch.cuda.stream(stream):
                while not stop_evt.is_set():
                    a = (a @ a).clamp(-1, 1) + 1e-3
            stream.synchronize()
        except Exception as exc:  # noqa: BLE001
            err_list.append(f"thread {tid}: {exc!r}")

    threads = [threading.Thread(target=loop, args=(t,), daemon=True)
               for t in range(nthreads)]
    for t in threads:
        t.start()
    return threads


# ── short-lived churn process ─────────────────────────────────────────────
def churn_worker(slot, ngpus, life_ms, jitter_ms):
    # Random delay so cuInit lands at unpredictable times relative to a
    # concurrently-running checkpoint's enumeration passes.
    time.sleep(random.uniform(0, jitter_ms / 1000.0))
    import torch

    dev = f"cuda:{slot % ngpus}"
    torch.cuda.set_device(slot % ngpus)

    # Deterministic self-check.
    gen = torch.Generator(device=dev)
    gen.manual_seed(42)
    a = torch.randn(128, 128, device=dev, generator=gen)
    expect = float((a @ a).sum().item())

    stop = threading.Event()
    errs = []
    threads = _threaded_matmuls(torch, dev, THREADS, stop, errs)
    time.sleep(life_ms / 1000.0)
    stop.set()
    for t in threads:
        t.join(timeout=30)

    got = float((a @ a).sum().item())
    if errs or got != expect:
        _log(f"churn slot {slot}: FAIL errs={errs} got={got} expect={expect}")
        os._exit(1)
    os._exit(0)


# ── NVML: frontendFD + RM objects, never CUDA, never toggled ────────────
def _nvml_open():
    import ctypes

    nvml = ctypes.CDLL("libnvidia-ml.so.1")
    if nvml.nvmlInit_v2() != 0:
        raise RuntimeError("nvmlInit_v2 failed")
    return nvml


def _nvml_snapshot(nvml):
    """Deterministic NVML state: (device count, dev0 name, dev0 total mem)."""
    import ctypes

    count = ctypes.c_uint()
    if nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
        raise RuntimeError("nvmlDeviceGetCount_v2 failed")
    handle = ctypes.c_void_p()
    if nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
        raise RuntimeError("nvmlDeviceGetHandleByIndex_v2 failed")
    name = ctypes.create_string_buffer(96)
    if nvml.nvmlDeviceGetName(handle, name, 96) != 0:
        raise RuntimeError("nvmlDeviceGetName failed")

    class Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong)]

    mem = Mem()
    if nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) != 0:
        raise RuntimeError("nvmlDeviceGetMemoryInfo failed")
    return {"count": count.value, "name": name.value.decode(),
            "total_mb": mem.total // (1 << 20)}


def nvml_long_worker(rank, cmd_q, res_q):
    nvml = _nvml_open()
    ref = _nvml_snapshot(nvml)

    # Continuous control-ioctl traffic so RM calls are in flight at
    # checkpoint time.
    def poll():
        while True:
            try:
                _nvml_snapshot(nvml)
            except Exception as exc:  # noqa: BLE001
                _log(f"nvml worker {rank}: poll FAILED: {exc!r}")
                return
            time.sleep(0.05)

    threading.Thread(target=poll, daemon=True).start()
    res_q.put(("ready", f"nvml:{rank}", os.getpid(), 0.0))
    _log(f"nvml worker {rank} ready: {ref}")
    while True:
        try:
            cmd = cmd_q.get(timeout=1.0)
        except Exception:
            continue
        if cmd[0] == "verify":
            gen_id = cmd[1]
            try:
                snap = _nvml_snapshot(nvml)
                snap["kind"] = "nvml"
                res_q.put(("verify", f"nvml:{rank}", gen_id, snap))
            except Exception as exc:  # noqa: BLE001
                res_q.put(("verify", f"nvml:{rank}", gen_id, {"error": repr(exc)}))


def nvml_churn_worker(slot, life_ms, jitter_ms):
    time.sleep(random.uniform(0, jitter_ms / 1000.0))
    nvml = _nvml_open()
    deadline = time.time() + life_ms / 1000.0
    while time.time() < deadline:
        _nvml_snapshot(nvml)
        time.sleep(0.05)
    # Half the time, exit WITHOUT nvmlShutdown: a dangling RM session whose
    # frontendFD is closed only by process exit.
    if random.random() < 0.5:
        nvml.nvmlShutdown()
    os._exit(0)


# ── raw device FDs: no RM objects at all, never toggled ─────────────────
def _open_nvidia_devs():
    fds = []
    for path in ["/dev/nvidiactl", "/dev/nvidia-uvm"] + [
        f"/dev/nvidia{i}" for i in range(16)
    ]:
        try:
            fds.append(os.open(path, os.O_RDWR))
        except OSError:
            pass
    return fds


def rawfd_long_worker(rank, cmd_q, res_q):
    fds = _open_nvidia_devs()
    res_q.put(("ready", f"raw:{rank}", os.getpid(), 0.0))
    _log(f"rawfd worker {rank} ready: {len(fds)} nvidia FDs held, no ioctls")
    while True:
        try:
            cmd = cmd_q.get(timeout=1.0)
        except Exception:
            continue
        if cmd[0] == "verify":
            gen_id = cmd[1]
            try:
                for fd in fds:
                    os.fstat(fd)
                res_q.put(("verify", f"raw:{rank}", gen_id,
                           {"kind": "rawfd", "fds": len(fds)}))
            except Exception as exc:  # noqa: BLE001
                res_q.put(("verify", f"raw:{rank}", gen_id, {"error": repr(exc)}))


def rawfd_churn_worker(slot, life_ms, jitter_ms):
    time.sleep(random.uniform(0, jitter_ms / 1000.0))
    fds = _open_nvidia_devs()
    time.sleep(life_ms / 1000.0)
    for fd in fds:
        os.close(fd)
    os._exit(0)


# ── long-lived verifier process ───────────────────────────────────────────
def long_worker(rank, ngpus, cmd_q, res_q):
    import torch

    dev = f"cuda:{rank % ngpus}"
    torch.cuda.set_device(rank % ngpus)
    elems = TENSOR_MB * 1024 * 1024 // 4
    gen = torch.Generator(device=dev)
    gen.manual_seed(100 + rank)
    persisted = torch.randn(elems, device=dev, generator=gen)
    torch.cuda.synchronize()

    stop = threading.Event()
    errs = []
    _threaded_matmuls(torch, dev, THREADS, stop, errs)
    res_q.put(("ready", f"cuda:{rank}", os.getpid(), float(persisted.sum().item())))
    _log(f"long worker {rank} ready on {dev} ({THREADS} matmul threads)")

    while True:
        try:
            cmd = cmd_q.get(timeout=1.0)
        except Exception:
            continue
        if cmd[0] == "verify":
            gen_id = cmd[1]
            try:
                g2 = torch.Generator(device=dev)
                g2.manual_seed(7)
                b = torch.randn(256, 256, device=dev, generator=g2)
                fresh = float((b @ b).sum().item())
                torch.cuda.synchronize()
                res_q.put(("verify", f"cuda:{rank}", gen_id, {
                    "kind": "cuda",
                    "persisted": float(persisted.sum().item()),
                    "fresh": fresh,
                    "thread_errors": list(errs),
                }))
            except Exception as exc:  # noqa: BLE001
                res_q.put(("verify", f"cuda:{rank}", gen_id, {"error": repr(exc)}))


# ── parent ────────────────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.lock = threading.Condition()
        self.ready = {}
        self.results = {}
        self.births = 0
        self.deaths_ok = 0
        self.deaths_fail = 0
        self.births_by_kind = {}
        self.start_time = time.time()


STATE = State()
CMD_QS = []
EXPECTED = []
GEN = [0]


def _churn_spawner(kind, target, args_fn, slot, ctx):
    while True:
        p = ctx.Process(target=target, args=args_fn(slot), daemon=True)
        p.start()
        with STATE.lock:
            STATE.births += 1
            STATE.births_by_kind[kind] = STATE.births_by_kind.get(kind, 0) + 1
        p.join()
        with STATE.lock:
            if p.exitcode == 0:
                STATE.deaths_ok += 1
            else:
                STATE.deaths_fail += 1
                _log(f"{kind} churn slot {slot}: pid={p.pid} exitcode={p.exitcode}")


def _collector(res_q):
    while True:
        msg = res_q.get()
        with STATE.lock:
            if msg[0] == "ready":
                STATE.ready[msg[1]] = msg[2]
            else:
                _, rank, gen_id, payload = msg
                STATE.results.setdefault(gen_id, {})[rank] = payload
            STATE.lock.notify_all()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        try:
            if path == "/health":
                any_churn = (CHURN_SLOTS + NVML_CHURN_SLOTS + RAWFD_CHURN_SLOTS) > 0
                with STATE.lock:
                    ok = len(STATE.ready) == len(EXPECTED) and (
                        STATE.deaths_ok > 0 or not any_churn)
                self._json(200 if ok else 503,
                           {"status": "ok" if ok else "starting",
                            "ready": sorted(STATE.ready)})
            elif path == "/stats":
                with STATE.lock:
                    up = time.time() - STATE.start_time
                    self._json(200, {
                        "births": STATE.births,
                        "births_by_kind": dict(sorted(STATE.births_by_kind.items())),
                        "deaths_ok": STATE.deaths_ok,
                        "deaths_fail": STATE.deaths_fail,
                        "alive_now": STATE.births - STATE.deaths_ok - STATE.deaths_fail,
                        "long_worker_pids": dict(sorted(STATE.ready.items())),
                        "uptime_s": round(up, 1),
                        "churn_per_s": round(STATE.deaths_ok / up, 2) if up else 0,
                        "config": {"long_workers": LONG_WORKERS,
                                   "nvml_workers": NVML_WORKERS,
                                   "rawfd_workers": RAWFD_WORKERS,
                                   "churn_slots": CHURN_SLOTS,
                                   "nvml_churn_slots": NVML_CHURN_SLOTS,
                                   "rawfd_churn_slots": RAWFD_CHURN_SLOTS,
                                   "threads": THREADS},
                    })
            elif path == "/verify":
                self._verify()
            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": repr(exc)})

    def _verify(self):
        GEN[0] += 1
        gen_id = GEN[0]
        for q in CMD_QS:
            q.put(("verify", gen_id))
        deadline = time.time() + VERIFY_TIMEOUT
        with STATE.lock:
            while (len(STATE.results.get(gen_id, {})) < len(EXPECTED)
                   and time.time() < deadline):
                STATE.lock.wait(timeout=1.0)
            got = dict(STATE.results.get(gen_id, {}))
        hung = [k for k in EXPECTED if k not in got]
        errors = {k: v for k, v in got.items()
                  if "error" in v or v.get("thread_errors")}
        self._json(200, {
            "gen": gen_id,
            "ranks": {k: got[k] for k in sorted(got)},
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
    # Parent must stay CUDA-free: count GPUs without initializing CUDA.
    ngpus = len([d for d in os.listdir("/dev") if d.startswith("nvidia")
                 and d[6:].isdigit()])
    ngpus = max(ngpus, 1)
    _log(f"cuda_churn: {ngpus} GPUs, {LONG_WORKERS} long workers, "
         f"{CHURN_SLOTS} churn slots x {THREADS} threads, "
         f"life {MIN_LIFE_MS}-{MAX_LIFE_MS}ms, jitter {JITTER_MS}ms")

    ctx = mp.get_context("spawn")
    res_q = ctx.Queue()

    def add_long(key, target, args):
        cmd_q = ctx.Queue()
        CMD_QS.append(cmd_q)
        EXPECTED.append(key)
        ctx.Process(target=target, args=args + (cmd_q, res_q), daemon=True).start()

    for rank in range(LONG_WORKERS):
        add_long(f"cuda:{rank}", long_worker, (rank, ngpus))
    for rank in range(NVML_WORKERS):
        add_long(f"nvml:{rank}", nvml_long_worker, (rank,))
    for rank in range(RAWFD_WORKERS):
        add_long(f"raw:{rank}", rawfd_long_worker, (rank,))

    threading.Thread(target=_collector, args=(res_q,), daemon=True).start()

    def spawn_churn(kind, nslots, target, args_fn):
        for slot in range(nslots):
            threading.Thread(target=_churn_spawner,
                             args=(kind, target, args_fn, slot, ctx),
                             daemon=True).start()

    spawn_churn("cuda", CHURN_SLOTS, churn_worker,
                lambda s: (s, ngpus, random.randint(MIN_LIFE_MS, MAX_LIFE_MS), JITTER_MS))
    spawn_churn("nvml", NVML_CHURN_SLOTS, nvml_churn_worker,
                lambda s: (s, random.randint(MIN_LIFE_MS, MAX_LIFE_MS), JITTER_MS))
    spawn_churn("rawfd", RAWFD_CHURN_SLOTS, rawfd_churn_worker,
                lambda s: (s, random.randint(MIN_LIFE_MS, MAX_LIFE_MS), JITTER_MS))

    server = HTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _log(f"HTTP on 0.0.0.0:{PORT}")
    while True:
        time.sleep(60)
        with STATE.lock:
            _log(f"stats: births={STATE.births} ok={STATE.deaths_ok} "
                 f"fail={STATE.deaths_fail}")


if __name__ == "__main__":
    main()
