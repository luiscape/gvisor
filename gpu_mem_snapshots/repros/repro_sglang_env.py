# Local adaptation of gcr/qwen_repro.py — SGLang GPU-snapshot repro whose
# outcome is decided by environment variables, with no Modal dependency and a
# small model (Qwen2.5-0.5B-Instruct by default).
#
# Lifecycle (mirrors the Modal snap=True/snap=False hooks, driven locally by
# run_sglang_env.sh):
#   startup()  start SGLang, wait ready, warm up, release_memory_occupation
#              (so GPU memory is excluded from the snapshot), then
#              dump_gpu_process_env() as the closest point to the snapshot.
#   the runner then: runsc checkpoint -> restore -> POST /wake -> /verify.
#
# The env vars that decide snapshot success (set by the runner, read by
# SGLang/NCCL/torch at startup, NOT flippable at runtime):
#   NCCL_CUMEM_ENABLE, NCCL_CUMEM_HOST_ENABLE  (restore-side; matter at TP>1)
#   TORCHINDUCTOR_COMPILE_THREADS              (checkpoint-side; fork FD leak)
#   SGLANG_USE_CUDA_IPC_TRANSPORT / _IPC_POOL_HANDLE_CACHE (IPC handles)
#
# A wrapper HTTP server on CTRL_PORT exposes /health (always responsive, unlike
# SGLang's /health which is dead while asleep), /wake, /verify.

import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PORT = int(os.environ.get("SGLANG_PORT", "30000"))
CTRL_PORT = int(os.environ.get("PORT", "8199"))
TP = int(os.environ.get("TP", "1"))
MEM_FRACTION = os.environ.get("MEM_FRACTION", "0.7")
CONTEXT_LEN = os.environ.get("CONTEXT_LEN", "4096")
TORCH_COMPILE = os.environ.get("TORCH_COMPILE", "0") == "1"
MINUTES = 60

# Env vars that matter for GPU memory snapshotting (see qwen_repro.py).
SNAPSHOT_RELEVANT_ENV_PREFIXES = (
    "NCCL",
    "CUMEM",
    "CUDA_VISIBLE",
    "NVIDIA_VISIBLE",
    "TORCHINDUCTOR",
    "SGLANG_USE",
)
REQUIRED_ZERO_ENV = ("NCCL_CUMEM_ENABLE", "NCCL_CUMEM_HOST_ENABLE")


def _url(path: str) -> str:
    return f"http://127.0.0.1:{PORT}{path}"


def _post(path: str, payload: dict, timeout: float):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        _url(path), data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(path: str, timeout: float = 2.0):
    with urllib.request.urlopen(_url(path), timeout=timeout) as r:
        return r.read()


def dump_gpu_process_env() -> None:
    """Print, for every process holding an open /dev/nvidia* FD, its PPid, comm,
    and snapshot-relevant environment as read from /proc/<pid>/environ.

    /proc/<pid>/environ is ground truth: it is what the process inherited at
    exec time, so this reliably confirms whether e.g. NCCL_CUMEM_ENABLE=0
    reached the SGLang worker subprocesses, and which parent forked the
    FD-holding workers.
    """
    # The CUDA driver names its per-session thread 'cuda' + 11 hex digits. A
    # holder WITHOUT such a thread has no live CUDA context of its own: its
    # /dev/nvidia* FDs are inherited-via-fork copies, which cuda-checkpoint
    # cannot checkpoint and which therefore block the snapshot.
    cuda_thread_re = re.compile(r"^cuda[0-9a-f]{11}$")

    def has_cuda_thread(pid: int) -> bool:
        try:
            for task in Path(f"/proc/{pid}/task").iterdir():
                try:
                    if cuda_thread_re.match((task / "comm").read_text().strip()):
                        return True
                except OSError:
                    continue
        except OSError:
            pass
        return False

    def cmdline(pid: int) -> str:
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return "?"
        argv = [a.decode(errors="replace") for a in raw.split(b"\x00") if a]
        line = " ".join(argv)
        return (line[:157] + "...") if len(line) > 160 else line

    def relevant_env(pid: int) -> dict[str, str]:
        try:
            raw = Path(f"/proc/{pid}/environ").read_bytes()
        except OSError:
            return {}
        env = {}
        for entry in raw.split(b"\x00"):
            if not entry or b"=" not in entry:
                continue
            key, _, val = entry.partition(b"=")
            k = key.decode(errors="replace")
            if k.startswith(SNAPSHOT_RELEVANT_ENV_PREFIXES):
                env[k] = val.decode(errors="replace")
        return env

    def read(pid: int, name: str) -> str:
        try:
            return Path(f"/proc/{pid}/{name}").read_text().strip()
        except OSError:
            return "?"

    def ppid(pid: int) -> str:
        for line in read(pid, "status").splitlines():
            if line.startswith("PPid:"):
                return line.split()[1]
        return "?"

    print("=== GPU FD-holder environment (pre-snapshot) ===", flush=True)
    holders = 0
    flagged: list[str] = []
    inherited_by_parent: dict[str, list[int]] = defaultdict(list)
    for pid_dir in sorted(
        Path("/proc").iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0
    ):
        if not pid_dir.name.isdigit():
            continue
        pid = int(pid_dir.name)
        nvidia_fds = []
        try:
            for fd in (pid_dir / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target.startswith("/dev/nvidia"):
                    nvidia_fds.append(target)
        except OSError:
            continue
        if not nvidia_fds:
            continue
        holders += 1
        env = relevant_env(pid)
        env_str = " ".join(f"{k}={v}" for k, v in sorted(env.items())) or "(none set)"

        pid_ppid = ppid(pid)
        if has_cuda_thread(pid):
            kind = "CUDA-session"
        else:
            kind = "INHERITED-fork-fd"
            inherited_by_parent[pid_ppid].append(pid)

        problems = [
            f"{k}={env.get(k, '<missing>')}"
            for k in REQUIRED_ZERO_ENV
            if env.get(k) != "0"
        ]
        if problems:
            flagged.append(f"PID {pid} ({read(pid, 'comm')}): {', '.join(problems)}")
        marker = f"  <-- BAD env: {', '.join(problems)}" if problems else ""

        print(
            f"PID {pid} comm={read(pid, 'comm')} ppid={pid_ppid} kind={kind} "
            f"fds={sorted(set(nvidia_fds))} cmdline={cmdline(pid)!r} env: {env_str}{marker}",
            flush=True,
        )
    if holders == 0:
        print("(no processes hold /dev/nvidia* FDs)", flush=True)

    # Env verdict (restore-side).
    if flagged:
        print(
            f"WARNING: {len(flagged)}/{holders} GPU FD-holder(s) missing "
            f"{' or '.join(REQUIRED_ZERO_ENV)}=0; these will likely fail restore "
            f"with 'operation not supported':",
            flush=True,
        )
        for line in flagged:
            print(f"  {line}", flush=True)
    else:
        print(
            f"OK env: all {holders} GPU FD-holder(s) have "
            f"{' and '.join(REQUIRED_ZERO_ENV)}=0",
            flush=True,
        )

    # FD verdict (checkpoint-side).
    total_inherited = sum(len(v) for v in inherited_by_parent.values())
    if total_inherited:
        print(
            f"BLOCKER: {total_inherited} holder(s) carry inherited-fork "
            f"/dev/nvidia* FDs that cuda-checkpoint cannot checkpoint. These must "
            f"be gone before snapshot:",
            flush=True,
        )
        for parent, kids in sorted(
            inherited_by_parent.items(),
            key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0,
        ):
            comm = read(int(parent), "comm") if parent.isdigit() else "?"
            cl = cmdline(int(parent)) if parent.isdigit() else "?"
            print(
                f"  {len(kids)} child(ren) of PID {parent} ({comm}) [{cl!r}]: {sorted(kids)}",
                flush=True,
            )
    else:
        print("OK fds: no inherited-fork /dev/nvidia* FD holders", flush=True)
    print("=== end GPU FD-holder environment ===", flush=True)


def start_server() -> subprocess.Popen:
    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--tp",
        str(TP),
        "--mem-fraction-static",
        MEM_FRACTION,
        "--context-length",
        CONTEXT_LEN,
        "--dtype",
        "float16",
        "--attention-backend",
        "triton",
        "--sampling-backend",
        "pytorch",
        "--enable-memory-saver",
        "--enable-weights-cpu-backup",
    ]
    if TORCH_COMPILE:
        cmd.append("--enable-torch-compile")
    print("Starting SGLang:", " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, start_new_session=True)


def wait_ready(proc: subprocess.Popen, timeout: int = 15 * MINUTES):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"SGLang exited early rc={proc.returncode}")
        try:
            _get("/health_generate", timeout=3)
            return
        except Exception:  # noqa: BLE001
            time.sleep(3)
    raise TimeoutError("SGLang not ready")


def chat(prompt: str, max_tokens: int = 8) -> str:
    resp = _post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "seed": 42,
        },
        timeout=120,
    )
    return resp["choices"][0]["message"]["content"]


def warmup():
    chat("Hi", max_tokens=3)


def sleep_server(timeout: int = 5 * MINUTES):
    print("release_memory_occupation (sleep)", flush=True)
    _post("/release_memory_occupation", {}, timeout=timeout)
    print("SGLang asleep", flush=True)


def _retry_post(path: str, deadline: float, per_call: float = 30.0) -> bool:
    """POST until it succeeds or deadline. After restore the HTTP server
    answers before the scheduler has finished GPU-restore, so resume /
    continue must be retried (mirrors cr-bench SGLang handling)."""
    while time.time() < deadline:
        try:
            _post(path, {}, timeout=per_call)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return False


def wake_server(timeout: int = 5 * MINUTES):
    deadline = time.time() + timeout
    print("resume_memory_occupation (wake)", flush=True)
    if not _retry_post("/resume_memory_occupation", deadline):
        raise RuntimeError("resume_memory_occupation never accepted")
    # SGLang's /health tracks detokenizer heartbeats, dead while paused;
    # continue_generation restarts them.
    _retry_post("/continue_generation", min(deadline, time.time() + 60))
    # Readiness = a real generation succeeds.
    while time.time() < deadline:
        try:
            chat("Hi", max_tokens=3)
            print("SGLang awake", flush=True)
            return
        except Exception:  # noqa: BLE001
            time.sleep(2)
    raise RuntimeError("SGLang did not become generation-ready after wake")


def main():
    proc = start_server()
    wait_ready(proc)
    warmup()
    # Reference answer captured while awake, before releasing GPU memory.
    ref_answer = chat("Capital of France? One word.")
    print(f"reference answer (pre-sleep): {ref_answer!r}", flush=True)
    sleep_server()
    dump_gpu_process_env()  # closest observable point to the snapshot

    state = {"proc": proc, "awake": False, "ref": ref_answer}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            try:
                if path == "/health":
                    self._json(200, {"status": "ok", "awake": state["awake"]})
                elif path == "/wake":
                    wake_server()
                    wait_ready(state["proc"])
                    state["awake"] = True
                    self._json(200, {"status": "awake"})
                elif path == "/verify":
                    ans = chat("Capital of France? One word.")
                    self._json(
                        200,
                        {
                            "answer": ans,
                            "ref": state["ref"],
                            "match": ans == state["ref"],
                        },
                    )
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

    server = HTTPServer(("0.0.0.0", CTRL_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"READY, control server on :{CTRL_PORT}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
