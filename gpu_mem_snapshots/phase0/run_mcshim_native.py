#!/usr/bin/env python3
"""Native (no gVisor) e2e for the mcshim multicast interposer (Idea D, single
process).

Proves the interposition/track/replay CORE in isolation: a stock workload that
never calls any suspend API holds a LIVE multicast (0x00fd) object; the
LD_PRELOAD shim transparently tears the multicast layer down before
cuda-checkpoint and rebuilds it -- at IDENTICAL VAs -- after restore.

  CONTROL leg (optional, CONTROL=1): no shim suspend -> live multicast ->
    `cuda-checkpoint checkpoint` HANGS (the constraint that governs everything).
  MAIN leg:
    (a) pause workload  ->  (b) shim suspend  ->  (c) cuda-checkpoint
        lock/checkpoint/restore/unlock  ->  shim resume  ->  unpause  ->  verify.

Usage:
  sudo nvidia-smi -pm 1
  sudo python3 run_mcshim_native.py [--gpus 0,1] [--cuda-checkpoint PATH]
  sudo CONTROL=1 python3 run_mcshim_native.py     # also run the hang control
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = "/tmp/mcshim"


def sh(*a, **kw):
    return subprocess.run(a, capture_output=True, text=True, **kw)


def clean_dir():
    os.makedirs(DIR, exist_ok=True)
    for f in os.listdir(DIR):
        if f.startswith(("wl.status", "suspend", "resumed.", "suspended.",
                         "error", "pause", "mcgrp-")):
            os.remove(os.path.join(DIR, f))


def read(name):
    p = os.path.join(DIR, name)
    try:
        return open(p).read().strip()
    except OSError:
        return ""


def wait_status(pattern, timeout, poll=0.5):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        s = read("wl.status")
        if pattern in s:
            return s
        time.sleep(poll)
    return None


def wait_acks(prefix, count, timeout, poll=0.2):
    """Wait until `count` files named <prefix>.<pid> exist."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        acks = [f for f in os.listdir(DIR) if f.startswith(prefix + ".")]
        if len(acks) >= count:
            return True
        time.sleep(poll)
    return False


def errors():
    return [f for f in os.listdir(DIR) if f.startswith("error.")]


def touch(name):
    open(os.path.join(DIR, name), "w").close()


def rm(name):
    p = os.path.join(DIR, name)
    if os.path.exists(p):
        os.remove(p)


def launch(cc, gpus, preload):
    env = dict(os.environ)
    env["MCSHIM_DIR"] = DIR
    env["MCSHIM_LOG"] = os.path.join(DIR, "mcshim.log")
    if preload:
        env["LD_PRELOAD"] = os.path.join(HERE, "mcshim", "mcshim.so")
    else:
        env.pop("LD_PRELOAD", None)
    argv = [cc, "--launch-job", sys.executable,
            os.path.join(HERE, "mcshim_workload.py"),
            "--gpus", gpus, "--dir", DIR]
    return subprocess.Popen(argv, env=env)


def find_cuda_pid(cc, job_pid):
    out = sh("pgrep", "-P", str(job_pid))
    kids = [int(x) for x in out.stdout.split()]
    for pid in kids + [job_pid]:
        st = sh(cc, "--get-state", "--pid", str(pid))
        if "running" in (st.stdout + st.stderr):
            return pid
    return kids[0] if kids else job_pid


def action(cc, pid, act, timeout, extra=()):
    t0 = time.monotonic()
    try:
        p = subprocess.run([cc, "--action", act, "--pid", str(pid), *extra],
                           capture_output=True, text=True, timeout=timeout)
        rc, out, to = p.returncode, (p.stdout + p.stderr).strip(), False
    except subprocess.TimeoutExpired as e:
        rc, to = -1, True
        out = ((e.stdout or b"").decode() + (e.stderr or b"").decode()).strip()
    dt = time.monotonic() - t0
    print(f"[cc] {act}: rc={rc} {'TIMEOUT ' if to else ''}({dt:.1f}s) "
          f"out=[{out}]", flush=True)
    return rc, to


def control_leg(cc, gpus):
    print("\n==== CONTROL: live multicast, NO shim suspend ====", flush=True)
    clean_dir()
    job = launch(cc, gpus, preload=False)
    if not wait_status("mc-live pass", 120):
        print("[control] workload never became ready"); job.kill(); return
    pid = find_cuda_pid(cc, job.pid)
    action(cc, pid, "lock", 60, ("--timeout", "30000"))
    rc, to = action(cc, pid, "checkpoint", 60)
    if to:
        print("[control] EXPECTED: checkpoint HANGS on live multicast", flush=True)
    else:
        print(f"[control] UNEXPECTED: checkpoint returned rc={rc}", flush=True)
    job.kill()
    try:
        job.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def main_leg(cc, gpus):
    print("\n==== MAIN: shim transparently suspends/resumes multicast ====",
          flush=True)
    clean_dir()
    job = launch(cc, gpus, preload=True)
    if not wait_status("mc-live pass", 120):
        print("[main] workload never became ready"); job.kill(); return 1
    ready = read("wl.status")
    print(f"[main] workload ready: {ready}", flush=True)
    pid = find_cuda_pid(cc, job.pid)
    print(f"[main] cuda pid={pid} (job {job.pid})", flush=True)

    results = {}

    # (a) quiesce.
    touch("pause")
    if not wait_status("PAUSED", 30):
        print("[main] workload did not pause"); job.kill(); return 1

    # (b) shim suspend (transparent, in-process through libcuda).
    touch("suspend")
    if not wait_acks("suspended", 1, 30):
        print(f"[main] shim did not suspend (errors={errors()})")
        job.kill(); return 1
    print("[main] shim SUSPENDED (multicast released at libcuda level)",
          flush=True)

    # (c) cuda-checkpoint: the blocker is gone, so this must NOT hang.
    rc, _ = action(cc, pid, "lock", 60, ("--timeout", "30000"))
    results["lock"] = rc
    if rc == 0:
        rc, to = action(cc, pid, "checkpoint", 90)
        results["checkpoint"] = "HANG" if to else rc
        if rc == 0:
            rc, _ = action(cc, pid, "restore", 90)
            results["restore"] = rc
        action(cc, pid, "unlock", 30)

    # shim resume (recreate multicast at identical VAs): remove the marker.
    rm("suspend")
    if not wait_acks("resumed", 1, 60):
        print(f"[main] shim did not resume (errors={errors()})")
        results["resume"] = "FAIL"
    else:
        print("[main] shim RESUMED (multicast re-created at identical VAs)",
              flush=True)
        results["resume"] = 0

    # unpause + verify post-restore.
    rm("pause")
    final = wait_status("post-restore", 30)
    ok_iters = final is not None and "pass" in final and "FAIL" not in final
    time.sleep(3)
    final = read("wl.status")
    print(f"[main] final workload status: {final}", flush=True)

    job.kill()
    try:
        job.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass

    print("\n[main] ==== RESULTS ====", flush=True)
    print(f"[main] {results}", flush=True)
    ok = (results.get("checkpoint") == 0 and results.get("restore") == 0 and
          results.get("resume") == 0 and ok_iters and
          "FAIL" not in final and "pass" in final)
    print("[main] VERDICT:", "PASS" if ok else "FAIL", flush=True)
    if ok:
        print("[main] The generic libcuda interposer round-trips a live "
              "multicast process through cuda-checkpoint transparently: "
              "no NCCL fork, no engine hooks.", flush=True)
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--cuda-checkpoint",
                    default=os.environ.get("CUDA_CHECKPOINT",
                                           "/usr/local/bin/cuda-checkpoint"))
    args = ap.parse_args()

    if not os.path.exists(os.path.join(HERE, "mcshim", "mcshim.so")):
        print("mcshim.so not built; run mcshim/build.sh first", file=sys.stderr)
        return 1

    if os.environ.get("CONTROL") == "1":
        control_leg(args.cuda_checkpoint, args.gpus)
    return main_leg(args.cuda_checkpoint, args.gpus)


if __name__ == "__main__":
    sys.exit(main())
