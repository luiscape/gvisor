#!/usr/bin/env python3
"""Native (no gVisor) e2e for the mcshim interposer, MULTI-PROCESS ranks
(one process per GPU -- the vLLM/SGLang TP topology).

The launcher + all rank processes form one cuda-checkpoint job
(--launch-job). Ranks share one multicast group: rank 0 created+exported it,
peers imported it. The shims transparently:

  suspend: each rank unmaps its MC VA (reservation kept), unbinds its local
           memory, releases its group handle (creator + imported refs).
  resume:  rank 0's shim recreates + re-exports the group and serves the new
           fd on a unix socket; peer shims fetch + re-import; all ranks
           re-add + re-bind (BindMem blocking until all devices join = the
           cross-rank barrier) and re-map at IDENTICAL VAs.

Usage:
  sudo python3 run_mcshim_mp_native.py [--world 2] [--cuda-checkpoint PATH]
  sudo CONTROL=1 python3 run_mcshim_mp_native.py   # also run the hang control
"""

import argparse
import os
import signal
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
        if f.startswith(("wl.", "suspend", "resumed.", "suspended.",
                         "error", "pause", "mcgrp-")):
            os.remove(os.path.join(DIR, f))


def rank_status(r):
    try:
        return open(os.path.join(DIR, f"wl.status.rank{r}")).read().strip()
    except OSError:
        return ""


def wait_all_status(world, pattern, timeout, poll=0.5):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if all(pattern in rank_status(r) for r in range(world)):
            return True
        time.sleep(poll)
    return False


def wait_acks(prefix, count, timeout, poll=0.2):
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


def rank_pids(world):
    pids = []
    for r in range(world):
        pids.append(int(open(os.path.join(DIR, f"wl.pid.rank{r}")).read()))
    return pids


def launch(cc, world, preload):
    env = dict(os.environ)
    env["MCSHIM_DIR"] = DIR
    env["MCSHIM_LOG"] = os.path.join(DIR, "mcshim.log")
    if preload:
        env["LD_PRELOAD"] = os.path.join(HERE, "mcshim", "mcshim.so")
    else:
        env.pop("LD_PRELOAD", None)
    argv = [cc, "--launch-job", sys.executable,
            os.path.join(HERE, "mcshim_mp.py"),
            "--world", str(world), "--dir", DIR]
    # Own session: the launcher forks rank children, and killing only the
    # launcher would reparent them (they then hold our stdout pipe open
    # forever). kill_job() nukes the whole process group instead.
    return subprocess.Popen(argv, env=env, start_new_session=True)


def kill_job(job):
    try:
        os.killpg(os.getpgid(job.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        job.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


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
    print(f"[cc] {act} pid={pid}: rc={rc} {'TIMEOUT ' if to else ''}({dt:.1f}s)"
          f" out=[{out}]", flush=True)
    return rc, to


def control_leg(cc, world):
    print(f"\n==== CONTROL: {world} live-multicast ranks, NO shim suspend ====",
          flush=True)
    clean_dir()
    job = launch(cc, world, preload=False)
    if not wait_all_status(world, "mc-live pass", 180):
        print("[control] ranks never became ready"); kill_job(job); return
    pid = rank_pids(world)[0]
    action(cc, pid, "lock", 60, ("--timeout", "30000"))
    rc, to = action(cc, pid, "checkpoint", 60)
    if to:
        print("[control] EXPECTED: checkpoint HANGS on live multicast",
              flush=True)
    else:
        print(f"[control] UNEXPECTED: checkpoint rc={rc}", flush=True)
    kill_job(job)
    time.sleep(3)


def main_leg(cc, world):
    print(f"\n==== MAIN: {world} ranks, transparent shim suspend/resume ====",
          flush=True)
    clean_dir()
    job = launch(cc, world, preload=True)
    if not wait_all_status(world, "mc-live pass", 180):
        print("[main] ranks never became ready:")
        for r in range(world):
            print(f"  rank{r}: {rank_status(r)}")
        kill_job(job); return 1
    pids = rank_pids(world)
    print(f"[main] all {world} ranks ready; pids={pids}", flush=True)
    for r in range(world):
        print(f"  rank{r}: {rank_status(r)}")

    results = {}

    # (a) quiesce all ranks.
    touch("pause")
    if not wait_all_status(world, "PAUSED", 30):
        print("[main] ranks did not pause"); kill_job(job); return 1

    # (b) shim suspend on every rank.
    touch("suspend")
    if not wait_acks("suspended", world, 60):
        print(f"[main] shims did not all suspend (errors={errors()})")
        kill_job(job); return 1
    print(f"[main] all {world} shims SUSPENDED", flush=True)

    # (c) cuda-checkpoint job: sequential per-pid lock/checkpoint/restore.
    ok = True
    for pid in pids:
        rc, _ = action(cc, pid, "lock", 60, ("--timeout", "30000"))
        ok &= rc == 0
    results["lock"] = ok
    if ok:
        for pid in pids:
            rc, to = action(cc, pid, "checkpoint", 120)
            ok &= rc == 0 and not to
        results["checkpoint"] = ok
    if ok:
        for pid in pids:
            rc, _ = action(cc, pid, "restore", 120)
            ok &= rc == 0
        results["restore"] = ok
    for pid in pids:
        action(cc, pid, "unlock", 30)

    # shim resume on every rank (creator serves fd; peers re-import).
    rm("suspend")
    if not wait_acks("resumed", world, 90):
        print(f"[main] shims did not all resume (errors={errors()})")
        results["resume"] = False
    else:
        print(f"[main] all {world} shims RESUMED", flush=True)
        results["resume"] = True

    # unpause + verify post-restore on every rank.
    rm("pause")
    verified = wait_all_status(world, "post-restore+mc-live pass", 60)
    time.sleep(3)
    finals = [rank_status(r) for r in range(world)]
    for r, s in enumerate(finals):
        print(f"[main] final rank{r}: {s}", flush=True)

    kill_job(job)

    print("\n[main] ==== RESULTS ====", flush=True)
    print(f"[main] {results}", flush=True)
    ok = (all(results.get(k) for k in ("lock", "checkpoint", "restore",
                                       "resume"))
          and verified
          and all("post-restore+mc-live pass" in s and "failures=0" in s
                  for s in finals))
    print("[main] VERDICT:", "PASS" if ok else "FAIL", flush=True)
    if ok:
        print("[main] Multi-process multicast (create/export/import topology) "
              "round-trips cuda-checkpoint transparently: the shim brokered "
              "the cross-rank fd rendezvous itself.", flush=True)
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=int(os.environ.get("WORLD", 2)))
    ap.add_argument("--cuda-checkpoint",
                    default=os.environ.get("CUDA_CHECKPOINT",
                                           "/usr/local/bin/cuda-checkpoint"))
    args = ap.parse_args()

    if not os.path.exists(os.path.join(HERE, "mcshim", "mcshim.so")):
        print("mcshim.so not built; run mcshim/build.sh first", file=sys.stderr)
        return 1

    if os.environ.get("CONTROL") == "1":
        control_leg(args.cuda_checkpoint, args.world)
    return main_leg(args.cuda_checkpoint, args.world)


if __name__ == "__main__":
    sys.exit(main())
