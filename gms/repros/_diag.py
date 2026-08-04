# Dependency-free diagnostic helper shared by all GPU-memory-snapshot repros.
#
# Dumps every process in the container that holds an open /dev/nvidia* file
# descriptor. Called as the LAST line of each @modal.enter(snap=True) hook,
# i.e. at the closest observable point to the snapshot, so the output shows
# exactly which nvidia FD holders the checkpointer must deal with.

import os
import time


def nvidia_fd_paths(pid: int) -> list[str]:
    """Return the /dev/nvidia* fd targets currently open in <pid>."""
    paths = set()
    fd_dir = f"/proc/{pid}/fd"
    try:
        for fd in os.listdir(fd_dir):
            try:
                target = os.readlink(f"{fd_dir}/{fd}")
            except OSError:
                continue
            if target.startswith("/dev/nvidia"):
                paths.add(target)
    except OSError:
        pass
    return sorted(paths)


def wait_for_nvidia_fd(pid: int, timeout: float = 120.0) -> None:
    """Block until <pid> has an open /dev/nvidia* fd (what the dump checks)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if nvidia_fd_paths(pid):
            return
        time.sleep(0.25)
    raise RuntimeError(f"pid {pid} never opened a /dev/nvidia* fd")


def dump_pre_snapshot_state(label: str) -> None:
    """Print `PID <pid> (<comm>): [paths]` for every /dev/nvidia* FD holder."""
    holders: dict[int, tuple[str, list[str]]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            paths = set()
            fd_dir = f"/proc/{entry}/fd"
            for fd in os.listdir(fd_dir):
                try:
                    target = os.readlink(f"{fd_dir}/{fd}")
                except OSError:
                    continue
                if target.startswith("/dev/nvidia"):
                    paths.add(target)
            if not paths:
                continue
            with open(f"/proc/{entry}/comm") as f:
                comm = f.read().strip()
            holders[pid] = (comm, sorted(paths))
        except OSError:
            # Processes exit mid-scan; permissions may vary.
            continue

    print(f"=== [{label}] nvidia FD holders: {len(holders)} ===", flush=True)
    for pid in sorted(holders):
        comm, paths = holders[pid]
        print(f"PID {pid} ({comm}): {paths}", flush=True)
    print(f"=== [{label}] end ===", flush=True)
