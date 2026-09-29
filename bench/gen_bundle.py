#!/usr/bin/env python3
"""Writes a bench run's OCI config.json to stdout. Inputs come from env."""
import json
import os
import sys

e = os.environ
gpus = [g for g in e["GPUS"].split(",") if g]

# Image env, then overrides (a later KEY= replaces an earlier one).
env = [l for l in open(e["ENVFILE"]).read().splitlines() if l.strip()]
overrides = ["HOME=/root", "HF_HOME=/hf", "HF_HUB_OFFLINE=1",
             "NVIDIA_VISIBLE_DEVICES=" + ",".join(gpus)]
overrides += [l for l in e.get("EXTRA_ENV", "").splitlines() if l.strip()]
for kv in overrides:
    key = kv.split("=", 1)[0]
    env = [x for x in env if x.split("=", 1)[0] != key] + [kv]

# Docker's default set, plus CAP_SYS_PTRACE unless NO_PTRACE_CAP=1 (without
# it, the interposer's pidfd_getfd relies on PR_SET_PTRACER_ANY).
caps = ["CAP_CHOWN", "CAP_DAC_OVERRIDE", "CAP_FSETID", "CAP_FOWNER",
        "CAP_MKNOD", "CAP_NET_RAW", "CAP_SETGID", "CAP_SETUID", "CAP_SETFCAP",
        "CAP_SETPCAP", "CAP_NET_BIND_SERVICE", "CAP_SYS_CHROOT", "CAP_KILL",
        "CAP_AUDIT_WRITE"]
if e.get("NO_PTRACE_CAP", "0") != "1":
    caps.append("CAP_SYS_PTRACE")

uvm = os.stat("/dev/nvidia-uvm").st_rdev
uvm_tools = os.stat("/dev/nvidia-uvm-tools").st_rdev


def dev(path, major, minor):
    return {"path": path, "type": "c", "major": major, "minor": minor,
            "fileMode": 0o666, "uid": 0, "gid": 0}


devices = [dev(f"/dev/nvidia{g}", 195, int(g)) for g in gpus]
devices += [dev("/dev/nvidiactl", 195, 255),
            dev("/dev/nvidia-uvm", os.major(uvm), os.minor(uvm)),
            dev("/dev/nvidia-uvm-tools", os.major(uvm_tools), os.minor(uvm_tools))]

# /tmp is a sentry tmpfs, so the interposer's rendezvous dir (/tmp/mcshim) is
# part of the checkpoint image, as it must be.
mounts = [
    {"destination": "/proc", "type": "proc", "source": "proc"},
    {"destination": "/dev", "type": "tmpfs", "source": "tmpfs",
     "options": ["nosuid", "strictatime", "mode=755", "size=65536k"]},
    {"destination": "/dev/pts", "type": "devpts", "source": "devpts",
     "options": ["nosuid", "noexec", "newinstance", "ptmxmode=0666", "mode=0620"]},
    {"destination": "/dev/shm", "type": "tmpfs", "source": "shm",
     "options": ["nosuid", "noexec", "nodev", "mode=1777", "size=17179869184"]},
    {"destination": "/sys", "type": "sysfs", "source": "sysfs",
     "options": ["nosuid", "noexec", "nodev", "ro"]},
    {"destination": "/tmp", "type": "tmpfs", "source": "tmpfs",
     "options": ["nosuid", "mode=1777"]},
    {"destination": "/run", "type": "tmpfs", "source": "tmpfs",
     "options": ["nosuid", "strictatime", "mode=755", "size=65536k"]},
    {"destination": "/applog", "type": "bind", "source": e["APPLOG"],
     "options": ["rbind", "rw"]},
    {"destination": "/hf", "type": "bind", "source": e["HF"],
     "options": ["rbind", "rw"]},
]

unlimited = 2**64 - 1
spec = {
    "ociVersion": "1.0.0",
    "process": {
        "terminal": False,
        "user": {"uid": 0, "gid": 0},
        "args": ["sh", "-c", e["CMD"]],
        "env": env,
        "cwd": e.get("CWD") or "/",
        "capabilities": {k: caps for k in
                         ("bounding", "effective", "permitted", "ambient")},
        "rlimits": [
            {"type": "RLIMIT_NOFILE", "hard": 1048576, "soft": 1048576},
            {"type": "RLIMIT_MEMLOCK", "hard": unlimited, "soft": unlimited},
        ],
    },
    "root": {"path": e["ROOTFS"], "readonly": False},
    "hostname": e["NAME"],
    "mounts": mounts,
    "linux": {
        "namespaces": [{"type": t} for t in
                       ("pid", "mount", "ipc", "uts", "network")],
        "devices": devices,
    },
}
json.dump(spec, sys.stdout, indent=2)
print()
