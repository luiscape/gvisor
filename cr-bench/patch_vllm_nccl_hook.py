#!/usr/bin/env python3
"""Patch vLLM's per-worker sleep/wake_up to call the NCCL suspend/resume API.

Prototype of the "engine fork" arm: rather than NCCL's control thread being
driven by gVisor markers, the engine itself calls ncclCommSuspend as part of
its existing sleep lifecycle, and ncclCommResume as part of wake_up.

Both hooks are placed to satisfy the API's ordering contract:

  * suspend at the END of Worker.sleep, so vLLM has already stopped
    scheduling, synchronized the device and released what it releases --
    ncclCommSuspend copies UC contents to a CPU backup and cannot run under a
    live collective.
  * resume at the START of Worker.wake_up, so the communicators are whole
    before the engine issues anything collective. ncclCommResume is itself
    collective (cuMulticastBindMem blocks until every device re-joins), and
    vLLM runs wake_up in all workers concurrently via collective_rpc, which is
    what makes that safe.

Idempotent: running it twice is a no-op. Usage:
    python3 patch_vllm_nccl_hook.py <rootfs>
"""

import os
import re
import sys

MARK = "# --- vllm_nccl_ckpt hook (cr-bench) ---"

SLEEP_HOOK = f"""        {MARK}
        try:
            import vllm_nccl_ckpt
            vllm_nccl_ckpt.suspend()
        except Exception as _e:
            import logging
            logging.getLogger(__name__).error("NCCL suspend hook failed: %s", _e)
            raise
"""

WAKE_HOOK = f"""        {MARK}
        try:
            import vllm_nccl_ckpt
            vllm_nccl_ckpt.resume()
        except Exception as _e:
            import logging
            logging.getLogger(__name__).error("NCCL resume hook failed: %s", _e)
            raise
"""


def patch(path):
    with open(path) as f:
        src = f.read()
    if MARK in src:
        print(f"already patched: {path}")
        return True

    lines = src.split("\n")
    out = []
    i = 0
    patched_sleep = patched_wake = False
    while i < len(lines):
        line = lines[i]
        out.append(line)

        # Insert the resume hook as the first statement of wake_up().
        if re.match(r"^    def wake_up\(", line):
            # Skip over the signature continuation and any docstring.
            j = i + 1
            while j < len(lines) and not lines[j].strip().endswith(":") and lines[j].strip() and not lines[j].startswith("        "):
                out.append(lines[j]); j += 1
            out.append(WAKE_HOOK.rstrip("\n"))
            patched_wake = True
            i += 1
            continue

        i += 1

    src2 = "\n".join(out)

    # Insert the suspend hook at the end of sleep(): find the def, then the
    # next def at the same indentation, and splice in just before it.
    m = re.search(r"\n    def sleep\(", src2)
    if m is None:
        print(f"ERROR: no Worker.sleep in {path}", file=sys.stderr)
        return False
    nxt = re.search(r"\n    def (?!sleep)", src2[m.end():])
    if nxt is None:
        print(f"ERROR: cannot find end of sleep() in {path}", file=sys.stderr)
        return False
    cut = m.end() + nxt.start()
    src2 = src2[:cut] + "\n" + SLEEP_HOOK.rstrip("\n") + src2[cut:]
    patched_sleep = True

    with open(path, "w") as f:
        f.write(src2)
    print(f"patched {path} (sleep={patched_sleep} wake_up={patched_wake})")
    return patched_sleep and patched_wake


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    rootfs = sys.argv[1]
    target = None
    for py in ("python3.12", "python3.10", "python3.11"):
        p = os.path.join(rootfs, "usr/local/lib", py,
                         "dist-packages/vllm/v1/worker/gpu_worker.py")
        if os.path.exists(p):
            target = p
            break
    if target is None:
        print("ERROR: could not find vllm/v1/worker/gpu_worker.py in rootfs",
              file=sys.stderr)
        return 1
    return 0 if patch(target) else 1


if __name__ == "__main__":
    sys.exit(main())
