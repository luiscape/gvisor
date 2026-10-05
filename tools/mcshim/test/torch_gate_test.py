# Copyright 2026 The gVisor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checks that the interposer's gate stops GPU work that PyTorch submits.

Runs in an image with PyTorch, with mcshim LD_PRELOADed (see run.sh). The
parent plays the sentry: it creates the gate marker, waits for the workers'
gated acks, and checks that their submission loops stop until it removes the
gate.

  torch_gate_test.py kernel  elementwise kernels, one process
  torch_gate_test.py symm    multimem all-reduce on symmetric memory, 2 ranks
"""

import multiprocessing as mp
import os
import sys
import time

D = "/tmp/mcshim"


def wait_for(path, timeout=120):
  end = time.time() + timeout
  while not os.path.exists(path):
    if time.time() > end:
      raise SystemExit(f"FAIL: timed out waiting for {path}")
    time.sleep(0.01)


def worker(rank, world, counter, pids):
  import torch  # pylint: disable=g-import-not-at-top

  torch.cuda.set_device(rank)
  if world > 1:
    import torch.distributed as dist  # pylint: disable=g-import-not-at-top
    import torch.distributed._symmetric_memory as symm_mem  # pylint: disable=g-import-not-at-top

    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world,
        init_method="tcp://127.0.0.1:29511",
        device_id=torch.device("cuda", rank),
    )
    buf = symm_mem.empty(1 << 20, device=f"cuda:{rank}")
    symm_mem.rendezvous(buf, dist.group.WORLD)
    name = dist.group.WORLD.group_name

    def step():
      torch.ops.symm_mem.multimem_all_reduce_(buf, "sum", name)

  else:
    x = torch.zeros(1 << 20, device="cuda")

    def step():
      x.add_(1)

  step()
  torch.cuda.synchronize()
  wait_for(f"{D}/present.{os.getpid()}")
  pids.put(os.getpid())
  while True:
    step()
    torch.cuda.synchronize()
    with counter.get_lock():
      counter.value += 1


def main():
  mode = sys.argv[1] if len(sys.argv) > 1 else "kernel"
  world = 2 if mode == "symm" else 1
  ctx = mp.get_context("spawn")
  counter = ctx.Value("q", 0)
  pids = ctx.Queue()
  procs = [
      ctx.Process(target=worker, args=(r, world, counter, pids), daemon=True)
      for r in range(world)
  ]
  for p in procs:
    p.start()
  ranks = [pids.get(timeout=300) for _ in procs]
  failed = []

  def check(ok, msg):
    if not ok:
      failed.append(msg)
      print(f"FAIL: {msg}", flush=True)

  c0 = counter.value
  time.sleep(0.5)
  check(counter.value > c0, "no progress before the gate")
  open(f"{D}/gate", "w").close()
  for pid in ranks:
    wait_for(f"{D}/gated.{pid}")
  time.sleep(0.3)  # an iteration already past its launch may still finish
  c1 = counter.value
  time.sleep(1.5)
  c2 = counter.value
  os.unlink(f"{D}/gate")
  time.sleep(1.5)
  c3 = counter.value
  check(c2 == c1, f"{c2 - c1} iterations while gated")
  check(c3 > c2, "no progress after the gate was removed")
  for p in procs:
    p.terminate()
  print(f"{mode}: before={c1 - c0} gated={c2 - c1} after={c3 - c2}: "
        f"{'FAIL' if failed else 'ok'}", flush=True)
  sys.exit(1 if failed else 0)


if __name__ == "__main__":
  main()
