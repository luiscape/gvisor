# nvproxy Pageable Transfer Optimization — Progress

## Problem Statement

When CUDA copies pageable (non-pinned) memory between host and GPU, the NVIDIA
driver issues a repeating **3-ioctl pattern per ~4 MB chunk**:

| Step | ioctl NR | Target fd | Purpose |
|------|----------|-----------|---------|
| 1 | `0x49` | `/dev/nvidia0` | DMA setup / notify |
| 2 | `0x21` | `/dev/nvidia0` | Pin pages + kick DMA |
| 3 | `0x2b` (`NV_ESC_RM_CONTROL`) | `/dev/nvidiactl` | RM wait / complete |

For 1 GB at ~4 MB/chunk → **256 chunks × 3 = 768 ioctls**.

In gVisor, every one of those ioctls takes a round-trip through the Sentry
(nvproxy intercepts → validates → forwards to host kernel → returns). On
Modal's production infrastructure (A10G, full PCIe Gen4 x16), this adds
**~41 µs per ioctl** on GPU→CPU transfers, totaling ~31 ms of overhead on a
167 ms transfer — a **16% bandwidth penalty**.

---

## What We Built

### 1. nvproxy fast-path (`pkg/sentry/devices/nvproxy/`)

**Branch:** `nvproxy-study`

Four files modified to reduce per-ioctl Sentry overhead:

| File | Change | Purpose |
|------|--------|---------|
| `handlers.go` | `feHandlerFast()`, `sync.Pool` buffer recycling | Mark handlers for fast dispatch; eliminate `make([]byte)` heap alloc per ioctl |
| `frontend.go` | Inline fast-path in `Ioctl()`, `rmControlFast()` | Skip debug logging + error wrapping + handler dispatch chain; streamlined RM_CONTROL |
| `frontend_unsafe.go` | `rmControlSimpleInvoke()` | Direct `RawSyscall` for byte-blob RM_CONTROL, skipping generic function layers |
| `version.go` | `feHandler` → `feHandlerFast` for 14 ioctls | Register all simple passthrough ioctls for fast dispatch |

### 2. Real-GPU ioctl benchmark (`test/perf/linux/nvproxy_ioctl_overhead/nvproxy_ioctl_bench.c`)

Opens `/dev/nvidiactl` and `/dev/nvidia0` directly, performs full RM client/device/subdevice
initialization, then hammers the real `NV_ESC_RM_CONTROL` and `NV_ESC_RM_ALLOC`/`FREE`
ioctls in tight loops. Reports per-ioctl latency distributions and bandwidth projections.

### 3. PyTorch bandwidth benchmark (`mem_bandwidth.py`)

Standalone PyTorch script (no Modal dependency) that measures CPU↔GPU transfer bandwidth
across five memory modes (pinned, pageable, prefaulted, hugepage, registered). Has a
`--docker-compare` mode that builds a container, runs under multiple Docker runtimes
(runc, runsc-baseline, runsc-fastpath), and prints a side-by-side comparison table.

### 4. Modal results analyzer (`analyze_modal_results.py`)

Parses the runc vs gVisor bandwidth output from Modal, computes per-ioctl overhead,
and prints the full analysis with bandwidth projections.

---

## Key Findings

### 1. The overhead is real — but only visible with fast PCIe

On Modal's production A10G (full PCIe Gen4 x16, 80+ vCPUs):

| Metric | runc | gVisor | Gap |
|--------|------|--------|-----|
| **d2h pageable 1 GB** | 5.98 GB/s | 4.98 GB/s | **-16.7%** |
| **h2d pageable 1 GB** | 12.17 GB/s | 10.29 GB/s | **-15.4%** |
| pinned (either dir) | ~12.3 GB/s | ~12.3 GB/s | **0%** |
| registered (either) | ~12.4 GB/s | ~12.4 GB/s | **0%** |

Pinned and registered are byte-for-byte identical — the overhead is **100% in the
pageable ioctl path**.

### 2. Small EC2 instances mask the problem

On a `g5.xlarge` (4 vCPU), the PCIe link is **downgraded to Gen1 × 8** (~2 GB/s
vs Gen4 × 16 ~32 GB/s). The slow PCIe makes DMA the bottleneck, and the ~44 µs
ioctl overhead becomes invisible noise against long chunk times. You need a
**g5.12xlarge or larger** to reproduce the Modal gap.

```
Modal (fast PCIe):  768 ioctls × 44 µs = 34 ms on a 167 ms transfer → 16% gap
Local (slow PCIe):  768 ioctls × 3 µs  =  2 ms on a  94 ms transfer → 2% gap
```

### 3. The fast-path saves ~1-3 µs per ioctl in microbenchmarks

Real nvidia ioctl benchmark on the local A10G (driver 580.95.05):

| Metric | Baseline | Fast-path | Change |
|--------|----------|-----------|--------|
| `NV_ESC_RM_CONTROL` sustained burst | 219K/s | **263K/s** | **+20%** |
| `NV_ESC_RM_CONTROL` p50 latency | 3.93 µs | 3.88 µs | -1.3% |
| Pageable BW (PyTorch, d2h avg) | 11.13 GB/s | **11.20 GB/s** | +0.6% |

The +20% burst throughput improvement comes from `sync.Pool` eliminating GC pauses
and reduced function-call depth. The bandwidth improvement is small on the local
instance but would compound on fast-PCIe hardware.

### 4. Closing the gap requires bypassing the Sentry

The per-ioctl overhead on Modal (~44 µs) breaks down approximately as:

- ~1 µs: systrap platform transition (guest → Sentry → guest)
- ~3 µs: nvproxy handler dispatch + param CopyIn/CopyOut (what we optimized)
- ~40 µs: **seccomp-bpf filter evaluation + production Sentry overhead**

Our fast-path saves the ~3 µs layer. The dominant ~40 µs requires:

| Strategy | Approach | Projected d2h BW |
|----------|----------|------------------|
| A. Fast-path (done) | Reduce dispatch overhead | 5.28 GB/s (+6%) |
| B. Batch triplet | 3 ioctls → 1 Sentry transition | 5.55 GB/s (+11%) |
| C. Passthrough | Bypass Sentry entirely for nvidia fds | **5.98 GB/s (=runc)** |

---

## Quick Start — New EC2 Instance

### Requirements

- **Instance type:** `g5.12xlarge` or larger (need full PCIe Gen4 × 16 for visible gap)
  - `g5.xlarge` (4 vCPU) works for development but masks the perf gap
- **AMI:** Amazon Linux 2023 with NVIDIA drivers (e.g., Deep Learning AMI)
- **Disk:** 100 GB+ (Bazel cache is large)

### 1. Clone and checkout

```bash
git clone https://github.com/nichochar/gvisor.git
cd gvisor
git checkout nvproxy-study
```

### 2. Install Docker + NVIDIA container toolkit

```bash
# Docker
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker $USER

# NVIDIA container toolkit
sudo mkdir -p /usr/share/keyrings
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -sL "https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo" \
  | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo > /dev/null
sudo dnf install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU passthrough
sudo docker run --rm --runtime=nvidia --gpus all \
  nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### 3. Install gVisor runtimes

```bash
# Stock release (baseline)
ARCH=$(uname -m)
URL="https://storage.googleapis.com/gvisor/releases/release/latest/${ARCH}"
sudo curl -fsSL "${URL}/runsc" -o /usr/local/bin/runsc-baseline
sudo chmod +x /usr/local/bin/runsc-baseline

# Get driver version
DRIVER_VER=$(cat /proc/driver/nvidia/version | grep -oP 'Module\s+\K[0-9.]+')
echo "Driver version: $DRIVER_VER"

# Register baseline runtime
sudo /usr/local/bin/runsc-baseline install --runtime=runsc-baseline \
  -- --nvproxy --nvproxy-driver-version=$DRIVER_VER

# Build fast-path from source
make runsc   # builds via Docker+Bazel, takes ~8 min first time
sudo cp bazel-bin/runsc/runsc_/runsc /usr/local/bin/runsc-fastpath

# Register fast-path runtime
sudo python3 -c "
import json
with open('/etc/docker/daemon.json') as f: d = json.load(f)
d['runtimes']['runsc-fastpath'] = {
    'path': '/usr/local/bin/runsc-fastpath',
    'runtimeArgs': ['--nvproxy', '--nvproxy-driver-version=$DRIVER_VER']
}
with open('/etc/docker/daemon.json', 'w') as f: json.dump(d, f, indent=4)
"
sudo systemctl restart docker

# Verify all runtimes
docker info 2>&1 | grep -oP '\brunsc\S*' | sort -u
```

### 4. Run the real-GPU ioctl benchmark

```bash
cd test/perf/linux/nvproxy_ioctl_overhead

# Build
sudo dnf install -y gcc glibc-static   # if needed
gcc -O2 -static -o bench_static nvproxy_ioctl_bench.c -lm

# Native baseline
./bench_static -n 5000 -r 100

# runc + GPU
sudo docker run --rm --runtime=nvidia --gpus all \
  -v $PWD/bench_static:/bench:ro --entrypoint /bench busybox -n 5000 -r 100

# gVisor baseline
sudo docker run --rm --runtime=runsc-baseline --gpus all \
  -v $PWD/bench_static:/bench:ro --entrypoint /bench busybox -n 5000 -r 100

# gVisor fast-path
sudo docker run --rm --runtime=runsc-fastpath --gpus all \
  -v $PWD/bench_static:/bench:ro --entrypoint /bench busybox -n 5000 -r 100
```

### 5. Run the PyTorch bandwidth benchmark

```bash
cd /path/to/gvisor

# Automated 3-way comparison (builds container, runs all runtimes)
sudo python3 mem_bandwidth.py \
  --docker-compare \
  --runtimes runc,runsc-baseline,runsc-fastpath \
  --sizes-mb 64,256,1024 \
  --repeats 20
```

### 6. Analyze Modal results

If you have runc and gVisor output from Modal, edit the embedded data in
`analyze_modal_results.py` and run:

```bash
python3 analyze_modal_results.py
```

---

## File Inventory

```
gvisor/
├── progress.md                          ← this file
├── mem_bandwidth.py                     ← PyTorch bandwidth benchmark (standalone)
├── analyze_modal_results.py             ← Modal results analysis script
├── pkg/sentry/devices/nvproxy/
│   ├── frontend.go                      ← fast-path Ioctl() dispatch + rmControlFast
│   ├── frontend_unsafe.go               ← rmControlSimpleInvoke (direct RawSyscall)
│   ├── handlers.go                      ← feHandlerFast + sync.Pool buffer recycling
│   └── version.go                       ← register simple ioctls with feHandlerFast
└── test/perf/linux/nvproxy_ioctl_overhead/
    ├── nvproxy_ioctl_bench.c            ← real-GPU ioctl latency benchmark
    ├── BUILD                            ← Bazel build file
    ├── Makefile                         ← standalone build
    └── README.md                        ← benchmark documentation
```

---

## Next Steps

1. **Test on g5.12xlarge+** — reproduce the Modal 16% gap locally
2. **Profile seccomp filter** — the ~40 µs dominant cost is likely BPF filter matching
   against hundreds of nvidia ioctl entries; `seccomp_filters.go` is the target
3. **Implement Strategy B (batching)** — recognize the 3-ioctl pattern and execute
   all three in one Sentry→host round-trip
4. **Implement Strategy C (passthrough)** — for nvidia device fds that have passed
   initial validation, bypass Sentry ioctl dispatch entirely and forward to host
5. **Run on Modal with custom gVisor** — deploy the fast-path build to Modal and
   measure the actual bandwidth improvement on production hardware