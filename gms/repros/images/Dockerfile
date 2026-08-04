# Image for the local gVisor GPU-snapshot repros.
#
# CUDA base + torch + pynvml (B1) + cuda-checkpoint + the repro apps and
# shared helpers. Host driver userspace is injected into the rootfs by
# run_repro.sh (via cr-bench/common.sh), not baked here.
#
# Build context: gms/repros/
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install --no-cache-dir pynvml==11.5.3

# NVIDIA cuda-checkpoint utility (links libcuda at runtime).
RUN curl -fSL -o /usr/local/bin/cuda-checkpoint \
        https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint \
    && chmod 0755 /usr/local/bin/cuda-checkpoint

RUN mkdir -p /app
COPY _diag.py _harness.py /app/
COPY repro_a_cuda_context.py \
     repro_b1_nvml_fd.py \
     repro_b2_inherited_fork_fd.py \
     repro_c1_cuda_ipc_handle.py \
     repro_c2_cumem_vmm.py \
     repro_inductor.py /app/

ENV PORT=8199
WORKDIR /app
