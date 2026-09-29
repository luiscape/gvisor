#!/usr/bin/env bash
# Engine images, the gate's models and cuda-checkpoint for the bench.
set -eu
HF_DIR=${HF_DIR:-/data/hf}
sudo docker pull -q vllm/vllm-openai:v0.29.0
sudo docker pull -q lmsysorg/sglang:v0.5.20
sudo mkdir -p "$HF_DIR"
sudo docker run --rm --network host --entrypoint python3 -e HF_HOME=/hf -v "$HF_DIR":/hf \
  vllm/vllm-openai:v0.29.0 -c "
from huggingface_hub import snapshot_download as s
for m in ('Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-3B-Instruct'):
    print(s(m), flush=True)"
curl -fsSL -o /tmp/cuda-checkpoint \
  https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint
sudo install -m 0755 /tmp/cuda-checkpoint /usr/local/bin/cuda-checkpoint
