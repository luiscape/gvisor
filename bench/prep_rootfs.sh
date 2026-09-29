#!/usr/bin/env bash
# Usage: prep_rootfs.sh vllm|sglang
# Builds the engine's cached rootfs under $BENCH_DIR/rootfs once: the image's
# filesystem plus what nvidia-container-cli would inject (the host driver's
# userspace), and cuda-checkpoint, which the sentry execs in the container.
set -euo pipefail
ENGINE="$1"
case "$ENGINE" in
  vllm) IMAGE=vllm/vllm-openai:v0.29.0 ;;
  sglang) IMAGE=lmsysorg/sglang:v0.5.20 ;;
  *) echo "unknown engine: $ENGINE" >&2; exit 2 ;;
esac
B=${BENCH_DIR:-/data/bench}
R=$B/rootfs/$ENGINE
DRV=$(nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader)
# Rebuild when the host driver changes: the injected libraries are versioned.
if [ -f "$R/.ready" ] && [ "$(cat "$R/.ready")" = "$DRV" ]; then exit 0; fi

echo "building $R from $IMAGE (driver $DRV)"
sudo rm -rf "$R"
sudo mkdir -p "$R"
cid=$(sudo docker create "$IMAGE" /bin/true)
sudo docker export "$cid" | sudo tar -C "$R" -xf -
sudo docker rm "$cid" >/dev/null
sudo docker image inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$IMAGE" \
  | grep . | sudo tee "$B/rootfs/$ENGINE.env" >/dev/null
sudo docker image inspect --format '{{.Config.WorkingDir}}' "$IMAGE" | sudo tee "$B/rootfs/$ENGINE.cwd" >/dev/null

sudo cp -a /usr/lib/x86_64-linux-gnu/*.so."$DRV" "$R/usr/lib/x86_64-linux-gnu/"
sudo cp -a /usr/bin/nvidia-smi "$R/usr/bin/"
sudo ldconfig -r "$R"
# The images also carry an R580 forward-compat libcuda in /usr/local/cuda/compat,
# which fails against the newer host driver; make sure it is not the one found.
first=$(sudo chroot "$R" /sbin/ldconfig -p | grep -m1 'libcuda.so.1 ' | awk '{print $NF}')
if [ "$(sudo readlink -f "$R$first")" != "$R/usr/lib/x86_64-linux-gnu/libcuda.so.$DRV" ]; then
  echo "libcuda.so.1 resolves to $first, not the host driver" >&2
  exit 1
fi
sudo install -m 0755 /usr/local/bin/cuda-checkpoint "$R/usr/local/bin/cuda-checkpoint"
echo "$DRV" | sudo tee "$R/.ready" >/dev/null
echo "ready: $R ($(sudo du -sh "$R" | cut -f1))"
