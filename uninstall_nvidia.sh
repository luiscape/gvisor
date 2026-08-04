#!/bin/bash
#
# This script is used on Github Actions.
# The Github Actions GPU runners are, as of Apr 2024, on NVIDIA driver version 535.54.03.
# This version does not match production, AND gvisor no longer supports driver versions this old.
# So we uninstall that version on the runner and reinstall the correct one.

set -o errexit
set -o nounset
set -o pipefail

resume_udev_queue() {
  if [ -f /etc/modprobe.d/blacklist-nvidia-current.conf ]; then
    sudo udevadm control --start-exec-queue || true
    sudo udevadm settle || true
    sudo rm -f /etc/modprobe.d/blacklist-nvidia-current.conf
  fi
}

cleanup_on_exit() {
    local exit_code=$?

    if [ "$exit_code" -ne 0 ]; then
        resume_udev_queue
    fi

    if [ "$exit_code" -ne 0 ] && [ -f /var/log/nvidia-installer.log ]; then
        echo "--- /var/log/nvidia-installer.log ---"
        cat /var/log/nvidia-installer.log
        echo "--- end /var/log/nvidia-installer.log ---"
    fi

    return "$exit_code"
}
trap cleanup_on_exit EXIT

unload_nvidia_modules() {
  if ! lsmod | grep -q '^nvidia'; then
    return
  fi

  sudo systemctl stop nvidia-fabricmanager 2>/dev/null || true
  sudo systemctl stop nvidia-dcgm 2>/dev/null || true
  sudo systemctl stop nvidia-persistenced 2>/dev/null || true
  if pgrep -x nv-hostengine >/dev/null; then
    sudo pkill nv-hostengine || true
  fi
  if pgrep -x nvidia-persiste >/dev/null; then
    sudo pkill nvidia-persiste || true
  fi
  sudo fuser -k /dev/nvidia* 2>/dev/null || true

  for _ in 1 2 3; do
    sudo modprobe --remove nvidia_drm 2>/dev/null || true
    sudo modprobe --remove nvidia_modeset 2>/dev/null || true
    sudo modprobe --remove nvidia_uvm 2>/dev/null || true
    sudo modprobe --remove nvidia_peermem 2>/dev/null || true
    sudo modprobe --remove nvidia 2>/dev/null || true
    sudo rmmod nvidia_drm 2>/dev/null || true
    sudo rmmod nvidia_modeset 2>/dev/null || true
    sudo rmmod nvidia_uvm 2>/dev/null || true
    sudo rmmod nvidia_peermem 2>/dev/null || true
    sudo rmmod nvidia 2>/dev/null || true

    if ! lsmod | grep -q '^nvidia'; then
      return
    fi

    sleep 1
  done

  echo "nvidia kernel modules still loaded after removal attempts"
  lsmod | grep '^nvidia'
  lsof /dev/nvidia* || true
  return 1
}

# IMPORTANT: the version must be compatible with gvisor.
DRIVER_VERSION="610.57.04"

# Detect architecture for platform-specific downloads
ARCH=$(uname -m)
case "$ARCH" in
  x86_64)  NVIDIA_ARCH="x86_64" ;;
  aarch64) NVIDIA_ARCH="aarch64" ;;
  *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers are not installed. Exiting early..."
    exit 0
fi

# Unload nouveau if present as it conflicts with the nvidia drivers
if lsmod | grep nouveau; then
  sudo modprobe --remove nouveau
fi
cat <<EOF | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
blacklist nouveau
options nouveau modeset=0
EOF
# shellcheck source=/dev/null
if [ "$(. /etc/os-release; echo "$ID")" = "ol" ] || [ "$(. /etc/os-release; echo "$ID")" = "rocky" ]; then
  sudo dracut --force
else
  sudo update-initramfs -u
fi

# On Lambda Labs the nvidia-fabricmanager is running and stops the unload
# of the nvidia kernel modules: "modprobe: FATAL: Module nvidia is in use."
if sudo systemctl is-active --quiet nvidia-fabricmanager; then
  echo "nvidia-fabricmanager is running. Stopping the service to allow nvidia mod unload"
  sudo systemctl stop nvidia-fabricmanager
  echo "nvidia-fabricmanager has been stopped."
fi

# If DCGM is still running it keeps a file descriptor open on /dev/nvidia-uvm.
# This prevents the kernel module unload from completing.
if systemctl is-active --quiet nvidia-dcgm; then
  sudo systemctl stop nvidia-dcgm
fi

# Stop nvidia-persistenced to prevent the nvidia kernel modules from being unloaded.
# This fails when in a Github Actions job.
if systemctl is-active --quiet nvidia-persistenced && [ "${GITHUB_ACTIONS:-}" != "true" ]; then
  sudo systemctl stop nvidia-persistenced
  pgrep -x nv-hostengine > /dev/null && sudo pkill nv-hostengine
fi

# Blacklist nvidia modules and pause the udev event queue BEFORE unloading.
# The GPU hardware triggers automatic module loading via udev rules; without
# this, udev immediately reloads the modules between the uninstall and install
# scripts, causing the runfile installer to fail.
if lsmod | grep -q nvidia; then
  cat <<'BLEOF' | sudo tee /etc/modprobe.d/blacklist-nvidia-current.conf
blacklist nvidia
blacklist nvidia_drm
blacklist nvidia_modeset
blacklist nvidia_uvm
blacklist nvidia_peermem
BLEOF
  sudo udevadm control --stop-exec-queue || true
fi

# Turn off persistence mode, which can prevent removal of the 'nvidia' kernel module.
# But handle non-zero exit, as it's ok if 'No devices were found'.
sudo nvidia-smi -pm 0 || true
# Unload all nvidia kernel modules if present as they conflict with the nvidia install
if lsmod | grep -q '^nvidia'; then
  unload_nvidia_modules
  echo "nvidia kernel modules after removal"
  lsmod | grep nvidia || true
fi


# NVIDIA Linux Accelerated Graphics Driver
# https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html#runfile
#
# We use the runfile installation method here, rather than manually installing cuda-drivers,
# fabricmanager, NVML, and all the other packages manually. This allows us to specify the exact
# version and is platform-independent.
echo "uninstalling cuda with runfile method"
curl -fSsl -O "https://us.download.nvidia.com/tesla/$DRIVER_VERSION/NVIDIA-Linux-${NVIDIA_ARCH}-$DRIVER_VERSION.run"
sudo sh "NVIDIA-Linux-${NVIDIA_ARCH}-$DRIVER_VERSION.run" \
   --ui=none \
   --no-questions \
   --disable-nouveau \
   --uninstall
rm "NVIDIA-Linux-${NVIDIA_ARCH}-$DRIVER_VERSION.run"

# shellcheck source=/dev/null
if [ "$(. /etc/os-release; echo "$ID")" = "ol" ] || [ "$(. /etc/os-release; echo "$ID")" = "rocky" ]; then
  # In Vultr's Rocky Linux this post-transaction-action will reinstall NVIDIA drivers
  # on *every* yum or dnf install, which makes image build extremely slow and also broken
  # because it installs a driver version different from what we specify.
  rm -rf /etc/dnf/plugins/post-transaction-actions.d/01-vendor-watchgpudriver.action
  sudo yum remove -y 'nvidia-*'
  sudo yum remove -y 'libnvidia-*'
else
  # Block -server driver packages from the NVIDIA apt repo. The repo sometimes
  # publishes libnvidia-*-NNN-server packages before their
  # nvidia-kernel-common-NNN-server dependency is available, which causes apt to
  # fail with unresolvable dependency errors even during purge/install operations.
  sudo tee /etc/apt/preferences.d/block-nvidia-server-pkgs <<'PINEOF'
Package: libnvidia-*-server nvidia-*-server libnvidia-*-server-* nvidia-*-server-*
Pin: version *
Pin-Priority: -1
PINEOF

  mapfile -t installed_nvidia_packages < <(
    dpkg-query -W -f='${db:Status-Abbrev}\t${binary:Package}\n' 2>/dev/null |
      awk '$1 ~ /^i/ && $2 ~ /nvidia|datacenter-gpu-manager|nvlink|nvlsm|libnvsdm/ { print $2 }'
  )
  if [ "${#installed_nvidia_packages[@]}" -gt 0 ]; then
    sudo dpkg --purge --force-depends "${installed_nvidia_packages[@]}"
  else
    echo "No installed NVIDIA apt packages to purge"
  fi

  # Remove apt package state for NVIDIA packages that were installed outside apt
  # by the runner image. Their maintainer scripts can otherwise trigger apt's
  # resolver for the broken -server package graph.
  mapfile -t residual_nvidia_packages < <(
    dpkg-query -W -f='${db:Status-Abbrev}\t${binary:Package}\n' 2>/dev/null |
      awk '$1 ~ /^.[^n]/ && $2 ~ /nvidia|datacenter-gpu-manager|nvlink|nvlsm|libnvsdm/ { print $2 }'
  )
  if [ "${#residual_nvidia_packages[@]}" -gt 0 ]; then
    sudo dpkg --purge --force-depends "${residual_nvidia_packages[@]}"
  fi
fi
