#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

# Resume the udev event queue whenever NVIDIA module reloads are re-enabled.
resume_udev_queue() {
  if [ -f /etc/modprobe.d/blacklist-nvidia-current.conf ]; then
    sudo udevadm control --start-exec-queue || true
    sudo udevadm settle || true
    sudo rm -f /etc/modprobe.d/blacklist-nvidia-current.conf
  fi
}

cleanup_on_exit() {
    local exit_code=$?

    resume_udev_queue

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
# Check with `runsc nvproxy list-supported-drivers`.

# Use default value if NVIDIA_DRIVER_VERSION is not provided
# 610.57.04 is the latest 580-branch driver supported by gVisor's nvproxy
# (580.95.05 is marked unsupported), and it compiles against kernels >= 6.10
# where the __assign_str() trace macro lost its second argument.
: "${NVIDIA_DRIVER_VERSION:=610.57.04}"

# Set the full driver version
DRIVER_VERSION="$NVIDIA_DRIVER_VERSION"

# Extract major version (everything before the first dot)
DRIVER_VERSION_MAJOR="${DRIVER_VERSION%%.*}"

# Use CUDA version for packages that aren't bound to a driver version
: "${CUDA_MAJOR_VERSION:=13}"

# Validate that major version is numeric
if ! [[ "$DRIVER_VERSION_MAJOR" =~ ^[0-9]+$ ]]; then
    echo "Error: Invalid driver version format. Major version must be numeric."
    exit 1
fi

# Detect architecture for platform-specific downloads
ARCH=$(uname -m)
case "$ARCH" in
  x86_64)  NVIDIA_ARCH="x86_64"; CUDA_REPO_ARCH="x86_64" ;;
  aarch64) NVIDIA_ARCH="aarch64"; CUDA_REPO_ARCH="sbsa" ;;
  *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

# You can now use DRIVER_VERSION and DRIVER_VERSION_MAJOR in your script
echo "Full driver version: $DRIVER_VERSION"
echo "Major version: $DRIVER_VERSION_MAJOR"
echo "Architecture: $ARCH (NVIDIA: $NVIDIA_ARCH, CUDA repo: $CUDA_REPO_ARCH)"

# Unload nouveau if present as it conflicts with the nvidia drivers
if lsmod | grep nouveau; then
  sudo modprobe -r nouveau
fi

# Stop nouveau coming back on reboot
cat <<EOF | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
  blacklist nouveau
  options nouveau modeset=0
EOF

# Build nvidia driver with same GCC version that was used to build UEK
# shellcheck source=/dev/null
if [[ $(. /etc/os-release; echo "$ID") = 'ol' ]]; then
  sudo dnf -y install gcc-toolset-14
  source /opt/rh/gcc-toolset-14/enable
fi

# The uninstall script blacklists nvidia modules and pauses the udev event queue
# to prevent automatic reload between scripts. Verify modules are still unloaded.
if lsmod | grep -q "^nvidia"; then
  echo "WARNING: nvidia modules still loaded despite blacklist; attempting removal"
  unload_nvidia_modules
fi

# NVIDIA Linux Accelerated Graphics Driver
# https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html#runfile
#
# We use the runfile installation method here, rather than manually installing cuda-drivers,
# fabricmanager, NVML, and all the other packages manually. This allows us to specify the exact
# version and is platform-independent.
# Starting with driver version 560, the open source kernel module is the default.
curl -fSsl -O "https://us.download.nvidia.com/tesla/$DRIVER_VERSION/NVIDIA-Linux-${NVIDIA_ARCH}-$DRIVER_VERSION.run"
if [ "${GITHUB_JOB:-}" = "worker-gpu-test" ] || [ "${GITHUB_JOB:-}" = "gpu-health-gpu-test" ] || [ "${GITHUB_JOB:-}" = "gpu-test" ]; then
  sudo env PATH="$PATH" sh "NVIDIA-Linux-${NVIDIA_ARCH}-$DRIVER_VERSION.run" \
    --ui=none \
    --no-questions \
    --disable-nouveau \
    --kernel-module-type=open \
    --no-drm
else
  sudo env PATH="$PATH" sh "NVIDIA-Linux-${NVIDIA_ARCH}-$DRIVER_VERSION.run" \
    --ui=none \
    --no-questions \
    --kernel-module-type=open \
    --disable-nouveau
fi
rm "NVIDIA-Linux-${NVIDIA_ARCH}-$DRIVER_VERSION.run"

resume_udev_queue

# TODO: We pin to 1.17.7 in both branches below for two reasons:
# * There is a bug in the latest version of the nvidia toolkit (1.18.0, as of Nov 5, 2025) that causes our instance startup script to fail.
#   - Context in Slack thread: https://modal-com.slack.com/archives/C0975RNQ03X/p1762355277951609?thread_ts=1762206668.229339&cid=C0975RNQ03X
#   - The bug has been fixed on `main`, but it isn't included in a named release yet: https://github.com/NVIDIA/nvidia-container-toolkit/commit/d8e61f9f433fcefb5d2e9b626205bc0929552e41
# * We haven't rolled out this gVisor commit: https://github.com/google/gvisor/commit/1c1fa726f046ccfe4f9659d4ca0b51889e0706d9
#   - EDIT (Nov 5, 2025): We actually *have* rolled out this commit to production.
#   - See this analysis: https://gist.githubusercontent.com/abhagwat/fa926c398b4ddd862b58ed4dd19bcd19/raw/525876262aefecf0449866b48104dd72d9a5315a/gistfile1.txt

# shellcheck source=/dev/null
if [[ $(. /etc/os-release; echo "$ID") = 'ol' ]]; then
  sudo dnf config-manager --add-repo "https://developer.download.nvidia.com/compute/cuda/repos/rhel9/${CUDA_REPO_ARCH}/cuda-rhel9.repo"

  # DCGM and Container toolkit
  # XXX(dano): The below install frequently fails on importing the GPG key of the above repo,
  # which does not seem to be retried by dnf.
  #
  # For RDMA builds, we need to install the `datacenter-gpu-manager` package, which conflicts
  # with the existing installation. We use `--allowerasing` to allow the installation to
  # continue even though there are conflicts.
  #
  # TODO: We pin to 1.17.7 because we haven't rolled out this gVisor commit: https://github.com/google/gvisor/commit/1c1fa726f046ccfe4f9659d4ca0b51889e0706d9
  timeout 300 bash -c "until sudo dnf -y --allowerasing install datacenter-gpu-manager-4-cuda$CUDA_MAJOR_VERSION 'nvidia-container-toolkit < 1.17.7' 'libnvidia-container-tools < 1.17.7' 'libnvidia-container1 < 1.17.7'; do echo 'retrying after 1s'; sleep 1; done"

  # Oracle Linux RDMA workers need ib_umad available before fabric manager and
  # NVLSM initialize, so load it now and persist it across reboots.
  sudo tee /etc/modules-load.d/rdma.conf >/dev/null <<'EOF'
ib_umad
EOF
  sudo modprobe ib_umad

  # Fabric manager. Retry like the toolkit install above: the NVIDIA repo
  # intermittently fails fetches, and dnf does not retry these on its own.
  timeout 300 bash -c "until sudo dnf install -y nvlsm infiniband-diags \
    && sudo dnf install -y --setopt='cuda-rhel9-${CUDA_REPO_ARCH}.module_hotfixes=true' nvidia-fabricmanager-$DRIVER_VERSION-1; do echo 'retrying after 1s'; sleep 1; done"
else
  # DCGM and Container toolkit
  distribution=$(. /etc/os-release;echo "$ID$VERSION_ID" | sed -e 's/\.//g')

  # NVIDIA has not populated the ubuntu2604 CUDA repo with the container
  # toolkit or 580-branch fabricmanager packages yet; fall back to the
  # ubuntu2404 repo, whose packages install fine on 26.04.
  if [ "$distribution" = "ubuntu2604" ]; then
    distribution=ubuntu2404
  fi

  # On Oracle Ubuntu, we get some repos pre-installed. Installing cuda-keyring will install new, conflicting
  # repos. We need to remove the pre-installed repos before installing the new ones.
  sudo rm -f /etc/apt/sources.list.d/developer_download_nvidia_com_compute_cuda_repos_"${distribution}"_"${CUDA_REPO_ARCH}".list

  cuda_repo_base="https://developer.download.nvidia.com/compute/cuda/repos/$distribution/${CUDA_REPO_ARCH}"
  keyring_deb=$(curl -fsSL "$cuda_repo_base/" | grep -oP 'cuda-keyring_[\d.]+-\d+_all\.deb' | sort -V | tail -1)
  curl -fsSL -O "$cuda_repo_base/$keyring_deb"
  sudo dpkg -i --force-confdef --force-confold "$keyring_deb" && rm "$keyring_deb"

  # Retry apt-get update/install with a per-attempt timeout and mirror fallback.
  # Acquire::http::Timeout only fires on per-connection inactivity, so a
  # slow-but-trickling regional mirror (e.g. *.ec2.archive.ubuntu.com) can hang
  # for many minutes without tripping it. A per-attempt timeout lets us fail
  # fast, and on retry we switch away from the cloud regional mirror to
  # archive.ubuntu.com so we stop hitting the same slow server.
  _switched_mirror=0
  _switch_mirror() {
    if [ "$_switched_mirror" -eq 0 ]; then
      echo "Switching apt mirror from cloud regional Ubuntu mirrors to archive.ubuntu.com..." >&2
      # azure.archive.ubuntu.com (Azure runners) and *.ec2.archive.ubuntu.com
      # (AWS self-hosted runners). Basic sed has no alternation, so two passes.
      for _apt_sources in /etc/apt/sources.list /etc/apt/apt-mirrors.txt \
          /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do
        [ -e "$_apt_sources" ] || continue
        sudo sed -i \
          -e 's|azure\.archive\.ubuntu\.com|archive.ubuntu.com|g' \
          -e 's|[a-z0-9.-]*\.ec2\.archive\.ubuntu\.com|archive.ubuntu.com|g' \
          "$_apt_sources" 2>/dev/null || true
      done
      timeout 180 sudo apt-get update -q || echo "warning: apt-get update after mirror switch failed (exit $?), continuing with retry..." >&2
      _switched_mirror=1
    fi
  }
  for attempt in 1 2 3; do
    rc=0
    timeout 180 sudo apt-get update || rc=$?
    if [ "$rc" -eq 0 ]; then
      break
    fi
    if [ "$attempt" -eq 3 ]; then
      echo "apt-get update failed after 3 attempts (last exit $rc)" >&2
      exit 1
    fi
    echo "apt-get update attempt $attempt failed (exit $rc), retrying in 10s..." >&2
    _switch_mirror
    sleep 10
  done

  # Block -server driver packages from the NVIDIA apt repo. The driver is
  # installed via runfile, so we never need the apt -server variants. The repo
  # sometimes publishes libnvidia-*-NNN-server packages before their
  # nvidia-kernel-common-NNN-server dependency is available, causing
  # unresolvable dependency errors when apt pulls them in transitively.
  sudo tee /etc/apt/preferences.d/block-nvidia-server-pkgs <<'EOF'
Package: libnvidia-*-server nvidia-*-server libnvidia-*-server-* nvidia-*-server-*
Pin: version *
Pin-Priority: -1
EOF

  # Install all APT-managed NVIDIA packages in a single invocation so that
  # a mid-script cancellation (e.g. from cancel-in-progress) cannot leave
  # the machine with a partial toolkit (driver + nvidia-ctk but no
  # fabricmanager).
  # Note: the `cuda-drivers-fabricmanager-#` metapackage also has cuda-drivers, which conflicts with runfile.
  #
  # nvlsm is pinned because NVIDIA's repo occasionally publishes a version in
  # its Packages index and then removes the corresponding .deb, leaving the
  # floating "latest" pointing at an artifact that 404s. The retry below
  # refreshes the apt index (with --fix-missing) so a runner holding a stale
  # index that references a since-removed file recovers instead of failing.
  nvidia_apt_pkgs=(
    "datacenter-gpu-manager-4-cuda$CUDA_MAJOR_VERSION"
    nvidia-container-toolkit=1.17.8-1
    nvidia-container-toolkit-base=1.17.8-1
    libnvidia-container-tools=1.17.8-1
    libnvidia-container1=1.17.8-1
    nvlsm=2025.10.12-1
    infiniband-diags
    # Debian revision glob: older packages use -1, newer ones -1ubuntu1.
    "nvidia-fabricmanager=$DRIVER_VERSION-*"
  )
  for attempt in 1 2 3; do
    rc=0
    # Refresh the index each attempt with --fix-missing: NVIDIA's repo
    # occasionally serves a stale Packages index referencing a since-removed
    # .deb, so a re-update lets a runner recover instead of failing.
    timeout 180 sudo apt-get update || true
    timeout 300 sudo apt-get install -y --allow-downgrades --fix-missing \
      "${nvidia_apt_pkgs[@]}" || rc=$?
    if [ "$rc" -eq 0 ]; then
      break
    fi
    if [ "$attempt" -eq 3 ]; then
      echo "nvidia apt install failed after 3 attempts (last exit $rc)" >&2
      exit 1
    fi
    echo "nvidia apt install attempt $attempt failed (exit $rc), retrying in 10s..." >&2
    # Recover from any interrupted dpkg state before retrying.
    sudo dpkg --configure -a || true
    _switch_mirror
    sleep 10
  done

  # Disable display manager, just in case it is installed (though it's not)
  sudo apt-get remove -y gdm3

  if [ "${PACKER_BUILDER_TYPE:-}" != "qemu" ] && [ "${PACKER_BUILDER_TYPE:-}" != "crusoe" ]; then
    sudo nvidia-smi -pm 1
  fi
fi

# Enable Fabric Manager for NVSwitch systems.
# It is needed for Oracle A100 instances. If NVSwitch is not enabled,
# the worker cannot find its GPU devices.
sudo systemctl enable nvidia-fabricmanager

# Allow non-admin users to access GPU performance counters.
# For details, see https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters
# and WRK-844.
sudo touch /etc/modprobe.d/nvidia.conf
# We use an authoritative write here to avoid retries creating multiple entries.
sudo bash -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia.conf'

# shellcheck source=/dev/null
if [[ $(. /etc/os-release; echo "$ID") = 'ubuntu' ]]; then
  sudo update-initramfs -u
else
  # Add the nvidia.conf file to the dracut initramfs.
  echo 'install_items+=" /etc/modprobe.d/nvidia.conf "' | \
    sudo tee /etc/dracut.conf.d/20-nvidia-profiling.conf
  sudo dracut -f
fi

# Rebooting here seems to prevent issues leaving the drivers in a bad
# state.
if [ "${GITHUB_JOB:-}" = "worker-gpu-test" ] || [ "${GITHUB_JOB:-}" = "gpu-health-gpu-test" ] || [ "${GITHUB_JOB:-}" = "gpu-test" ]; then
  echo "skipping reboot on Github Action GPU tests"
else
  echo "rebooting now after installing nvidia drivers"
  sudo reboot now
fi
