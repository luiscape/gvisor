#!/usr/bin/env bash
# --------------------------------------------------------------------------
#  common.sh — shared machinery for the cr-bench checkpoint/restore
#  benchmarks.
#
#  Provides:
#    - rootfs extraction from a Docker image (cached) + NVIDIA driver
#      userspace injection + per-run overlay mount
#    - OCI bundle (config.json) generation, CPU-only or with N GPUs
#    - pure-runsc container lifecycle: run / checkpoint / restore / exec
#    - timing + verification helpers and the summary table
#
#  GPU checkpoints use gVisor's NATIVE cuda-checkpoint integration:
#      runsc checkpoint --cuda-checkpoint-path=<path in container>
#  The sentry finds every CUDA process, execs `cuda-checkpoint --toggle`
#  on each (in parallel), serializes the sandbox, and automatically
#  toggles the processes back after restore/resume.  No LD_PRELOAD, no
#  helper daemon.
#
#  A benchmark script sets its knobs (BENCH_NAME, IMAGE, DOCKERFILE,
#  PORT, CB_GPU, GPU_DEVICES, ...) then calls the cb_* functions.
# --------------------------------------------------------------------------

# ── Global defaults (overridable via environment) ─────────────────────────
# Note: the GPU benchmarks need a runsc that includes both
# --cuda-checkpoint-path support AND the SetSaver fix from commit
# 1e693aa6e ("Add application-driven checkpoint/restore support") — older
# builds nil-panic in invokeCudaCheckpoint.  Build with:
#   make build TARGETS=//runsc && cp bazel-bin/runsc/runsc_/runsc /usr/local/bin/runsc-crbench
RUNSC="${RUNSC:-/usr/local/bin/runsc-crbench}"
COMPRESSION="${COMPRESSION:-none}"
EXCLUDE_ZERO="${EXCLUDE_ZERO:-1}"
REBUILD_ROOTFS="${REBUILD_ROOTFS:-0}"

# Where cb_prepare_rootfs stages the multicast interposer inside the container,
# and where it is built from on the host.
CUDA_MULTICAST_SHIM_PATH="${CUDA_MULTICAST_SHIM_PATH:-/usr/local/lib/mcshim.so}"
CUDA_MULTICAST_SHIM_SRC="${CUDA_MULTICAST_SHIM_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/gpu_mem_snapshots/phase0/mcshim/mcshim.so}"

# NCCL_CKPT_PATCH=1 selects the NCCL-native multicast path instead of the
# interposer: a patched libnccl whose ncclCommSuspend/Resume also release and
# rebuild the NVLS multicast layer, plus NCCL's own control thread so the
# engine needs no code change. See gpu_mem_snapshots/phase0/NCCL_PATCH_TESTS.md.
NCCL_CKPT_PATCH_PATH="${NCCL_CKPT_PATCH_PATH:-/usr/local/lib/libnccl-patched.so.2}"
NCCL_CKPT_PATCH_SRC="${NCCL_CKPT_PATCH_SRC:-/opt/phase0/nccl-patched/libnccl.so.2}"
NCCL_CKPT_CTRL_DIR="${NCCL_CKPT_CTRL_DIR:-/tmp/ncclckpt}"
DATA_ROOT="${DATA_ROOT:-/data/cr-bench}"
CUDA_CHECKPOINT_PATH="${CUDA_CHECKPOINT_PATH:-/usr/local/bin/cuda-checkpoint}"
CUDA_CKPT_SEQUENTIAL="${CUDA_CKPT_SEQUENTIAL:-0}"
# CKPT_TMPFS=1: back the checkpoint image directory with tmpfs so
# checkpoint.img/pages.img are memory-resident (equivalent to memfd: both
# are shmem-backed). Restore then reads pages at memory speed.
CKPT_TMPFS="${CKPT_TMPFS:-0}"
CKPT_TMPFS_SIZE="${CKPT_TMPFS_SIZE:-64g}"
# RESTORE_BACKGROUND=1: pass --background to runsc restore (restore returns
# while pages.img continues loading asynchronously; faulting tasks wait for
# their pages). Requires an uncompressed checkpoint.
RESTORE_BACKGROUND="${RESTORE_BACKGROUND:-0}"
# PAGES_IDENTITY=1: checkpoint with --pages-layout=identity (pages.img is
# sparse, page contents at file offset == MemoryFile offset). Required for
# ADOPT_PAGES=1. Requires COMPRESSION=none.
PAGES_IDENTITY="${PAGES_IDENTITY:-0}"
# ADOPT_PAGES=1: restore with --adopt-pages-file: the sentry adopts pages.img
# as its memory file and uses the pages in place (zero-copy) instead of
# copying them into a new memfd. pages.img is consumed (mutated) by the
# restored sandbox, so the image can only be restored once. Pair with
# CKPT_TMPFS=1 so the adopted file is shmem-backed.
ADOPT_PAGES="${ADOPT_PAGES:-0}"
NVPROXY_DRIVER_VER="${NVPROXY_DRIVER_VER:-latest}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"

# ── Colours / logging ─────────────────────────────────────────────────────
if [ -t 1 ]; then
    B="\033[1m" G="\033[32m" R="\033[31m" C="\033[36m" Y="\033[33m" Z="\033[0m"
else
    B="" G="" R="" C="" Y="" Z=""
fi
info()   { echo -e "${B}${C}==> $*${Z}"; }
ok()     { echo -e "    ${G}✓ $*${Z}"; }
warn()   { echo -e "    ${Y}⚠ $*${Z}"; }
fail()   { echo -e "    ${R}✗ $*${Z}"; }
banner() { echo -e "${B}$*${Z}"; }
ts_ms()  { echo $(( $(date +%s%N) / 1000000 )); }

CB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Init: directory layout, container ids, cleanup trap ──────────────────
# Requires: BENCH_NAME, IMAGE
cb_init() {
    if [[ "$(id -u)" -ne 0 ]]; then
        fail "Must run as root (sudo)"; exit 1
    fi
    BASE_DIR="$DATA_ROOT/$BENCH_NAME-$$"
    ROOTFS_DIR="$DATA_ROOT/rootfs-${IMAGE//[\/:]/_}"  # cached across runs
    ROOTFS_UPPER="$BASE_DIR/rootfs-upper"
    ROOTFS_WORK="$BASE_DIR/rootfs-work"
    ROOTFS_MERGED="$BASE_DIR/rootfs-merged"
    BUNDLE_DIR="$BASE_DIR/bundle"
    RESTORE_BUNDLE_DIR="$BASE_DIR/bundle-restore"
    CKPT_DIR="$BASE_DIR/ckpt"
    LOG_DIR="$BASE_DIR/logs"
    RUNSC_ROOT="$BASE_DIR/runsc-state"
    APPLOG_DIR="$BASE_DIR/applog"   # bind-mounted at /applog in the container

    CONTAINER_ID="$BENCH_NAME"
    RESTORE_ID="$BENCH_NAME-restored"

    mkdir -p "$BUNDLE_DIR" "$CKPT_DIR" "$LOG_DIR" "$RUNSC_ROOT" "$APPLOG_DIR" \
             "$ROOTFS_UPPER" "$ROOTFS_WORK" "$ROOTFS_MERGED"

    if [[ "$RESTORE_BACKGROUND" = "1" && "$COMPRESSION" != "none" ]]; then
        fail "--background restore requires an uncompressed checkpoint (COMPRESSION=none)"
        exit 1
    fi
    if [[ "$PAGES_IDENTITY" = "1" && "$COMPRESSION" != "none" ]]; then
        fail "PAGES_IDENTITY=1 requires COMPRESSION=none"
        exit 1
    fi
    if [[ "$ADOPT_PAGES" = "1" && "$PAGES_IDENTITY" != "1" ]]; then
        fail "ADOPT_PAGES=1 requires PAGES_IDENTITY=1 (identity-layout pages.img)"
        exit 1
    fi
    if [[ "$CKPT_TMPFS" = "1" ]]; then
        mount -t tmpfs -o "size=$CKPT_TMPFS_SIZE" tmpfs "$CKPT_DIR" || {
            fail "Failed to mount tmpfs at $CKPT_DIR"; exit 1
        }
        ok "Checkpoint image dir on tmpfs ($CKPT_TMPFS_SIZE) at $CKPT_DIR"
    fi

    trap cb_cleanup EXIT
}

cb_cleanup() {
    echo ""
    info "Cleaning up …"
    "$RUNSC" --root "$RUNSC_ROOT" kill "$CONTAINER_ID" KILL 2>/dev/null || true
    "$RUNSC" --root "$RUNSC_ROOT" delete --force "$CONTAINER_ID" 2>/dev/null || true
    "$RUNSC" --root "$RUNSC_ROOT" kill "$RESTORE_ID" KILL 2>/dev/null || true
    "$RUNSC" --root "$RUNSC_ROOT" delete --force "$RESTORE_ID" 2>/dev/null || true

    if mountpoint -q "$ROOTFS_MERGED" 2>/dev/null; then
        umount -lR "$ROOTFS_MERGED" 2>/dev/null || true
    fi
    if [[ "$CKPT_TMPFS" = "1" ]] && mountpoint -q "$CKPT_DIR" 2>/dev/null; then
        umount -l "$CKPT_DIR" 2>/dev/null || true
    fi

    info "Artifacts at $BASE_DIR"
    info "  Logs:       $LOG_DIR"
    info "  Checkpoint: $CKPT_DIR"
}

# ── runsc helpers ─────────────────────────────────────────────────────────
_rexec() {
    local cid="$1"; shift
    "$RUNSC" --root "$RUNSC_ROOT" exec "$cid" "$@" 2>/dev/null
}

cb_state() {
    "$RUNSC" --root "$RUNSC_ROOT" state "$1" 2>/dev/null \
        | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])" \
        2>/dev/null || echo "dead"
}

# With --network=none gVisor creates a loopback-only netstack; the host
# cannot reach the container's port, so curl runs INSIDE the sandbox.
cb_curl() {
    local cid="$1"; shift
    _rexec "$cid" /usr/bin/curl -sf --max-time 120 "$@"
}

cb_health() {
    _rexec "$1" /usr/bin/curl -sf --max-time 2 \
        "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1
}

# ── GPU detection ─────────────────────────────────────────────────────────
cb_detect_gpu() {
    _nvsmi() { nvidia-smi -i 0 "$@" 2>/dev/null || true; }
    HOST_DRIVER_VER=$(_nvsmi --query-gpu=driver_version --format=csv,noheader,nounits | tr -d '[:space:]')
    GPU_NAME=$(       _nvsmi --query-gpu=name           --format=csv,noheader          | xargs)
    GPU_MEM_TOTAL=$(  _nvsmi --query-gpu=memory.total   --format=csv,noheader,nounits  | tr -d '[:space:]')
    if [[ -z "$HOST_DRIVER_VER" ]]; then
        fail "Cannot detect NVIDIA driver (nvidia-smi failed)"; exit 1
    fi
    local major="${HOST_DRIVER_VER%%.*}"
    if (( major < 550 )); then
        fail "cuda-checkpoint requires driver >= R550 (found $HOST_DRIVER_VER)"; exit 1
    elif (( major < 570 )); then
        warn "Driver $HOST_DRIVER_VER: multi-process / IPC-heavy workloads may fail; >= R570 recommended"
    fi

    # Pick the nvproxy driver version.  "latest" is wrong when the host
    # driver is not the newest supported one (the ABIs differ and CUDA
    # init fails with "invalid argument").  Prefer an exact match, then
    # the closest supported version on the same major branch.
    if [[ "$NVPROXY_DRIVER_VER" == "latest" ]]; then
        local supported
        supported=$("$RUNSC" nvproxy list-supported-drivers 2>/dev/null || true)
        if echo "$supported" | grep -qx "$HOST_DRIVER_VER"; then
            NVPROXY_DRIVER_VER="$HOST_DRIVER_VER"
        else
            local same_major
            same_major=$(echo "$supported" | grep "^${major}\." | head -1 || true)
            if [[ -n "$same_major" ]]; then
                NVPROXY_DRIVER_VER="$same_major"
                warn "Host driver $HOST_DRIVER_VER not supported by this runsc;"
                warn "using same-branch nvproxy ABI $NVPROXY_DRIVER_VER"
            else
                warn "No supported R${major} driver in this runsc; keeping 'latest'"
            fi
        fi
    fi
}

# ── runsc global flags ────────────────────────────────────────────────────
# Requires: CB_GPU (0/1)
cb_runsc_flags() {
    RUNSC_FLAGS=(
        --root "$RUNSC_ROOT"
        --network=none
        --debug
        --debug-log="$LOG_DIR/"
        --platform=systrap
        --directfs
        --overlay2=root:self
        --host-uds=all
        --net-disconnect-ok
        --restore-spec-validation=ignore
        --cpu-num-from-quota
        --save-restore-netstack=false
    )
    # Note: we deliberately do NOT pass --nvproxy-docker.  In hook mode the
    # restore-side device set for nvproxy device remapping is derived from a
    # dev-gofer directory listing (ALL host GPUs), which turns cross-GPU
    # remapping into an identity mapping.  With explicit spec.Linux.Devices
    # (GKE-style), the device set is exactly the bundle's GPUs, so
    # checkpointing on one GPU set and restoring on another works.  Driver
    # userspace is baked into the rootfs by cb_prepare_rootfs, so the
    # nvidia-container-cli hook isn't needed.
    if [[ "$CB_GPU" = "1" ]]; then
        RUNSC_FLAGS+=(
            --nvproxy
            --nvproxy-driver-version="$NVPROXY_DRIVER_VER"
            --nvproxy-allowed-driver-capabilities=all
        )
        # Opt-in: group each GPU container's CUDA processes into a
        # cuda-checkpoint job (driver R610+) so CUDA IPC state can be
        # checkpointed/restored coherently. Set CUDA_CKPT_JOB_FILE=1.
        if [[ "${CUDA_CKPT_JOB_FILE:-0}" = "1" ]]; then
            RUNSC_FLAGS+=(--cuda-checkpoint-path="$CUDA_CHECKPOINT_PATH")
        fi
        # Opt-in: LD_PRELOAD the multicast suspend/resume interposer, which is
        # what makes a process holding NVLS / symmetric-memory multicast
        # objects checkpointable at all. gVisor drives its suspend and rebuild
        # around the cuda-checkpoint phases; the application is untouched.
        # Set CUDA_MULTICAST_SHIM=1 (cb_prepare_rootfs stages the library).
        if [[ "${CUDA_MULTICAST_SHIM:-0}" = "1" ]]; then
            RUNSC_FLAGS+=(--cuda-multicast-shim-path="$CUDA_MULTICAST_SHIM_PATH")
        fi
    fi
}

# ── Phase: rootfs prep ────────────────────────────────────────────────────
# Build IMAGE from DOCKERFILE if missing, export its filesystem to
# ROOTFS_DIR (cached), inject NVIDIA driver userspace if CB_GPU=1, then
# mount a per-run overlay at ROOTFS_MERGED.
cb_prepare_rootfs() {
    info "Prepare rootfs"

    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        info "  Building Docker image '$IMAGE' …"
        docker build -t "$IMAGE" -f "$CB_DIR/$DOCKERFILE" "$CB_DIR" 2>&1 | tail -5 || true
        if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
            fail "Docker build failed"; exit 1
        fi
    fi

    [[ "$REBUILD_ROOTFS" = "1" ]] && rm -rf "$ROOTFS_DIR"

    if [[ "${CUDA_MULTICAST_SHIM:-0}" = "1" ]]; then
        if [[ ! -f "$CUDA_MULTICAST_SHIM_SRC" ]]; then
            die "CUDA_MULTICAST_SHIM=1 but $CUDA_MULTICAST_SHIM_SRC is missing (build it: bash gpu_mem_snapshots/phase0/mcshim/build.sh)"
        fi
    fi

    if [[ -f "$ROOTFS_DIR/.ready" ]]; then
        ok "Rootfs cached at $ROOTFS_DIR"
    else
        info "  Extracting rootfs from image '$IMAGE' …"
        rm -rf "$ROOTFS_DIR"
        mkdir -p "$ROOTFS_DIR"
        local tmp_cid
        tmp_cid=$(docker create "$IMAGE" /bin/true 2>/dev/null)
        docker export "$tmp_cid" | tar -C "$ROOTFS_DIR" -xf -
        docker rm "$tmp_cid" >/dev/null 2>&1
        ok "Rootfs exported"

        if [[ "$CB_GPU" = "1" ]]; then
            _cb_inject_nvidia_libs
        fi

        touch "$ROOTFS_DIR/.ready"
        ok "Rootfs ready ($(du -sh "$ROOTFS_DIR" 2>/dev/null | cut -f1))"
    fi

    info "  Mounting per-run rootfs overlay …"
    if [[ "${CUDA_MULTICAST_SHIM:-0}" = "1" ]]; then
        install -D -m 0755 "$CUDA_MULTICAST_SHIM_SRC" \
            "${ROOTFS_DIR}${CUDA_MULTICAST_SHIM_PATH}" \
            && ok "Staged multicast interposer at ${CUDA_MULTICAST_SHIM_PATH}" \
            || die "failed to stage $CUDA_MULTICAST_SHIM_SRC"
    fi
    if [[ "${NCCL_CKPT_PATCH:-0}" = "1" ]]; then
        [[ -f "$NCCL_CKPT_PATCH_SRC" ]] || \
            die "NCCL_CKPT_PATCH=1 but $NCCL_CKPT_PATCH_SRC is missing (build nccl/ and stage it; see NCCL_PATCH_TESTS.md)"
        install -D -m 0755 "$NCCL_CKPT_PATCH_SRC" \
            "${ROOTFS_DIR}${NCCL_CKPT_PATCH_PATH}" \
            && ok "Staged patched NCCL at ${NCCL_CKPT_PATCH_PATH}" \
            || die "failed to stage $NCCL_CKPT_PATCH_SRC"
    fi

    mount -t overlay overlay \
        -o "lowerdir=${ROOTFS_DIR},upperdir=${ROOTFS_UPPER},workdir=${ROOTFS_WORK}" \
        "$ROOTFS_MERGED" && ok "Overlay mounted at $ROOTFS_MERGED" || {
        fail "Failed to mount rootfs overlay"; exit 1
    }
}

# The image has the CUDA runtime but NOT the host driver userspace
# (libcuda.so, libnvidia-*.so).  Copy them into the rootfs so they are
# available when runsc overlays on top.
_cb_inject_nvidia_libs() {
    info "  Injecting host NVIDIA driver libraries …"
    local dst_lib="$ROOTFS_DIR/usr/lib/x86_64-linux-gnu"
    local dst_bin="$ROOTFS_DIR/usr/bin"
    mkdir -p "$dst_lib" "$dst_bin"
    local nlibs=0 src dst
    while IFS= read -r src; do
        [[ -z "$src" ]] && continue
        [[ "$src" == /dev/* ]] && continue    # device nodes come from the OCI spec
        if [[ "$src" == *.so* ]]; then
            # Only copy 64-bit (ELF64) shared libraries; nvidia-container-cli
            # lists 32-bit variants too which would clobber the 64-bit ones.
            if file -L "$src" 2>/dev/null | grep -q "ELF 64-bit"; then
                cp -aL "$src" "$dst_lib/" 2>/dev/null && nlibs=$((nlibs + 1)) || true
            fi
        elif [[ -x "$src" ]]; then
            cp -aL "$src" "$dst_bin/" 2>/dev/null || true
        else
            dst="$ROOTFS_DIR$src"
            mkdir -p "$(dirname "$dst")"
            cp -aL "$src" "$dst" 2>/dev/null || true
        fi
    done < <(nvidia-container-cli list 2>/dev/null || true)

    if [[ -d /usr/lib/firmware/nvidia ]]; then
        mkdir -p "$ROOTFS_DIR/usr/lib/firmware"
        cp -a /usr/lib/firmware/nvidia "$ROOTFS_DIR/usr/lib/firmware/" 2>/dev/null || true
    fi

    ldconfig -r "$ROOTFS_DIR" 2>/dev/null || \
        chroot "$ROOTFS_DIR" /sbin/ldconfig 2>/dev/null || true

    if chroot "$ROOTFS_DIR" /sbin/ldconfig -p 2>/dev/null | grep -q libcuda; then
        ok "NVIDIA driver libs injected ($nlibs files, libcuda.so found)"
    elif [[ $nlibs -gt 0 ]]; then
        warn "Copied $nlibs nvidia libs but libcuda.so not in ldconfig cache"
    else
        warn "No NVIDIA libs found on host — GPU init may fail"
    fi
}

# ── Phase: OCI bundle generation ──────────────────────────────────────
# Requires: CB_CMD              shell command line to exec (run via sh -c)
#           CB_ENV              newline-separated KEY=VALUE pairs (may be empty)
#           GPU_DEVICES         comma-separated host GPU indices ("" for CPU-only)
#           RESTORE_GPU_DEVICES optional different GPU set for restore; if set
#                               (and different), a second bundle is generated
#                               and used by cb_restore_and_wait_health.  The
#                               sentry records the saved device set (minor +
#                               UUID) in checkpoint metadata and nvproxy remaps
#                               device FDs onto the restore bundle's GPUs
#                               (sorted by minor, positionally).
#           CB_SHM_MB           /dev/shm size in MiB (default 4096)
cb_write_bundle() {
    info "Generate OCI bundle"
    _cb_write_bundle_config "${GPU_DEVICES:-}" "$BUNDLE_DIR"
    ok "Bundle at $BUNDLE_DIR/config.json (GPUs: ${GPU_DEVICES:-none})"
    if [[ -n "${RESTORE_GPU_DEVICES:-}" && "$RESTORE_GPU_DEVICES" != "${GPU_DEVICES:-}" ]]; then
        mkdir -p "$RESTORE_BUNDLE_DIR"
        _cb_write_bundle_config "$RESTORE_GPU_DEVICES" "$RESTORE_BUNDLE_DIR"
        ok "Restore bundle at $RESTORE_BUNDLE_DIR/config.json (GPUs: $RESTORE_GPU_DEVICES)"
    else
        RESTORE_BUNDLE_DIR="$BUNDLE_DIR"
    fi
}

_cb_write_bundle_config() {
    local gpus="$1" outdir="$2"
    CB_ROOTFS="$ROOTFS_MERGED" \
    CB_HOSTNAME="$BENCH_NAME" \
    CB_GPUS="$gpus" \
    CB_SHM_MB="${CB_SHM_MB:-4096}" \
    CB_CMD="$CB_CMD" \
    CB_ENV="${CB_ENV:-}" \
    CB_DRIVER_VER="${HOST_DRIVER_VER:-}" \
    CB_APPLOG_DIR="$APPLOG_DIR" \
    python3 - > "$outdir/config.json" << 'PYEOF'
import json, os, sys

rootfs   = os.environ["CB_ROOTFS"]
hostname = os.environ["CB_HOSTNAME"]
gpus     = [g for g in os.environ["CB_GPUS"].split(",") if g != ""]
shm_mb   = int(os.environ["CB_SHM_MB"])
cmd      = os.environ["CB_CMD"]
extra_env = [e for e in os.environ["CB_ENV"].splitlines() if e.strip()]
drv_ver  = os.environ["CB_DRIVER_VER"]

caps = [
    "CAP_CHOWN", "CAP_DAC_OVERRIDE", "CAP_FSETID", "CAP_FOWNER",
    "CAP_MKNOD", "CAP_NET_RAW", "CAP_SETGID", "CAP_SETUID",
    "CAP_SETFCAP", "CAP_SETPCAP", "CAP_NET_BIND_SERVICE",
    "CAP_SYS_CHROOT", "CAP_KILL", "CAP_AUDIT_WRITE",
]

env = [
    "PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "HOME=/root",
    "TERM=xterm",
    "XDG_CACHE_HOME=/tmp",
]
if gpus:
    env += [
        "LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:"
            "/usr/lib/x86_64-linux-gnu",
        "NVIDIA_VISIBLE_DEVICES=" + ",".join(gpus),
        "NVIDIA_DRIVER_CAPABILITIES=all",
        f"NVIDIA_DRIVER_VERSION={drv_ver}",
    ]
# extra_env overrides base env: duplicate variables resolve to the FIRST
# occurrence in environ, so drop overridden base entries. As a special
# case, LD_LIBRARY_PATH in extra_env is appended to the base value.
for e in extra_env:
    key = e.split("=", 1)[0]
    for i, b in enumerate(env):
        if b.split("=", 1)[0] == key:
            if key == "LD_LIBRARY_PATH":
                e = b + ":" + e.split("=", 1)[1]
            del env[i]
            break
    env.append(e)

mounts = [
    {"destination": "/proc", "type": "proc", "source": "proc"},
    {"destination": "/dev", "type": "tmpfs", "source": "tmpfs",
     "options": ["nosuid", "strictatime", "mode=755", "size=65536k"]},
    {"destination": "/dev/pts", "type": "devpts", "source": "devpts",
     "options": ["nosuid", "noexec", "newinstance",
                 "ptmxmode=0666", "mode=0620"]},
    {"destination": "/dev/shm", "type": "tmpfs", "source": "shm",
     "options": ["nosuid", "noexec", "nodev", "mode=1777",
                 f"size={shm_mb * 1024 * 1024}"]},
    {"destination": "/sys", "type": "sysfs", "source": "sysfs",
     "options": ["nosuid", "noexec", "nodev", "ro"]},
    {"destination": "/tmp", "type": "tmpfs", "source": "tmpfs",
     "options": ["nosuid", "mode=1777"]},
    {"destination": "/run", "type": "tmpfs", "source": "tmpfs",
     "options": ["nosuid", "strictatime", "mode=755", "size=65536k"]},
    # Host-backed app log dir so application output survives crashes.
    {"destination": "/applog", "type": "bind",
     "source": os.environ["CB_APPLOG_DIR"], "options": ["bind", "rw"]},
]

devices = []
for g in gpus:
    devices.append({"path": f"/dev/nvidia{g}", "type": "c",
                    "major": 195, "minor": int(g),
                    "fileMode": 0o666, "uid": 0, "gid": 0})
if gpus:
    devices += [
        {"path": "/dev/nvidiactl", "type": "c", "major": 195, "minor": 255,
         "fileMode": 0o666, "uid": 0, "gid": 0},
        {"path": "/dev/nvidia-uvm", "type": "c", "major": 240, "minor": 0,
         "fileMode": 0o666, "uid": 0, "gid": 0},
        {"path": "/dev/nvidia-uvm-tools", "type": "c", "major": 240, "minor": 1,
         "fileMode": 0o666, "uid": 0, "gid": 0},
    ]

spec = {
    "ociVersion": "1.0.0",
    "process": {
        "terminal": False,
        "user": {"uid": 0, "gid": 0},
        "args": ["sh", "-c", cmd],
        "env": env,
        "cwd": "/app",
        "capabilities": {
            "bounding":  caps,
            "effective": caps,
            "permitted": caps,
            "ambient":   caps,
        },
        "rlimits": [
            {"type": "RLIMIT_NOFILE", "hard": 1048576, "soft": 1048576},
            {"type": "RLIMIT_MEMLOCK", "hard": 18446744073709551615,
                                       "soft": 18446744073709551615},
        ],
    },
    "root": {"path": rootfs, "readonly": False},
    "hostname": hostname,
    "mounts": mounts,
    "linux": {
        "namespaces": [
            {"type": "pid"},
            {"type": "mount"},
            {"type": "ipc"},
            {"type": "uts"},
            {"type": "network"},
        ],
        "devices": devices,
    },
}

json.dump(spec, sys.stdout, indent=2)
print()
PYEOF
}

# ── Phase: cold boot ──────────────────────────────────────────────────────
# Boots CONTAINER_ID and waits for /health.  Sets T_COLD_BOOT (ms).
# APP_LOG (default /tmp/app.log) is tailed for progress.
cb_run_and_wait_health() {
    local app_log="${APP_LOG:-/applog/app.log}"
    local app_log_host="$APPLOG_DIR/${app_log##*/}"
    info "runsc run (cold boot)"

    local t0 rc=0
    t0=$(ts_ms)
    "$RUNSC" "${RUNSC_FLAGS[@]}" run \
        --detach \
        --pid-file="$LOG_DIR/run.pid" \
        --bundle="$BUNDLE_DIR" \
        "$CONTAINER_ID" \
        >"$LOG_DIR/runsc-run.log" 2>&1 || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        fail "runsc run failed (exit $rc)"
        tail -20 "$LOG_DIR/runsc-run.log" 2>/dev/null
        _cb_dump_boot_log
        exit 1
    fi
    ok "runsc run returned (container=$CONTAINER_ID)"

    info "  Waiting for health on localhost:${PORT} (timeout ${HEALTH_TIMEOUT}s) …"
    local i cs
    for i in $(seq 1 "$HEALTH_TIMEOUT"); do
        cs=$(cb_state "$CONTAINER_ID")
        if [[ "$cs" != "running" && "$cs" != "created" ]]; then
            fail "Container exited ($cs) during cold boot"
            echo "--- $app_log_host (last 40 lines) ---"
            tail -40 "$app_log_host" 2>/dev/null || true
            _cb_dump_boot_log
            exit 1
        fi
        cb_health "$CONTAINER_ID" && break
        if (( i % 15 == 0 )); then
            local line
            line=$(tail -1 "$app_log_host" 2>/dev/null || true)
            echo "    [${i}s] ${line:-waiting… (container=$cs)}"
        fi
        if [[ "$i" -eq "$HEALTH_TIMEOUT" ]]; then
            fail "Timed out waiting for health (${HEALTH_TIMEOUT}s)"
            tail -40 "$app_log_host" 2>/dev/null || true
            exit 1
        fi
        sleep 1
    done
    T_COLD_BOOT=$(( $(ts_ms) - t0 ))
    ok "App ready — cold boot: ${T_COLD_BOOT} ms"
}

# ── Phase: checkpoint ─────────────────────────────────────────────────────
# Sets T_CHECKPOINT (ms), PAGES_SIZE, STATE_SIZE, TOTAL_SIZE.
# With CB_GPU=1 this passes --cuda-checkpoint-path, activating gVisor's
# native CUDA checkpoint support.
cb_checkpoint() {
    info "runsc checkpoint"
    local ckpt_flags=(--image-path="$CKPT_DIR" --compression="$COMPRESSION")
    [[ "$EXCLUDE_ZERO" = "1" ]] && ckpt_flags+=(--exclude-committed-zero-pages)
    [[ "$PAGES_IDENTITY" = "1" ]] && ckpt_flags+=(--pages-layout=identity)
    if [[ "$CB_GPU" = "1" ]]; then
        ckpt_flags+=(--cuda-checkpoint-path="$CUDA_CHECKPOINT_PATH")
        [[ "$CUDA_CKPT_SEQUENTIAL" = "1" ]] && ckpt_flags+=(--cuda-checkpoint-sequential)
    fi
    info "  flags: ${ckpt_flags[*]}"

    local t0 rc=0
    t0=$(ts_ms)
    "$RUNSC" "${RUNSC_FLAGS[@]}" checkpoint "${ckpt_flags[@]}" "$CONTAINER_ID" \
        >"$LOG_DIR/runsc-checkpoint.log" 2>&1 || rc=$?
    T_CHECKPOINT=$(( $(ts_ms) - t0 ))

    if [[ "$rc" -ne 0 ]]; then
        fail "runsc checkpoint failed (exit $rc, ${T_CHECKPOINT} ms)"
        tail -20 "$LOG_DIR/runsc-checkpoint.log" 2>/dev/null
        _cb_dump_boot_log
        exit 1
    fi
    if [[ ! -f "$CKPT_DIR/pages.img" ]]; then
        fail "No pages.img after checkpoint"
        ls -lha "$CKPT_DIR/" 2>/dev/null | sed 's/^/    /'
        exit 1
    fi

    PAGES_SIZE=$(du -sh "$CKPT_DIR/pages.img" 2>/dev/null | cut -f1)
    STATE_SIZE=$(du -sh "$CKPT_DIR/checkpoint.img" 2>/dev/null | cut -f1)
    TOTAL_SIZE=$(du -sh "$CKPT_DIR" 2>/dev/null | cut -f1)
    ok "Checkpoint: ${T_CHECKPOINT} ms"
    if [[ "$PAGES_IDENTITY" = "1" ]]; then
        # Identity pages.img is sparse; show both apparent and actual size.
        local pages_apparent
        pages_apparent=$(du -sh --apparent-size "$CKPT_DIR/pages.img" 2>/dev/null | cut -f1)
        ok "  pages.img:      $PAGES_SIZE (sparse, apparent $pages_apparent)"
    fi
    [[ "$PAGES_IDENTITY" != "1" ]] && ok "  pages.img:      $PAGES_SIZE"
    ok "  checkpoint.img: $STATE_SIZE"
    ok "  Total:          $TOTAL_SIZE"
}

# ── Phase: restore ──────────────────────────────────────────────────────────
# cb_restore deletes the checkpointed container and restores into
# RESTORE_ID; sets T_RESTORE_START and T_RESTORE_RETURNED. On restore, the
# sentry automatically re-execs cuda-checkpoint --toggle on all CUDA
# processes recorded at checkpoint time — no flags needed.
cb_restore() {
    info "runsc restore"
    "$RUNSC" --root "$RUNSC_ROOT" delete --force "$CONTAINER_ID" 2>/dev/null || true
    "$RUNSC" --root "$RUNSC_ROOT" delete --force "$RESTORE_ID"  2>/dev/null || true

    # DROP_CACHES=1: drop the page cache before restoring, so pages.img
    # reads hit storage rather than the cache the checkpoint just
    # populated. (No effect on tmpfs-backed images: shmem pages are not
    # dropped.) Gives honest cold-restore numbers.
    if [[ "${DROP_CACHES:-0}" = "1" ]]; then
        sync
        echo 3 > /proc/sys/vm/drop_caches
        info "  dropped page caches (cold restore)"
    fi

    local restore_flags=()
    if [[ "$RESTORE_BACKGROUND" = "1" ]]; then
        restore_flags+=(--background)
        info "  restore mode: --background (async page loading)"
    fi
    if [[ "$ADOPT_PAGES" = "1" ]]; then
        restore_flags+=(--adopt-pages-file)
        info "  restore mode: --adopt-pages-file (zero-copy: pages.img adopted as memory file)"
    fi

    local rc=0
    T_RESTORE_START=$(ts_ms)
    "$RUNSC" "${RUNSC_FLAGS[@]}" restore \
        --detach \
        "${restore_flags[@]}" \
        --image-path="$CKPT_DIR" \
        --bundle="$RESTORE_BUNDLE_DIR" \
        --pid-file="$LOG_DIR/restore.pid" \
        "$RESTORE_ID" \
        >"$LOG_DIR/runsc-restore.log" 2>&1 || rc=$?
    T_RESTORE_RETURNED=$(( $(ts_ms) - T_RESTORE_START ))

    if [[ "$rc" -ne 0 ]]; then
        fail "runsc restore exited $rc (${T_RESTORE_RETURNED} ms)"
        tail -20 "$LOG_DIR/runsc-restore.log" 2>/dev/null
        _cb_dump_boot_log
        exit 1
    fi
    ok "runsc restore returned: ${T_RESTORE_RETURNED} ms"

    local rstate
    rstate=$(cb_state "$RESTORE_ID")
    if [[ "$rstate" != "running" ]]; then
        fail "Container not running after restore (state=$rstate)"
        tail -30 "$LOG_DIR/runsc-restore.log" 2>/dev/null
        _cb_dump_boot_log
        exit 1
    fi
}

# cb_wait_health_restored waits (wall-clock bounded, default 120s, override
# with RESTORE_HEALTH_TIMEOUT) for RESTORE_ID's /health. Sets HEALTH_OK and
# T_HEALTH_MS (from restore start). Progress labels are real elapsed time —
# each failed probe is a runsc exec + curl and can take seconds, so
# iteration counts would lie.
cb_wait_health_restored() {
    HEALTH_OK=0
    T_HEALTH_MS=0
    local deadline_ms=$(( ${RESTORE_HEALTH_TIMEOUT:-120} * 1000 ))
    local elapsed last_print=0 cs
    while :; do
        if cb_health "$RESTORE_ID"; then
            T_HEALTH_MS=$(( $(ts_ms) - T_RESTORE_START ))
            ok "Health: ${T_HEALTH_MS} ms after restore"
            HEALTH_OK=1
            break
        fi
        elapsed=$(( $(ts_ms) - T_RESTORE_START ))
        if (( elapsed > deadline_ms )); then
            warn "Health never responded (${RESTORE_HEALTH_TIMEOUT:-120}s)"
            tail -20 "$APPLOG_DIR/$(basename "${APP_LOG:-/applog/app.log}")" 2>/dev/null || true
            break
        fi
        if (( elapsed - last_print > 5000 )); then
            cs=$(cb_state "$RESTORE_ID")
            if [[ "$cs" != "running" ]]; then
                fail "Container died waiting for health ($cs)"
                tail -40 "$APPLOG_DIR/$(basename "${APP_LOG:-/applog/app.log}")" 2>/dev/null || true
                _cb_dump_boot_log
                break
            fi
            echo "    $(( elapsed / 1000 ))s … container=$cs"
            last_print=$elapsed
        fi
        sleep 0.1
    done
}

cb_restore_and_wait_health() {
    cb_restore
    cb_wait_health_restored
}

# ── GPU placement verification (device remapping) ────────────────────────
# After a cross-GPU restore, verify from the HOST side that GPU memory is
# resident on the restore GPUs and NOT on the original ones.
# Sets PLACEMENT_OK (1/0).  Args: <min_mib_expected_per_restore_gpu>
cb_verify_gpu_placement() {
    local min_mib="${1:-200}"
    PLACEMENT_OK=1
    if [[ -z "${RESTORE_GPU_DEVICES:-}" || "$RESTORE_GPU_DEVICES" == "${GPU_DEVICES:-}" ]]; then
        return 0
    fi
    info "Verify GPU placement (host nvidia-smi)"
    local g used
    for g in ${RESTORE_GPU_DEVICES//,/ }; do
        used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || echo 0)
        if (( used >= min_mib )); then
            ok "GPU $g (restore set): ${used} MiB in use"
        else
            fail "GPU $g (restore set): only ${used} MiB in use (expected >= ${min_mib})"
            PLACEMENT_OK=0
        fi
    done
    for g in ${GPU_DEVICES//,/ }; do
        # Skip GPUs that are in both sets.
        case ",$RESTORE_GPU_DEVICES," in *",$g,"*) continue ;; esac
        used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || echo 0)
        if (( used < min_mib )); then
            ok "GPU $g (original set): ${used} MiB in use (released)"
        else
            fail "GPU $g (original set): still ${used} MiB in use"
            PLACEMENT_OK=0
        fi
    done
}

# ── Debug-log profiling: cuda-checkpoint timings ──────────────────────────
cb_cuda_ckpt_log_summary() {
    info "cuda-checkpoint sentry log summary"
    local found=0 blog
    for blog in $(ls -t "$LOG_DIR"/*boot* 2>/dev/null); do
        local lines
        lines=$(grep -hE "cuda-checkpoint on [0-9]+ processes took|starting cuda-ckpt|cuda-ckpts (invoked|waited|done)|Remapping [0-9]+ nvproxy devices|nvproxy.*=> " "$blog" 2>/dev/null | head -30 || true)
        if [[ -n "$lines" ]]; then
            found=1
            echo "    --- $(basename "$blog") ---"
            echo "$lines" | sed 's/^/      /'
        fi
    done
    if [[ "$found" = "0" ]]; then
        warn "No cuda-checkpoint entries in sentry logs"
    fi
}

_cb_dump_boot_log() {
    local blog
    blog=$(ls -t "$LOG_DIR"/*boot* 2>/dev/null | head -1 || true)
    if [[ -n "$blog" ]]; then
        echo "--- gVisor boot log (last 30 lines) ---"
        tail -30 "$blog"
        echo "---"
    fi
}

# ── Summary helpers ───────────────────────────────────────────────────────
row()    { printf "  %-46s %s\n" "$1" "$2"; }
speedup() {
    # speedup <numerator_ms> <denominator_ms>
    python3 -c "print(f'{$1 / $2:.1f}x')" 2>/dev/null || echo "?"
}
