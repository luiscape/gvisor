# Multi-GPU checkpoint/restore bench (dev only, not for upstream)

Validation harness for the multicast interposer flow: one
sleep/checkpoint/restore/wake cycle of vLLM or SGLang per run, under runsc.

Run it from a copy outside the working tree (e.g. `~/bench`): switching
branches removes this directory, which would break a running gate.

## Files

-   `pull.sh` -- engine images, the gate's models (into `$HF_DIR`, default
    `/data/hf`) and `cuda-checkpoint` (into `/usr/local/bin`).
-   `prep_rootfs.sh ENGINE` -- cached rootfs in `$BENCH_DIR/rootfs/ENGINE`
    (default `/data/bench`): the image's filesystem, the host driver's
    userspace and `cuda-checkpoint`. Rebuilt when the host driver changes.
-   `gen_bundle.py` -- OCI spec: explicit `/dev/nvidiaN` devices (no toolkit
    hook), and `/tmp` as a sentry tmpfs so the interposer's `/tmp/mcshim` is
    part of the checkpoint image.
-   `bench.sh` -- one cycle: boot, reference completion, sleep, checkpoint,
    restore (onto `--restore-gpus` if given), wake, then compare the output
    and the GPU placement. `--no-sleep` checkpoints an awake engine.
-   `gate.sh BIN_DIR PREFIX CELLS...` -- runs cells one at a time and writes
    `$BENCH_DIR/PREFIX-gate.txt`; `KEEP_CKPT=0` deletes passing images.

## Host setup (Ubuntu 26.04, 8 GPUs with NVSwitch)

1.  Driver and fabric manager 610.57.04. On B300 the fabric manager also
    needs `nvlsm` and the `ib_umad` module (`/etc/modules-load.d/rdma.conf`).
2.  `/data` on the instance-store NVMe disks (check `lsblk` for the names;
    lost on stop/start):

        sudo mdadm --create /dev/md0 --level=0 --raid-devices=8 --run /dev/nvme[1-8]n1
        sudo mkfs.xfs -f -K /dev/md0 && sudo mount -o noatime /dev/md0 /data

3.  Docker with `{"data-root": "/data/docker"}` in `/etc/docker/daemon.json`,
    and your user in the `docker` group. Optionally
    `ln -s /data/cache/bazel ~/.cache/bazel`.
4.  `echo always | sudo tee /sys/kernel/mm/transparent_hugepage/shmem_enabled`
    (restore performance; not persistent).
5.  Stop the OS from restarting the fabric manager: `unattended-upgrades` runs
    `needrestart`, which restarts `nvidia-fabricmanager` mid-run. That
    desyncs the NVLink fabric (Xid 145, then `NV_ERR_FABRIC_STATE_OUT_OF_SYNC`
    on every multicast setup) until all GPUs are reset:

        sudo systemctl disable --now unattended-upgrades apt-daily.timer apt-daily-upgrade.timer
        echo '$nrconf{override_rc}{qr(^nvidia-)} = 0;' | sudo tee /etc/needrestart/conf.d/90-gpu.conf

    To recover: stop the fabric manager and persistenced, `nvidia-smi -r`,
    then start them again.
6.  `bash pull.sh`.

## Build and run

    sg docker -c 'make build TARGETS=//:release'
    sudo mkdir -p /data/bench/NAME-bin/gvisor-bin
    sudo cp bazel-bin/release/runsc /data/bench/NAME-bin/
    sudo cp bazel-bin/release/gvisor-bin/* /data/bench/NAME-bin/gvisor-bin/
    KEEP_CKPT=0 nohup bash ~/bench/gate.sh /data/bench/NAME-bin NAME [CELLS...] \
        > /data/bench/NAME.nohup 2>&1 &

`gate.sh` lists the cells; with none given it runs a 7-cell subset.

## Gotchas

-   vLLM serves `/sleep` and `/wake_up` only with `VLLM_SERVER_DEV_MODE=1`
    (`bench.sh` sets it).
-   The images carry an R580 forward-compat `libcuda` in
    `/usr/local/cuda/compat`; `prep_rootfs.sh` fails unless the host driver's
    library is the one resolved.
-   runsc chowns files it gets as stdio to root, so never point a runsc
    command's output at a file this shell appends to later.
-   TP=8 cells serve Qwen2.5-3B-Instruct: the 1.5B model's 12 attention heads
    do not split 8 ways.
