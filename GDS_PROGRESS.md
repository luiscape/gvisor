# GPUDirect Storage (GDS) in nvproxy — Design & Handoff

**Status:** the full proxy is implemented and compiles. `REMOVE` and `mmap` are
hardware-validated. **`MAP` is BLOCKED**: with the real GDS stack installed, a
real `cuFileBufRegister`-equivalent `MAP` succeeds on bare metal but fails with
`EFAULT` inside the sandbox, because the sentry cannot mirror the application's
mapping of `/dev/nvidia-fs`. This is a design-level blocker with a known fix —
see [section 6a](#6a-blocker-shadow-buffer-mirroring-is-a-dead-end), which is
**the next thing to work on**. `READ`/`WRITE` is implemented but unreachable
until `MAP` works.

Branch: `luis/gds-capability`.

---

## 1. Goal & scope

Add a `gpudirect-storage` driver capability to gVisor's `nvproxy` that
intercepts GPUDirect Storage (cuFile / nvidia-fs) calls and proxies the
`/dev/nvidia-fs*` character-device ioctls and `mmap` from the sandbox to the
host — the same way nvproxy already proxies CUDA — so NVMe→GPU DMA works inside
the sandbox.

**In scope:** local NVMe→GPU DMA via the non-RDMA, non-batch nvidia-fs build.

**Out of scope (for now):** GDS-over-RDMA (`*_RDMA_REG_INFO` ioctls, `use_rkeys`),
the batch ioctl, checkpoint/restore of live GDS state.

## 2. Design decisions

- **Intercept in nvproxy. No `LD_PRELOAD`, no daemon.** We proxy
  `/dev/nvidia-fs<N>` ioctls/mmap directly, exactly like the existing GPU
  device proxying. (`gds_proxy/` and `tools/gds_proxy/` in the worktree are a
  reference-only userspace prototype and are *not* part of the implementation.)
- **Data-file FDs reach the Sentry via DirectFS.** The gofer donates host FDs;
  `gpudirect-storage` therefore **requires `--directfs`**, enforced as a
  precondition in `runsc/config`.
- **Capability is privileged/opt-in**, modeled on `CapProfiling`: present in
  `SupportedDriverCaps`, excluded from `AllContainerDriverCaps` (so `all` does
  not grant it). Gated via `nvconf.CapGPUDirectStorage`.

## 3. ABI ground truth

Verified against `github.com/NVIDIA/gds-nvidia-fs` (`nvfs-core.c`,
`nvfs-mmap.c`, `master`) and confirmed by `strace` on a real host.

- `/dev/nvidia-fs0..15` — 16 nodes (`nvgpu.NVIDIA_FS_DEV_COUNT = 16`); cuFile
  opens all of them. Host major was 239 on the test box; the Sentry remaps to
  whatever `/proc/devices` reports inside the sandbox.
- ioctls are `_IOW('t', nr, int)` = `0x40047400 | nr`:
  `REMOVE=0x40047401`, `READ=0x40047402`, `MAP=0x40047403`, `WRITE=0x40047404`.
- `nvfs_ioctl()` does `copy_from_user(sizeof(union))` for **every** command, so
  we must forward the full 80-byte `NvfsIoctlParamUnion`, never the per-command
  struct.
- `REMOVE` is a no-op in the driver.
- **The shadow buffer is kernel-owned, not app memory.**
  `nvfs_mgroup_mmap_internal()` `alloc_page()`s fresh pages, tags each with an
  encoded `page->index` (a random `base_index`), and `vm_insert_page()`s them.
  `nvfs_mgroup_pin_shadow_pages()` then asserts the pinned pages are *those
  exact `struct page`s* — via **`BUG_ON`, not an error return**.
- **`mmap` must be at offset 0.** `nvfs_mgroup_mmap()` returns `-EIO` for any
  non-zero `vm_pgoff`, and allocates **distinct** shadow pages per mmap call.
- `nvfs_io_init()` (the `READ`/`WRITE` entry point) validates, in order:
  offset/size 4096-aligned; `fdget(ioargs->fd)`; `FMODE_READ`/`FMODE_WRITE`
  permissions; `file_args.inum != 0` else `EINVAL`; `inum == inode->i_ino`,
  `majdev/mindev == get_major/get_minor(inode)` else **`ESTALE`**; `generation`
  checked only when non-zero; `nvfs_get_mgroup_from_vaddr(cpuvaddr)` which
  requires `cpuvaddr == mgroup->cpu_base_vaddr` recorded at `MAP`; and
  `file->f_flags & O_DIRECT` else `EINVAL`.
- `nvfs_io_start_op()` then passes `cpuvaddr` as the userspace buffer to
  `call_read_iter`/`call_write_iter` — i.e. the shadow VA is *also* the
  O_DIRECT DMA target, `get_user_pages`'d in the **sentry's** mm.

## 4. Architecture

Every nvidia-fs ioctl is forwarded from the sentry, so every sandbox-relative
field must be rewritten to a host-relative one first.

### `MAP` (cuFileBufRegister)
1. Translate `cpuvaddr` (app VA of the shadow buffer) to a **sentry VA** backed
   by the same host nvidia-fs pages: `MemoryManager.Pin` → `MapInternal` →
   `mremap` into a fresh reservation (mirrors `rmAllocOSDescriptor`).
2. Translate `end_fence_addr` the same way.
3. Forward; on success **retain** the shadow mapping in a registry (below) and
   release only the end-fence mapping (the host holds its own pin and kmaps the
   page directly, so it does not need our mapping).
4. Restore the guest-visible addresses before copy-out so the app never sees
   sentry VAs.

### The shadow registry (`nvproxy.gdsShadows`)
`READ`/`WRITE` re-pin `cpuvaddr` and require it to equal the `cpu_base_vaddr`
recorded at `MAP`, so the mapping must persist at the *same* sentry VA.

- Key: `(app *mm.MemoryManager, shadow app VA)` — mirrors the driver resolving
  `cpuvaddr` in `current->mm`. The MM pointer is an opaque key; no reference is
  held (safe because a registered buffer keeps its process alive, and process
  death closes its `nvfsFD`s, releasing the entries).
- Scope: **per-`nvproxy`, not per-FD** — cuFile spreads I/O across the 16
  device FDs, so `MAP` and `READ` can legitimately land on different FDs.
- Ownership: the `nvfsFD` that handled the `MAP` owns cleanup; released in
  `nvfsFD.Release`, or replaced on re-registration at the same VA.

### `READ`/`WRITE` (cuFileRead/Write)
1. `cpuvaddr` → the stable sentry VA from the registry (fail fast if absent).
2. Data-file FD → gofer's donated host FD (`hostFDForGDSer`) → **re-opened
   `O_DIRECT`** with a matching access mode via `/proc/self/fd/<hostFD>`,
   because the gofer's FD guarantees neither `O_DIRECT` nor the right
   `FMODE_*`.
3. `file_args.{Inum,MajDev,MinDev}` rewritten from an `fstat` of the host FD;
   `Generation` zeroed (the driver only checks it when non-zero, and it isn't
   recoverable via `fstat`). `DevPtrOff` is preserved.
4. Forward, then restore all guest-observable inputs before copy-out, keeping
   only the driver-written `IoctlReturn`.

## 5. Code map

**ABI** — `pkg/abi/nvgpu/nvfs.go` (+`nvfs_test.go`): ioctl constants,
`NvfsIoctlMap` (48B), `NvfsFileArgs` (32B, packed), `NvfsIoctlIoargs` (80B),
`NvfsIoctlParamUnion` (80B), `NVFS_BLOCK_SIZE`, `NVIDIA_FS_DEV_COUNT`.

**nvproxy** — `pkg/sentry/devices/nvproxy/`:
- `nvfs.go` — device (16 nodes), `nvfsFD`, ioctl dispatch, and:
  `nvfsIoctlRemove`, `nvfsIoctlMap` (+ registry persistence), `nvfsIoctlReadWrite`
  (all three translations), `nvfsPinAndMap`, `nvfsOpenDirectHostFD`,
  `nvfsRewriteFileArgs`, and the registry (`gdsShadowKey`, `gdsShadowMapping`,
  `storeGDSShadow`, `lookupGDSShadow`, `releaseGDSShadows`).
- `nvfs_mmap.go` — `ConfigureMMap`/`Translate`, host-FD-backed `memmap.File`.
- `nvfs_unsafe.go` — `nvfsIoctlInvoke`.
- `nvproxy.go` — `gdsShadows` map + mutex; registers the 16 nodes when the cap
  is enabled; `DeviceInfo.NvfsDevMajor`.
- `version.go`, `handlers.go`, `seccomp_filters.go`, `save_restore.go`,
  `nvconf/caps.go`, `README.md`.

**runsc** — `runsc/config/config.go` (directfs precondition),
`runsc/cmd/sandboxsetup/gofer_mount.go` (`ShouldExposeNvidiaDevice`),
`runsc/boot/vfs.go` (device-node discovery/creation).

**gofer** — `pkg/sentry/fsimpl/gofer/regular_file.go`:
`HostFDForGPUDirectStorage(write)` (GDS accessor) and `HostFD()`
(`vfs.HostFDProvider` convention), with compile-time assertions for both.

## 6. Validation status

**Hardware-validated** (AWS g5, 4×A10G, driver 580.95.05, kernel 6.1.163) — a
direct-ioctl harness produced results *byte-for-byte identical* between bare
metal and the sandbox:

| Step | Bare metal | Through `runsc-gds` |
|---|---|---|
| `open /dev/nvidia-fs0..15` | ok | ok |
| `REMOVE` ×16 | `0` | `0` |
| `mmap` (4 KiB) | ok | ok (host-backed) |
| `MAP` (`gpuvaddr=0`) | `-1 EPERM` | `-1 EPERM` |

This proves: device-node creation, dev-gofer/directfs FD donation into the
handler, seccomp, ioctl dispatch, `REMOVE`, and the `mmap` proxy.

**Executed and FAILING:** `MAP` against a *real* registered shadow buffer —
see [section 6a](#6a-blocker-shadow-buffer-mirroring-is-a-dead-end). This is now
the top blocker.

**Never executed** (compile- and source-verified only):
- The shadow registry (`MAP` fails before reaching `storeGDSShadow`).
- The entire `nvfsIoctlReadWrite` body.

**Why it can't be validated here:** the test host has
`# CONFIG_PCI_P2PDMA is not set`, so cuFile refuses GDS and falls back to compat
mode (POSIX + `cudaMemcpy`), never opening `/dev/nvidia-fs`. A `gdsio` run here
measures compat-mode bandwidth that bypasses the proxy entirely.

## 6a. BLOCKER: shadow-buffer mirroring is a dead end

**Hardware-proven on 2026-08-08.** With the full GDS stack installed (nvidia-fs
2.29.4, libcufile/gds-tools 1.17.1.22), a purpose-built harness
(`cuMemAlloc` + `mmap(/dev/nvidia-fs0)` + `NVFS_IOCTL_MAP`, i.e. exactly what
`cuFileBufRegister` does) gives:

| | bare metal | through `runsc-gds` |
|---|---|---|
| real `MAP` (real GPU buf + real shadow buf) | **0 (success)** | **-1 `EFAULT`** |

Sentry diagnostics localise it precisely:

```
pin-and-map: pinned range src=0x… len=65536 File=*nvproxy.nvfsFDMemmapFile off=0x0
pin-and-map: MapInternal ok, numBlocks=1
pin-and-map: mremap(src=… len=65536 dst=…) failed: bad address
```

So `Pin` ✓, reservation `mmap` ✓, `MapInternal` ✓ (correct File, offset 0) — and
then **`mremap` fails with `EFAULT`**. The host driver logged nothing in
`dmesg`, so the call never reached nvfs's `vm_ops`; the kernel rejected the
duplication first.

**Conclusion: `nvfsPinAndMap`'s "mirror the application's mapping into the
sentry" strategy cannot work for nvidia-fs shadow buffers.** The pattern is
borrowed from `rmAllocOSDescriptor`, which mirrors *anonymous application
memory* (`pr.File` is the sentry's `MemoryFile`, freely duplicable). Here
`pr.File` is a mapping of the nvidia-fs character device, and nvidia-fs is
actively hostile to VMA duplication — its `vm_operations_struct` defines
`.mremap = nvfs_vma_mremap` (returns `-ENOMEM`, `WARN_ON_ONCE`), `.open =
nvfs_vma_open` (`WARN_ON_ONCE`, clears `vm_private_data`), and `.split` /
`.fault` as hard errors.

### Recommended redesign: the sentry should OWN the shadow buffer

The shadow buffer is **not** a data buffer the application reads. It is a
kernel-side handle: `nvfs_io_start_op()` passes `cpuvaddr` as the O_DIRECT
buffer, the block layer builds a bio from those pages, and nvfs's DMA ops
(`nvfs_get_dma`) substitute **GPU BAR** addresses — the data never lands in the
shadow pages. So its contents need not be visible to the application.

Therefore, instead of mirroring the app's mapping:

1. At `MAP` (or at the app's `mmap` of the device), have the **sentry** call
   `mmap(hostFD, offset 0, MAP_SHARED)` itself, creating its own shadow buffer
   with its own driver-allocated pages and `base_index`.
2. Pass **that** sentry address as `cpuvaddr` for `MAP` and every
   `READ`/`WRITE`; store it in the existing `gdsShadows` registry keyed by the
   application's shadow VA.
3. Leave the application's own mapping as-is; its contents are irrelevant.

This removes the `mremap` entirely and, as a bonus, **fixes the multi-buffer
aliasing problem** (risk #1 below): one sentry-side host `mmap` per registered
buffer naturally yields distinct shadow pages, which is exactly what the driver
expects.

Until this is done, `MAP` (and therefore all of `READ`/`WRITE`) fails inside a
sandbox on any host, GDS-capable or not.

## 7. Known gaps & risks (ranked)

1. **Multiple concurrent registered buffers will almost certainly break.**
   *Architectural, not a small bug.* `nvfs_mgroup_mmap()` allocates distinct
   shadow pages per mmap and rejects non-zero `vm_pgoff`. But gVisor's
   `Translate` maps by file offset: two app mmaps of the device, both at offset
   0, yield the same `MappableRange{0,…}` and therefore **alias the same host
   pages**. The registry fixes *cpuvaddr→sentryVA*; it does **not** give each
   buffer its own host mmap. Expect single-buffer to have a chance and
   multi-buffer (e.g. multi-threaded `gdsio`) to collide. **Fixing this needs
   work in the mmap layer: one host mmap per registered buffer, tracked
   per-VA-range instead of per-offset.**

2. **Shadow-page identity may not survive the pin+`mremap`, and the failure
   mode can be a host kernel panic.** `nvfs_mgroup_pin_shadow_pages()` uses
   `BUG_ON` (not error returns) to assert the pinned pages are the driver's
   exact `struct page`s and in the right order. Most likely a mismatch fails
   earlier and benignly (`base_index` lookup → `NULL` → `EINVAL`), but a
   partial match panics the host. **Test on a machine you can afford to crash.**

3. **`file_args` major/minor semantics are inferred.** We map the driver's
   `get_major(inode)`/`get_minor(inode)` to `st_dev`. Reasonable but
   unverified; a mismatch surfaces as `ESTALE`.

4. **No refcounting on shadow mappings.** They're released on `nvfsFD` close or
   re-`MAP`, not per-I/O. An app deregistering a buffer with I/O in flight could
   have the sentry unmap a range the host is mid-DMA into. cuFile shouldn't do
   this; a refcount on `gdsShadowMapping` would harden it.

5. **Mappings persist until `nvfsFD` close, not `cuFileBufDeregister`.** We
   don't hook the shadow-buffer munmap, so register/deregister-heavy workloads
   hold sentry mappings + pins until the device FD closes (bounded by peak
   concurrent registrations). Hooking `RemoveMapping` would allow eager release.

6. **`O_DIRECT` FD re-opened per I/O** — correct but adds an open/close per op;
   cache per (file, mode) if it shows up in profiles.

7. **Async (`sync==0`) is reasoned-through but untested.** The end-fence
   completion path should work (host writes the pinned page; app polls its own
   mapping), but only sync `cuFileRead/Write` has been analysed.

8. **RDMA/batch nvfs builds unsupported.** `NvfsIoctlParamUnion` assumes the
   common non-RDMA, non-batch build; a host with the RDMA-enabled build has a
   different union size and extra ioctls.

## 8. How to validate on a capable host

**Prerequisite: fix the `MAP` blocker in [section 6a](#6a-blocker-shadow-buffer-mirroring-is-a-dead-end) first.**
Until the sentry owns the shadow buffer, `MAP` fails with `EFAULT` in the
sandbox on *every* host, so none of the steps below can pass.

**Backend** (need one): a kernel built with `CONFIG_PCI_P2PDMA` for local
NVMe→GPU DMA, or an RDMA-capable DFS (FSx for Lustre, WekaFS). Note the DFS
route additionally needs an **RDMA NIC** — on the g5 test box `gdscheck -p`
reported `Userspace RDMA: Unsupported` and `rdma devices: Not configured`, so
every backend showed `compat`. The data file must be on an `O_DIRECT`-capable
mount (not `/tmp`).

**Reproducing the current blocker** (works on any box with the GDS stack, no
P2PDMA needed): build a harness that does `cuMemAlloc` + `mmap(/dev/nvidia-fs0)`
+ `NVFS_IOCTL_MAP`. It returns 0 on bare metal and `EFAULT` through `runsc`.
This is the fastest signal that the redesign works.

**Note on cuFile in a sandbox:** cuFile probes `/proc/driver/nvidia-fs/*` and
`/proc/modules`. gVisor serves neither; the former can be bind-mounted
(`-v /proc/driver/nvidia-fs:/proc/driver/nvidia-fs:ro`) which gets cuFile to
open the 16 devices, but `/proc/modules` cannot be (procfs refuses the
mountpoint) and cuFile then stalls. Driving the ioctls from a harness avoids
this entirely; serving those files from nvproxy's `procfs.go` would fix it
properly.

**Start minimal:** single thread, **one** registered buffer, one `cuFileRead` of
a few MB, with `--strace --debug --debug-log=...`. Then walk the boot log in
order; each stage passed is real signal, and the first failure localises the
broken assumption:

1. `REMOVE` ×16 forward, return 0.
2. `MAP` returns **0** (not `EPERM`) — the shadow-page path now works.
3. No `nvproxy: nvidia-fs READ/WRITE on unregistered shadow buffer` warning —
   registry populated and looked up.
4. No `ESTALE` from the host — `nvfsRewriteFileArgs` is correct.
5. No `failed to open data file O_DIRECT` warning.
6. Data actually lands in the GPU buffer (verify contents, not just rc).

Only after single-buffer works, try multi-buffer / multi-threaded — and expect
risk #1.

**Container needs:** explicit `--device /dev/nvidia-fs0..15` (the
nvidia-container-cli does *not* inject them), `NVIDIA_VISIBLE_DEVICES`,
`NVIDIA_DRIVER_CAPABILITIES=...,gpudirect-storage`, and runsc flags including
`--directfs --nvproxy --nvproxy-docker
--nvproxy-allowed-driver-capabilities=all,gpudirect-storage`.

**Optional:** cuFile probes `/proc/driver/nvidia-fs/{devcount,version,modules}`
and silently falls back to compat mode on `ENOENT`. If cuFile won't take the GDS
path, expose these via nvproxy's `procfs.go` (mirroring `ProcDriverNvidiaParams`).

## 9. Relationship to GPUDirect RDMA (`CapRDMA`)

`CapRDMA` (GPU↔NIC) is a different feature from GDS (GPU↔storage) and they are
independent for the local NVMe path. Two points of contact:

- **Host-FD handoff convention.** RDMA is a two-proxy handoff: nvproxy's
  `NV_ESC_EXPORT_TO_DMABUF_FD` wraps the dma-buf host fd in a `dmaBufFDWrapper`
  implementing `vfs.HostFDProvider`, which `rdmaproxy` recovers for
  `ibv_reg_dmabuf_mr`. The gofer's `regularFileFD` now implements
  `vfs.HostFDProvider` too (convention adherence), but GDS uses the richer
  `HostFDForGPUDirectStorage(write)` because it needs read/write selection,
  `O_DIRECT`, and an error return. *Security note:* implementing
  `HostFDProvider` exposes a host FD for every directfs-backed file to any code
  type-asserting it — an explicit tradeoff for convention adherence.
- **GDS-over-RDMA (future).** cuFile on WekaFS/GPFS/some Lustre uses the nvfs
  RDMA path and registers the GPU buffer with the NIC through the *same*
  `ibv_reg_dmabuf_mr` path `CapRDMA` implements, so it should **compose** the
  two capabilities rather than duplicate them.

## 10. Before upstreaming

- Resolve risk #1 (multi-buffer) — shipping a capability that only supports one
  registered buffer is likely a blocker.
- Decide whether `READ`/`WRITE` should fail loudly (`ENOSYS`) until validated,
  so the capability can't be enabled into a silently-broken data path.
- Verify ABI behaviour for each supported driver version (per repo ABI policy).
- Confirm capability gating intent (currently privileged/opt-in, like
  `CapProfiling`/`CapRDMA`).
- This document lives at the repo root for handoff; move it to `g3doc/` or drop
  it from the upstream PR.
- Build/test **only** via `make build` / `make test TARGETS=...` (Bazel in
  Docker). Plain `bazel`/`go test` fail here (missing aarch64 cross-linker for
  the dual VDSO; `go test` chokes on `pkg/sync/*.tmpl.s`). gopls's "missing
  CopyIn/SizeBytes" on `+marshal` structs is a false positive (codegen is
  Bazel-only).
