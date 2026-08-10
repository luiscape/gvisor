// Copyright 2026 The gVisor Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package nvproxy

import (
	"fmt"

	"golang.org/x/sys/unix"
	"gvisor.dev/gvisor/pkg/abi/linux"
	"gvisor.dev/gvisor/pkg/abi/nvgpu"
	"gvisor.dev/gvisor/pkg/cleanup"
	"gvisor.dev/gvisor/pkg/context"
	"gvisor.dev/gvisor/pkg/errors/linuxerr"
	"gvisor.dev/gvisor/pkg/fdnotifier"
	"gvisor.dev/gvisor/pkg/hostarch"
	"gvisor.dev/gvisor/pkg/log"
	"gvisor.dev/gvisor/pkg/sentry/arch"
	"gvisor.dev/gvisor/pkg/sentry/kernel"
	"gvisor.dev/gvisor/pkg/sentry/kernel/auth"
	"gvisor.dev/gvisor/pkg/sentry/memmap"
	"gvisor.dev/gvisor/pkg/sentry/mm"
	"gvisor.dev/gvisor/pkg/sentry/vfs"
	"gvisor.dev/gvisor/pkg/usermem"
	"gvisor.dev/gvisor/pkg/waiter"
)

// nvfsDevice implements vfs.Device for /dev/nvidia-fs<minor>.
//
// The nvidia-fs driver exposes nvgpu.NVIDIA_FS_DEV_COUNT device nodes
// (nvidia-fs0 .. nvidia-fs15); cuFile opens all of them.
//
// +stateify savable
type nvfsDevice struct {
	nvp   *nvproxy
	minor uint32
}

func (dev *nvfsDevice) basename() string {
	return fmt.Sprintf("nvidia-fs%d", dev.minor)
}

// Open implements vfs.Device.Open.
func (dev *nvfsDevice) Open(ctx context.Context, mnt *vfs.Mount, vfsd *vfs.Dentry, opts vfs.OpenOptions) (*vfs.FileDescription, error) {
	fd := &nvfsFD{
		dev: dev,
	}
	var err error
	fd.hostFD, fd.containerName, err = openHostDevFile(ctx, dev.basename(), dev.nvp.useDevGofer, opts.Flags)
	if err != nil {
		return nil, err
	}
	if err := fd.vfsfd.Init(fd, opts.Flags, auth.CredentialsFromContext(ctx), mnt, vfsd, &vfs.FileDescriptionOptions{
		UseDentryMetadata: true,
	}); err != nil {
		unix.Close(int(fd.hostFD))
		return nil, err
	}
	if err := fdnotifier.AddFD(fd.hostFD, &fd.queue); err != nil {
		unix.Close(int(fd.hostFD))
		return nil, err
	}
	// Back application mmaps of /dev/nvidia-fs with mmaps of the host device so
	// the application sees a real shadow-buffer mapping. (The buffer used for the
	// ioctls is the sentry's own; see nvfsIoctlMap.) Unlike nvidia-uvm, nvidia-fs
	// does not require addr == file offset, so RequireAddrEqualsFileOffset is not
	// set.
	fd.memmapFile.SetFD(int(fd.hostFD))
	return &fd.vfsfd, nil
}

// nvfsFD implements vfs.FileDescriptionImpl for /dev/nvidia-fs.
//
// +stateify savable
type nvfsFD struct {
	vfsfd vfs.FileDescription
	vfs.FileDescriptionDefaultImpl
	vfs.DentryMetadataFileDescriptionImpl
	vfs.NoLockFD
	memmap.MappableNoTrackMappings

	dev           *nvfsDevice
	containerName string
	hostFD        int32
	memmapFile    nvfsFDMemmapFile

	queue waiter.Queue
}

// Release implements vfs.FileDescriptionImpl.Release.
func (fd *nvfsFD) Release(context.Context) {
	// Drop any GPUDirect Storage shadow buffers registered through this FD.
	fd.dev.nvp.releaseGDSShadows(fd)
	fdnotifier.RemoveFD(fd.hostFD)
	fd.queue.Notify(waiter.EventHUp)
	fd.memmapFile.MappableRelease() // eventually closes fd.hostFD
}

// EventRegister implements waiter.Waitable.EventRegister.
func (fd *nvfsFD) EventRegister(e *waiter.Entry) error {
	fd.queue.EventRegister(e)
	if err := fdnotifier.UpdateFD(fd.hostFD); err != nil {
		fd.queue.EventUnregister(e)
		return err
	}
	return nil
}

// EventUnregister implements waiter.Waitable.EventUnregister.
func (fd *nvfsFD) EventUnregister(e *waiter.Entry) {
	fd.queue.EventUnregister(e)
	if err := fdnotifier.UpdateFD(fd.hostFD); err != nil {
		panic(fmt.Sprint("UpdateFD:", err))
	}
}

// Readiness implements waiter.Waitable.Readiness.
func (fd *nvfsFD) Readiness(mask waiter.EventMask) waiter.EventMask {
	return fdnotifier.NonBlockingPoll(fd.hostFD, mask)
}

// Epollable implements vfs.FileDescriptionImpl.Epollable.
func (fd *nvfsFD) Epollable() bool {
	return true
}

// Ioctl implements vfs.FileDescriptionImpl.Ioctl.
func (fd *nvfsFD) Ioctl(ctx context.Context, uio usermem.IO, sysno uintptr, args arch.SyscallArguments) (uintptr, error) {
	cmd := args[1].Uint()
	argPtr := args[2].Pointer()

	t := kernel.TaskFromContext(ctx)
	if t == nil {
		panic("Ioctl should be called from a task context")
	}

	if ctx.IsLogging(log.Debug) {
		ctx.Debugf("nvproxy: nvidia-fs ioctl %d = %#x", cmd, cmd)
	}

	ni := nvfsIoctlState{
		fd:              fd,
		ctx:             ctx,
		t:               t,
		cmd:             cmd,
		ioctlParamsAddr: argPtr,
	}
	result, err := fd.dev.nvp.abi.nvfsIoctl[cmd].handle(&ni)
	if err != nil {
		if handleErr, ok := err.(*errHandler); ok {
			ctx.Warningf("nvproxy: %v for nvidia-fs ioctl %d = %#x", handleErr, cmd, cmd)
			return 0, linuxerr.EINVAL
		}
	}
	return result, err
}

// IsNvidiaDeviceFD implements NvidiaDeviceFD.IsNvidiaDeviceFD.
func (fd *nvfsFD) IsNvidiaDeviceFD() {}

// nvfsIoctlState holds the state of a call to nvfsFD.Ioctl().
type nvfsIoctlState struct {
	fd              *nvfsFD
	ctx             context.Context
	t               *kernel.Task
	cmd             uint32
	ioctlParamsAddr hostarch.Addr
}

// nvfsIoctlRemove handles NVFS_IOCTL_REMOVE. In the driver (nvfs_remove()) this
// is a no-op that touches no parameter fields, so it needs no translation; it
// is forwarded as a union-sized buffer like any other simple command. cuFile
// issues it on every /dev/nvidia-fs<N> node at startup.
func nvfsIoctlRemove(ni *nvfsIoctlState) (uintptr, error) {
	var param nvgpu.NvfsIoctlParamUnion
	if _, err := param.CopyIn(ni.t, ni.ioctlParamsAddr); err != nil {
		return 0, err
	}
	// nvfs_remove() does not copy_to_user, so no copy-out is needed.
	return nvfsIoctlInvoke(ni, &param)
}

// nvfsIoctlMap handles NVFS_IOCTL_MAP (cuFileBufRegister). The GPU buffer
// (NvfsIoctlMap.GPUVAddr) is resolved by the GPU driver via nvidia_p2p and needs
// no translation, but nvidia-fs pin_user_pages_fast()es the shadow buffer
// (CPUVAddr) and the end-fence page (EndFenceAddr) in the *calling* process,
// which for nvproxy is the sentry. The two are handled differently:
//
//   - CPUVAddr (shadow buffer): the sentry creates and owns its OWN shadow
//     buffer by mmap()ing the host /dev/nvidia-fs FD, and passes that address.
//     The application's own mapping is ignored. This is required because
//     nvidia-fs allocates driver-private pages for each mmap and then asserts
//     (via BUG_ON) that the pinned pages are exactly those; its vm_ops also
//     reject mremap/VMA duplication, so the sentry cannot mirror the
//     application's mapping. This is sound because the shadow buffer is a
//     kernel-side handle, not a data buffer: nvfs_get_dma() substitutes GPU BAR
//     addresses during DMA, so payload never lands in the shadow pages and the
//     application never reads them. It also gives each registered buffer its own
//     distinct shadow pages, which is what the driver expects.
//   - EndFenceAddr: ordinary anonymous application memory, so the usual
//     pin + MapInternal + mremap translation works (see nvfsPinAndMap).
//
// The shadow mapping is retained past MAP in a per-nvproxy registry: the host
// records cpu_base_vaddr == the address passed here and re-pins it in the
// sentry on every READ/WRITE (nvfs-core.c:nvfs_get_mgroup_from_vaddr), so it
// must stay mapped at the SAME sentry VA. The end-fence page does not need to
// persist: the host holds its own pin_user_pages_fast() reference and kmaps the
// page directly, so its sentry mapping is released once MAP returns.
func nvfsIoctlMap(ni *nvfsIoctlState) (uintptr, error) {
	var param nvgpu.NvfsIoctlParamUnion
	if _, err := param.CopyIn(ni.t, ni.ioctlParamsAddr); err != nil {
		return 0, err
	}
	var mapArgs nvgpu.NvfsIoctlMap
	mapArgs.UnmarshalBytes(param.Data[:])

	shadowLen := uint64(mapArgs.SBufBlock) * nvgpu.NVFS_BLOCK_SIZE
	if shadowLen == 0 {
		return 0, linuxerr.EINVAL
	}
	appCPUVAddr := mapArgs.CPUVAddr
	shadowAddr, shadowCleanup, err := nvfsMapOwnShadowBuffer(ni, shadowLen)
	if err != nil {
		return 0, err
	}
	// On any failure the shadow mapping is released; on success its ownership is
	// transferred to the registry (persisted == true).
	persisted := false
	defer func() {
		if !persisted {
			shadowCleanup()
		}
	}()
	mapArgs.CPUVAddr = uint64(shadowAddr)

	if mapArgs.EndFenceAddr != 0 {
		fenceAddr, fenceCleanup, err := nvfsPinAndMap(ni, mapArgs.EndFenceAddr, hostarch.PageSize, true /* write */)
		if err != nil {
			return 0, err
		}
		defer fenceCleanup()
		mapArgs.EndFenceAddr = uint64(fenceAddr)
	}

	mapArgs.MarshalBytes(param.Data[:])
	n, err := nvfsIoctlInvoke(ni, &param)
	if err != nil {
		return n, err
	}
	// nvfs_map() does not write the parameter struct back, so there is no
	// copy-out; leaving the application's buffer untouched also keeps the sentry
	// VAs substituted above from ever being exposed.
	ni.fd.dev.nvp.storeGDSShadow(ni.fd, ni.t.MemoryManager(), appCPUVAddr, shadowAddr, shadowCleanup)
	persisted = true
	return n, nil
}

// nvfsMapOwnShadowBuffer creates a sentry-owned GPUDirect Storage shadow buffer
// by mmap()ing the host /dev/nvidia-fs FD directly, returning the sentry
// virtual address and a cleanup function that unmaps it.
//
// nvidia-fs allocates driver-private pages per mmap (nvfs-mmap.c:
// nvfs_mgroup_mmap_internal), tags them with an encoded page->index, and later
// asserts via BUG_ON that the pages pinned at NVFS_IOCTL_MAP are exactly those.
// The sentry therefore cannot mirror the application's mapping of the device
// (its vm_ops also reject mremap and VMA duplication outright); it must own a
// shadow buffer of its own. The offset must be 0: nvfs_mgroup_mmap() returns
// -EIO for any non-zero vm_pgoff.
func nvfsMapOwnShadowBuffer(ni *nvfsIoctlState, length uint64) (uintptr, func(), error) {
	m, _, errno := unix.RawSyscall6(unix.SYS_MMAP, 0 /* addr */, uintptr(length), unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED, uintptr(ni.fd.hostFD), 0 /* offset */)
	if errno != 0 {
		ni.ctx.Warningf("nvproxy: nvidia-fs failed to mmap host shadow buffer (len=%d): %v", length, errno)
		return 0, nil, errno
	}
	return m, func() { unix.RawSyscall(unix.SYS_MUNMAP, m, uintptr(length), 0) }, nil
}

// nvfsPinAndMap pins the application memory range [addr, addr+length) and
// mirrors it into a fresh reservation in the sentry's address space, returning
// the sentry virtual address and a cleanup function that unmaps it and unpins
// the application pages. The host nvidia-fs driver pin_user_pages_fast()es the
// returned address in the sentry process.
//
// This works only for ordinary (anonymous) application memory, whose pages are
// backed by the sentry's MemoryFile and can be duplicated; it is used for the
// end-fence page. It must NOT be used for the shadow buffer -- see
// nvfsMapOwnShadowBuffer.
func nvfsPinAndMap(ni *nvfsIoctlState, addr uint64, length uint64, write bool) (uintptr, func(), error) {
	appAR, ok := hostarch.Addr(addr).ToRange(length)
	if !ok {
		return 0, nil, linuxerr.EINVAL
	}
	at := hostarch.Read
	if write {
		at.Write = true
	}
	prs, err := ni.t.MemoryManager().Pin(ni.ctx, appAR, at, false /* ignorePermissions */)
	cu := cleanup.Make(func() { mm.Unpin(prs) })
	defer cu.Clean()
	if err != nil {
		ni.ctx.Warningf("nvproxy: nvidia-fs pin-and-map: Pin(%#x, %d) failed: %v", addr, length, err)
		return 0, nil, err
	}
	// Reserve a range in our address space, then mirror the pinned pages into
	// it. old_size == 0 in MREMAP duplicates the (shared) host mapping rather
	// than moving it, leaving the original MapInternal mapping intact.
	m, _, errno := unix.RawSyscall6(unix.SYS_MMAP, 0 /* addr */, uintptr(length), unix.PROT_NONE, unix.MAP_PRIVATE|unix.MAP_ANONYMOUS, ^uintptr(0) /* fd */, 0 /* offset */)
	if errno != 0 {
		ni.ctx.Warningf("nvproxy: nvidia-fs pin-and-map: reservation mmap(%d) failed: %v", length, errno)
		return 0, nil, errno
	}
	cu.Add(func() { unix.RawSyscall(unix.SYS_MUNMAP, m, uintptr(length), 0) })
	sentryAddr := m
	for _, pr := range prs {
		ims, err := pr.File.MapInternal(memmap.FileRange{pr.Offset, pr.Offset + uint64(pr.Source.Length())}, at)
		if err != nil {
			ni.ctx.Warningf("nvproxy: nvidia-fs pin-and-map: MapInternal(off=%#x len=%d) failed: %v", pr.Offset, pr.Source.Length(), err)
			return 0, nil, err
		}
		for !ims.IsEmpty() {
			im := ims.Head()
			if _, _, errno := unix.RawSyscall6(unix.SYS_MREMAP, im.Addr(), 0 /* old_size */, uintptr(im.Len()), linux.MREMAP_MAYMOVE|linux.MREMAP_FIXED, sentryAddr, 0); errno != 0 {
				ni.ctx.Warningf("nvproxy: nvidia-fs pin-and-map: mremap(src=%#x len=%d dst=%#x) failed: %v", im.Addr(), im.Len(), sentryAddr, errno)
				return 0, nil, errno
			}
			sentryAddr += uintptr(im.Len())
			ims = ims.Tail()
		}
	}
	return m, cu.Release(), nil
}

// hostFDForGDSer is implemented by vfs.FileDescriptionImpls that can expose a
// host file descriptor for use as the data-file FD in nvidia-fs (GPUDirect
// Storage) ioctls. The gofer's directfs regular-file FD implements it.
//
// This mirrors how GPUDirect RDMA hands a host FD between proxies (nvproxy's
// dmaBufFDWrapper implements vfs.HostFDProvider, recovered by rdmaproxy). The
// gofer regular-file FD implements the vfs.HostFDProvider convention too, but
// GDS uses this richer interface because it needs a superset of HostFD() int:
// write-vs-read selection, O_DIRECT reconciliation, and an error return. See
// regularFileFD.HostFDForGPUDirectStorage.
type hostFDForGDSer interface {
	HostFDForGPUDirectStorage(write bool) (int32, error)
}

// gdsShadowKey identifies a registered GPUDirect Storage shadow buffer by the
// registering application's address space and the buffer's application VA. This
// mirrors how the host driver resolves a shadow buffer: pin_user_pages_fast()
// of cpuvaddr in current->mm (here, the app's MemoryManager).
//
// The MemoryManager pointer is used only as an opaque key; no reference is
// held. This is safe because a registered buffer keeps its process alive, and
// the process's death closes its nvfsFDs (releasing the registrations, see
// releaseGDSShadows), so a live entry's MemoryManager pointer cannot be reused
// by a different address space.
type gdsShadowKey struct {
	mm   *mm.MemoryManager
	addr uint64
}

// gdsShadowMapping is a persistent sentry-owned shadow buffer (a host mmap of
// /dev/nvidia-fs), established at NVFS_IOCTL_MAP and reused by every
// NVFS_IOCTL_READ/WRITE against that buffer until the owning nvfsFD is released.
type gdsShadowMapping struct {
	// owner is the nvfsFD whose MAP established this mapping; it owns cleanup.
	owner *nvfsFD
	// sentryAddr is the stable sentry VA passed to the host as cpuvaddr.
	sentryAddr uintptr
	// release unmaps the sentry-owned shadow buffer.
	release func()
}

// storeGDSShadow records a shadow-buffer mapping, taking ownership of release.
// A pre-existing registration at the same (mm, addr) is released first (the
// application re-registered a buffer at the same VA).
func (nvp *nvproxy) storeGDSShadow(owner *nvfsFD, appMM *mm.MemoryManager, addr uint64, sentryAddr uintptr, release func()) {
	key := gdsShadowKey{mm: appMM, addr: addr}
	nvp.gdsShadowsMu.Lock()
	defer nvp.gdsShadowsMu.Unlock()
	if old, ok := nvp.gdsShadows[key]; ok {
		old.release()
	}
	nvp.gdsShadows[key] = &gdsShadowMapping{
		owner:      owner,
		sentryAddr: sentryAddr,
		release:    release,
	}
}

// lookupGDSShadow returns the stable sentry VA registered for the shadow buffer
// at (appMM, addr), or ok == false if none is registered.
func (nvp *nvproxy) lookupGDSShadow(appMM *mm.MemoryManager, addr uint64) (uintptr, bool) {
	nvp.gdsShadowsMu.Lock()
	defer nvp.gdsShadowsMu.Unlock()
	m, ok := nvp.gdsShadows[gdsShadowKey{mm: appMM, addr: addr}]
	if !ok {
		return 0, false
	}
	return m.sentryAddr, true
}

// releaseGDSShadows releases every shadow mapping owned by fd. Called from
// nvfsFD.Release.
func (nvp *nvproxy) releaseGDSShadows(fd *nvfsFD) {
	nvp.gdsShadowsMu.Lock()
	defer nvp.gdsShadowsMu.Unlock()
	for key, m := range nvp.gdsShadows {
		if m.owner == fd {
			m.release()
			delete(nvp.gdsShadows, key)
		}
	}
}

// nvfsIoctlReadWrite handles NVFS_IOCTL_READ and NVFS_IOCTL_WRITE, the actual
// NVMe->GPU (READ) and GPU->NVMe (WRITE) DMA path. nvfs_io_init() runs in the
// ioctl-issuing (sentry) process and dereferences three request fields that
// are all sandbox-relative and must be rewritten to host-relative before
// forwarding (verified against nvfs-core.c):
//
//   - CPUVAddr: re-pinned via pin_user_pages_fast() in the sentry and required
//     to equal the cpu_base_vaddr recorded at MAP
//     (nvfs-mmap.c:nvfs_get_mgroup_from_vaddr). Translated to the persistent
//     sentry VA established for this shadow buffer at NVFS_IOCTL_MAP.
//   - FD: the data-file FD, fget()'d and subjected to direct I/O. Translated to
//     the gofer's donated host FD (directfs, which CapGPUDirectStorage
//     requires), then re-opened O_DIRECT with a matching access mode
//     (nvfs_io_init() rejects the file unless O_DIRECT and FMODE_READ/WRITE are
//     set).
//   - FileArgs.{Inum,MajDev,MinDev,Generation}: validated against the HOST
//     inode; rewritten from an fstat of the host FD, else nvfs returns ESTALE.
//
// The guest-observable inputs are restored before copy-out; only the
// driver-written IoctlReturn is surfaced back to the application.
func nvfsIoctlReadWrite(ni *nvfsIoctlState) (uintptr, error) {
	var ioctlParams nvgpu.NvfsIoctlIoargs
	if _, err := ioctlParams.CopyIn(ni.t, ni.ioctlParamsAddr); err != nil {
		return 0, err
	}
	write := ni.cmd == nvgpu.NVFS_IOCTL_WRITE

	// Translate the shadow-buffer VA to the persistent sentry mapping recorded
	// at MAP. Without a matching registration the host would fault or reject the
	// address, so fail early.
	sentryAddr, ok := ni.fd.dev.nvp.lookupGDSShadow(ni.t.MemoryManager(), ioctlParams.CPUVAddr)
	if !ok {
		op := "READ"
		if write {
			op = "WRITE"
		}
		ni.ctx.Warningf("nvproxy: nvidia-fs %s on unregistered shadow buffer %#x", op, ioctlParams.CPUVAddr)
		return 0, linuxerr.EINVAL
	}

	// Translate the application's data-file FD to a host FD opened O_DIRECT.
	dataFile, _ := ni.t.FDTable().Get(ioctlParams.FD)
	if dataFile == nil {
		return 0, linuxerr.EINVAL
	}
	defer dataFile.DecRef(ni.ctx)
	hostFDer, ok := dataFile.Impl().(hostFDForGDSer)
	if !ok {
		ni.ctx.Warningf("nvproxy: nvidia-fs I/O on FD %d that is not backed by a host FD (directfs is required for GPUDirect Storage)", ioctlParams.FD)
		return 0, linuxerr.EINVAL
	}
	baseHostFD, err := hostFDer.HostFDForGPUDirectStorage(write)
	if err != nil {
		return 0, err
	}
	directFD, err := nvfsOpenDirectHostFD(baseHostFD, write)
	if err != nil {
		ni.ctx.Warningf("nvproxy: nvidia-fs failed to open data file O_DIRECT (GPUDirect Storage requires O_DIRECT-capable storage): %v", err)
		return 0, err
	}
	defer unix.Close(directFD)

	// Rewrite the file-identity fields to the host inode.
	if err := nvfsRewriteFileArgs(&ioctlParams.FileArgs, directFD); err != nil {
		return 0, err
	}

	origFD := ioctlParams.FD
	origCPUVAddr := ioctlParams.CPUVAddr
	origFileArgs := ioctlParams.FileArgs
	ioctlParams.FD = int32(directFD)
	ioctlParams.CPUVAddr = uint64(sentryAddr)
	n, err := nvfsIoctlInvoke(ni, &ioctlParams)
	// Restore guest-observable inputs; keep driver outputs (IoctlReturn).
	ioctlParams.FD = origFD
	ioctlParams.CPUVAddr = origCPUVAddr
	ioctlParams.FileArgs = origFileArgs
	if err != nil {
		return n, err
	}
	if _, err := ioctlParams.CopyOut(ni.t, ni.ioctlParamsAddr); err != nil {
		return n, err
	}
	return n, nil
}

// nvfsOpenDirectHostFD re-opens the data file identified by hostFD with O_DIRECT
// and an access mode matching the operation. nvfs_io_init() rejects the data
// file unless O_DIRECT is set and FMODE_READ/FMODE_WRITE match the op; the host
// FD donated by the gofer carries neither guarantee, so a fresh open of
// /proc/self/fd/<hostFD> (which resolves to the same host inode) is used. The
// caller owns the returned FD and must close it.
func nvfsOpenDirectHostFD(hostFD int32, write bool) (int, error) {
	flags := unix.O_DIRECT | unix.O_CLOEXEC
	if write {
		flags |= unix.O_WRONLY
	} else {
		flags |= unix.O_RDONLY
	}
	return unix.Open(fmt.Sprintf("/proc/self/fd/%d", hostFD), flags, 0)
}

// nvfsRewriteFileArgs overwrites the file-identity fields of file_args with the
// HOST inode's values. cuFile fills these from the sandbox (gVisor) stat, but
// nvfs_io_init() validates them against the host inode (inum == i_ino, majdev/
// mindev == the device the inode resides on) and returns ESTALE on mismatch.
// Generation is zeroed because nvfs only checks it when non-zero and it is not
// recoverable via fstat. DevPtrOff (the GPU-buffer offset) is left untouched.
func nvfsRewriteFileArgs(fa *nvgpu.NvfsFileArgs, hostFD int) error {
	var st unix.Stat_t
	if err := unix.Fstat(hostFD, &st); err != nil {
		return err
	}
	fa.Inum = st.Ino
	fa.MajDev = unix.Major(st.Dev)
	fa.MinDev = unix.Minor(st.Dev)
	fa.Generation = 0
	return nil
}
