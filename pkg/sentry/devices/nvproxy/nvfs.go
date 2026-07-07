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
	// Back application mmaps of /dev/nvidia-fs with mmaps of the host device,
	// so that the shadow buffer's pages and fault handler are real. Unlike
	// nvidia-uvm, nvidia-fs does not require addr == file offset, so
	// RequireAddrEqualsFileOffset is not set.
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
// no translation, but nvidia-fs also pin_user_pages_fast()es the shadow buffer
// (CPUVAddr) and the end-fence page (EndFenceAddr) in the *calling* process.
// Since nvproxy forwards from the sentry, those application VAs are translated
// to sentry-host VAs backed by the same pages (pin + MapInternal + mremap into
// a fresh reservation, mirroring rmAllocOSDescriptor).
//
// TODO(GDS): the sentry mappings are released once MAP returns, which suffices
// for buffer registration (the host pins the pages itself). The READ/WRITE path
// additionally requires CPUVAddr to stay mapped at the SAME sentry VA, since the
// host stores it as cpu_base_vaddr and re-pins it; that needs a per-mapping
// registry (see nvfsIoctlReadWrite).
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
	shadowAddr, shadowCleanup, err := nvfsPinAndMap(ni, mapArgs.CPUVAddr, shadowLen, true /* write */)
	if err != nil {
		return 0, err
	}
	defer shadowCleanup()
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
	if _, err := param.CopyOut(ni.t, ni.ioctlParamsAddr); err != nil {
		return n, err
	}
	return n, nil
}

// nvfsPinAndMap pins the application memory range [addr, addr+length) and
// mirrors it into a fresh reservation in the sentry's address space, returning
// the sentry virtual address and a cleanup function that unmaps it and unpins
// the application pages. The host nvidia-fs driver pin_user_pages_fast()es the
// returned address in the sentry process.
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
		return 0, nil, err
	}
	// Reserve a range in our address space, then mirror the pinned pages into
	// it. old_size == 0 in MREMAP duplicates the (shared) host mapping rather
	// than moving it, leaving the original MapInternal mapping intact.
	m, _, errno := unix.RawSyscall6(unix.SYS_MMAP, 0 /* addr */, uintptr(length), unix.PROT_NONE, unix.MAP_PRIVATE|unix.MAP_ANONYMOUS, ^uintptr(0) /* fd */, 0 /* offset */)
	if errno != 0 {
		return 0, nil, errno
	}
	cu.Add(func() { unix.RawSyscall(unix.SYS_MUNMAP, m, uintptr(length), 0) })
	sentryAddr := m
	for _, pr := range prs {
		ims, err := pr.File.MapInternal(memmap.FileRange{pr.Offset, pr.Offset + uint64(pr.Source.Length())}, at)
		if err != nil {
			return 0, nil, err
		}
		for !ims.IsEmpty() {
			im := ims.Head()
			if _, _, errno := unix.RawSyscall6(unix.SYS_MREMAP, im.Addr(), 0 /* old_size */, uintptr(im.Len()), linux.MREMAP_MAYMOVE|linux.MREMAP_FIXED, sentryAddr, 0); errno != 0 {
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
type hostFDForGDSer interface {
	HostFDForGPUDirectStorage(write bool) (int32, error)
}

// nvfsIoctlReadWrite handles NVFS_IOCTL_READ and NVFS_IOCTL_WRITE. nvidia-fs
// performs direct I/O (and NVMe-to-GPU DMA) on the file descriptor carried in
// NvfsIoctlIoargs.FD via fget() in the calling (sentry) process, so the
// application's data-file FD must be translated to its host FD before
// forwarding. The host FD is only reachable when directfs donated it to the
// sentry, which CapGPUDirectStorage requires.
//
// Confirmed against nvfs-core.c:nvfs_io_init(), three more translations are
// required before READ/WRITE can succeed (none are exercisable on a
// compat-mode host, which never issues these ioctls):
//
//   - CPUVAddr: nvfs_get_mgroup_from_vaddr() pins it via pin_user_pages_fast()
//     in current->mm and requires it to equal the cpu_base_vaddr recorded at
//     MAP. It must be translated to a STABLE sentry VA backing the shadow
//     buffer (the same value used at MAP), via a per-fd registry of shadow
//     mappings (pin app range + MapInternal + mremap into a reserved range).
//   - FileArgs.{Inum,MajDev,MinDev,Generation}: validated against the HOST
//     inode (inum == inode->i_ino, etc.). cuFile fills these from the sandbox
//     (gVisor) stat, so nvproxy must fstat the host FD and overwrite them with
//     host values, else nvfs returns ESTALE.
//   - O_DIRECT: nvfs_io_init() rejects the data-file FD unless O_DIRECT is set;
//     the directfs host FD may not have it, so it must be reconciled (e.g. a
//     per-op O_DIRECT host FD).
func nvfsIoctlReadWrite(ni *nvfsIoctlState) (uintptr, error) {
	var ioctlParams nvgpu.NvfsIoctlIoargs
	if _, err := ioctlParams.CopyIn(ni.t, ni.ioctlParamsAddr); err != nil {
		return 0, err
	}

	write := ni.cmd == nvgpu.NVFS_IOCTL_WRITE

	// Translate the application's data-file FD to its host FD.
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
	hostFD, err := hostFDer.HostFDForGPUDirectStorage(write)
	if err != nil {
		return 0, err
	}

	origFD := ioctlParams.FD
	ioctlParams.FD = hostFD
	n, err := nvfsIoctlInvoke(ni, &ioctlParams)
	ioctlParams.FD = origFD
	if err != nil {
		return n, err
	}
	if _, err := ioctlParams.CopyOut(ni.t, ni.ioctlParamsAddr); err != nil {
		return n, err
	}
	return n, nil
}
