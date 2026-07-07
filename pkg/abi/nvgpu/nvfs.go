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

package nvgpu

import (
	"structs"
)

// nvidia-fs is the GPUDirect Storage (GDS) kernel driver, exposing
// /dev/nvidia-fs. The cuFile library (libcufile.so) drives GDS through ioctls
// on this device. These definitions come from
// github.com/NVIDIA/gds-nvidia-fs:src/nvfs-core.h.

// NVFS_BLOCK_SIZE is nvidia-fs's I/O block size (the unit of
// nvfs_ioctl_map_s.sbuf_block). From github.com/NVIDIA/gds-nvidia-fs.
const NVFS_BLOCK_SIZE = 4096

// NVIDIA_FS_DEV_COUNT is the number of /dev/nvidia-fs<N> character device nodes
// the nvidia-fs driver creates (nvidia-fs0 .. nvidia-fs15). The cuFile library
// opens all of them. This matches /proc/driver/nvidia-fs/devcount.
const NVIDIA_FS_DEV_COUNT = 16

// nvidia-fs ioctl commands.
//
// These are the full ioctl command values, encoded as
// _IOW(NVFS_MAGIC, nr, int), where NVFS_MAGIC is 't' (0x74) and the size field
// is sizeof(int) == 4 regardless of the actual parameter struct. On the
// asm-generic ioctl encoding used by amd64 and arm64:
//
//	_IOC(dir, type, nr, size) = (dir<<30) | (size<<16) | (type<<8) | nr
//	_IOW(0x74, nr, int)       = (1<<30) | (4<<16) | (0x74<<8) | nr
//	                          = 0x40047400 | nr
const (
	NVFS_IOCTL_REMOVE = 0x40047401 // _IOW('t', 1, int)
	NVFS_IOCTL_READ   = 0x40047402 // _IOW('t', 2, int)
	NVFS_IOCTL_MAP    = 0x40047403 // _IOW('t', 3, int)
	NVFS_IOCTL_WRITE  = 0x40047404 // _IOW('t', 4, int)
)

// Bits of NvfsIoctlIoargs.Flags, which corresponds to the bitfield
// "sync:1, hipri:1, allowreads:1, use_rkeys:1, optype:3, reserved:1" in
// nvfs_ioctl_ioargs. On little-endian, the first-declared bitfield occupies the
// least-significant bits.
const (
	NVFS_IOARGS_FLAG_SYNC       = 1 << 0
	NVFS_IOARGS_FLAG_HIPRI      = 1 << 1
	NVFS_IOARGS_FLAG_ALLOWREADS = 1 << 2
	NVFS_IOARGS_FLAG_USE_RKEYS  = 1 << 3

	// NVFS_IOARGS_OPTYPE_SHIFT and NVFS_IOARGS_OPTYPE_MASK select the optype
	// field (bits 4-6) within NvfsIoctlIoargs.Flags.
	NVFS_IOARGS_OPTYPE_SHIFT = 4
	NVFS_IOARGS_OPTYPE_MASK  = 0x7 << NVFS_IOARGS_OPTYPE_SHIFT
)

// optype values, stored in NvfsIoctlIoargs.Flags at NVFS_IOARGS_OPTYPE_SHIFT.
const (
	NVFS_IO_READ  = 0
	NVFS_IO_WRITE = 1
)

// NvfsIoctlMap is nvfs_ioctl_map_s, the parameter to NVFS_IOCTL_MAP. It
// registers a GPU buffer (and its shadow CPU mapping) with nvidia-fs.
//
// The C struct is __attribute__((packed, aligned(8))); all members are
// naturally aligned here, so HostLayout reproduces the same 48-byte layout.
//
// +marshal
type NvfsIoctlMap struct {
	_              structs.HostLayout
	Size           int64    // offset 0:  GPU buffer size
	PDevInfo       uint64   // offset 8:  PCI domain/bus/device/func info
	CPUVAddr       uint64   // offset 16: shadow buffer address
	GPUVAddr       uint64   // offset 24: GPU buffer address
	EndFenceAddr   uint64   // offset 32: end fence address
	SBufBlock      uint32   // offset 40: number of 4K blocks
	IsBounceBuffer uint16   // offset 44: bounce buffer
	Pad0           [2]uint8 // offset 46: padding
}

// NvfsFileArgs is nvfs_file_args, the optional file-identity sub-struct
// embedded in NvfsIoctlIoargs.
//
// The C struct is __attribute__((packed, aligned(8))), so devptroff (a u64) is
// placed at the unaligned offset 20. Go would naturally align a uint64 to
// offset 24, so DevPtrOff is represented as a byte array to match the kernel's
// packed layout exactly. Use the host's native byte order to interpret it.
//
// +marshal
type NvfsFileArgs struct {
	_          structs.HostLayout
	Inum       uint64   // offset 0:  inode number
	Generation uint32   // offset 8:  inode generation
	MajDev     uint32   // offset 12: device major
	MinDev     uint32   // offset 16: device minor
	DevPtrOff  [8]uint8 // offset 20: device buffer offset (packed u64)
	Pad0       [4]uint8 // offset 28: trailing padding from aligned(8)
}

// NvfsIoctlIoargs is nvfs_ioctl_ioargs, the parameter to NVFS_IOCTL_READ and
// NVFS_IOCTL_WRITE. FD is the file descriptor that nvidia-fs performs the
// (direct) I/O on; nvproxy must translate it from a sentry FD to a host FD.
//
// The C struct is __attribute__((packed, aligned(8))). With FileArgs occupying
// a fixed 32 bytes at offset 40, HostLayout reproduces the same 80-byte layout.
//
// +marshal
type NvfsIoctlIoargs struct {
	_             structs.HostLayout
	CPUVAddr      uint64       // offset 0:  shadow buffer VA
	Offset        int64        // offset 8:  file offset (loff_t)
	Size          uint64       // offset 16: read/write length
	EndFenceValue uint64       // offset 24: end fence value for DMA completion
	IoctlReturn   int64        // offset 32: ioctl return
	FileArgs      NvfsFileArgs // offset 40: optional file identity (32 bytes)
	FD            int32        // offset 72: file descriptor
	Flags         uint8        // offset 76: sync/hipri/allowreads/use_rkeys/optype/reserved
	Pad0          [3]uint8     // offset 77: padding
}

// NvfsIoctlParamUnion is nvfs_ioctl_param_union. The kernel's nvfs_ioctl()
// copies sizeof(this union) from userspace for every nvidia-fs command (the
// ioctl request encodes only sizeof(int)), so a proxied parameter buffer must
// be at least this large regardless of the specific command. For the common
// (non-RDMA, non-batch) driver build the largest member is NvfsIoctlIoargs.
//
// +marshal
type NvfsIoctlParamUnion struct {
	_    structs.HostLayout
	Data [80]uint8 // == SizeofNvfsIoctlIoargs
}

// Sizes of nvidia-fs ioctl parameter structs. nvidia-fs encodes sizeof(int) in
// the ioctl command rather than the parameter size, so the host driver copies a
// fixed-size struct based on its own definition; these must match the kernel's
// sizeof exactly.
var (
	SizeofNvfsIoctlMap    = uint32((*NvfsIoctlMap)(nil).SizeBytes())
	SizeofNvfsIoctlIoargs = uint32((*NvfsIoctlIoargs)(nil).SizeBytes())
)
