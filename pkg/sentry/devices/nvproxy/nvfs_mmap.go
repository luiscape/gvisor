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
	"gvisor.dev/gvisor/pkg/context"
	"gvisor.dev/gvisor/pkg/hostarch"
	"gvisor.dev/gvisor/pkg/log"
	"gvisor.dev/gvisor/pkg/safemem"
	"gvisor.dev/gvisor/pkg/sentry/fsutil"
	"gvisor.dev/gvisor/pkg/sentry/memmap"
	"gvisor.dev/gvisor/pkg/sentry/vfs"
)

// ConfigureMMap implements vfs.FileDescriptionImpl.ConfigureMMap.
//
// The cuFile library mmaps /dev/nvidia-fs to obtain the "shadow buffer" used by
// GPUDirect Storage. Backing the application's mapping with an mmap of the host
// device makes the shadow pages (and the host driver's fault handling) real.
func (fd *nvfsFD) ConfigureMMap(ctx context.Context, opts *memmap.MMapOpts) error {
	return vfs.GenericProxyDeviceConfigureMMap(&fd.vfsfd, fd, opts)
}

// Translate implements memmap.Mappable.Translate.
func (fd *nvfsFD) Translate(ctx context.Context, required, optional memmap.MappableRange, at hostarch.AccessType) ([]memmap.Translation, error) {
	return []memmap.Translation{
		{
			Source: optional,
			File:   &fd.memmapFile,
			Offset: optional.Start,
			Perms:  hostarch.AnyAccess,
		},
	}, nil
}

// nvfsFDMemmapFile implements memmap.File by extending fsutil.MmapPreciseFile
// with fallback buffered I/O.
//
// +stateify savable
type nvfsFDMemmapFile struct {
	fsutil.MmapPreciseFile
}

// MapInternal implements memmap.File.MapInternal.
func (mf *nvfsFDMemmapFile) MapInternal(fr memmap.FileRange, at hostarch.AccessType) (safemem.BlockSeq, error) {
	bs, err := mf.MmapPreciseFile.MapInternal(fr, at)
	if err != nil {
		log.Warningf("nvfsFDMemmapFile.MapInternal(%v) failed: %v; falling back to buffered I/O", fr, err)
		return safemem.BlockSeq{}, memmap.BufferedIOFallbackErr{}
	}
	return bs, nil
}

// NOTE: This establishes the host-device-backed shadow-buffer mapping, but two
// pieces remain before GPUDirect Storage works end-to-end, both of which
// require validation against GPU hardware:
//
//  1. cpuvaddr translation. nvidia-fs resolves the shadow-buffer address in
//     NVFS_IOCTL_MAP/READ/WRITE via pin_user_pages_fast(cpuvaddr) in the
//     ioctl-issuing process (see nvfs-mmap.c:nvfs_get_mgroup_from_vaddr).
//     nvproxy forwards from the sentry's host process, so the application VA
//     must be translated to the sentry-host VA of the corresponding host mmap
//     before forwarding. nvfsIoctlSimple/nvfsIoctlReadWrite do not yet do this.
//
//  2. Per-mapping tracking. nvidia-fs requires vm_pgoff == 0 for every mmap and
//     allocates distinct shadow pages (keyed by a random base_index) per call
//     (see nvfs-mmap.c:nvfs_mgroup_mmap). The offset-indexed memmap.File model
//     used here therefore cannot, on its own, distinguish multiple concurrent
//     shadow buffers; a custom registry mapping each application VA range to its
//     host mmap is needed for correct multi-buffer behavior.
