// Copyright 2024 The gVisor Authors.
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
	"gvisor.dev/gvisor/pkg/abi/nvgpu"
	"gvisor.dev/gvisor/pkg/log"
	"gvisor.dev/gvisor/pkg/sync"
)

// allocTracker tracks GPU memory allocations across all processes in the
// sandbox by observing UVM ioctls as they flow through nvproxy. It maintains
// a map of GPU virtual addresses to allocation metadata so that the
// checkpoint/restore path knows which GPU memory regions to stage (copy to
// CPU) during checkpoint and restore (copy back to GPU) after restore.
//
// Both the cuMemAlloc path (classic allocator) and the cuMemCreate/cuMemMap
// path (VMM / expandable_segments) ultimately issue UVM_MAP_EXTERNAL_ALLOCATION,
// which is the single tracking point for recording allocations.
//
// The tracker is serialized as part of the nvproxy state during checkpoint,
// so the allocation map survives across checkpoint/restore boundaries.
//
// NOTE: To use this tracker, a field `tracker allocTracker` must be added to
// the nvproxy struct in nvproxy.go, and Init() must be called during Register().
//
// +stateify savable
type allocTracker struct {
	// mu protects all mutable fields below.
	mu sync.Mutex `state:"nosave"`

	// allocs maps GPU device virtual addresses to allocation metadata.
	// The key is the base virtual address of each allocation.
	allocs map[uint64]gpuAlloc

	// totalBytes is the sum of all tracked allocation sizes.
	totalBytes uint64
}

// gpuAlloc describes a single GPU memory allocation as observed via the
// UVM_MAP_EXTERNAL_ALLOCATION ioctl.
//
// +stateify savable
type gpuAlloc struct {
	// Base is the device virtual address of the allocation.
	Base uint64

	// Length is the size of the allocation in bytes.
	Length uint64
}

// Init initializes the allocation tracker. Must be called once before use,
// typically from nvproxy.Register(). On restore, stateify deserializes the
// allocs map directly, so Init() is only needed for fresh initialization.
func (at *allocTracker) Init() {
	at.allocs = make(map[uint64]gpuAlloc)
	at.totalBytes = 0
}

// RecordAlloc records a GPU memory allocation at the given base address with
// the given length. If an allocation already exists at that address, it is
// replaced (with a warning).
func (at *allocTracker) RecordAlloc(base, length uint64) {
	at.mu.Lock()
	defer at.mu.Unlock()

	if prev, exists := at.allocs[base]; exists {
		at.totalBytes -= prev.Length
		log.Warningf("nvproxy: tracker: overwriting existing allocation at VA %#x (old len=%d, new len=%d)", base, prev.Length, length)
	}
	at.allocs[base] = gpuAlloc{Base: base, Length: length}
	at.totalBytes += length

	if log.IsLogging(log.Debug) {
		log.Debugf("nvproxy: tracker: recorded alloc VA=%#x len=%d (%d MB) [total: %d allocs, %d MB]",
			base, length, length>>20, len(at.allocs), at.totalBytes>>20)
	}
}

// RemoveAlloc removes the GPU memory allocation at the given base address.
// It is a no-op if no allocation exists at that address.
func (at *allocTracker) RemoveAlloc(base uint64) {
	at.mu.Lock()
	defer at.mu.Unlock()

	alloc, exists := at.allocs[base]
	if !exists {
		return
	}
	at.totalBytes -= alloc.Length
	delete(at.allocs, base)

	if log.IsLogging(log.Debug) {
		log.Debugf("nvproxy: tracker: removed alloc VA=%#x len=%d [total: %d allocs, %d MB]",
			base, alloc.Length, len(at.allocs), at.totalBytes>>20)
	}
}

// Snapshot returns a copy of all tracked allocations. The returned map is
// safe to iterate without holding the tracker lock.
func (at *allocTracker) Snapshot() map[uint64]gpuAlloc {
	at.mu.Lock()
	defer at.mu.Unlock()

	snap := make(map[uint64]gpuAlloc, len(at.allocs))
	for k, v := range at.allocs {
		snap[k] = v
	}
	return snap
}

// TotalBytes returns the total bytes of tracked GPU memory.
func (at *allocTracker) TotalBytes() uint64 {
	at.mu.Lock()
	defer at.mu.Unlock()
	return at.totalBytes
}

// NumAllocs returns the number of tracked allocations.
func (at *allocTracker) NumAllocs() int {
	at.mu.Lock()
	defer at.mu.Unlock()
	return len(at.allocs)
}

// Reset clears all tracked allocations. This is used during restore setup
// when GPU state is being reconstructed from scratch.
func (at *allocTracker) Reset() {
	at.mu.Lock()
	defer at.mu.Unlock()
	at.allocs = make(map[uint64]gpuAlloc)
	at.totalBytes = 0
}

// --- Tracked UVM ioctl handler functions ---
//
// These functions wrap the standard UVM ioctl handlers with allocation
// tracking. They are intended to replace the generic handlers in the
// driverABI.uvmIoctl table (version.go) for the specific ioctls that
// create and destroy GPU memory mappings.
//
// Registration in version.go (replaces the generic handlers):
//
//   nvgpu.UVM_MAP_EXTERNAL_ALLOCATION: uvmHandler(uvmMapExtAllocTracked, compUtil),
//   nvgpu.UVM_FREE:                    uvmHandler(uvmFreeTracked, compUtil),
//   nvgpu.UVM_UNMAP_EXTERNAL:          uvmHandler(uvmUnmapExternalTracked, compUtil),
//
// And for version-specific variants:
//
//   V550: UVM_MAP_EXTERNAL_ALLOCATION → uvmMapExtAllocTrackedV550
//   V590: UVM_FREE → uvmFreeTrackedV590

// uvmMapExtAllocTracked wraps the UVM_MAP_EXTERNAL_ALLOCATION handler with
// allocation tracking. On success (RMStatus == 0), it records {Base, Length}
// in the allocation tracker.
func uvmMapExtAllocTracked(ui *uvmIoctlState) (uintptr, error) {
	n, err := uvmIoctlHasFrontendFD[nvgpu.UVM_MAP_EXTERNAL_ALLOCATION_PARAMS](ui)
	if err != nil {
		return n, err
	}
	// Re-read params from user memory to get the post-ioctl result status
	// and the allocation base/length. The inner handler already did CopyOut.
	var params nvgpu.UVM_MAP_EXTERNAL_ALLOCATION_PARAMS
	if _, copyErr := params.CopyIn(ui.t, ui.ioctlParamsAddr); copyErr == nil {
		if params.RMStatus == 0 && params.Length > 0 {
			ui.fd.dev.nvp.tracker.RecordAlloc(params.Base, params.Length)
		}
	}
	return n, nil
}

// uvmMapExtAllocTrackedV550 is the V550+ variant of uvmMapExtAllocTracked.
// The params struct has a larger PerGPUAttributes array but the same
// Base/Length/RMStatus fields.
func uvmMapExtAllocTrackedV550(ui *uvmIoctlState) (uintptr, error) {
	n, err := uvmIoctlHasFrontendFD[nvgpu.UVM_MAP_EXTERNAL_ALLOCATION_PARAMS_V550](ui)
	if err != nil {
		return n, err
	}
	var params nvgpu.UVM_MAP_EXTERNAL_ALLOCATION_PARAMS_V550
	if _, copyErr := params.CopyIn(ui.t, ui.ioctlParamsAddr); copyErr == nil {
		if params.RMStatus == 0 && params.Length > 0 {
			ui.fd.dev.nvp.tracker.RecordAlloc(params.Base, params.Length)
		}
	}
	return n, nil
}

// uvmFreeTracked wraps the UVM_FREE handler with allocation tracking.
// On success (RMStatus == 0), it removes the allocation at Base from the
// tracker.
func uvmFreeTracked(ui *uvmIoctlState) (uintptr, error) {
	// Read Base before invoking the ioctl (it's an input parameter).
	var preParams nvgpu.UVM_FREE_PARAMS
	if _, err := preParams.CopyIn(ui.t, ui.ioctlParamsAddr); err != nil {
		return 0, err
	}
	base := preParams.Base

	n, err := uvmIoctlSimple[nvgpu.UVM_FREE_PARAMS](ui)
	if err != nil {
		return n, err
	}

	// Re-read to check status.
	var postParams nvgpu.UVM_FREE_PARAMS
	if _, copyErr := postParams.CopyIn(ui.t, ui.ioctlParamsAddr); copyErr == nil {
		if postParams.RMStatus == 0 {
			ui.fd.dev.nvp.tracker.RemoveAlloc(base)
		}
	}
	return n, nil
}

// uvmFreeTrackedV590 is the V590+ variant of uvmFreeTracked. The params
// struct has Base but no Length field.
func uvmFreeTrackedV590(ui *uvmIoctlState) (uintptr, error) {
	var preParams nvgpu.UVM_FREE_PARAMS_V590
	if _, err := preParams.CopyIn(ui.t, ui.ioctlParamsAddr); err != nil {
		return 0, err
	}
	base := preParams.Base

	n, err := uvmIoctlSimple[nvgpu.UVM_FREE_PARAMS_V590](ui)
	if err != nil {
		return n, err
	}

	var postParams nvgpu.UVM_FREE_PARAMS_V590
	if _, copyErr := postParams.CopyIn(ui.t, ui.ioctlParamsAddr); copyErr == nil {
		if postParams.RMStatus == 0 {
			ui.fd.dev.nvp.tracker.RemoveAlloc(base)
		}
	}
	return n, nil
}

// uvmUnmapExternalTracked wraps the UVM_UNMAP_EXTERNAL handler with
// allocation tracking. On success (RMStatus == 0), it removes the
// allocation at Base from the tracker.
func uvmUnmapExternalTracked(ui *uvmIoctlState) (uintptr, error) {
	var preParams nvgpu.UVM_UNMAP_EXTERNAL_PARAMS
	if _, err := preParams.CopyIn(ui.t, ui.ioctlParamsAddr); err != nil {
		return 0, err
	}
	base := preParams.Base

	n, err := uvmIoctlSimple[nvgpu.UVM_UNMAP_EXTERNAL_PARAMS](ui)
	if err != nil {
		return n, err
	}

	var postParams nvgpu.UVM_UNMAP_EXTERNAL_PARAMS
	if _, copyErr := postParams.CopyIn(ui.t, ui.ioctlParamsAddr); copyErr == nil {
		if postParams.RMStatus == 0 {
			ui.fd.dev.nvp.tracker.RemoveAlloc(base)
		}
	}
	return n, nil
}
