// Copyright 2025 The gVisor Authors.
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
	"gvisor.dev/gvisor/pkg/context"
	"gvisor.dev/gvisor/pkg/log"
	"gvisor.dev/gvisor/pkg/sentry/vfs"
)

// fabricClasses are the RM object classes that back CUDA fabric / multicast
// memory allocated via cuMemExportToShareableHandle() (e.g. NCCL NVLS and
// PyTorch symmetric memory). cuda-checkpoint cannot serialize this memory and
// hangs on it during `--action checkpoint`, so nvproxy drains (frees) these
// objects while the application is quiesced, before cuda-checkpoint runs.
var fabricClasses = map[nvgpu.ClassID]struct{}{
	nvgpu.NV_MEMORY_FABRIC:              {},
	nvgpu.NV_MEMORY_MULTICAST_FABRIC:    {},
	nvgpu.NV_MEMORY_FABRIC_IMPORTED_REF: {},
	nvgpu.NV_MEMORY_EXPORT:              {},
}

// DrainedFabricObject describes a fabric RM object freed by DrainFabricMemory.
// It carries what is needed to log and (in a future slice) replay the object.
type DrainedFabricObject struct {
	Client nvgpu.Handle
	Parent nvgpu.Handle
	Handle nvgpu.Handle
	Class  nvgpu.ClassID
}

// FabricCensus returns, without modifying any state, the fabric RM objects
// currently tracked by the nvproxy registered in vfsObj. It is the read-only
// counterpart of DrainFabricMemory, used to inventory cuda-checkpoint-
// unsupported memory before a checkpoint.
func FabricCensus(ctx context.Context, vfsObj *vfs.VirtualFilesystem) []DrainedFabricObject {
	nvp := nvproxyFromVFS(vfsObj)
	if nvp == nil {
		return nil
	}
	var out []DrainedFabricObject
	nvp.clientsMu.RLock()
	clients := make([]*rootClient, 0, len(nvp.clients))
	for _, c := range nvp.clients {
		clients = append(clients, c)
	}
	nvp.clientsMu.RUnlock()
	for _, client := range clients {
		client.objsMu.Lock()
		if !client.released {
			for h, o := range client.resources {
				if _, ok := fabricClasses[o.class]; ok {
					out = append(out, DrainedFabricObject{
						Client: client.handle, Parent: o.parent, Handle: h, Class: o.class,
					})
				}
			}
		}
		client.objsMu.Unlock()
	}
	return out
}

// DrainFabricMemory frees all CUDA fabric / multicast RM objects tracked by the
// nvproxy registered in vfsObj, so that a subsequent `cuda-checkpoint --action
// checkpoint` does not hang on memory it cannot serialize. It must be called
// only while all CUDA processes are locked (quiesced). It returns the drained
// objects, for logging and future replay.
//
// NOTE (PoC): this drains fabric objects across all containers sharing this
// nvproxy, and does not yet replay them on restore. Both are addressed in later
// slices; this slice validates that draining unblocks the checkpoint.
func DrainFabricMemory(ctx context.Context, vfsObj *vfs.VirtualFilesystem) ([]DrainedFabricObject, error) {
	nvp := nvproxyFromVFS(vfsObj)
	if nvp == nil {
		return nil, nil
	}
	return nvp.drainFabricMemory(ctx)
}

func (nvp *nvproxy) drainFabricMemory(ctx context.Context) ([]DrainedFabricObject, error) {
	// Phase A: unmap all UVM external ranges. CUDA fabric/multicast memory is
	// mapped into the application's address space via these ranges; on peer
	// ranks that join a multicast group they are the *only* trace of the fabric
	// memory (no local RM object). They must be unmapped before the RM objects
	// are freed, so cuda-checkpoint does not walk a dangling external mapping.
	nvp.fdsMu.Lock()
	uvmFDs := make([]*uvmFD, 0, len(nvp.uvmFDs))
	for fd := range nvp.uvmFDs {
		uvmFDs = append(uvmFDs, fd)
	}
	nvp.fdsMu.Unlock()
	numRanges := 0
	for _, fd := range uvmFDs {
		fd.extRangesMu.Lock()
		for base, length := range fd.extRanges {
			status, err := freeUvmRangeOnHost(fd.hostFD, base)
			if err != nil || status != nvgpu.NV_OK {
				log.Warningf("nvproxy: failed to drain UVM external range %#x (len %#x): err=%v status=%#x", base, length, err, status)
				continue
			}
			delete(fd.extRanges, base)
			numRanges++
		}
		fd.extRangesMu.Unlock()
	}
	if numRanges > 0 {
		log.Infof("nvproxy: drained %d UVM external range(s)", numRanges)
	}

	// Phase B: free the fabric RM objects themselves.
	// Snapshot the set of clients so we don't hold clientsMu while taking
	// per-client objsMu (lock ordering: objsMu -> clientsMu).
	nvp.clientsMu.RLock()
	clients := make([]*rootClient, 0, len(nvp.clients))
	for _, c := range nvp.clients {
		clients = append(clients, c)
	}
	nvp.clientsMu.RUnlock()

	var drained []DrainedFabricObject
	for _, client := range clients {
		client.objsMu.Lock()
		if client.released {
			client.objsMu.Unlock()
			continue
		}
		// Collect fabric handles first; freeing mutates client.resources.
		var handles []nvgpu.Handle
		for h, o := range client.resources {
			if _, ok := fabricClasses[o.class]; ok {
				handles = append(handles, h)
			}
		}
		var deferReleases []func()
		for _, h := range handles {
			o, ok := client.resources[h]
			if !ok {
				// Already freed as a dependent of an earlier free.
				continue
			}
			fd := client.params.fd
			if fd == nil {
				log.Warningf("nvproxy: cannot drain fabric object %v:%v (class %v): client has no frontend FD", client.handle, h, o.class)
				continue
			}
			rec := DrainedFabricObject{Client: client.handle, Parent: o.parent, Handle: h, Class: o.class}
			status, err := freeObjectOnHost(fd.hostFD, client.handle, o.parent, h)
			if err != nil || status != nvgpu.NV_OK {
				log.Warningf("nvproxy: failed to drain fabric object %v:%v (class %v): err=%v status=%#x", client.handle, h, o.class, err, status)
				continue
			}
			deferReleases = append(deferReleases, nvp.objFree(ctx, client, h)...)
			drained = append(drained, rec)
			log.Infof("nvproxy: drained fabric object %v:%v (class %v)", client.handle, h, o.class)
		}
		client.objsMu.Unlock()
		for _, release := range deferReleases {
			release()
		}
	}
	return drained, nil
}
