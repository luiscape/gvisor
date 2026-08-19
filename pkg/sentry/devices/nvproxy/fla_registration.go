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
	goContext "context"
	"fmt"

	"gvisor.dev/gvisor/pkg/abi/nvgpu"
	"gvisor.dev/gvisor/pkg/context"
	"gvisor.dev/gvisor/pkg/log"
	"gvisor.dev/gvisor/pkg/sentry/vfs"
)

// This file implements suspend/replay of NV_MEMORY_FABRIC (00f8) objects
// around cuda-checkpoint.
//
// On fabric-attached GPUs with the RM fabric probe allowed
// (fabric-imex-mgmt), libcuda lazily allocates one 00f8 object per VMM
// allocation that is shared between processes: an FLA *registration* whose
// alloc params carry Map.HVidMem != 0, naming the covered vidmem object. It
// is metadata over content that lives elsewhere; nothing in userspace frees
// it while the covered allocation lives, and no CUDA API can. cuda-checkpoint
// checkpoints a live 00f8 but cannot restore it: the restored process dies in
// NV_ERR_OBJECT_NOT_FOUND storms (measured; see the blocker gate, which
// exists to refuse such snapshots).
//
// nvproxy is the only layer holding both the handle and the original alloc
// params, so it owns the fix:
//
//   - Checkpoint: SuspendFLARegistrations host-frees each registration and
//     records it on nvproxy.suspendedFLARegs. cuda-checkpoint then never
//     sees one.
//   - Restore/resume: ReplayFLARegistrations recreates each with identical
//     client/handle/params, after the cuda-checkpoint toggle has rebuilt
//     the vidmem it covers (under its original handle -- that identity is
//     cuda-checkpoint's contract with libcuda's restored bookkeeping) and
//     before the multicast interposer's resume, whose re-exports make
//     libcuda reference the registration handle again (measured: re-export
//     fails with OBJECT_NOT_FOUND if the replay is skipped).
//
// The record deliberately lives OUTSIDE the object graph: cuda-checkpoint's
// checkpoint phase frees the process's remaining RM objects through nvproxy,
// cascading graph entries away before the sandbox state is saved, so replay
// bookkeeping attached to graph objects would be gone exactly when needed.
// nvproxy.suspendedFLARegs survives both the checkpoint phase and the state
// round-trip.

// memoryFabricObject is an objectImpl tracking an NV_MEMORY_FABRIC (00f8)
// allocation.
//
// +stateify savable
type memoryFabricObject struct {
	object

	params capturedRmAllocParams
}

// Release implements objectImpl.Release.
func (o *memoryFabricObject) Release(ctx context.Context) func() {
	// no-op
	return nil
}

// Restore implements restorableObjectImpl.Restore.
func (o *memoryFabricObject) Restore(ctx goContext.Context) error {
	return o.params.restore()
}

// isFLARegistration returns true if this 00f8 object is a driver-internal
// FLA registration over an existing vidmem allocation (Map.HVidMem != 0),
// as opposed to application-requested fabric memory.
func (o *memoryFabricObject) isFLARegistration() bool {
	var allocParams nvgpu.NV00F8_ALLOCATION_PARAMETERS
	if len(o.params.allocParams) < allocParams.SizeBytes() {
		return false
	}
	allocParams.UnmarshalBytes(o.params.allocParams[:allocParams.SizeBytes()])
	return allocParams.Map.HVidMem.Val != 0
}

// suspendedFLARegistration records a host-freed FLA registration for replay.
//
// +stateify savable
type suspendedFLARegistration struct {
	clientH nvgpu.Handle
	objectH nvgpu.Handle
	parentH nvgpu.Handle
	hVidMem nvgpu.Handle
	params  capturedRmAllocParams
	// clientFD is the frontendFD that allocated the root client. RM binds
	// clients to their allocating fd (frees or allocs on behalf of the
	// client via any other fd fail with NV_ERR_INVALID_CLIENT, measured), so
	// the replay must ride this fd. The pointer survives the state
	// round-trip; frontendFD.load() gives it a fresh hostFD at restore.
	clientFD *frontendFD
}

// rmAllocMemoryFabric is the NV_ESC_RM_ALLOC handler for NV_MEMORY_FABRIC
// (00f8): rmAllocSimple plus diagnostic logging of the allocation parameters
// (fabric objects block CUDA checkpoints, and attributing one to the VMM
// allocation it covers is what makes such a blocker debuggable), a
// suspendable object implementation, and a restore-ordering dependency on
// the covered vidmem object so that, in flows where the afterLoad object
// replay runs, the registration is recreated only after the memory it maps.
func rmAllocMemoryFabric(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS64_PARAMETERS, isNVOS64 bool) (uintptr, error) {
	return rmAllocSimpleParams[nvgpu.NV00F8_ALLOCATION_PARAMETERS](fi, ioctlParams, isNVOS64,
		func(fi *frontendIoctlState, client *rootClient, ioctlParams *nvgpu.NVOS64_PARAMETERS, rightsRequested nvgpu.RS_ACCESS_MASK, allocParams *nvgpu.NV00F8_ALLOCATION_PARAMETERS) {
			if allocParams != nil {
				fi.ctx.Debugf("nvproxy: NV_MEMORY_FABRIC alloc: allocSize=%#x pageSize=%#x alignment=%#x allocFlags=%#x map.hVidMem=%#x map.flags=%#x",
					allocParams.AllocSize, allocParams.PageSize, allocParams.Alignment, allocParams.AllocFlags, allocParams.Map.HVidMem.Val, allocParams.Map.Flags)
			}
			impl := &memoryFabricObject{
				params: captureRmAllocParams(fi.fd, ioctlParams, rightsRequested, allocParams),
			}
			fi.fd.dev.nvp.objAdd(fi.ctx, client, ioctlParams.HObjectNew, ioctlParams.HClass, impl, ioctlParams.HObjectParent)
			if allocParams != nil && allocParams.Map.HVidMem.Val != 0 {
				// Restore-ordering only: freeing the covered vidmem does not
				// free the registration in the driver's model, so a hard
				// deps edge would over-free.
				client.objAddRestoreDep(ioctlParams.HObjectNew, allocParams.Map.HVidMem)
			}
		})
}

// SuspendFLARegistrations host-frees every live FLA registration (see file
// comment) so that cuda-checkpoint never sees one, recording each for
// ReplayFLARegistrations. It returns the number of registrations freed.
//
// Preconditions: the application's CUDA processes must be quiesced (the
// caller runs this within the cuda-checkpoint lock/suspend window).
func SuspendFLARegistrations(vfsObj *vfs.VirtualFilesystem) (int, error) {
	nvp := nvproxyFromVFS(vfsObj)
	if nvp == nil {
		return 0, nil
	}
	return nvp.suspendFLARegistrations()
}

func (nvp *nvproxy) suspendFLARegistrations() (int, error) {
	type target struct {
		client *rootClient
		obj    *memoryFabricObject
	}
	var targets []target
	nvp.clientsMu.RLock()
	clients := make([]*rootClient, 0, len(nvp.clients))
	for _, c := range nvp.clients {
		clients = append(clients, c)
	}
	nvp.clientsMu.RUnlock()
	for _, client := range clients {
		client.objsMu.Lock()
		if !client.released {
			for _, o := range client.resources {
				if mf, ok := o.impl.(*memoryFabricObject); ok && mf.isFLARegistration() {
					targets = append(targets, target{client, mf})
				}
			}
		}
		client.objsMu.Unlock()
	}
	ctx := context.Background()
	freed := 0
	for _, t := range targets {
		// The client's allocating fd carries the free (see
		// suspendedFLARegistration.clientFD). It is necessarily open: RM
		// frees the whole client when its fd closes, and this client is
		// live.
		clientFD := t.client.params.fd
		status, err := rmFreeOnHost(clientFD.hostFD, t.client.handle, t.obj.handle)
		if err != nil {
			return freed, fmt.Errorf("host RM_FREE of FLA registration %v:%v failed: %w", t.client.handle, t.obj.handle, err)
		}
		if status != nvgpu.NV_OK {
			return freed, fmt.Errorf("host RM_FREE of FLA registration %v:%v failed: status=%#x", t.client.handle, t.obj.handle, status)
		}
		var allocParams nvgpu.NV00F8_ALLOCATION_PARAMETERS
		allocParams.UnmarshalBytes(t.obj.params.allocParams[:allocParams.SizeBytes()])
		rec := suspendedFLARegistration{
			clientH:  t.client.handle,
			objectH:  t.obj.handle,
			parentH:  t.obj.parent,
			hVidMem:  allocParams.Map.HVidMem,
			params:   t.obj.params,
			clientFD: clientFD,
		}
		// Drop the graph entry: the driver-side object no longer exists, and
		// the entry would otherwise be reported by the blocker gate and
		// cascaded away by cuda-checkpoint's own frees anyway.
		t.client.objsMu.Lock()
		cleanup := nvp.objFree(ctx, t.client, t.obj.handle)
		t.client.objsMu.Unlock()
		for _, f := range cleanup {
			f()
		}
		nvp.suspendedFLARegs = append(nvp.suspendedFLARegs, rec)
		freed++
		log.Debugf("nvproxy: host-freed FLA registration %v:%v for checkpoint", t.client.handle, t.obj.handle)
	}
	return freed, nil
}

// ReplayFLARegistrations recreates the registrations freed by
// SuspendFLARegistrations with their original handles and parameters, and
// re-records them in the object graph. It applies only to flows where the
// process's driver state survived: checkpoint-failure unwind and
// resume-after-save. After a TRUE restore it must not (and does not) run:
// nvproxy.afterLoad drops the suspended list, because the cuda-checkpoint
// restore toggle rebuilds the process's RM objects under fresh handles
// (measured: the covered vidmem's original handle appears nowhere in the
// rebuilt graph), so the saved identities are unreplayable -- and
// unnecessary: libcuda lazily re-registers FLA on the next export, exactly
// as it did on first use (measured: the fusion workload's restore recreates
// its registrations itself and passes end-to-end).
// It returns the number of registrations recreated.
func ReplayFLARegistrations(vfsObj *vfs.VirtualFilesystem) (int, error) {
	nvp := nvproxyFromVFS(vfsObj)
	if nvp == nil {
		return 0, nil
	}
	return nvp.replayFLARegistrations()
}


func (nvp *nvproxy) replayFLARegistrations() (int, error) {
	regs := nvp.suspendedFLARegs
	replayed := 0
	ctx := context.Background()
	for i := range regs {
		rec := &regs[i]
		// Fd selection. RM accepts operations on a client only from fds it
		// associates with that client, and the fd that allocated the client
		// may have been closed by the application since (its saved hostFD
		// then dangles; measured as ENOTTY). The one fd that is both
		// guaranteed live and guaranteed associated: whichever fd the
		// application (or the cuda-checkpoint restore toggle, acting through
		// the application's fds) most recently used to allocate the vidmem
		// this registration covers -- it is in the current object graph.
		fd := int32(-1)
		nvp.clientsMu.RLock()
		if client := nvp.clients[rec.clientH]; client != nil {
			client.objsMu.Lock()
			if vidmem, ok := client.resources[rec.hVidMem]; ok {
				switch impl := vidmem.impl.(type) {
				case *rmAllocObject:
					fd = impl.params.fd.hostFD
				case *osDescMem:
					// Not expected for vidmem, but harmless to skip.
				}
			}
			client.objsMu.Unlock()
			if fd < 0 && client.params.fd != nil {
				fd = client.params.fd.hostFD
			}
		}
		nvp.clientsMu.RUnlock()
		if fd < 0 && rec.clientFD != nil {
			fd = rec.clientFD.hostFD
		}
		if err := rec.params.restoreOnFD(fd); err != nil {
			// Drop what was replayed; keep the rest recorded for another
			// attempt or for diagnosis.
			nvp.suspendedFLARegs = regs[replayed:]
			return replayed, fmt.Errorf("replaying FLA registration %v:%v failed: %w", rec.clientH, rec.objectH, err)
		}
		// Re-record in the object graph so a later checkpoint sees the truth.
		nvp.clientsMu.RLock()
		client := nvp.clients[rec.clientH]
		nvp.clientsMu.RUnlock()
		if client != nil {
			impl := &memoryFabricObject{params: rec.params}
			client.objsMu.Lock()
			nvp.objAdd(ctx, client, rec.objectH, nvgpu.NV_MEMORY_FABRIC, impl, rec.parentH)
			client.objAddRestoreDep(rec.objectH, rec.hVidMem)
			client.objsMu.Unlock()
		} else {
			log.Warningf("nvproxy: replayed FLA registration %v:%v, but its client is gone from the object graph", rec.clientH, rec.objectH)
		}
		replayed++
	}
	nvp.suspendedFLARegs = nil
	return replayed, nil
}
