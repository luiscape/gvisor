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
// On fabric-attached GPUs (NVSwitch with the fabric manager running -- the
// RM fabric probe succeeds in default sandboxes), libcuda lazily allocates
// one 00f8 object per VMM allocation that is shared between processes: an
// FLA *registration* whose alloc params carry Map.HVidMem != 0, naming the
// covered vidmem object. It is metadata over content that lives elsewhere;
// nothing in userspace frees it while the covered allocation lives, and no
// CUDA API can. cuda-checkpoint checkpoints a live 00f8 but cannot restore
// it: the restored process dies in NV_ERR_OBJECT_NOT_FOUND storms
// (measured; the blocker gate exists to refuse such snapshots).
//
// nvproxy is the only layer holding both the handle and the original alloc
// params, so it owns the fix: SuspendFLARegistrations host-frees each
// registration inside the cuda-checkpoint suspend window and records it on
// nvproxy.suspendedFLARegs, so cuda-checkpoint never sees one. What happens
// afterwards depends on the fate of the process's driver state:
//
//   - Checkpoint-failure unwind, including resume after a FAILED save
//     (driver state survived in place): ReplayFLARegistrations recreates
//     each registration with identical client/handle/params before the
//     application runs again.
//   - Resume after a SUCCESSFUL save: out of scope (single-pass
//     checkpoint->restore only); postResumeCuda fails loudly if any
//     registration is pending.
//   - True restore: replay is impossible and unnecessary; see the comment
//     in nvproxy.afterLoad, which drops the record. libcuda lazily
//     re-registers on the next export.
//
// The record deliberately lives OUTSIDE the object graph: cuda-checkpoint's
// checkpoint phase frees the process's remaining RM objects through nvproxy,
// cascading graph entries away before the sandbox state is saved, so replay
// bookkeeping attached to graph objects would be gone exactly when needed.

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
	params  capturedRmAllocParams
	// client is the rootClient the registration belonged to, by IDENTITY:
	// replay is valid only while nvp.clients[clientH] is still this exact
	// object. If cuda-checkpoint's teardown freed the client (and the
	// restore toggle rebuilt one under the same handle), the record is
	// stale and must be dropped instead -- see replayFLARegistrations.
	client *rootClient
	// clientFD is the frontendFD that allocated the root client -- the one
	// carrier known-good for sentry-driven RM calls on this client
	// (another process's fd fails with NV_ERR_INVALID_CLIENT, measured).
	// The pointer survives the state round-trip.
	clientFD *frontendFD
}

// rmAllocMemoryFabric is the NV_ESC_RM_ALLOC handler for NV_MEMORY_FABRIC
// (00f8): rmAllocSimple plus diagnostic logging of the allocation parameters
// (fabric objects block CUDA checkpoints, and attributing one to the VMM
// allocation it covers is what makes such a blocker debuggable) and a
// suspendable object implementation.
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
		rec := suspendedFLARegistration{
			clientH:  t.client.handle,
			objectH:  t.obj.handle,
			parentH:  t.obj.parent,
			params:   t.obj.params,
			client:   t.client,
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
		// Append, never reset: records left by an earlier failed unwind
		// are still-freed registrations that the next replay opportunity
		// must also cover.
		nvp.suspendedFLARegs = append(nvp.suspendedFLARegs, rec)
		freed++
		log.Debugf("nvproxy: host-freed FLA registration %v:%v for checkpoint", t.client.handle, t.obj.handle)
	}
	return freed, nil
}

// ReplayFLARegistrations recreates the registrations freed by
// SuspendFLARegistrations with their original handles and parameters, and
// re-records them in the object graph. It serves the checkpoint-failure
// paths: the application must keep running (or be re-checkpointed) with its
// driver state whole. Records whose client was torn down by cuda-checkpoint
// (failures past the checkpoint action) are dropped instead of replayed --
// for those the restore toggle rebuilt driver state fresh and libcuda
// re-registers lazily, exactly as after a true restore. After a TRUE restore nvproxy.afterLoad drops the record
// instead: sentry-driven allocation into the restored client is impossible
// (measured exhaustively -- see the comment in afterLoad), and libcuda
// re-registers lazily on the next export. The residual gap is workloads
// whose userspace caches registration state across the export: torch
// symm-mem's post-restore re-export retries EXPORT_OBJECT_TO_FD against the
// stale registration handle into OBJECT_NOT_FOUND. That gap is a
// userspace-bookkeeping/driver mismatch only NVIDIA can close (the
// checkpointed image believes in an object that cannot be checkpointed).
// It returns the number of registrations recreated.
func ReplayFLARegistrations(vfsObj *vfs.VirtualFilesystem) (int, error) {
	nvp := nvproxyFromVFS(vfsObj)
	if nvp == nil {
		return 0, nil
	}
	return nvp.replayFLARegistrations()
}

// PendingFLARegistrations returns the number of FLA registrations that were
// host-freed for a checkpoint and not yet recreated. Nonzero outside the
// checkpoint sequence means the sandbox kept running past a successful save
// (save-and-resume) with fabric users -- a flow that is out of scope
// (single-pass checkpoint->restore only) and must fail loudly rather than
// let the application dereference freed registrations.
func PendingFLARegistrations(vfsObj *vfs.VirtualFilesystem) int {
	nvp := nvproxyFromVFS(vfsObj)
	if nvp == nil {
		return 0
	}
	return len(nvp.suspendedFLARegs)
}

func (nvp *nvproxy) replayFLARegistrations() (int, error) {
	regs := nvp.suspendedFLARegs
	replayed := 0
	ctx := context.Background()
	for i := range regs {
		rec := &regs[i]
		// Which failure geometry is this record in? Decided structurally, by
		// client identity in the sentry's own object graph -- errno probing
		// is unreliable here because the restore toggle recycles fd numbers:
		//
		//   - Failure BEFORE the cuda-checkpoint checkpoint action: the
		//     application was frozen the whole time, so the client that
		//     allocated the registration is still live and IS
		//     nvp.clients[clientH]. Its allocating fd is open by RM
		//     invariant (RM frees the client when its fd closes), and the
		//     replay must recreate the registration on it.
		//   - Failure AFTER the checkpoint action (e.g. the save failed in
		//     encoding): the checkpoint action freed the client through
		//     nvproxy (removing it from nvp.clients), and the restore toggle
		//     rebuilt driver state fresh -- possibly a NEW client under the
		//     same handle. That is the same geometry as a true restore,
		//     where identity replay is impossible and unnecessary (measured;
		//     see afterLoad): libcuda re-registers lazily on the next
		//     export. Drop the record instead of failing the resume.
		// rootClient.Release removes the client from nvp.clients under
		// clientsMu, so identity under that lock is exact: same pointer =>
		// never released.
		nvp.clientsMu.RLock()
		client := nvp.clients[rec.clientH]
		nvp.clientsMu.RUnlock()
		if client != rec.client {
			log.Infof("nvproxy: dropped FLA registration %v:%v (client torn down by cuda-checkpoint); libcuda re-registers lazily on next export", rec.clientH, rec.objectH)
			continue
		}
		if err := rec.params.restoreOnFD(rec.clientFD.hostFD); err != nil {
			// Keep this record and everything unprocessed for another
			// attempt or for diagnosis; drop what was already handled.
			nvp.suspendedFLARegs = append([]suspendedFLARegistration{}, regs[i:]...)
			return replayed, fmt.Errorf("replaying FLA registration %v:%v failed: %w", rec.clientH, rec.objectH, err)
		}
		// Re-record in the object graph so a later checkpoint sees the truth.
		impl := &memoryFabricObject{params: rec.params}
		client.objsMu.Lock()
		nvp.objAdd(ctx, client, rec.objectH, nvgpu.NV_MEMORY_FABRIC, impl, rec.parentH)
		client.objsMu.Unlock()
		replayed++
	}
	nvp.suspendedFLARegs = nil
	return replayed, nil
}
