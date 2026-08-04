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

	"golang.org/x/sys/unix"
	"gvisor.dev/gvisor/pkg/abi/nvgpu"
	"gvisor.dev/gvisor/pkg/context"
	"gvisor.dev/gvisor/pkg/errors/linuxerr"
	"gvisor.dev/gvisor/pkg/log"
	"gvisor.dev/gvisor/pkg/sentry/vfs"
)

// This file implements tracking of NV_MEMORY_MULTICAST_FABRIC (00FD) objects
// for checkpoint/restore. A multicast object's state is built after
// allocation by state-mutating control calls (NV00FD_CTRL_CMD_ATTACH_GPU,
// NV00FD_CTRL_CMD_ATTACH_MEM); replaying only the allocation would produce an
// empty multicast group. So a narrow allowlist of controls is recorded on the
// object and replayed after the alloc during restore.
//
// Do not record controls outside this allowlist; the object graph is live
// state, not an ioctl log.

// mcastAttachGPU records a successful NV00FD_CTRL_CMD_ATTACH_GPU.
//
// +stateify savable
type mcastAttachGPU struct {
	// params are the application-provided parameters. DevDescriptor holds the
	// application FD number, which is meaningless at replay time; devMinor is
	// used to reconstruct a device descriptor instead.
	params nvgpu.NV00FD_CTRL_ATTACH_GPU_PARAMS

	// devMinor is the minor number of the /dev/nvidia# frontend device that
	// params.DevDescriptor referred to when the control was recorded.
	devMinor uint32
}

// mcastAttachMem records a successful NV00FD_CTRL_CMD_ATTACH_MEM.
//
// +stateify savable
type mcastAttachMem struct {
	params nvgpu.NV00FD_CTRL_ATTACH_MEM_PARAMS
}

// multicastFabricObject is an objectImpl tracking a
// NV_MEMORY_MULTICAST_FABRIC (00FD) driver object, together with the
// state-mutating control calls that built up its state, so that the whole
// object can be replayed on restore.
//
// +stateify savable
type multicastFabricObject struct {
	object

	params capturedRmAllocParams

	// attachedGPUs and attachedMems are recorded state-mutating controls,
	// replayed in order (all GPU attaches, then all mem attaches) after the
	// alloc replay. Both are protected by client.objsMu.
	attachedGPUs []mcastAttachGPU
	attachedMems []mcastAttachMem
}

func newMulticastFabricObject[Params any](fd *frontendFD, ioctlParams *nvgpu.NVOS64_PARAMETERS, rightsRequested nvgpu.RS_ACCESS_MASK, allocParams *Params) *multicastFabricObject {
	return &multicastFabricObject{
		params: captureRmAllocParams(fd, ioctlParams, rightsRequested, allocParams),
	}
}

// Release implements objectImpl.Release.
func (o *multicastFabricObject) Release(ctx context.Context) func() {
	// no-op
	return nil
}

// getMulticastObjectWithLock looks up the multicastFabricObject with the given
// handles. On success, the caller must call the returned unlock function when
// done with the object.
func (nvp *nvproxy) getMulticastObjectWithLock(ctx context.Context, clientH, objectH nvgpu.Handle) (*multicastFabricObject, func()) {
	client, unlock := nvp.getClientWithLock(ctx, clientH)
	if client == nil {
		return nil, nil
	}
	obj := client.getObject(ctx, objectH)
	if obj == nil {
		unlock()
		return nil, nil
	}
	mcObj, ok := obj.impl.(*multicastFabricObject)
	if !ok {
		ctx.Warningf("nvproxy: object %v:%v (class %v) is not a multicast fabric object", clientH, objectH, obj.class)
		unlock()
		return nil, nil
	}
	return mcObj, unlock
}

// recordAttachGPU records a successful NV00FD_CTRL_CMD_ATTACH_GPU on o.
//
// Precondition: o.client.objsMu must be locked.
func (o *multicastFabricObject) recordAttachGPU(params nvgpu.NV00FD_CTRL_ATTACH_GPU_PARAMS, devMinor uint32) {
	o.attachedGPUs = append(o.attachedGPUs, mcastAttachGPU{
		params:   params,
		devMinor: devMinor,
	})
}

// recordAttachMem records a successful NV00FD_CTRL_CMD_ATTACH_MEM on o.
//
// Precondition: o.client.objsMu must be locked.
func (o *multicastFabricObject) recordAttachMem(params nvgpu.NV00FD_CTRL_ATTACH_MEM_PARAMS) {
	o.attachedMems = append(o.attachedMems, mcastAttachMem{params: params})
}

// removeAttachMem removes the recorded attach matching a successful
// NV00FD_CTRL_CMD_DETACH_MEM (which identifies the binding by subdevice and
// offset into the multicast object).
//
// Precondition: o.client.objsMu must be locked.
func (o *multicastFabricObject) removeAttachMem(params nvgpu.NV00FD_CTRL_DETACH_MEM_PARAMS) {
	for i, am := range o.attachedMems {
		if am.params.HSubDevice == params.HSubDevice && am.params.Offset == params.Offset {
			o.attachedMems = append(o.attachedMems[:i], o.attachedMems[i+1:]...)
			return
		}
	}
	log.Warningf("nvproxy: DETACH_MEM on %v:%v (subdevice %v, offset %#x) does not match any recorded attach", o.client.handle, o.handle, params.HSubDevice, params.Offset)
}

// Restore implements restorableObjectImpl.Restore.
//
// It replays the saved allocation, then the recorded ATTACH_GPU and
// ATTACH_MEM controls, in order. Per the Phase 0 attach_blocking measurement
// (gpu_mem_snapshots/phase0/), ATTACH_GPU does not block, but ATTACH_MEM
// blocks until ALL participating GPUs have joined the multicast object; a
// caller restoring multiple clients of one multicast object must therefore
// complete every client's ATTACH_GPU replay before any client's ATTACH_MEM
// replay (see restoreAttachesSplit).
func (o *multicastFabricObject) Restore(ctx goContext.Context) error {
	if err := o.params.restore(); err != nil {
		return fmt.Errorf("failed to replay multicast fabric alloc: %w", err)
	}
	if err := o.restoreAttachGPUs(ctx); err != nil {
		return err
	}
	return o.restoreAttachMems(ctx)
}

// restoreAttachGPUs replays the recorded ATTACH_GPU controls.
func (o *multicastFabricObject) restoreAttachGPUs(ctx goContext.Context) error {
	for _, ag := range o.attachedGPUs {
		devFD, err := o.hostDevFDForMinor(ctx, ag.devMinor)
		if err != nil {
			return fmt.Errorf("failed to get device descriptor for ATTACH_GPU replay (minor %d): %w", ag.devMinor, err)
		}
		params := ag.params
		params.DevDescriptor = uint64(devFD.fd)
		status, err := controlObjectOnHost(o.params.fd.hostFD, o.client.handle, o.handle, nvgpu.NV00FD_CTRL_CMD_ATTACH_GPU, &params)
		devFD.close()
		if err != nil || status != nvgpu.NV_OK {
			return fmt.Errorf("failed to replay NV00FD_CTRL_CMD_ATTACH_GPU (subdevice %v, minor %d): errno=%v status=%#x", ag.params.HSubDevice, ag.devMinor, err, status)
		}
	}
	return nil
}

// restoreAttachMems replays the recorded ATTACH_MEM controls.
//
// NOTE: ATTACH_MEM blocks until all participating GPUs have attached (see
// Restore); the caller is responsible for ordering across clients.
func (o *multicastFabricObject) restoreAttachMems(ctx goContext.Context) error {
	for _, am := range o.attachedMems {
		params := am.params
		status, err := controlObjectOnHost(o.params.fd.hostFD, o.client.handle, o.handle, nvgpu.NV00FD_CTRL_CMD_ATTACH_MEM, &params)
		if err != nil || status != nvgpu.NV_OK {
			return fmt.Errorf("failed to replay NV00FD_CTRL_CMD_ATTACH_MEM (subdevice %v, hMemory %v, offset %#x): errno=%v status=%#x", am.params.HSubDevice, am.params.HMemory, am.params.Offset, err, status)
		}
	}
	return nil
}

// hostDevFD is a host device FD used during replay, with an optional cleanup.
type hostDevFD struct {
	fd    int32
	close func()
}

// hostDevFDForMinor returns a host FD for /dev/nvidia<minor>, preferring an
// existing live frontendFD (whose hostFD was reopened during restore) and
// falling back to opening a fresh host FD.
func (o *multicastFabricObject) hostDevFDForMinor(ctx goContext.Context, minor uint32) (hostDevFD, error) {
	nvp := o.nvp
	nvp.fdsMu.Lock()
	for fd := range nvp.frontendFDs {
		if fd.dev.minor == minor && fd.hostFD >= 0 {
			nvp.fdsMu.Unlock()
			return hostDevFD{fd: fd.hostFD, close: func() {}}, nil
		}
	}
	nvp.fdsMu.Unlock()
	// Fall back to opening the device directly. This requires the dev gofer
	// client (or host /dev) to be reachable from ctx.
	// openHostDevFileForRestore panics on failure, consistent with the restore
	// path this runs on.
	dev := nvp.regularDevs[minor]
	if dev == nil {
		return hostDevFD{}, fmt.Errorf("no /dev/nvidia%d device", minor)
	}
	hostFD := openHostDevFileForRestore(ctx, dev.basename(), nvp.useDevGofer, o.params.fd.containerName, unix.O_RDWR)
	return hostDevFD{fd: hostFD, close: func() { unix.Close(int(hostFD)) }}, nil
}

// suspendedMulticastObject is a multicast object that nvproxy tore down
// host-side before a cuda-checkpoint checkpoint action (which hangs on live
// multicast objects), retaining everything needed to replay it after the
// post-restore toggle.
//
// It lives outside the per-client object graph because the checkpoint action
// releases the entire root client (Phase 0 census measurement): the graph
// entry is cascade-freed with the client, and the toggle recreates the client
// under a NEW handle. Child object handles are client-chosen and measured to
// be stable across the toggle, so replay needs only client-handle remapping.
//
// +stateify savable
type suspendedMulticastObject struct {
	// fd is the frontendFD the object was allocated through. It remains open
	// across checkpoint/restore (hostFD is reopened on restore), and the
	// toggle recreates the object's client on this same FD.
	fd *frontendFD

	oldClient    nvgpu.Handle
	handle       nvgpu.Handle
	params       capturedRmAllocParams
	attachedGPUs []mcastAttachGPU
	attachedMems []mcastAttachMem
}

// SuspendMulticastObjects tears down all live multicast (00FD) objects
// host-side, so that a subsequent `cuda-checkpoint --action checkpoint` does
// not hang on them, and stashes them for replay after the post-restore
// toggle. It must be called only while all CUDA processes are locked
// (quiesced). It returns the number of objects suspended.
func SuspendMulticastObjects(ctx context.Context, vfsObj *vfs.VirtualFilesystem) (int, error) {
	nvp := nvproxyFromVFS(vfsObj)
	if nvp == nil {
		return 0, nil
	}
	return nvp.suspendMulticastObjects(ctx)
}

// dummySubstituteParams builds NV_ESC_RM_ALLOC parameters for a plain vidmem
// allocation with the same handle as mcObj, cloned from the alloc params of
// one of the memory objects bound into the multicast group (guaranteeing
// libcuda-compatible size/attributes on the right device).
//
// Rationale (measured, phase0): after nvproxy frees a multicast object
// host-side, `cuda-checkpoint --action checkpoint` still fails: libcuda walks
// its (CRIU-preserved) handle table and creates a UVM mapping of every
// allocation to save its contents; the mapping of the freed multicast handle
// fails with OBJECT_NOT_FOUND => "Could not checkpoint: out of memory".
// Substituting a plain allocation under the same handle gives libcuda
// something it can checkpoint and restore; the substitute is replaced by the
// real multicast object during replay.
//
// Precondition: client.objsMu must be locked.
func dummySubstituteParams(client *rootClient, mcObj *multicastFabricObject, mcAllocSize uint64) *capturedRmAllocParams {
	clone := func(memObj *object) *capturedRmAllocParams {
		rmObj, ok := memObj.impl.(*rmAllocObject)
		if !ok {
			return nil
		}
		dummy := rmObj.params
		dummy.ioctlParams.HObjectNew = mcObj.handle
		dummy.allocParams = append([]byte(nil), rmObj.params.allocParams...)
		if mcAllocSize != 0 && !patchAllocSize(dummy.allocParams, mcAllocSize) {
			log.Warningf("nvproxy: could not patch substitute alloc size for multicast object %v:%v (params size %d)", client.handle, mcObj.handle, len(dummy.allocParams))
		}
		return &dummy
	}
	// Prefer a memory object bound into this multicast group (right device and
	// attributes by construction).
	for _, am := range mcObj.attachedMems {
		if memObj, ok := client.resources[am.params.HMemory]; ok {
			if dummy := clone(memObj); dummy != nil {
				return dummy
			}
		}
	}
	// Otherwise clone any vidmem allocation in the client (an unbound
	// multicast object still needs a substitute for cuda-checkpoint's
	// content-save pass).
	for _, o := range client.resources {
		if o.class != nvgpu.NV01_MEMORY_LOCAL_USER {
			continue
		}
		if dummy := clone(o); dummy != nil {
			return dummy
		}
	}
	return nil
}

// patchAllocSize sets the Size field of marshaled NV_MEMORY_ALLOCATION_PARAMS
// (or its V545+ extension, which embeds it at offset 0) to size. The
// substitute must have the multicast object's driver-rounded allocation size
// (e.g. 512MiB NVSwitch pages): libcuda's checkpoint content-save pass maps
// the handle with that length, and a smaller substitute fails with
// NV_ERR_INVALID_OFFSET (measured).
func patchAllocSize(allocParams []byte, size uint64) bool {
	var base nvgpu.NV_MEMORY_ALLOCATION_PARAMS
	switch len(allocParams) {
	case base.SizeBytes(), (&nvgpu.NV_MEMORY_ALLOCATION_PARAMS_V545{}).SizeBytes():
		base.UnmarshalBytes(allocParams[:base.SizeBytes()])
		base.Size = size
		base.MarshalBytes(allocParams[:base.SizeBytes()])
		return true
	default:
		return false
	}
}

func (nvp *nvproxy) suspendMulticastObjects(ctx context.Context) (int, error) {
	nvp.clientsMu.RLock()
	clients := make([]*rootClient, 0, len(nvp.clients))
	for _, c := range nvp.clients {
		clients = append(clients, c)
	}
	nvp.clientsMu.RUnlock()

	suspended := 0
	for _, client := range clients {
		client.objsMu.Lock()
		if client.released {
			client.objsMu.Unlock()
			continue
		}
		// Collect first: freeing mutates client.resources.
		var mcObjs []*multicastFabricObject
		for _, o := range client.resources {
			if mcObj, ok := o.impl.(*multicastFabricObject); ok {
				mcObjs = append(mcObjs, mcObj)
			}
		}
		var deferReleases []func()
		for _, mcObj := range mcObjs {
			fd := client.params.fd
			if fd == nil {
				client.objsMu.Unlock()
				return suspended, fmt.Errorf("cannot suspend multicast object %v:%v: client has no frontend FD", client.handle, mcObj.handle)
			}
			// The substitute must match the multicast object's driver-rounded
			// size, not the requested size; query it before freeing.
			var mcAllocSize uint64
			var infoParams nvgpu.NV00FD_CTRL_GET_INFO_PARAMS
			if status, err := controlObjectOnHost(fd.hostFD, client.handle, mcObj.handle, nvgpu.NV00FD_CTRL_CMD_GET_INFO, &infoParams); err == nil && status == nvgpu.NV_OK {
				mcAllocSize = infoParams.AllocSize
			} else {
				log.Warningf("nvproxy: NV00FD_CTRL_CMD_GET_INFO on %v:%v failed (errno=%v status=%#x); substitute will use the bound memory's size", client.handle, mcObj.handle, err, status)
			}
			dummy := dummySubstituteParams(client, mcObj, mcAllocSize)
			status, err := freeObjectOnHost(fd.hostFD, client.handle, mcObj.parent, mcObj.handle)
			if err != nil || status != nvgpu.NV_OK {
				client.objsMu.Unlock()
				return suspended, fmt.Errorf("failed to free multicast object %v:%v host-side: errno=%v status=%#x", client.handle, mcObj.handle, err, status)
			}
			// Substitute a plain allocation under the same handle so libcuda's
			// checkpoint content-save pass doesn't fail on the freed handle.
			var substitute *rmAllocObject
			if dummy == nil {
				log.Warningf("nvproxy: no bound memory to clone a substitute for multicast object %v:%v; cuda-checkpoint may fail on its handle", client.handle, mcObj.handle)
			} else if err := dummy.restore(); err != nil {
				client.objsMu.Unlock()
				return suspended, fmt.Errorf("failed to allocate substitute for multicast object %v:%v: %w", client.handle, mcObj.handle, err)
			} else {
				substitute = &rmAllocObject{params: *dummy}
			}
			nvp.suspendedMulticast = append(nvp.suspendedMulticast, &suspendedMulticastObject{
				fd:           fd,
				oldClient:    client.handle,
				handle:       mcObj.handle,
				params:       mcObj.params,
				attachedGPUs: mcObj.attachedGPUs,
				attachedMems: mcObj.attachedMems,
			})
			deferReleases = append(deferReleases, nvp.objFree(ctx, client, mcObj.handle)...)
			if substitute != nil {
				// Track the substitute in the graph so nvproxy bookkeeping
				// matches host reality; replay recognizes and frees it by its
				// non-multicast class.
				nvp.objAdd(ctx, client, mcObj.handle, dummy.ioctlParams.HClass, substitute, dummy.ioctlParams.HObjectParent)
			}
			suspended++
			log.Infof("nvproxy: suspended multicast object %v:%v (%d GPU attach(es), %d mem attach(es); substitute=%t)", client.handle, mcObj.handle, len(mcObj.attachedGPUs), len(mcObj.attachedMems), substitute != nil)
		}
		client.objsMu.Unlock()
		for _, release := range deferReleases {
			release()
		}
	}
	return suspended, nil
}

// ReplayMulticastObjects replays multicast objects suspended by
// SuspendMulticastObjects. It must be called after the post-restore
// cuda-checkpoint toggle (which recreates each CUDA process's root client,
// under a new handle), while sandbox tasks are still frozen.
//
// It is self-adapting: if the toggle already recreated a multicast object
// (same handle, same class) under the remapped client, replay of that object
// is skipped. This distinguishes "libcuda's restore path recreates multicast
// itself" from "nvproxy must replay" and logs which case occurred.
func ReplayMulticastObjects(ctx goContext.Context, vfsObj *vfs.VirtualFilesystem) error {
	nvp := nvproxyFromVFS(vfsObj)
	if nvp == nil {
		return nil
	}
	return nvp.replayMulticastObjects(ctx)
}

func (nvp *nvproxy) replayMulticastObjects(ctx goContext.Context) error {
	suspended := nvp.suspendedMulticast
	if len(suspended) == 0 {
		return nil
	}
	nvp.suspendedMulticast = nil

	for _, s := range suspended {
		newClientH, err := nvp.remapClientForFD(s.fd, s.oldClient)
		if err != nil {
			return fmt.Errorf("multicast object %v (old client %v): %w", s.handle, s.oldClient, err)
		}

		// If the toggle already recreated the object, don't replay.
		nvp.clientsMu.RLock()
		newClient := nvp.clients[newClientH]
		nvp.clientsMu.RUnlock()
		if newClient == nil {
			return fmt.Errorf("multicast object %v: remapped client %v disappeared", s.handle, newClientH)
		}
		newClient.objsMu.Lock()
		existing, exists := newClient.resources[s.handle]
		var deferReleases []func()
		if exists {
			if existing.class == nvgpu.NV_MEMORY_MULTICAST_FABRIC {
				// libcuda's restore toggle recreated the multicast object
				// itself; nothing to replay.
				newClient.objsMu.Unlock()
				log.Infof("nvproxy: multicast object %v:%v was recreated by the restore toggle; skipping nvproxy replay", newClientH, s.handle)
				continue
			}
			// The toggle recreated the SUBSTITUTE allocation (class %v) that
			// suspend planted under this handle; free it to make room for the
			// real multicast object.
			log.Infof("nvproxy: freeing substitute object %v:%v (class %v) before multicast replay", newClientH, s.handle, existing.class)
			status, err := freeObjectOnHost(s.fd.hostFD, newClientH, existing.parent, s.handle)
			if err != nil || status != nvgpu.NV_OK {
				newClient.objsMu.Unlock()
				return fmt.Errorf("failed to free substitute object %v:%v: errno=%v status=%#x", newClientH, s.handle, err, status)
			}
			deferReleases = nvp.objFree(context.Background(), newClient, s.handle)
		}
		newClient.objsMu.Unlock()
		for _, release := range deferReleases {
			release()
		}

		log.Infof("nvproxy: replaying multicast object %v (old client %v -> new client %v): alloc + %d GPU attach(es) + %d mem attach(es)", s.handle, s.oldClient, newClientH, len(s.attachedGPUs), len(s.attachedMems))
		if err := s.replay(ctx, nvp, newClientH, newClient); err != nil {
			return err
		}
	}
	return nil
}

// remapClientForFD maps a pre-checkpoint root client handle to the root
// client now live on the same frontendFD. The cuda-checkpoint restore toggle
// recreates each process's client through the same (reopened) device FD, but
// RM assigns it a new handle.
func (nvp *nvproxy) remapClientForFD(fd *frontendFD, oldClient nvgpu.Handle) (nvgpu.Handle, error) {
	nvp.clientsMu.RLock()
	defer nvp.clientsMu.RUnlock()
	var candidates []nvgpu.Handle
	for h, c := range nvp.clients {
		if c.params.fd == fd && !c.released {
			candidates = append(candidates, h)
		}
	}
	switch len(candidates) {
	case 1:
		return candidates[0], nil
	case 0:
		return nvgpu.Handle{}, fmt.Errorf("no live client on the object's frontend FD (toggle did not recreate the client?)")
	default:
		// Ambiguous. Prefer the old handle if it exists (client survived).
		for _, h := range candidates {
			if h == oldClient {
				return h, nil
			}
		}
		return nvgpu.Handle{}, fmt.Errorf("ambiguous client remap: %d live clients on the object's frontend FD: %v", len(candidates), candidates)
	}
}

// replay recreates the multicast object under newClientH: alloc with the
// original object handle, then ATTACH_GPU and ATTACH_MEM replays. On success
// the object is re-registered in the object graph so that future controls
// keep being recorded and future checkpoints can suspend it again.
func (s *suspendedMulticastObject) replay(ctx goContext.Context, nvp *nvproxy, newClientH nvgpu.Handle, newClient *rootClient) error {
	// Replay the allocation with remapped client handles. The 00FD object's
	// parent is the root client itself.
	params := s.params
	params.ioctlParams.HRoot = newClientH
	if params.ioctlParams.HObjectParent == s.oldClient {
		params.ioctlParams.HObjectParent = newClientH
	}
	params.ioctlParams.HObjectNew = s.handle
	if err := params.restore(); err != nil {
		return fmt.Errorf("failed to replay multicast alloc %v:%v: %w", newClientH, s.handle, err)
	}

	mcObj := &multicastFabricObject{
		params:       params,
		attachedGPUs: s.attachedGPUs,
		attachedMems: s.attachedMems,
	}
	// Register in the graph first so the object's client/handle fields are
	// initialized for the attach replays below.
	newClient.objsMu.Lock()
	nvp.objAdd(context.Background(), newClient, s.handle, nvgpu.NV_MEMORY_MULTICAST_FABRIC, mcObj, params.ioctlParams.HObjectParent)
	for _, am := range s.attachedMems {
		newClient.objAddDep(s.handle, am.params.HMemory)
	}
	newClient.objsMu.Unlock()

	if err := mcObj.restoreAttachGPUs(ctx); err != nil {
		return fmt.Errorf("multicast object %v:%v: %w", newClientH, s.handle, err)
	}
	if err := mcObj.restoreAttachMems(ctx); err != nil {
		return fmt.Errorf("multicast object %v:%v: %w", newClientH, s.handle, err)
	}
	return nil
}

// ctrlMemoryMulticastFabricAttachMem proxies NV00FD_CTRL_CMD_ATTACH_MEM like
// rmControlSimple, and additionally records the successful attach on the
// target multicast object for replay at restore time. It also records a
// dependency of the multicast object on the attached memory object, so that
// the memory is restored first and a free of the memory cascades to the
// multicast object, mirroring the driver.
func ctrlMemoryMulticastFabricAttachMem(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	var ctrlParams nvgpu.NV00FD_CTRL_ATTACH_MEM_PARAMS
	if ctrlParams.SizeBytes() != int(ioctlParams.ParamsSize) {
		return 0, linuxerr.EINVAL
	}
	if _, err := ctrlParams.CopyIn(fi.t, addrFromP64(ioctlParams.Params)); err != nil {
		return 0, err
	}
	n, err := rmControlInvoke(fi, ioctlParams, &ctrlParams)
	if err != nil || ioctlParams.Status != nvgpu.NV_OK {
		return n, err
	}
	if _, err := ctrlParams.CopyOut(fi.t, addrFromP64(ioctlParams.Params)); err != nil {
		return n, err
	}

	nvp := fi.fd.dev.nvp
	if mcObj, unlock := nvp.getMulticastObjectWithLock(fi.ctx, ioctlParams.HClient, ioctlParams.HObject); mcObj != nil {
		mcObj.recordAttachMem(ctrlParams)
		mcObj.client.objAddDep(ioctlParams.HObject, ctrlParams.HMemory)
		unlock()
	}
	return n, nil
}

// ctrlMemoryMulticastFabricDetachMem proxies NV00FD_CTRL_CMD_DETACH_MEM and
// removes the matching recorded attach.
func ctrlMemoryMulticastFabricDetachMem(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	var ctrlParams nvgpu.NV00FD_CTRL_DETACH_MEM_PARAMS
	if ctrlParams.SizeBytes() != int(ioctlParams.ParamsSize) {
		return 0, linuxerr.EINVAL
	}
	if _, err := ctrlParams.CopyIn(fi.t, addrFromP64(ioctlParams.Params)); err != nil {
		return 0, err
	}
	n, err := rmControlInvoke(fi, ioctlParams, &ctrlParams)
	if err != nil || ioctlParams.Status != nvgpu.NV_OK {
		return n, err
	}
	if _, err := ctrlParams.CopyOut(fi.t, addrFromP64(ioctlParams.Params)); err != nil {
		return n, err
	}

	nvp := fi.fd.dev.nvp
	if mcObj, unlock := nvp.getMulticastObjectWithLock(fi.ctx, ioctlParams.HClient, ioctlParams.HObject); mcObj != nil {
		mcObj.removeAttachMem(ctrlParams)
		// Note: the dependency on the memory object is intentionally retained;
		// dependencies record freeing order, and the driver keeps the
		// association until the objects are freed.
		unlock()
	}
	return n, nil
}
