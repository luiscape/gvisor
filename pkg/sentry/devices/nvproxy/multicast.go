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
