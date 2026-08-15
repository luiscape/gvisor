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
	"time"

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

	// numGPUs is the allocation's numGpus: how many GPUs must ATTACH_GPU
	// before the object is usable (and before ATTACH_MEM can succeed).
	numGPUs uint32

	// attachedGPUs and attachedMems are recorded state-mutating controls,
	// replayed in order (all GPU attaches, then all mem attaches) after the
	// alloc replay. Both are protected by client.objsMu.
	attachedGPUs []mcastAttachGPU
	attachedMems []mcastAttachMem
}

func newMulticastFabricObject[Params any](fd *frontendFD, ioctlParams *nvgpu.NVOS64_PARAMETERS, rightsRequested nvgpu.RS_ACCESS_MASK, allocParams *Params) *multicastFabricObject {
	o := &multicastFabricObject{
		params: captureRmAllocParams(fd, ioctlParams, rightsRequested, allocParams),
	}
	if ng, ok := any(allocParams).(interface{ GetNumGPUs() uint32 }); ok {
		o.numGPUs = ng.GetNumGPUs()
	}
	return o
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
// offset into the multicast object), and returns the removed attach's
// HMemory and true, or false if no recorded attach matched.
//
// Matching on HSubDevice assumes one subdevice handle per GPU per client
// (the driver itself matches on the GPU behind the subdevice); a client
// holding multiple subdevice handles for one GPU and detaching via a
// different handle than it attached with would leave a stale record.
//
// Precondition: o.client.objsMu must be locked.
func (o *multicastFabricObject) removeAttachMem(ctx context.Context, params nvgpu.NV00FD_CTRL_DETACH_MEM_PARAMS) (nvgpu.Handle, bool) {
	for i, am := range o.attachedMems {
		if am.params.HSubDevice == params.HSubDevice && am.params.Offset == params.Offset {
			hMemory := am.params.HMemory
			o.attachedMems = append(o.attachedMems[:i], o.attachedMems[i+1:]...)
			return hMemory, true
		}
	}
	ctx.Warningf("nvproxy: DETACH_MEM on %v:%v (subdevice %v, offset %#x) does not match any recorded attach", o.client.handle, o.handle, params.HSubDevice, params.Offset)
	return nvgpu.Handle{}, false
}

// attachMemsReference reports whether any recorded ATTACH_MEM still
// references hMemory.
//
// Precondition: o.client.objsMu must be locked.
func (o *multicastFabricObject) attachMemsReference(hMemory nvgpu.Handle) bool {
	for _, am := range o.attachedMems {
		if am.params.HMemory == hMemory {
			return true
		}
	}
	return false
}

// checkReplayable returns an error if o's recorded attach controls can no
// longer be replayed, i.e. if a referenced object has been freed. The driver
// dups hMemory internally on ATTACH_MEM, so the application may legally free
// its handle while the attachment lives on -- but then no faithful replay
// exists, and the save must fail rather than the restore.
//
// Precondition: the sandbox is paused (called from beforeSave).
func (o *multicastFabricObject) checkReplayable() error {
	for _, ag := range o.attachedGPUs {
		if _, ok := o.client.resources[ag.params.HSubDevice]; !ok {
			return fmt.Errorf("recorded ATTACH_GPU references freed subdevice handle %v", ag.params.HSubDevice)
		}
	}
	for _, am := range o.attachedMems {
		if _, ok := o.client.resources[am.params.HMemory]; !ok {
			return fmt.Errorf("recorded ATTACH_MEM references freed memory handle %v", am.params.HMemory)
		}
		if _, ok := o.client.resources[am.params.HSubDevice]; !ok {
			return fmt.Errorf("recorded ATTACH_MEM references freed subdevice handle %v", am.params.HSubDevice)
		}
	}
	return nil
}

// Restore implements restorableObjectImpl.Restore.
//
// It replays the saved allocation, then the recorded ATTACH_GPU and
// ATTACH_MEM controls, in order.
//
// KNOWN LIMITATION (cross-client ordering): ATTACH_GPU does not block, but
// once every participating GPU has joined, ATTACH_MEM is completed by the
// driver for all of them together: on R610 an early ATTACH_MEM blocks in the
// driver until the team is complete (measured); other driver versions may
// instead fail it with NV_ERR_NOT_READY. afterLoad restores objects serially,
// so if several clients of ONE multicast object were restored through this
// path, the first client's ATTACH_MEM replay would wait for peers whose
// ATTACH_GPU has not been replayed yet -- a deadlock. restoreAttachMems
// therefore fails fast when the recorded state itself proves the team cannot
// be complete (fewer recorded ATTACH_GPUs than the allocation's numGpus),
// and a watchdog names the object loudly if the driver blocks anyway. This
// path is unreachable in the cuda-checkpoint flow (the interposer empties
// the multicast graph before save); it matters only for a plain sentry
// checkpoint taken with live multicast objects. Lifting the limitation
// requires splitting the replay into an all-clients ATTACH_GPU pass followed
// by an all-clients ATTACH_MEM pass.
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
// ATTACH_MEM blocks in the driver until all participating GPUs have attached
// (see Restore). The replay is a raw host ioctl that cannot be cancelled, so
// a cross-client ordering deadlock cannot be timed out -- but it must not be
// silent either. A watchdog names the stuck object loudly if the replay
// exceeds a generous bound.
func (o *multicastFabricObject) restoreAttachMems(ctx goContext.Context) error {
	if len(o.attachedMems) > 0 && uint32(len(o.attachedGPUs)) < o.numGPUs {
		// This object's replay alone cannot complete the team, and a serial
		// restore cannot guarantee that peer clients' ATTACH_GPU replays have
		// run (the topological sort imposes no cross-client order), so the
		// first ATTACH_MEM may block (R610) or fail (NV_ERR_NOT_READY); see
		// Restore's known limitation. Fail deterministically instead of
		// depending on restore order.
		return fmt.Errorf("multicast object records %d ATTACH_GPUs but requires %d GPUs; a serial restore cannot guarantee the peer attaches ATTACH_MEM would wait for", len(o.attachedGPUs), o.numGPUs)
	}
	for _, am := range o.attachedMems {
		params := am.params
		const attachMemWatchdog = 60 * time.Second
		wd := time.AfterFunc(attachMemWatchdog, func() {
			log.Warningf("nvproxy: ATTACH_MEM replay on multicast object %v:%v (hMemory %v) still blocked after %s; "+
				"likely waiting for a peer client's ATTACH_GPU that the serial restore has not replayed yet (see Restore's known limitation)",
				o.client.handle, o.handle, am.params.HMemory, attachMemWatchdog)
		})
		status, err := controlObjectOnHost(o.params.fd.hostFD, o.client.handle, o.handle, nvgpu.NV00FD_CTRL_CMD_ATTACH_MEM, &params)
		wd.Stop()
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
	if minor > nvgpu.NV_MINOR_DEVICE_NUMBER_REGULAR_MAX {
		return hostDevFD{}, fmt.Errorf("recorded device minor %d exceeds NV_MINOR_DEVICE_NUMBER_REGULAR_MAX", minor)
	}
	dev := nvp.regularDevs[minor]
	if dev == nil {
		return hostDevFD{}, fmt.Errorf("no /dev/nvidia%d device", minor)
	}
	hostFD := openHostDevFileForRestore(ctx, dev.basename(), nvp.useDevGofer, o.params.fd.containerName, unix.O_RDWR)
	return hostDevFD{fd: hostFD, close: func() { unix.Close(int(hostFD)) }}, nil
}

// ctrlMemoryMulticastFabricAttachMem proxies NV00FD_CTRL_CMD_ATTACH_MEM like
// rmControlSimple, and additionally records the successful attach on the
// target multicast object for replay at restore time. It also records
// restore-ordering-only dependencies of the multicast object on the attached
// memory and subdevice objects, so that the replay finds them already
// restored. The edges must not cascade frees: the driver dups hMemory
// internally (see
// src/nvidia/src/kernel/mem_mgr/mem_multicast_fabric.c:memorymulticastfabricCtrlAttachMem_IMPL()),
// so the application may legally free its hMemory while the attachment --
// and the multicast object -- live on.
func ctrlMemoryMulticastFabricAttachMem(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	var ctrlParams nvgpu.NV00FD_CTRL_ATTACH_MEM_PARAMS
	if ctrlParams.SizeBytes() != int(ioctlParams.ParamsSize) {
		return 0, linuxerr.EINVAL
	}
	if _, err := ctrlParams.CopyIn(fi.t, addrFromP64(ioctlParams.Params)); err != nil {
		return 0, err
	}
	n, err := rmControlInvoke(fi, ioctlParams, &ctrlParams)
	if err != nil {
		return n, err
	}
	// Record before CopyOut: the shadow graph must reflect host-side reality
	// even if the copy-out to the application faults afterwards (an
	// unrecorded host-side attach would replay an incomplete binding at
	// restore). This matches rmAllocInvoke, which records on host success
	// before any copy-out; the CopyOut itself is unconditional after a
	// successful syscall, like rmControlSimple.
	if ioctlParams.Status == nvgpu.NV_OK {
		nvp := fi.fd.dev.nvp
		// The lock is taken after the host invoke, diverging from the
		// rmDupObject/rmFree convention (lock held across the call):
		// ATTACH_MEM can block in the driver until all participating GPUs
		// have attached, and holding objsMu across it would deadlock a peer
		// thread of the same client issuing that ATTACH_GPU. The cost is a
		// benign race: a concurrent free+realloc of the same handle could
		// mis-attribute this record, which only a racy application can
		// trigger.
		if mcObj, unlock := nvp.getMulticastObjectWithLock(fi.ctx, ioctlParams.HClient, ioctlParams.HObject); mcObj != nil {
			mcObj.recordAttachMem(ctrlParams)
			mcObj.client.objAddRestoreDep(ioctlParams.HObject, ctrlParams.HMemory)
			mcObj.client.objAddRestoreDep(ioctlParams.HObject, ctrlParams.HSubDevice)
			unlock()
		}
	}
	if _, err := ctrlParams.CopyOut(fi.t, addrFromP64(ioctlParams.Params)); err != nil {
		return n, err
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
	if err != nil {
		return n, err
	}
	// Unrecord before CopyOut, for the same reason ATTACH_MEM records before
	// it: a stale record surviving a copy-out fault would replay an attach
	// the application detached.
	if ioctlParams.Status == nvgpu.NV_OK {
		nvp := fi.fd.dev.nvp
		if mcObj, unlock := nvp.getMulticastObjectWithLock(fi.ctx, ioctlParams.HClient, ioctlParams.HObject); mcObj != nil {
			if hMemory, ok := mcObj.removeAttachMem(fi.ctx, ctrlParams); ok && !mcObj.attachMemsReference(hMemory) {
				// No remaining recorded attach references the memory object,
				// so the replay no longer needs it restored first. The
				// subdevice restore-edge is deliberately retained: it is
				// redundant with the one ATTACH_GPU added (there is no
				// DETACH_GPU), and objFree cleans both sides when either
				// object goes away.
				mcObj.client.objRemoveRestoreDep(ioctlParams.HObject, hMemory)
			}
			unlock()
		}
	}
	if _, err := ctrlParams.CopyOut(fi.t, addrFromP64(ioctlParams.Params)); err != nil {
		return n, err
	}
	return n, nil
}
