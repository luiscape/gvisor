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
	"encoding/binary"
	"fmt"

	"gvisor.dev/gvisor/pkg/abi/nvgpu"
	"gvisor.dev/gvisor/pkg/context"
	"gvisor.dev/gvisor/pkg/errors/linuxerr"
)

// exportedObjInfo records that an RM object was exported into a frontendFD
// via NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT(S)_TO_FD.
//
// +stateify savable
type exportedObjInfo struct {
	client nvgpu.Handle
	object nvgpu.Handle
	class  nvgpu.ClassID
}

// exportedObjInfoLocked returns the exported object in the lowest slot of fd,
// which is the fd's identity for the fdinfo oracle (libcuda exports one
// object per fd, in slot 0).
//
// Preconditions: nvp.fdsMu must be locked.
func (fd *frontendFD) exportedObjInfoLocked() (exportedObjInfo, bool) {
	var best exportedObjInfo
	bestSlot, found := uint16(0), false
	for slot, eo := range fd.exportedObjs {
		if !found || slot < bestSlot {
			best, bestSlot, found = eo, slot, true
		}
	}
	return best, found
}

// ProcFDInfoExtra implements proc's procFDInfoExtra (duck-typed): expose the
// exported RM object's identity in /proc/[pid]/fdinfo/[fd], analogous to
// Linux's dmabuf show_fdinfo.
//
// This is the identity oracle for CUDA IPC fds: every fd from
// cuMemExportToShareableHandle is an open of /dev/nvidiactl, so fstat gives
// all of them the device node's single inode, and SCM_RIGHTS recipients have
// no way to tell which exported allocation a received fd refers to. The
// (client, object) pair recorded at export time IS that identity — RM client
// handles are globally unique on the host — and because SCM_RIGHTS passes
// the same FileDescription, exporter and importers read identical lines.
// Userspace (e.g. a checkpoint interposer that must re-import the same
// allocation after restore) parses the nvproxy_exported_object line; its
// format is a contract, locked by TestProcFDInfoExtraFormat.
func (fd *frontendFD) ProcFDInfoExtra(ctx context.Context) string {
	nvp := fd.dev.nvp
	nvp.fdsMu.Lock()
	exp, ok := fd.exportedObjInfoLocked()
	nvp.fdsMu.Unlock()
	if !ok {
		return ""
	}
	return procFDInfoExportedObjectLine(exp)
}

// procFDInfoExportedObjectLine formats the fdinfo oracle line; see
// ProcFDInfoExtra.
func procFDInfoExportedObjectLine(exp exportedObjInfo) string {
	return fmt.Sprintf("nvproxy_exported_object:\tclient=%#x object=%#x class=%#x\n",
		exp.client.Val, exp.object.Val, uint32(exp.class))
}

// ctrlExportToFDInvoke performs the frontend-FD-translating control sequence
// shared by the export-to-fd handlers, mirroring ctrlHasFrontendFD: CopyIn,
// translate the params' FD to the corresponding host FD, invoke, restore the
// application FD value, CopyOut. If the invoke succeeded, it calls post with
// the populated params and the destination frontendFD (with a reference
// held); post is responsible for checking ioctlParams.Status.
func ctrlExportToFDInvoke[Params any, PtrParams hasFrontendFDPtr[Params]](fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS, post func(params PtrParams, ctlFile *frontendFD)) (uintptr, error) {
	var ctrlParamsValue Params
	ctrlParams := PtrParams(&ctrlParamsValue)
	if ctrlParams.SizeBytes() != int(ioctlParams.ParamsSize) {
		return 0, linuxerr.EINVAL
	}
	if _, err := ctrlParams.CopyIn(fi.t, addrFromP64(ioctlParams.Params)); err != nil {
		return 0, err
	}

	origFD := ctrlParams.GetFrontendFD()
	ctlFileGeneric, _ := fi.t.FDTable().Get(origFD)
	if ctlFileGeneric == nil {
		return 0, linuxerr.EINVAL
	}
	defer ctlFileGeneric.DecRef(fi.ctx)
	ctlFile, ok := ctlFileGeneric.Impl().(*frontendFD)
	if !ok {
		return 0, linuxerr.EINVAL
	}

	ctrlParams.SetFrontendFD(ctlFile.hostFD)
	n, err := rmControlInvoke(fi, ioctlParams, ctrlParams)
	ctrlParams.SetFrontendFD(origFD)
	if err != nil {
		return n, err
	}
	// post runs before CopyOut: the accounting must reflect host-side reality
	// even if the copy-out to the application faults afterwards (an unmarked
	// host-exported fd would silently under-report the blocker inventory).
	post(ctrlParams, ctlFile)
	if _, cerr := ctrlParams.CopyOut(fi.t, addrFromP64(ioctlParams.Params)); cerr != nil {
		return n, cerr
	}
	return n, nil
}

// ctrlClientExportObjectsToFD proxies
// NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECTS_TO_FD (the batched form used by
// current libcuda, e.g. for cuMemExportToShareableHandle) like
// ctrlHasFrontendFD, and additionally marks the destination frontendFD as
// holding exported RM objects, so that it is reported as a checkpoint blocker
// until closed.
func ctrlClientExportObjectsToFD(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	return ctrlExportToFDInvoke(fi, ioctlParams, func(ctrlParams *nvgpu.NV0000_CTRL_OS_UNIX_EXPORT_OBJECTS_TO_FD_PARAMS, ctlFile *frontendFD) {
		if ioctlParams.Status != nvgpu.NV_OK {
			n := int(ctrlParams.NumObjects)
			if n > len(ctrlParams.Objects) {
				n = len(ctrlParams.Objects)
			}
			fi.ctx.Debugf("nvproxy: EXPORT_OBJECTS_TO_FD failed: client %v objects %v index %d status %#x", ioctlParams.HClient, ctrlParams.Objects[:n], ctrlParams.Index, ioctlParams.Status)
			return
		}
		// Batch semantics (see ctrl0000unix.h): each call (re)writes NumObjects
		// slots starting at Index, and a zero handle unexports its slot.
		// Mirror that exactly, so the blocker inventory reports every live
		// export (libcuda exports one object per fd, in slot 0, but nothing
		// prevents another client from filling several slots).
		n := int(ctrlParams.NumObjects)
		if n > len(ctrlParams.Objects) {
			n = len(ctrlParams.Objects)
		}
		for i := 0; i < n; i++ {
			slot := ctrlParams.Index + uint16(i)
			if h := ctrlParams.Objects[i]; h.Val != 0 {
				markExportedObjFD(fi, ctlFile, slot, ioctlParams.HClient, h)
			} else {
				unmarkExportedObjFD(fi, ctlFile, slot)
			}
		}
	})
}

// markExportedObjFD records that slot of fd holds an RM object exported from
// the given client (attributed to objectH, which may be a zero handle if
// unknown).
func markExportedObjFD(fi *frontendIoctlState, fd *frontendFD, slot uint16, clientH, objectH nvgpu.Handle) {
	var class nvgpu.ClassID
	nvp := fi.fd.dev.nvp
	if objectH.Val != 0 {
		if client, unlock := nvp.getClientWithLock(fi.ctx, clientH); client != nil {
			if obj, ok := client.resources[objectH]; ok {
				class = obj.class
			}
			unlock()
		}
	}
	nvp.fdsMu.Lock()
	if fd.exportedObjs == nil {
		fd.exportedObjs = make(map[uint16]exportedObjInfo, 1)
	}
	fd.exportedObjs[slot] = exportedObjInfo{
		client: clientH,
		object: objectH,
		class:  class,
	}
	nvp.fdsMu.Unlock()
}

// unmarkExportedObjFD records that slot of fd no longer holds an exported
// object.
func unmarkExportedObjFD(fi *frontendIoctlState, fd *frontendFD, slot uint16) {
	nvp := fi.fd.dev.nvp
	nvp.fdsMu.Lock()
	delete(fd.exportedObjs, slot)
	if len(fd.exportedObjs) == 0 {
		fd.exportedObjs = nil
	}
	nvp.fdsMu.Unlock()
}

// ctrlClientExportObjectToFD proxies NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD
// like ctrlHasFrontendFD, and additionally marks the destination frontendFD
// as holding an exported RM object, so that it is reported as a checkpoint
// blocker until closed.
func ctrlClientExportObjectToFD(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	return ctrlExportToFDInvoke(fi, ioctlParams, func(ctrlParams *nvgpu.NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_PARAMS, ctlFile *frontendFD) {
		if ioctlParams.Status != nvgpu.NV_OK {
			// A failed export is the signature of userspace presenting a
			// stale object handle (e.g. a cached fabric registration after
			// a restore); log the handle so the failure is attributable.
			var objectH nvgpu.Handle
			if ctrlParams.Object.Type == nvgpu.NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM {
				objectH.Val = binary.LittleEndian.Uint32(ctrlParams.Object.Data[8:12])
			}
			fi.ctx.Debugf("nvproxy: EXPORT_OBJECT_TO_FD failed: client %v object %v flags %#x status %#x", ioctlParams.HClient, objectH, ctrlParams.Flags, ioctlParams.Status)
			return
		}
		// With EMPTY_FD the export succeeds but associates no object with the
		// fd (objects are attached later, e.g. by EXPORT_OBJECTS_TO_FD), so
		// there is nothing to report as a blocker yet.
		if ctrlParams.Flags&nvgpu.NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_FLAGS_EMPTY_FD != 0 {
			return
		}
		// For type NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM, the union is
		// struct {hDevice, hParent, hObject}; record hObject for attribution.
		var objectH nvgpu.Handle
		if ctrlParams.Object.Type == nvgpu.NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM {
			objectH.Val = binary.LittleEndian.Uint32(ctrlParams.Object.Data[8:12])
		}
		markExportedObjFD(fi, ctlFile, 0 /* slot */, ioctlParams.HClient, objectH)
	})
}
