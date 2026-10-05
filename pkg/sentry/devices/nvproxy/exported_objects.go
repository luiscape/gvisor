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

	// obj is the exported object, if it was tracked. Unlike the handle, it
	// tells the object apart from a later one that reuses the handle.
	obj *object `state:"nosave"`
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
	exp := fd.exportedObj
	nvp.fdsMu.Unlock()
	if exp.object.Val == 0 {
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
// shared by the export and import handlers, mirroring ctrlHasFrontendFD: CopyIn,
// translate the params' FD to the corresponding host FD, invoke, restore the
// application FD value, CopyOut. If the invoke succeeded, it calls post with
// the populated params and the params' frontendFD (with a reference held);
// post is responsible for checking ioctlParams.Status.
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
	// post runs before CopyOut, so the recorded identity matches the host even
	// if the copy-out faults.
	post(ctrlParams, ctlFile)
	if _, cerr := ctrlParams.CopyOut(fi.t, addrFromP64(ioctlParams.Params)); cerr != nil {
		return n, cerr
	}
	return n, nil
}

// ctrlClientExportObjectsToFD proxies
// NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECTS_TO_FD (the batched form used by
// current libcuda, e.g. for cuMemExportToShareableHandle) like
// ctrlHasFrontendFD, and records the object exported into slot 0 of the
// destination frontendFD.
func ctrlClientExportObjectsToFD(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	return ctrlExportToFDInvoke(fi, ioctlParams, func(ctrlParams *nvgpu.NV0000_CTRL_OS_UNIX_EXPORT_OBJECTS_TO_FD_PARAMS, ctlFile *frontendFD) {
		// Each call writes NumObjects slots starting at Index, and a zero
		// handle clears its slot. libcuda exports one object per fd, in slot 0.
		if ioctlParams.Status == nvgpu.NV_OK && ctrlParams.Index == 0 && ctrlParams.NumObjects > 0 {
			setExportedObj(fi, ctlFile, ioctlParams.HClient, ctrlParams.Objects[0])
		}
	})
}

// setExportedObj records objectH, exported from clientH, as fd's exported
// object; a zero objectH clears it.
func setExportedObj(fi *frontendIoctlState, fd *frontendFD, clientH, objectH nvgpu.Handle) {
	var exp exportedObjInfo
	nvp := fi.fd.dev.nvp
	if objectH.Val != 0 {
		exp = exportedObjInfo{client: clientH, object: objectH}
		if client, unlock := nvp.getClientWithLock(fi.ctx, clientH); client != nil {
			if obj, ok := client.resources[objectH]; ok {
				exp.class = obj.class
				exp.obj = obj
			}
			unlock()
		}
	}
	nvp.fdsMu.Lock()
	fd.exportedObj = exp
	nvp.fdsMu.Unlock()
}

// ctrlClientExportObjectToFD proxies NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD
// like ctrlHasFrontendFD, and records the exported object as the destination
// frontendFD's.
func ctrlClientExportObjectToFD(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	return ctrlExportToFDInvoke(fi, ioctlParams, func(ctrlParams *nvgpu.NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_PARAMS, ctlFile *frontendFD) {
		// With EMPTY_FD no object is attached yet (EXPORT_OBJECTS_TO_FD adds
		// it). For type RM, the union is struct {hDevice, hParent, hObject}.
		if ioctlParams.Status != nvgpu.NV_OK ||
			ctrlParams.Flags&nvgpu.NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_FLAGS_EMPTY_FD != 0 ||
			ctrlParams.Object.Type != nvgpu.NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM {
			return
		}
		objectH := nvgpu.Handle{Val: binary.LittleEndian.Uint32(ctrlParams.Object.Data[8:12])}
		setExportedObj(fi, ctlFile, ioctlParams.HClient, objectH)
	})
}

// importedObject tracks an object that a client imported from an exported fd,
// duped under one of its own objects. cuda-checkpoint cannot carry imports (a
// restore fails, and for multicast objects the checkpoint hangs), so they are
// checkpoint blockers, and like other dups they are not restorable.
type importedObject struct {
	object

	// src is the exported object, as the fd recorded it; src.obj is nil if
	// unknown.
	src exportedObjInfo

	// multicast is true for an imported multicast object.
	multicast bool
}

// Release implements objectImpl.Release.
func (o *importedObject) Release(ctx context.Context) func() {
	return nil
}

// addImportedObj records objectH, duped under parentH in clientH, as
// imported from fd slot index.
func addImportedObj(fi *frontendIoctlState, fd *frontendFD, clientH, parentH, objectH nvgpu.Handle, index int, multicast bool) {
	nvp := fi.fd.dev.nvp
	var src exportedObjInfo
	if index == 0 {
		nvp.fdsMu.Lock()
		src = fd.exportedObj
		nvp.fdsMu.Unlock()
	}
	client, unlock := nvp.getClientWithLock(fi.ctx, clientH)
	if client == nil {
		return
	}
	nvp.objAdd(fi.ctx, client, objectH, src.class, &importedObject{src: src, multicast: multicast}, parentH)
	unlock()
}

// ctrlClientImportObjectFromFD proxies
// NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD like ctrlHasFrontendFD, and
// records the imported object.
func ctrlClientImportObjectFromFD(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	return ctrlExportToFDInvoke(fi, ioctlParams, func(ctrlParams *nvgpu.NV0000_CTRL_OS_UNIX_IMPORT_OBJECT_FROM_FD_PARAMS, ctlFile *frontendFD) {
		// For type RM, the union is struct {hDevice, hParent, hObject}.
		if ioctlParams.Status != nvgpu.NV_OK || ctrlParams.Object.Type != nvgpu.NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM {
			return
		}
		parentH := nvgpu.Handle{Val: binary.LittleEndian.Uint32(ctrlParams.Object.Data[4:8])}
		objectH := nvgpu.Handle{Val: binary.LittleEndian.Uint32(ctrlParams.Object.Data[8:12])}
		addImportedObj(fi, ctlFile, ioctlParams.HClient, parentH, objectH, 0, false)
	})
}

// ctrlClientImportObjectsFromFD proxies
// NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECTS_FROM_FD (used by current libcuda, e.g.
// for cuMemImportFromShareableHandle) like ctrlHasFrontendFD, and records the
// imported objects.
func ctrlClientImportObjectsFromFD(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	return ctrlExportToFDInvoke(fi, ioctlParams, func(ctrlParams *nvgpu.NV0000_CTRL_OS_UNIX_IMPORT_OBJECTS_FROM_FD_PARAMS, ctlFile *frontendFD) {
		if ioctlParams.Status != nvgpu.NV_OK {
			return
		}
		n := min(int(ctrlParams.NumObjects), len(ctrlParams.Objects))
		for i := 0; i < n; i++ {
			t := ctrlParams.ObjectTypes[i]
			if ctrlParams.Objects[i].Val == 0 || t == nvgpu.NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_TYPE_NONE {
				continue
			}
			addImportedObj(fi, ctlFile, ioctlParams.HClient, ctrlParams.HParent, ctrlParams.Objects[i],
				int(ctrlParams.Index)+i, t == nvgpu.NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_TYPE_FABRIC_MC)
		}
	})
}
