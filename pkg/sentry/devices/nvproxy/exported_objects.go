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

// ProcFDInfoExtra implements proc's procFDInfoExtra (duck-typed): expose the
// exported RM object's identity in /proc/[pid]/fdinfo/[fd], analogous to
// Linux's dmabuf show_fdinfo.
//
// Every fd from cuMemExportToShareableHandle is an open of /dev/nvidiactl, so
// fstat gives all of them the device node's single inode, and SCM_RIGHTS
// recipients have no way to tell which exported allocation a received fd
// refers to. The (client, object) pair recorded at export time is that
// identity (RM client handles are unique on the host), and because SCM_RIGHTS
// passes the same FileDescription, exporter and importers read identical
// lines. Userspace (e.g. a checkpoint interposer that must re-import the same
// allocation after restore) parses the nvproxy_exported_object line; its
// format is locked by TestProcFDInfoExtraFormat.
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

// procFDInfoExportedObjectLine formats the fdinfo line; see ProcFDInfoExtra.
func procFDInfoExportedObjectLine(exp exportedObjInfo) string {
	return fmt.Sprintf("nvproxy_exported_object:\tclient=%#x object=%#x class=%#x\n",
		exp.client.Val, exp.object.Val, uint32(exp.class))
}

// ctrlExportToFDInvoke proxies an export control like ctrlHasFrontendFD, and,
// if the host exported an object, records it as the destination frontendFD's.
// exported returns the handle of the object the populated params exported
// (zero to clear the fd's), or false if they leave the fd's unchanged.
func ctrlExportToFDInvoke[Params any, PtrParams hasFrontendFDPtr[Params]](fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS, exported func(params PtrParams) (nvgpu.Handle, bool)) (uintptr, error) {
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
	if ioctlParams.Status == nvgpu.NV_OK {
		if objectH, ok := exported(ctrlParams); ok {
			setExportedObj(fi.fd.dev.nvp, ctlFile, exportedObjOf(fi, ioctlParams.HClient, objectH))
		}
	}
	if _, err := ctrlParams.CopyOut(fi.t, addrFromP64(ioctlParams.Params)); err != nil {
		return n, err
	}
	return n, nil
}

// exportedObjOf returns the identity of objectH, exported from clientH. A zero
// objectH returns the zero identity, which clears an fd's. The class is
// informational, and zero if the object is not tracked.
func exportedObjOf(fi *frontendIoctlState, clientH, objectH nvgpu.Handle) exportedObjInfo {
	if objectH.Val == 0 {
		return exportedObjInfo{}
	}
	exp := exportedObjInfo{client: clientH, object: objectH}
	if client, unlock := fi.fd.dev.nvp.getClientWithLock(fi.ctx, clientH); client != nil {
		if obj, ok := client.resources[objectH]; ok {
			exp.class = obj.class
		}
		unlock()
	}
	return exp
}

// setExportedObj records exp as fd's exported object.
func setExportedObj(nvp *nvproxy, fd *frontendFD, exp exportedObjInfo) {
	nvp.fdsMu.Lock()
	fd.exportedObj = exp
	nvp.fdsMu.Unlock()
}

// ctrlClientExportObjectsToFD proxies
// NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECTS_TO_FD (the batched form used by
// current libcuda, e.g. for cuMemExportToShareableHandle), and records the
// object exported into slot 0 of the destination frontendFD.
func ctrlClientExportObjectsToFD(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	return ctrlExportToFDInvoke(fi, ioctlParams, func(ctrlParams *nvgpu.NV0000_CTRL_OS_UNIX_EXPORT_OBJECTS_TO_FD_PARAMS) (nvgpu.Handle, bool) {
		// Each call writes NumObjects slots starting at Index, and a zero
		// handle clears its slot. libcuda exports an allocation into slot 0,
		// and sometimes companion objects into later slots.
		if ctrlParams.Index != 0 || ctrlParams.NumObjects == 0 {
			return nvgpu.Handle{}, false
		}
		return ctrlParams.Objects[0], true
	})
}

// ctrlClientExportObjectToFD proxies NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD,
// and records the exported object as the destination frontendFD's.
func ctrlClientExportObjectToFD(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	return ctrlExportToFDInvoke(fi, ioctlParams, func(ctrlParams *nvgpu.NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_PARAMS) (nvgpu.Handle, bool) {
		// With EMPTY_FD no object is attached yet (EXPORT_OBJECTS_TO_FD adds
		// it). For type RM, the union is struct {hDevice, hParent, hObject}.
		if ctrlParams.Flags&nvgpu.NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_FLAGS_EMPTY_FD != 0 ||
			ctrlParams.Object.Type != nvgpu.NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM {
			return nvgpu.Handle{}, false
		}
		return nvgpu.Handle{Val: binary.LittleEndian.Uint32(ctrlParams.Object.Data[8:12])}, true
	})
}
