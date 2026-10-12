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
	if !nvp.trackExports {
		return ""
	}
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

// exportRecorder records what an export or import did, for
// ctrlExportToFDInvoke. It is called only if the host call succeeded (it must
// still check ioctlParams.Status), with client's objsMu locked (client is nil
// if the params' client is unknown or released), src the params frontendFD's
// exported object as of before the call, and the populated params. It may
// return a function to run once objsMu is released, since fdsMu is ordered
// before it.
type exportRecorder[PtrParams any] func(client *rootClient, src exportedObjInfo, params PtrParams, ctlFile *frontendFD) func()

// ctrlExportToFDInvoke performs the frontend-FD-translating control sequence
// shared by the export and import handlers, mirroring ctrlHasFrontendFD: CopyIn,
// translate the params' FD to the corresponding host FD, invoke, restore the
// application FD value, CopyOut. Unlike it, the client's objsMu is held across
// the host call, as rmAllocInvoke does for allocations, so that a concurrent
// free cannot run between the host's import or export and record; and record
// runs before anything is copied back to the application, so that an object
// the host has imported is recorded even if the copy-out faults (an import
// nvproxy did not know of would escape CheckpointBlockers).
//
// Without Options.TrackExports, it is exactly ctrlHasFrontendFD.
func ctrlExportToFDInvoke[Params any, PtrParams hasFrontendFDPtr[Params]](fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS, record exportRecorder[PtrParams]) (uintptr, error) {
	if !fi.fd.dev.nvp.trackExports {
		return ctrlHasFrontendFD[Params, PtrParams](fi, ioctlParams)
	}
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

	nvp := fi.fd.dev.nvp
	nvp.fdsMu.Lock()
	src := ctlFile.exportedObj
	nvp.fdsMu.Unlock()

	client, unlock := nvp.getClientWithLock(fi.ctx, ioctlParams.HClient)
	ctrlParams.SetFrontendFD(ctlFile.hostFD)
	n, err := rmControlInvokeHost(fi, ioctlParams, ctrlParams)
	ctrlParams.SetFrontendFD(origFD)
	var after func()
	if err == nil {
		after = record(client, src, ctrlParams, ctlFile)
	}
	if client != nil {
		unlock()
	}
	if after != nil {
		after()
	}
	if err != nil {
		return n, err
	}
	if _, err := ioctlParams.CopyOut(fi.t, fi.ioctlParamsAddr); err != nil {
		return n, err
	}
	if _, err := ctrlParams.CopyOut(fi.t, addrFromP64(ioctlParams.Params)); err != nil {
		return n, err
	}
	return n, nil
}

// ctrlClientExportObjectsToFD proxies
// NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECTS_TO_FD (the batched form used by
// current libcuda, e.g. for cuMemExportToShareableHandle) like
// ctrlHasFrontendFD, and records the object exported into slot 0 of the
// destination frontendFD.
func ctrlClientExportObjectsToFD(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	return ctrlExportToFDInvoke(fi, ioctlParams, func(client *rootClient, _ exportedObjInfo, ctrlParams *nvgpu.NV0000_CTRL_OS_UNIX_EXPORT_OBJECTS_TO_FD_PARAMS, ctlFile *frontendFD) func() {
		// Each call writes NumObjects slots starting at Index, and a zero
		// handle clears its slot. libcuda exports an allocation into slot 0,
		// and sometimes companion objects into later slots.
		if ioctlParams.Status != nvgpu.NV_OK || ctrlParams.Index != 0 || ctrlParams.NumObjects == 0 {
			return nil
		}
		return setExportedObj(fi, ctlFile, exportedObjOf(client, ioctlParams.HClient, ctrlParams.Objects[0]))
	})
}

// exportedObjOf returns the identity of objectH, exported from clientH, whose
// objsMu must be locked if client (clientH's rootClient, or nil if unknown) is
// non-nil. A zero objectH returns the zero identity, which clears an fd's.
func exportedObjOf(client *rootClient, clientH, objectH nvgpu.Handle) exportedObjInfo {
	if objectH.Val == 0 {
		return exportedObjInfo{}
	}
	exp := exportedObjInfo{client: clientH, object: objectH}
	if client != nil {
		if obj, ok := client.resources[objectH]; ok {
			exp.class = obj.class
			exp.obj = obj
		}
	}
	return exp
}

// setExportedObj returns a function that records exp as fd's exported object.
// It takes fdsMu, so it must run with no client's objsMu locked.
func setExportedObj(fi *frontendIoctlState, fd *frontendFD, exp exportedObjInfo) func() {
	return func() {
		nvp := fi.fd.dev.nvp
		nvp.fdsMu.Lock()
		fd.exportedObj = exp
		nvp.fdsMu.Unlock()
	}
}

// ctrlClientExportObjectToFD proxies NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD
// like ctrlHasFrontendFD, and records the exported object as the destination
// frontendFD's.
func ctrlClientExportObjectToFD(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	return ctrlExportToFDInvoke(fi, ioctlParams, func(client *rootClient, _ exportedObjInfo, ctrlParams *nvgpu.NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_PARAMS, ctlFile *frontendFD) func() {
		// With EMPTY_FD no object is attached yet (EXPORT_OBJECTS_TO_FD adds
		// it). For type RM, the union is struct {hDevice, hParent, hObject}.
		if ioctlParams.Status != nvgpu.NV_OK ||
			ctrlParams.Flags&nvgpu.NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_FLAGS_EMPTY_FD != 0 ||
			ctrlParams.Object.Type != nvgpu.NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM {
			return nil
		}
		objectH := nvgpu.Handle{Val: binary.LittleEndian.Uint32(ctrlParams.Object.Data[8:12])}
		return setExportedObj(fi, ctlFile, exportedObjOf(client, ioctlParams.HClient, objectH))
	})
}

// importedObject tracks an object that a client imported from an exported fd,
// duped under one of its own objects. cuda-checkpoint cannot carry imports (a
// restore fails, and for multicast objects the checkpoint hangs), so they are
// checkpoint blockers, and like other dups they are not restorable.
type importedObject struct {
	object

	// src is the fd's exported object (see addImportedObj); src.obj is nil if
	// unknown.
	src exportedObjInfo

	// multicast is true for an imported multicast object.
	multicast bool
}

// Release implements objectImpl.Release.
func (o *importedObject) Release(ctx context.Context) func() {
	return nil
}

// addImportedObj records objectH, duped under parentH in client, as imported
// from slot index of an fd whose exported object is src. Objects from any
// slot are attributed to the allocation exported into slot 0 (see
// ctrlClientExportObjectsToFD).
//
// Precondition: client.objsMu must be locked, if client is non-nil.
func addImportedObj(fi *frontendIoctlState, client *rootClient, src exportedObjInfo, parentH, objectH nvgpu.Handle, index int, multicast bool) {
	if client == nil {
		return
	}
	var class nvgpu.ClassID
	if index == 0 {
		class = src.class
	}
	fi.fd.dev.nvp.objAdd(fi.ctx, client, objectH, class, &importedObject{src: src, multicast: multicast}, parentH)
}

// ctrlClientImportObjectFromFD proxies
// NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD like ctrlHasFrontendFD, and
// records the imported object.
func ctrlClientImportObjectFromFD(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	return ctrlExportToFDInvoke(fi, ioctlParams, func(client *rootClient, src exportedObjInfo, ctrlParams *nvgpu.NV0000_CTRL_OS_UNIX_IMPORT_OBJECT_FROM_FD_PARAMS, _ *frontendFD) func() {
		// For type RM, the union is struct {hDevice, hParent, hObject}.
		if ioctlParams.Status != nvgpu.NV_OK || ctrlParams.Object.Type != nvgpu.NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM {
			return nil
		}
		parentH := nvgpu.Handle{Val: binary.LittleEndian.Uint32(ctrlParams.Object.Data[4:8])}
		objectH := nvgpu.Handle{Val: binary.LittleEndian.Uint32(ctrlParams.Object.Data[8:12])}
		addImportedObj(fi, client, src, parentH, objectH, 0, false)
		return nil
	})
}

// ctrlClientImportObjectsFromFD proxies
// NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECTS_FROM_FD (used by current libcuda, e.g.
// for cuMemImportFromShareableHandle) like ctrlHasFrontendFD, and records the
// imported objects.
func ctrlClientImportObjectsFromFD(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	return ctrlExportToFDInvoke(fi, ioctlParams, func(client *rootClient, src exportedObjInfo, ctrlParams *nvgpu.NV0000_CTRL_OS_UNIX_IMPORT_OBJECTS_FROM_FD_PARAMS, _ *frontendFD) func() {
		if ioctlParams.Status != nvgpu.NV_OK {
			return nil
		}
		n := min(int(ctrlParams.NumObjects), len(ctrlParams.Objects))
		for i := 0; i < n; i++ {
			t := ctrlParams.ObjectTypes[i]
			if ctrlParams.Objects[i].Val == 0 || t == nvgpu.NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_TYPE_NONE {
				continue
			}
			addImportedObj(fi, client, src, ctrlParams.HParent, ctrlParams.Objects[i],
				int(ctrlParams.Index)+i, t == nvgpu.NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_TYPE_FABRIC_MC)
		}
		return nil
	})
}
