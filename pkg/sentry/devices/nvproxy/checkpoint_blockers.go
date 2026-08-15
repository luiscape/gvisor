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
	"sort"
	"strings"

	"gvisor.dev/gvisor/pkg/abi/nvgpu"
	"gvisor.dev/gvisor/pkg/context"
	"gvisor.dev/gvisor/pkg/errors/linuxerr"
	"gvisor.dev/gvisor/pkg/sentry/kernel"
	"gvisor.dev/gvisor/pkg/sentry/vfs"
)

// This file implements the checkpoint-blocker inventory: an accounting of
// live driver resources that `cuda-checkpoint` cannot serialize, so that a
// sandbox checkpoint can be refused (or preceded by a teardown) instead of
// hanging inside `cuda-checkpoint --action checkpoint`. The blockers are
// fabric/multicast RM objects allocated via cuMemExportToShareableHandle()
// and NVLS multicast (e.g. NCCL NVLS and PyTorch symmetric memory), and FDs
// holding exported RM objects (live CUDA IPC).

// blockerClasses are the RM object classes reported as checkpoint blockers.
var blockerClasses = map[nvgpu.ClassID]struct{}{
	nvgpu.NV_MEMORY_FABRIC:              {},
	nvgpu.NV_MEMORY_MULTICAST_FABRIC:    {},
	nvgpu.NV_MEMORY_FABRIC_IMPORTED_REF: {},
	nvgpu.NV_MEMORY_EXPORT:              {},
}

// BlockerKind labels the kind of resource blocking a CUDA checkpoint.
type BlockerKind string

// BlockerKind values.
const (
	// BlockerKindMulticast is a live NV_MEMORY_MULTICAST_FABRIC object.
	BlockerKindMulticast BlockerKind = "multicast"
	// BlockerKindFabric is a live NV_MEMORY_FABRIC object.
	BlockerKindFabric BlockerKind = "fabric"
	// BlockerKindFabricImport is a live NV_MEMORY_FABRIC_IMPORTED_REF object.
	BlockerKindFabricImport BlockerKind = "fabric-import"
	// BlockerKindExport is a live NV_MEMORY_EXPORT object.
	BlockerKindExport BlockerKind = "export"
	// BlockerKindExportedFD is an open FD holding an exported RM object
	// (NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT(S)_TO_FD).
	BlockerKindExportedFD BlockerKind = "exported-fd"
)

// CheckpointBlocker describes a live driver resource that blocks
// cuda-checkpoint from checkpointing the sandbox: multicast/fabric RM objects
// (which cuda-checkpoint cannot serialize) and FDs holding exported RM
// objects (live CUDA IPC).
type CheckpointBlocker struct {
	ClientHandle nvgpu.Handle
	ObjectHandle nvgpu.Handle
	Class        nvgpu.ClassID
	TaskID       int32 // thread group ID of the allocating task, or 0
	Kind         BlockerKind
}

// String implements fmt.Stringer.
func (b CheckpointBlocker) String() string {
	return fmt.Sprintf("{client %v object %v class %v task %d kind %q}", b.ClientHandle, b.ObjectHandle, b.Class, b.TaskID, b.Kind)
}

func blockerKind(class nvgpu.ClassID) BlockerKind {
	switch class {
	case nvgpu.NV_MEMORY_MULTICAST_FABRIC:
		return BlockerKindMulticast
	case nvgpu.NV_MEMORY_FABRIC_IMPORTED_REF:
		return BlockerKindFabricImport
	case nvgpu.NV_MEMORY_EXPORT:
		return BlockerKindExport
	default:
		return BlockerKindFabric
	}
}

// CheckpointBlockers returns the current checkpoint blockers tracked by the
// nvproxy registered in vfsObj, or nil if there are none (or nvproxy is not
// registered). It does not modify any state.
func CheckpointBlockers(vfsObj *vfs.VirtualFilesystem) []CheckpointBlocker {
	nvp := nvproxyFromVFS(vfsObj)
	if nvp == nil {
		return nil
	}
	return nvp.checkpointBlockers()
}

func (nvp *nvproxy) checkpointBlockers() []CheckpointBlocker {
	var out []CheckpointBlocker

	// Fabric/multicast RM objects.
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
				if _, ok := blockerClasses[o.class]; ok {
					out = append(out, CheckpointBlocker{
						ClientHandle: client.handle,
						ObjectHandle: h,
						Class:        o.class,
						TaskID:       o.taskID,
						Kind:         blockerKind(o.class),
					})
				}
			}
		}
		client.objsMu.Unlock()
	}

	// FDs holding exported RM objects (NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD).
	// The blocker disappears when the FD is closed (removed from frontendFDs).
	nvp.fdsMu.Lock()
	for fd := range nvp.frontendFDs {
		if eo := fd.exportedObj; eo != nil {
			out = append(out, CheckpointBlocker{
				ClientHandle: eo.client,
				ObjectHandle: eo.object,
				Class:        eo.class,
				TaskID:       eo.taskID,
				Kind:         BlockerKindExportedFD,
			})
		}
	}
	nvp.fdsMu.Unlock()

	sort.Slice(out, func(i, j int) bool {
		if out[i].ClientHandle.Val != out[j].ClientHandle.Val {
			return out[i].ClientHandle.Val < out[j].ClientHandle.Val
		}
		if out[i].ObjectHandle.Val != out[j].ObjectHandle.Val {
			return out[i].ObjectHandle.Val < out[j].ObjectHandle.Val
		}
		return out[i].Kind < out[j].Kind
	})
	return out
}

// FormatBlockersByClient formats blockers as one line per client, e.g.
// "task 42 (client 0xc1d00922): 3 multicast, 1 exported-fd". This is the
// per-rank message required by the checkpoint gate.
func FormatBlockersByClient(blockers []CheckpointBlocker) string {
	type key struct {
		client nvgpu.Handle
		task   int32
	}
	counts := make(map[key]map[BlockerKind]int)
	var order []key
	for _, b := range blockers {
		k := key{b.ClientHandle, b.TaskID}
		if _, ok := counts[k]; !ok {
			counts[k] = make(map[BlockerKind]int)
			order = append(order, k)
		}
		counts[k][b.Kind]++
	}
	var lines []string
	for _, k := range order {
		kinds := make([]string, 0, len(counts[k]))
		for kind := range counts[k] {
			kinds = append(kinds, string(kind))
		}
		sort.Strings(kinds)
		parts := make([]string, 0, len(kinds))
		for _, kind := range kinds {
			parts = append(parts, fmt.Sprintf("%d %s", counts[k][BlockerKind(kind)], kind))
		}
		lines = append(lines, fmt.Sprintf("task %d (client %v): %s", k.task, k.client, strings.Join(parts, ", ")))
	}
	return strings.Join(lines, "; ")
}

// exportedObjInfo records that an RM object was exported into a frontendFD
// via NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT(S)_TO_FD.
//
// +stateify savable
type exportedObjInfo struct {
	client nvgpu.Handle
	object nvgpu.Handle
	class  nvgpu.ClassID
	taskID int32
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
	if exp == nil {
		return ""
	}
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
		if ioctlParams.Status != nvgpu.NV_OK || ctrlParams.NumObjects == 0 {
			return
		}
		// Batch semantics (see ctrl0000unix.h): each call (re)writes NumObjects
		// slots starting at Index, and a zero handle unexports its slot. The
		// accounting here is deliberately simpler — the fd is attributed to
		// Objects[0] of the most recent non-empty batch, and cleared only when
		// an all-zero batch covers slot 0 (a partial-slot unexport must not
		// clear the mark: under-reporting a live export is the dangerous
		// direction for the blocker gate) — because libcuda exports a single
		// object per fd. Log when that assumption is visibly exceeded.
		if ctrlParams.NumObjects > 1 || ctrlParams.Index != 0 {
			fi.ctx.Infof("nvproxy: EXPORT_OBJECTS_TO_FD with NumObjects=%d Index=%d; blocker accounting attributes the fd to the first object only", ctrlParams.NumObjects, ctrlParams.Index)
		}
		allZero := true
		for i := uint16(0); i < ctrlParams.NumObjects && int(i) < len(ctrlParams.Objects); i++ {
			if ctrlParams.Objects[i].Val != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			if ctrlParams.Index == 0 {
				nvp := fi.fd.dev.nvp
				nvp.fdsMu.Lock()
				ctlFile.exportedObj = nil
				nvp.fdsMu.Unlock()
			}
			return
		}
		markExportedObjFD(fi, ctlFile, ioctlParams.HClient, ctrlParams.Objects[0])
	})
}

// markExportedObjFD marks fd as holding an RM object exported from the given
// client (attributed to objectH, which may be a zero handle if unknown).
func markExportedObjFD(fi *frontendIoctlState, fd *frontendFD, clientH, objectH nvgpu.Handle) {
	var taskID int32
	if t := kernel.TaskFromContext(fi.ctx); t != nil {
		taskID = int32(t.ThreadGroup().ID())
	}
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
	fd.exportedObj = &exportedObjInfo{
		client: clientH,
		object: objectH,
		class:  class,
		taskID: taskID,
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
		markExportedObjFD(fi, ctlFile, ioctlParams.HClient, objectH)
	})
}
