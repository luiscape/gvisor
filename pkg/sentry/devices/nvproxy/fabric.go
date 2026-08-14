// Copyright 2025 The gVisor Authors.
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
	"fmt"
	"sort"
	"strings"

	"gvisor.dev/gvisor/pkg/abi/nvgpu"
	"gvisor.dev/gvisor/pkg/context"
	"gvisor.dev/gvisor/pkg/errors/linuxerr"
	"gvisor.dev/gvisor/pkg/sentry/kernel"
	"gvisor.dev/gvisor/pkg/sentry/vfs"
)

// fabricClasses are the RM object classes that back CUDA fabric / multicast
// memory allocated via cuMemExportToShareableHandle() (e.g. NCCL NVLS and
// PyTorch symmetric memory). cuda-checkpoint cannot serialize this memory and
// hangs on it during `--action checkpoint`, so nvproxy drains (frees) these
// objects while the application is quiesced, before cuda-checkpoint runs.
var fabricClasses = map[nvgpu.ClassID]struct{}{
	nvgpu.NV_MEMORY_FABRIC:              {},
	nvgpu.NV_MEMORY_MULTICAST_FABRIC:    {},
	nvgpu.NV_MEMORY_FABRIC_IMPORTED_REF: {},
	nvgpu.NV_MEMORY_EXPORT:              {},
}

// CheckpointBlocker describes a live driver resource that blocks
// cuda-checkpoint from checkpointing the sandbox: multicast/fabric RM objects
// (which cuda-checkpoint cannot serialize) and FDs holding exported RM
// objects (live CUDA IPC).
type CheckpointBlocker struct {
	ClientHandle nvgpu.Handle
	ObjectHandle nvgpu.Handle
	Class        nvgpu.ClassID
	TaskID       int32  // thread group ID of the allocating task, or 0
	Kind         string // "multicast" | "fabric" | "fabric-import" | "exported-fd"
}

// String implements fmt.Stringer.
func (b *CheckpointBlocker) String() string {
	return fmt.Sprintf("{client %v object %v class %v task %d kind %q}", b.ClientHandle, b.ObjectHandle, b.Class, b.TaskID, b.Kind)
}

func blockerKind(class nvgpu.ClassID) string {
	switch class {
	case nvgpu.NV_MEMORY_MULTICAST_FABRIC:
		return "multicast"
	case nvgpu.NV_MEMORY_FABRIC_IMPORTED_REF:
		return "fabric-import"
	default:
		return "fabric"
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
				if _, ok := fabricClasses[o.class]; ok {
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
				ClientHandle: eo.Client,
				ObjectHandle: eo.Object,
				Class:        eo.Class,
				TaskID:       eo.TaskID,
				Kind:         "exported-fd",
			})
		}
	}
	nvp.fdsMu.Unlock()

	sort.Slice(out, func(i, j int) bool {
		if out[i].ClientHandle.Val != out[j].ClientHandle.Val {
			return out[i].ClientHandle.Val < out[j].ClientHandle.Val
		}
		return out[i].ObjectHandle.Val < out[j].ObjectHandle.Val
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
	counts := make(map[key]map[string]int)
	var order []key
	for _, b := range blockers {
		k := key{b.ClientHandle, b.TaskID}
		if _, ok := counts[k]; !ok {
			counts[k] = make(map[string]int)
			order = append(order, k)
		}
		counts[k][b.Kind]++
	}
	var lines []string
	for _, k := range order {
		kinds := make([]string, 0, len(counts[k]))
		for kind := range counts[k] {
			kinds = append(kinds, kind)
		}
		sort.Strings(kinds)
		parts := make([]string, 0, len(kinds))
		for _, kind := range kinds {
			parts = append(parts, fmt.Sprintf("%d %s", counts[k][kind], kind))
		}
		lines = append(lines, fmt.Sprintf("task %d (client %v): %s", k.task, k.client, strings.Join(parts, ", ")))
	}
	return strings.Join(lines, "; ")
}

// exportedObjInfo records that an RM object was exported into a frontendFD
// via NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD.
//
// +stateify savable
type exportedObjInfo struct {
	Client nvgpu.Handle
	Object nvgpu.Handle
	Class  nvgpu.ClassID
	TaskID int32
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
// allocation after restore) parses the nvproxy_exported_object line.
func (fd *frontendFD) ProcFDInfoExtra(ctx context.Context) string {
	nvp := fd.dev.nvp
	nvp.fdsMu.Lock()
	exp := fd.exportedObj
	nvp.fdsMu.Unlock()
	if exp == nil {
		return ""
	}
	return fmt.Sprintf("nvproxy_exported_object:\tclient=%#x object=%#x class=%#x\n",
		exp.Client.Val, exp.Object.Val, uint32(exp.Class))
}

// ctrlClientExportObjectsToFD proxies
// NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECTS_TO_FD (the batched form used by
// current libcuda, e.g. for cuMemExportToShareableHandle) like
// ctrlHasFrontendFD, and additionally marks the destination frontendFD as
// holding exported RM objects, so that it is reported as a checkpoint blocker
// until closed.
func ctrlClientExportObjectsToFD(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	var ctrlParams nvgpu.NV0000_CTRL_OS_UNIX_EXPORT_OBJECTS_TO_FD_PARAMS
	if ctrlParams.SizeBytes() != int(ioctlParams.ParamsSize) {
		return 0, linuxerr.EINVAL
	}
	if _, err := ctrlParams.CopyIn(fi.t, addrFromP64(ioctlParams.Params)); err != nil {
		return 0, err
	}

	origFD := ctrlParams.FD
	ctlFileGeneric, _ := fi.t.FDTable().Get(origFD)
	if ctlFileGeneric == nil {
		return 0, linuxerr.EINVAL
	}
	defer ctlFileGeneric.DecRef(fi.ctx)
	ctlFile, ok := ctlFileGeneric.Impl().(*frontendFD)
	if !ok {
		return 0, linuxerr.EINVAL
	}

	ctrlParams.FD = ctlFile.hostFD
	n, err := rmControlInvoke(fi, ioctlParams, &ctrlParams)
	ctrlParams.FD = origFD
	if err != nil {
		return n, err
	}
	if _, cerr := ctrlParams.CopyOut(fi.t, addrFromP64(ioctlParams.Params)); cerr != nil {
		return n, cerr
	}

	if ioctlParams.Status == nvgpu.NV_OK && ctrlParams.NumObjects > 0 {
		markExportedObjFD(fi, ctlFile, ioctlParams.HClient, ctrlParams.Objects[0])
	}
	return n, nil
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
		Client: clientH,
		Object: objectH,
		Class:  class,
		TaskID: taskID,
	}
	nvp.fdsMu.Unlock()
}

// ctrlClientExportObjectToFD proxies NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD
// like ctrlHasFrontendFD, and additionally marks the destination frontendFD
// as holding an exported RM object, so that it is reported as a checkpoint
// blocker until closed.
func ctrlClientExportObjectToFD(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	var ctrlParams nvgpu.NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_PARAMS
	if ctrlParams.SizeBytes() != int(ioctlParams.ParamsSize) {
		return 0, linuxerr.EINVAL
	}
	if _, err := ctrlParams.CopyIn(fi.t, addrFromP64(ioctlParams.Params)); err != nil {
		return 0, err
	}

	origFD := ctrlParams.FD
	ctlFileGeneric, _ := fi.t.FDTable().Get(origFD)
	if ctlFileGeneric == nil {
		return 0, linuxerr.EINVAL
	}
	defer ctlFileGeneric.DecRef(fi.ctx)
	ctlFile, ok := ctlFileGeneric.Impl().(*frontendFD)
	if !ok {
		return 0, linuxerr.EINVAL
	}

	ctrlParams.FD = ctlFile.hostFD
	n, err := rmControlInvoke(fi, ioctlParams, &ctrlParams)
	ctrlParams.FD = origFD
	if err != nil {
		return n, err
	}
	if _, cerr := ctrlParams.CopyOut(fi.t, addrFromP64(ioctlParams.Params)); cerr != nil {
		return n, cerr
	}

	if ioctlParams.Status == nvgpu.NV_OK {
		// For type NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM (0), the union is
		// struct {hDevice, hParent, hObject}; record hObject for attribution.
		var objectH nvgpu.Handle
		if ctrlParams.Object.Type == 0 /* NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM */ {
			objectH.Val = uint32(ctrlParams.Object.Data[8]) | uint32(ctrlParams.Object.Data[9])<<8 | uint32(ctrlParams.Object.Data[10])<<16 | uint32(ctrlParams.Object.Data[11])<<24
		}
		markExportedObjFD(fi, ctlFile, ioctlParams.HClient, objectH)
	}
	return n, nil
}

// censusHandleClasses are the classes whose individual handles (not just
// counts) are logged by ObjectGraphCensus. NV01_MEMORY_LOCAL_USER is included
// to measure whether libcuda's cuda-checkpoint restore path recreates
// physical memory objects with identical handles — which determines how
// multicast ATTACH_MEM replay must resolve its hMemory references.
var censusHandleClasses = map[nvgpu.ClassID]struct{}{
	nvgpu.NV01_MEMORY_LOCAL_USER:        {},
	nvgpu.NV_MEMORY_FABRIC:              {},
	nvgpu.NV_MEMORY_MULTICAST_FABRIC:    {},
	nvgpu.NV_MEMORY_FABRIC_IMPORTED_REF: {},
}

// ClientObjectCensus summarizes the live RM objects of one root client, as a
// class -> count histogram.
type ClientObjectCensus struct {
	Client  nvgpu.Handle
	Total   int
	Classes map[nvgpu.ClassID]int
	// Handles lists individual object handles for classes in
	// censusHandleClasses, sorted ascending.
	Handles map[nvgpu.ClassID][]uint32
}

// String implements fmt.Stringer. Classes are printed in ascending order;
// classes that cuda-checkpoint cannot serialize (fabricClasses) are marked
// with a trailing "!".
func (c *ClientObjectCensus) String() string {
	ids := make([]nvgpu.ClassID, 0, len(c.Classes))
	for id := range c.Classes {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	var sb strings.Builder
	fmt.Fprintf(&sb, "client %v: %d object(s):", c.Client, c.Total)
	for _, id := range ids {
		mark := ""
		if _, ok := fabricClasses[id]; ok {
			mark = "!"
		}
		fmt.Fprintf(&sb, " %v%s x%d", id, mark, c.Classes[id])
		if hs, ok := c.Handles[id]; ok {
			sb.WriteString("[")
			for i, h := range hs {
				if i > 0 {
					sb.WriteString(" ")
				}
				fmt.Fprintf(&sb, "%#x", h)
			}
			sb.WriteString("]")
		}
	}
	return sb.String()
}

// ObjectGraphCensus returns, without modifying any state, a per-client class
// histogram of all live RM objects tracked by the nvproxy registered in
// vfsObj, sorted by client handle.
//
// This is Phase 0 instrumentation for multicast suspend/replay: logging the
// census immediately before the cuda-checkpoint lock phase and immediately
// after the checkpoint phase reveals which objects libcuda frees during
// `cuda-checkpoint --action checkpoint` and which survive in the graph. In
// particular, whether the physical memory objects that multicast ATTACH_MEM
// references survive determines whether multicast replay can ride
// nvproxy.afterLoad()'s topological sort, or must be deferred until after the
// post-restore cuda-checkpoint toggle.
func ObjectGraphCensus(vfsObj *vfs.VirtualFilesystem) []ClientObjectCensus {
	nvp := nvproxyFromVFS(vfsObj)
	if nvp == nil {
		return nil
	}
	return nvp.objectGraphCensus()
}

func (nvp *nvproxy) objectGraphCensus() []ClientObjectCensus {
	nvp.clientsMu.RLock()
	clients := make([]*rootClient, 0, len(nvp.clients))
	for _, c := range nvp.clients {
		clients = append(clients, c)
	}
	nvp.clientsMu.RUnlock()
	var out []ClientObjectCensus
	for _, client := range clients {
		client.objsMu.Lock()
		if !client.released {
			cc := ClientObjectCensus{
				Client:  client.handle,
				Classes: make(map[nvgpu.ClassID]int),
				Handles: make(map[nvgpu.ClassID][]uint32),
			}
			for h, o := range client.resources {
				cc.Classes[o.class]++
				cc.Total++
				if _, ok := censusHandleClasses[o.class]; ok {
					cc.Handles[o.class] = append(cc.Handles[o.class], h.Val)
				}
			}
			for _, hs := range cc.Handles {
				sort.Slice(hs, func(i, j int) bool { return hs[i] < hs[j] })
			}
			out = append(out, cc)
		}
		client.objsMu.Unlock()
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Client.Val < out[j].Client.Val })
	return out
}

