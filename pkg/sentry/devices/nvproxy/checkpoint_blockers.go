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
	"fmt"
	"sort"
	"strings"

	"gvisor.dev/gvisor/pkg/abi/nvgpu"
	"gvisor.dev/gvisor/pkg/sentry/kernel"
	"gvisor.dev/gvisor/pkg/sentry/vfs"
)

// checkpointBlockerClasses are RM object classes on which cuda-checkpoint
// hangs: NVLink multicast groups (NCCL NVLS, PyTorch symmetric memory) and
// imports of another process's fabric memory. NV_MEMORY_FABRIC itself is not
// included: libcuda allocates it per context on fabric-attached GPUs and it
// checkpoints fine.
var checkpointBlockerClasses = map[nvgpu.ClassID]string{
	nvgpu.NV_MEMORY_MULTICAST_FABRIC:    "multicast",
	nvgpu.NV_MEMORY_FABRIC_IMPORTED_REF: "fabric-import",
}

// blockerKind returns the kind of checkpoint blocker o is, or "".
// Objects imported from an exported fd are blockers too: cuda-checkpoint
// cannot restore them, or for multicast objects hangs.
func blockerKind(o *object) string {
	if imp, ok := o.impl.(*importedObject); ok {
		if imp.multicast {
			return "multicast-import"
		}
		return "import"
	}
	return checkpointBlockerClasses[o.class]
}

// CheckpointBlockers returns a description of live RM objects that would
// make cuda-checkpoint hang or fail, one line per owning process, or "" if
// there are none. Processes in except are skipped.
func CheckpointBlockers(vfsObj *vfs.VirtualFilesystem, except map[kernel.ThreadID]bool) string {
	if nvp := nvproxyFromVFS(vfsObj); nvp != nil {
		return nvp.checkpointBlockers(except)
	}
	return ""
}

func (nvp *nvproxy) checkpointBlockers(except map[kernel.ThreadID]bool) string {
	type owner struct {
		tgid   kernel.ThreadID
		client nvgpu.Handle
	}
	// Lock order is objsMu before clientsMu, so snapshot the clients first.
	nvp.clientsMu.RLock()
	clients := make([]*rootClient, 0, len(nvp.clients))
	for _, client := range nvp.clients {
		clients = append(clients, client)
	}
	nvp.clientsMu.RUnlock()
	counts := make(map[owner]map[string]int)
	for _, client := range clients {
		if except[client.tgid] {
			continue
		}
		client.objsMu.Lock()
		if !client.released {
			for _, o := range client.resources {
				if kind := blockerKind(o); kind != "" {
					k := owner{client.tgid, client.handle}
					if counts[k] == nil {
						counts[k] = make(map[string]int)
					}
					counts[k][kind]++
				}
			}
		}
		client.objsMu.Unlock()
	}

	owners := make([]owner, 0, len(counts))
	for k := range counts {
		owners = append(owners, k)
	}
	sort.Slice(owners, func(i, j int) bool {
		if owners[i].tgid != owners[j].tgid {
			return owners[i].tgid < owners[j].tgid
		}
		return owners[i].client.Val < owners[j].client.Val
	})
	var lines []string
	for _, k := range owners {
		kinds := make([]string, 0, len(counts[k]))
		for kind := range counts[k] {
			kinds = append(kinds, kind)
		}
		sort.Strings(kinds)
		for i, kind := range kinds {
			kinds[i] = fmt.Sprintf("%d %s", counts[k][kind], kind)
		}
		lines = append(lines, fmt.Sprintf("PID %d (client %v): %s", k.tgid, k.client, strings.Join(kinds, ", ")))
	}
	return strings.Join(lines, "; ")
}

// UnresolvableImports returns a description of the objects that processes in
// procs imported, but whose exported object no longer exists in a process in
// procs, one per line, or "" if there are none. The multicast interposer
// rebuilds an import after a restore from its exporter's re-export, which such
// an import would wait for in vain. procs must be quiesced, so that nothing is
// freed meanwhile.
func UnresolvableImports(vfsObj *vfs.VirtualFilesystem, procs map[kernel.ThreadID]bool) string {
	if nvp := nvproxyFromVFS(vfsObj); nvp != nil {
		return nvp.unresolvableImports(procs)
	}
	return ""
}

func (nvp *nvproxy) unresolvableImports(procs map[kernel.ThreadID]bool) string {
	// Lock order is objsMu before clientsMu, so snapshot the clients first.
	nvp.clientsMu.RLock()
	clients := make([]*rootClient, 0, len(nvp.clients))
	for _, client := range nvp.clients {
		clients = append(clients, client)
	}
	nvp.clientsMu.RUnlock()
	type objID struct{ client, object nvgpu.Handle }
	type imp struct {
		tgid kernel.ThreadID
		id   objID
		src  exportedObjInfo
	}
	live := make(map[objID]bool)
	var imps []imp
	for _, client := range clients {
		if !procs[client.tgid] {
			continue
		}
		client.objsMu.Lock()
		if !client.released {
			for h, o := range client.resources {
				live[objID{client.handle, h}] = true
				if i, ok := o.impl.(*importedObject); ok {
					imps = append(imps, imp{client.tgid, objID{client.handle, h}, i.src})
				}
			}
		}
		client.objsMu.Unlock()
	}
	var lines []string
	for _, i := range imps {
		if i.src.object.Val == 0 {
			lines = append(lines, fmt.Sprintf("PID %d: object %v:%v imported from an unknown export", i.tgid, i.id.client, i.id.object))
		} else if !live[objID{i.src.client, i.src.object}] {
			lines = append(lines, fmt.Sprintf("PID %d: object %v:%v imported from %v:%v, which no longer exists", i.tgid, i.id.client, i.id.object, i.src.client, i.src.object))
		}
	}
	sort.Strings(lines)
	return strings.Join(lines, "; ")
}
