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
	"testing"

	"gvisor.dev/gvisor/pkg/abi/nvgpu"
	"gvisor.dev/gvisor/pkg/context"
)

// TestProcFDInfoExtraFormat locks the fdinfo line format: userspace (e.g. the
// checkpoint interposer at https://github.com/modal-projects/mcshim) parses it
// to identify exported allocations across processes.
func TestProcFDInfoExtraFormat(t *testing.T) {
	exp := exportedObjInfo{
		client: nvgpu.Handle{Val: 0xc1d00922},
		object: nvgpu.Handle{Val: 0x5c000123},
		class:  nvgpu.NV01_MEMORY_LOCAL_USER,
	}
	got := procFDInfoExportedObjectLine(exp)
	want := "nvproxy_exported_object:\tclient=0xc1d00922 object=0x5c000123 class=0x40\n"
	if got != want {
		t.Errorf("fdinfo line = %q, want %q", got, want)
	}
}

// TestExportedObj covers what an export records on an fd, and what fdinfo then
// shows: the exported object's identity, with its class if nvproxy tracks it;
// nothing for an fd without an export, or one cleared by a zero handle.
func TestExportedObj(t *testing.T) {
	ctx := context.Background()
	nvp := &nvproxy{clients: make(map[nvgpu.Handle]*rootClient)}
	fi := &frontendIoctlState{fd: &frontendFD{dev: &frontendDevice{nvp: nvp}}, ctx: ctx}
	clientH := nvgpu.Handle{Val: 0xc1d00001}
	c := &rootClient{resources: make(map[nvgpu.Handle]*object)}
	nvp.clients[clientH] = c
	memH := nvgpu.Handle{Val: 0x5c000001}
	c.objsMu.Lock()
	nvp.objAdd(ctx, c, clientH, nvgpu.NV01_ROOT_CLIENT, c, nvgpu.Handle{Val: nvgpu.NV01_NULL_OBJECT})
	nvp.objAdd(ctx, c, memH, nvgpu.NV_MEMORY_MULTICAST_FABRIC, &miscObject{}, clientH)
	c.objsMu.Unlock()

	fd := &frontendFD{dev: &frontendDevice{nvp: nvp}}
	if got := fd.ProcFDInfoExtra(ctx); got != "" {
		t.Errorf("ProcFDInfoExtra() before any export = %q, want empty", got)
	}

	exp := exportedObjOf(fi, clientH, memH)
	want := exportedObjInfo{client: clientH, object: memH, class: nvgpu.NV_MEMORY_MULTICAST_FABRIC}
	if exp != want {
		t.Fatalf("exportedObjOf() = %+v, want %+v", exp, want)
	}
	setExportedObj(nvp, fd, exp)
	if got, want := fd.ProcFDInfoExtra(ctx), procFDInfoExportedObjectLine(want); got != want {
		t.Errorf("ProcFDInfoExtra() = %q, want %q", got, want)
	}

	unknown := nvgpu.Handle{Val: 0xc1d00002}
	if got := exportedObjOf(fi, unknown, memH); got != (exportedObjInfo{client: unknown, object: memH}) {
		t.Errorf("exportedObjOf(unknown client) = %+v, want the handles only", got)
	}

	setExportedObj(nvp, fd, exportedObjOf(fi, clientH, nvgpu.Handle{}))
	if got := fd.ProcFDInfoExtra(ctx); got != "" {
		t.Errorf("ProcFDInfoExtra() after clearing = %q, want empty", got)
	}
}
