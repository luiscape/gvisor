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
)

// TestProcFDInfoExtraFormat locks the fdinfo oracle line format: the
// interposer (tools/mcshim) parses it to identify exported allocations across
// processes.
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

func TestExportedObjSlots(t *testing.T) {
	fd := &frontendFD{}
	if _, ok := fd.exportedObjInfoLocked(); ok {
		t.Fatal("empty fd reported an exported object")
	}
	fd.exportedObjs = map[uint16]exportedObjInfo{
		2: {object: nvgpu.Handle{Val: 2}},
		0: {object: nvgpu.Handle{Val: 0xa}},
		1: {object: nvgpu.Handle{Val: 1}},
	}
	// The identity is the lowest slot (libcuda's single export lands in 0).
	if exp, ok := fd.exportedObjInfoLocked(); !ok || exp.object.Val != 0xa {
		t.Errorf("exportedObjInfoLocked() = %v, %v; want slot 0 (object 0xa)", exp, ok)
	}
	delete(fd.exportedObjs, 0)
	if exp, ok := fd.exportedObjInfoLocked(); !ok || exp.object.Val != 1 {
		t.Errorf("after unexporting slot 0, exportedObjInfoLocked() = %v, %v; want slot 1", exp, ok)
	}
}
