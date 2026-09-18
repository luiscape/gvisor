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

// testGraph builds an nvproxy with one live client holding the given objects
// (handle -> class), the client object itself included under clientH.
func testGraph(t *testing.T, clientH nvgpu.Handle, objs map[uint32]nvgpu.ClassID) *nvproxy {
	t.Helper()
	ctx := context.Background()
	nvp := &nvproxy{
		clients:     make(map[nvgpu.Handle]*rootClient),
		frontendFDs: make(map[*frontendFD]struct{}),
	}
	client := &rootClient{resources: make(map[nvgpu.Handle]*object)}
	client.objsMu.Lock()
	defer client.objsMu.Unlock()
	nvp.objAdd(ctx, client, clientH, nvgpu.NV01_ROOT_CLIENT, client, nvgpu.Handle{})
	nvp.clients[clientH] = client
	for h, class := range objs {
		nvp.objAdd(ctx, client, nvgpu.Handle{Val: h}, class, &miscObject{}, clientH)
	}
	return nvp
}

func TestCheckpointBlockersReportsOnlyFabricClasses(t *testing.T) {
	clientH := nvgpu.Handle{Val: 0xc1d00001}
	nvp := testGraph(t, clientH, map[uint32]nvgpu.ClassID{
		0x10: nvgpu.NV01_DEVICE_0,
		0x11: nvgpu.NV20_SUBDEVICE_0,
		0x20: nvgpu.NV_MEMORY_MULTICAST_FABRIC,
		0x21: nvgpu.NV_MEMORY_FABRIC,
		0x22: nvgpu.NV_MEMORY_FABRIC_IMPORTED_REF,
		0x30: nvgpu.NV50_MEMORY_VIRTUAL,
	})
	got := nvp.checkpointBlockers()
	want := []struct {
		h    uint32
		kind BlockerKind
	}{
		{0x20, BlockerKindMulticast},
		{0x21, BlockerKindFabric},
		{0x22, BlockerKindFabricImport},
	}
	if len(got) != len(want) {
		t.Fatalf("checkpointBlockers() = %v, want %d blockers", got, len(want))
	}
	for i, w := range want {
		if got[i].ObjectHandle.Val != w.h || got[i].Kind != w.kind || got[i].ClientHandle != clientH {
			t.Errorf("blocker %d = %v, want handle %#x kind %q", i, got[i], w.h, w.kind)
		}
	}
}

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

func TestFormatBlockersByClient(t *testing.T) {
	c1 := nvgpu.Handle{Val: 0xc1d00001}
	c2 := nvgpu.Handle{Val: 0xc1d00002}
	blockers := []CheckpointBlocker{
		{ClientHandle: c1, ObjectHandle: nvgpu.Handle{Val: 1}, Kind: BlockerKindMulticast, TaskID: 7},
		{ClientHandle: c1, ObjectHandle: nvgpu.Handle{Val: 2}, Kind: BlockerKindMulticast, TaskID: 7},
		{ClientHandle: c1, ObjectHandle: nvgpu.Handle{Val: 3}, Kind: BlockerKindExportedFD, TaskID: 7},
		{ClientHandle: c2, ObjectHandle: nvgpu.Handle{Val: 4}, Kind: BlockerKindFabric, TaskID: 9},
	}
	got := FormatBlockersByClient(blockers)
	want := "task 7 (client 0xc1d00001): 1 exported-fd, 2 multicast; task 9 (client 0xc1d00002): 1 fabric"
	if got != want {
		t.Errorf("FormatBlockersByClient() =\n  %s\nwant\n  %s", got, want)
	}
}
