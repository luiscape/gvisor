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

// TestProcFDInfoExtraFormat locks the fdinfo oracle line format: the
// interposer (https://github.com/modal-projects/mcshim) parses it to identify exported allocations across
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

// TestRecordExportImport covers what the export and import handlers record,
// with the client's objsMu held as ctrlExportToFDInvoke holds it across the
// host call: an export's identity includes the tracked object, an import is
// attributed to its fd's export (with the export's class in slot 0 only), and
// nothing is recorded for an unknown client.
func TestRecordExportImport(t *testing.T) {
	ctx := context.Background()
	nvp := &nvproxy{clients: make(map[nvgpu.Handle]*rootClient)}
	fi := &frontendIoctlState{fd: &frontendFD{dev: &frontendDevice{nvp: nvp}}, ctx: ctx}
	exporterH, importerH := nvgpu.Handle{Val: 0xc1d00001}, nvgpu.Handle{Val: 0xc1d00002}
	newClient := func(h nvgpu.Handle) *rootClient {
		c := &rootClient{resources: make(map[nvgpu.Handle]*object)}
		nvp.clients[h] = c
		c.objsMu.Lock()
		nvp.objAdd(ctx, c, h, nvgpu.NV01_ROOT_CLIENT, c, nvgpu.Handle{Val: nvgpu.NV01_NULL_OBJECT})
		c.objsMu.Unlock()
		return c
	}
	exporter, importer := newClient(exporterH), newClient(importerH)
	memH := nvgpu.Handle{Val: 0x5c000001}
	exporter.objsMu.Lock()
	nvp.objAdd(ctx, exporter, memH, nvgpu.NV_MEMORY_MULTICAST_FABRIC, &miscObject{}, exporterH)

	// Export: the identity names the tracked object.
	exp := exportedObjOf(exporter, exporterH, memH)
	exporter.objsMu.Unlock()
	if exp.class != nvgpu.NV_MEMORY_MULTICAST_FABRIC || exp.obj == nil || exp.obj != exporter.resources[memH] {
		t.Fatalf("exportedObjOf() = %+v, want the tracked multicast object", exp)
	}
	if got := exportedObjOf(nil, exporterH, memH); got.obj != nil || got.object != memH {
		t.Errorf("exportedObjOf(unknown client) = %+v, want the handles only", got)
	}
	if got := exportedObjOf(exporter, exporterH, nvgpu.Handle{}); got != (exportedObjInfo{}) {
		t.Errorf("exportedObjOf(0) = %+v, want zero (clears the fd)", got)
	}
	fd := &frontendFD{}
	setExportedObj(fi, fd, exp)()
	if fd.exportedObj != exp {
		t.Errorf("fd.exportedObj = %+v, want %+v", fd.exportedObj, exp)
	}

	// Import, slots 0 and 1 of the same fd.
	slot0, slot1 := nvgpu.Handle{Val: 0x5c000010}, nvgpu.Handle{Val: 0x5c000011}
	importer.objsMu.Lock()
	addImportedObj(fi, importer, exp, importerH, slot0, 0, true)
	addImportedObj(fi, importer, exp, importerH, slot1, 1, false)
	importer.objsMu.Unlock()
	for _, tc := range []struct {
		h         nvgpu.Handle
		class     nvgpu.ClassID
		multicast bool
	}{
		{slot0, nvgpu.NV_MEMORY_MULTICAST_FABRIC, true},
		{slot1, 0, false},
	} {
		o, ok := importer.resources[tc.h]
		if !ok {
			t.Fatalf("import %v not recorded", tc.h)
		}
		imp, ok := o.impl.(*importedObject)
		if !ok || o.class != tc.class || imp.multicast != tc.multicast || imp.src.obj != exp.obj {
			t.Errorf("import %v = class %v %+v, want class %v, multicast %v, attributed to the export", tc.h, o.class, imp, tc.class, tc.multicast)
		}
	}

	// An unknown client records nothing (and does not crash).
	addImportedObj(fi, nil, exp, importerH, nvgpu.Handle{Val: 0x5c000012}, 0, false)
}

// TestProcFDInfoExtraTracking checks that the fdinfo line is only shown with
// Options.TrackExports: without the multicast interposer, /proc shows nothing
// new.
func TestProcFDInfoExtraTracking(t *testing.T) {
	ctx := context.Background()
	exp := exportedObjInfo{
		client: nvgpu.Handle{Val: 0xc1d00922},
		object: nvgpu.Handle{Val: 0x5c000123},
		class:  nvgpu.NV01_MEMORY_LOCAL_USER,
	}
	for _, track := range []bool{false, true} {
		nvp := &nvproxy{trackExports: track}
		fd := &frontendFD{dev: &frontendDevice{nvp: nvp}, exportedObj: exp}
		got := fd.ProcFDInfoExtra(ctx)
		want := ""
		if track {
			want = procFDInfoExportedObjectLine(exp)
		}
		if got != want {
			t.Errorf("TrackExports=%v: ProcFDInfoExtra() = %q, want %q", track, got, want)
		}
	}
}
