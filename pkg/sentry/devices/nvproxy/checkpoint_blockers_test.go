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
	"gvisor.dev/gvisor/pkg/sentry/kernel"
)

func TestCheckpointBlockers(t *testing.T) {
	ctx := context.Background()
	nvp := &nvproxy{clients: make(map[nvgpu.Handle]*rootClient)}
	addClient := func(h uint32, tgid kernel.ThreadID) *rootClient {
		c := &rootClient{resources: make(map[nvgpu.Handle]*object), tgid: tgid}
		nvp.clients[nvgpu.Handle{Val: h}] = c
		c.objsMu.Lock()
		nvp.objAdd(ctx, c, nvgpu.Handle{Val: h}, nvgpu.NV01_ROOT_CLIENT, c, nvgpu.Handle{Val: nvgpu.NV01_NULL_OBJECT})
		c.objsMu.Unlock()
		return c
	}
	addObj := func(c *rootClient, h uint32, class nvgpu.ClassID) {
		c.objsMu.Lock()
		nvp.objAdd(ctx, c, nvgpu.Handle{Val: h}, class, &miscObject{}, c.handle)
		c.objsMu.Unlock()
	}

	rank1 := addClient(0xc1d00002, 42)
	rank0 := addClient(0xc1d00001, 41)
	addObj(rank0, 0x5c000001, nvgpu.NV01_DEVICE_0)    // not a blocker
	addObj(rank0, 0x5c000002, nvgpu.NV_MEMORY_FABRIC) // not a blocker
	if got := nvp.checkpointBlockers(nil); got != "" {
		t.Fatalf("no blockers expected, got %q", got)
	}

	addObj(rank0, 0x5c000003, nvgpu.NV_MEMORY_MULTICAST_FABRIC)
	addObj(rank0, 0x5c000004, nvgpu.NV_MEMORY_MULTICAST_FABRIC)
	addObj(rank1, 0x5c000005, nvgpu.NV_MEMORY_FABRIC_IMPORTED_REF)
	want := "PID 41 (client 0xc1d00001): 2 multicast; PID 42 (client 0xc1d00002): 1 fabric-import"
	if got := nvp.checkpointBlockers(nil); got != want {
		t.Fatalf("got %q, want %q", got, want)
	}

	// Freed objects drop out.
	rank0.objsMu.Lock()
	nvp.objFree(ctx, rank0, nvgpu.Handle{Val: 0x5c000003})
	rank0.objsMu.Unlock()
	want = "PID 41 (client 0xc1d00001): 1 multicast; PID 42 (client 0xc1d00002): 1 fabric-import"
	if got := nvp.checkpointBlockers(nil); got != want {
		t.Fatalf("after free: got %q, want %q", got, want)
	}

	// Processes running the interposer can be exempted.
	want = "PID 42 (client 0xc1d00002): 1 fabric-import"
	if got := nvp.checkpointBlockers(map[kernel.ThreadID]bool{41: true}); got != want {
		t.Fatalf("except 41: got %q, want %q", got, want)
	}

	// Objects imported from an exported fd are blockers, whatever their class.
	addImport := func(c *rootClient, h uint32, src exportedObjInfo, multicast bool) {
		c.objsMu.Lock()
		nvp.objAdd(ctx, c, nvgpu.Handle{Val: h}, src.class, &importedObject{src: src, multicast: multicast}, c.handle)
		c.objsMu.Unlock()
	}
	addImport(rank1, 0x5c000006, exportedObjInfo{client: rank0.handle, object: nvgpu.Handle{Val: 0x5c000001}, class: nvgpu.NV01_MEMORY_LOCAL_USER}, false)
	addImport(rank1, 0x5c000007, exportedObjInfo{client: rank0.handle, object: nvgpu.Handle{Val: 0x5c000004}, class: nvgpu.NV_MEMORY_MULTICAST_FABRIC}, true)
	want = "PID 42 (client 0xc1d00002): 1 fabric-import, 1 import, 1 multicast-import"
	if got := nvp.checkpointBlockers(map[kernel.ThreadID]bool{41: true}); got != want {
		t.Fatalf("imports: got %q, want %q", got, want)
	}
}

func TestUnresolvableImports(t *testing.T) {
	ctx := context.Background()
	nvp := &nvproxy{clients: make(map[nvgpu.Handle]*rootClient)}
	addClient := func(h uint32, tgid kernel.ThreadID) *rootClient {
		c := &rootClient{resources: make(map[nvgpu.Handle]*object), tgid: tgid}
		nvp.clients[nvgpu.Handle{Val: h}] = c
		c.objsMu.Lock()
		nvp.objAdd(ctx, c, nvgpu.Handle{Val: h}, nvgpu.NV01_ROOT_CLIENT, c, nvgpu.Handle{Val: nvgpu.NV01_NULL_OBJECT})
		c.objsMu.Unlock()
		return c
	}
	add := func(c *rootClient, h uint32, oi objectImpl, class nvgpu.ClassID) {
		c.objsMu.Lock()
		nvp.objAdd(ctx, c, nvgpu.Handle{Val: h}, class, oi, c.handle)
		c.objsMu.Unlock()
	}
	exporter := addClient(0xc1d00001, 41)
	importer := addClient(0xc1d00002, 42)
	exported := &miscObject{}
	add(exporter, 0x5c000001, exported, nvgpu.NV01_MEMORY_LOCAL_USER)
	src := exportedObjInfo{client: exporter.handle, object: nvgpu.Handle{Val: 0x5c000001}, class: nvgpu.NV01_MEMORY_LOCAL_USER, obj: exported.Object()}
	add(importer, 0x5c000002, &importedObject{src: src}, src.class)
	both := map[kernel.ThreadID]bool{41: true, 42: true}
	if got := nvp.unresolvableImports(both); got != "" {
		t.Fatalf("live exporter: got %q, want none", got)
	}

	// The exporter is not among the processes that will rebuild.
	want := "PID 42: object 0xc1d00002:0x5c000002 imported from 0xc1d00001:0x5c000001, which no longer exists"
	if got := nvp.unresolvableImports(map[kernel.ThreadID]bool{42: true}); got != want {
		t.Fatalf("exporter not managed: got %q, want %q", got, want)
	}

	// The exporter freed the object, and libcuda reused its handle.
	exporter.objsMu.Lock()
	nvp.objFree(ctx, exporter, nvgpu.Handle{Val: 0x5c000001})
	exporter.objsMu.Unlock()
	if got := nvp.unresolvableImports(both); got != want {
		t.Fatalf("exporter freed: got %q, want %q", got, want)
	}
	add(exporter, 0x5c000001, &miscObject{}, nvgpu.NV01_MEMORY_LOCAL_USER)
	if got := nvp.unresolvableImports(both); got != want {
		t.Fatalf("handle reused: got %q, want %q", got, want)
	}

	// Freeing the import itself, or its parent, drops it.
	importer.objsMu.Lock()
	nvp.objFree(ctx, importer, nvgpu.Handle{Val: 0x5c000002})
	importer.objsMu.Unlock()
	if got := nvp.unresolvableImports(both); got != "" {
		t.Fatalf("import freed: got %q, want none", got)
	}

	// An import whose export is unknown cannot be rebuilt.
	add(importer, 0x5c000003, &importedObject{}, 0)
	want = "PID 42: object 0xc1d00002:0x5c000003 imported from an unknown export"
	if got := nvp.unresolvableImports(both); got != want {
		t.Fatalf("unknown export: got %q, want %q", got, want)
	}
}
