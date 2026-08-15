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
	"strings"
	"testing"

	"gvisor.dev/gvisor/pkg/abi/nvgpu"
	"gvisor.dev/gvisor/pkg/context"
)

// newTestClientWithObjects builds an nvproxy with a single rootClient
// containing a multicastFabricObject and a plain memory object, without any
// host driver interaction.
func newTestClientWithObjects(ctx context.Context) (*nvproxy, *rootClient, *multicastFabricObject, nvgpu.Handle, nvgpu.Handle) {
	nvp := &nvproxy{
		clients:     make(map[nvgpu.Handle]*rootClient),
		frontendFDs: make(map[*frontendFD]struct{}),
	}
	clientH := nvgpu.Handle{Val: 0xc1d00001}
	client := &rootClient{
		resources: make(map[nvgpu.Handle]*object),
	}
	client.objsMu.Lock()
	defer client.objsMu.Unlock()
	nvp.objAdd(ctx, client, clientH, nvgpu.NV01_ROOT_CLIENT, client, nvgpu.Handle{Val: nvgpu.NV01_NULL_OBJECT})
	nvp.clients[clientH] = client

	memH := nvgpu.Handle{Val: 0x5c000002}
	mem := &miscObject{}
	nvp.objAdd(ctx, client, memH, nvgpu.NV01_MEMORY_LOCAL_USER, mem, clientH)

	mcH := nvgpu.Handle{Val: 0x5c000003}
	mcObj := &multicastFabricObject{}
	nvp.objAdd(ctx, client, mcH, nvgpu.NV_MEMORY_MULTICAST_FABRIC, mcObj, clientH)

	return nvp, client, mcObj, mcH, memH
}

func TestMulticastAttachRecording(t *testing.T) {
	ctx := context.Background()
	_, client, mcObj, mcH, memH := newTestClientWithObjects(ctx)

	subdev0 := nvgpu.Handle{Val: 0x5c000010}
	subdev1 := nvgpu.Handle{Val: 0x5c000011}

	client.objsMu.Lock()
	defer client.objsMu.Unlock()

	// Record two GPU attaches and two mem attaches.
	mcObj.recordAttachGPU(nvgpu.NV00FD_CTRL_ATTACH_GPU_PARAMS{HSubDevice: subdev0, DevDescriptor: 42}, 0)
	mcObj.recordAttachGPU(nvgpu.NV00FD_CTRL_ATTACH_GPU_PARAMS{HSubDevice: subdev1, DevDescriptor: 43}, 1)
	mcObj.recordAttachMem(nvgpu.NV00FD_CTRL_ATTACH_MEM_PARAMS{HSubDevice: subdev0, HMemory: memH, Offset: 0, MapLength: 0x200000})
	mcObj.recordAttachMem(nvgpu.NV00FD_CTRL_ATTACH_MEM_PARAMS{HSubDevice: subdev1, HMemory: memH, Offset: 0x200000, MapLength: 0x200000})
	client.objAddRestoreDep(mcH, memH)

	if got := len(mcObj.attachedGPUs); got != 2 {
		t.Errorf("attachedGPUs: got %d, want 2", got)
	}
	if got := len(mcObj.attachedMems); got != 2 {
		t.Errorf("attachedMems: got %d, want 2", got)
	}
	if mcObj.attachedGPUs[1].devMinor != 1 {
		t.Errorf("attachedGPUs[1].devMinor: got %d, want 1", mcObj.attachedGPUs[1].devMinor)
	}

	// The multicast object must restore-depend on the attached memory so that
	// the topological sort restores memory first.
	memObj := client.resources[memH]
	if _, ok := client.resources[mcH].restoreDeps[memObj]; !ok {
		t.Errorf("multicast object does not restore-depend on attached memory")
	}

	// Detach removes exactly the matching (subdevice, offset) record and
	// returns its HMemory.
	if hMem, ok := mcObj.removeAttachMem(ctx, nvgpu.NV00FD_CTRL_DETACH_MEM_PARAMS{HSubDevice: subdev0, Offset: 0}); !ok || hMem != memH {
		t.Errorf("removeAttachMem: got (%v, %t), want (%v, true)", hMem, ok, memH)
	}
	if got := len(mcObj.attachedMems); got != 1 {
		t.Fatalf("attachedMems after detach: got %d, want 1", got)
	}
	if mcObj.attachedMems[0].params.HSubDevice != subdev1 {
		t.Errorf("wrong attach removed: remaining subdevice %v, want %v", mcObj.attachedMems[0].params.HSubDevice, subdev1)
	}
	// The remaining attach still references memH, so the restore dependency
	// must survive the first detach.
	if !mcObj.attachMemsReference(memH) {
		t.Errorf("attachMemsReference(%v) = false, want true", memH)
	}
	// Detaching a non-existent binding must not remove anything.
	if _, ok := mcObj.removeAttachMem(ctx, nvgpu.NV00FD_CTRL_DETACH_MEM_PARAMS{HSubDevice: subdev0, Offset: 0xdead0000}); ok {
		t.Errorf("bogus detach unexpectedly matched a record")
	}
	if got := len(mcObj.attachedMems); got != 1 {
		t.Errorf("attachedMems after bogus detach: got %d, want 1", got)
	}
	// Detaching the last reference allows the restore dependency to be
	// dropped (as ctrlMemoryMulticastFabricDetachMem does).
	if hMem, ok := mcObj.removeAttachMem(ctx, nvgpu.NV00FD_CTRL_DETACH_MEM_PARAMS{HSubDevice: subdev1, Offset: 0x200000}); !ok || hMem != memH {
		t.Fatalf("removeAttachMem: got (%v, %t), want (%v, true)", hMem, ok, memH)
	}
	if mcObj.attachMemsReference(memH) {
		t.Errorf("attachMemsReference(%v) = true after last detach, want false", memH)
	}
	client.objRemoveRestoreDep(mcH, memH)
	if _, ok := client.resources[mcH].restoreDeps[memObj]; ok {
		t.Errorf("restore dependency survived objRemoveRestoreDep")
	}
}

// TestMulticastSurvivesMemoryFree verifies that freeing an attached memory
// object does NOT cascade-free the multicast object: the driver dups hMemory
// internally on ATTACH_MEM, so the multicast object legally outlives the
// application's memory handle — and it must keep being reported as a
// checkpoint blocker.
func TestMulticastSurvivesMemoryFree(t *testing.T) {
	ctx := context.Background()
	nvp, client, mcObj, mcH, memH := newTestClientWithObjects(ctx)

	subdev0 := nvgpu.Handle{Val: 0x5c000010}
	client.objsMu.Lock()
	mcObj.recordAttachMem(nvgpu.NV00FD_CTRL_ATTACH_MEM_PARAMS{HSubDevice: subdev0, HMemory: memH})
	client.objAddRestoreDep(mcH, memH)
	nvp.objFree(ctx, client, memH)
	client.objsMu.Unlock()

	client.objsMu.Lock()
	mc, live := client.resources[mcH]
	client.objsMu.Unlock()
	if !live {
		t.Fatalf("multicast object was cascade-freed by freeing attached memory")
	}
	// The freed memory object must have been dropped from the multicast
	// object's restore dependencies (no dangling edge).
	if len(mc.restoreDeps) != 0 {
		t.Errorf("restoreDeps not cleaned up on free: %v", mc.restoreDeps)
	}
	// And the blocker gate must still see the multicast object.
	blockers := nvp.checkpointBlockers()
	if len(blockers) != 1 || blockers[0].Kind != BlockerKindMulticast {
		t.Errorf("checkpointBlockers after memory free: got %v, want 1 multicast", blockers)
	}
	// The recorded attach now references a freed handle, so the object is no
	// longer replayable and a save must fail loudly.
	if err := mcObj.checkReplayable(); err == nil {
		t.Errorf("checkReplayable succeeded with a freed hMemory; want error")
	}
}

func TestRestoreAttachMemsFailsFastOnIncompleteTeam(t *testing.T) {
	ctx := context.Background()
	_, client, mcObj, _, memH := newTestClientWithObjects(ctx)

	subdev0 := nvgpu.Handle{Val: 0x5c000010}
	client.objsMu.Lock()
	mcObj.numGPUs = 2
	mcObj.recordAttachGPU(nvgpu.NV00FD_CTRL_ATTACH_GPU_PARAMS{HSubDevice: subdev0}, 0)
	mcObj.recordAttachMem(nvgpu.NV00FD_CTRL_ATTACH_MEM_PARAMS{HSubDevice: subdev0, HMemory: memH})
	client.objsMu.Unlock()

	// One recorded ATTACH_GPU of two required: the replay can never complete
	// the team, so it must fail deterministically instead of issuing an
	// ATTACH_MEM that would hang.
	if err := mcObj.restoreAttachMems(ctx); err == nil {
		t.Errorf("restoreAttachMems succeeded with an incomplete team; want error")
	}
}

func TestCheckpointBlockers(t *testing.T) {
	ctx := context.Background()
	nvp, client, _, mcH, _ := newTestClientWithObjects(ctx)

	blockers := nvp.checkpointBlockers()
	if len(blockers) != 1 {
		t.Fatalf("checkpointBlockers: got %d, want 1 (multicast only): %v", len(blockers), blockers)
	}
	b := blockers[0]
	if b.ObjectHandle != mcH || b.Kind != BlockerKindMulticast || b.Class != nvgpu.NV_MEMORY_MULTICAST_FABRIC {
		t.Errorf("unexpected blocker: %v", b)
	}

	// Add a fabric object and an imported ref; both must be reported with
	// their own kinds.
	client.objsMu.Lock()
	fabH := nvgpu.Handle{Val: 0x5c000020}
	nvp.objAdd(ctx, client, fabH, nvgpu.NV_MEMORY_FABRIC, &miscObject{}, client.handle)
	impH := nvgpu.Handle{Val: 0x5c000021}
	nvp.objAdd(ctx, client, impH, nvgpu.NV_MEMORY_FABRIC_IMPORTED_REF, &miscObject{}, client.handle)
	client.objsMu.Unlock()

	// An FD holding an exported RM object is a blocker of its own kind.
	expFD := &frontendFD{
		dev: &frontendDevice{nvp: nvp},
		exportedObj: &exportedObjInfo{
			client: client.handle,
			object: nvgpu.Handle{Val: 0x5c000030},
			class:  nvgpu.NV01_MEMORY_LOCAL_USER,
			taskID: 7,
		},
	}
	nvp.fdsMu.Lock()
	nvp.frontendFDs[expFD] = struct{}{}
	nvp.fdsMu.Unlock()

	blockers = nvp.checkpointBlockers()
	if len(blockers) != 4 {
		t.Fatalf("checkpointBlockers: got %d, want 4: %v", len(blockers), blockers)
	}
	kinds := make(map[BlockerKind]int)
	for _, b := range blockers {
		kinds[b.Kind]++
	}
	if kinds[BlockerKindMulticast] != 1 || kinds[BlockerKindFabric] != 1 || kinds[BlockerKindFabricImport] != 1 || kinds[BlockerKindExportedFD] != 1 {
		t.Errorf("unexpected blocker kinds: %v", kinds)
	}

	// Per-client formatting names the client and aggregates kinds.
	msg := FormatBlockersByClient(blockers)
	if !strings.Contains(msg, client.handle.String()) {
		t.Errorf("blocker message %q does not name client %v", msg, client.handle)
	}
	for _, want := range []string{"1 multicast", "1 fabric", "1 fabric-import", "1 exported-fd"} {
		if !strings.Contains(msg, want) {
			t.Errorf("blocker message %q missing %q", msg, want)
		}
	}

	// Closing the export FD and freeing the objects clears the blockers.
	nvp.fdsMu.Lock()
	delete(nvp.frontendFDs, expFD)
	nvp.fdsMu.Unlock()
	client.objsMu.Lock()
	for _, h := range []nvgpu.Handle{mcH, fabH, impH} {
		nvp.objFree(ctx, client, h)
	}
	client.objsMu.Unlock()
	if blockers = nvp.checkpointBlockers(); len(blockers) != 0 {
		t.Errorf("checkpointBlockers after free: got %v, want none", blockers)
	}
}

func TestFormatBlockersByClientMultiClient(t *testing.T) {
	blockers := []CheckpointBlocker{
		{ClientHandle: nvgpu.Handle{Val: 0xc1d00001}, TaskID: 11, Kind: BlockerKindMulticast},
		{ClientHandle: nvgpu.Handle{Val: 0xc1d00001}, TaskID: 11, Kind: BlockerKindMulticast},
		{ClientHandle: nvgpu.Handle{Val: 0xc1d00002}, TaskID: 12, Kind: BlockerKindExportedFD},
	}
	msg := FormatBlockersByClient(blockers)
	// One line per (client, task), joined by "; ".
	lines := strings.Split(msg, "; ")
	if len(lines) != 2 {
		t.Fatalf("FormatBlockersByClient: got %d lines, want 2: %q", len(lines), msg)
	}
	if !strings.Contains(lines[0], "task 11") || !strings.Contains(lines[0], "2 multicast") {
		t.Errorf("line 0 %q missing task 11 / 2 multicast", lines[0])
	}
	if !strings.Contains(lines[1], "task 12") || !strings.Contains(lines[1], "1 exported-fd") {
		t.Errorf("line 1 %q missing task 12 / 1 exported-fd", lines[1])
	}
}

// TestProcFDInfoExtraFormat locks the fdinfo line format: it is parsed by
// userspace (the checkpoint interposer's exported-object identity oracle),
// making it a de-facto ABI.
func TestProcFDInfoExtraFormat(t *testing.T) {
	ctx := context.Background()
	nvp, _, _, _, _ := newTestClientWithObjects(ctx)
	fd := &frontendFD{
		dev: &frontendDevice{nvp: nvp},
		exportedObj: &exportedObjInfo{
			client: nvgpu.Handle{Val: 0xc1d00001},
			object: nvgpu.Handle{Val: 0xcaf00002},
			class:  nvgpu.ClassID(0x40),
		},
	}
	got := fd.ProcFDInfoExtra(ctx)
	want := "nvproxy_exported_object:\tclient=0xc1d00001 object=0xcaf00002 class=0x40\n"
	if got != want {
		t.Errorf("ProcFDInfoExtra:\n got %q\nwant %q", got, want)
	}
	// No exported object => no extra line.
	fd.exportedObj = nil
	if got := fd.ProcFDInfoExtra(ctx); got != "" {
		t.Errorf("ProcFDInfoExtra without export: got %q, want empty", got)
	}
}

func TestMulticastABIStructSizes(t *testing.T) {
	// Sizes verified against src/common/sdk/nvidia/inc/ctrl/ctrl00fd.h
	// (open-gpu-kernel-modules 535 and 580).
	for _, tc := range []struct {
		name string
		got  int
		want int
	}{
		{"ATTACH_GPU", (&nvgpu.NV00FD_CTRL_ATTACH_GPU_PARAMS{}).SizeBytes(), 16},
		{"ATTACH_MEM", (&nvgpu.NV00FD_CTRL_ATTACH_MEM_PARAMS{}).SizeBytes(), 40},
		{"DETACH_MEM", (&nvgpu.NV00FD_CTRL_DETACH_MEM_PARAMS{}).SizeBytes(), 24},
		{"GET_INFO", (&nvgpu.NV00FD_CTRL_GET_INFO_PARAMS{}).SizeBytes(), 32},
	} {
		if tc.got != tc.want {
			t.Errorf("%s.SizeBytes() = %d, want %d", tc.name, tc.got, tc.want)
		}
	}
}
