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
		clients: make(map[nvgpu.Handle]*rootClient),
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
	client.objAddDep(mcH, memH)

	if got := len(mcObj.attachedGPUs); got != 2 {
		t.Errorf("attachedGPUs: got %d, want 2", got)
	}
	if got := len(mcObj.attachedMems); got != 2 {
		t.Errorf("attachedMems: got %d, want 2", got)
	}
	if mcObj.attachedGPUs[1].devMinor != 1 {
		t.Errorf("attachedGPUs[1].devMinor: got %d, want 1", mcObj.attachedGPUs[1].devMinor)
	}

	// The multicast object must depend on the attached memory so that the
	// topological sort restores memory first.
	memObj := client.resources[memH]
	if _, ok := client.resources[mcH].deps[memObj]; !ok {
		t.Errorf("multicast object does not depend on attached memory")
	}

	// Detach removes exactly the matching (subdevice, offset) record.
	mcObj.removeAttachMem(nvgpu.NV00FD_CTRL_DETACH_MEM_PARAMS{HSubDevice: subdev0, Offset: 0})
	if got := len(mcObj.attachedMems); got != 1 {
		t.Fatalf("attachedMems after detach: got %d, want 1", got)
	}
	if mcObj.attachedMems[0].params.HSubDevice != subdev1 {
		t.Errorf("wrong attach removed: remaining subdevice %v, want %v", mcObj.attachedMems[0].params.HSubDevice, subdev1)
	}
	// Detaching a non-existent binding must not remove anything.
	mcObj.removeAttachMem(nvgpu.NV00FD_CTRL_DETACH_MEM_PARAMS{HSubDevice: subdev0, Offset: 0xdead0000})
	if got := len(mcObj.attachedMems); got != 1 {
		t.Errorf("attachedMems after bogus detach: got %d, want 1", got)
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
	if b.ObjectHandle != mcH || b.Kind != "multicast" || b.Class != nvgpu.NV_MEMORY_MULTICAST_FABRIC {
		t.Errorf("unexpected blocker: %v", &b)
	}

	// Add a fabric object and an imported ref; both must be reported with
	// their own kinds.
	client.objsMu.Lock()
	fabH := nvgpu.Handle{Val: 0x5c000020}
	nvp.objAdd(ctx, client, fabH, nvgpu.NV_MEMORY_FABRIC, &miscObject{}, client.handle)
	impH := nvgpu.Handle{Val: 0x5c000021}
	nvp.objAdd(ctx, client, impH, nvgpu.NV_MEMORY_FABRIC_IMPORTED_REF, &miscObject{}, client.handle)
	client.objsMu.Unlock()

	blockers = nvp.checkpointBlockers()
	if len(blockers) != 3 {
		t.Fatalf("checkpointBlockers: got %d, want 3: %v", len(blockers), blockers)
	}
	kinds := make(map[string]int)
	for _, b := range blockers {
		kinds[b.Kind]++
	}
	if kinds["multicast"] != 1 || kinds["fabric"] != 1 || kinds["fabric-import"] != 1 {
		t.Errorf("unexpected blocker kinds: %v", kinds)
	}

	// Per-client formatting names the client and aggregates kinds.
	msg := FormatBlockersByClient(blockers)
	if !strings.Contains(msg, client.handle.String()) {
		t.Errorf("blocker message %q does not name client %v", msg, client.handle)
	}
	for _, want := range []string{"1 multicast", "1 fabric", "1 fabric-import"} {
		if !strings.Contains(msg, want) {
			t.Errorf("blocker message %q missing %q", msg, want)
		}
	}

	// Freeing the objects clears the blockers.
	client.objsMu.Lock()
	for _, h := range []nvgpu.Handle{mcH, fabH, impH} {
		nvp.objFree(ctx, client, h)
	}
	client.objsMu.Unlock()
	if blockers = nvp.checkpointBlockers(); len(blockers) != 0 {
		t.Errorf("checkpointBlockers after free: got %v, want none", blockers)
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
