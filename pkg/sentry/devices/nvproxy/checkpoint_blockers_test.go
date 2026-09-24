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
	if got := nvp.checkpointBlockers(); got != "" {
		t.Fatalf("no blockers expected, got %q", got)
	}

	addObj(rank0, 0x5c000003, nvgpu.NV_MEMORY_MULTICAST_FABRIC)
	addObj(rank0, 0x5c000004, nvgpu.NV_MEMORY_MULTICAST_FABRIC)
	addObj(rank1, 0x5c000005, nvgpu.NV_MEMORY_FABRIC_IMPORTED_REF)
	want := "PID 41 (client 0xc1d00001): 2 multicast; PID 42 (client 0xc1d00002): 1 fabric-import"
	if got := nvp.checkpointBlockers(); got != want {
		t.Fatalf("got %q, want %q", got, want)
	}

	// Freed objects drop out.
	rank0.objsMu.Lock()
	nvp.objFree(ctx, rank0, nvgpu.Handle{Val: 0x5c000003})
	rank0.objsMu.Unlock()
	want = "PID 41 (client 0xc1d00001): 1 multicast; PID 42 (client 0xc1d00002): 1 fabric-import"
	if got := nvp.checkpointBlockers(); got != want {
		t.Fatalf("after free: got %q, want %q", got, want)
	}
}
