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

package nvgpu

import (
	"testing"
	"unsafe"
)

// TestNvfsStructSizes verifies that the nvidia-fs parameter structs have the
// exact sizes the kernel driver expects. nvidia-fs copies fixed-size structs
// based on its own sizeof (the ioctl command encodes only sizeof(int)), so any
// drift here silently corrupts host ioctls. Sizes are from
// github.com/NVIDIA/gds-nvidia-fs:src/nvfs-core.h.
func TestNvfsStructSizes(t *testing.T) {
	for _, tc := range []struct {
		name string
		got  int
		want int
	}{
		{"NvfsIoctlMap", (*NvfsIoctlMap)(nil).SizeBytes(), 48},
		{"NvfsFileArgs", (*NvfsFileArgs)(nil).SizeBytes(), 32},
		{"NvfsIoctlIoargs", (*NvfsIoctlIoargs)(nil).SizeBytes(), 80},
		// The union must be at least as large as its largest member
		// (NvfsIoctlIoargs), since the kernel copies sizeof(union) per ioctl.
		{"NvfsIoctlParamUnion", (*NvfsIoctlParamUnion)(nil).SizeBytes(), 80},
	} {
		if tc.got != tc.want {
			t.Errorf("%s.SizeBytes() = %d, want %d", tc.name, tc.got, tc.want)
		}
	}
}

// TestNvfsFieldOffsets verifies the packed field offsets that Go's natural
// alignment would otherwise get wrong, locking in the kernel's
// __attribute__((packed, aligned(8))) layout.
func TestNvfsFieldOffsets(t *testing.T) {
	var fileArgs NvfsFileArgs
	if got := unsafe.Offsetof(fileArgs.DevPtrOff); got != 20 {
		t.Errorf("NvfsFileArgs.DevPtrOff offset = %d, want 20", got)
	}

	var ioargs NvfsIoctlIoargs
	for _, tc := range []struct {
		name string
		got  uintptr
		want uintptr
	}{
		{"FileArgs", unsafe.Offsetof(ioargs.FileArgs), 40},
		{"FD", unsafe.Offsetof(ioargs.FD), 72},
		{"Flags", unsafe.Offsetof(ioargs.Flags), 76},
	} {
		if tc.got != tc.want {
			t.Errorf("NvfsIoctlIoargs.%s offset = %d, want %d", tc.name, tc.got, tc.want)
		}
	}
}
