// Copyright 2025 The gVisor Authors.
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
	"unsafe"

	"golang.org/x/sys/unix"
	"gvisor.dev/gvisor/pkg/abi/nvgpu"
)

// freeObjectOnHost issues NV_ESC_RM_FREE on the given host frontend FD to free
// the object with handle `object` (child of `parent`) in root client `root`.
// It returns the driver status and any ioctl errno. This mirrors the host-side
// effect of rmFree(), but is driven by the sentry rather than the application.
func freeObjectOnHost(hostFD int32, root, parent, object nvgpu.Handle) (uint32, error) {
	params := nvgpu.NVOS00_PARAMETERS{
		HRoot:         root,
		HObjectParent: parent,
		HObjectOld:    object,
	}
	if _, _, errno := unix.RawSyscall(unix.SYS_IOCTL, uintptr(hostFD), frontendIoctlCmd(nvgpu.NV_ESC_RM_FREE, nvgpu.SizeofNVOS00Parameters), uintptr(unsafe.Pointer(&params))); errno != 0 {
		return params.Status, errno
	}
	return params.Status, nil
}

// freeUvmRangeOnHost issues UVM_FREE on the given host nvidia-uvm FD to free
// the VA range starting at base (previously created via
// UVM_CREATE_EXTERNAL_RANGE), tearing down any external allocations mapped into
// it. It returns the driver status and any ioctl errno. The V590 param layout
// (base only) matches driver R590+ (including R610).
func freeUvmRangeOnHost(hostFD int32, base uint64) (uint32, error) {
	params := nvgpu.UVM_FREE_PARAMS_V590{
		Base: base,
	}
	if _, _, errno := unix.RawSyscall(unix.SYS_IOCTL, uintptr(hostFD), uintptr(nvgpu.UVM_FREE), uintptr(unsafe.Pointer(&params))); errno != 0 {
		return params.RMStatus, errno
	}
	return params.RMStatus, nil
}
