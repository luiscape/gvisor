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
	"unsafe"

	"golang.org/x/sys/unix"
	"gvisor.dev/gvisor/pkg/abi/nvgpu"
)

// rmFreeOnHost issues NV_ESC_RM_FREE for the object with the given handle in
// the given root client, on the host frontend FD. It mirrors the host-side
// effect of an application NV_ESC_RM_FREE, but is driven by the sentry (to
// free driver resources that cuda-checkpoint cannot serialize and that the
// application cannot release itself; see SuspendFLARegistrations).
func rmFreeOnHost(hostFD int32, client, object nvgpu.Handle) (uint32, error) {
	ioctlParams := nvgpu.NVOS00_PARAMETERS{
		HRoot:         client,
		HObjectParent: client,
		HObjectOld:    object,
	}
	if _, _, errno := unix.Syscall(unix.SYS_IOCTL, uintptr(hostFD), frontendIoctlCmd(nvgpu.NV_ESC_RM_FREE, nvgpu.SizeofNVOS00Parameters), uintptr(unsafe.Pointer(&ioctlParams))); errno != 0 {
		return ioctlParams.Status, errno
	}
	return ioctlParams.Status, nil
}
