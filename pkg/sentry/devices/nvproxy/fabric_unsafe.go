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
	"gvisor.dev/gvisor/pkg/marshal"
)

// controlObjectOnHost issues NV_ESC_RM_CONTROL with command cmd and the given
// marshalable params on the host frontend FD, targeting the object with
// handle `object` in root client `root`. It returns the driver status and any
// ioctl errno. This mirrors the host-side effect of an application
// NV_ESC_RM_CONTROL, but is driven by the sentry (e.g. to replay recorded
// multicast attach controls at restore time).
func controlObjectOnHost(hostFD int32, root, object nvgpu.Handle, cmd uint32, ctrlParams marshal.Marshallable) (uint32, error) {
	buf := make([]byte, ctrlParams.SizeBytes())
	ctrlParams.MarshalBytes(buf)
	ioctlParams := nvgpu.NVOS54_PARAMETERS{
		HClient:    root,
		HObject:    object,
		Cmd:        cmd,
		Params:     p64FromPtr(unsafe.Pointer(&buf[0])),
		ParamsSize: uint32(len(buf)),
	}
	_, _, errno := unix.RawSyscall(unix.SYS_IOCTL, uintptr(hostFD), frontendIoctlCmd(nvgpu.NV_ESC_RM_CONTROL, nvgpu.SizeofNVOS54Parameters), uintptr(unsafe.Pointer(&ioctlParams)))
	// buf must remain alive for the duration of the ioctl; referencing it
	// below also prevents the GC from collecting it earlier.
	ctrlParams.UnmarshalBytes(buf)
	if errno != 0 {
		return ioctlParams.Status, errno
	}
	return ioctlParams.Status, nil
}
