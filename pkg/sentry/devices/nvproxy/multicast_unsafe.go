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
	"runtime"
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
	if len(buf) == 0 {
		return 0, unix.EINVAL
	}
	ctrlParams.MarshalBytes(buf)
	ioctlParams := nvgpu.NVOS54_PARAMETERS{
		HClient:    root,
		HObject:    object,
		Cmd:        cmd,
		Params:     p64FromPtr(unsafe.Pointer(&buf[0])),
		ParamsSize: uint32(len(buf)),
	}
	// unix.Syscall, not RawSyscall: some replayed controls block in the driver
	// (e.g. NV00FD_CTRL_CMD_ATTACH_MEM waits for all participating GPUs), and
	// a blocking ioctl outside entersyscall would wedge the M's P and could
	// stall the scheduler and GC.
	_, _, errno := unix.Syscall(unix.SYS_IOCTL, uintptr(hostFD), frontendIoctlCmd(nvgpu.NV_ESC_RM_CONTROL, nvgpu.SizeofNVOS54Parameters), uintptr(unsafe.Pointer(&ioctlParams)))
	ctrlParams.UnmarshalBytes(buf)
	runtime.KeepAlive(buf)
	if errno != 0 {
		return ioctlParams.Status, errno
	}
	return ioctlParams.Status, nil
}
