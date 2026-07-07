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
)

// nvfsIoctlInvoke forwards an nvidia-fs ioctl to the host /dev/nvidia-fs FD.
// The command is forwarded verbatim; nvidia-fs encodes sizeof(int) in the
// command rather than the parameter size, so the caller-supplied command is
// already correct for the host.
func nvfsIoctlInvoke[Params any](ni *nvfsIoctlState, ioctlParams *Params) (uintptr, error) {
	n, _, errno := unix.RawSyscall(unix.SYS_IOCTL, uintptr(ni.fd.hostFD), uintptr(ni.cmd), uintptr(unsafe.Pointer(ioctlParams)))
	if errno != 0 {
		return n, errno
	}
	return n, nil
}
