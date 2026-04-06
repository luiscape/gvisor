// Copyright 2023 The gVisor Authors.
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
	"golang.org/x/sys/unix"
	"gvisor.dev/gvisor/pkg/abi/linux"
	"gvisor.dev/gvisor/pkg/abi/nvgpu"
	"gvisor.dev/gvisor/pkg/seccomp"
	"gvisor.dev/gvisor/pkg/sentry/devices/nvproxy/nvconf"
)

// Shorthands for NVIDIA driver capabilities.
const (
	// Shorthand for compute+utility capabilities.
	// This is the default set of capabilities when capabilities are not
	// explicitly specified, and using a shorthand for this makes ABI
	// definitions in `version.go` more readable.
	compUtil = nvconf.CapCompute | nvconf.CapUtility
)

func frontendIoctlFilters(enabledCaps nvconf.DriverCaps) []seccomp.SyscallRule {
	const (
		// for ioctls taking arbitrary size
		notIocSizeMask = ^(((uintptr(1) << linux.IOC_SIZEBITS) - 1) << linux.IOC_SIZESHIFT)
	)
	var ioctlRules []seccomp.SyscallRule
	for _, feIoctl := range []struct {
		arg1 seccomp.ValueMatcher
		caps nvconf.DriverCaps
	}{
		{seccomp.MaskedEqual(notIocSizeMask, frontendIoctlCmd(nvgpu.NV_ESC_CARD_INFO, 0)), compUtil},
		{seccomp.EqualTo(frontendIoctlCmd(nvgpu.NV_ESC_CHECK_VERSION_STR, nvgpu.SizeofRMAPIVersion)), compUtil},
		{seccomp.MaskedEqual(notIocSizeMask, frontendIoctlCmd(nvgpu.NV_ESC_ATTACH_GPUS_TO_FD, 0)), compUtil},
		{seccomp.EqualTo(frontendIoctlCmd(nvgpu.NV_ESC_REGISTER_FD, nvgpu.SizeofIoctlRegisterFD)), compUtil},
		{seccomp.EqualTo(frontendIoctlCmd(nvgpu.NV_ESC_ALLOC_OS_EVENT, nvgpu.SizeofIoctlAllocOSEvent)), compUtil},
		{seccomp.EqualTo(frontendIoctlCmd(nvgpu.NV_ESC_FREE_OS_EVENT, nvgpu.SizeofIoctlFreeOSEvent)), compUtil},
		{seccomp.EqualTo(frontendIoctlCmd(nvgpu.NV_ESC_SYS_PARAMS, nvgpu.SizeofIoctlSysParams)), compUtil},
		{seccomp.EqualTo(frontendIoctlCmd(nvgpu.NV_ESC_WAIT_OPEN_COMPLETE, nvgpu.SizeofIoctlWaitOpenComplete)), compUtil},
		{seccomp.EqualTo(frontendIoctlCmd(nvgpu.NV_ESC_RM_ALLOC_MEMORY, nvgpu.SizeofIoctlNVOS02ParametersWithFD)), compUtil | nvconf.CapGraphics},
		{seccomp.EqualTo(frontendIoctlCmd(nvgpu.NV_ESC_RM_FREE, nvgpu.SizeofNVOS00Parameters)), compUtil},
		{seccomp.EqualTo(frontendIoctlCmd(nvgpu.NV_ESC_RM_CONTROL, nvgpu.SizeofNVOS54Parameters)), compUtil},
		{seccomp.EqualTo(frontendIoctlCmd(nvgpu.NV_ESC_RM_ALLOC, nvgpu.SizeofNVOS64Parameters)), compUtil},
		{seccomp.EqualTo(frontendIoctlCmd(nvgpu.NV_ESC_RM_DUP_OBJECT, nvgpu.SizeofNVOS55Parameters)), compUtil},
		{seccomp.EqualTo(frontendIoctlCmd(nvgpu.NV_ESC_RM_SHARE, nvgpu.SizeofNVOS57Parameters)), compUtil},
		{seccomp.EqualTo(frontendIoctlCmd(nvgpu.NV_ESC_RM_IDLE_CHANNELS, nvgpu.SizeofNVOS30Parameters)), nvconf.CapGraphics},
		{seccomp.EqualTo(frontendIoctlCmd(nvgpu.NV_ESC_RM_VID_HEAP_CONTROL, nvgpu.SizeofNVOS32Parameters)), compUtil},
		{seccomp.EqualTo(frontendIoctlCmd(nvgpu.NV_ESC_RM_MAP_MEMORY, nvgpu.SizeofIoctlNVOS33ParametersWithFD)), compUtil},
		{seccomp.EqualTo(frontendIoctlCmd(nvgpu.NV_ESC_RM_UNMAP_MEMORY, nvgpu.SizeofNVOS34Parameters)), compUtil},
		{seccomp.EqualTo(frontendIoctlCmd(nvgpu.NV_ESC_RM_ALLOC_CONTEXT_DMA2, nvgpu.SizeofNVOS39Parameters)), nvconf.CapGraphics},
		{seccomp.MaskedEqual(notIocSizeMask, frontendIoctlCmd(nvgpu.NV_ESC_RM_MAP_MEMORY_DMA, 0)), nvconf.CapGraphics | nvconf.CapVideo},
		{seccomp.MaskedEqual(notIocSizeMask, frontendIoctlCmd(nvgpu.NV_ESC_RM_UNMAP_MEMORY_DMA, 0)), nvconf.CapGraphics | nvconf.CapVideo},
		{seccomp.EqualTo(frontendIoctlCmd(nvgpu.NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO, nvgpu.SizeofNVOS56Parameters)), compUtil},
	} {
		if feIoctl.caps&enabledCaps != 0 {
			ioctlRules = append(ioctlRules, seccomp.PerArg{
				seccomp.NonNegativeFD{},
				feIoctl.arg1,
			})
		}
	}
	return ioctlRules
}

func uvmIoctlFilters(enabledCaps nvconf.DriverCaps) []seccomp.SyscallRule {
	var ioctlRules []seccomp.SyscallRule
	for _, uvmIoctl := range []struct {
		arg1 seccomp.ValueMatcher
		caps nvconf.DriverCaps
	}{
		{seccomp.EqualTo(nvgpu.UVM_INITIALIZE), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_MM_INITIALIZE), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_DEINITIALIZE), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_CREATE_RANGE_GROUP), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_DESTROY_RANGE_GROUP), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_REGISTER_GPU_VASPACE), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_UNREGISTER_GPU_VASPACE), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_REGISTER_CHANNEL), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_UNREGISTER_CHANNEL), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_ENABLE_PEER_ACCESS), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_DISABLE_PEER_ACCESS), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_SET_RANGE_GROUP), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_MAP_EXTERNAL_ALLOCATION), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_FREE), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_REGISTER_GPU), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_UNREGISTER_GPU), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_PAGEABLE_MEM_ACCESS), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_SET_PREFERRED_LOCATION), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_UNSET_PREFERRED_LOCATION), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_ENABLE_READ_DUPLICATION), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_DISABLE_READ_DUPLICATION), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_SET_ACCESSED_BY), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_UNSET_ACCESSED_BY), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_MIGRATE), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_MIGRATE_RANGE_GROUP), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_TOOLS_READ_PROCESS_MEMORY), nvconf.ValidCapabilities},
		{seccomp.EqualTo(nvgpu.UVM_TOOLS_WRITE_PROCESS_MEMORY), nvconf.ValidCapabilities},
		{seccomp.EqualTo(nvgpu.UVM_MAP_DYNAMIC_PARALLELISM_REGION), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_UNMAP_EXTERNAL), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_PAGEABLE_MEM_ACCESS_ON_GPU), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_POPULATE_PAGEABLE), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_ALLOC_SEMAPHORE_POOL), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_VALIDATE_VA_RANGE), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_CREATE_EXTERNAL_RANGE), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_MAP_EXTERNAL_SPARSE), compUtil},
		{seccomp.EqualTo(nvgpu.UVM_ALLOC_DEVICE_P2P), compUtil},
	} {
		if uvmIoctl.caps&enabledCaps != 0 {
			ioctlRules = append(ioctlRules, seccomp.PerArg{
				seccomp.NonNegativeFD{},
				uvmIoctl.arg1,
			})
		}
	}
	return ioctlRules
}

// Filters returns seccomp-bpf filters for this package when using the given
// set of capabilities.
func Filters(enabledCaps nvconf.DriverCaps) seccomp.SyscallRules {
	var ioctlRules []seccomp.SyscallRule
	ioctlRules = append(ioctlRules, frontendIoctlFilters(enabledCaps)...)
	ioctlRules = append(ioctlRules, uvmIoctlFilters(enabledCaps)...)
	return seccomp.MakeSyscallRules(map[uintptr]seccomp.SyscallRule{
		unix.SYS_IOCTL: seccomp.Or(ioctlRules),
		// SYS_OPENAT and SYS_CLOSE are needed to re-open nvidia device
		// files during checkpoint/restore (afterLoadImpl).
		unix.SYS_OPENAT: seccomp.MatchAll{},
		unix.SYS_CLOSE:  seccomp.MatchAll{},
		// The following syscalls are needed to fork+exec the GPU
		// checkpoint restore helper binary (cuda_checkpoint_helper)
		// from the sentry process during restore.  Go's os/exec
		// package uses pidfd, vfork, waitid, and various setup
		// syscalls that the base seccomp filter restricts.
		unix.SYS_EXECVE:            seccomp.MatchAll{},
		unix.SYS_CLONE:             seccomp.MatchAll{},
		unix.SYS_CLONE3:            seccomp.MatchAll{},
		unix.SYS_WAIT4:             seccomp.MatchAll{},
		unix.SYS_WAITID:            seccomp.MatchAll{},
		unix.SYS_PIPE2:             seccomp.MatchAll{},
		unix.SYS_DUP3:              seccomp.MatchAll{},
		unix.SYS_READLINKAT:        seccomp.MatchAll{},
		unix.SYS_NEWFSTATAT:        seccomp.MatchAll{},
		unix.SYS_PIDFD_OPEN:        seccomp.MatchAll{},
		unix.SYS_PIDFD_GETFD:       seccomp.MatchAll{},
		424:                        seccomp.MatchAll{}, // SYS_PIDFD_SEND_SIGNAL
		unix.SYS_FCNTL:             seccomp.MatchAll{},
		unix.SYS_RT_SIGACTION:      seccomp.MatchAll{},
		unix.SYS_RT_SIGPROCMASK:    seccomp.MatchAll{},
		unix.SYS_SIGALTSTACK:       seccomp.MatchAll{},
		unix.SYS_GETPID:            seccomp.MatchAll{},
		unix.SYS_GETTID:            seccomp.MatchAll{},
		unix.SYS_SET_TID_ADDRESS:   seccomp.MatchAll{},
		unix.SYS_SET_ROBUST_LIST:   seccomp.MatchAll{},
		unix.SYS_PRCTL:             seccomp.MatchAll{},
		unix.SYS_RSEQ:              seccomp.MatchAll{},
		unix.SYS_MPROTECT:          seccomp.MatchAll{},
		unix.SYS_MEMBARRIER:        seccomp.MatchAll{},
		unix.SYS_SCHED_GETAFFINITY: seccomp.MatchAll{},
		unix.SYS_READ:              seccomp.MatchAll{},
		unix.SYS_WRITE:             seccomp.MatchAll{},
		unix.SYS_MMAP: seccomp.PerArg{
			seccomp.AnyValue{},
			seccomp.AnyValue{},
			seccomp.AnyValue{},
			seccomp.EqualTo(unix.MAP_SHARED | unix.MAP_FIXED_NOREPLACE),
		},
		unix.SYS_MREMAP: seccomp.PerArg{
			seccomp.AnyValue{},
			seccomp.EqualTo(0), /* old_size */
			seccomp.AnyValue{},
			seccomp.EqualTo(linux.MREMAP_MAYMOVE | linux.MREMAP_FIXED),
			seccomp.AnyValue{},
			seccomp.EqualTo(0),
		},
	})
}
