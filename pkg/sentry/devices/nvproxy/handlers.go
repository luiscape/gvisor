// Copyright 2024 The gVisor Authors.
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
	"sync"

	"gvisor.dev/gvisor/pkg/abi/nvgpu"
	"gvisor.dev/gvisor/pkg/sentry/devices/nvproxy/nvconf"
)

// errHandler is an error returned by an ioctl handler function.
type errHandler struct {
	Err string
}

// Error implements the error interface.
func (e *errHandler) Error() string {
	return e.Err
}

// ioctl handler errors.
var (
	errUndefinedHandler  = errHandler{"handler is undefined"}
	errMissingCapability = errHandler{"missing capability"}
)

type frontendIoctlHandler struct {
	// handler is the function to call if a capability in capSet is enabled.
	handler func(*frontendIoctlState) (uintptr, error)
	// capSet is a bitmask of capabilities that this handler is available for.
	capSet nvconf.DriverCaps
	// fastPath indicates this handler is a simple passthrough whose
	// parameters are flat byte blobs requiring no pointer translation, FD
	// translation, or object tracking. When set, frontendFD.Ioctl may use
	// a streamlined copy-in / host-ioctl / copy-out sequence that avoids
	// intermediate allocations and reduces function-call depth.
	fastPath bool
}

// feHandler returns a frontendIoctlHandler that wraps the given function.
// The handler will be called if any of the given capabilities are enabled.
func feHandler(handler func(*frontendIoctlState) (uintptr, error), caps nvconf.DriverCaps) frontendIoctlHandler {
	return frontendIoctlHandler{
		handler: handler,
		capSet:  caps,
	}
}

// feHandlerFast returns a frontendIoctlHandler marked for fast-path
// dispatch. Fast-path handlers are simple passthroughs: parameters are
// copied in from guest memory as an opaque byte blob, the host ioctl is
// invoked via RawSyscall, and parameters are copied back out. No pointer
// translation, FD translation, or object tracking is performed, so
// frontendFD.Ioctl can bypass the normal handler call chain for these.
func feHandlerFast(handler func(*frontendIoctlState) (uintptr, error), caps nvconf.DriverCaps) frontendIoctlHandler {
	return frontendIoctlHandler{
		handler:  handler,
		capSet:   caps,
		fastPath: true,
	}
}

// handle calls the handler if the capability is enabled.
// Returns errMissingCapability if the caller is missing the required
// capabilities for this handler.
// Returns errUndefinedHandler if the handler does not exist.
func (h frontendIoctlHandler) handle(fi *frontendIoctlState) (uintptr, error) {
	if h.handler == nil {
		return 0, &errUndefinedHandler
	}
	if h.capSet&fi.fd.dev.nvp.capsEnabled == 0 {
		return 0, &errMissingCapability
	}
	return h.handler(fi)
}

type controlCmdHandler struct {
	// handler is the function to call if a capability in capSet is enabled.
	handler func(*frontendIoctlState, *nvgpu.NVOS54_PARAMETERS) (uintptr, error)
	// capSet is a bitmask of capabilities that this handler is available for.
	capSet nvconf.DriverCaps
}

// ctrlHandler returns a controlCmdHandler that wraps the given function.
// The handler will be called if any of the given capabilities are enabled.
func ctrlHandler(handler func(*frontendIoctlState, *nvgpu.NVOS54_PARAMETERS) (uintptr, error), caps nvconf.DriverCaps) controlCmdHandler {
	return controlCmdHandler{
		handler: handler,
		capSet:  caps,
	}
}

// handle calls the handler if the capability is enabled.
// Returns errMissingCapability if the caller is missing the required
// capabilities for this handler.
// Returns errUndefinedHandler if the handler does not exist.
func (h controlCmdHandler) handle(fi *frontendIoctlState, params *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	if h.handler == nil {
		return 0, &errUndefinedHandler
	}
	if h.capSet&fi.fd.dev.nvp.capsEnabled == 0 {
		return 0, &errMissingCapability
	}
	return h.handler(fi, params)
}

type allocationClassHandler struct {
	// handler is the function to call if a capability in capSet is enabled.
	handler func(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS64_PARAMETERS, isNVOS64 bool) (uintptr, error)
	// capSet is a bitmask of capabilities that this handler is available for.
	capSet nvconf.DriverCaps
}

// allocHandler returns a allocationClassHandler that wraps the given function.
// The handler will be called if any of the given capabilities are enabled.
func allocHandler(handler func(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS64_PARAMETERS, isNVOS64 bool) (uintptr, error), caps nvconf.DriverCaps) allocationClassHandler {
	return allocationClassHandler{
		handler: handler,
		capSet:  caps,
	}
}

// handle calls the handler if the capability is enabled.
// Returns errMissingCapability if the caller is missing the required
// capabilities for this handler.
// Returns errUndefinedHandler if the handler does not exist.
func (h allocationClassHandler) handle(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS64_PARAMETERS, isNVOS64 bool) (uintptr, error) {
	if h.handler == nil {
		return 0, &errUndefinedHandler
	}
	if h.capSet&fi.fd.dev.nvp.capsEnabled == 0 {
		return 0, &errMissingCapability
	}
	return h.handler(fi, ioctlParams, isNVOS64)
}

type uvmIoctlHandler struct {
	// handler is the function to call if a capability in capSet is enabled.
	handler func(*uvmIoctlState) (uintptr, error)
	// capSet is a bitmask of capabilities that this handler is available for.
	capSet nvconf.DriverCaps
}

// uvmHandler returns a uvmIoctlHandler that wraps the given function.
// The handler will be called if any of the given capabilities are enabled.
func uvmHandler(handler func(*uvmIoctlState) (uintptr, error), caps nvconf.DriverCaps) uvmIoctlHandler {
	return uvmIoctlHandler{
		handler: handler,
		capSet:  caps,
	}
}

// handle calls the handler if the capability is enabled.
// Returns errMissingCapability if the caller is missing the required
// capabilities for this handler.
// Returns errUndefinedHandler if the handler does not exist.
func (h uvmIoctlHandler) handle(ui *uvmIoctlState) (uintptr, error) {
	if h.handler == nil {
		return 0, &errUndefinedHandler
	}
	if h.capSet&ui.fd.dev.nvp.capsEnabled == 0 {
		return 0, &errMissingCapability
	}
	return h.handler(ui)
}

// ---------------------------------------------------------------------------
// Fast-path infrastructure for reducing per-ioctl overhead.
//
// Pageable GPU memory transfers (cudaMemcpy on non-pinned memory) use a
// tight 3-ioctl-per-chunk loop. At ~256 chunks for 1 GB, the 768 ioctls
// accumulate significant overhead from the Sentry round-trip if each one
// traverses the full handler dispatch chain and heap-allocates parameter
// buffers. The constants and pools below support a streamlined path.
// ---------------------------------------------------------------------------

const (
	// rmControlParamsPoolBufSize is the capacity of pooled byte buffers
	// used by rmControlSimple for control-command parameter copying.
	// Buffers up to this size are recycled via sync.Pool, eliminating
	// per-ioctl heap allocation for the DMA-transfer hot loop. Larger
	// requests fall back to make([]byte, n).
	//
	// 4 KiB covers virtually all control commands observed during
	// pageable transfers; RMAPI_PARAM_COPY_MAX_PARAMS_SIZE (1 MiB) is
	// the driver's upper bound, but such sizes are exceedingly rare.
	rmControlParamsPoolBufSize = 4096
)

// rmControlParamsPool recycles byte buffers used by rmControlSimple to
// copy control-command parameters between guest memory and host ioctl
// invocations. This eliminates the per-ioctl make([]byte, ParamsSize)
// allocation that otherwise creates GC pressure on the hot transfer path.
var rmControlParamsPool = sync.Pool{
	New: func() any {
		b := make([]byte, rmControlParamsPoolBufSize)
		return &b
	},
}

// getRmControlBuf returns a byte slice of the requested size for
// rmControlSimple parameter copying. If size <= rmControlParamsPoolBufSize
// a recycled buffer is returned and pooled will be true; the caller must
// pass the buffer to putRmControlBuf when done. Otherwise a fresh heap
// allocation is returned and pooled is false.
func getRmControlBuf(size uint32) (buf []byte, pooled bool) {
	if size <= rmControlParamsPoolBufSize {
		bp := rmControlParamsPool.Get().(*[]byte)
		return (*bp)[:size], true
	}
	return make([]byte, size), false
}

// putRmControlBuf returns a pooled buffer to rmControlParamsPool. It is
// safe to call with any buffer; non-pooled buffers (cap < pool size) are
// silently ignored.
func putRmControlBuf(buf []byte) {
	if cap(buf) < rmControlParamsPoolBufSize {
		return
	}
	b := buf[:cap(buf)]
	rmControlParamsPool.Put(&b)
}
