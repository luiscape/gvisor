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

//go:build cgo

package boot

/*
#cgo LDFLAGS: -ldl

#include <dlfcn.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

// CUDA types.
typedef int CUresult;
typedef CUresult (*cuInit_fn)(unsigned int);
typedef CUresult (*cuCheckpointProcessRestore_fn)(int pid, void *args);
typedef CUresult (*cuCheckpointProcessUnlock_fn)(int pid, void *args);

// gpuRestoreState holds resolved function pointers.
static void *g_cuda_handle;
static cuInit_fn                       g_cuInit;
static cuCheckpointProcessRestore_fn   g_cuRestore;
static cuCheckpointProcessUnlock_fn    g_cuUnlock;

// gpuRestoreLoadCuda loads libcuda.so and resolves checkpoint symbols.
// Returns 0 on success, -1 if libcuda is not found (not an error for
// non-GPU containers), -2 if checkpoint API symbols are missing.
static int gpuRestoreLoadCuda(char *errbuf, int errbuf_len) {
    const char *paths[] = {
        "libcuda.so.1",
        "libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
        "/usr/local/nvidia/lib64/libcuda.so.1",
        "/usr/local/cuda/lib64/libcuda.so.1",
        "/host/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        NULL
    };

    for (int i = 0; paths[i]; i++) {
        g_cuda_handle = dlopen(paths[i], RTLD_LAZY | RTLD_GLOBAL);
        if (g_cuda_handle) break;
    }
    if (!g_cuda_handle) {
        snprintf(errbuf, errbuf_len, "libcuda.so not found: %s", dlerror());
        return -1;
    }

    g_cuInit = (cuInit_fn)dlsym(g_cuda_handle, "cuInit");
    if (!g_cuInit) {
        snprintf(errbuf, errbuf_len, "cuInit not found in libcuda.so");
        return -2;
    }

    g_cuRestore = (cuCheckpointProcessRestore_fn)dlsym(
        g_cuda_handle, "cuCheckpointProcessRestore");
    g_cuUnlock = (cuCheckpointProcessUnlock_fn)dlsym(
        g_cuda_handle, "cuCheckpointProcessUnlock");

    if (!g_cuRestore || !g_cuUnlock) {
        snprintf(errbuf, errbuf_len,
            "CUDA checkpoint API not available (Restore=%p Unlock=%p)",
            (void*)g_cuRestore, (void*)g_cuUnlock);
        return -2;
    }
    return 0;
}

// gpuRestoreInit calls cuInit(0).
// Returns the CUresult (0 = success).
static int gpuRestoreInit(void) {
    return (int)g_cuInit(0);
}

// gpuRestoreProcess calls cuCheckpointProcessRestore + Unlock for one PID.
// Returns 0 on success, the CUresult on failure.
static int gpuRestoreProcess(int pid, char *errbuf, int errbuf_len) {
    CUresult r = g_cuRestore(pid, NULL);
    if (r != 0) {
        snprintf(errbuf, errbuf_len,
            "cuCheckpointProcessRestore(pid=%d) failed: %d", pid, (int)r);
        return (int)r;
    }

    r = g_cuUnlock(pid, NULL);
    if (r != 0) {
        snprintf(errbuf, errbuf_len,
            "cuCheckpointProcessUnlock(pid=%d) failed: %d", pid, (int)r);
        return (int)r;
    }
    return 0;
}

// gpuRestoreGetPid returns getpid() from C (the host PID of the sentry).
static int gpuRestoreGetPid(void) {
    return (int)getpid();
}
*/
import "C"

import (
	"fmt"
	"time"
	"unsafe"

	"gvisor.dev/gvisor/pkg/log"
)

// gpuRestoreCgoAvailable is true when this file is compiled (cgo build).
const gpuRestoreCgoAvailable = true

// RestoreGPUCheckpointInProcess loads libcuda.so via dlopen and calls
// cuCheckpointProcessRestore + cuCheckpointProcessUnlock directly in the
// sentry process.  This runs BEFORE the kernel starts application threads
// (before onStart()), so there is:
//
//   - No race with application threads issuing CUDA ioctls
//   - No nvidia driver per-process mutex contention (we ARE the only thread)
//   - No mm mismatch (current->mm is the sentry's mm, which the UVM driver
//     registered via UVM_MM_INITIALIZE)
//
// The function is a no-op if libcuda.so is not found (non-GPU container)
// or if the CUDA checkpoint API is not available (old driver).
//
// Called from restore.go after kernel state is loaded but before onStart().
func RestoreGPUCheckpointInProcess() error {
	start := time.Now()
	log.Infof("gpu-restore: starting in-process CUDA checkpoint restore (sentry PID=%d)",
		int(C.gpuRestoreGetPid()))

	// Step 1: Load libcuda.so and resolve symbols.
	errbuf := make([]byte, 512)
	rc := C.gpuRestoreLoadCuda((*C.char)(unsafe.Pointer(&errbuf[0])), C.int(len(errbuf)))
	if rc == -1 {
		// libcuda not found — not a GPU container, nothing to do.
		log.Infof("gpu-restore: %s (non-GPU container, skipping)", C.GoString((*C.char)(unsafe.Pointer(&errbuf[0]))))
		return nil
	}
	if rc == -2 {
		// Checkpoint API not available — old driver.
		log.Warningf("gpu-restore: %s (skipping GPU restore)", C.GoString((*C.char)(unsafe.Pointer(&errbuf[0]))))
		return nil
	}
	log.Infof("gpu-restore: libcuda.so loaded, checkpoint API resolved")

	// Step 2: cuInit(0) — initialize the CUDA driver in the sentry process.
	// This opens fresh nvidia device FDs that belong to the sentry.
	// Since app threads are not running yet and restored FDs have hostFD=-1,
	// there is no driver-level lock contention.
	initResult := C.gpuRestoreInit()
	if initResult != 0 {
		return fmt.Errorf("gpu-restore: cuInit failed: %d", int(initResult))
	}
	log.Infof("gpu-restore: cuInit succeeded")

	// Step 3: Call cuCheckpointProcessRestore targeting our own PID.
	//
	// The sentry IS the host process that owns all nvidia FDs and all
	// CUDA checkpoint data (which cuCheckpointProcessCheckpoint placed
	// in our address space during the save phase).  Passing getpid()
	// tells the CUDA driver to look in our own memory for checkpoint
	// data and restore GPU state.
	pid := int(C.gpuRestoreGetPid())
	log.Infof("gpu-restore: calling cuCheckpointProcessRestore(pid=%d)...", pid)

	errbuf2 := make([]byte, 512)
	rc2 := C.gpuRestoreProcess(C.int(pid),
		(*C.char)(unsafe.Pointer(&errbuf2[0])), C.int(len(errbuf2)))
	if rc2 != 0 {
		errMsg := C.GoString((*C.char)(unsafe.Pointer(&errbuf2[0])))
		log.Warningf("gpu-restore: %s", errMsg)
		// Don't fail the entire restore — the container can still run,
		// just without GPU state.  The application may reinitialize.
		return fmt.Errorf("gpu-restore: %s", errMsg)
	}

	elapsed := time.Since(start)
	log.Infof("gpu-restore: GPU state restored and unlocked in %v", elapsed.Round(time.Millisecond))
	return nil
}
