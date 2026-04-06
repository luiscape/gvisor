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

//go:build !false
// +build !false

package nvproxy

import (
	goContext "context"
	"fmt"
	"path/filepath"

	"golang.org/x/sys/unix"
	"gvisor.dev/gvisor/pkg/fdnotifier"
	"gvisor.dev/gvisor/pkg/log"
	"gvisor.dev/gvisor/pkg/waiter"
)

func (nvp *nvproxy) beforeSaveImpl() {
	// Stage GPU data BEFORE closing host FDs — the UVM ioctls require a
	// live host FD to read device memory.
	if err := nvp.StageGPUData(); err != nil {
		log.Warningf("nvproxy: beforeSave: StageGPUData failed: %v (continuing without GPU data)", err)
	}

	nvp.fdsMu.Lock()
	feCount := len(nvp.frontendFDs)
	for fd := range nvp.frontendFDs {
		if fd.hostFD >= 0 {
			fdnotifier.RemoveFD(fd.hostFD)
			unix.Close(int(fd.hostFD))
			fd.hostFD = -1
			fd.memmapFile.hostFD = -1
		}
	}
	uvmCount := len(nvp.uvmFDs)
	for fd := range nvp.uvmFDs {
		if fd.hostFD >= 0 {
			fdnotifier.RemoveFD(fd.hostFD)
			unix.Close(int(fd.hostFD))
			fd.hostFD = -1
		}
	}
	nvp.fdsMu.Unlock()

	nvp.clientsMu.RLock()
	clientCount := len(nvp.clients)
	nvp.clientsMu.RUnlock()

	log.Infof("nvproxy: beforeSave: neutralized %d frontend + %d UVM FDs, %d live clients, %d MiB staged",
		feCount, uvmCount, clientCount, nvp.StagedDataBytes()>>20)
}

func (nvp *nvproxy) afterLoadImpl(goContext.Context) {
	nvp.frontendFDs = make(map[*frontendFD]struct{})
	nvp.uvmFDs = make(map[*uvmFD]struct{})

	// Arm the ioctl gate so that application-issued nvidia ioctls on
	// restored FDs return EAGAIN until the save-restore-exec helper has
	// finished calling cuCheckpointProcessRestore.  This prevents
	// vLLM's engine core threads from interfering with GPU state
	// reconstruction.  The gate is opened by postRestoreImpl after the
	// helper exits.
	nvp.ArmIoctlGate()

	log.Infof("nvproxy: afterLoad: %d allocs tracked, %d MiB staged data to restore, ioctl gate armed",
		nvp.tracker.NumAllocs(), nvp.StagedDataBytes()>>20)
}

func (fd *frontendFD) beforeSaveImpl() {
	log.Infof("nvproxy: saving frontendFD for %s", fd.dev.basename())
}

// openHostDevDirect opens an nvidia device file directly from the host,
// bypassing the device gofer.  This is used during restore because the
// gofer context is not available during stateify afterLoad callbacks.
func openHostDevDirect(relpath string) (int32, error) {
	abspath := filepath.Join("/dev", relpath)
	hostFD, err := unix.Openat(-1, abspath, unix.O_RDWR|unix.O_NOFOLLOW, 0)
	if err != nil {
		return -1, fmt.Errorf("open %s: %w", abspath, err)
	}
	return int32(hostFD), nil
}

func (fd *frontendFD) afterLoadImpl(goContext.Context) {
	// Mark this FD as restored so the ioctl gate applies to it.
	// Fresh FDs opened by the save-restore-exec helper are NOT marked,
	// so the helper's ioctls bypass the gate and avoid deadlock.
	fd.restoredFromCheckpoint = true

	// Do NOT re-open the host device yet.  Leave hostFD = -1 so that
	// any ioctl on this FD cannot reach the host nvidia driver.  This
	// prevents driver-level lock contention with the save-restore-exec
	// helper, which opens its OWN fresh FDs and calls
	// cuCheckpointProcessRestore.
	//
	// Host FDs are re-opened later by ReopenAllFDs (called from
	// postRestoreImpl after the helper finishes).  Until then, the
	// ioctl gate blocks app threads so they never see hostFD = -1.
	fd.dev.nvp.fdsMu.Lock()
	fd.dev.nvp.frontendFDs[fd] = struct{}{}
	fd.dev.nvp.fdsMu.Unlock()
	log.Infof("nvproxy: afterLoad frontendFD for %s (hostFD deferred)", fd.dev.basename())
}

func (fd *uvmFD) beforeSaveImpl() {
	// hostFD already neutralized by nvproxy.beforeSaveImpl; just log.
	log.Infof("nvproxy: saving uvmFD (hostFD=%d)", fd.hostFD)
}

func (fd *uvmFD) afterLoadImpl(goContext.Context) {
	// Mark this FD as restored so the ioctl gate applies to it.
	fd.restoredFromCheckpoint = true

	// Do NOT re-open the host device yet — same reasoning as frontendFD.
	// Host FDs are re-opened by ReopenAllFDs after the helper finishes.
	fd.dev.nvp.fdsMu.Lock()
	fd.dev.nvp.uvmFDs[fd] = struct{}{}
	fd.dev.nvp.fdsMu.Unlock()
	log.Infof("nvproxy: afterLoad uvmFD (hostFD deferred)")
}

func (fd *openOnlyFD) beforeSaveImpl() {
	if fd.hostFD >= 0 {
		unix.Close(int(fd.hostFD))
		fd.hostFD = -1
	}
}

func (fd *openOnlyFD) afterLoadImpl(goContext.Context) {
	// Do NOT re-open yet — same reasoning as frontendFD.
	log.Infof("nvproxy: afterLoad openOnlyFD for %s (hostFD deferred)", fd.dev.relpath)
}

// ReopenAllFDs re-opens all nvidia host device FDs that were deferred
// during afterLoadImpl.  This must be called AFTER the save-restore-exec
// helper has finished cuCheckpointProcessRestore, so there is no
// driver-level lock contention between the helper's FDs and the app's FDs.
//
// Called from postRestoreImpl via ReopenAllFDsFromVFS.
func (nvp *nvproxy) ReopenAllFDs() {
	nvp.fdsMu.Lock()
	feCount := 0
	for fd := range nvp.frontendFDs {
		if fd.hostFD >= 0 {
			continue // already open
		}
		newHostFD, err := openHostDevDirect(fd.dev.basename())
		if err != nil {
			log.Warningf("nvproxy: ReopenAllFDs: failed to re-open %s: %v", fd.dev.basename(), err)
			continue
		}
		fd.hostFD = newHostFD
		fd.memmapFile.SetFD(int(newHostFD))
		fd.internalEntry.Init(fd, waiter.AllEvents)
		fd.internalQueue.EventRegister(&fd.internalEntry)
		if err := fdnotifier.AddFD(newHostFD, &fd.internalQueue); err != nil {
			log.Warningf("nvproxy: ReopenAllFDs: fdnotifier.AddFD for %s failed: %v", fd.dev.basename(), err)
		}
		feCount++
	}
	uvmCount := 0
	for fd := range nvp.uvmFDs {
		if fd.hostFD >= 0 {
			continue
		}
		newHostFD, err := openHostDevDirect("nvidia-uvm")
		if err != nil {
			log.Warningf("nvproxy: ReopenAllFDs: failed to re-open nvidia-uvm: %v", err)
			continue
		}
		fd.hostFD = newHostFD
		fd.memmapFile.SetFD(int(newHostFD))
		fd.memmapFile.RequireAddrEqualsFileOffset()
		if err := fdnotifier.AddFD(newHostFD, &fd.queue); err != nil {
			log.Warningf("nvproxy: ReopenAllFDs: fdnotifier.AddFD for nvidia-uvm failed: %v", err)
		}
		uvmCount++
	}
	nvp.fdsMu.Unlock()
	log.Infof("nvproxy: ReopenAllFDs: re-opened %d frontend + %d UVM FDs", feCount, uvmCount)
}
