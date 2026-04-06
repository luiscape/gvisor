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
	"fmt"
	"time"

	"golang.org/x/sys/unix"
	"gvisor.dev/gvisor/pkg/log"
	"gvisor.dev/gvisor/pkg/sentry/vfs"
)

// stageChunkSize is the maximum number of bytes staged (copied from GPU to
// host) in a single UVM_TOOLS_READ_PROCESS_MEMORY ioctl.  Large allocations
// are read in chunks of this size so that progress can be logged and so that
// a single failed ioctl doesn't lose all preceding data.
const stageChunkSize = 64 << 20 // 64 MiB

// StageGPUData copies every tracked GPU allocation from device memory into
// host memory (nvproxy.stagedData).  The staged buffers are serialised as
// part of the checkpoint image so the data survives across the
// checkpoint/restore boundary.
//
// This function must be called BEFORE beforeSaveImpl neutralises the host
// FDs, because it issues UVM ioctls through the live UVM host FD.
//
// Preconditions:
//   - Application threads are paused (no concurrent GPU access).
//   - At least one uvmFD is open and initialised.
func (nvp *nvproxy) StageGPUData() error {
	uvmFD := nvp.getAnyUVMFD()
	if uvmFD == nil {
		log.Infof("nvproxy: StageGPUData: no UVM FD available, nothing to stage")
		return nil
	}

	allocs := nvp.tracker.Snapshot()
	if len(allocs) == 0 {
		log.Infof("nvproxy: StageGPUData: tracker has 0 allocations, nothing to stage")
		return nil
	}

	totalBytes := uint64(0)
	for _, a := range allocs {
		totalBytes += a.Length
	}
	log.Infof("nvproxy: StageGPUData: staging %d allocations (%d MiB) from GPU",
		len(allocs), totalBytes>>20)

	staged := make(map[uint64][]byte, len(allocs))
	start := time.Now()

	skipped := 0
	stagedBytes := uint64(0)
	for _, alloc := range allocs {
		buf, err := nvp.readGPUAlloc(uvmFD, alloc)
		if err != nil {
			log.Warningf("nvproxy: StageGPUData: skipping VA %#x (len %d): %v", alloc.Base, alloc.Length, err)
			skipped++
			continue
		}
		staged[alloc.Base] = buf
		stagedBytes += alloc.Length
	}

	elapsed := time.Since(start)
	bw := float64(0)
	if elapsed > 0 {
		bw = float64(totalBytes) / elapsed.Seconds() / (1 << 30)
	}
	log.Infof("nvproxy: StageGPUData: staged %d MiB (%d allocs) in %v (%.1f GiB/s), skipped %d allocs",
		stagedBytes>>20, len(staged), elapsed.Round(time.Millisecond), bw, skipped)

	nvp.stagedData = staged
	return nil
}

// WriteGPUManifest writes the allocation tracker's contents to
// /tmp/.gpu_checkpoint_manifest inside the container's tmpfs.
// Each line is: hex_gpu_va hex_size
// The CLI restore helper reads this manifest to know which GPU
// allocations to recreate and where to read the staged data from
// in the sandbox's process memory.
//
// This must be called AFTER cuCheckpointProcessCheckpoint has
// staged GPU data to host memory and BEFORE gVisor serializes
// the container state (so the manifest file is included in the
// checkpoint image).
//
// The manifest is written by issuing a write to a tmpfs file via
// the container's filesystem.  Since we're in the sentry, we
// write to a well-known path that the restore helper can find
// at /proc/<sandbox_pid>/root/tmp/.gpu_checkpoint_manifest.
func (nvp *nvproxy) WriteGPUManifest() error {
	allocs := nvp.tracker.Snapshot()
	if len(allocs) == 0 {
		log.Infof("nvproxy: WriteGPUManifest: no allocations to write")
		return nil
	}

	// Write to a host-accessible path.  The sentry process can write
	// to /tmp directly (it's on the host filesystem).
	// The restore helper (running as a CLI child process) can read it
	// from /proc/<sandbox_pid>/root/tmp/ or directly from /tmp if the
	// sandbox shares the host /tmp.
	//
	// For simplicity, write to the host's /tmp.  The manifest is small
	// (a few KB) and ephemeral.
	fd, err := unix.Open("/tmp/.gpu_checkpoint_manifest",
		unix.O_WRONLY|unix.O_CREAT|unix.O_TRUNC, 0644)
	if err != nil {
		return fmt.Errorf("open manifest: %w", err)
	}
	defer unix.Close(fd)

	for _, alloc := range allocs {
		line := fmt.Sprintf("%x %x\n", alloc.Base, alloc.Length)
		if _, err := unix.Write(fd, []byte(line)); err != nil {
			return fmt.Errorf("write manifest entry: %w", err)
		}
	}

	log.Infof("nvproxy: WriteGPUManifest: wrote %d allocations to /tmp/.gpu_checkpoint_manifest",
		len(allocs))
	return nil
}

// WriteGPUManifestFromVFS writes the GPU allocation manifest via the
// nvproxy instance found in the given VFS.
func WriteGPUManifestFromVFS(vfsObj *vfs.VirtualFilesystem) error {
	nvp := nvproxyFromVFS(vfsObj)
	if nvp == nil {
		return nil
	}
	return nvp.WriteGPUManifest()
}

// We also need to write the staged data locations.  After
// cuCheckpointProcessCheckpoint, the GPU data is in the sandbox's
// host memory at addresses that differ from the GPU VAs.
// The manifest records GPU VAs, but the restore helper needs to
// know WHERE in host memory the data actually is.
//
// The approach: after checkpoint, we use UVM_TOOLS_READ_PROCESS_MEMORY
// to verify the data is accessible at the GPU VAs.  If it is, the
// restore helper can read from /proc/<pid>/mem at those VAs.
// If not, we need to scan host memory (more complex).
//
// For now, we record the GPU VAs in the manifest and the restore
// helper will try reading from those VAs via /proc/<pid>/mem.
// If that fails, it will zero-fill the allocation.

// RestoreGPUData writes every buffer in nvproxy.stagedData back to device
// memory at its original virtual address, then clears the staged data.
//
// This function must be called AFTER afterLoadImpl has re-opened the UVM
// host FD and AFTER the CUDA driver state has been restored (so that the
// target GPU virtual addresses are valid).
//
// Preconditions:
//   - Application threads are paused (no concurrent GPU access).
//   - The CUDA driver has restored VA reservations (e.g. via
//     cuCheckpointProcessRestore).
//   - At least one uvmFD is open and initialised.
func (nvp *nvproxy) RestoreGPUData() error {
	if len(nvp.stagedData) == 0 {
		log.Infof("nvproxy: RestoreGPUData: no staged data, nothing to restore")
		return nil
	}

	uvmFD := nvp.getAnyUVMFD()
	if uvmFD == nil {
		return fmt.Errorf("nvproxy: RestoreGPUData: no UVM FD available")
	}

	totalBytes := uint64(0)
	for _, buf := range nvp.stagedData {
		totalBytes += uint64(len(buf))
	}
	log.Infof("nvproxy: RestoreGPUData: restoring %d allocations (%d MiB) to GPU",
		len(nvp.stagedData), totalBytes>>20)

	start := time.Now()

	for va, buf := range nvp.stagedData {
		if err := nvp.writeGPUAlloc(uvmFD, va, buf); err != nil {
			return fmt.Errorf("restoring VA %#x (len %d): %w", va, len(buf), err)
		}
	}

	elapsed := time.Since(start)
	bw := float64(0)
	if elapsed > 0 {
		bw = float64(totalBytes) / elapsed.Seconds() / (1 << 30)
	}
	log.Infof("nvproxy: RestoreGPUData: restored %d MiB in %v (%.1f GiB/s)",
		totalBytes>>20, elapsed.Round(time.Millisecond), bw)

	// Free the host-side copies.  From this point the canonical data is on
	// the GPU only.
	nvp.stagedData = nil
	return nil
}

// ClearStagedData discards all staged GPU data without writing it to the
// device.  This is used if GPU restore takes an alternative path (e.g. the
// CUDA Checkpoint API restores data internally).
func (nvp *nvproxy) ClearStagedData() {
	if n := len(nvp.stagedData); n > 0 {
		log.Infof("nvproxy: ClearStagedData: discarding %d staged buffers", n)
	}
	nvp.stagedData = nil
}

// HasStagedData returns true if there is staged GPU data waiting to be
// restored.
func (nvp *nvproxy) HasStagedData() bool {
	return len(nvp.stagedData) > 0
}

// StagedDataBytes returns the total size of staged GPU data in bytes.
func (nvp *nvproxy) StagedDataBytes() uint64 {
	total := uint64(0)
	for _, buf := range nvp.stagedData {
		total += uint64(len(buf))
	}
	return total
}

// --- helpers ---

// getAnyUVMFD returns an arbitrary open uvmFD, or nil if none are tracked.
// The returned FD must have a valid hostFD (>= 0).
func (nvp *nvproxy) getAnyUVMFD() *uvmFD {
	nvp.fdsMu.Lock()
	defer nvp.fdsMu.Unlock()
	for fd := range nvp.uvmFDs {
		if fd.hostFD >= 0 {
			return fd
		}
	}
	return nil
}

// readGPUAlloc reads a single GPU allocation from device memory into a new
// host-side byte slice, using the given UVM FD for the ioctl.  Large
// allocations are read in stageChunkSize chunks.
func (nvp *nvproxy) readGPUAlloc(fd *uvmFD, alloc gpuAlloc) ([]byte, error) {
	buf := make([]byte, alloc.Length)
	remaining := alloc.Length
	offset := uint64(0)

	for remaining > 0 {
		chunk := remaining
		if chunk > stageChunkSize {
			chunk = stageChunkSize
		}
		n, err := fd.memmapFile.BufferReadAt(alloc.Base+offset, buf[offset:offset+chunk])
		if err != nil {
			return nil, fmt.Errorf("BufferReadAt(VA=%#x, off=%d, len=%d): read %d bytes: %w",
				alloc.Base, offset, chunk, n, err)
		}
		offset += chunk
		remaining -= chunk
	}

	if log.IsLogging(log.Debug) {
		log.Debugf("nvproxy: read GPU alloc VA=%#x len=%d (%d MiB)",
			alloc.Base, alloc.Length, alloc.Length>>20)
	}
	return buf, nil
}

// writeGPUAlloc writes a host-side byte buffer back to a GPU virtual address,
// using the given UVM FD for the ioctl.  Large buffers are written in
// stageChunkSize chunks.
func (nvp *nvproxy) writeGPUAlloc(fd *uvmFD, va uint64, buf []byte) error {
	total := uint64(len(buf))
	remaining := total
	offset := uint64(0)

	for remaining > 0 {
		chunk := remaining
		if chunk > stageChunkSize {
			chunk = stageChunkSize
		}
		n, err := fd.memmapFile.BufferWriteAt(va+offset, buf[offset:offset+chunk])
		if err != nil {
			return fmt.Errorf("BufferWriteAt(VA=%#x, off=%d, len=%d): wrote %d bytes: %w",
				va, offset, chunk, n, err)
		}
		offset += chunk
		remaining -= chunk
	}

	if log.IsLogging(log.Debug) {
		log.Debugf("nvproxy: wrote GPU alloc VA=%#x len=%d (%d MiB)",
			va, total, total>>20)
	}
	return nil
}

// NvproxyFromVFS returns the nvproxy instance registered in the given VFS, or
// nil if nvproxy is not registered.  This is exported so that the
// control/state package can access nvproxy for checkpoint/restore
// orchestration.
func NvproxyFromVFS(vfsObj *vfs.VirtualFilesystem) *nvproxy {
	return nvproxyFromVFS(vfsObj)
}

// StageGPUDataFromVFS looks up the nvproxy instance from the VFS and stages
// all tracked GPU allocations to host memory.  Returns nil if nvproxy is not
// registered or has no allocations to stage.
//
// This is the primary entry point for the checkpoint path in the control
// package (state_impl.go).
func StageGPUDataFromVFS(vfsObj *vfs.VirtualFilesystem) error {
	nvp := nvproxyFromVFS(vfsObj)
	if nvp == nil {
		return nil // nvproxy not registered; nothing to do
	}
	return nvp.StageGPUData()
}

// RestoreGPUDataFromVFS looks up the nvproxy instance from the VFS and
// writes all staged GPU data back to device memory.  Returns nil if nvproxy
// is not registered or has no staged data.
//
// This is the primary entry point for the restore path in the control
// package (state_impl.go).
func RestoreGPUDataFromVFS(vfsObj *vfs.VirtualFilesystem) error {
	nvp := nvproxyFromVFS(vfsObj)
	if nvp == nil {
		return nil
	}
	return nvp.RestoreGPUData()
}

// HasStagedDataFromVFS returns true if nvproxy has GPU data staged in host
// memory that needs to be restored.
func HasStagedDataFromVFS(vfsObj *vfs.VirtualFilesystem) bool {
	nvp := nvproxyFromVFS(vfsObj)
	if nvp == nil {
		return false
	}
	return nvp.HasStagedData()
}

// ReopenAllFDsFromVFS re-opens all nvidia host device FDs that were
// deferred during afterLoadImpl.  This must be called AFTER the
// save-restore-exec helper has finished cuCheckpointProcessRestore,
// so there is no driver-level lock contention.
func ReopenAllFDsFromVFS(vfsObj *vfs.VirtualFilesystem) {
	nvp := nvproxyFromVFS(vfsObj)
	if nvp == nil {
		return
	}
	nvp.ReopenAllFDs()
}

// OpenIoctlGateFromVFS opens the ioctl gate on the nvproxy instance,
// unblocking all application ioctls that were blocked during restore.
// This must be called after the save-restore-exec helper has finished
// cuCheckpointProcessRestore, so GPU state is fully reconstructed before
// application threads can issue CUDA ioctls.
//
// Safe to call when nvproxy is not registered or when the gate is not armed.
func OpenIoctlGateFromVFS(vfsObj *vfs.VirtualFilesystem) {
	nvp := nvproxyFromVFS(vfsObj)
	if nvp == nil {
		return
	}
	nvp.OpenIoctlGate()
}

// TrackerInfoFromVFS returns a summary string describing the current state
// of the allocation tracker.  Useful for logging/debugging.
func TrackerInfoFromVFS(vfsObj *vfs.VirtualFilesystem) string {
	nvp := nvproxyFromVFS(vfsObj)
	if nvp == nil {
		return "nvproxy not registered"
	}
	return fmt.Sprintf("%d allocs, %d MiB tracked; %d MiB staged",
		nvp.tracker.NumAllocs(), nvp.tracker.TotalBytes()>>20,
		nvp.StagedDataBytes()>>20)
}
