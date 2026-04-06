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

//go:build !false
// +build !false

package control

import (
	"gvisor.dev/gvisor/pkg/abi/linux"
	"gvisor.dev/gvisor/pkg/context"
	"gvisor.dev/gvisor/pkg/log"
	"gvisor.dev/gvisor/pkg/sentry/devices/nvproxy"
	"gvisor.dev/gvisor/pkg/sentry/kernel"
	"gvisor.dev/gvisor/pkg/sentry/state"
	"gvisor.dev/gvisor/pkg/sentry/vfs"
	"gvisor.dev/gvisor/pkg/timing"
)

type SaveOptsExtra struct{}

func setSaveOptsImpl(o *SaveOpts, saveOpts *state.SaveOpts) error {
	return setSaveOptsForLocalCheckpointFiles(o, saveOpts)
}

func preSaveImpl(k *kernel.Kernel, o *state.SaveOpts) error {
	// Stage GPU memory to host memory before the checkpoint serialises
	// kernel state.  This must happen while the UVM host FDs are still
	// live (beforeSaveImpl will neutralise them).
	//
	// StageGPUDataFromVFS is a no-op when nvproxy is not registered or
	// when there are no tracked GPU allocations.
	info := nvproxy.TrackerInfoFromVFS(k.VFS())
	log.Infof("control: preSave: nvproxy tracker: %s", info)

	if err := nvproxy.StageGPUDataFromVFS(k.VFS()); err != nil {
		log.Warningf("control: preSave: GPU data staging failed: %v", err)
	}
	// NOTE: GPU allocation manifest is written in preSave() (state.go)
	// BEFORE SaveRestoreExec runs, because the CUDA checkpoint helper
	// frees all allocations from the tracker during checkpoint.
	return nil
}

func postRestoreImpl(k *kernel.Kernel, _ *timing.Timeline) error {
	// At this point, SaveRestoreExec has already run the restore helper
	// (cuda_checkpoint_helper restore), which called
	// cuCheckpointProcessRestore + cuCheckpointProcessUnlock for all
	// checkpointed PIDs.  GPU state is now fully restored.
	//
	// Now re-open the nvidia host device FDs that were deferred during
	// afterLoadImpl.  We deferred them to avoid driver-level lock
	// contention between the app's FDs and the helper's FDs (all
	// gVisor processes share one host process, so the nvidia driver
	// sees them as competing for the same per-process locks).
	log.Infof("control: postRestore: re-opening nvidia host FDs")
	nvproxy.ReopenAllFDsFromVFS(k.VFS())

	// Open the ioctl gate so application threads can proceed with
	// their CUDA ioctls (they were getting EAGAIN until now).
	log.Infof("control: postRestore: opening ioctl gate")
	nvproxy.OpenIoctlGateFromVFS(k.VFS())

	// Write any staged GPU data back to device memory (from the
	// nvproxy sentry-side staging path, if it was used).
	if nvproxy.HasStagedDataFromVFS(k.VFS()) {
		log.Infof("control: postRestore: restoring staged GPU data")
		if err := nvproxy.RestoreGPUDataFromVFS(k.VFS()); err != nil {
			log.Warningf("control: postRestore: GPU data restore failed: %v", err)
		}
	}
	return nil
}

// StopGPUThreadGroups sends SIGSTOP to every thread group that has open
// nvidia device FDs.  This prevents application threads from issuing CUDA
// ioctls while the save-restore-exec helper restores GPU state via
// cuCheckpointProcessRestore.
//
// Call this from the PostRestore path BEFORE SaveRestoreExec runs the
// restore helper.  After the helper finishes, postRestoreImpl calls
// ResumeGPUThreadGroups to send SIGCONT and open the ioctl gate.
//
// Thread groups whose leader has Origin == OriginExec (i.e. the helper
// process itself) are skipped so the helper can run freely.
func StopGPUThreadGroups(k *kernel.Kernel) {
	tgs := findGPUThreadGroups(k)
	if len(tgs) == 0 {
		return
	}
	for _, tg := range tgs {
		pid := tg.ID()
		log.Infof("control: SIGSTOP → GPU thread group pid=%d", pid)
		if err := k.SendExternalSignalThreadGroup(tg, &linux.SignalInfo{
			Signo: int32(linux.SIGSTOP),
		}); err != nil {
			log.Warningf("control: SIGSTOP → pid=%d failed: %v", pid, err)
		}
	}
	log.Infof("control: sent SIGSTOP to %d GPU thread group(s)", len(tgs))
}

// ResumeGPUThreadGroups sends SIGCONT to every thread group that has open
// nvidia device FDs, reversing a previous StopGPUThreadGroups call.
func ResumeGPUThreadGroups(k *kernel.Kernel) {
	tgs := findGPUThreadGroups(k)
	if len(tgs) == 0 {
		return
	}
	for _, tg := range tgs {
		pid := tg.ID()
		log.Infof("control: SIGCONT → GPU thread group pid=%d", pid)
		if err := k.SendExternalSignalThreadGroup(tg, &linux.SignalInfo{
			Signo: int32(linux.SIGCONT),
		}); err != nil {
			log.Warningf("control: SIGCONT → pid=%d failed: %v", pid, err)
		}
	}
	log.Infof("control: sent SIGCONT to %d GPU thread group(s)", len(tgs))
}

// findGPUThreadGroups returns thread groups that have at least one open
// nvidia device FD.  Thread groups created by SaveRestoreExec (Origin ==
// OriginExec) are excluded so the restore helper is never stopped.
func findGPUThreadGroups(k *kernel.Kernel) []*kernel.ThreadGroup {
	var result []*kernel.ThreadGroup
	ctx := context.Background()
	for _, tg := range k.RootPIDNamespace().ThreadGroups() {
		leader := tg.Leader()
		if leader == nil {
			continue
		}
		// Skip the save-restore-exec helper process.
		if leader.Origin == kernel.OriginExec {
			continue
		}
		if tgHasNvidiaFDs(ctx, leader) {
			result = append(result, tg)
		}
	}
	return result
}

// tgHasNvidiaFDs checks whether the given task's FD table contains at
// least one nvidia device FD (frontendFD, uvmFD, or openOnlyFD).
func tgHasNvidiaFDs(ctx context.Context, t *kernel.Task) bool {
	fdt := t.FDTable()
	if fdt == nil {
		return false
	}
	found := false
	fdt.ForEach(ctx, func(fd int32, file *vfs.FileDescription, _ kernel.FDFlags) bool {
		if _, ok := file.Impl().(nvproxy.NvidiaDeviceFD); ok {
			found = true
			return false // stop iteration
		}
		return true
	})
	return found
}

func postResumeImpl(k *kernel.Kernel, _ *timing.Timeline) error {
	return nil
}
