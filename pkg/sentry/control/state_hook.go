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

package control

import (
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"gvisor.dev/gvisor/pkg/context"
	"gvisor.dev/gvisor/pkg/log"
	"gvisor.dev/gvisor/pkg/sentry/devices/nvproxy"
	"gvisor.dev/gvisor/pkg/sentry/kernel"
	"gvisor.dev/gvisor/pkg/sentry/state"
	"gvisor.dev/gvisor/pkg/sentry/vfs"
	"gvisor.dev/gvisor/pkg/timing"
)

// A checkpoint hook is a program inside each GPU container that checkpoints
// the container's GPU state in place of cuda-checkpoint (state_cuda.go). It
// runs with the container's namespaces and environment as:
//
//	<hook> checkpoint                  before the sandbox is saved
//	<hook> restore [--device-map MAP]  after it is restored or resumed
//	<hook> resume                      after restore, once page faults are released
//
// Page faults of GPU processes are held from before checkpoint until after
// restore (b/456299722), so work that touches GPU memory belongs in resume.
//
// If checkpoint exits 1, it undid its work except for what resume undoes, and
// resume runs. Any other non-zero status means it could not undo its work.
// Any failure of restore or resume fails the restore.

const (
	checkpointHookKey        = "checkpoint-hook"
	checkpointHookTargetsKey = "checkpoint-hook-targets"
	checkpointHookLockedKey  = "checkpoint-hook-locked"
)

// hookStatusRecoverable is the checkpoint exit status of a failure that resume
// finishes undoing.
const hookStatusRecoverable = 1

var errHookNotStarted = errors.New("could not start the checkpoint hook")

// hookOps runs the hook in one container per target (a *kernel.ThreadGroup,
// except in tests), and holds and releases the GPU processes' page faults.
type hookOps[T comparable] struct {
	// run returns the hook's exit status and output. err wraps
	// errHookNotStarted if it did not start, and is set if it did not exit.
	run                      func(target T, args ...string) (status uint32, out string, err error)
	lockFaults, unlockFaults func()
	name                     func(target T) string
}

func hookError[T comparable](ops hookOps[T], target T, phase string, status uint32, out string, err error) error {
	if err != nil {
		return fmt.Errorf("checkpoint hook %s in %s: %w", phase, ops.name(target), err)
	}
	return fmt.Errorf("checkpoint hook %s in %s failed with exit status %d; output: %q", phase, ops.name(target), status, out)
}

// checkpointWithHook runs checkpoint in each target with page faults held,
// which stay held on success. On failure, it restores the targets already
// checkpointed, releases page faults, and runs resume wherever the failure can
// be undone.
func checkpointWithHook[T comparable](ops hookOps[T], targets []T) error {
	ops.lockFaults()
	var done []T
	for _, target := range targets {
		status, out, err := ops.run(target, "checkpoint")
		if err == nil && status == 0 {
			done = append(done, target)
			continue
		}
		cerr := hookError(ops, target, "checkpoint", status, out, err)
		toResume := slices.Clone(done)
		switch {
		case err == nil && status == hookStatusRecoverable:
			toResume = append(toResume, target)
		case errors.Is(err, errHookNotStarted):
		default:
			cerr = fmt.Errorf("%w; it could not be undone, and the application in %s may be left blocked", cerr, ops.name(target))
		}
		var errs []error
		for _, d := range done {
			if status, out, err := ops.run(d, "restore"); err != nil || status != 0 {
				errs = append(errs, hookError(ops, d, "restore", status, out, err))
				toResume = slices.DeleteFunc(toResume, func(t T) bool { return t == d })
			}
		}
		ops.unlockFaults()
		for _, r := range toResume {
			if status, out, err := ops.run(r, "resume"); err != nil || status != 0 {
				errs = append(errs, hookError(ops, r, "resume", status, out, err))
			}
		}
		if len(errs) > 0 {
			return fmt.Errorf("%w; undoing it also failed: %w", cerr, errors.Join(errs...))
		}
		return cerr
	}
	return nil
}

// restoreWithHook runs restore in each target, releases page faults, and then
// runs resume in each if every restore succeeded.
func restoreWithHook[T comparable](ops hookOps[T], targets []T, deviceMap string) error {
	args := []string{"restore"}
	if deviceMap != "" {
		args = append(args, "--device-map", deviceMap)
	}
	var errs []error
	for _, target := range targets {
		if status, out, err := ops.run(target, args...); err != nil || status != 0 {
			errs = append(errs, hookError(ops, target, "restore", status, out, err))
		}
	}
	ops.unlockFaults()
	if len(errs) > 0 {
		return errors.Join(errs...)
	}
	for _, target := range targets {
		if status, out, err := ops.run(target, "resume"); err != nil || status != 0 {
			errs = append(errs, hookError(ops, target, "resume", status, out, err))
		}
	}
	return errors.Join(errs...)
}

// gpuProcs returns the thread groups in k with an NVIDIA device open, and one
// of them per container.
//
// Callers must not hold any thread-group leader's Task.mu in k.
func gpuProcs(sctx context.Context, k *kernel.Kernel) (procs, perContainer []*kernel.ThreadGroup) {
	seen := make(map[string]bool)
	k.TaskSet().ForEachThreadGroup(func(tg *kernel.ThreadGroup, tgLeader *kernel.Task) {
		found := false
		var contID string
		tgLeader.WithMuLocked(func(t *kernel.Task) {
			contID = t.ContainerID()
			t.FDTable().ForEach(sctx, func(_ int32, file *vfs.FileDescription, _ kernel.FDFlags) bool {
				_, found = file.Impl().(nvproxy.NvidiaDeviceFD)
				return !found
			})
		})
		if !found {
			return
		}
		procs = append(procs, tg)
		if !seen[contID] {
			seen[contID] = true
			perContainer = append(perContainer, tg)
		}
	})
	return procs, perContainer
}

func kernelHookOps(sctx context.Context, k *kernel.Kernel, path string, locked []*kernel.ThreadGroup, nullFD *vfs.FileDescription) hookOps[*kernel.ThreadGroup] {
	proc := &Proc{Kernel: k}
	return hookOps[*kernel.ThreadGroup]{
		run: func(target *kernel.ThreadGroup, args ...string) (uint32, string, error) {
			argv := append([]string{path}, args...)
			p, cleanup, err := execInContainerOf(sctx, k, proc, target, path, argv, "checkpoint hook "+args[0], nullFD)
			if err != nil {
				return 0, "", fmt.Errorf("%w: %w", errHookNotStarted, err)
			}
			if p.tg == nil {
				return 0, "", fmt.Errorf("%w: PID %d has exited", errHookNotStarted, target.ID())
			}
			defer cleanup()
			start := time.Now()
			p.tg.WaitExited()
			out := ""
			if p.out != nil {
				out = strings.TrimSpace(p.out.String())
			}
			ws := p.tg.ExitStatus()
			log.Infof("%s took [%s]; status %v; output: %q", p.desc, time.Since(start), ws, out)
			if !ws.Exited() {
				return 0, out, fmt.Errorf("terminated by %v; output: %q", ws.TerminationSignal(), out)
			}
			return ws.ExitStatus(), out, nil
		},
		lockFaults: func() {
			for _, tg := range locked {
				tg.SigsegvLock()
			}
		},
		unlockFaults: func() {
			for _, tg := range locked {
				tg.SigsegvUnlock()
			}
		},
		name: func(target *kernel.ThreadGroup) string {
			if leader := target.Leader(); leader != nil {
				return fmt.Sprintf("container %q", leader.ContainerID())
			}
			return fmt.Sprintf("the container of PID %d", target.ID())
		},
	}
}

// preSaveHook runs the checkpoint hook before the kernel is saved, and records
// what postRestoreHook needs.
func preSaveHook(k *kernel.Kernel, o *state.SaveOpts) error {
	if k.IsPaused() {
		// Docker's checkpoint pauses first, but the hook needs the
		// application's processes to run, including for any undo.
		k.Unpause()
		if k.IsPaused() {
			k.Pause()
			return fmt.Errorf("kernel is double paused before checkpoint")
		}
		defer k.Pause()
	}
	sctx := k.SupervisorContext()
	locked, targets := gpuProcs(sctx, k)
	if len(targets) == 0 {
		return nil
	}
	nullFD, cleanup := openNullFD(sctx, k)
	defer cleanup()
	if err := checkpointWithHook(kernelHookOps(sctx, k, o.CheckpointHook, locked, nullFD), targets); err != nil {
		return err
	}
	k.AddStateToCheckpoint(checkpointHookKey, o.CheckpointHook)
	k.AddStateToCheckpoint(checkpointHookTargetsKey, targets)
	k.AddStateToCheckpoint(checkpointHookLockedKey, locked)
	return nil
}

// postRestoreHook runs the checkpoint hook's restore and resume phases if
// preSaveHook ran it. nvproxyRemapping is non-nil only after a restore that
// remapped GPUs.
func postRestoreHook(k *kernel.Kernel, timeline *timing.Timeline, nvproxyRemapping *nvproxy.DeviceRemapping) error {
	pathVal := k.PopCheckpointState(checkpointHookKey)
	if pathVal == nil {
		return nil
	}
	path := pathVal.(string)
	targets := k.PopCheckpointState(checkpointHookTargetsKey).([]*kernel.ThreadGroup)
	locked := k.PopCheckpointState(checkpointHookLockedKey).([]*kernel.ThreadGroup)
	sctx := k.SupervisorContext()
	nullFD, cleanup := openNullFD(sctx, k)
	defer cleanup()
	ops := kernelHookOps(sctx, k, path, locked, nullFD)
	deviceMap, err := cudaCheckpointDeviceMap(nvproxyRemapping)
	if err == nil && deviceMap != "" {
		if major := k.NvidiaDriverVersion.Major(); major < 580 {
			err = fmt.Errorf("GPUs changed across restore, but cuda-checkpoint --device-map requires driver >= R580 (have R%d)", major)
		}
	}
	if err != nil {
		// The sandbox is killed after a failed restore: let its processes die.
		ops.unlockFaults()
		return err
	}
	timeline.Reached("starting checkpoint hook")
	err = restoreWithHook(ops, targets, deviceMap)
	timeline.Reached("checkpoint hook done")
	return err
}
