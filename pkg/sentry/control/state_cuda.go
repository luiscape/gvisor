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

package control

import (
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"gvisor.dev/gvisor/pkg/cleanup"
	"gvisor.dev/gvisor/pkg/context"
	"gvisor.dev/gvisor/pkg/log"
	"gvisor.dev/gvisor/pkg/sentry/devices/memdev"
	"gvisor.dev/gvisor/pkg/sentry/devices/nvproxy"
	"gvisor.dev/gvisor/pkg/sentry/fdcollector"
	"gvisor.dev/gvisor/pkg/sentry/fsimpl/pipefs"
	"gvisor.dev/gvisor/pkg/sentry/kernel"
	"gvisor.dev/gvisor/pkg/sentry/state"
	"gvisor.dev/gvisor/pkg/sentry/vfs"
	"gvisor.dev/gvisor/pkg/timing"
)

const (
	cudaProcsKey = "cuda-procs"

	// cudaCheckpointPathKey is the checkpoint state key for the path to the
	// cuda-checkpoint binary.
	cudaCheckpointPathKey = "cuda-checkpoint-path"

	// cudaCheckpointSequentialKey is the checkpoint state key for whether to run
	// cuda-checkpoint sequentially.
	cudaCheckpointSequentialKey = "cuda-checkpoint-sequential"

	// cudaLockTimeoutMS is how long (in milliseconds) each `cuda-checkpoint
	// --action lock` invocation waits for a process to reach a lockable state.
	// NCCL/CUDA-IPC-coupled processes only become lockable once every job
	// member is locking in parallel (a rank spinning in an unfinished collective
	// cannot be quiesced until its peers are too), so this must be generous.
	cudaLockTimeoutMS = 30000
)

func preSaveCuda(k *kernel.Kernel, o *state.SaveOpts) error {
	if o.CudaCheckpointPath == "" {
		return nil
	}

	wasPaused := k.IsPaused()
	if wasPaused {
		// It is possible that the kernel is paused when we are trying to save it.
		// Unpause it temporarily so that we can execute cuda-checkpoint. We can
		// expect such a state when using Docker. Docker's checkpoint command
		// calls pause first and then calls the checkpoint command.
		log.Infof("Unpausing kernel to execute cuda-checkpoint")
		k.Unpause()
		if k.IsPaused() {
			// If the kernel is still paused, we don't understand/expect this state.
			k.Pause() // Revert the unpause from above.
			return fmt.Errorf("kernel is double paused before checkpoint")
		}
	}
	sctx := k.SupervisorContext()
	cudaProcs := cudaProcs(sctx, k, o.CudaCheckpointPath, k.NvidiaDriverVersion.Major())
	// FIXME: b/456299722
	for _, tg := range cudaProcs {
		tg.SigsegvLock()
	}
	err := checkpointCudaProcs(sctx, k, o.CudaCheckpointPath, cudaProcs, o.CudaCheckpointSequential)
	if wasPaused {
		k.Pause()
	}
	if err != nil {
		// FIXME: b/456299722
		for _, tg := range cudaProcs {
			tg.SigsegvUnlock()
		}
		return err
	}
	k.AddStateToCheckpoint(cudaCheckpointPathKey, o.CudaCheckpointPath)
	k.AddStateToCheckpoint(cudaCheckpointSequentialKey, o.CudaCheckpointSequential)
	k.AddStateToCheckpoint(cudaProcsKey, cudaProcs)
	return nil
}

// cudaProcs returns a list of all CUDA processes in the sandbox. It selects
// them by collecting processes whose FD table has an open file descriptor to
// any CUDA device.
func cudaProcs(sctx context.Context, k *kernel.Kernel, cudaCheckpointPath string, nvidiaDriverVersionMajor int) []*kernel.ThreadGroup {
	var procs []*kernel.ThreadGroup
	k.TaskSet().ForEachThreadGroup(func(tg *kernel.ThreadGroup, tgLeader *kernel.Task) {
		found := false
		// Note that it is possible for tasks in a thread group to have various FD
		// tables (via clone(2) with CLONE_THREAD set and CLONE_FILES *not* set).
		// However, we don't expect this to happen in practice for CUDA processes.
		// So for efficiency, we just check the tgLeader's FD table, instead of
		// iterating over all tasks' FD tables in all thread groups.
		tgLeader.WithMuLocked(func(t *kernel.Task) {
			t.FDTable().ForEach(sctx, func(_ int32, file *vfs.FileDescription, _ kernel.FDFlags) bool {
				if _, ok := file.Impl().(nvproxy.NvidiaDeviceFD); ok {
					found = true
					return false
				}
				return true
			})
		})
		if found {
			procs = append(procs, tg)
		}
	})

	// procs may contain NVML-only processes, which don't use CUDA. As of
	// writing, calling cuda-checkpoint on them will fail for all tested drivers.
	// This includes R570, which supposedly has "NVML support". We suspect this
	// means that R570 supports CUDA+NVML processes, but not NVML-only processes.
	//
	// To filter out NVML-only processes, there are two approaches:
	// 1. Call cuda-checkpoint --get-state on all candidates. The checkpoint-able
	//    ones will return "running" and the others will fail. This is the
	//    recommendation in https://github.com/NVIDIA/cuda-checkpoint/issues/10.
	// 2. CUDA processes will have a thread named 'cudaXXXXXXXXXXX', where X is a
	//    hex digit. cuda-checkpoint interacts with these threads. Filter out
	//    processes that don't have such a thread.
	//
	// Option 1 is more robust, however, support for --get-state was only added
	// in R555. Prefer option 1 if possible, otherwise fall back to option 2.
	if nvidiaDriverVersionMajor < 550 {
		log.Warningf("cuda-checkpoint requires driver >=R550, driver major = %d, expect failures with message \"Insufficient driver\"", nvidiaDriverVersionMajor)
	} else if nvidiaDriverVersionMajor < 555 {
		procs = filterCudaProcsUsingThreadName(sctx, procs)
	} else {
		procs = filterCudaProcsUsingGetState(sctx, k, cudaCheckpointPath, procs)
	}
	return procs
}

func postRestoreCuda(k *kernel.Kernel, timeline *timing.Timeline) error {
	return postResumeCuda(k, timeline)
}

func postResumeCuda(k *kernel.Kernel, timeline *timing.Timeline) error {
	cudaCheckpointPathVal := k.PopCheckpointState(cudaCheckpointPathKey)
	if cudaCheckpointPathVal == nil {
		return nil
	}
	cudaCheckpointPath := cudaCheckpointPathVal.(string)
	cudaCheckpointSequential := k.PopCheckpointState(cudaCheckpointSequentialKey).(bool)
	cudaProcs := k.PopCheckpointState(cudaProcsKey).([]*kernel.ThreadGroup)
	timeline.Reached("starting cuda-ckpt")
	// FIXME: b/460451448 - pass --device-map to cuda-checkpoint if accepted
	err := restoreCudaProcs(k.SupervisorContext(), k, cudaCheckpointPath, cudaProcs, timeline, cudaCheckpointSequential)
	// FIXME: b/456299722
	for _, tg := range cudaProcs {
		tg.SigsegvUnlock()
	}
	return err
}

type checkpointProc struct {
	desc string
	tg   *kernel.ThreadGroup
	out  *fdcollector.Agent
}

// invokeCudaCheckpoint invokes cuda-checkpoint on the given CUDA process with
// the given operation flag. On success it returns a checkpointProc struct
// containing the running cuda-checkpoint process and a cleanup function which
// must be called to release resources. If cudaProc has exited, it returns
// (checkpointProc.tg == nil, err == nil).
func invokeCudaCheckpoint(sctx context.Context, k *kernel.Kernel, proc *Proc, cudaCheckpointPath string, cudaProc *kernel.ThreadGroup, opArgs []string, nullFD *vfs.FileDescription) (checkpointProc, func(), error) {
	pid := cudaProc.ID()
	leader := cudaProc.Leader()
	contID := leader.ContainerID()
	mntns := leader.MountNamespace()
	if mntns == nil || !mntns.TryIncRef() {
		log.Warningf("PID %d in container %q has exited, skipping CUDA checkpoint for it", pid, contID)
		return checkpointProc{}, nil, nil
	}
	root := mntns.Root(sctx)
	cu := cleanup.Make(func() {
		root.DecRef(sctx)
	})
	defer cu.Clean()
	ctx := vfs.WithRoot(sctx, root)
	cu.Add(func() {
		mntns.DecRef(ctx)
	})
	argv := append([]string{"cuda-checkpoint"}, opArgs...)
	argv = append(argv, "--pid", strconv.FormatInt(int64(pid), 10))
	args := &ExecArgs{
		Filename:       cudaCheckpointPath,
		Argv:           argv,
		ContainerID:    contID,
		MountNamespace: mntns,
		PIDNamespace:   leader.PIDNamespace(),
	}
	// Provision environment variables from leader's container spec.
	contName := k.ContainerName(contID)
	args.Envv = k.Saver().SpecEnviron(contName)

	// Provide standard streams to cuda-checkpoint. Use /dev/null as stdin
	// and direct cuda-checkpoint's stdout/stderr to a pipe.
	ckptDesc := fmt.Sprintf("cuda-checkpoint %s for PID %d in container %q", strings.Join(opArgs, " "), pid, contID)
	args.FDTable = k.NewFDTable()
	cu.Add(func() {
		args.FDTable.DecRef(ctx)
	})
	if nullFD != nil {
		if _, err := args.FDTable.NewFDAt(ctx, 0, nullFD, kernel.FDFlags{}); err != nil {
			log.Warningf("Failed to make /dev/null stdin for %s: %v", ckptDesc, err)
		}
	}
	var ckptOut *fdcollector.Agent
	rfd, wfd, err := pipefs.NewConnectedPipeFDs(ctx, k.PipeMount(), 0 /* flags */)
	if err != nil {
		log.Warningf("Failed to create stdout/stderr pipe for %s: %v", ckptDesc, err)
	} else {
		if _, err := args.FDTable.NewFDAt(ctx, 1, wfd, kernel.FDFlags{}); err != nil {
			log.Warningf("Failed to make pipe stdout for %s: %v", ckptDesc, err)
		}
		if _, err := args.FDTable.NewFDAt(ctx, 2, wfd, kernel.FDFlags{}); err != nil {
			log.Warningf("Failed to make pipe stderr for %s: %v", ckptDesc, err)
		}
		wfd.DecRef(ctx)
		ckptOut = fdcollector.NewAgent(ctx, rfd, ckptDesc) // transfers ownership of rfd
		cu.Add(ckptOut.Stop)
	}
	// FIXME(ayushranjan): Get WorkDirectory, Limits and Capabilities from spec?
	ckptTG, _, _, err := ExecAsync(proc, args)
	if err != nil {
		return checkpointProc{}, nil, fmt.Errorf("failed to exec %s: %w", ckptDesc, err)
	}
	return checkpointProc{
		desc: ckptDesc,
		tg:   ckptTG,
		out:  ckptOut,
	}, cu.Release(), nil
}

func filterCudaProcsUsingThreadName(sctx context.Context, cudaProcs []*kernel.ThreadGroup) []*kernel.ThreadGroup {
	log.Debugf("Filtering CUDA processes using thread name")
	cudaThreadRegex := regexp.MustCompile(`^cuda[0-9a-f]{11}$`)
	var res []*kernel.ThreadGroup
	for _, cudaProc := range cudaProcs {
		found := false
		cudaProc.ForEachTask(func(t *kernel.Task) bool {
			if cudaThreadRegex.MatchString(t.Name()) {
				found = true
				return false
			}
			return true
		})
		if found {
			res = append(res, cudaProc)
		}
	}
	return res
}

func filterCudaProcsUsingGetState(sctx context.Context, k *kernel.Kernel, cudaCheckpointPath string, cudaProcs []*kernel.ThreadGroup) []*kernel.ThreadGroup {
	log.Debugf("Filtering CUDA processes using 'cuda-checkpoint --get-state'")
	// Open /dev/null once for the stdin of all cuda-checkpoint processes.
	nullVD := k.VFS().NewAnonVirtualDentry("null")
	defer nullVD.DecRef(sctx)
	nullFD, err := memdev.NewNullFD(sctx, nullVD.Mount(), nullVD.Dentry(), vfs.OpenOptions{})
	if err != nil {
		log.Warningf("Failed to open /dev/null for cuda-checkpoint stdin: %v", err)
	} else {
		defer nullFD.DecRef(sctx)
	}

	// Call cuda-checkpoint for each CUDA PID parallelly.
	proc := &Proc{Kernel: k}
	ckptProcs := make(map[*kernel.ThreadGroup]checkpointProc)
	for _, cudaProc := range cudaProcs {
		ckptProc, cleanup, err := invokeCudaCheckpoint(sctx, k, proc, cudaCheckpointPath, cudaProc, []string{"--get-state"}, nullFD)
		if err != nil {
			log.Warningf("Failed to get state for PID %d: %v", cudaProc.ID(), err)
			continue
		}
		if ckptProc.tg == nil {
			continue
		}
		ckptProcs[cudaProc] = ckptProc
		defer cleanup()
	}
	// Check the output of all cuda-checkpoint processes. We want the ones with
	// output "running".
	var res []*kernel.ThreadGroup
	for cudaProc, ckptProc := range ckptProcs {
		ckptProc.tg.WaitExited()
		if status := ckptProc.tg.ExitStatus(); status == 0 {
			res = append(res, cudaProc)
			if ckptProc.out != nil {
				output := strings.TrimSpace(ckptProc.out.String())
				if output != "running" {
					log.Warningf("CUDA PID %d in unexpected state %q", cudaProc.ID(), output)
				}
				log.Debugf("%s succeeded; output: %q", ckptProc.desc, output)
			}
		} else {
			if ckptProc.out != nil {
				log.Warningf("%q failed with exit status %d, skipping CUDA checkpoint for PID %d; output: %q", ckptProc.desc, status, cudaProc.ID(), ckptProc.out.String())
			} else {
				log.Warningf("%q failed with exit status %d, skipping CUDA checkpoint for PID %d", ckptProc.desc, status, cudaProc.ID())
			}
		}
	}
	return res
}

// openCudaCheckpointNullFD opens /dev/null to use as stdin for cuda-checkpoint
// child processes. The returned cleanup must be called when the caller is done.
func openCudaCheckpointNullFD(sctx context.Context, k *kernel.Kernel) (*vfs.FileDescription, func()) {
	nullVD := k.VFS().NewAnonVirtualDentry("null")
	nullFD, err := memdev.NewNullFD(sctx, nullVD.Mount(), nullVD.Dentry(), vfs.OpenOptions{})
	if err != nil {
		log.Warningf("Failed to open /dev/null for cuda-checkpoint stdin: %v", err)
		return nil, func() { nullVD.DecRef(sctx) }
	}
	return nullFD, func() {
		nullFD.DecRef(sctx)
		nullVD.DecRef(sctx)
	}
}

// runCudaAction invokes `cuda-checkpoint <opArgs...> --pid <pid>` on every
// process in cudaProcs. When parallel is true all invocations run concurrently;
// otherwise they run one at a time. It returns the processes for which the
// action succeeded (exit status 0) and a combined error describing any failures.
func runCudaAction(sctx context.Context, k *kernel.Kernel, cudaCheckpointPath string, cudaProcs []*kernel.ThreadGroup, opArgs []string, parallel bool, nullFD *vfs.FileDescription) ([]*kernel.ThreadGroup, error) {
	proc := &Proc{Kernel: k}
	ckptProcs := make(map[*kernel.ThreadGroup]checkpointProc)
	var errs []error
	for _, cudaProc := range cudaProcs {
		ckptProc, cleanup, err := invokeCudaCheckpoint(sctx, k, proc, cudaCheckpointPath, cudaProc, opArgs, nullFD)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		if ckptProc.tg == nil {
			continue
		}
		ckptProcs[cudaProc] = ckptProc
		defer cleanup()
		// In sequential mode, wait for each invocation to finish before starting
		// the next. In parallel mode, all invocations are launched first and
		// waited on below.
		if !parallel {
			ckptProc.tg.WaitExited()
		}
	}
	var succeeded []*kernel.ThreadGroup
	for cudaProc, ckptProc := range ckptProcs {
		if parallel {
			ckptProc.tg.WaitExited()
		}
		if status := ckptProc.tg.ExitStatus(); status != 0 {
			out := ""
			if ckptProc.out != nil {
				out = ckptProc.out.String()
			}
			errs = append(errs, fmt.Errorf("%q failed with exit status %d; output: %q", ckptProc.desc, status, out))
		} else {
			succeeded = append(succeeded, cudaProc)
			if log.IsLogging(log.Debug) && ckptProc.out != nil {
				log.Debugf("%s succeeded; output: %q", ckptProc.desc, ckptProc.out.String())
			}
		}
	}
	return succeeded, errors.Join(errs...)
}

// checkpointCudaProcs suspends all CUDA processes in cudaProcs using
// cuda-checkpoint's two-phase lock/checkpoint protocol. The two phases are
// required for correctness when the processes are coupled through NCCL and/or
// CUDA IPC (as in tensor-parallel inference engines):
//
//  1. Lock ALL processes in parallel. Locking every job member before
//     checkpointing any is essential: a rank spinning inside an unfinished
//     collective can only be quiesced once its peers are locking too. Issuing a
//     full per-process --toggle (lock+checkpoint) instead lets one rank finish
//     checkpointing while its peer keeps spinning waiting for it, deadlocking
//     the snapshot.
//  2. Checkpoint all locked processes, releasing their GPU state.
func checkpointCudaProcs(sctx context.Context, k *kernel.Kernel, cudaCheckpointPath string, cudaProcs []*kernel.ThreadGroup, sequential bool) error {
	start := time.Now()
	nullFD, cleanup := openCudaCheckpointNullFD(sctx, k)
	defer cleanup()

	// Phase 1: lock every process, in parallel, so coupled ranks quiesce
	// together. --timeout bounds how long each lock waits for its process to
	// become lockable (rather than hanging forever).
	lockArgs := []string{"--action", "lock", "--timeout", strconv.Itoa(cudaLockTimeoutMS)}
	locked, err := runCudaAction(sctx, k, cudaCheckpointPath, cudaProcs, lockArgs, true /* parallel */, nullFD)
	if err != nil {
		// Best-effort: unlock whatever we locked so the application keeps running.
		if _, uerr := runCudaAction(sctx, k, cudaCheckpointPath, locked, []string{"--action", "unlock"}, true, nullFD); uerr != nil {
			log.Warningf("cuda-checkpoint unlock after lock-phase failure also failed: %v", uerr)
		}
		return fmt.Errorf("cuda-checkpoint lock phase failed: %w", err)
	}

	// Phase 2: checkpoint all locked processes.
	// NOTE: an experimental nvproxy fabric drain (nvproxy.DrainFabricMemory) was
	// prototyped here to work around cuda-checkpoint's inability to serialize
	// fabric/multicast memory, but it cannot succeed (libcuda retains the fabric
	// allocation and cuda-checkpoint still tries to serialize it), so it is not
	// invoked. See pkg/sentry/devices/nvproxy/fabric.go.
	if _, err := runCudaAction(sctx, k, cudaCheckpointPath, locked, []string{"--action", "checkpoint"}, !sequential, nullFD); err != nil {
		// Best-effort undo: restore then unlock, returning the app to running.
		if _, rerr := runCudaAction(sctx, k, cudaCheckpointPath, locked, []string{"--action", "restore"}, !sequential, nullFD); rerr != nil {
			log.Warningf("cuda-checkpoint restore after checkpoint-phase failure also failed: %v", rerr)
		}
		if _, uerr := runCudaAction(sctx, k, cudaCheckpointPath, locked, []string{"--action", "unlock"}, true, nullFD); uerr != nil {
			log.Warningf("cuda-checkpoint unlock after checkpoint-phase failure also failed: %v", uerr)
		}
		return fmt.Errorf("cuda-checkpoint checkpoint phase failed: %w", err)
	}
	log.Infof("cuda-checkpoint lock+checkpoint on %d processes took [%s]", len(locked), time.Since(start))
	return nil
}

// restoreCudaProcs resumes all CUDA processes in cudaProcs, the inverse of
// checkpointCudaProcs.
//
// Unlike the save side (which must lock all coupled processes in parallel
// before checkpointing any), the restore side uses a full per-process --toggle
// (restore + unlock in one atomic step), sequentially by default: with a
// cuda-checkpoint job (--launch-job), members must be toggled one at a time so
// the job file can re-establish shared CUDA IPC mappings deterministically —
// each importer's restore rendezvouses with an already-running exporter.
// Splitting restore and unlock into separate all-process phases breaks that
// protocol (observed: importer restore fails with "unknown error" or resumed
// ranks hit "unspecified launch failure" on their first kernel launch).
func restoreCudaProcs(sctx context.Context, k *kernel.Kernel, cudaCheckpointPath string, cudaProcs []*kernel.ThreadGroup, timeline *timing.Timeline, sequential bool) error {
	start := time.Now()
	nullFD, cleanup := openCudaCheckpointNullFD(sctx, k)
	defer cleanup()

	restored, err := runCudaAction(sctx, k, cudaCheckpointPath, cudaProcs, []string{"--toggle"}, !sequential, nullFD)
	timeline.Reached("cuda toggled to running")
	if err != nil {
		return fmt.Errorf("cuda-checkpoint restore toggle failed: %w", err)
	}
	log.Infof("cuda-checkpoint restore toggle on %d processes took [%s]", len(restored), time.Since(start))
	return nil
}
