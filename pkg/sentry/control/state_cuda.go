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
	"maps"
	"regexp"
	"slices"
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
	fail := func(err error) error {
		if wasPaused {
			k.Pause()
		}
		return err
	}

	// cuda-checkpoint hangs on processes holding NVLink multicast memory (e.g.
	// NCCL with NVLS) and cannot restore memory imported from an exported fd,
	// so refuse up front, unless they run the multicast interposer, which
	// releases both before cuda-checkpoint runs (checkpointCudaProcs re-checks
	// afterwards).
	managed := cudaShimManagedProcs(sctx, k, cudaProcs)
	shim := len(managed) != 0
	except := make(map[kernel.ThreadID]bool, len(managed))
	for _, tg := range managed {
		except[tg.ID()] = true
	}
	if blockers := nvproxy.CheckpointBlockers(k.VFS(), except); blockers != "" {
		return fail(fmt.Errorf("cannot checkpoint CUDA processes holding multicast or imported memory without the multicast interposer (e.g. NCCL_NVLS_ENABLE=0 to disable NVLS): %s", blockers))
	}
	// FIXME: b/456299722
	for _, tg := range cudaProcs {
		tg.SigsegvLock()
	}
	err := checkpointCudaProcs(sctx, k, o.CudaCheckpointPath, cudaProcs, o.CudaCheckpointSequential, shim)
	if err != nil {
		// Unwind BEFORE re-pausing (the docker flow): the interposer rebuild
		// needs the application's shim control threads to run and
		// acknowledge, which a paused kernel cannot do -- it would stall for
		// the full ack timeout and converge only after the eventual unpause.
		// The SIGSEGV unlock must precede the rebuild for the same reason
		// the resume path orders them: the rebuild touches GPU memory.
		// FIXME: b/456299722
		for _, tg := range cudaProcs {
			tg.SigsegvUnlock()
		}
		// Bring multicast back and let the application run again, unless the
		// teardown failed partway (see errCudaShimTornDown).
		if shim && !errors.Is(err, errCudaShimTornDown) {
			unwindCudaMulticastShim(sctx, k, cudaProcs)
		}
		return fail(err)
	}
	if wasPaused {
		k.Pause()
	}
	k.AddStateToCheckpoint(cudaCheckpointPathKey, o.CudaCheckpointPath)
	k.AddStateToCheckpoint(cudaCheckpointSequentialKey, o.CudaCheckpointSequential)
	k.AddStateToCheckpoint(cudaProcsKey, cudaProcs)
	return nil
}

// cudaProcs returns a list of all CUDA processes in the sandbox. It selects
// them by collecting processes whose FD table has an open file descriptor to
// any CUDA device.
//
// Callers must not hold any thread-group leader's Task.mu in k.
// checklocks cannot name the leader mutexes selected by ForEachThreadGroup.
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

// postRestoreCuda restores CUDA processes that were checkpointed by
// preSaveCuda. nvproxyRemapping is non-nil only when called after a restore
// that remapped GPUs.
func postRestoreCuda(k *kernel.Kernel, timeline *timing.Timeline, nvproxyRemapping *nvproxy.DeviceRemapping) error {
	cudaCheckpointPathVal := k.PopCheckpointState(cudaCheckpointPathKey)
	if cudaCheckpointPathVal == nil {
		return nil
	}
	cudaCheckpointPath := cudaCheckpointPathVal.(string)
	cudaCheckpointSequential := k.PopCheckpointState(cudaCheckpointSequentialKey).(bool)
	cudaProcs := k.PopCheckpointState(cudaProcsKey).([]*kernel.ThreadGroup)
	timeline.Reached("starting cuda-ckpt")
	err := restoreCudaProcs(k.SupervisorContext(), k, cudaCheckpointPath, cudaProcs, timeline, cudaCheckpointSequential, nvproxyRemapping)
	// FIXME: b/456299722
	for _, tg := range cudaProcs {
		tg.SigsegvUnlock()
	}

	// Rebuild the interposer's multicast objects and CUDA IPC imports.
	//
	// Ordering here is doubly constrained. It must run after the restore
	// toggle has finished on EVERY process, or a rank rebuilds on a context
	// whose device state is not restored yet and latches a sticky 719. It
	// must also run after SigsegvUnlock above: the rebuild touches GPU
	// memory, and while the SIGSEGV lock is held the resulting page faults
	// cannot be serviced, which faults every rank's context with
	// CUDA_ERROR_ILLEGAL_ADDRESS (700). Both were observed.
	if err == nil {
		if rerr := resumeCudaMulticastShim(k.SupervisorContext(), k, cudaProcs); rerr != nil {
			err = fmt.Errorf("failed to resume multicast interposer: %w", rerr)
		} else {
			timeline.Reached("multicast interposer resumed")
		}
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
	if leader == nil {
		// The thread group fully exited between enumeration and now.
		log.Warningf("PID %d has exited, skipping CUDA checkpoint for it", pid)
		return checkpointProc{}, nil, nil
	}
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
	// The multicast interposer may be preloaded into every container binary
	// via /etc/ld.so.preload, including this cuda-checkpoint process. Disable
	// it here: it has no business interposing cuda-checkpoint, and its load
	// banner on stderr would corrupt the output this exec's caller parses
	// (e.g. --get-state's "running").
	args.Envv = append(args.Envv, "MCSHIM_DISABLE=1")

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
	// Collect results in input order, not map order, so that later phases
	// handle the processes in the same order.
	var succeeded []*kernel.ThreadGroup
	for _, cudaProc := range cudaProcs {
		ckptProc, ok := ckptProcs[cudaProc]
		if !ok {
			continue
		}
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
//
// On failure it leaves the processes unlocked; the caller unwinds the
// interposer.
func checkpointCudaProcs(sctx context.Context, k *kernel.Kernel, cudaCheckpointPath string, cudaProcs []*kernel.ThreadGroup, sequential bool, shim bool) error {
	start := time.Now()
	nullFD, cleanup := openCudaCheckpointNullFD(sctx, k)
	defer cleanup()
	unlockArgs := []string{"--action", "unlock"}
	unlock := func(tgs []*kernel.ThreadGroup) {
		if _, err := runCudaAction(sctx, k, cudaCheckpointPath, tgs, unlockArgs, true /* parallel */, nullFD); err != nil {
			log.Warningf("cuda-checkpoint unlock after failure also failed: %v", err)
		}
	}

	// Phase 1: gate the application off the GPU, which stops new submissions,
	// then lock every process, which drains work in flight. A collective that
	// straddles the gate starves its peers, and the lock then times out.
	lockArgs := []string{"--action", "lock", "--timeout", strconv.Itoa(cudaLockTimeoutMS)}
	if shim {
		if err := armCudaMulticastShimGate(sctx, k, cudaProcs); err != nil {
			return err
		}
	}
	locked, err := runCudaAction(sctx, k, cudaCheckpointPath, cudaProcs, lockArgs, true /* parallel */, nullFD)
	if err != nil {
		unlock(locked)
		return fmt.Errorf("cuda-checkpoint lock phase failed: %w", err)
	}

	// Interposer teardown, between two locks: it issues libcuda calls, which a
	// locked process cannot make. The gate keeps the application off the GPU.
	if shim {
		if _, err := runCudaAction(sctx, k, cudaCheckpointPath, locked, unlockArgs, true /* parallel */, nullFD); err != nil {
			unlock(locked)
			return fmt.Errorf("cuda-checkpoint unlock before multicast teardown failed: %w", err)
		}
		if err := suspendCudaMulticastShim(sctx, k, locked); err != nil {
			return err
		}
		// cuda-checkpoint would hang on anything the interposer left behind.
		if blockers := nvproxy.CheckpointBlockers(k.VFS(), nil); blockers != "" {
			return fmt.Errorf("multicast interposer suspended but resources remain: %s", blockers)
		}
		if locked, err = runCudaAction(sctx, k, cudaCheckpointPath, cudaProcs, lockArgs, true /* parallel */, nullFD); err != nil {
			unlock(locked)
			return fmt.Errorf("cuda-checkpoint re-lock after multicast teardown failed: %w", err)
		}
	}

	// Phase 2: checkpoint all locked processes.
	if _, err := runCudaAction(sctx, k, cudaCheckpointPath, locked, []string{"--action", "checkpoint"}, !sequential, nullFD); err != nil {
		// Best-effort undo: restore then unlock, returning the app to running.
		if _, rerr := runCudaAction(sctx, k, cudaCheckpointPath, locked, []string{"--action", "restore"}, !sequential, nullFD); rerr != nil {
			log.Warningf("cuda-checkpoint restore after checkpoint-phase failure also failed: %v", rerr)
		}
		unlock(locked)
		return fmt.Errorf("cuda-checkpoint checkpoint phase failed: %w", err)
	}

	log.Infof("cuda-checkpoint lock+checkpoint on %d processes took [%s]", len(locked), time.Since(start))
	return nil
}

// cudaCheckpointDeviceMap returns the value to pass to cuda-checkpoint's
// --device-map flag to restore CUDA state checkpointed on dr's old devices
// onto its new devices, in the format "oldUuid1=newUuid1,oldUuid2=newUuid2".
// cuda-checkpoint requires the map to list all checkpointed devices, so all
// saved devices are included even if only some are remapped. It returns ""
// if dr is nil or an identity, in which case no device map is needed.
func cudaCheckpointDeviceMap(dr *nvproxy.DeviceRemapping) (string, error) {
	if dr == nil {
		return "", nil
	}
	identity := true
	pairs := make([]string, 0, len(dr.OldDeviceByMinor))
	for _, oldMinor := range slices.Sorted(maps.Keys(dr.OldDeviceByMinor)) {
		oldID := dr.OldDeviceByMinor[oldMinor]
		newID := dr.NewDeviceByOld[oldID]
		if oldID.UUID == "" || newID.UUID == "" {
			return "", fmt.Errorf("nvproxy device has no UUID: %v => %v", oldID, newID)
		}
		if oldID.UUID != newID.UUID {
			identity = false
		}
		pairs = append(pairs, oldID.UUID+"="+newID.UUID)
	}
	if identity {
		return "", nil
	}
	return strings.Join(pairs, ","), nil
}

// restoreCudaProcs restores CUDA state in all of the given (currently
// checkpointed) CUDA processes, the inverse of checkpointCudaProcs. If
// nvproxyRemapping maps any device to a different one, CUDA state is restored
// onto the new devices via cuda-checkpoint's --device-map flag.
//
// Failures are not undone: if CUDA can't be restored, the sandbox can't make
// progress regardless, so the error is simply returned to the caller.
func restoreCudaProcs(sctx context.Context, k *kernel.Kernel, cudaCheckpointPath string, cudaProcs []*kernel.ThreadGroup, timeline *timing.Timeline, sequential bool, nvproxyRemapping *nvproxy.DeviceRemapping) error {
	deviceMap, err := cudaCheckpointDeviceMap(nvproxyRemapping)
	if err != nil {
		return err
	}
	start := time.Now()
	nullFD, cleanup := openCudaCheckpointNullFD(sctx, k)
	defer cleanup()
	if deviceMap == "" {
		// --toggle transitions checkpointed => running in a single invocation.
		restored, err := runCudaAction(sctx, k, cudaCheckpointPath, cudaProcs, []string{"--toggle"}, !sequential, nullFD)
		timeline.Reached("cuda toggled to running")
		if err != nil {
			return fmt.Errorf("cuda-checkpoint restore toggle failed: %w", err)
		}
		log.Infof("cuda-checkpoint restore toggle on %d processes took [%s]", len(restored), time.Since(start))
		return nil
	}
	// GPU migration via --device-map requires driver >= R580.
	if major := k.NvidiaDriverVersion.Major(); major < 580 {
		return fmt.Errorf("GPUs changed across restore, but cuda-checkpoint --device-map requires driver >= R580 (have R%d)", major)
	}
	log.Infof("cuda-checkpoint device map: %s", deviceMap)
	restored, err := runCudaAction(sctx, k, cudaCheckpointPath, cudaProcs, []string{"--action", "restore", "--device-map", deviceMap}, !sequential, nullFD)
	timeline.Reached("cuda restored")
	if err != nil {
		return fmt.Errorf("cuda-checkpoint restore failed: %w", err)
	}
	if _, err := runCudaAction(sctx, k, cudaCheckpointPath, restored, []string{"--action", "unlock"}, !sequential, nullFD); err != nil {
		return fmt.Errorf("cuda-checkpoint unlock failed: %w", err)
	}
	timeline.Reached("cuda unlocked")
	log.Infof("cuda-checkpoint restore+unlock on %d processes took [%s]", len(restored), time.Since(start))
	return nil
}
