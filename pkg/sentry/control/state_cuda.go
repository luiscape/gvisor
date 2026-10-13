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

	"gvisor.dev/gvisor/pkg/abi/linux"
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

	// cudaMulticastShimKey records, in the checkpoint, that the CUDA processes
	// were checkpointed with the multicast interposer enabled, so that restore
	// runs cuda-checkpoint the same way (see withCudaMulticastShim). It is only
	// added when the interposer is enabled.
	cudaMulticastShimKey = "cuda-multicast-shim"

	// cudaCheckpointControlTimeout bounds, with the multicast interposer, each
	// cuda-checkpoint invocation that runs before any GPU state is saved:
	// --get-state, --action lock, and --action unlock. Those finish in seconds
	// (lock bounds itself, see cudaLockTimeoutMS), and one killed on this bound
	// is undone: the checkpoint fails and the application keeps running.
	// --action checkpoint and everything on restore are not bounded, as without
	// the interposer: their duration grows with GPU memory, and killing one
	// partway leaves the process in a state cuda-checkpoint does not define.
	cudaCheckpointControlTimeout = 2 * time.Minute

	// cudaCheckpointKillGrace bounds the wait for a killed cuda-checkpoint to
	// exit. A task blocked in a host driver call cannot exit until it returns.
	cudaCheckpointKillGrace = 10 * time.Second

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
	// Without the multicast interposer, checkpoints are as they were before
	// it: one unbounded cuda-checkpoint --toggle per process.
	var controlTimeout time.Duration
	if o.CudaMulticastShim {
		controlTimeout = cudaCheckpointControlTimeout
		sctx = withCudaMulticastShim(sctx)
	}
	fail := func(err error) error {
		if wasPaused {
			k.Pause()
		}
		return err
	}
	cudaProcs, err := cudaProcs(sctx, k, o.CudaCheckpointPath, k.NvidiaDriverVersion.Major(), controlTimeout)
	if err != nil {
		return fail(err)
	}

	shim := false
	if o.CudaMulticastShim {
		// cuda-checkpoint hangs on processes holding NVLink multicast memory
		// (e.g. NCCL with NVLS) and cannot restore memory imported from an
		// exported fd, so refuse up front, unless they run the multicast
		// interposer, which releases both before cuda-checkpoint runs
		// (checkpointCudaProcs re-checks afterwards).
		managed := cudaShimManagedProcs(sctx, k, cudaProcs)
		shim = len(managed) != 0
		except := make(map[kernel.ThreadID]bool, len(managed))
		for _, tg := range managed {
			except[tg.ID()] = true
		}
		if blockers := nvproxy.CheckpointBlockers(k.VFS(), except); blockers != "" {
			return fail(fmt.Errorf("cannot checkpoint CUDA processes holding multicast or imported memory that do not run the multicast interposer: %s", blockers))
		}
	}
	// FIXME: b/456299722
	for _, tg := range cudaProcs {
		tg.SigsegvLock()
	}
	if !o.CudaMulticastShim {
		// Undo the --toggle on success procs if any fail, so the application can
		// continue running after the failed checkpoint is reported.
		err = runCudaCheckpointOp(sctx, k, o.CudaCheckpointPath, cudaProcs, nil, o.CudaCheckpointSequential, cudaCheckpointToggle, &cudaCheckpointToggle)
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
		addCudaCheckpointState(k, o, cudaProcs)
		return nil
	}

	if err := checkpointCudaProcs(sctx, k, o.CudaCheckpointPath, cudaProcs, o.CudaCheckpointSequential, shim, controlTimeout); err != nil {
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
		// teardown failed partway (see errCudaShimTornDown) or a process could
		// not be returned to running (see errCudaUnrecoverable).
		if shim && !errors.Is(err, errCudaShimTornDown) && !errors.Is(err, errCudaUnrecoverable) {
			unwindCudaMulticastShim(sctx, k, cudaProcs)
		}
		return fail(err)
	}
	if wasPaused {
		k.Pause()
	}
	addCudaCheckpointState(k, o, cudaProcs)
	k.AddStateToCheckpoint(cudaMulticastShimKey, true)
	return nil
}

// addCudaCheckpointState records in the checkpoint what postRestoreCuda needs
// to restore cudaProcs.
func addCudaCheckpointState(k *kernel.Kernel, o *state.SaveOpts, cudaProcs []*kernel.ThreadGroup) {
	k.AddStateToCheckpoint(cudaCheckpointPathKey, o.CudaCheckpointPath)
	k.AddStateToCheckpoint(cudaCheckpointSequentialKey, o.CudaCheckpointSequential)
	k.AddStateToCheckpoint(cudaProcsKey, cudaProcs)
}

// cudaMulticastShimCtxKey is the context key under which withCudaMulticastShim
// marks a context.
type cudaMulticastShimCtxKey struct{}

// withCudaMulticastShim returns ctx marked as running cuda-checkpoint for
// processes that run with the multicast interposer enabled; see
// invokeCudaCheckpoint.
func withCudaMulticastShim(ctx context.Context) context.Context {
	return context.WithValue(ctx, cudaMulticastShimCtxKey{}, true)
}

// cudaMulticastShimFromContext reports whether ctx was marked by
// withCudaMulticastShim.
func cudaMulticastShimFromContext(ctx context.Context) bool {
	return ctx.Value(cudaMulticastShimCtxKey{}) != nil
}

// cudaProcs returns a list of all CUDA processes in the sandbox. It selects
// them by collecting processes whose FD table has an open file descriptor to
// any CUDA device. With a timeout, it fails if a process cannot be classified
// in time (see filterCudaProcsUsingGetState).
//
// Callers must not hold any thread-group leader's Task.mu in k.
// checklocks cannot name the leader mutexes selected by ForEachThreadGroup.
func cudaProcs(sctx context.Context, k *kernel.Kernel, cudaCheckpointPath string, nvidiaDriverVersionMajor int, timeout time.Duration) ([]*kernel.ThreadGroup, error) {
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
		return filterCudaProcsUsingGetState(sctx, k, cudaCheckpointPath, procs, timeout)
	}
	return procs, nil
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
	shim := k.PopCheckpointState(cudaMulticastShimKey) != nil
	sctx := k.SupervisorContext()
	if shim {
		sctx = withCudaMulticastShim(sctx)
	}
	timeline.Reached("starting cuda-ckpt")
	err := restoreCudaProcs(sctx, k, cudaCheckpointPath, cudaProcs, timeline, cudaCheckpointSequential, nvproxyRemapping)
	// FIXME: b/456299722
	for _, tg := range cudaProcs {
		tg.SigsegvUnlock()
	}
	if !shim {
		return err
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
		if rerr := resumeCudaMulticastShim(sctx, k, cudaProcs); rerr != nil {
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

// wait waits for p to exit until deadline (forever if zero), then kills it.
// It returns an error if p had to be killed, or would not exit.
func (p checkpointProc) wait(deadline time.Time) error {
	if deadline.IsZero() {
		p.tg.WaitExited()
		return nil
	}
	done := make(chan struct{})
	go func() {
		p.tg.WaitExited()
		close(done)
	}()
	timer := time.NewTimer(time.Until(deadline))
	defer timer.Stop()
	select {
	case <-done:
		return nil
	case <-timer.C:
	}
	select {
	case <-done:
		return nil
	default:
	}
	p.tg.SendSignal(&linux.SignalInfo{Signo: int32(linux.SIGKILL)})
	select {
	case <-done:
		return fmt.Errorf("%s did not finish in time and was killed", p.desc)
	case <-time.After(cudaCheckpointKillGrace):
		return fmt.Errorf("%s did not finish in time, and did not exit when killed", p.desc)
	}
}

// invokeCudaCheckpoint invokes cuda-checkpoint on the given CUDA process with
// the given operation flag. On success it returns a checkpointProc struct
// containing the running cuda-checkpoint process and a cleanup function which
// must be called to release resources. If cudaProc has exited, it returns
// (checkpointProc.tg == nil, err == nil).
func invokeCudaCheckpoint(sctx context.Context, k *kernel.Kernel, proc *Proc, cudaCheckpointPath string, cudaProc *kernel.ThreadGroup, op cudaCheckpointOp, nullFD *vfs.FileDescription) (checkpointProc, func(), error) {
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
	argv := make([]string, 0, 3+len(op.args))
	argv = append(argv, "cuda-checkpoint")
	argv = append(argv, op.args...)
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
	if cudaMulticastShimFromContext(sctx) {
		// The multicast interposer is preloaded into every container binary via
		// /etc/ld.so.preload, including this cuda-checkpoint process. Disable it
		// here: it has no business interposing cuda-checkpoint, which must not
		// start a control thread or take part in the interposer's protocol.
		args.Envv = append(slices.Clip(args.Envv), "MCSHIM_DISABLE=1")
	}

	// Provide standard streams to cuda-checkpoint. Use /dev/null as stdin
	// and direct cuda-checkpoint's stdout/stderr to a pipe.
	ckptDesc := fmt.Sprintf("cuda-checkpoint %s for PID %d in container %q", op.desc, pid, contID)
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

// filterCudaProcsUsingGetState returns the processes in cudaProcs that
// cuda-checkpoint --get-state reports as CUDA processes. One that fails
// --get-state is not one (NVML-only, say) and is left out. One whose
// --get-state does not finish within timeout, if there is one, is an error:
// it is a CUDA process in some state cuda-checkpoint cannot handle, and
// skipping it would leave its live GPU objects to fail the save later, after
// the other processes have been checkpointed.
func filterCudaProcsUsingGetState(sctx context.Context, k *kernel.Kernel, cudaCheckpointPath string, cudaProcs []*kernel.ThreadGroup, timeout time.Duration) ([]*kernel.ThreadGroup, error) {
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
	deadline := cudaCheckpointDeadline(timeout)
	for _, cudaProc := range cudaProcs {
		ckptProc, cleanup, err := invokeCudaCheckpoint(sctx, k, proc, cudaCheckpointPath, cudaProc, cudaCheckpointGetState, nullFD)
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
	// output "running". Every invocation is waited for, and killed on the
	// deadline, before a timeout is reported; one stuck in a host driver call
	// may still not have exited (see checkpointProc.wait).
	var res []*kernel.ThreadGroup
	var errs []error
	for cudaProc, ckptProc := range ckptProcs {
		if err := ckptProc.wait(deadline); err != nil {
			errs = append(errs, fmt.Errorf("cannot checkpoint PID %d: %w", cudaProc.ID(), err))
			continue
		}
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
	if len(errs) > 0 {
		return nil, errors.Join(errs...)
	}
	return res, nil
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

// cudaCheckpointDeadline returns the deadline for an invocation started now:
// zero (none) if timeout is zero.
func cudaCheckpointDeadline(timeout time.Duration) time.Time {
	if timeout <= 0 {
		return time.Time{}
	}
	return time.Now().Add(timeout)
}

// cudaCheckpointOp describes an operation to perform with cuda-checkpoint.
type cudaCheckpointOp struct {
	// args are the arguments to cuda-checkpoint that specify the operation, not
	// including --pid.
	args []string
	// desc is a short name for the operation, used in logs and timeline names.
	desc string
}

var (
	cudaCheckpointToggle = cudaCheckpointOp{
		args: []string{"--toggle"},
		desc: "toggle",
	}
	cudaCheckpointGetState = cudaCheckpointOp{
		args: []string{"--get-state"},
		desc: "get-state",
	}

	cudaCheckpointUnlock = cudaCheckpointOp{
		args: []string{"--action", "unlock"},
		desc: "unlock",
	}

	// The operations below are used only with the multicast interposer.
	cudaCheckpointLock = cudaCheckpointOp{
		args: []string{"--action", "lock", "--timeout", strconv.Itoa(cudaLockTimeoutMS)},
		desc: "lock",
	}
	cudaCheckpointCheckpoint = cudaCheckpointOp{
		args: []string{"--action", "checkpoint"},
		desc: "checkpoint",
	}
	cudaCheckpointRestore = cudaCheckpointOp{
		args: []string{"--action", "restore"},
		desc: "restore",
	}
)

// runCudaCheckpointOp invokes cuda-checkpoint with the given operation on all
// of the given CUDA processes and waits for them to finish. If any invocation
// fails and undoOp is non-nil, undoOp is invoked on all processes for which
// op succeeded, on a best-effort basis, to restore their original state.
func runCudaCheckpointOp(sctx context.Context, k *kernel.Kernel, cudaCheckpointPath string, cudaProcs []*kernel.ThreadGroup, timeline *timing.Timeline, sequential bool, op cudaCheckpointOp, undoOp *cudaCheckpointOp) error {
	start := time.Now()

	// Open /dev/null once for the stdin of all cuda-checkpoint processes.
	nullVD := k.VFS().NewAnonVirtualDentry("null")
	defer nullVD.DecRef(sctx)
	nullFD, err := memdev.NewNullFD(sctx, nullVD.Mount(), nullVD.Dentry(), vfs.OpenOptions{})
	if err != nil {
		log.Warningf("Failed to open /dev/null for cuda-checkpoint stdin: %v", err)
	} else {
		defer nullFD.DecRef(sctx)
	}

	// Call cuda-checkpoint for each CUDA PID.
	ckptTimerNames := make([]string, len(cudaProcs))
	for i, cudaProc := range cudaProcs {
		ckptTimerNames[i] = fmt.Sprintf("cuda-ckpt %s %s", op.desc, cudaProc.ID())
	}
	var ckptTimings []*timing.Lease
	if !sequential {
		ckptTimelines := timeline.MultiFork(ckptTimerNames)
		ckptTimings = make([]*timing.Lease, len(cudaProcs))
		for i := range cudaProcs {
			ckptTimings[i] = ckptTimelines[i].Lease()
		}
	}
	defer func() {
		for _, t := range ckptTimings {
			t.End()
		}
	}()
	proc := &Proc{Kernel: k}
	ckptProcs := make(map[*kernel.ThreadGroup]checkpointProc)
	var errs []error
	for i, cudaProc := range cudaProcs {
		var ckptTiming *timing.Lease
		if sequential {
			ckptTiming = timeline.Fork(ckptTimerNames[i]).Lease()
		} else {
			ckptTiming = ckptTimings[i]
		}
		ckptProc, cleanup, err := invokeCudaCheckpoint(sctx, k, proc, cudaCheckpointPath, cudaProc, op, nullFD)
		if err != nil {
			ckptTiming.Reached("invoke error")
			errs = append(errs, err)
			break
		}
		if ckptProc.tg == nil {
			ckptTiming.Reached("tg nil")
			continue
		}
		ckptProcs[cudaProc] = ckptProc
		ckptTimeline := ckptTiming.Transfer()
		if sequential {
			ckptProc.tg.WaitExited()
			if status := ckptProc.tg.ExitStatus(); status != 0 {
				ckptTimeline.Reached("exec error")
			}
			ckptTimeline.End()
		} else {
			go func() {
				defer ckptTimeline.End()
				ckptProc.tg.WaitExited()
				if status := ckptProc.tg.ExitStatus(); status != 0 {
					ckptTimeline.Reached("exec error")
				}
			}()
		}
		defer cleanup()
	}
	timeline.Reached("cuda-ckpts invoked")
	// Wait for all cuda-checkpoint processes to exit. Remove all failed
	// cuda-checkpoint attempts from ckptProcs, so ckptProcs only contains the
	// successful ones.
	for cudaProc, ckptProc := range ckptProcs {
		if !sequential {
			ckptProc.tg.WaitExited()
		}
		if status := ckptProc.tg.ExitStatus(); status != 0 {
			if ckptProc.out != nil {
				errs = append(errs, fmt.Errorf("%q failed with exit status %d; output: %q", ckptProc.desc, status, ckptProc.out.String()))
			} else {
				errs = append(errs, fmt.Errorf("%q failed with exit status %d", ckptProc.desc, status))
			}
			delete(ckptProcs, cudaProc)
		} else if log.IsLogging(log.Debug) && ckptProc.out != nil {
			log.Debugf("%s succeeded; output: %q", ckptProc.desc, ckptProc.out.String())
		}
	}
	timeline.Reached("cuda-ckpts waited")
	if len(errs) > 0 {
		if undoOp == nil {
			return errors.Join(errs...)
		}
		// Undo the operation for all the successful ones to restore the original
		// state. This is best-effort.
		undoCkptProcs := make(map[string]checkpointProc)
		for cudaProc, ckptProc := range ckptProcs {
			undoCkptProc, cleanup, err := invokeCudaCheckpoint(sctx, k, proc, cudaCheckpointPath, cudaProc, *undoOp, nullFD)
			if err != nil {
				log.Warningf("Failed to invoke cuda-checkpoint to undo %q: %v", ckptProc.desc, err)
				continue
			}
			if undoCkptProc.tg == nil {
				continue
			}
			if sequential {
				undoCkptProc.tg.WaitExited()
			}
			undoCkptProcs[ckptProc.desc] = undoCkptProc
			defer cleanup()
		}
		for ckptProcDesc, undoCkptProc := range undoCkptProcs {
			if !sequential {
				undoCkptProc.tg.WaitExited()
			}
			if status := undoCkptProc.tg.ExitStatus(); status != 0 {
				if undoCkptProc.out != nil {
					log.Warningf("Undoing %q failed with exit status %d; output: %q", ckptProcDesc, status, undoCkptProc.out.String())
				} else {
					log.Warningf("Undoing %q failed with exit status %d", ckptProcDesc, status)
				}
			}
		}
		// Combine all errors and return.
		return errors.Join(errs...)
	}
	log.Infof("cuda-checkpoint %s on %d processes took [%s]", op.desc, len(ckptProcs), time.Since(start))
	timeline.Reached("cuda-ckpts done")
	return nil
}

// runCudaAction invokes `cuda-checkpoint <op> --pid <pid>` on every process in
// cudaProcs, for the multicast interposer's protocol (checkpoints without it
// use runCudaCheckpointOp). When parallel is true all invocations run
// concurrently; otherwise they run one at a time. Each invocation still
// running after timeout (if non-zero) is killed and counts as failed. It
// returns the processes for which the action succeeded (exit status 0) and a
// combined error describing any failures.
func runCudaAction(sctx context.Context, k *kernel.Kernel, cudaCheckpointPath string, cudaProcs []*kernel.ThreadGroup, op cudaCheckpointOp, parallel bool, nullFD *vfs.FileDescription, timeout time.Duration) ([]*kernel.ThreadGroup, error) {
	return runCudaActionStop(sctx, k, cudaCheckpointPath, cudaProcs, op, parallel, false /* stopOnError */, nullFD, timeout)
}

// runCudaActionStop is runCudaAction, except that when sequential (!parallel)
// and stopOnError, it invokes nothing after the first invocation that fails.
func runCudaActionStop(sctx context.Context, k *kernel.Kernel, cudaCheckpointPath string, cudaProcs []*kernel.ThreadGroup, op cudaCheckpointOp, parallel, stopOnError bool, nullFD *vfs.FileDescription, timeout time.Duration) ([]*kernel.ThreadGroup, error) {
	proc := &Proc{Kernel: k}
	ckptProcs := make(map[*kernel.ThreadGroup]checkpointProc)
	deadlines := make(map[*kernel.ThreadGroup]time.Time)
	waitErrs := make(map[*kernel.ThreadGroup]error)
	var errs []error
	stop := stopOnError && !parallel
	for _, cudaProc := range cudaProcs {
		ckptProc, cleanup, err := invokeCudaCheckpoint(sctx, k, proc, cudaCheckpointPath, cudaProc, op, nullFD)
		if err != nil {
			errs = append(errs, err)
			if stop {
				break
			}
			continue
		}
		if ckptProc.tg == nil {
			continue
		}
		ckptProcs[cudaProc] = ckptProc
		deadlines[cudaProc] = cudaCheckpointDeadline(timeout)
		defer cleanup()
		// In sequential mode, wait for each invocation to finish before starting
		// the next. In parallel mode, all invocations are launched first and
		// waited on below.
		if !parallel {
			waitErrs[cudaProc] = ckptProc.wait(deadlines[cudaProc])
			if stop && (waitErrs[cudaProc] != nil || ckptProc.tg.ExitStatus() != 0) {
				break
			}
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
			waitErrs[cudaProc] = ckptProc.wait(deadlines[cudaProc])
		}
		if err := waitErrs[cudaProc]; err != nil {
			errs = append(errs, err)
		} else if status := ckptProc.tg.ExitStatus(); status != 0 {
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
//
// controlTimeout bounds each lock and unlock invocation (see
// cudaCheckpointControlTimeout); checkpoint and its undo are not bounded.
func checkpointCudaProcs(sctx context.Context, k *kernel.Kernel, cudaCheckpointPath string, cudaProcs []*kernel.ThreadGroup, sequential bool, shim bool, controlTimeout time.Duration) error {
	start := time.Now()
	nullFD, cleanup := openCudaCheckpointNullFD(sctx, k)
	defer cleanup()
	unlock := func(tgs []*kernel.ThreadGroup) {
		if _, err := runCudaAction(sctx, k, cudaCheckpointPath, tgs, cudaCheckpointUnlock, true /* parallel */, nullFD, controlTimeout); err != nil {
			log.Warningf("cuda-checkpoint unlock after failure also failed: %v", err)
		}
	}

	// Phase 1: gate the application off the GPU, which stops new submissions,
	// then lock every process, which drains work in flight. A collective that
	// straddles the gate starves its peers, and the lock then times out.
	if shim {
		if err := armCudaMulticastShimGate(sctx, k, cudaProcs); err != nil {
			return err
		}
	}
	// On a lock failure, unlock every process, not just the ones reported
	// locked: a lock invocation that was killed on timeout may have left its
	// process locked. Unlocking a process that is not locked only fails, and
	// unlock logs that.
	locked, err := runCudaAction(sctx, k, cudaCheckpointPath, cudaProcs, cudaCheckpointLock, true /* parallel */, nullFD, controlTimeout)
	if err != nil {
		unlock(cudaProcs)
		return fmt.Errorf("cuda-checkpoint lock phase failed: %w", err)
	}

	// Interposer teardown, between two locks: it issues libcuda calls, which a
	// locked process cannot make. The gate keeps the application off the GPU.
	if shim {
		if _, err := runCudaAction(sctx, k, cudaCheckpointPath, locked, cudaCheckpointUnlock, true /* parallel */, nullFD, controlTimeout); err != nil {
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
		if locked, err = runCudaAction(sctx, k, cudaCheckpointPath, cudaProcs, cudaCheckpointLock, true /* parallel */, nullFD, controlTimeout); err != nil {
			unlock(cudaProcs)
			return fmt.Errorf("cuda-checkpoint re-lock after multicast teardown failed: %w", err)
		}
	}

	// Phase 2: checkpoint all locked processes.
	actions := cudaActions[*kernel.ThreadGroup]{
		run: func(procs []*kernel.ThreadGroup, op cudaCheckpointOp, parallel, stopOnError bool, timeout time.Duration) ([]*kernel.ThreadGroup, error) {
			return runCudaActionStop(sctx, k, cudaCheckpointPath, procs, op, parallel, stopOnError, nullFD, timeout)
		},
		state: func(proc *kernel.ThreadGroup) (string, error) {
			return cudaProcState(sctx, k, cudaCheckpointPath, proc, nullFD, controlTimeout)
		},
		name: func(proc *kernel.ThreadGroup) string { return fmt.Sprintf("PID %d", proc.ID()) },
	}
	if err := checkpointLocked(actions, locked, sequential, controlTimeout); err != nil {
		return err
	}

	log.Infof("cuda-checkpoint lock+checkpoint on %d processes took [%s]", len(locked), time.Since(start))
	return nil
}

// errCudaUnrecoverable reports that a failed checkpoint could not be undone:
// some process is in a state cuda-checkpoint could not bring back to running.
// The caller must not unwind the interposer, which needs every process
// running; the application stays gated.
var errCudaUnrecoverable = errors.New("the CUDA processes could not be returned to running; the application is left blocked and must be restarted")

// cudaActions runs cuda-checkpoint for checkpointLocked on processes of type
// P (*kernel.ThreadGroup, except in tests): run is runCudaActionStop, state
// reports a process's state as `cuda-checkpoint --get-state` prints it
// ("running", "locked", "checkpointed" or "failed"), and name names a process
// in errors.
type cudaActions[P comparable] struct {
	run   func(procs []P, op cudaCheckpointOp, parallel, stopOnError bool, timeout time.Duration) ([]P, error)
	state func(proc P) (string, error)
	name  func(proc P) string
}

// checkpointLocked checkpoints each of locked, which are all locked, one at a
// time if sequential, and stops at the first failure. It then returns every
// process to running: it asks each one that did not report success for its
// state, restores those that are checkpointed (cuda-checkpoint does not say
// where a failed checkpoint leaves a process), and unlocks all of them. If
// that cannot be done, it returns an error wrapping errCudaUnrecoverable.
// controlTimeout bounds each state query and unlock; checkpoint and restore
// are not bounded (see cudaCheckpointControlTimeout).
func checkpointLocked[P comparable](a cudaActions[P], locked []P, sequential bool, controlTimeout time.Duration) error {
	ckpted, err := a.run(locked, cudaCheckpointCheckpoint, !sequential, true /* stopOnError */, 0 /* timeout */)
	if err == nil {
		return nil
	}
	err = fmt.Errorf("cuda-checkpoint checkpoint phase failed: %w", err)
	unrecoverable := func(why error) error {
		return fmt.Errorf("%w, and undoing it failed (%v): %w", err, why, errCudaUnrecoverable)
	}

	toRestore := slices.Clone(ckpted)
	for _, tg := range locked {
		if slices.Contains(ckpted, tg) {
			continue
		}
		state, serr := a.state(tg)
		switch {
		case serr != nil:
			return unrecoverable(fmt.Errorf("getting the state of %s: %w", a.name(tg), serr))
		case state == "checkpointed":
			toRestore = append(toRestore, tg)
		case state == "locked" || state == "running":
		default:
			return unrecoverable(fmt.Errorf("%s is in state %q", a.name(tg), state))
		}
	}
	if len(toRestore) > 0 {
		if _, rerr := a.run(toRestore, cudaCheckpointRestore, !sequential, false /* stopOnError */, 0 /* timeout */); rerr != nil {
			return unrecoverable(fmt.Errorf("restoring the checkpointed processes: %w", rerr))
		}
	}
	// Unlocking a process that is running only fails; one that stays locked
	// cannot make the CUDA calls the interposer's rebuild needs.
	if unlocked, uerr := a.run(locked, cudaCheckpointUnlock, true /* parallel */, false /* stopOnError */, controlTimeout); uerr != nil && len(unlocked) < len(locked) {
		for _, tg := range locked {
			if slices.Contains(unlocked, tg) {
				continue
			}
			if state, serr := a.state(tg); serr != nil || state != "running" {
				return unrecoverable(fmt.Errorf("unlocking: %w", uerr))
			}
		}
	}
	return err
}

// cudaProcState returns cudaProc's state as `cuda-checkpoint --get-state`
// prints it, bounded by timeout.
func cudaProcState(sctx context.Context, k *kernel.Kernel, cudaCheckpointPath string, cudaProc *kernel.ThreadGroup, nullFD *vfs.FileDescription, timeout time.Duration) (string, error) {
	ckptProc, cleanup, err := invokeCudaCheckpoint(sctx, k, &Proc{Kernel: k}, cudaCheckpointPath, cudaProc, cudaCheckpointGetState, nullFD)
	if err != nil {
		return "", err
	}
	if ckptProc.tg == nil {
		return "", fmt.Errorf("PID %d has exited", cudaProc.ID())
	}
	defer cleanup()
	if err := ckptProc.wait(cudaCheckpointDeadline(timeout)); err != nil {
		return "", err
	}
	out := ""
	if ckptProc.out != nil {
		out = strings.TrimSpace(ckptProc.out.String())
	}
	if status := ckptProc.tg.ExitStatus(); status != 0 {
		return "", fmt.Errorf("%q failed with exit status %d; output: %q", ckptProc.desc, status, out)
	}
	return out, nil
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
// checkpointed) CUDA processes. If nvproxyRemapping maps any device to a
// different one, CUDA state is restored onto the new devices via
// cuda-checkpoint's --device-map flag.
//
// Failures are not undone: if CUDA can't be restored, the sandbox can't make
// progress regardless, so the error is simply returned to the caller.
func restoreCudaProcs(sctx context.Context, k *kernel.Kernel, cudaCheckpointPath string, cudaProcs []*kernel.ThreadGroup, timeline *timing.Timeline, sequential bool, nvproxyRemapping *nvproxy.DeviceRemapping) error {
	deviceMap, err := cudaCheckpointDeviceMap(nvproxyRemapping)
	if err != nil {
		return err
	}
	if deviceMap == "" {
		// --toggle transitions checkpointed => running in a single invocation.
		return runCudaCheckpointOp(sctx, k, cudaCheckpointPath, cudaProcs, timeline, sequential, cudaCheckpointToggle, nil)
	}
	// GPU migration via --device-map requires driver >= R580.
	if major := k.NvidiaDriverVersion.Major(); major < 580 {
		return fmt.Errorf("GPUs changed across restore, but cuda-checkpoint --device-map requires driver >= R580 (have R%d)", major)
	}
	log.Infof("cuda-checkpoint device map: %s", deviceMap)
	restoreOp := cudaCheckpointOp{
		args: []string{"--action", "restore", "--device-map", deviceMap},
		desc: "restore",
	}
	if err := runCudaCheckpointOp(sctx, k, cudaCheckpointPath, cudaProcs, timeline, sequential, restoreOp, nil); err != nil {
		return err
	}
	return runCudaCheckpointOp(sctx, k, cudaCheckpointPath, cudaProcs, timeline, sequential, cudaCheckpointUnlock, nil)
}
