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
	"sort"
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

	// cudaBlockerPollInterval is how often the checkpoint-blocker gate
	// re-polls nvproxy for live blockers before giving up.
	cudaBlockerPollInterval = 500 * time.Millisecond

	// cudaLockGateAttempts bounds how many times the (gate, lock) pair is
	// retried when the lock cannot quiesce every rank. Only meaningful with
	// the multicast interposer, whose gate is what gets released between
	// attempts to let a deadlocked collective drain.
	cudaLockGateAttempts = 8

	// cudaLockGateRetryDelay is how long the gate stays released between
	// attempts, giving in-flight collectives time to complete.
	cudaLockGateRetryDelay = 500 * time.Millisecond
)

// DefaultCudaBlockerTimeout is the default for SaveOpts.CudaBlockerTimeout:
// how long preSaveCuda waits for checkpoint blockers (multicast/fabric
// objects, exported-object FDs) to disappear before failing the checkpoint.
// Blockers only disappear if the application tears the resources down itself,
// so a short default just bounds the wait; nvproxy-driven multicast suspend
// (which removes the blockers) is a separate, later step.
const DefaultCudaBlockerTimeout = 10 * time.Second

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

	// Gate: cuda-checkpoint cannot serialize multicast/fabric memory or live
	// CUDA IPC exports; attempting to checkpoint with such resources live
	// hangs or corrupts the snapshot. Poll (the app may be tearing them down)
	// and fail loudly with a per-client attribution if they persist.
	//
	// Multicast/fabric objects are exempt when the LD_PRELOADed interposer is
	// present: checkpointCudaProcs drives it to release them between the
	// cuda-checkpoint lock and checkpoint phases (see state_cuda_shim.go).
	// All other blockers (e.g. exported fds) always gate.
	shimDir := cudaShimDir(k, cudaProcs)
	if err := waitForCudaCheckpointBlockers(k, o.CudaBlockerTimeout, shimDir != "" /* shimWillRelease */); err != nil {
		if wasPaused {
			k.Pause()
		}
		return err
	}
	// FIXME: b/456299722
	for _, tg := range cudaProcs {
		tg.SigsegvLock()
	}
	err := checkpointCudaProcs(sctx, k, o.CudaCheckpointPath, cudaProcs, o.CudaCheckpointSequential, o.CudaBlockerTimeout, shimDir)
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
		// Bring multicast back and let the application run again.
		if shimDir != "" {
			unwindCudaMulticastShim(sctx, k, cudaProcs, shimDir)
		}
		if wasPaused {
			k.Pause()
		}
		return err
	}
	if wasPaused {
		k.Pause()
	}
	k.AddStateToCheckpoint(cudaCheckpointPathKey, o.CudaCheckpointPath)
	k.AddStateToCheckpoint(cudaCheckpointSequentialKey, o.CudaCheckpointSequential)
	k.AddStateToCheckpoint(cudaProcsKey, cudaProcs)
	return nil
}

// waitForCudaCheckpointBlockers polls nvproxy for checkpoint blockers
// (fabric RM objects and exported-object FDs) until they disappear or timeout
// elapses, in which case it returns an error attributing the blockers to
// their owning clients/tasks.
//
// When shimWillRelease is true, blockers the interposer's suspend releases
// are NOT waited for: multicast objects (unbound + released between the
// cuda-checkpoint lock and checkpoint phases, rebuilt after the post-restore
// toggle) and the fabric / fabric-import companion objects of VMM
// exports/imports (the driver allocates an NV_MEMORY_FABRIC object for a
// POSIX-FD cuMemExportToShareableHandle on fabric-attached GPUs; it is
// released along with the export/import state the interposer tears down).
// The application need not release these itself. A second, strict gate runs
// after the interposer's suspend and before cuda-checkpoint, so an object
// this exemption mispredicts still fails the checkpoint loudly rather than
// hanging cuda-checkpoint.
func waitForCudaCheckpointBlockers(k *kernel.Kernel, timeout time.Duration, shimWillRelease bool) error {
	if timeout <= 0 {
		timeout = DefaultCudaBlockerTimeout
	}
	deadline := time.Now().Add(timeout)
	var lastLog time.Time
	blockers := gatedBlockers(nvproxy.CheckpointBlockers(k.VFS()), shimWillRelease)
	for len(blockers) != 0 {
		if time.Now().After(deadline) {
			return fmt.Errorf("cuda-checkpoint cannot proceed: %d resource(s) it cannot serialize are still live after %s: %s",
				len(blockers), timeout, nvproxy.FormatBlockersByClient(blockers))
		}
		// Rate-limit: with a long user-set timeout, logging the full blocker
		// list every poll would flood the log.
		if time.Since(lastLog) >= 5*time.Second {
			lastLog = time.Now()
			log.Infof("Waiting for CUDA checkpoint blockers to be released: %s", nvproxy.FormatBlockersByClient(blockers))
		}
		time.Sleep(cudaBlockerPollInterval)
		blockers = gatedBlockers(nvproxy.CheckpointBlockers(k.VFS()), shimWillRelease)
	}
	return nil
}

// gatedBlockers returns the blockers that must gate the checkpoint. When
// shimWillRelease is true, the kinds the interposer's suspend releases later
// in the checkpoint sequence (multicast objects and fabric / fabric-import
// companions of VMM exports and imports) are not gated; all other blockers
// always gate.
func gatedBlockers(blockers []nvproxy.CheckpointBlocker, shimWillRelease bool) []nvproxy.CheckpointBlocker {
	if !shimWillRelease {
		return blockers
	}
	var out []nvproxy.CheckpointBlocker
	for _, b := range blockers {
		switch b.Kind {
		case nvproxy.BlockerKindMulticast, nvproxy.BlockerKindFabric, nvproxy.BlockerKindFabricImport:
			// Released by the interposer's suspend.
		default:
			out = append(out, b)
		}
	}
	return out
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
	// ForEachThreadGroup above iterates a map, so without this the order in
	// which cuda-checkpoint actions are issued -- and therefore which process
	// is toggled first on restore -- would differ on every run. That never
	// caused a failure by itself (measured: no ordering rule fits the observed
	// toggle failures, and parallel toggling is no better), but it makes any
	// failure unreproducible run to run, which is a debugging tax on every
	// investigation downstream of this list. Sorting last keeps the invariant
	// local: nothing after this line may reorder the slice.
	sort.Slice(procs, func(i, j int) bool { return procs[i].ID() < procs[j].ID() })
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

	// Recreate host-freed FLA registrations. Only the resume-after-save path
	// needs this (the registrations were freed for the checkpoint and
	// cuda-checkpoint's restore knows nothing about them); after a true
	// restore this is a no-op, because the afterLoad object replay already
	// recreated them. Ordering: after the toggle (the covered vidmem must
	// exist again), before the interposer resume (whose re-exports assume
	// exporter-side state is whole).
	if err == nil {
		if n, rerr := nvproxy.ReplayFLARegistrations(k.VFS()); rerr != nil {
			err = fmt.Errorf("replaying FLA registrations: %w", rerr)
		} else if n > 0 {
			log.Infof("nvproxy: replayed %d FLA registrations after resume", n)
		}
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
		if rerr := resumeCudaMulticastShim(k.SupervisorContext(), k, cudaCheckpointPath, cudaProcs); rerr != nil {
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
	// output "running". Iterate the input slice rather than the map so the
	// caller's order is preserved.
	var res []*kernel.ThreadGroup
	for _, cudaProc := range cudaProcs {
		ckptProc, ok := ckptProcs[cudaProc]
		if !ok {
			continue
		}
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
	// Collect results by iterating the input slice, not the map: succeeded
	// feeds later sequential phases, and map iteration order would silently
	// re-randomize the deterministic order cudaProcs() established.
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
func checkpointCudaProcs(sctx context.Context, k *kernel.Kernel, cudaCheckpointPath string, cudaProcs []*kernel.ThreadGroup, sequential bool, blockerTimeout time.Duration, shimDir string) error {
	start := time.Now()
	nullFD, cleanup := openCudaCheckpointNullFD(sctx, k)
	defer cleanup()

	// Phase 1: bar the application from the GPU, then lock every process in
	// parallel so coupled ranks quiesce together. --timeout bounds how long
	// each lock waits for its process to become lockable.
	//
	// The gate and the lock cover different halves of the problem, and neither
	// suffices alone. The gate stops *new* submissions but cannot drain work
	// already in flight. The lock drains and preempts in-flight work but
	// cannot keep up with a workload that never idles, and then reports
	// "device not ready". So gate first, leaving the lock only the in-flight
	// work to deal with.
	//
	// Gating can however deadlock a collective: a rank gated just before
	// submitting collective N starves peers already spinning in N, and the
	// lock cannot quiesce those peers either. Releasing the gate lets that
	// collective complete, so retry the pair a bounded number of times -- each
	// attempt is a fresh chance to catch every rank between collectives. If it
	// never converges, fail cleanly with the application still running.
	lockArgs := []string{"--action", "lock", "--timeout", strconv.Itoa(cudaLockTimeoutMS)}
	var locked []*kernel.ThreadGroup
	var err error
	for attempt := 1; ; attempt++ {
		if shimDir != "" {
			if err = armCudaMulticastShimGate(sctx, k, cudaProcs, shimDir); err != nil {
				return err
			}
		}
		locked, err = runCudaAction(sctx, k, cudaCheckpointPath, cudaProcs, lockArgs, true /* parallel */, nullFD)
		if err == nil {
			break
		}
		// Unlock whatever did lock, so ranks holding peers back can make
		// progress before the next attempt.
		if _, uerr := runCudaAction(sctx, k, cudaCheckpointPath, locked, []string{"--action", "unlock"}, true, nullFD); uerr != nil {
			log.Warningf("cuda-checkpoint unlock between lock attempts failed: %v", uerr)
		}
		if shimDir == "" || attempt >= cudaLockGateAttempts {
			break
		}
		log.Infof("cuda-checkpoint lock attempt %d/%d did not quiesce all ranks; releasing the interposer gate to let in-flight collectives drain, then retrying: %v",
			attempt, cudaLockGateAttempts, err)
		if rerr := cudaShimSetMarker(sctx, k, cudaProcs, shimDir, cudaShimGateMarker, false /* set */); rerr != nil {
			log.Warningf("failed to release multicast interposer gate between lock attempts: %v", rerr)
		}
		time.Sleep(cudaLockGateRetryDelay)
	}
	if err != nil {
		if shimDir != "" {
			unwindCudaMulticastShim(sctx, k, cudaProcs, shimDir)
		}
		return fmt.Errorf("cuda-checkpoint lock phase failed: %w", err)
	}

	// Interposer teardown, sandwiched inside the lock.
	//
	// Two constraints collide. The interposer must issue libcuda calls
	// (cuMemUnmap / cuMulticastUnbind / cuMemRelease), which a locked process
	// cannot do. But it must not tear multicast down while the application is
	// still using it, and the application cannot simply be gated first: a rank
	// gated before submitting its next collective starves peers already
	// spinning in that collective, so the gate alone deadlocks the drain
	// (observed with NVLS disabled and a workload with no idle gap).
	//
	// cuda-checkpoint's parallel lock is precisely the thing that can quiesce
	// coupled ranks. So: arm the gate while still locked -- the interposer only
	// flips a flag and issues no CUDA calls, so this is safe -- then unlock, so
	// the teardown runs against an already-drained GPU that the application is
	// barred from touching. Then re-lock for the checkpoint.
	if shimDir != "" {
		// undo returns the application to running after a failure in this
		// window. Unlock FIRST: the unwind's rebuild issues libcuda calls
		// that a locked process cannot make, so unwinding first would stall
		// until the ack timeout. stillLocked names the processes that hold a
		// cuda-checkpoint lock at the failure point (nil when the failure
		// happened with everything already unlocked, where a blanket unlock
		// would only produce misleading "unlock failed" warnings).
		undo := func(stillLocked []*kernel.ThreadGroup) {
			// Recreate host-freed FLA registrations FIRST (sentry-driven, so
			// process lock state is irrelevant): once the application runs
			// again, libcuda may reference their handles.
			if n, rerr := nvproxy.ReplayFLARegistrations(k.VFS()); rerr != nil {
				log.Warningf("replaying FLA registrations during checkpoint unwind failed after %d: %v", n, rerr)
			}
			if len(stillLocked) != 0 {
				if _, uerr := runCudaAction(sctx, k, cudaCheckpointPath, stillLocked, []string{"--action", "unlock"}, true, nullFD); uerr != nil {
					log.Warningf("cuda-checkpoint unlock during checkpoint unwind failed: %v", uerr)
				}
			}
			// Unwind with the full, never-reassigned process list: `locked`
			// is reassigned by the re-lock attempt below and can be a
			// partial set (or nil) at the time undo runs. Extra entries are
			// harmless -- procs without a suspended.<pid> ack are not waited
			// on. preSaveCuda's outer unwind is the idempotent backstop for
			// anything this misses.
			unwindCudaMulticastShim(sctx, k, cudaProcs, shimDir)
		}
		if _, err := runCudaAction(sctx, k, cudaCheckpointPath, locked, []string{"--action", "unlock"}, true, nullFD); err != nil {
			undo(locked)
			return fmt.Errorf("cuda-checkpoint unlock before multicast teardown failed: %w", err)
		}
		if err := suspendCudaMulticastShim(sctx, k, locked, shimDir); err != nil {
			undo(nil)
			return err
		}
		// Host-free driver-internal FLA registrations (NV_MEMORY_FABRIC
		// objects covering peer-shared VMM allocations). The interposer
		// cannot release these -- no CUDA API frees them -- and
		// cuda-checkpoint checkpoints them but cannot restore them. They
		// stay in the object graph marked hostFreed; the afterLoad object
		// replay recreates them (after the vidmem they cover, via a
		// restore-ordering dependency) before the post-restore toggle.
		if n, err := nvproxy.SuspendFLARegistrations(k.VFS()); err != nil {
			undo(nil)
			return fmt.Errorf("suspending FLA registrations: %w", err)
		} else if n > 0 {
			log.Infof("nvproxy: host-freed %d FLA registrations for checkpoint", n)
		}
		// Verify rather than trust: the interposer acknowledging its suspend
		// does not by itself prove the process is serializable. Re-run the
		// gate with nothing exempt, so anything left unreleased fails here
		// instead of becoming a snapshot that only misbehaves after restore.
		if err := waitForCudaCheckpointBlockers(k, blockerTimeout, false /* shimWillRelease */); err != nil {
			undo(nil)
			return fmt.Errorf("multicast interposer suspended but resources remain: %w", err)
		}
		var err error
		if locked, err = runCudaAction(sctx, k, cudaCheckpointPath, cudaProcs, lockArgs, true /* parallel */, nullFD); err != nil {
			// locked now holds the partial set that DID re-lock.
			undo(locked)
			return fmt.Errorf("cuda-checkpoint re-lock after multicast teardown failed: %w", err)
		}
	}

	// Phase 2: checkpoint all locked processes.
	if _, err := runCudaAction(sctx, k, cudaCheckpointPath, locked, []string{"--action", "checkpoint"}, !sequential, nullFD); err != nil {
		// Best-effort undo: restore then unlock, returning the app to running.
		if _, rerr := runCudaAction(sctx, k, cudaCheckpointPath, locked, []string{"--action", "restore"}, !sequential, nullFD); rerr != nil {
			log.Warningf("cuda-checkpoint restore after checkpoint-phase failure also failed: %v", rerr)
		}
		// Recreate host-freed FLA registrations before the app runs again
		// (no-op when none were freed, e.g. the shimless path).
		if n, rerr := nvproxy.ReplayFLARegistrations(k.VFS()); rerr != nil {
			log.Warningf("replaying FLA registrations after checkpoint-phase failure failed after %d: %v", n, rerr)
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
// (restore + unlock in one atomic step), sequentially by default.
//
// What is measured: splitting restore and unlock into separate all-process
// phases fails (importer restore hits "unknown error", or resumed ranks hit
// "unspecified launch failure" on their first kernel launch), and a parallel
// toggle is no better than a sequential one. What is NOT established is any
// ordering rule among the members: captured failures fit no exporter-first or
// position-based pattern, so the sequencing here should be read as "one
// atomic toggle per process, one at a time", not as a dependency order.
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
