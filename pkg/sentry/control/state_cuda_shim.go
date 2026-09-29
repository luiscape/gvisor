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
	"strings"
	"time"

	"gvisor.dev/gvisor/pkg/abi/linux"
	"gvisor.dev/gvisor/pkg/context"
	"gvisor.dev/gvisor/pkg/errors/linuxerr"
	"gvisor.dev/gvisor/pkg/fspath"
	"gvisor.dev/gvisor/pkg/log"
	"gvisor.dev/gvisor/pkg/sentry/kernel"
	"gvisor.dev/gvisor/pkg/sentry/kernel/auth"
	"gvisor.dev/gvisor/pkg/sentry/vfs"
)

// Multicast interposer (mcshim) integration.
//
// cuda-checkpoint cannot checkpoint a process that holds live multicast
// (NV_MEMORY_MULTICAST_FABRIC, 0x00fd) objects, which both NCCL NVLS and torch
// _symmetric_memory create. The interposer, LD_PRELOADed by
// Loader.setupCudaMulticastShim, tracks every multicast group and CUDA IPC
// import at the libcuda layer. On request it releases them (keeping the VA
// reservations) and later rebuilds them at byte-identical virtual addresses,
// so application pointers and captured CUDA graphs remain valid.
//
// gVisor owns the ordering, which is the part that must be exactly right:
//
//	suspend  after cuda-checkpoint has locked (quiesced) every process, and
//	         before it checkpoints any of them, so the multicast blocker set
//	         is empty when the checkpoint runs;
//	resume   strictly after the post-restore cuda-checkpoint toggle has
//	         finished rebuilding GPU state on EVERY process.
//
// The resume ordering is not merely tidy. `runsc restore` makes tasks runnable
// before the toggle completes, and an application's non-CUDA threads (such as
// the interposer's own control thread) run concurrently with it. Rebuilding
// multicast on a context whose device state has not been restored yet latches
// an unrecoverable fault into that context (sticky CUDA_ERROR_LAUNCH_FAILED,
// 719), which then surfaces much later as a collective failure on a single,
// arbitrary rank. Driving the transition from here -- after restoreCudaProcs
// returns, while tasks are still frozen -- removes that race by construction.
//
// The protocol is existence-based, which keeps it race-free for any number of
// ranks sharing one directory:
//
//	create <dir>/suspend      -> each process suspends, acks <dir>/suspended.<pid>
//	unlink <dir>/suspend      -> each process resumes,  acks <dir>/resumed.<pid>
//
// The directory lives in the container filesystem, so the marker is part of
// the checkpoint image: after a restore it still exists and the interposer
// stays suspended until gVisor removes it.

const (
	// cudaShimDir is the rendezvous directory; the interposer uses the same
	// fixed path. It must be on a filesystem that is part of the checkpoint
	// image (not a host mount): the suspend marker's survival across restore
	// is what keeps the interposer suspended until the sentry orders the
	// rebuild (see the file comment).
	cudaShimDir = "/tmp/mcshim"

	// cudaShimSuspendMarker is created to request the teardown and removed to
	// request the rebuild.
	cudaShimSuspendMarker = "suspend"

	// cudaShimGateMarker is created to bar the application from submitting
	// GPU work. Handling it involves no CUDA calls, so it can be created
	// while the processes are locked by cuda-checkpoint.
	cudaShimGateMarker = "gate"

	// cudaShimAckTimeout bounds how long to wait for every process to
	// acknowledge a transition. Rebuilding multicast is collective (each
	// cuMulticastBindMem blocks until every device has joined the group), so
	// a rank that never acks would otherwise hang the whole operation
	// indefinitely; time out loudly instead.
	cudaShimAckTimeout = 5 * time.Minute

	// cudaShimPollInterval is how often to re-check for acknowledgements.
	cudaShimPollInterval = 100 * time.Millisecond

	// cudaShimRunningPollInterval is how often waitCudaProcsRunning re-polls
	// `cuda-checkpoint --get-state`. Deliberately coarser than
	// cudaShimPollInterval: every poll execs one cuda-checkpoint process per
	// pending CUDA process.
	cudaShimRunningPollInterval = 500 * time.Millisecond

	// cudaShimSuspendedKey records, in the checkpoint, that the interposer was
	// suspended, which tells postRestoreCuda that a rebuild is owed.
	cudaShimSuspendedKey = "cuda-multicast-shim-suspended"
)

// cudaShimPathOp builds a PathOperation for path within tg's mount namespace.
// The returned cleanup must be called by the caller.
func cudaShimPathOp(sctx context.Context, tg *kernel.ThreadGroup, path string) (context.Context, *vfs.PathOperation, func(), bool) {
	leader := tg.Leader()
	if leader == nil {
		return nil, nil, nil, false
	}
	mntns := leader.MountNamespace()
	if mntns == nil || !mntns.TryIncRef() {
		return nil, nil, nil, false
	}
	root := mntns.Root(sctx)
	ctx := vfs.WithRoot(sctx, root)
	cleanup := func() {
		root.DecRef(ctx)
		mntns.DecRef(ctx)
	}
	pop := &vfs.PathOperation{
		Root:  root,
		Start: root,
		Path:  fspath.Parse(path),
	}
	return ctx, pop, cleanup, true
}

// cudaShimCreds returns the credentials to use for interposer marker file
// operations. The markers are sentry-managed control state, so full privilege
// is appropriate and avoids depending on the container's user.
func cudaShimCreds(k *kernel.Kernel) *auth.Credentials {
	return auth.NewRootCredentials(k.RootUserNamespace())
}

// cudaShimSetMarker creates (set) or removes (clear) the suspend marker in
// every distinct mount namespace among cudaProcs. Ranks of a job usually share
// one namespace, in which case this touches the file once.
func cudaShimSetMarker(sctx context.Context, k *kernel.Kernel, cudaProcs []*kernel.ThreadGroup, marker string, set bool) error {
	creds := cudaShimCreds(k)
	path := cudaShimDir + "/" + marker
	seen := make(map[*vfs.MountNamespace]bool)
	var done bool
	for _, tg := range cudaProcs {
		leader := tg.Leader()
		if leader == nil {
			continue
		}
		if mntns := leader.MountNamespace(); mntns != nil {
			if seen[mntns] {
				continue
			}
			seen[mntns] = true
		}
		ctx, pop, cleanup, ok := cudaShimPathOp(sctx, tg, path)
		if !ok {
			continue
		}
		var err error
		if set {
			var fd *vfs.FileDescription
			fd, err = k.VFS().OpenAt(ctx, creds, pop, &vfs.OpenOptions{
				Flags: linux.O_CREAT | linux.O_WRONLY,
				Mode:  0666,
			})
			if err == nil {
				fd.DecRef(ctx)
			}
		} else {
			err = k.VFS().UnlinkAt(ctx, creds, pop)
			if linuxerr.Equals(linuxerr.ENOENT, err) {
				err = nil
			}
		}
		cleanup()
		if err != nil {
			return fmt.Errorf("multicast interposer: %s marker %q: %w", markerVerb(set), path, err)
		}
		done = true
	}
	if !done {
		return fmt.Errorf("multicast interposer: no live process to %s marker %q", markerVerb(set), path)
	}
	return nil
}

func markerVerb(set bool) string {
	if set {
		return "create"
	}
	return "remove"
}

// cudaShimWaitAcks waits until every process in cudaProcs has written its
// acknowledgement file for the given prefix ("suspended" or "resumed").
func cudaShimWaitAcks(sctx context.Context, k *kernel.Kernel, cudaProcs []*kernel.ThreadGroup, prefix string) error {
	creds := cudaShimCreds(k)
	deadline := time.Now().Add(cudaShimAckTimeout)
	pending := make(map[*kernel.ThreadGroup]bool, len(cudaProcs))
	for _, tg := range cudaProcs {
		pending[tg] = true
	}
	for {
		for tg := range pending {
			// The interposer names its ack after getpid(), which is the
			// same value gVisor passes to cuda-checkpoint as --pid.
			path := fmt.Sprintf("%s/%s.%d", cudaShimDir, prefix, tg.ID())
			ctx, pop, cleanup, ok := cudaShimPathOp(sctx, tg, path)
			if !ok {
				// The process exited; it cannot hold multicast state.
				delete(pending, tg)
				continue
			}
			_, err := k.VFS().StatAt(ctx, creds, pop, &vfs.StatOptions{})
			cleanup()
			if err == nil {
				delete(pending, tg)
				continue
			}
			// A failed transition acks with error.<pid> instead. Fail
			// fast: without this, a shim-side failure surfaces as this
			// loop's full timeout. (The sentry removes stale error files
			// before requesting each transition -- see
			// cudaShimClearErrorAcks -- and the interposer does the same
			// when it observes the edge, so an error here is from the
			// transition being waited on.)
			errPath := fmt.Sprintf("%s/error.%d", cudaShimDir, tg.ID())
			ctx, pop, cleanup, ok = cudaShimPathOp(sctx, tg, errPath)
			if !ok {
				delete(pending, tg)
				continue
			}
			_, err = k.VFS().StatAt(ctx, creds, pop, &vfs.StatOptions{})
			cleanup()
			if err == nil {
				return fmt.Errorf("multicast interposer: process %d reported an error during %q (see its MCSHIM_LOG for the cause)", tg.ID(), prefix)
			}
		}
		if len(pending) == 0 {
			return nil
		}
		if time.Now().After(deadline) {
			var missing []kernel.ThreadID
			for tg := range pending {
				missing = append(missing, tg.ID())
			}
			return fmt.Errorf("multicast interposer: %d process(es) did not acknowledge %q within %s (pids %v)",
				len(missing), prefix, cudaShimAckTimeout, missing)
		}
		time.Sleep(cudaShimPollInterval)
	}
}

// waitCudaProcsRunning polls `cuda-checkpoint --get-state` until every process
// in cudaProcs reports the state "running" (processes that exit are dropped
// from the wait, matching cudaShimWaitAcks).
//
// `--action restore`/`--toggle` returning success means the driver accepted the
// restore, not that the process has finished coming back. Issuing CUDA work
// before then is what faults the context, so this is the readiness condition
// for the interposer's rebuild.
func waitCudaProcsRunning(sctx context.Context, k *kernel.Kernel, cudaCheckpointPath string, cudaProcs []*kernel.ThreadGroup) error {
	deadline := time.Now().Add(cudaShimAckTimeout)
	nullFD, cleanup := openCudaCheckpointNullFD(sctx, k)
	defer cleanup()
	proc := &Proc{Kernel: k}
	pending := make(map[*kernel.ThreadGroup]bool, len(cudaProcs))
	for _, tg := range cudaProcs {
		pending[tg] = true
	}
	for {
		for tg := range pending {
			ckptProc, cleanup, err := invokeCudaCheckpoint(sctx, k, proc, cudaCheckpointPath, tg, []string{"--get-state"}, nullFD)
			if err != nil {
				log.Warningf("Failed to get CUDA state for PID %d: %v", tg.ID(), err)
				continue
			}
			if ckptProc.tg == nil {
				// The process exited; nothing to wait for.
				delete(pending, tg)
				continue
			}
			ckptProc.tg.WaitExited()
			status := ckptProc.tg.ExitStatus()
			output := ""
			if ckptProc.out != nil {
				output = ckptProc.out.String()
			}
			cleanup()
			// Only the literal state "running" is readiness; a locked or
			// checkpointed process also exits 0 but is NOT safe to rebuild
			// on. Match by line: stdout and stderr share the collection
			// pipe, so other preloaded libraries' stderr chatter must not
			// mask the state.
			if status == 0 && outputHasLine(output, "running") {
				delete(pending, tg)
			}
		}
		if len(pending) == 0 {
			return nil
		}
		if time.Now().After(deadline) {
			var waiting []kernel.ThreadID
			for tg := range pending {
				waiting = append(waiting, tg.ID())
			}
			return fmt.Errorf("multicast interposer: %d of %d process(es) did not report running within %s (pids %v)",
				len(pending), len(cudaProcs), cudaShimAckTimeout, waiting)
		}
		time.Sleep(cudaShimRunningPollInterval)
	}
}

// outputHasLine reports whether any whitespace-trimmed line of out equals
// want.
func outputHasLine(out, want string) bool {
	for _, line := range strings.Split(out, "\n") {
		if strings.TrimSpace(line) == want {
			return true
		}
	}
	return false
}

// cudaShimClearErrorAcks removes any stale error.<pid> ack files for
// cudaProcs. Called before requesting a suspend or resume: cudaShimWaitAcks
// fast-fails on error acks, and the interposer only clears its own error file
// when it observes the next marker edge, so an error left over from an
// earlier, timed-out transition could otherwise fail the new one instantly.
func cudaShimClearErrorAcks(sctx context.Context, k *kernel.Kernel, cudaProcs []*kernel.ThreadGroup) {
	creds := cudaShimCreds(k)
	for _, tg := range cudaProcs {
		path := fmt.Sprintf("%s/error.%d", cudaShimDir, tg.ID())
		ctx, pop, cleanup, ok := cudaShimPathOp(sctx, tg, path)
		if !ok {
			continue
		}
		if err := k.VFS().UnlinkAt(ctx, creds, pop); err != nil && !linuxerr.Equals(linuxerr.ENOENT, err) {
			log.Warningf("Failed to clear stale interposer error ack %q: %v", path, err)
		}
		cleanup()
	}
}

// cudaShimProcsWith returns the subset of cudaProcs that have written the file
// "<prefix>.<pid>" in the rendezvous directory.
func cudaShimProcsWith(sctx context.Context, k *kernel.Kernel, cudaProcs []*kernel.ThreadGroup, prefix string) []*kernel.ThreadGroup {
	creds := cudaShimCreds(k)
	var out []*kernel.ThreadGroup
	for _, tg := range cudaProcs {
		path := fmt.Sprintf("%s/%s.%d", cudaShimDir, prefix, tg.ID())
		ctx, pop, cleanup, ok := cudaShimPathOp(sctx, tg, path)
		if !ok {
			continue
		}
		_, err := k.VFS().StatAt(ctx, creds, pop, &vfs.StatOptions{})
		cleanup()
		if err == nil {
			out = append(out, tg)
		}
	}
	return out
}

// cudaShimManagedProcs returns the processes the interposer is actually
// managing, i.e. that announced a control thread.
//
// cudaProcs is selected by looking for open NVIDIA device FDs, which is
// deliberately broad. Processes such as a vLLM API server or engine-core hold
// those FDs without ever resolving a multicast entry point, so the interposer
// never starts a control thread in them and they can never acknowledge a
// transition. Waiting on them would hang every checkpoint.
//
// A process that does hold multicast state necessarily resolved a tracked entry
// point first, so it is always in this set.
func cudaShimManagedProcs(sctx context.Context, k *kernel.Kernel, cudaProcs []*kernel.ThreadGroup) []*kernel.ThreadGroup {
	return cudaShimProcsWith(sctx, k, cudaProcs, "present")
}

// errCudaShimTornDown reports that the interposer's teardown failed. Other
// processes may already have released state that a rebuild needs from the one
// that failed, so the application is left blocked rather than unwound.
var errCudaShimTornDown = errors.New("multicast interposer teardown failed; the application is left blocked and must be restarted")

// unwindCudaMulticastShim returns the application to a runnable state after a
// checkpoint failed before the interposer's teardown, or after it completed on
// every process: rebuild whatever was torn down, then release the gate.
//
// The gate is released unconditionally rather than as a side effect of a
// successful rebuild: it is armed before the teardown, so a failure between
// the two leaves the application barred from the GPU with nothing to rebuild.
// Only processes that acknowledged the teardown will acknowledge a rebuild.
func unwindCudaMulticastShim(sctx context.Context, k *kernel.Kernel, cudaProcs []*kernel.ThreadGroup) {
	// Drop any recorded rebuild state: this instance is handling it.
	k.PopCheckpointState(cudaShimSuspendedKey)
	tornDown := cudaShimProcsWith(sctx, k, cudaProcs, "suspended")
	if err := cudaShimSetMarker(sctx, k, cudaProcs, cudaShimSuspendMarker, false /* set */); err != nil {
		log.Warningf("Multicast interposer unwind: %v", err)
	}
	if len(tornDown) != 0 {
		if err := cudaShimWaitAcks(sctx, k, tornDown, "resumed"); err != nil {
			log.Warningf("Multicast interposer unwind: %v", err)
		}
	}
	if err := cudaShimSetMarker(sctx, k, cudaProcs, cudaShimGateMarker, false /* set */); err != nil {
		log.Warningf("Multicast interposer unwind: %v", err)
	}
	log.Infof("Multicast interposer unwound (%d process(es) had been torn down)", len(tornDown))
}

// armCudaMulticastShimGate bars the application from submitting GPU work, and
// waits until every process confirms it. It makes no CUDA calls in the target
// processes; the interposer only flips a flag. A process whose state the
// interposer could not fully track refuses here, failing the checkpoint before
// anything is torn down.
func armCudaMulticastShimGate(sctx context.Context, k *kernel.Kernel, cudaProcs []*kernel.ThreadGroup) error {
	start := time.Now()
	// Clear stale error acks: the gate wait fails fast on them, and the
	// interposer clears its own only on suspend/resume edges.
	cudaShimClearErrorAcks(sctx, k, cudaProcs)
	if err := cudaShimSetMarker(sctx, k, cudaProcs, cudaShimGateMarker, true /* set */); err != nil {
		return err
	}
	managed := cudaShimManagedProcs(sctx, k, cudaProcs)
	if err := cudaShimWaitAcks(sctx, k, managed, "gated"); err != nil {
		if rerr := cudaShimSetMarker(sctx, k, cudaProcs, cudaShimGateMarker, false /* set */); rerr != nil {
			log.Warningf("Failed to clear multicast interposer gate after arm failure: %v", rerr)
		}
		return err
	}
	log.Infof("Multicast interposer gated %d of %d CUDA process(es) in %s", len(managed), len(cudaProcs), time.Since(start))
	return nil
}

// suspendCudaMulticastShim asks the interposer to release multicast objects and
// CUDA IPC imports on every process, and waits for all of them to finish. If
// any fails, it returns errCudaShimTornDown and leaves the markers in place.
//
// Precondition: the application is gated (the processes were just unlocked by
// checkpointCudaProcs so the interposer can issue libcuda calls, and the gate
// is what keeps the application off the GPU meanwhile).
func suspendCudaMulticastShim(sctx context.Context, k *kernel.Kernel, cudaProcs []*kernel.ThreadGroup) error {
	start := time.Now()
	cudaShimClearErrorAcks(sctx, k, cudaProcs)
	if err := cudaShimSetMarker(sctx, k, cudaProcs, cudaShimSuspendMarker, true /* set */); err != nil {
		return err
	}
	managed := cudaShimManagedProcs(sctx, k, cudaProcs)
	if err := cudaShimWaitAcks(sctx, k, managed, "suspended"); err != nil {
		return fmt.Errorf("%w: %v", errCudaShimTornDown, err)
	}
	k.AddStateToCheckpoint(cudaShimSuspendedKey, true)
	log.Infof("Multicast interposer suspended on %d of %d CUDA process(es) in %s", len(managed), len(cudaProcs), time.Since(start))
	return nil
}

// resumeCudaMulticastShim asks the interposer to rebuild multicast objects and
// CUDA IPC imports, and waits for every process to finish.
//
// Precondition: the post-restore cuda-checkpoint toggle has completed on EVERY
// process. Resuming earlier rebuilds on a context whose device state is not
// restored yet and permanently faults it; see the file comment.
func resumeCudaMulticastShim(sctx context.Context, k *kernel.Kernel, cudaCheckpointPath string, cudaProcs []*kernel.ThreadGroup) error {
	if k.PopCheckpointState(cudaShimSuspendedKey) == nil {
		log.Infof("Multicast interposer: no suspend recorded in the checkpoint; nothing to rebuild")
		return nil
	}
	start := time.Now()
	// The restore toggle returning is necessary but not sufficient: wait until
	// every process actually reports "running" before rebuilding on top of it.
	if err := waitCudaProcsRunning(sctx, k, cudaCheckpointPath, cudaProcs); err != nil {
		return err
	}
	cudaShimClearErrorAcks(sctx, k, cudaProcs)
	if err := cudaShimSetMarker(sctx, k, cudaProcs, cudaShimSuspendMarker, false /* set */); err != nil {
		return err
	}
	managed := cudaShimManagedProcs(sctx, k, cudaProcs)
	if err := cudaShimWaitAcks(sctx, k, managed, "resumed"); err != nil {
		return err
	}
	// The interposer releases the application itself once the rebuild
	// succeeds; clear the marker too so a later checkpoint starts clean.
	if err := cudaShimSetMarker(sctx, k, cudaProcs, cudaShimGateMarker, false /* set */); err != nil {
		log.Warningf("Failed to clear multicast interposer gate marker: %v", err)
	}
	log.Infof("Multicast interposer resumed on %d of %d CUDA process(es) in %s", len(managed), len(cudaProcs), time.Since(start))
	return nil
}
