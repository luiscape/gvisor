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
	"gvisor.dev/gvisor/pkg/sentry/devices/nvproxy"
	"gvisor.dev/gvisor/pkg/sentry/kernel"
	"gvisor.dev/gvisor/pkg/sentry/kernel/auth"
	"gvisor.dev/gvisor/pkg/sentry/vfs"
	"gvisor.dev/gvisor/pkg/usermem"
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
// arbitrary rank. Tasks are already running when this code runs, but the
// interposer rebuilds only when the sentry removes the suspend marker, which it
// does only after restoreCudaProcs has returned for every process. That removes
// the race by construction.
//
// The protocol is existence-based, which keeps it race-free for any number of
// ranks sharing one directory:
//
//	create <dir>/gate         -> each process blocks GPU work, acks <dir>/gated.<pid>
//	create <dir>/suspend      -> each process suspends, acks <dir>/suspended.<pid>
//	unlink <dir>/suspend      -> each process resumes,  acks <dir>/resumed.<pid>
//	unlink <dir>/gate         -> each process releases the application
//
// The directory lives in the container filesystem, so the markers are part of
// the checkpoint image: after a restore they still exist and the interposer
// stays suspended until gVisor removes them. The gate is removed only once
// every process has resumed, since a rank released earlier could reach a
// multicast group that a peer is still binding.

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
			// loop's full timeout. (The sentry removes stale acks before
			// requesting each transition -- see cudaShimClearAcks -- so an
			// error here is from the transition being waited on.)
			errPath := fmt.Sprintf("%s/error.%d", cudaShimDir, tg.ID())
			ctx, pop, cleanup, ok = cudaShimPathOp(sctx, tg, errPath)
			if !ok {
				delete(pending, tg)
				continue
			}
			_, err = k.VFS().StatAt(ctx, creds, pop, &vfs.StatOptions{})
			cleanup()
			if err == nil {
				return fmt.Errorf("multicast interposer: process %d reported an error during %q: %s", tg.ID(), prefix, cudaShimErrorDetail(sctx, k, tg))
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

// cudaShimErrorDetail describes the error tg reported: the reason in its
// error.<pid> ack, and its last lines in the interposer's default log.
func cudaShimErrorDetail(sctx context.Context, k *kernel.Kernel, tg *kernel.ThreadGroup) string {
	reason := strings.TrimSpace(cudaShimReadTail(sctx, k, tg, fmt.Sprintf("%s/error.%d", cudaShimDir, tg.ID()), 512))
	detail := fmt.Sprintf("%q", reason)
	var lines []string
	tag := fmt.Sprintf(" pid=%d] ", tg.ID())
	for _, l := range strings.Split(cudaShimReadTail(sctx, k, tg, cudaShimDir+"/mcshim.log", 64<<10), "\n") {
		if strings.Contains(l, tag) {
			lines = append(lines, l)
		}
	}
	if len(lines) > 5 {
		lines = lines[len(lines)-5:]
	}
	if len(lines) > 0 {
		detail += fmt.Sprintf("; last log lines: %q", strings.Join(lines, " | "))
	}
	return detail
}

// cudaShimReadTail returns at most max bytes from the end of path, resolved in
// tg's mount namespace, or "" if it cannot be read.
func cudaShimReadTail(sctx context.Context, k *kernel.Kernel, tg *kernel.ThreadGroup, path string, max int64) string {
	ctx, pop, cleanup, ok := cudaShimPathOp(sctx, tg, path)
	if !ok {
		return ""
	}
	defer cleanup()
	fd, err := k.VFS().OpenAt(ctx, cudaShimCreds(k), pop, &vfs.OpenOptions{Flags: linux.O_RDONLY})
	if err != nil {
		return ""
	}
	defer fd.DecRef(ctx)
	stat, err := fd.Stat(ctx, vfs.StatOptions{Mask: linux.STATX_SIZE})
	if err != nil {
		return ""
	}
	off := int64(stat.Size) - max
	if off < 0 {
		off = 0
	}
	buf := make([]byte, int64(stat.Size)-off)
	n, _ := fd.PRead(ctx, usermem.BytesIOSequence(buf), off, vfs.ReadOptions{})
	return string(buf[:n])
}

// cudaShimClearAcks removes the error.<pid> and <prefix>.<pid> acks of
// cudaProcs before a transition is requested, so that an ack left over from an
// earlier attempt can neither fail nor satisfy cudaShimWaitAcks.
func cudaShimClearAcks(sctx context.Context, k *kernel.Kernel, cudaProcs []*kernel.ThreadGroup, prefix string) {
	creds := cudaShimCreds(k)
	for _, tg := range cudaProcs {
		for _, p := range []string{"error", prefix} {
			path := fmt.Sprintf("%s/%s.%d", cudaShimDir, p, tg.ID())
			ctx, pop, cleanup, ok := cudaShimPathOp(sctx, tg, path)
			if !ok {
				continue
			}
			if err := k.VFS().UnlinkAt(ctx, creds, pop); err != nil && !linuxerr.Equals(linuxerr.ENOENT, err) {
				log.Warningf("Failed to clear stale interposer ack %q: %v", path, err)
			}
			cleanup()
		}
	}
}

// cudaShimMarkerExists returns whether marker exists for any of cudaProcs.
func cudaShimMarkerExists(sctx context.Context, k *kernel.Kernel, cudaProcs []*kernel.ThreadGroup, marker string) bool {
	creds := cudaShimCreds(k)
	for _, tg := range cudaProcs {
		ctx, pop, cleanup, ok := cudaShimPathOp(sctx, tg, cudaShimDir+"/"+marker)
		if !ok {
			continue
		}
		_, err := k.VFS().StatAt(ctx, creds, pop, &vfs.StatOptions{})
		cleanup()
		if err == nil {
			return true
		}
	}
	return false
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
// deliberately broad. Processes such as a vLLM API server hold those FDs (e.g.
// through NVML) without initializing CUDA, so the interposer never starts a
// control thread in them and they can never acknowledge a transition. Waiting
// on them would hang every checkpoint.
//
// The control thread starts at cuInit and at the first tracked create or
// import, so a process holding state the interposer saw is always in this set.
// One holding state it did not see is not, and the blocker check refuses it.
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
	cudaShimClearAcks(sctx, k, tornDown, "resumed")
	if err := cudaShimSetMarker(sctx, k, cudaProcs, cudaShimSuspendMarker, false /* set */); err != nil {
		log.Warningf("Multicast interposer unwind: %v", err)
	}
	if len(tornDown) != 0 {
		if err := cudaShimWaitAcks(sctx, k, tornDown, "resumed"); err != nil {
			log.Warningf("Multicast interposer unwind: %v", err)
		}
	}
	if err := cudaShimSetMarker(sctx, k, cudaProcs, cudaShimGateMarker, false /* set */); err != nil {
		log.Warningf("Multicast interposer unwind: the application stays blocked: %v", err)
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
	// A failed teardown leaves its markers in place and the application
	// blocked; a new attempt would wait for acks that never come.
	if cudaShimMarkerExists(sctx, k, cudaProcs, cudaShimGateMarker) || cudaShimMarkerExists(sctx, k, cudaProcs, cudaShimSuspendMarker) {
		return fmt.Errorf("%w (an earlier attempt's markers remain)", errCudaShimTornDown)
	}
	cudaShimClearAcks(sctx, k, cudaProcs, "gated")
	if err := cudaShimSetMarker(sctx, k, cudaProcs, cudaShimGateMarker, true /* set */); err != nil {
		return err
	}
	managed := cudaShimManagedProcs(sctx, k, cudaProcs)
	err := cudaShimWaitAcks(sctx, k, managed, "gated")
	if err == nil {
		// After a restore the interposer rebuilds each import from its
		// exporter's re-export. Every process is gated, so none can free an
		// exported object while this checks.
		procs := make(map[kernel.ThreadID]bool, len(managed))
		for _, tg := range managed {
			procs[tg.ID()] = true
		}
		if imports := nvproxy.UnresolvableImports(k.VFS(), procs); imports != "" {
			err = fmt.Errorf("multicast interposer: imports that could not be rebuilt after a restore: %s", imports)
		}
	}
	if err != nil {
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
	cudaShimClearAcks(sctx, k, cudaProcs, "suspended")
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
// Precondition: restoreCudaProcs has returned, so every process is running
// again. Resuming earlier rebuilds on a context whose device state is not
// restored yet and permanently faults it; see the file comment.
func resumeCudaMulticastShim(sctx context.Context, k *kernel.Kernel, cudaProcs []*kernel.ThreadGroup) error {
	if k.PopCheckpointState(cudaShimSuspendedKey) == nil {
		log.Infof("Multicast interposer: no suspend recorded in the checkpoint; nothing to rebuild")
		return nil
	}
	start := time.Now()
	cudaShimClearAcks(sctx, k, cudaProcs, "resumed")
	if err := cudaShimSetMarker(sctx, k, cudaProcs, cudaShimSuspendMarker, false /* set */); err != nil {
		return err
	}
	managed := cudaShimManagedProcs(sctx, k, cudaProcs)
	if err := cudaShimWaitAcks(sctx, k, managed, "resumed"); err != nil {
		return err
	}
	// Every process has resumed: release the application.
	if err := cudaShimSetMarker(sctx, k, cudaProcs, cudaShimGateMarker, false /* set */); err != nil {
		return fmt.Errorf("multicast interposer: releasing the application: %w", err)
	}
	log.Infof("Multicast interposer resumed on %d of %d CUDA process(es) in %s", len(managed), len(cudaProcs), time.Since(start))
	return nil
}
