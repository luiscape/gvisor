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
	"strconv"
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

	// cudaShimProtocol is the version of the protocol with the interposer,
	// which each interposer announces in its present.<pid> marker ("mcshim
	// protocol N"); runsc and the interposer ship separately. It must equal
	// MCSHIM_PROTOCOL in tools/mcshim/mcshim.c. Bump both with any change the
	// other side would misread: marker names or meanings, acks, the order of
	// transitions, the published fd format, or the fdinfo line the interposer
	// parses.
	cudaShimProtocol = 1

	// cudaShimProtocolPrefix starts the body of a present.<pid> marker.
	cudaShimProtocolPrefix = "mcshim protocol "
)

// cudaShimCreds returns the credentials to use for interposer marker file
// operations. The markers are sentry-managed control state, so full privilege
// is appropriate and avoids depending on the container's user; cudaShimFS
// keeps that privilege inside the control directory.
func cudaShimCreds(k *kernel.Kernel) *auth.Credentials {
	return auth.NewRootCredentials(k.RootUserNamespace())
}

// cudaShimFS is the control directory as one mount namespace sees it.
//
// The directory is world-writable, in world-writable /tmp, and the sentry acts
// in it with root credentials, so any container user can plant things there:
// the directory itself as a symlink (to /etc, or a writable host mount), or a
// symlink, hard link or FIFO in place of a marker or ack. cudaShimFS resolves
// the directory once, without following a symlink at its last component, and
// refuses it unless it is a directory; reaches every file as a single
// component relative to it, without following a final symlink; never opens
// anything but a regular file, and opens non-blocking, so a FIFO cannot stall
// the checkpoint. What a container user can still do is forge or remove
// markers and acks, which only fails its own container's checkpoint.
type cudaShimFS struct {
	ctx   context.Context
	vfs   *vfs.VirtualFilesystem
	creds *auth.Credentials
	root  vfs.VirtualDentry
	dir   vfs.VirtualDentry
	done  func()
}

// newCudaShimFS returns the control directory under root, mntns's root. The
// caller must call release.
func newCudaShimFS(ctx context.Context, vfsObj *vfs.VirtualFilesystem, creds *auth.Credentials, mntns *vfs.MountNamespace, root vfs.VirtualDentry) (*cudaShimFS, error) {
	// Unlinking checks for mount points in the context's mount namespace.
	ctx = vfs.WithMountNamespace(vfs.WithRoot(ctx, root), mntns)
	dir, err := vfsObj.GetDentryAt(ctx, creds, &vfs.PathOperation{
		Root:  root,
		Start: root,
		Path:  fspath.Parse(cudaShimDir),
	}, &vfs.GetDentryOptions{CheckSearchable: true})
	if err != nil {
		return nil, fmt.Errorf("multicast interposer: control directory %s: %w", cudaShimDir, err)
	}
	return &cudaShimFS{ctx: ctx, vfs: vfsObj, creds: creds, root: root, dir: dir}, nil
}

// openCudaShimFS returns the control directory in tg's mount namespace, or
// ok false if tg has exited. The caller must call release.
func openCudaShimFS(sctx context.Context, k *kernel.Kernel, tg *kernel.ThreadGroup) (fs *cudaShimFS, ok bool, err error) {
	leader := tg.Leader()
	if leader == nil {
		return nil, false, nil
	}
	mntns := leader.MountNamespace()
	if mntns == nil || !mntns.TryIncRef() {
		return nil, false, nil
	}
	root := mntns.Root(sctx)
	fs, err = newCudaShimFS(sctx, k.VFS(), cudaShimCreds(k), mntns, root)
	if err != nil {
		root.DecRef(sctx)
		mntns.DecRef(sctx)
		return nil, true, err
	}
	fs.done = func() {
		root.DecRef(sctx)
		mntns.DecRef(sctx)
	}
	return fs, true, nil
}

func (fs *cudaShimFS) release() {
	fs.dir.DecRef(fs.ctx)
	if fs.done != nil {
		fs.done()
	}
}

// pop returns a PathOperation for name, a single path component, in the
// directory. It does not follow a final symlink.
func (fs *cudaShimFS) pop(name string) *vfs.PathOperation {
	return &vfs.PathOperation{Root: fs.root, Start: fs.dir, Path: fspath.Parse(name)}
}

// open opens name, which must be a regular file with a single link, with
// flags (plus O_NOFOLLOW and O_NONBLOCK).
func (fs *cudaShimFS) open(name string, flags uint32, mode linux.FileMode) (*vfs.FileDescription, error) {
	fd, err := fs.vfs.OpenAt(fs.ctx, fs.creds, fs.pop(name), &vfs.OpenOptions{
		Flags: flags | linux.O_NOFOLLOW | linux.O_NONBLOCK,
		Mode:  mode,
	})
	if err != nil {
		return nil, err
	}
	stat, err := fd.Stat(fs.ctx, vfs.StatOptions{Mask: linux.STATX_TYPE | linux.STATX_NLINK})
	switch {
	case err != nil:
	case stat.Mode&linux.S_IFMT != linux.S_IFREG:
		err = fmt.Errorf("not a regular file (mode %#o)", stat.Mode)
	case stat.Nlink != 1:
		// A hard link to a file elsewhere (whose contents the sentry would
		// otherwise quote in its error).
		err = fmt.Errorf("has %d links", stat.Nlink)
	}
	if err != nil {
		fd.DecRef(fs.ctx)
		return nil, err
	}
	return fd, nil
}

// create creates name, or leaves it if it is already a regular file.
func (fs *cudaShimFS) create(name string) error {
	fd, err := fs.open(name, linux.O_CREAT|linux.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	fd.DecRef(fs.ctx)
	return nil
}

// remove removes name; it is not an error if it does not exist.
func (fs *cudaShimFS) remove(name string) error {
	if err := fs.vfs.UnlinkAt(fs.ctx, fs.creds, fs.pop(name)); err != nil && !linuxerr.Equals(linuxerr.ENOENT, err) {
		return err
	}
	return nil
}

// exists returns whether name exists (as anything).
func (fs *cudaShimFS) exists(name string) bool {
	_, err := fs.vfs.StatAt(fs.ctx, fs.creds, fs.pop(name), &vfs.StatOptions{})
	return err == nil
}

// readTail returns at most limit bytes from the end of name, or "" if it is
// not a regular file it can read.
func (fs *cudaShimFS) readTail(name string, limit int64) string {
	fd, err := fs.open(name, linux.O_RDONLY, 0)
	if err != nil {
		return ""
	}
	defer fd.DecRef(fs.ctx)
	stat, err := fd.Stat(fs.ctx, vfs.StatOptions{Mask: linux.STATX_SIZE})
	if err != nil {
		return ""
	}
	off := max(int64(stat.Size)-limit, 0)
	buf := make([]byte, int64(stat.Size)-off)
	n, _ := fd.PRead(fs.ctx, usermem.BytesIOSequence(buf), off, vfs.ReadOptions{})
	return string(buf[:n])
}

// withCudaShimFS calls f with tg's control directory. It returns false if tg
// has exited, and an error if the directory is not usable.
func withCudaShimFS(sctx context.Context, k *kernel.Kernel, tg *kernel.ThreadGroup, f func(fs *cudaShimFS)) (bool, error) {
	fs, ok, err := openCudaShimFS(sctx, k, tg)
	if !ok || err != nil {
		return ok, err
	}
	defer fs.release()
	f(fs)
	return true, nil
}

// cudaShimSetMarker creates (set) or removes (clear) the suspend marker in
// every distinct mount namespace among cudaProcs. Ranks of a job usually share
// one namespace, in which case this touches the file once.
func cudaShimSetMarker(sctx context.Context, k *kernel.Kernel, cudaProcs []*kernel.ThreadGroup, marker string, set bool) error {
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
		var err error
		ok, derr := withCudaShimFS(sctx, k, tg, func(fs *cudaShimFS) {
			if set {
				err = fs.create(marker)
			} else {
				err = fs.remove(marker)
			}
		})
		if derr != nil {
			err = derr
		}
		if !ok {
			continue
		}
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
	deadline := time.Now().Add(cudaShimAckTimeout)
	pending := make(map[*kernel.ThreadGroup]bool, len(cudaProcs))
	for _, tg := range cudaProcs {
		pending[tg] = true
	}
	for {
		for tg := range pending {
			// The interposer names its ack after getpid(), which is the
			// same value gVisor passes to cuda-checkpoint as --pid.
			var acked, failed bool
			var detail string
			ok, err := withCudaShimFS(sctx, k, tg, func(fs *cudaShimFS) {
				acked = fs.exists(fmt.Sprintf("%s.%d", prefix, tg.ID()))
				// A failed transition acks with error.<pid> instead. Fail
				// fast: without this, a shim-side failure surfaces as this
				// loop's full timeout. (The sentry removes stale acks
				// before requesting each transition -- see
				// cudaShimClearAcks -- so an error here is from the
				// transition being waited on.)
				if !acked && fs.exists(fmt.Sprintf("error.%d", tg.ID())) {
					failed = true
					detail = cudaShimErrorDetail(fs, tg)
				}
			})
			switch {
			case !ok:
				// The process exited; it cannot hold multicast state.
				delete(pending, tg)
			case err != nil:
				return err
			case acked:
				delete(pending, tg)
			case failed:
				return fmt.Errorf("multicast interposer: process %d reported an error during %q: %s", tg.ID(), prefix, detail)
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
func cudaShimErrorDetail(fs *cudaShimFS, tg *kernel.ThreadGroup) string {
	reason := strings.TrimSpace(fs.readTail(fmt.Sprintf("error.%d", tg.ID()), 512))
	detail := fmt.Sprintf("%q", reason)
	var lines []string
	tag := fmt.Sprintf(" pid=%d] ", tg.ID())
	for _, l := range strings.Split(fs.readTail("mcshim.log", 64<<10), "\n") {
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

// cudaShimClearAcks removes the error.<pid> and <prefix>.<pid> acks of
// cudaProcs before a transition is requested, so that an ack left over from an
// earlier attempt can neither fail nor satisfy cudaShimWaitAcks.
func cudaShimClearAcks(sctx context.Context, k *kernel.Kernel, cudaProcs []*kernel.ThreadGroup, prefix string) {
	for _, tg := range cudaProcs {
		_, err := withCudaShimFS(sctx, k, tg, func(fs *cudaShimFS) {
			for _, p := range []string{"error", prefix} {
				name := fmt.Sprintf("%s.%d", p, tg.ID())
				if err := fs.remove(name); err != nil {
					log.Warningf("Failed to clear stale interposer ack %s/%s: %v", cudaShimDir, name, err)
				}
			}
		})
		if err != nil {
			log.Warningf("Failed to clear stale interposer acks: %v", err)
		}
	}
}

// cudaShimMarkerExists returns whether marker exists for any of cudaProcs.
func cudaShimMarkerExists(sctx context.Context, k *kernel.Kernel, cudaProcs []*kernel.ThreadGroup, marker string) bool {
	for _, tg := range cudaProcs {
		var exists bool
		withCudaShimFS(sctx, k, tg, func(fs *cudaShimFS) { exists = fs.exists(marker) })
		if exists {
			return true
		}
	}
	return false
}

// cudaShimProcsWith returns the subset of cudaProcs that have written the file
// "<prefix>.<pid>" in the rendezvous directory.
func cudaShimProcsWith(sctx context.Context, k *kernel.Kernel, cudaProcs []*kernel.ThreadGroup, prefix string) []*kernel.ThreadGroup {
	var out []*kernel.ThreadGroup
	for _, tg := range cudaProcs {
		var exists bool
		withCudaShimFS(sctx, k, tg, func(fs *cudaShimFS) { exists = fs.exists(fmt.Sprintf("%s.%d", prefix, tg.ID())) })
		if exists {
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

// protocolMismatch returns why the interposer in the process with the given
// pid cannot be driven by this sentry, from its present.<pid> marker, or ""
// if it speaks cudaShimProtocol. The marker's contents come from the
// container; a forged one can only refuse that container's checkpoint, or let
// a mismatched interposer through, as running a mismatched one could anyway.
func (fs *cudaShimFS) protocolMismatch(pid kernel.ThreadID) string {
	body := strings.TrimSpace(fs.readTail(fmt.Sprintf("present.%d", pid), 64))
	v, ok := strings.CutPrefix(body, cudaShimProtocolPrefix)
	if !ok {
		// Interposers built before the protocol was versioned write "ok".
		return fmt.Sprintf("PID %d announces no protocol version (%q)", pid, body)
	}
	if n, err := strconv.Atoi(v); err != nil || n != cudaShimProtocol {
		return fmt.Sprintf("PID %d speaks protocol %q", pid, v)
	}
	return ""
}

// cudaShimCheckProtocol returns an error naming every process in managed whose
// interposer does not speak cudaShimProtocol.
func cudaShimCheckProtocol(sctx context.Context, k *kernel.Kernel, managed []*kernel.ThreadGroup) error {
	var bad []string
	for _, tg := range managed {
		var why string
		ok, err := withCudaShimFS(sctx, k, tg, func(fs *cudaShimFS) { why = fs.protocolMismatch(tg.ID()) })
		if err != nil {
			return err
		}
		if ok && why != "" {
			bad = append(bad, why)
		}
	}
	if len(bad) > 0 {
		return fmt.Errorf("multicast interposer: this runsc speaks protocol %d, but %s; use the interposer built from the same gVisor release as runsc", cudaShimProtocol, strings.Join(bad, "; "))
	}
	return nil
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
	// The restoring runsc may not be the one that checkpointed. Fail the
	// restore rather than drive a rebuild this sentry may get wrong.
	managed := cudaShimManagedProcs(sctx, k, cudaProcs)
	if err := cudaShimCheckProtocol(sctx, k, managed); err != nil {
		return err
	}
	cudaShimClearAcks(sctx, k, cudaProcs, "resumed")
	if err := cudaShimSetMarker(sctx, k, cudaProcs, cudaShimSuspendMarker, false /* set */); err != nil {
		return err
	}
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
