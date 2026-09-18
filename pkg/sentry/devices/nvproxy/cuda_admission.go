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

package nvproxy

import (
	"gvisor.dev/gvisor/pkg/errors/linuxerr"
	"gvisor.dev/gvisor/pkg/log"
	"gvisor.dev/gvisor/pkg/sentry/kernel"
	"gvisor.dev/gvisor/pkg/sentry/vfs"
	"gvisor.dev/gvisor/pkg/sync"
)

// CUDA admission gate.
//
// A CUDA checkpoint sequence (pkg/sentry/control/state_cuda.go) collects the
// set of CUDA processes, then locks, tears down, and checkpoints them, then
// saves the sandbox. A process that initializes CUDA AFTER the set was
// collected -- engines spawn helpers that take seconds to reach cuInit; the
// window is tens of seconds -- ends up with GPU state that cuda-checkpoint
// never serialized but that nvproxy's object graph would replay blind on
// restore, where it fails (a CUDA context cannot be recreated by replaying
// its RM allocations). This gate removes the race.
//
// While the gate is closed, a process's FIRST acquisition of GPU state -- the
// allocation of an RM root client, which every other RM object descends from
// -- blocks until the gate opens. A blocked process has no GPU state, so it
// is saved as an ordinary process sleeping in a syscall; after a restore it
// retries the allocation against the restored devices and initializes CUDA
// normally, as if it had started a moment later. Processes the sequence must
// let through (the cuda-checkpoint invocations themselves, and the processes
// being checkpointed, which re-allocate their clients on a failure-path
// restore) are exempted by the caller's predicate.
//
// The gate must be open before any interposer rebuild (the rebuild execs a
// fresh helper process that initializes CUDA), and is deliberately not saved:
// a checkpointed image is restored into a kernel where no sequence is in
// progress.
type cudaAdmission struct {
	mu sync.Mutex `state:"nosave"`
	// closed is non-nil while the gate is closed; it is closed (the channel)
	// to release every waiter when the gate opens.
	closed chan struct{} `state:"nosave"`
	exempt func(*kernel.ThreadGroup) bool `state:"nosave"`
}

// close closes the gate. exempt names the thread groups allowed through; nil
// exempts none. Closing an already-closed gate replaces the predicate and
// keeps existing waiters waiting.
func (a *cudaAdmission) close(exempt func(*kernel.ThreadGroup) bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed == nil {
		a.closed = make(chan struct{})
	}
	a.exempt = exempt
}

// open opens the gate, releasing every waiter. No-op if already open.
func (a *cudaAdmission) open() {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed != nil {
		close(a.closed)
		a.closed = nil
		a.exempt = nil
	}
}

// wait returns (nil, true) if tg may proceed now, or (ch, false) with the
// channel that is closed when the gate next opens.
func (a *cudaAdmission) wait(tg *kernel.ThreadGroup) (<-chan struct{}, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed == nil || (a.exempt != nil && a.exempt(tg)) {
		return nil, true
	}
	return a.closed, false
}

// CloseCudaAdmission closes the admission gate of the nvproxy registered in
// vfsObj (no-op if nvproxy is not registered). See cudaAdmission.close.
func CloseCudaAdmission(vfsObj *vfs.VirtualFilesystem, exempt func(*kernel.ThreadGroup) bool) {
	if nvp := nvproxyFromVFS(vfsObj); nvp != nil {
		nvp.admission.close(exempt)
	}
}

// OpenCudaAdmission opens the admission gate, releasing every process blocked
// in it. No-op if the gate is open or nvproxy is not registered.
func OpenCudaAdmission(vfsObj *vfs.VirtualFilesystem) {
	if nvp := nvproxyFromVFS(vfsObj); nvp != nil {
		nvp.admission.open()
	}
}

// awaitCudaAdmission blocks t while the admission gate is closed for its
// thread group. It returns nil once admitted, or ERESTARTNOINTR if the block
// was interrupted (by a save, or a signal), so that the ioctl is transparently
// retried -- after the restore, against the restored devices.
func (nvp *nvproxy) awaitCudaAdmission(t *kernel.Task) error {
	logged := false
	for {
		ch, admitted := nvp.admission.wait(t.ThreadGroup())
		if admitted {
			return nil
		}
		if !logged {
			logged = true
			log.Infof("nvproxy: task %d is initializing CUDA during a checkpoint sequence; holding its RM root client allocation until the sequence ends", t.ThreadGroup().ID())
		}
		if err := t.Block(ch); err != nil {
			return linuxerr.ERESTARTNOINTR
		}
	}
}
