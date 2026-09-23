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

// cudaAdmission gates a process's first RM root client allocation, i.e. CUDA
// initialization, while a CUDA checkpoint is in progress. A process that
// initializes CUDA after the checkpoint collected its set of CUDA processes
// would hold GPU state that cuda-checkpoint never saved; holding it here
// instead makes it initialize after the save, or after the restore.
//
// Not saved: a restored kernel has no checkpoint in progress.
type cudaAdmission struct {
	mu sync.Mutex `state:"nosave"`
	// closed is non-nil while the gate is closed, and is closed to release
	// all waiters when the gate opens.
	// +checklocks:mu
	closed chan struct{} `state:"nosave"`
	// +checklocks:mu
	exempt func(*kernel.ThreadGroup) bool `state:"nosave"`
}

func (a *cudaAdmission) close(exempt func(*kernel.ThreadGroup) bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed == nil {
		a.closed = make(chan struct{})
	}
	a.exempt = exempt
}

func (a *cudaAdmission) open() {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed != nil {
		close(a.closed)
		a.closed = nil
		a.exempt = nil
	}
}

// wait returns nil if tg may proceed, or the channel closed when the gate
// next opens.
func (a *cudaAdmission) wait(tg *kernel.ThreadGroup) <-chan struct{} {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed == nil || (a.exempt != nil && a.exempt(tg)) {
		return nil
	}
	return a.closed
}

// CloseCudaAdmission holds CUDA initialization in all processes except those
// for which exempt returns true, until OpenCudaAdmission is called. It is a
// no-op if nvproxy is not registered.
func CloseCudaAdmission(vfsObj *vfs.VirtualFilesystem, exempt func(*kernel.ThreadGroup) bool) {
	if nvp := nvproxyFromVFS(vfsObj); nvp != nil {
		nvp.admission.close(exempt)
	}
}

// OpenCudaAdmission releases processes held by CloseCudaAdmission. It is a
// no-op if the gate is open or nvproxy is not registered.
func OpenCudaAdmission(vfsObj *vfs.VirtualFilesystem) {
	if nvp := nvproxyFromVFS(vfsObj); nvp != nil {
		nvp.admission.open()
	}
}

// awaitCudaAdmission blocks t while the gate is closed for its thread group.
// If the block is interrupted, the ioctl is restarted so that it retries
// after the save or restore completes.
func (nvp *nvproxy) awaitCudaAdmission(t *kernel.Task) error {
	ch := nvp.admission.wait(t.ThreadGroup())
	if ch == nil {
		return nil
	}
	log.Infof("nvproxy: holding CUDA initialization in PID %d until the in-progress checkpoint completes", t.ThreadGroup().ID())
	for ch != nil {
		if err := t.Block(ch); err != nil {
			return linuxerr.ERESTARTNOINTR
		}
		ch = nvp.admission.wait(t.ThreadGroup())
	}
	return nil
}
