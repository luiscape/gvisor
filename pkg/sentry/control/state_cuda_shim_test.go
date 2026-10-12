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
	"testing"
	"time"

	"gvisor.dev/gvisor/pkg/abi/linux"
	"gvisor.dev/gvisor/pkg/context"
	"gvisor.dev/gvisor/pkg/fspath"
	"gvisor.dev/gvisor/pkg/sentry/contexttest"
	"gvisor.dev/gvisor/pkg/sentry/fsimpl/tmpfs"
	"gvisor.dev/gvisor/pkg/sentry/kernel/auth"
	"gvisor.dev/gvisor/pkg/sentry/vfs"
	"gvisor.dev/gvisor/pkg/usermem"
)

// shimTestFS is a tmpfs root with /tmp, /etc and a 0600 /etc/victim, as root.
type shimTestFS struct {
	t     *testing.T
	ctx   context.Context
	vfs   *vfs.VirtualFilesystem
	creds *auth.Credentials
	mntns *vfs.MountNamespace
	root  vfs.VirtualDentry
}

func newShimTestFS(t *testing.T) *shimTestFS {
	ctx := contexttest.Context(t)
	creds := auth.NewRootCredentials(auth.NewRootUserNamespace())
	vfsObj := &vfs.VirtualFilesystem{}
	if err := vfsObj.Init(ctx); err != nil {
		t.Fatalf("VFS init: %v", err)
	}
	vfsObj.MustRegisterFilesystemType("tmpfs", tmpfs.FilesystemType{}, &vfs.RegisterFilesystemTypeOptions{AllowUserMount: true})
	mntns, err := vfsObj.NewMountNamespace(ctx, creds, "", "tmpfs", &vfs.MountOptions{}, nil)
	if err != nil {
		t.Fatalf("tmpfs root: %v", err)
	}
	root := mntns.Root(ctx)
	t.Cleanup(func() {
		root.DecRef(ctx)
		mntns.DecRef(ctx)
	})
	// As root throughout: linking a file the caller cannot read and write
	// is refused otherwise (vfs.MayLink), and root can plant anything.
	ctx = vfs.WithMountNamespace(auth.ContextWithCredentials(ctx, creds), mntns)
	f := &shimTestFS{t: t, ctx: ctx, vfs: vfsObj, creds: creds, mntns: mntns, root: root}
	f.mkdir("/tmp", 01777)
	f.mkdir("/etc", 0755)
	f.write("/etc/victim", "secret", 0600)
	return f
}

func (f *shimTestFS) pop(path string) *vfs.PathOperation {
	return &vfs.PathOperation{Root: f.root, Start: f.root, Path: fspath.Parse(path)}
}

func (f *shimTestFS) mkdir(path string, mode linux.FileMode) {
	if err := f.vfs.MkdirAt(f.ctx, f.creds, f.pop(path), &vfs.MkdirOptions{Mode: mode}); err != nil {
		f.t.Fatalf("mkdir %s: %v", path, err)
	}
}

func (f *shimTestFS) write(path, data string, mode linux.FileMode) {
	fd, err := f.vfs.OpenAt(f.ctx, f.creds, f.pop(path), &vfs.OpenOptions{Flags: linux.O_CREAT | linux.O_WRONLY, Mode: mode})
	if err != nil {
		f.t.Fatalf("create %s: %v", path, err)
	}
	defer fd.DecRef(f.ctx)
	if _, err := fd.Write(f.ctx, usermem.BytesIOSequence([]byte(data)), vfs.WriteOptions{}); err != nil {
		f.t.Fatalf("write %s: %v", path, err)
	}
}

func (f *shimTestFS) symlink(target, path string) {
	if err := f.vfs.SymlinkAt(f.ctx, f.creds, f.pop(path), target); err != nil {
		f.t.Fatalf("symlink %s: %v", path, err)
	}
}

func (f *shimTestFS) link(oldpath, path string) {
	if err := f.vfs.LinkAt(f.ctx, f.creds, f.pop(oldpath), f.pop(path)); err != nil {
		f.t.Fatalf("link %s: %v", path, err)
	}
}

func (f *shimTestFS) mkfifo(path string) {
	if err := f.vfs.MknodAt(f.ctx, f.creds, f.pop(path), &vfs.MknodOptions{Mode: linux.S_IFIFO | 0666}); err != nil {
		f.t.Fatalf("mkfifo %s: %v", path, err)
	}
}

// stat returns path's mode, or 0 if it does not exist (without following a
// final symlink).
func (f *shimTestFS) stat(path string) linux.FileMode {
	st, err := f.vfs.StatAt(f.ctx, f.creds, f.pop(path), &vfs.StatOptions{Mask: linux.STATX_TYPE | linux.STATX_MODE})
	if err != nil {
		return 0
	}
	return linux.FileMode(st.Mode)
}

// victimIntact fails the test unless /etc/victim is unchanged.
func (f *shimTestFS) victimIntact() {
	f.t.Helper()
	if m := f.stat("/etc/victim"); m != linux.S_IFREG|0600 {
		f.t.Errorf("/etc/victim has mode %#o, want %#o", m, linux.S_IFREG|0600)
	}
	fd, err := f.vfs.OpenAt(f.ctx, f.creds, f.pop("/etc/victim"), &vfs.OpenOptions{Flags: linux.O_RDONLY})
	if err != nil {
		f.t.Fatalf("open /etc/victim: %v", err)
	}
	defer fd.DecRef(f.ctx)
	buf := make([]byte, 64)
	n, _ := fd.PRead(f.ctx, usermem.BytesIOSequence(buf), 0, vfs.ReadOptions{})
	if got := string(buf[:n]); got != "secret" {
		f.t.Errorf("/etc/victim contains %q, want %q", got, "secret")
	}
}

func (f *shimTestFS) shimFS() (*cudaShimFS, error) {
	return newCudaShimFS(f.ctx, f.vfs, f.creds, f.mntns, f.root)
}

// within fails the test if fn does not return within a second: a FIFO opened
// blocking would hang it.
func within(t *testing.T, what string, fn func()) {
	t.Helper()
	done := make(chan struct{})
	go func() {
		fn()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatalf("%s blocked", what)
	}
}

// TestCudaShimFS covers the sentry's access to the interposer's control
// directory: markers and acks work in a real directory, and nothing a
// container user can plant there (the directory as a symlink, or a symlink,
// hard link or FIFO in place of a marker or ack) makes the sentry act outside
// the directory or block.
func TestCudaShimFS(t *testing.T) {
	t.Run("markers and acks", func(t *testing.T) {
		f := newShimTestFS(t)
		f.mkdir(cudaShimDir, 01777)
		f.write(cudaShimDir+"/error.7", "managed memory\n", 0644)
		fs, err := f.shimFS()
		if err != nil {
			t.Fatalf("newCudaShimFS: %v", err)
		}
		defer fs.release()
		if err := fs.create(cudaShimGateMarker); err != nil {
			t.Fatalf("create gate: %v", err)
		}
		if err := fs.create(cudaShimGateMarker); err != nil {
			t.Errorf("create gate again: %v", err)
		}
		if !fs.exists(cudaShimGateMarker) || f.stat(cudaShimDir+"/gate") != linux.S_IFREG|0644 {
			t.Errorf("gate is %#o, want a 0644 regular file", f.stat(cudaShimDir+"/gate"))
		}
		if got := fs.readTail("error.7", 512); got != "managed memory\n" {
			t.Errorf("readTail(error.7) = %q", got)
		}
		if got := fs.readTail("error.7", 7); got != "memory\n" {
			t.Errorf("readTail(error.7, 7) = %q", got)
		}
		if err := fs.remove(cudaShimGateMarker); err != nil || fs.exists(cudaShimGateMarker) {
			t.Errorf("remove gate: %v, exists %v", err, fs.exists(cudaShimGateMarker))
		}
		if err := fs.remove(cudaShimGateMarker); err != nil {
			t.Errorf("remove a missing gate: %v", err)
		}
	})

	t.Run("missing directory", func(t *testing.T) {
		f := newShimTestFS(t)
		if _, err := f.shimFS(); err == nil {
			t.Errorf("newCudaShimFS succeeded without %s", cudaShimDir)
		}
	})

	t.Run("directory is a symlink", func(t *testing.T) {
		f := newShimTestFS(t)
		f.symlink("/etc", cudaShimDir)
		if fs, err := f.shimFS(); err == nil {
			fs.release()
			t.Fatalf("newCudaShimFS followed %s -> /etc", cudaShimDir)
		}
		if f.stat("/etc/gate") != 0 {
			t.Errorf("/etc/gate was created")
		}
	})

	t.Run("planted files", func(t *testing.T) {
		f := newShimTestFS(t)
		f.mkdir(cudaShimDir, 01777)
		f.symlink("/etc/victim", cudaShimDir+"/gate")
		f.symlink("/etc/created", cudaShimDir+"/suspend")
		f.link("/etc/victim", cudaShimDir+"/error.7")
		f.symlink("/etc/victim", cudaShimDir+"/error.8")
		f.mkfifo(cudaShimDir + "/error.9")
		f.mkfifo(cudaShimDir + "/mcshim.log")
		f.mkfifo(cudaShimDir + "/present.10")
		fs, err := f.shimFS()
		if err != nil {
			t.Fatalf("newCudaShimFS: %v", err)
		}
		defer fs.release()

		if err := fs.create("gate"); err == nil {
			t.Errorf("create through a symlink to an existing file succeeded")
		}
		if err := fs.create("suspend"); err == nil {
			t.Errorf("create through a dangling symlink succeeded")
		}
		if f.stat("/etc/created") != 0 {
			t.Errorf("/etc/created was created through the symlink")
		}
		for _, name := range []string{"error.7", "error.8"} {
			if got := fs.readTail(name, 512); got != "" {
				t.Errorf("readTail(%s) = %q, read through a link", name, got)
			}
		}
		within(t, "readTail of a FIFO", func() {
			if got := fs.readTail("error.9", 512); got != "" {
				t.Errorf("readTail(FIFO) = %q", got)
			}
			if got := fs.readTail("mcshim.log", 512); got != "" {
				t.Errorf("readTail(FIFO log) = %q", got)
			}
		})
		within(t, "create over a FIFO", func() {
			if err := fs.create("present.10"); err == nil {
				t.Errorf("create over a FIFO succeeded")
			}
		})
		// Removing a planted name removes it, not what it points to.
		for _, name := range []string{"gate", "error.7"} {
			if err := fs.remove(name); err != nil {
				t.Errorf("remove(%s): %v", name, err)
			}
		}
		f.victimIntact()
	})
}
