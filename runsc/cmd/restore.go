// Copyright 2018 The gVisor Authors.
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

package cmd

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/google/subcommands"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"golang.org/x/sys/unix"
	"gvisor.dev/gvisor/pkg/cleanup"
	"gvisor.dev/gvisor/pkg/log"
	"gvisor.dev/gvisor/runsc/cmd/util"
	"gvisor.dev/gvisor/runsc/config"
	"gvisor.dev/gvisor/runsc/container"
	"gvisor.dev/gvisor/runsc/flag"
	"gvisor.dev/gvisor/runsc/specutils"
)

// Restore implements subcommands.Command for the "restore" command.
type Restore struct {
	// Restore flags are a super-set of those for Create.
	Create

	containerLoader

	// imagePath is the path to the saved container image
	imagePath string

	// detach indicates that runsc has to start a process and exit without waiting it.
	detach bool

	// direct indicates whether O_DIRECT should be used for reading the
	// checkpoint pages file. It is faster if the checkpoint files are not
	// already in the page cache (for example if its coming from an untouched
	// network block device). Usually the restore is done only once, so the cost
	// of adding the checkpoint files to the page cache can be redundant.
	direct bool

	// If background is true, the container image may continue to be read after
	// the restore command exits. For large images, this significantly shortens
	// the amount of time taken by the restore command. The checkpoint must be
	// uncompressed for background to work; if the checkpoint is compressed,
	// background has no effect.
	background bool
}

// Name implements subcommands.Command.Name.
func (*Restore) Name() string {
	return "restore"
}

// Synopsis implements subcommands.Command.Synopsis.
func (*Restore) Synopsis() string {
	return "restore a saved state of container (experimental)"
}

// Usage implements subcommands.Command.Usage.
func (*Restore) Usage() string {
	return "restore [flags] <container id> - restore saved state of container.\n"
}

// SetFlags implements subcommands.Command.SetFlags.
func (r *Restore) SetFlags(f *flag.FlagSet) {
	r.Create.SetFlags(f)
	f.StringVar(&r.imagePath, "image-path", "", "directory path to saved container image")
	f.BoolVar(&r.detach, "detach", false, "detach from the container's process")
	f.BoolVar(&r.direct, "direct", false, "use O_DIRECT for reading checkpoint pages file")
	f.BoolVar(&r.background, "background", false, "allow image loading to continue after restore exits (requires uncompressed checkpoint)")

	// Unimplemented flags necessary for compatibility with docker.

	var nsr bool
	f.BoolVar(&nsr, "no-subreaper", false, "ignored")

	var wp string
	f.StringVar(&wp, "work-path", "", "ignored")
}

// FetchSpec implements util.SubCommand.FetchSpec.
func (r *Restore) FetchSpec(conf *config.Config, f *flag.FlagSet) (string, *specs.Spec, error) {
	if f.NArg() != 1 {
		return "", nil, fmt.Errorf("a container id is required")
	}
	id := f.Arg(0)

	// If the spec is already set via Create.FetchSpec(), use it.
	if r.spec != nil {
		return id, r.spec, nil
	}

	// Try loading the container first, similar to Execute().
	c, err := r.loadContainer(conf, f, container.LoadOpts{})
	if err == nil {
		return c.ID, c.Spec, nil
	}
	if err != os.ErrNotExist {
		return "", nil, fmt.Errorf("loading container: %w", err)
	}

	// Container not found. Fallback to reading spec from bundle.
	return r.Create.FetchSpec(conf, f)
}

// Execute implements subcommands.Command.Execute.
func (r *Restore) Execute(_ context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if f.NArg() != 1 {
		f.Usage()
		return subcommands.ExitUsageError
	}

	id := f.Arg(0)
	conf := args[0].(*config.Config)
	waitStatus := args[1].(*unix.WaitStatus)

	if conf.Rootless {
		return util.Errorf("Rootless mode not supported with %q", r.Name())
	}

	bundleDir := r.bundleDir
	if bundleDir == "" {
		bundleDir = getwdOrDie()
	}
	if r.imagePath == "" {
		return util.Errorf("image-path flag must be provided")
	}

	var cu cleanup.Cleanup
	defer cu.Clean()

	runArgs := container.Args{
		ID:            id,
		Spec:          nil,
		BundleDir:     bundleDir,
		ConsoleSocket: r.consoleSocket,
		PIDFile:       r.pidFile,
		UserLog:       r.userLog,
		Attached:      !r.detach,
	}

	log.Debugf("Restore container, cid: %s, rootDir: %q", id, conf.RootDir)
	c, err := r.loadContainer(conf, f, container.LoadOpts{})
	if err != nil {
		if err != os.ErrNotExist {
			return util.Errorf("loading container: %v", err)
		}

		log.Warningf("Container not found, creating new one, cid: %s, spec from: %s", id, bundleDir)

		// Read the spec from the bundle directory.
		if r.spec == nil {
			if r.spec, err = specutils.ReadSpec(bundleDir, conf); err != nil {
				return util.Errorf("reading spec: %v", err)
			}
		}
		runArgs.Spec = r.spec
		specutils.LogSpecDebug(runArgs.Spec, conf.OCISeccomp)

		if c, err = container.New(conf, runArgs); err != nil {
			return util.Errorf("creating container: %v", err)
		}

		// Clean up partially created container if an error occurs.
		// Any errors returned by Destroy() itself are ignored.
		cu.Add(func() {
			c.Destroy()
		})
	} else {
		runArgs.Spec = c.Spec
	}

	// Run the GPU checkpoint restore helper from the CLI process BEFORE
	// sending the restore RPC to the sandbox.  The CLI process has NO
	// seccomp filters, so the helper can freely dlopen(libcuda.so) and
	// call cuCheckpointProcessRestore.  The helper targets the sandbox
	// process (which has the checkpoint data in its address space after
	// the restore RPC loads the checkpoint image).
	//
	// However, we need the sandbox to have loaded the checkpoint first
	// (so the GPU checkpoint data is in memory).  So we run the helper
	// AFTER c.Restore() returns — but the sandbox hasn't called onStart()
	// yet if we use a two-phase approach.
	//
	// For now, run the helper targeting the sandbox PID after Restore
	// returns.  The sandbox's ioctl gate blocks app threads from issuing
	// GPU ioctls until postRestoreImpl opens the gate.  The helper runs
	// here (no seccomp) and calls cuCheckpointProcessRestore(sandbox_pid).
	log.Debugf("Restore: %v", r.imagePath)
	err = c.Restore(conf, r.imagePath, r.direct, r.background)

	// After Restore returns, the sandbox has loaded the checkpoint and
	// started the kernel.  App threads are running but GPU ioctls are
	// blocked by the nvproxy ioctl gate.  Run the GPU restore helper
	// NOW from the CLI process (which has NO seccomp filters) targeting
	// the sandbox PID.  The helper calls cuCheckpointProcessRestore to
	// reconstruct GPU state from checkpoint data in the sandbox's memory.
	// Always try to run the GPU restore helper after Restore returns.
	// We don't check conf.NVProxy because the restore CLI doesn't receive
	// --nvproxy (it was only on the create command).  Instead we just
	// check if the helper binary exists on the host — if it does, we run
	// it.  If it doesn't, we silently skip.  The helper itself is a no-op
	// if the sandbox has no CUDA checkpoint data to restore.
	if err == nil {
		sandboxPid := c.SandboxPid()
		log.Infof("GPU restore CLI: sandboxPid=%d, looking for helper", sandboxPid)
		if sandboxPid > 0 {
			if herr := runGPURestoreFromCLI(sandboxPid); herr != nil {
				log.Warningf("GPU restore helper failed: %v (GPU state may not be restored)", herr)
			}
		}
	}
	if err != nil {
		return util.Errorf("starting container: %v", err)
	}

	// If we allocate a terminal, forward signals to the sandbox process.
	// Otherwise, Ctrl+C will terminate this process and its children,
	// including the terminal.
	if c.Spec.Process.Terminal {
		stopForwarding := c.ForwardSignals(0, true /* fgProcess */)
		defer stopForwarding()
	}

	var ws unix.WaitStatus
	if runArgs.Attached {
		if ws, err = c.Wait(); err != nil {
			return util.Errorf("running container: %v", err)
		}
	}
	*waitStatus = ws

	cu.Release()

	return subcommands.ExitSuccess
}

// runGPURestoreFromCLI exec's the cuda_checkpoint_helper binary from the
// CLI process (which has NO seccomp filters) targeting the given sandbox PID.
// The helper calls cuCheckpointProcessRestore(sandboxPid) to reconstruct
// GPU state from the checkpoint data in the sandbox process's memory.
func runGPURestoreFromCLI(sandboxPid int) error {
	searchPaths := []string{
		"/usr/local/bin/cuda_checkpoint_helper",
		"/tmp/cuda_checkpoint_helper",
	}
	if self, err := os.Executable(); err == nil {
		searchPaths = append([]string{filepath.Join(filepath.Dir(self), "cuda_checkpoint_helper")}, searchPaths...)
	}

	var helperPath string
	for _, p := range searchPaths {
		if info, err := os.Stat(p); err == nil && !info.IsDir() {
			helperPath = p
			break
		}
	}
	if helperPath == "" {
		log.Infof("GPU restore: cuda_checkpoint_helper not found, skipping")
		return nil
	}

	log.Infof("GPU restore: running %s from CLI (no seccomp), sandbox PID=%d", helperPath, sandboxPid)

	cmd := exec.Command(helperPath)
	cmd.Env = append(os.Environ(),
		"GVISOR_SAVE_RESTORE_AUTO_EXEC_MODE=restore",
		fmt.Sprintf("GVISOR_SANDBOX_PID=%d", sandboxPid),
	)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("starting %s: %w", helperPath, err)
	}

	tmout := 10 * time.Minute
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()

	select {
	case err := <-done:
		if err != nil {
			return fmt.Errorf("%s: %w", helperPath, err)
		}
		log.Infof("GPU restore: helper completed successfully")
		return nil
	case <-time.After(tmout):
		cmd.Process.Kill()
		return fmt.Errorf("%s timed out after %v", helperPath, tmout)
	}
}
