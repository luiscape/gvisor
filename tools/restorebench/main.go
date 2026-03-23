// Command restorebench checkpoints a container and measures restore latency.
// It must be run as root.
//
// Usage:
//
//	cd <gvisor repo root>
//	go build -o restorebench ./tools/restorebench
//	sudo ./restorebench -runsc=./bazel-out/k8-opt/bin/runsc/runsc_/runsc
//	sudo ./restorebench -runsc=./bazel-out/k8-opt/bin/runsc/runsc_/runsc -mem=1024
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

var (
	runscFlag   = flag.String("runsc", "", "path to a pre-built runsc binary (required)")
	iterations  = flag.Int("iterations", 5, "number of restore iterations")
	compression = flag.String("compression", "none", "checkpoint compression: none|flate-best-speed")
	platFlag    = flag.String("platform", "systrap", "runsc platform")
	keepTmp     = flag.Bool("keep-tmp", false, "keep temp dirs for debugging")
	debugLog    = flag.Bool("debug-log", false, "enable debug logging for first restore iteration")
	background  = flag.Bool("background", false, "pass --background to runsc restore")
	memMiB      = flag.Int("mem", 0, "MiB of memory the workload allocates and touches (0 = use sleep-only)")
	touchPct    = flag.Int("touch-pct", 100, "percentage of allocated pages to touch with non-zero data (1-100)")
	settleTime  = flag.Duration("settle", 2*time.Second, "time to wait after start before checkpoint")
	directCkpt    = flag.Bool("direct-checkpoint", false, "use O_DIRECT for writing checkpoint pages file")
	directRestore = flag.Bool("direct-restore", false, "use O_DIRECT for reading checkpoint pages file during restore")
	excludeZero   = flag.Bool("exclude-zero-pages", false, "pass --exclude-committed-zero-pages to checkpoint (scans all pages, skips zeros)")
	precreate     = flag.Bool("precreate", false, "pre-create sandbox with 'runsc create' before timing; measures only the restore RPC, not fork/exec")
)

// ---------------------------------------------------------------------------
// Workload binary: a tiny C program that malloc+memsets N MiB, then sleeps.
// We compile it with the host cc into the bundle directory.
// ---------------------------------------------------------------------------

const allocSrc = `
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        /* No argument: just sleep. */
        pause();
        return 0;
    }
    long mib = atol(argv[1]);
    if (mib <= 0) {
        pause();
        return 0;
    }
    /* Optional second arg: percentage of pages to touch (1-100, default 100). */
    int touch_pct = 100;
    if (argc >= 3) {
        touch_pct = atoi(argv[2]);
        if (touch_pct <= 0) touch_pct = 1;
        if (touch_pct > 100) touch_pct = 100;
    }
    long bytes = mib * 1024L * 1024L;
    char *p = malloc(bytes);
    if (!p) {
        perror("malloc");
        return 1;
    }
    /* Touch pages so the memory is committed.
     * If touch_pct < 100, only touch that fraction of pages, leaving
     * the rest as zero pages (simulates sparse workloads like JVMs). */
    long pagesz = sysconf(_SC_PAGESIZE);
    long total_pages = bytes / pagesz;
    long step = 1;
    if (touch_pct < 100) {
        step = 100 / touch_pct;
        if (step < 1) step = 1;
    }
    long touched = 0;
    for (long pg = 0; pg < total_pages; pg += step) {
        p[pg * pagesz] = (char)(pg >> 4);
        touched++;
    }
    fprintf(stderr, "allocated %ld MiB, touched %ld/%ld pages (%d%%)\n",
            mib, touched, total_pages, touch_pct);
    /* Keep the process alive. */
    for (;;) pause();
    return 0;
}
`

// buildAllocBin compiles the alloc helper into dir and returns its path.
func buildAllocBin(dir string) (string, error) {
	src := filepath.Join(dir, "alloc.c")
	bin := filepath.Join(dir, "alloc")
	if err := os.WriteFile(src, []byte(allocSrc), 0644); err != nil {
		return "", err
	}
	// Static link so it works inside the minimal container.
	cmd := exec.Command("cc", "-static", "-O2", "-o", bin, src)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("compiling alloc helper: %w", err)
	}
	return bin, nil
}

// ---------------------------------------------------------------------------
// OCI spec helpers
// ---------------------------------------------------------------------------

func ociSpec(args []string, extraBinds []mount) map[string]any {
	mnts := []map[string]any{
		{"destination": "/proc", "type": "proc", "source": "proc"},
		{"destination": "/tmp", "type": "tmpfs", "source": "tmpfs"},
	}
	for _, b := range extraBinds {
		mnts = append(mnts, map[string]any{
			"destination": b.dst,
			"type":        "bind",
			"source":      b.src,
			"options":     []string{"rbind", "ro"},
		})
	}
	return map[string]any{
		"ociVersion": "1.0.0",
		"process": map[string]any{
			"terminal": false,
			"user":     map[string]any{"uid": 0, "gid": 0},
			"args":     args,
			"env":      []string{"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"},
			"cwd":      "/",
		},
		"root": map[string]any{
			"path":     "/",
			"readonly": true,
		},
		"hostname": "restorebench",
		"mounts":   mnts,
		"linux": map[string]any{
			"namespaces": []map[string]any{
				{"type": "pid"},
				{"type": "ipc"},
				{"type": "uts"},
				{"type": "mount"},
			},
		},
	}
}

type mount struct{ src, dst string }

func writeBundle(dir string, args []string, extraBinds []mount) (string, error) {
	bundleDir := filepath.Join(dir, "bundle")
	if err := os.MkdirAll(bundleDir, 0755); err != nil {
		return "", err
	}
	spec := ociSpec(args, extraBinds)
	b, err := json.MarshalIndent(spec, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(bundleDir, "config.json"), b, 0644); err != nil {
		return "", err
	}
	return bundleDir, nil
}

// ---------------------------------------------------------------------------
// runsc helper
// ---------------------------------------------------------------------------

func runscCmd(bin, rootDir, logDir string, args ...string) *exec.Cmd {
	all := []string{
		"--rootless=false",
		"--root=" + rootDir,
		"--platform=" + *platFlag,
		"--network=none",
	}
	if logDir != "" {
		all = append(all,
			"--debug",
			"--debug-log="+filepath.Join(logDir, "runsc.log.%COMMAND%"),
		)
	}
	all = append(all, args...)
	cmd := exec.Command(bin, all...)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	return cmd
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

func main() {
	flag.Parse()
	log.SetFlags(log.Lmicroseconds)

	if os.Getuid() != 0 {
		log.Fatal("this program must be run as root")
	}
	if *runscFlag == "" {
		log.Fatal("-runsc flag is required")
	}

	// ---- Step 1: Resolve runsc binary ----
	log.Println("=== Step 1: Resolve runsc ===")
	runscBin, err := filepath.Abs(*runscFlag)
	if err != nil {
		log.Fatalf("resolving runsc path: %v", err)
	}
	if _, err := os.Stat(runscBin); err != nil {
		log.Fatalf("runsc binary not found at %s: %v", runscBin, err)
	}
	log.Printf("    runsc: %s", runscBin)
	if out, err := exec.Command(runscBin, "--version").CombinedOutput(); err == nil {
		log.Printf("    version: %s", strings.TrimSpace(string(out)))
	}

	// ---- Step 2: Prepare workload & bundle ----
	log.Println("=== Step 2: Prepare workload ===")

	tmpDir, err := os.MkdirTemp("", "restorebench-*")
	if err != nil {
		log.Fatalf("creating temp dir: %v", err)
	}
	if *keepTmp {
		log.Printf("    temp dir: %s", tmpDir)
	} else {
		defer os.RemoveAll(tmpDir)
	}

	var containerArgs []string
	var extraBinds []mount

	if *memMiB > 0 {
		allocBin, err := buildAllocBin(tmpDir)
		if err != nil {
			log.Fatalf("building alloc helper: %v", err)
		}
		log.Printf("    alloc binary: %s", allocBin)

		// Bind-mount the binary into the container at /workload/alloc.
		wlDir := filepath.Join(tmpDir, "workload")
		if err := os.MkdirAll(wlDir, 0755); err != nil {
			log.Fatalf("creating workload dir: %v", err)
		}
		// Copy the binary into the workload dir (bind-mount source).
		dst := filepath.Join(wlDir, "alloc")
		data, err := os.ReadFile(allocBin)
		if err != nil {
			log.Fatalf("reading alloc binary: %v", err)
		}
		if err := os.WriteFile(dst, data, 0755); err != nil {
			log.Fatalf("writing alloc binary: %v", err)
		}
		extraBinds = append(extraBinds, mount{src: wlDir, dst: "/workload"})
		containerArgs = []string{"/workload/alloc", fmt.Sprintf("%d", *memMiB), fmt.Sprintf("%d", *touchPct)}
		log.Printf("    workload: allocate %d MiB, touch %d%% of pages", *memMiB, *touchPct)
	} else {
		containerArgs = []string{"sleep", "infinity"}
		log.Println("    workload: sleep (no memory allocation)")
	}

	bundleDir, err := writeBundle(tmpDir, containerArgs, extraBinds)
	if err != nil {
		log.Fatalf("writing bundle: %v", err)
	}

	stateDir := filepath.Join(tmpDir, "state")
	os.MkdirAll(stateDir, 0755)
	checkpointDir := filepath.Join(tmpDir, "checkpoint")
	os.MkdirAll(checkpointDir, 0755)

	// ---- Step 3: Create, start, (wait for alloc), checkpoint ----
	log.Println("=== Step 3: Checkpoint ===")

	containerID := "restorebench-src"

	log.Println("    creating container...")
	if err := runscCmd(runscBin, stateDir, "", "create", "--bundle="+bundleDir, containerID).Run(); err != nil {
		log.Fatalf("runsc create: %v", err)
	}
	defer runscCmd(runscBin, stateDir, "", "delete", "-force", containerID).Run()

	log.Println("    starting container...")
	if err := runscCmd(runscBin, stateDir, "", "start", containerID).Run(); err != nil {
		log.Fatalf("runsc start: %v", err)
	}

	// Wait for memory allocation to complete.
	settle := *settleTime
	if *memMiB > 0 && settle < 2*time.Second {
		settle = 2 * time.Second
	}
	if *memMiB >= 512 {
		// Give bigger allocations more time.
		extra := time.Duration(*memMiB/512) * time.Second
		if settle < 2*time.Second+extra {
			settle = 2*time.Second + extra
		}
	}
	log.Printf("    waiting %v for workload to settle...", settle)
	time.Sleep(settle)

	log.Println("    checkpointing...")
	ckptArgs := []string{
		"checkpoint",
		"--image-path=" + checkpointDir,
		"--compression=" + *compression,
	}
	if *directCkpt {
		ckptArgs = append(ckptArgs, "--direct")
	}
	if *excludeZero {
		ckptArgs = append(ckptArgs, "--exclude-committed-zero-pages")
	}
	ckptArgs = append(ckptArgs, containerID)
	ckptStart := time.Now()
	if err := runscCmd(runscBin, stateDir, "", ckptArgs...).Run(); err != nil {
		log.Fatalf("runsc checkpoint: %v", err)
	}
	ckptDur := time.Since(ckptStart)
	log.Printf("    checkpoint completed in %v", ckptDur)

	// Print checkpoint file sizes.
	entries, _ := os.ReadDir(checkpointDir)
	var totalBytes int64
	for _, e := range entries {
		info, _ := e.Info()
		if info != nil {
			totalBytes += info.Size()
			log.Printf("    %s: %s", e.Name(), humanBytes(info.Size()))
		}
	}
	log.Printf("    total checkpoint size: %s", humanBytes(totalBytes))

	// ---- Step 4: Measure restore ----
	if *precreate {
		log.Println("=== Step 4: Measuring restore (precreate mode — sandbox pre-created before timing) ===")
	} else {
		log.Println("=== Step 4: Measuring restore ===")
	}

	durations := make([]time.Duration, 0, *iterations)
	for i := 0; i < *iterations; i++ {
		restoreID := fmt.Sprintf("restorebench-dst-%d", i)
		restoreStateDir := filepath.Join(tmpDir, fmt.Sprintf("state-restore-%d", i))
		os.MkdirAll(restoreStateDir, 0755)

		var restoreLogDir string
		if *debugLog && i == 0 {
			restoreLogDir = filepath.Join(tmpDir, "debug-logs")
			os.MkdirAll(restoreLogDir, 0755)
		}

		// In precreate mode, create the sandbox BEFORE we start the timer.
		// This simulates a pre-forked sandbox pool: the fork/exec/re-exec
		// and loader init (~67ms) have already happened. We measure only
		// the restore RPC (connect + send checkpoint + load state).
		if *precreate {
			log.Printf("    pre-creating sandbox %s...", restoreID)
			if err := runscCmd(runscBin, restoreStateDir, "", "create", "--bundle="+bundleDir, restoreID).Run(); err != nil {
				log.Fatalf("runsc create (precreate): %v", err)
			}
		}

		restoreArgs := []string{
			"restore",
			"--detach",
			"--image-path=" + checkpointDir,
			"--bundle=" + bundleDir,
		}
		if *background {
			restoreArgs = append(restoreArgs, "--background")
		}
		if *directRestore {
			restoreArgs = append(restoreArgs, "--direct")
		}
		restoreArgs = append(restoreArgs, restoreID)

		start := time.Now()
		if err := runscCmd(runscBin, restoreStateDir, restoreLogDir, restoreArgs...).Run(); err != nil {
			log.Fatalf("runsc restore: %v", err)
		}
		d := time.Since(start)
		durations = append(durations, d)
		log.Printf("    restore %d/%d: %v", i+1, *iterations, d)

		if restoreLogDir != "" {
			printDebugLogTimings(restoreLogDir)
		}

		// Cleanup.
		_ = runscCmd(runscBin, restoreStateDir, "", "kill", restoreID, "SIGKILL").Run()
		time.Sleep(200 * time.Millisecond)
		_ = runscCmd(runscBin, restoreStateDir, "", "delete", "-force", restoreID).Run()
	}

	// ---- Summary ----
	fmt.Println()
	fmt.Println("=== Restore Benchmark Results ===")
	fmt.Printf("Platform:       %s\n", *platFlag)
	fmt.Printf("Compression:    %s\n", *compression)
	fmt.Printf("Memory:         %d MiB\n", *memMiB)
	fmt.Printf("Checkpoint:     %s\n", humanBytes(totalBytes))
	fmt.Printf("Iterations:     %d\n", *iterations)
	if *precreate {
		fmt.Println("Precreate:      yes (sandbox pre-created, measuring only restore RPC)")
	}
	if *background {
		fmt.Println("Background:     yes")
	}
	if *directRestore {
		fmt.Println("Direct I/O:     yes")
	}
	fmt.Println()

	var total time.Duration
	mn, mx := durations[0], durations[0]
	for _, d := range durations {
		total += d
		if d < mn {
			mn = d
		}
		if d > mx {
			mx = d
		}
	}
	avg := total / time.Duration(len(durations))

	fmt.Printf("Min:     %v\n", mn)
	fmt.Printf("Max:     %v\n", mx)
	fmt.Printf("Avg:     %v\n", avg)
	fmt.Printf("Total:   %v\n", total)
	for i, d := range durations {
		fmt.Printf("  [%d] %v\n", i, d)
	}
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func humanBytes(b int64) string {
	switch {
	case b >= 1<<30:
		return fmt.Sprintf("%.2f GiB", float64(b)/(1<<30))
	case b >= 1<<20:
		return fmt.Sprintf("%.2f MiB", float64(b)/(1<<20))
	case b >= 1<<10:
		return fmt.Sprintf("%.2f KiB", float64(b)/(1<<10))
	default:
		return fmt.Sprintf("%d B", b)
	}
}

func printDebugLogTimings(logDir string) {
	entries, err := os.ReadDir(logDir)
	if err != nil {
		log.Printf("    (could not read debug logs: %v)", err)
		return
	}
	fmt.Fprintln(os.Stderr)
	fmt.Fprintln(os.Stderr, "    === Debug Log Timing Breakdown (first restore) ===")
	keywords := []string{
		"Restore", "restore", "load", "Load", "took", "timer", "Timer",
		"Reached", "Overall", "CPUID", "Kernel", "Memory", "MF ",
		"Starting sandbox", "kernel loaded", "specs validated",
		"page loading", "page file", "Async", "async",
		"MemoryFile", "pread", "throughput",
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		data, err := os.ReadFile(filepath.Join(logDir, entry.Name()))
		if err != nil {
			continue
		}
		for _, line := range strings.Split(string(data), "\n") {
			for _, kw := range keywords {
				if strings.Contains(line, kw) {
					fmt.Fprintf(os.Stderr, "    %s\n", strings.TrimSpace(line))
					break
				}
			}
		}
	}
	fmt.Fprintln(os.Stderr)
}
