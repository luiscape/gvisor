// Command restorebench checkpoints a container and measures restore latency
// using direct Go API calls (no shelling out to runsc).
//
// Usage:
//
//	cd <gvisor repo root>
//	go build -o restorebench ./tools/restorebench
//	sudo ./restorebench -runsc=./bazel-out/k8-opt/bin/runsc/runsc_/runsc
//	sudo ./restorebench -runsc=./bazel-out/k8-opt/bin/runsc/runsc_/runsc -mem=1024
//	sudo ./restorebench -runsc=./bazel-out/k8-opt/bin/runsc/runsc_/runsc -mem=10240 -precreate -background
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	goflag "flag"

	specs "github.com/opencontainers/runtime-spec/specs-go"
	"gvisor.dev/gvisor/pkg/state/statefile"
	"gvisor.dev/gvisor/runsc/config"
	"gvisor.dev/gvisor/runsc/container"
	"gvisor.dev/gvisor/runsc/flag"
	"gvisor.dev/gvisor/runsc/sandbox"
	"gvisor.dev/gvisor/runsc/specutils"
)

var (
	runscFlag  = goflag.String("runsc", "", "path to a pre-built runsc binary (required)")
	iterations = goflag.Int("iterations", 5, "number of restore iterations")
	memMiB     = goflag.Int("mem", 0, "MiB of memory the workload touches (0 = sleep only)")
	touchPct   = goflag.Int("touch-pct", 100, "percentage of allocated pages to touch (1-100)")
	settleTime = goflag.Duration("settle", 2*time.Second, "time to wait after start for workload to settle")
	precreate  = goflag.Bool("precreate", false, "pre-create sandbox before timing (simulates sandbox pool)")
	background = goflag.Bool("background", false, "use background page loading")
	keepTmp    = goflag.Bool("keep-tmp", false, "keep temp dirs for debugging")
	compFlag   = goflag.String("compression", "none", "checkpoint compression: none|flate-best-speed")
)

// ---------------------------------------------------------------------------
// Alloc helper: a small static C program that mallocs N MiB, touches pages,
// then sleeps forever.
// ---------------------------------------------------------------------------

const allocSrc = `
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main(int argc, char **argv) {
    if (argc < 2) { pause(); return 0; }
    long mib = atol(argv[1]);
    if (mib <= 0) { pause(); return 0; }
    int touch_pct = 100;
    if (argc >= 3) {
        touch_pct = atoi(argv[2]);
        if (touch_pct <= 0) touch_pct = 1;
        if (touch_pct > 100) touch_pct = 100;
    }
    long bytes = mib * 1024L * 1024L;
    char *p = malloc(bytes);
    if (!p) { perror("malloc"); return 1; }
    long pgsz = sysconf(_SC_PAGESIZE);
    long total_pages = bytes / pgsz;
    long step = (touch_pct < 100) ? (100 / touch_pct) : 1;
    if (step < 1) step = 1;
    long touched = 0;
    for (long pg = 0; pg < total_pages; pg += step) {
        p[pg * pgsz] = (char)(pg >> 4);
        touched++;
    }
    fprintf(stderr, "allocated %ld MiB, touched %ld/%ld pages (%d%%)\n",
            mib, touched, total_pages, touch_pct);
    for (;;) pause();
    return 0;
}
`

func buildAllocBin(dir string) (string, error) {
	src := filepath.Join(dir, "alloc.c")
	bin := filepath.Join(dir, "alloc")
	if err := os.WriteFile(src, []byte(allocSrc), 0644); err != nil {
		return "", err
	}
	cmd := exec.Command("cc", "-static", "-O2", "-o", bin, src)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("compiling alloc helper: %w", err)
	}
	return bin, nil
}

// ---------------------------------------------------------------------------
// OCI spec construction
// ---------------------------------------------------------------------------

type bindMount struct{ src, dst string }

func makeSpec(args []string, binds []bindMount) *specs.Spec {
	mnts := []specs.Mount{
		{Type: "proc", Destination: "/proc", Source: "proc"},
		{Type: "tmpfs", Destination: "/tmp", Source: "tmpfs"},
	}
	for _, b := range binds {
		mnts = append(mnts, specs.Mount{
			Type:        "bind",
			Destination: b.dst,
			Source:      b.src,
			Options:     []string{"rbind", "ro"},
		})
	}
	return &specs.Spec{
		Root: &specs.Root{
			Path:     "/",
			Readonly: true,
		},
		Process: &specs.Process{
			Args:         args,
			Env:          []string{"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"},
			Cwd:          "/",
			Capabilities: specutils.AllCapabilities(),
		},
		Hostname:    "restorebench",
		Mounts:      mnts,
		Annotations: map[string]string{},
	}
}

// ---------------------------------------------------------------------------
// Config helpers
// ---------------------------------------------------------------------------

func makeConfig(rootDir string) *config.Config {
	testFlags := flag.NewFlagSet("bench", goflag.ContinueOnError)
	config.RegisterFlags(testFlags)
	conf, err := config.NewFromFlags(testFlags)
	if err != nil {
		log.Fatalf("creating config: %v", err)
	}
	conf.RootDir = rootDir
	conf.Network = config.NetworkNone
	conf.TestOnlyAllowRunAsCurrentUserWithoutChroot = true
	return conf
}

func writeSpec(dir string, spec *specs.Spec) error {
	b, err := json.MarshalIndent(spec, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "config.json"), b, 0644)
}

// ---------------------------------------------------------------------------
// Helpers
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

func randomID(prefix string) string {
	return fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano()%1e9)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	goflag.Parse()
	log.SetFlags(log.Lmicroseconds)

	if os.Getuid() != 0 {
		log.Fatal("must be run as root")
	}
	if *runscFlag == "" {
		log.Fatal("-runsc flag is required")
	}

	// Point gvisor at the runsc binary for sandbox fork/exec.
	abs, err := filepath.Abs(*runscFlag)
	if err != nil {
		log.Fatalf("resolving runsc path: %v", err)
	}
	specutils.ExePath = abs
	log.Printf("runsc: %s", abs)

	// ---- Temp dir ----
	tmpDir, err := os.MkdirTemp("", "restorebench-*")
	if err != nil {
		log.Fatalf("creating temp dir: %v", err)
	}
	if *keepTmp {
		log.Printf("temp dir: %s", tmpDir)
	} else {
		defer os.RemoveAll(tmpDir)
	}

	// ---- Build workload ----
	var containerArgs []string
	var binds []bindMount

	if *memMiB > 0 {
		allocBin, err := buildAllocBin(tmpDir)
		if err != nil {
			log.Fatalf("building alloc helper: %v", err)
		}
		wlDir := filepath.Join(tmpDir, "workload")
		os.MkdirAll(wlDir, 0755)
		data, _ := os.ReadFile(allocBin)
		os.WriteFile(filepath.Join(wlDir, "alloc"), data, 0755)
		binds = append(binds, bindMount{src: wlDir, dst: "/workload"})
		containerArgs = []string{"/workload/alloc", fmt.Sprintf("%d", *memMiB), fmt.Sprintf("%d", *touchPct)}
		log.Printf("workload: allocate %d MiB, touch %d%%", *memMiB, *touchPct)
	} else {
		containerArgs = []string{"sleep", "infinity"}
		log.Println("workload: sleep (no memory allocation)")
	}

	spec := makeSpec(containerArgs, binds)

	// ---- Config + bundle for source container ----
	srcRootDir := filepath.Join(tmpDir, "root-src")
	os.MkdirAll(srcRootDir, 0711)
	conf := makeConfig(srcRootDir)

	srcBundleDir := filepath.Join(tmpDir, "bundle-src")
	os.MkdirAll(srcBundleDir, 0755)
	if err := writeSpec(srcBundleDir, spec); err != nil {
		log.Fatalf("writing spec: %v", err)
	}

	// ---- Create + start source container ----
	log.Println("=== Creating source container ===")
	srcID := randomID("src")
	srcCont, err := container.New(conf, container.Args{
		ID:        srcID,
		Spec:      spec,
		BundleDir: srcBundleDir,
		Attached:  false,
	})
	if err != nil {
		log.Fatalf("container.New (source): %v", err)
	}
	defer srcCont.Destroy()

	if err := srcCont.Start(conf); err != nil {
		log.Fatalf("Start (source): %v", err)
	}

	settle := *settleTime
	if *memMiB >= 512 {
		extra := time.Duration(*memMiB/512) * time.Second
		if settle < 2*time.Second+extra {
			settle = 2*time.Second + extra
		}
	}
	log.Printf("waiting %v for workload to settle...", settle)
	time.Sleep(settle)

	// ---- Checkpoint ----
	log.Println("=== Checkpointing ===")
	ckptDir := filepath.Join(tmpDir, "checkpoint")
	os.MkdirAll(ckptDir, 0755)

	compression, _ := statefile.CompressionLevelFromString(*compFlag)
	ckptStart := time.Now()
	if err := srcCont.Checkpoint(conf, ckptDir, sandbox.CheckpointOpts{
		Compression: compression,
	}); err != nil {
		log.Fatalf("Checkpoint: %v", err)
	}
	log.Printf("checkpoint completed in %v", time.Since(ckptStart))

	entries, _ := os.ReadDir(ckptDir)
	var totalBytes int64
	for _, e := range entries {
		info, _ := e.Info()
		if info != nil {
			totalBytes += info.Size()
			log.Printf("  %s: %s", e.Name(), humanBytes(info.Size()))
		}
	}
	log.Printf("  total: %s", humanBytes(totalBytes))

	// ---- Measure restore ----
	mode := "sync"
	if *precreate && *background {
		mode = "precreate+background"
	} else if *precreate {
		mode = "precreate"
	} else if *background {
		mode = "background"
	}
	log.Printf("=== Measuring restore (%s, %d iterations) ===", mode, *iterations)

	durations := make([]time.Duration, 0, *iterations)
	for i := 0; i < *iterations; i++ {
		restoreID := randomID(fmt.Sprintf("dst-%d", i))
		restoreRootDir := filepath.Join(tmpDir, fmt.Sprintf("root-restore-%d", i))
		os.MkdirAll(restoreRootDir, 0711)
		restoreConf := makeConfig(restoreRootDir)

		restoreBundleDir := filepath.Join(tmpDir, fmt.Sprintf("bundle-restore-%d", i))
		os.MkdirAll(restoreBundleDir, 0755)
		if err := writeSpec(restoreBundleDir, spec); err != nil {
			log.Fatalf("writing restore spec: %v", err)
		}

		// Create the restore container (includes sandbox fork/exec).
		// In precreate mode this happens BEFORE timing starts.
		var restoreCont *container.Container
		if *precreate {
			c, err := container.New(restoreConf, container.Args{
				ID:        restoreID,
				Spec:      spec,
				BundleDir: restoreBundleDir,
				Attached:  false,
			})
			if err != nil {
				log.Fatalf("container.New (precreate): %v", err)
			}
			restoreCont = c
		}

		// ---- Timed section ----
		start := time.Now()

		if !*precreate {
			c, err := container.New(restoreConf, container.Args{
				ID:        restoreID,
				Spec:      spec,
				BundleDir: restoreBundleDir,
				Attached:  false,
			})
			if err != nil {
				log.Fatalf("container.New (restore): %v", err)
			}
			restoreCont = c
		}

		if err := restoreCont.Restore(restoreConf, ckptDir, false /* direct */, *background); err != nil {
			log.Fatalf("Restore: %v", err)
		}

		d := time.Since(start)
		// ---- End timed section ----

		durations = append(durations, d)
		log.Printf("  restore %d/%d: %v", i+1, *iterations, d)

		restoreCont.Destroy()
	}

	// ---- Summary ----
	fmt.Println()
	fmt.Println("=== Restore Benchmark Results ===")
	fmt.Printf("Platform:       systrap\n")
	fmt.Printf("Compression:    %s\n", *compFlag)
	fmt.Printf("Memory:         %d MiB\n", *memMiB)
	fmt.Printf("Checkpoint:     %s\n", humanBytes(totalBytes))
	fmt.Printf("Iterations:     %d\n", *iterations)
	fmt.Printf("Mode:           %s\n", mode)
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
