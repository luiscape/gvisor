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

// The workload binary emulates a snapshottable serverless function for
// checkpoint/restore benchmarking.
//
// It allocates TOTAL_MB of memory as separate REGION_KB-sized mappings
// (emulating the fragmented address spaces of real language runtimes, and
// bounding the granularity at which gVisor demand-maps and demand-loads
// memory after restore) and fills every page (so that all of it is captured
// in a checkpoint). It then repeatedly touches a working set of HOT_MB worth
// of regions, chosen pseudo-randomly from across the whole allocation and
// traversed in a fixed scrambled order (emulating a request whose page
// access order is unrelated to virtual address order). Each touch pass
// appends a "PASS" line, with its duration, to the file named by OUT.
//
// Between passes it sleeps for SLEEP_MS, polling the file named by TRIGGER
// (on a bind mount shared with the host); when the trigger file's contents
// change, it ends the sleep early and immediately runs the next pass. The
// benchmark harness writes a fresh token to the trigger file just before
// restoring a checkpoint of this workload, so the first post-restore pass
// begins promptly.
package main

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"syscall"
	"time"
)

const pageSize = 4096

func envInt(name string, def int) int {
	if v := os.Getenv(name); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			panic(fmt.Sprintf("invalid %s: %v", name, err))
		}
		return n
	}
	return def
}

func envStr(name, def string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return def
}

func main() {
	totalMB := envInt("TOTAL_MB", 4096)
	hotMB := envInt("HOT_MB", 1024)
	regionKB := envInt("REGION_KB", 256)
	sleepMS := envInt("SLEEP_MS", 200)
	outPath := envStr("OUT", "/out/bench.log")
	triggerPath := envStr("TRIGGER", "/out/trigger")

	out, err := os.OpenFile(outPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		panic(err)
	}
	emit := func(format string, args ...any) {
		line := fmt.Sprintf(format+"\n", args...)
		if _, err := out.WriteString(line); err != nil {
			panic(err)
		}
		if err := out.Sync(); err != nil {
			panic(err)
		}
		fmt.Print(line)
	}

	regionBytes := regionKB << 10
	regionPages := regionBytes / pageSize
	regions := (totalMB << 20) / regionBytes
	hotRegions := (hotMB << 20) / regionBytes
	if hotRegions > regions {
		panic("HOT_MB > TOTAL_MB")
	}

	mems := make([][]byte, regions)
	for r := range mems {
		m, err := syscall.Mmap(-1, 0, regionBytes, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANONYMOUS|syscall.MAP_PRIVATE)
		if err != nil {
			panic(err)
		}
		mems[r] = m
	}

	// Fill every page with non-zero data so that the entire allocation is
	// captured in the checkpoint's pages file.
	fillStart := time.Now()
	for r, m := range mems {
		for base := 0; base < regionBytes; base += pageSize {
			for off := 0; off < 64; off++ {
				m[base+off*64] = byte(r + base + off | 1)
			}
		}
	}
	emit("INIT_DONE total_mb=%d hot_mb=%d regions=%d region_kb=%d fill_ms=%d", totalMB, hotMB, regions, regionKB, time.Since(fillStart).Milliseconds())

	// Choose the working set: a pseudo-random (but deterministic) subset of
	// regions scattered across the entire allocation, traversed in scrambled
	// order, so that access order is unrelated to address order.
	rng := rand.New(rand.NewSource(42))
	hot := rng.Perm(regions)[:hotRegions]

	readTrigger := func() string {
		b, err := os.ReadFile(triggerPath)
		if err != nil {
			return ""
		}
		return string(b)
	}
	lastToken := readTrigger()

	sum := uint64(0)
	for i := 1; ; i++ {
		passStart := time.Now()
		for _, r := range hot {
			m := mems[r]
			for pg := 0; pg < regionPages; pg++ {
				base := pg * pageSize
				sum += uint64(m[base]) // read fault
				m[base+1] = byte(i)    // write
			}
		}
		emit("PASS %d dur_ms=%d wall_ms=%d sum=%d", i, time.Since(passStart).Milliseconds(), time.Now().UnixMilli(), sum)

		// Sleep in small increments, polling the trigger file; if its
		// contents change (the harness writes a fresh token just before
		// restoring a checkpoint), start the next pass immediately.
		deadline := time.Now().Add(time.Duration(sleepMS) * time.Millisecond)
		for {
			time.Sleep(20 * time.Millisecond)
			if token := readTrigger(); token != lastToken {
				lastToken = token
				emit("TRIGGERED token=%q wall_ms=%d", token, time.Now().UnixMilli())
				break
			}
			if time.Now().After(deadline) {
				break
			}
		}
	}
}
