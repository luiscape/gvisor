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

package mcshimbin

import (
	"bytes"
	"debug/elf"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// glibcFloor is the newest glibc symbol version the embedded interposer may
// require, as a (major, minor) pair. The interposer is preloaded into every
// process of a container, and a library that needs a symbol version the
// image's glibc lacks makes the dynamic linker exit the process with status
// 1, so nothing in the image can start. 2.17 is CentOS 7 / Amazon Linux 2,
// and the base version of aarch64 glibc. See tools/mcshim/README.md.
var glibcFloor = [2]int{2, 17}

// glibcVersion parses "GLIBC_2.17" into (2, 17). Three-part versions such as
// GLIBC_2.2.5 keep their first two parts, which is enough to compare them
// against the floor.
func glibcVersion(v string) ([2]int, bool) {
	parts := strings.Split(strings.TrimPrefix(v, "GLIBC_"), ".")
	if !strings.HasPrefix(v, "GLIBC_") || len(parts) < 2 {
		return [2]int{}, false
	}
	var out [2]int
	for i := range out {
		n, err := strconv.Atoi(parts[i])
		if err != nil {
			return [2]int{}, false
		}
		out[i] = n
	}
	return out, true
}

func above(a, b [2]int) bool {
	return a[0] > b[0] || (a[0] == b[0] && a[1] > b[1])
}

// TestGlibcFloor checks that the embedded interposer needs no glibc symbol
// version above glibcFloor, and that it still depends on the libraries that
// defined those symbols before glibc 2.34.
func TestGlibcFloor(t *testing.T) {
	f, err := elf.NewFile(bytes.NewReader(Interposer()))
	if err != nil {
		t.Fatalf("parsing the embedded mcshim.so: %v", err)
	}
	syms, err := f.ImportedSymbols()
	if err != nil {
		t.Fatalf("reading imported symbols: %v", err)
	}
	var tooNew []string
	for _, s := range syms {
		v, ok := glibcVersion(s.Version)
		if !ok {
			continue
		}
		if above(v, glibcFloor) {
			tooNew = append(tooNew, fmt.Sprintf("%s@%s", s.Name, s.Version))
		}
	}
	sort.Strings(tooNew)
	if len(tooNew) > 0 {
		t.Errorf("mcshim.so needs symbol versions above GLIBC_%d.%d, so it cannot load in images with an older glibc: %s",
			glibcFloor[0], glibcFloor[1], strings.Join(tooNew, " "))
	}

	// Before glibc 2.34, dlopen and pthread_create live in libdl.so.2 and
	// libpthread.so.0; glibc >= 2.34 ships both as empty stubs, so always
	// depending on them is harmless.
	needed, err := f.ImportedLibraries()
	if err != nil {
		t.Fatalf("reading NEEDED entries: %v", err)
	}
	for _, want := range []string{"libdl.so.2", "libpthread.so.0"} {
		found := false
		for _, lib := range needed {
			found = found || lib == want
		}
		if !found {
			t.Errorf("mcshim.so does not depend on %s (NEEDED: %v); images with glibc < 2.34 need it for the dl*/pthread_* symbols", want, needed)
		}
	}
}
