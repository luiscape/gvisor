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

package pgalloc

import (
	"bytes"
	"fmt"
	"os"
	"reflect"
	"testing"

	"golang.org/x/sys/unix"
	"gvisor.dev/gvisor/pkg/context"
	"gvisor.dev/gvisor/pkg/hostarch"
	"gvisor.dev/gvisor/pkg/sentry/memmap"
	"gvisor.dev/gvisor/pkg/sentry/state/stateio"
	"gvisor.dev/gvisor/pkg/sentry/usage"
)

func testNewMemoryFile(t *testing.T) *MemoryFile {
	t.Helper()
	memfd, err := unix.MemfdCreate("test-memory-file", 0)
	if err != nil {
		t.Fatalf("MemfdCreate failed: %v", err)
	}
	f, err := NewMemoryFile(os.NewFile(uintptr(memfd), "test-memory-file"), MemoryFileOpts{
		DisableMemoryAccounting: true,
	})
	if err != nil {
		t.Fatalf("NewMemoryFile failed: %v", err)
	}
	return f
}

// testFillPages fills each page in fr with a deterministic pattern derived
// from its MemoryFile offset.
func testFillPages(t *testing.T, f *MemoryFile, fr memmap.FileRange) {
	t.Helper()
	off := fr.Start
	f.forEachMappingSlice(fr, func(bs []byte) {
		for i := 0; i < len(bs); i += hostarch.PageSize {
			pattern := fmt.Sprintf("MemoryFile page at offset %#x|", off+uint64(i))
			pg := bs[i : i+hostarch.PageSize]
			for len(pg) >= len(pattern) {
				pg = pg[copy(pg, pattern):]
			}
			copy(pg, pattern)
		}
		off += uint64(len(bs))
	})
}

func testCheckPages(t *testing.T, f *MemoryFile, fr memmap.FileRange) {
	t.Helper()
	off := fr.Start
	f.forEachMappingSlice(fr, func(bs []byte) {
		for i := 0; i < len(bs); i += hostarch.PageSize {
			pattern := fmt.Sprintf("MemoryFile page at offset %#x|", off+uint64(i))
			pg := bs[i : i+hostarch.PageSize]
			want := make([]byte, 0, hostarch.PageSize)
			for len(want) < hostarch.PageSize {
				want = append(want, pattern...)
			}
			want = want[:hostarch.PageSize]
			if !bytes.Equal(pg, want) {
				t.Errorf("page at offset %#x corrupted after restore: got %q..., want %q...", off+uint64(i), pg[:len(pattern)], pattern)
				return
			}
		}
		off += uint64(len(bs))
	})
}

// testSaveToPagesFile saves f to a temporary pages file, returning the
// metadata stream and the pages file.
func testSaveToPagesFile(t *testing.T, f *MemoryFile) (*bytes.Buffer, *os.File) {
	t.Helper()
	pagesFile, err := os.CreateTemp(t.TempDir(), "pages")
	if err != nil {
		t.Fatalf("CreateTemp failed: %v", err)
	}
	wfd, err := unix.Dup(int(pagesFile.Fd()))
	if err != nil {
		t.Fatalf("Dup failed: %v", err)
	}
	saveDone := make(chan error, 1)
	apfs, err := StartAsyncPagesFileSave(stateio.NewPagesFileFDWriterDefault(int32(wfd)), func(err error) {
		saveDone <- err
	})
	if err != nil {
		t.Fatalf("StartAsyncPagesFileSave failed: %v", err)
	}
	var metadata bytes.Buffer
	f.MarkSavable()
	if err := f.SaveTo(context.Background(), &metadata, &SaveOpts{PagesFile: apfs}); err != nil {
		t.Fatalf("SaveTo failed: %v", err)
	}
	apfs.MemoryFilesDone()
	if err := <-saveDone; err != nil {
		t.Fatalf("async page saving failed: %v", err)
	}
	return &metadata, pagesFile
}

// testLoadFromPagesFile loads a MemoryFile from the given metadata stream and
// pages file, blocking until all pages have been loaded.
func testLoadFromPagesFile(t *testing.T, f *MemoryFile, metadata *bytes.Buffer, pagesFile *os.File, opts AsyncPagesFileLoadOpts) {
	t.Helper()
	rfd, err := unix.Dup(int(pagesFile.Fd()))
	if err != nil {
		t.Fatalf("Dup failed: %v", err)
	}
	apfl, err := StartAsyncPagesFileLoad(stateio.NewPagesFileFDReaderDefault(int32(rfd)), nil, nil, opts)
	if err != nil {
		t.Fatalf("StartAsyncPagesFileLoad failed: %v", err)
	}
	if err := f.LoadFrom(context.Background(), metadata, &LoadOpts{PagesFile: apfl}); err != nil {
		t.Fatalf("LoadFrom failed: %v", err)
	}
	apfl.MemoryFilesDone()
	if err := f.AwaitLoadAll(); err != nil {
		t.Fatalf("AwaitLoadAll failed: %v", err)
	}
}

// testAllocateScattered allocates n page-sized ranges, each separated by a
// page-sized hole (which is deallocated and hence not saved).
func testAllocateScattered(t *testing.T, f *MemoryFile, n int) []memmap.FileRange {
	t.Helper()
	frs := make([]memmap.FileRange, 0, n)
	var holes []memmap.FileRange
	for i := 0; i < n; i++ {
		fr, err := f.Allocate(hostarch.PageSize, AllocOpts{Kind: usage.Anonymous, Dir: BottomUp})
		if err != nil {
			t.Fatalf("Allocate failed: %v", err)
		}
		frs = append(frs, fr)
		hole, err := f.Allocate(hostarch.PageSize, AllocOpts{Kind: usage.Anonymous, Dir: BottomUp})
		if err != nil {
			t.Fatalf("Allocate failed: %v", err)
		}
		holes = append(holes, hole)
		testFillPages(t, f, fr)
	}
	for _, hole := range holes {
		f.DecRef(hole)
	}
	return frs
}

// TestSaveRestoreReorderedPagesFile checks that a MemoryFile whose pages file
// was written in traced access order (rather than MemoryFile offset order)
// restores with correct contents.
func TestSaveRestoreReorderedPagesFile(t *testing.T) {
	f := testNewMemoryFile(t)
	defer f.Destroy()

	const numPages = 64
	frs := testAllocateScattered(t, f, numPages)

	// Synthesize an access trace covering a subset of the allocated pages in
	// a scrambled order, as if recorded by a previous traced restore. (13 is
	// coprime to numPages, so the trace entries are distinct, as required of
	// a genuine trace.) Untraced pages should be emitted after traced pages,
	// in offset order.
	for i := 0; i < numPages/2; i++ {
		j := (i*13 + 5) % numPages
		f.restoreAccessTrace = append(f.restoreAccessTrace, frs[j])
	}

	metadata, pagesFile := testSaveToPagesFile(t, f)
	defer pagesFile.Close()

	// The pages file must not be in offset order: the first page in the
	// pages file should be the first traced page.
	firstTraced := f.restoreAccessTrace[0]
	buf := make([]byte, hostarch.PageSize)
	if _, err := pagesFile.ReadAt(buf, 0); err != nil {
		t.Fatalf("reading pages file failed: %v", err)
	}
	wantPattern := fmt.Sprintf("MemoryFile page at offset %#x|", firstTraced.Start)
	if !bytes.HasPrefix(buf, []byte(wantPattern)) {
		t.Errorf("pages file does not begin with first traced page: got %q, want %q", buf[:len(wantPattern)], wantPattern)
	}

	f2 := testNewMemoryFile(t)
	defer f2.Destroy()
	testLoadFromPagesFile(t, f2, metadata, pagesFile, AsyncPagesFileLoadOpts{})
	for _, fr := range frs {
		testCheckPages(t, f2, fr)
	}
}

// TestSaveInvalidAccessTrace checks that an access trace containing
// overlapping ranges (which should be impossible, but would corrupt the
// checkpoint if used) causes SaveTo to fall back to an offset-ordered pages
// file that restores correctly.
func TestSaveInvalidAccessTrace(t *testing.T) {
	f := testNewMemoryFile(t)
	defer f.Destroy()

	const numPages = 8
	frs := testAllocateScattered(t, f, numPages)
	f.restoreAccessTrace = []memmap.FileRange{frs[3], frs[1], frs[3]}

	metadata, pagesFile := testSaveToPagesFile(t, f)
	defer pagesFile.Close()

	// The invalid trace must have been ignored: the pages file should be in
	// offset order, starting with the first allocated page.
	buf := make([]byte, hostarch.PageSize)
	if _, err := pagesFile.ReadAt(buf, 0); err != nil {
		t.Fatalf("reading pages file failed: %v", err)
	}
	wantPattern := fmt.Sprintf("MemoryFile page at offset %#x|", frs[0].Start)
	if !bytes.HasPrefix(buf, []byte(wantPattern)) {
		t.Errorf("pages file does not begin with first allocated page: got %q, want %q", buf[:len(wantPattern)], wantPattern)
	}

	f2 := testNewMemoryFile(t)
	defer f2.Destroy()
	testLoadFromPagesFile(t, f2, metadata, pagesFile, AsyncPagesFileLoadOpts{})
	for _, fr := range frs {
		testCheckPages(t, f2, fr)
	}
}

// TestLoadReorderedBackground checks that a reordered checkpoint restores
// correctly when pages are loaded entirely by the background loader (in
// pages file order), without any demand loads.
func TestLoadReorderedBackground(t *testing.T) {
	f := testNewMemoryFile(t)
	defer f.Destroy()

	const numPages = 64
	frs := testAllocateScattered(t, f, numPages)
	for i := 0; i < numPages; i++ {
		j := (i*29 + 17) % numPages
		f.restoreAccessTrace = append(f.restoreAccessTrace, frs[j])
	}

	metadata, pagesFile := testSaveToPagesFile(t, f)
	defer pagesFile.Close()

	f2 := testNewMemoryFile(t)
	defer f2.Destroy()
	rfd, err := unix.Dup(int(pagesFile.Fd()))
	if err != nil {
		t.Fatalf("Dup failed: %v", err)
	}
	loadDone := make(chan error, 1)
	apfl, err := StartAsyncPagesFileLoad(stateio.NewPagesFileFDReaderDefault(int32(rfd)), func(err error) {
		loadDone <- err
	}, nil, AsyncPagesFileLoadOpts{})
	if err != nil {
		t.Fatalf("StartAsyncPagesFileLoad failed: %v", err)
	}
	if err := f2.LoadFrom(context.Background(), metadata, &LoadOpts{PagesFile: apfl}); err != nil {
		t.Fatalf("LoadFrom failed: %v", err)
	}
	apfl.MemoryFilesDone()
	// Wait for background loading to complete without awaiting any pages, so
	// that all loads take the loadOrder-directed background path.
	if err := <-loadDone; err != nil {
		t.Fatalf("async page loading failed: %v", err)
	}
	if f2.IsAsyncLoading() {
		t.Errorf("async page loading still in progress after doneCallback")
	}
	for _, fr := range frs {
		testCheckPages(t, f2, fr)
	}
}

// TestSaveRestoreTraceAccess checks the full profiling flow: a traced restore
// records the order in which pages are demanded; a subsequent save writes the
// pages file in that order; and a subsequent restore of the re-saved
// checkpoint has correct contents.
func TestSaveRestoreTraceAccess(t *testing.T) {
	f := testNewMemoryFile(t)
	defer f.Destroy()

	const numPages = 64
	frs := testAllocateScattered(t, f, numPages)

	// Save without any trace; the pages file is in offset order.
	metadata, pagesFile := testSaveToPagesFile(t, f)
	defer pagesFile.Close()

	// Restore with access tracing enabled, and demand pages in a scrambled
	// order.
	f2 := testNewMemoryFile(t)
	defer f2.Destroy()
	rfd, err := unix.Dup(int(pagesFile.Fd()))
	if err != nil {
		t.Fatalf("Dup failed: %v", err)
	}
	apfl, err := StartAsyncPagesFileLoad(stateio.NewPagesFileFDReaderDefault(int32(rfd)), nil, nil, AsyncPagesFileLoadOpts{TraceAccess: true})
	if err != nil {
		t.Fatalf("StartAsyncPagesFileLoad failed: %v", err)
	}
	if err := f2.LoadFrom(context.Background(), metadata, &LoadOpts{PagesFile: apfl}); err != nil {
		t.Fatalf("LoadFrom failed: %v", err)
	}
	apfl.MemoryFilesDone()
	var accessOrder []memmap.FileRange
	for i := 0; i < numPages; i++ {
		j := (i*29 + 17) % numPages
		accessOrder = append(accessOrder, frs[j])
	}
	amfl := f2.asyncPageLoad.Load()
	if amfl == nil {
		t.Fatalf("no async page loading in progress")
	}
	for _, fr := range accessOrder {
		if err := amfl.awaitLoad(fr); err != nil {
			t.Fatalf("awaitLoad(%v) failed: %v", fr, err)
		}
		testCheckPages(t, f2, fr)
	}
	if err := f2.AwaitLoadAll(); err != nil {
		t.Fatalf("AwaitLoadAll failed: %v", err)
	}

	// The recorded trace should match the demanded access order.
	if got := f2.restoreAccessTrace[:numPages]; !reflect.DeepEqual(got, accessOrder) {
		t.Errorf("recorded access trace %v does not match access order %v", got, accessOrder)
	}

	// Save f2; the pages file should now be in access order.
	metadata2, pagesFile2 := testSaveToPagesFile(t, f2)
	defer pagesFile2.Close()
	buf := make([]byte, hostarch.PageSize)
	for i, fr := range accessOrder {
		if _, err := pagesFile2.ReadAt(buf, int64(i)*hostarch.PageSize); err != nil {
			t.Fatalf("reading pages file failed: %v", err)
		}
		wantPattern := fmt.Sprintf("MemoryFile page at offset %#x|", fr.Start)
		if !bytes.HasPrefix(buf, []byte(wantPattern)) {
			t.Fatalf("pages file page %d: got %q, want %q", i, buf[:len(wantPattern)], wantPattern)
		}
	}

	// Restore the re-saved checkpoint and check contents.
	f3 := testNewMemoryFile(t)
	defer f3.Destroy()
	testLoadFromPagesFile(t, f3, metadata2, pagesFile2, AsyncPagesFileLoadOpts{})
	for _, fr := range frs {
		testCheckPages(t, f3, fr)
	}
}
