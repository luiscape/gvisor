// Copyright 2025 The gVisor Authors.
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

package stateio

import (
	"golang.org/x/sys/unix"

	"gvisor.dev/gvisor/pkg/hostarch"
	"gvisor.dev/gvisor/pkg/log"
)

const (
	// pagesFileFDDefaultMaxIOBytes is the default maximum number of bytes
	// per I/O operation. Increased to 2MB to reduce syscall overhead for
	// large checkpoint files.
	pagesFileFDDefaultMaxIOBytes = 2 << 20
	// pagesFileFDDefaultMaxParallel is the default maximum number of
	// parallel I/O operations. Reduced from 128 to 32 to avoid excessive
	// memory usage from larger I/O buffers while still saturating storage.
	pagesFileFDDefaultMaxParallel = 32
)

// NewPagesFileFDReader returns a FDReader that reads a pages file from the
// given host file descriptor. It takes ownership of the file descriptor.
//
// Preconditions:
// - hostarch.PageSize <= maxReadBytes <= linux.MAX_RW_COUNT.
// - maxParallel > 0.
func NewPagesFileFDReader(fd int32, maxReadBytes uint64, maxParallel int) *FDReader {
	// Hint to the kernel that we will read this file sequentially, which
	// enables readahead and improves restore performance.
	if err := unix.Fadvise(int(fd), 0, 0, unix.FADV_SEQUENTIAL); err != nil {
		log.Warningf("fadvise(FADV_SEQUENTIAL) on pages file FD %d failed: %v", fd, err)
	}
	// Provision one range per page of maxReadBytes, since this is the maximum
	// number of ranges that async page loading will use.
	return NewFDReader(fd, maxReadBytes, int(maxReadBytes/hostarch.PageSize), maxParallel)
}

// NewPagesFileFDReaderDefault returns a FDReader that reads a pages file from
// the given host file descriptor, using defaults for MaxReadBytes and
// MaxParallel. It takes ownership of the file descriptor.
func NewPagesFileFDReaderDefault(fd int32) *FDReader {
	return NewPagesFileFDReader(fd, pagesFileFDDefaultMaxIOBytes, pagesFileFDDefaultMaxParallel)
}

// NewPagesFileFDWriter returns a FDWriter that writes a pages file to the
// given host file descriptor. It takes ownership of the file descriptor.
//
// Preconditions:
// - hostarch.PageSize <= maxWriteBytes <= linux.MAX_RW_COUNT.
// - maxParallel > 0.
func NewPagesFileFDWriter(fd int32, maxWriteBytes uint64, maxParallel int) *FDWriter {
	// Provision one range per page of maxWriteBytes, since this is the maximum
	// number of ranges that async page saving will use.
	return NewFDWriter(fd, maxWriteBytes, int(maxWriteBytes/hostarch.PageSize), maxParallel)
}

// NewPagesFileFDWriterDefault returns a FDWriter that writes to the given host
// file descriptor, using defaults for MaxWriteBytes and MaxParallel. It takes
// ownership of the file descriptor.
func NewPagesFileFDWriterDefault(fd int32) *FDWriter {
	return NewPagesFileFDWriter(fd, pagesFileFDDefaultMaxIOBytes, pagesFileFDDefaultMaxParallel)
}
