// Copyright 2024 The gVisor Authors.
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

//go:build !cgo

package boot

import "gvisor.dev/gvisor/pkg/log"

// gpuRestoreCgoAvailable is false when cgo is disabled.
const gpuRestoreCgoAvailable = false

// RestoreGPUCheckpointInProcess is a no-op when cgo is disabled.
// GPU checkpoint/restore requires cgo to dlopen libcuda.so and call
// cuCheckpointProcessRestore directly in the sentry process.
func RestoreGPUCheckpointInProcess() error {
	log.Infof("gpu-restore: cgo not available, skipping in-process CUDA restore")
	return nil
}
