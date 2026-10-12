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
	"errors"
	"fmt"
	"slices"
	"testing"
	"time"

	"gvisor.dev/gvisor/pkg/sentry/devices/nvproxy"
)

func TestCudaCheckpointDeviceMap(t *testing.T) {
	const (
		uuidA = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
		uuidB = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
		uuidC = "GPU-cccccccc-cccc-cccc-cccc-cccccccccccc"
	)
	id := func(minor uint32, uuid string) nvproxy.DeviceRemapID {
		return nvproxy.DeviceRemapID{Minor: minor, DeviceInstance: minor, UUID: uuid}
	}
	for _, tc := range []struct {
		name    string
		oldIDs  []nvproxy.DeviceRemapID
		newIDs  []nvproxy.DeviceRemapID
		want    string
		wantErr bool
	}{
		{
			name:   "identity",
			oldIDs: []nvproxy.DeviceRemapID{id(0, uuidA), id(1, uuidB)},
			newIDs: []nvproxy.DeviceRemapID{id(0, uuidA), id(1, uuidB)},
			want:   "",
		},
		{
			name:   "single device",
			oldIDs: []nvproxy.DeviceRemapID{id(0, uuidA)},
			newIDs: []nvproxy.DeviceRemapID{id(1, uuidB)},
			want:   uuidA + "=" + uuidB,
		},
		{
			// Only one device moves, but all checkpointed devices must be
			// listed, ordered by old minor.
			name:   "partial remap",
			oldIDs: []nvproxy.DeviceRemapID{id(1, uuidB), id(0, uuidA)},
			newIDs: []nvproxy.DeviceRemapID{id(2, uuidC), id(0, uuidA)},
			want:   uuidA + "=" + uuidA + "," + uuidB + "=" + uuidC,
		},
		{
			name:    "missing UUID",
			oldIDs:  []nvproxy.DeviceRemapID{id(0, uuidA)},
			newIDs:  []nvproxy.DeviceRemapID{id(1, "")},
			wantErr: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dr, err := nvproxy.MakeDeviceRemapping(tc.oldIDs, tc.newIDs)
			if err != nil {
				t.Fatalf("MakeDeviceRemapping failed: %v", err)
			}
			got, err := cudaCheckpointDeviceMap(dr)
			if (err != nil) != tc.wantErr {
				t.Fatalf("cudaCheckpointDeviceMap() error = %v, wantErr %v", err, tc.wantErr)
			}
			if got != tc.want {
				t.Errorf("cudaCheckpointDeviceMap() = %q, want %q", got, tc.want)
			}
		})
	}
}

// fakeCuda is a fake cuda-checkpoint over processes 0..n-1 for
// checkpointLocked. Each process has a state; an action listed in fail for a
// process fails there (leaving the state given by failState), and every
// call is recorded.
type fakeCuda struct {
	state     map[int]string
	fail      map[string]map[int]bool
	failState map[int]string
	stateErr  map[int]bool
	calls     []string
}

func (f *fakeCuda) actions() cudaActions[int] {
	return cudaActions[int]{
		run: func(procs []int, opArgs []string, parallel, stopOnError bool, timeout time.Duration) ([]int, error) {
			action := opArgs[len(opArgs)-1]
			f.calls = append(f.calls, fmt.Sprintf("%s %v", action, procs))
			var ok []int
			var errs []error
			for _, p := range procs {
				if f.fail[action][p] {
					if s, set := f.failState[p]; set {
						f.state[p] = s
					}
					errs = append(errs, fmt.Errorf("%s on %d failed", action, p))
					if stopOnError && !parallel {
						break
					}
					continue
				}
				want := map[string]string{"checkpoint": "locked", "restore": "checkpointed", "unlock": "locked"}[action]
				if f.state[p] != want {
					errs = append(errs, fmt.Errorf("%s on %d in state %s", action, p, f.state[p]))
					continue
				}
				f.state[p] = map[string]string{"checkpoint": "checkpointed", "restore": "locked", "unlock": "running"}[action]
				ok = append(ok, p)
			}
			return ok, errors.Join(errs...)
		},
		state: func(p int) (string, error) {
			f.calls = append(f.calls, fmt.Sprintf("get-state %d", p))
			if f.stateErr[p] {
				return "", fmt.Errorf("get-state on %d failed", p)
			}
			return f.state[p], nil
		},
		name: func(p int) string { return fmt.Sprintf("process %d", p) },
	}
}

// TestCheckpointLocked covers how a failed checkpoint is undone: stop at the
// first failure, restore only the processes that are checkpointed, unlock
// every process, and report what cannot be undone as errCudaUnrecoverable.
func TestCheckpointLocked(t *testing.T) {
	for _, tc := range []struct {
		name          string
		fail          map[string]map[int]bool
		failState     map[int]string
		stateErr      map[int]bool
		wantErr       bool
		unrecoverable bool
		wantCalls     []string
		wantState     string // of every process
	}{
		{
			name:      "success",
			wantCalls: []string{"checkpoint [0 1 2]"},
			wantState: "checkpointed",
		},
		{
			name:    "second fails, still locked",
			fail:    map[string]map[int]bool{"checkpoint": {1: true}},
			wantErr: true,
			wantCalls: []string{
				"checkpoint [0 1 2]",
				"get-state 1", "get-state 2",
				"restore [0]",
				"unlock [0 1 2]",
			},
			wantState: "running",
		},
		{
			name:      "second fails after checkpointing",
			fail:      map[string]map[int]bool{"checkpoint": {1: true}},
			failState: map[int]string{1: "checkpointed"},
			wantErr:   true,
			wantCalls: []string{
				"checkpoint [0 1 2]",
				"get-state 1", "get-state 2",
				"restore [0 1]",
				"unlock [0 1 2]",
			},
			wantState: "running",
		},
		{
			name:      "first fails",
			fail:      map[string]map[int]bool{"checkpoint": {0: true}},
			wantErr:   true,
			wantCalls: []string{"checkpoint [0 1 2]", "get-state 0", "get-state 1", "get-state 2", "unlock [0 1 2]"},
			wantState: "running",
		},
		{
			name:          "failed process in an unknown state",
			fail:          map[string]map[int]bool{"checkpoint": {1: true}},
			failState:     map[int]string{1: "failed"},
			wantErr:       true,
			unrecoverable: true,
			wantCalls:     []string{"checkpoint [0 1 2]", "get-state 1"},
		},
		{
			name:          "state unknown",
			fail:          map[string]map[int]bool{"checkpoint": {1: true}},
			stateErr:      map[int]bool{1: true},
			wantErr:       true,
			unrecoverable: true,
			wantCalls:     []string{"checkpoint [0 1 2]", "get-state 1"},
		},
		{
			name:          "restore fails",
			fail:          map[string]map[int]bool{"checkpoint": {2: true}, "restore": {0: true}},
			wantErr:       true,
			unrecoverable: true,
			wantCalls:     []string{"checkpoint [0 1 2]", "get-state 2", "restore [0 1]"},
		},
		{
			name:          "unlock fails",
			fail:          map[string]map[int]bool{"checkpoint": {2: true}, "unlock": {1: true}},
			wantErr:       true,
			unrecoverable: true,
			wantCalls:     []string{"checkpoint [0 1 2]", "get-state 2", "restore [0 1]", "unlock [0 1 2]", "get-state 1"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := &fakeCuda{
				state:     map[int]string{0: "locked", 1: "locked", 2: "locked"},
				fail:      tc.fail,
				failState: tc.failState,
				stateErr:  tc.stateErr,
			}
			err := checkpointLocked(f.actions(), []int{0, 1, 2}, true /* sequential */, time.Minute)
			if (err != nil) != tc.wantErr {
				t.Fatalf("checkpointLocked() = %v, want error: %v", err, tc.wantErr)
			}
			if got := errors.Is(err, errCudaUnrecoverable); got != tc.unrecoverable {
				t.Errorf("checkpointLocked() = %v; unrecoverable = %v, want %v", err, got, tc.unrecoverable)
			}
			if !slices.Equal(f.calls, tc.wantCalls) {
				t.Errorf("calls = %q, want %q", f.calls, tc.wantCalls)
			}
			if tc.wantState != "" {
				for p, s := range f.state {
					if s != tc.wantState {
						t.Errorf("process %d is %s, want %s", p, s, tc.wantState)
					}
				}
			}
		})
	}
}
