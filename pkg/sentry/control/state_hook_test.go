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
	"fmt"
	"slices"
	"strings"
	"testing"
)

// fakeHook records the hook sequences' effects over containers 0..n-1. status
// maps "<phase> <container>" to the exit status of that run (0 if absent);
// broken lists runs that cannot be run at all.
type fakeHook struct {
	status map[string]uint32
	broken map[string]bool
	calls  []string
	locked bool
}

func (f *fakeHook) ops() hookOps[int] {
	return hookOps[int]{
		run: func(target int, args ...string) (uint32, string, error) {
			call := fmt.Sprintf("%s %d", strings.Join(args, " "), target)
			f.calls = append(f.calls, call)
			key := fmt.Sprintf("%s %d", args[0], target)
			if f.broken[key] {
				return 0, "", fmt.Errorf("%w: exec failed", errHookNotStarted)
			}
			if args[0] == "resume" && f.locked {
				// The rebuild would fault forever with page faults held.
				return 99, "resume with page faults held", nil
			}
			return f.status[key], "hook output", nil
		},
		lockFaults: func() {
			f.locked = true
			f.calls = append(f.calls, "lock faults")
		},
		unlockFaults: func() {
			f.locked = false
			f.calls = append(f.calls, "unlock faults")
		},
		name: func(target int) string { return fmt.Sprintf("container %d", target) },
	}
}

func TestCheckpointWithHook(t *testing.T) {
	for _, tc := range []struct {
		name       string
		status     map[string]uint32
		broken     map[string]bool
		wantErr    string // substring; "" for success
		wantCalls  []string
		wantLocked bool
	}{
		{
			name:       "success",
			wantCalls:  []string{"lock faults", "checkpoint 0", "checkpoint 1"},
			wantLocked: true,
		},
		{
			name:    "recoverable failure in the first container",
			status:  map[string]uint32{"checkpoint 0": 1},
			wantErr: "checkpoint in container 0 failed with exit status 1",
			wantCalls: []string{
				"lock faults", "checkpoint 0", "unlock faults", "resume 0",
			},
		},
		{
			name:    "recoverable failure after a checkpointed container",
			status:  map[string]uint32{"checkpoint 1": 1},
			wantErr: "exit status 1",
			wantCalls: []string{
				"lock faults", "checkpoint 0", "checkpoint 1",
				"restore 0", "unlock faults", "resume 0", "resume 1",
			},
		},
		{
			name:    "unrecoverable failure is not resumed",
			status:  map[string]uint32{"checkpoint 1": 2},
			wantErr: "could not be undone, and the application in container 1 may be left blocked",
			wantCalls: []string{
				"lock faults", "checkpoint 0", "checkpoint 1",
				"restore 0", "unlock faults", "resume 0",
			},
		},
		{
			name:    "hook could not start",
			broken:  map[string]bool{"checkpoint 1": true},
			wantErr: "could not start the checkpoint hook",
			wantCalls: []string{
				"lock faults", "checkpoint 0", "checkpoint 1",
				"restore 0", "unlock faults", "resume 0",
			},
		},
		{
			name:    "undo fails",
			status:  map[string]uint32{"checkpoint 1": 1, "restore 0": 1},
			wantErr: "undoing it also failed",
			wantCalls: []string{
				"lock faults", "checkpoint 0", "checkpoint 1",
				"restore 0", "unlock faults", "resume 1",
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := &fakeHook{status: tc.status, broken: tc.broken}
			err := checkpointWithHook(f.ops(), []int{0, 1})
			if tc.wantErr == "" && err != nil || tc.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErr)) {
				t.Errorf("checkpointWithHook() = %v, want error containing %q", err, tc.wantErr)
			}
			if !slices.Equal(f.calls, tc.wantCalls) {
				t.Errorf("calls = %q, want %q", f.calls, tc.wantCalls)
			}
			if f.locked != tc.wantLocked {
				t.Errorf("page faults held = %v, want %v", f.locked, tc.wantLocked)
			}
		})
	}
}

func TestRestoreWithHook(t *testing.T) {
	for _, tc := range []struct {
		name      string
		deviceMap string
		status    map[string]uint32
		wantErr   string
		wantCalls []string
	}{
		{
			name:      "success",
			wantCalls: []string{"restore 0", "restore 1", "unlock faults", "resume 0", "resume 1"},
		},
		{
			name:      "device map",
			deviceMap: "GPU-a=GPU-b",
			wantCalls: []string{"restore --device-map GPU-a=GPU-b 0", "restore --device-map GPU-a=GPU-b 1", "unlock faults", "resume 0", "resume 1"},
		},
		{
			name:      "restore fails: faults released, no resume",
			status:    map[string]uint32{"restore 1": 1},
			wantErr:   "restore in container 1 failed",
			wantCalls: []string{"restore 0", "restore 1", "unlock faults"},
		},
		{
			name:      "resume fails",
			status:    map[string]uint32{"resume 0": 1},
			wantErr:   "resume in container 0 failed",
			wantCalls: []string{"restore 0", "restore 1", "unlock faults", "resume 0", "resume 1"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := &fakeHook{status: tc.status, locked: true}
			err := restoreWithHook(f.ops(), []int{0, 1}, tc.deviceMap)
			if tc.wantErr == "" && err != nil || tc.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErr)) {
				t.Errorf("restoreWithHook() = %v, want error containing %q", err, tc.wantErr)
			}
			if !slices.Equal(f.calls, tc.wantCalls) {
				t.Errorf("calls = %q, want %q", f.calls, tc.wantCalls)
			}
			if f.locked {
				t.Errorf("page faults still held")
			}
		})
	}
}
