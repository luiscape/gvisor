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

package nvproxy

import (
	"testing"

	"gvisor.dev/gvisor/pkg/sentry/kernel"
)

func TestCudaAdmission(t *testing.T) {
	var a cudaAdmission
	exempted := &kernel.ThreadGroup{}
	held := &kernel.ThreadGroup{}
	isExempted := func(tg *kernel.ThreadGroup) bool { return tg == exempted }

	if a.wait(held) != nil {
		t.Fatal("open gate held a thread group")
	}

	a.close(isExempted)
	ch := a.wait(held)
	if ch == nil {
		t.Fatal("closed gate admitted a non-exempt thread group")
	}
	if a.wait(exempted) != nil {
		t.Fatal("closed gate held an exempt thread group")
	}
	select {
	case <-ch:
		t.Fatal("wait channel closed while the gate is closed")
	default:
	}

	// Closing again keeps waiters waiting but may replace the predicate.
	a.close(nil)
	if a.wait(exempted) == nil {
		t.Fatal("re-closed gate with no exemptions admitted a thread group")
	}
	select {
	case <-ch:
		t.Fatal("wait channel closed by a second close")
	default:
	}

	a.open()
	select {
	case <-ch:
	default:
		t.Fatal("open did not release waiters")
	}
	if a.wait(held) != nil {
		t.Fatal("reopened gate held a thread group")
	}
	a.open() // no-op
}
