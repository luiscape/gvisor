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

func TestCudaAdmissionGate(t *testing.T) {
	var a cudaAdmission
	tg := &kernel.ThreadGroup{}
	exemptTG := &kernel.ThreadGroup{}
	isExempt := func(x *kernel.ThreadGroup) bool { return x == exemptTG }

	// Open by default.
	if _, ok := a.wait(tg); !ok {
		t.Fatal("fresh gate should admit")
	}

	// Closed: non-exempt waits, exempt passes.
	a.close(isExempt)
	ch, ok := a.wait(tg)
	if ok || ch == nil {
		t.Fatal("closed gate should hand out a wait channel")
	}
	if _, ok := a.wait(exemptTG); !ok {
		t.Fatal("exempt thread group should pass a closed gate")
	}
	select {
	case <-ch:
		t.Fatal("wait channel closed while the gate is closed")
	default:
	}

	// Re-closing replaces the predicate but keeps the same waiters waiting.
	a.close(nil)
	if _, ok := a.wait(exemptTG); ok {
		t.Fatal("replaced predicate should no longer exempt")
	}
	ch2, _ := a.wait(tg)
	if ch2 != ch {
		t.Fatal("re-close must not drop existing waiters (channel changed)")
	}

	// Open releases everyone and admits newcomers; a second open is a no-op.
	a.open()
	select {
	case <-ch:
	default:
		t.Fatal("open did not release waiters")
	}
	if _, ok := a.wait(tg); !ok {
		t.Fatal("open gate should admit")
	}
	a.open()
}
