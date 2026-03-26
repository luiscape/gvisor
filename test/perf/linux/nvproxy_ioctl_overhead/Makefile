# Copyright 2024 The gVisor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?= -lm

BINARY := ioctl_overhead_bench
SRC := ioctl_overhead_bench.c

# Static build for portability across container runtimes.
# The benchmark must run in minimal containers (e.g. scratch, busybox)
# without depending on the container's libc.
STATIC_BINARY := $(BINARY)_static
STATIC_CFLAGS := $(CFLAGS) -static
STATIC_LDFLAGS := $(LDFLAGS)

.PHONY: all clean static run run-compare help

all: $(BINARY)

$(BINARY): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

static: $(STATIC_BINARY)

$(STATIC_BINARY): $(SRC)
	$(CC) $(STATIC_CFLAGS) -o $@ $< $(STATIC_LDFLAGS)

clean:
	rm -f $(BINARY) $(STATIC_BINARY)

# Quick run with default settings.
run: $(BINARY)
	./$(BINARY)

# Run with more iterations for higher-confidence results.
run-long: $(BINARY)
	./$(BINARY) --iterations 50000 --transfer-reps 500 --warmup 2000

# Run the three-way comparison: native, runc, gVisor.
# Requires Docker with runsc runtime installed.
# Uses the static binary so it works in a scratch-like container.
run-compare: $(STATIC_BINARY)
	@echo "============================================================"
	@echo " Native"
	@echo "============================================================"
	./$(STATIC_BINARY)
	@echo ""
	@echo "============================================================"
	@echo " runc (Docker default runtime)"
	@echo "============================================================"
	docker run --rm \
		-v $(CURDIR)/$(STATIC_BINARY):/bench/$(BINARY):ro \
		--entrypoint /bench/$(BINARY) \
		busybox:latest
	@echo ""
	@echo "============================================================"
	@echo " gVisor (runsc)"
	@echo "============================================================"
	docker run --rm --runtime=runsc \
		-v $(CURDIR)/$(STATIC_BINARY):/bench/$(BINARY):ro \
		--entrypoint /bench/$(BINARY) \
		busybox:latest

help:
	@echo "Targets:"
	@echo "  all          Build dynamically-linked benchmark (default)"
	@echo "  static       Build statically-linked benchmark (for containers)"
	@echo "  run          Run benchmark with default settings"
	@echo "  run-long     Run benchmark with higher iteration counts"
	@echo "  run-compare  Run native vs runc vs gVisor comparison (needs Docker+runsc)"
	@echo "  clean        Remove built binaries"
	@echo ""
	@echo "Variables:"
	@echo "  CC           C compiler (default: gcc)"
	@echo "  CFLAGS       Compiler flags (default: -O2 -Wall -Wextra)"
