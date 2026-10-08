/*
 * Copyright 2026 The gVisor Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* A typical LD_PRELOAD interposer (see run.sh preload): it wraps puts and
 * finds the next definition with dlsym(RTLD_NEXT). If the shim's dlsym makes
 * itself the caller, RTLD_NEXT finds this wrapper again; exit 99 rather than
 * recurse until the stack overflows. Built with -DMAIN, the program. */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>

#ifdef MAIN
int main(void) { return puts("ok") < 0; }
#else
int puts(const char* s) {
  static int (*next)(const char*);
  static __thread int depth;
  if (!next) *(void**)&next = dlsym(RTLD_NEXT, "puts");
  if (!next || ++depth > 8) _exit(99);
  int rc = next(s);
  depth--;
  return rc;
}
#endif
