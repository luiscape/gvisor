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

/* A CUDA process for ctlsec.sh, over the fake libcuda (fakecuda.c): it calls
 * cuInit, which starts the shim's control thread, and waits for the thread to
 * announce itself with a regular file present.<pid> (lstat, not access: a
 * planted symlink must not count). Exits 0 once it has, 3 if it has not after
 * 2 s (the shim declined to take part). Either way the shim has opened its
 * log and written its markers by then. */

#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

int cuInit(unsigned);

int main(void) {
  if (cuInit(0) != 0) return 2;
  char p[64];
  snprintf(p, sizeof(p), "/tmp/mcshim/present.%d", (int)getpid());
  for (int i = 0; i < 200; i++) {
    struct stat st;
    if (lstat(p, &st) == 0 && S_ISREG(st.st_mode)) {
      usleep(200000); /* the thread logs right after writing present */
      return 0;
    }
    usleep(10000);
  }
  return 3;
}
