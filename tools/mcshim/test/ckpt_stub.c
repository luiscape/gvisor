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

/* A stand-in for cuda-checkpoint (see run.sh). It appends each invocation to
 * /mnt/ckpt_stub.log, hangs on any invocation containing the text in
 * /mnt/ckpt_stub.hang, fails any containing the text in /mnt/ckpt_stub.fail,
 * answers --get-state with "running", and otherwise succeeds without touching
 * the process. --launch-job execs the rest of argv. */

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Whether line contains the first line of file. */
static int matches(const char* file, const char* line) {
  char want[128] = "";
  FILE* f = fopen(file, "r");
  if (!f) return 0;
  if (!fgets(want, sizeof(want), f)) want[0] = 0;
  fclose(f);
  want[strcspn(want, "\n")] = 0;
  return want[0] && strstr(line, want);
}

int main(int argc, char** argv) {
  char line[512] = "";
  for (int i = 1; i < argc; i++) {
    strncat(line, argv[i], sizeof(line) - strlen(line) - 2);
    strcat(line, i + 1 < argc ? " " : "\n");
  }
  int fd = open("/mnt/ckpt_stub.log", O_WRONLY | O_APPEND | O_CREAT, 0666);
  if (fd >= 0) {
    if (write(fd, line, strlen(line)) < 0) {
    }
    close(fd);
  }
  if (argc > 2 && strcmp(argv[1], "--launch-job") == 0) {
    execv(argv[2], argv + 2);
    perror("execv");
    return 127;
  }
  if (matches("/mnt/ckpt_stub.hang", line))
    for (;;) pause();
  if (matches("/mnt/ckpt_stub.fail", line)) return 1;
  if (argc > 1 && strcmp(argv[1], "--get-state") == 0) puts("running");
  return 0;
}
