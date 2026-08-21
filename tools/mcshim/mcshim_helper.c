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

/*
 * mcshim-helper: multicast create/attach proxy for the interposer.
 *
 * On the supported drivers (measured on R580), a cuda-checkpoint-restored
 * process can import multicast group fds, bind memory to them, and map
 * multicast VAs -- but cuMulticastCreate
 * and cuMulticastAddDevice fail with CUDA_ERROR_INVALID_DEVICE (and
 * cuCtxCreate with OOM): the restore blocks fresh device admission at the
 * process level. This helper is exec'd by mcshim.so during its rebuild as a
 * NEVER-checkpointed process and performs exactly those two operations on
 * the rank's behalf. The group persists once the rank holds an import of
 * it (measured: gpu_mem_snapshots/phase0/native_mc_proxy_restore.py), so
 * the helper exits as soon as the rebuild is done.
 *
 * Protocol, over the socketpair fd passed as argv[1] (one text line per
 * command; replies are "OK\n" or "ERR <CUresult>\n"; fds travel as
 * SCM_RIGHTS):
 *
 *   CREATE <numDevices> <size> <handleTypes> <flags>   -> OK + group fd
 *   IMPORT                        (fd attached)        -> OK
 *   ADDDEV <device ordinal>                            -> OK
 *   EXIT                                               -> OK, then exit
 *
 * The current group is whatever the last CREATE/IMPORT produced. The helper
 * also exits on EOF, so a dying parent can never leak it.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

typedef unsigned long long CUmemGenericAllocationHandle;

typedef struct {
	unsigned int numDevices;
	size_t size;
	unsigned long long handleTypes;
	unsigned long long flags;
} CUmulticastObjectProp;

#define CU_MEM_HANDLE_TYPE_POSIX_FD 0x1

static int (*p_cuInit)(unsigned int);
static int (*p_cuMulticastCreate)(CUmemGenericAllocationHandle *,
                                  const CUmulticastObjectProp *);
static int (*p_cuMulticastAddDevice)(CUmemGenericAllocationHandle, int);
static int (*p_cuMemExportToShareableHandle)(void *,
                                             CUmemGenericAllocationHandle,
                                             int, unsigned long long);
static int (*p_cuMemImportFromShareableHandle)(
    CUmemGenericAllocationHandle *, void *, int);

static void die(const char *msg)
{
	fprintf(stderr, "mcshim-helper: %s\n", msg);
	_exit(1);
}

static int send_reply(int s, const char *line, int fd)
{
	struct iovec iov = {(void *)line, strlen(line)};
	char cbuf[CMSG_SPACE(sizeof(int))];
	struct msghdr mh = {0};
	mh.msg_iov = &iov;
	mh.msg_iovlen = 1;
	if (fd >= 0) {
		mh.msg_control = cbuf;
		mh.msg_controllen = sizeof(cbuf);
		struct cmsghdr *c = CMSG_FIRSTHDR(&mh);
		c->cmsg_level = SOL_SOCKET;
		c->cmsg_type = SCM_RIGHTS;
		c->cmsg_len = CMSG_LEN(sizeof(int));
		memcpy(CMSG_DATA(c), &fd, sizeof(int));
	}
	return sendmsg(s, &mh, 0) < 0 ? -1 : 0;
}

static ssize_t recv_cmd(int s, char *buf, size_t n, int *fd_out)
{
	struct iovec iov = {buf, n - 1};
	char cbuf[CMSG_SPACE(sizeof(int))];
	struct msghdr mh = {0};
	mh.msg_iov = &iov;
	mh.msg_iovlen = 1;
	mh.msg_control = cbuf;
	mh.msg_controllen = sizeof(cbuf);
	ssize_t r = recvmsg(s, &mh, MSG_CMSG_CLOEXEC);
	*fd_out = -1;
	if (r <= 0)
		return r;
	buf[r] = 0;
	for (struct cmsghdr *c = CMSG_FIRSTHDR(&mh); c; c = CMSG_NXTHDR(&mh, c))
		if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS)
			memcpy(fd_out, CMSG_DATA(c), sizeof(int));
	return r;
}

static void reply_err(int s, int rc)
{
	char line[64];
	snprintf(line, sizeof(line), "ERR %d\n", rc);
	send_reply(s, line, -1);
}

int main(int argc, char **argv)
{
	if (argc != 2)
		die("usage: mcshim-helper <sockfd>");
	int s = atoi(argv[1]);

	void *lib = dlopen("libcuda.so.1", RTLD_NOW);
	if (!lib)
		die("dlopen libcuda.so.1 failed");
	/* Resolve with dlsym on the library handle: even if mcshim.so is
	 * preloaded into this process too (via ld.so.preload), its dlsym
	 * interposition only redirects tracked entry points to wrappers that
	 * forward to the same reals; MCSHIM_DISABLE (set by the spawner)
	 * keeps its control machinery quiet either way. */
	p_cuInit = dlsym(lib, "cuInit");
	p_cuMulticastCreate = dlsym(lib, "cuMulticastCreate");
	p_cuMulticastAddDevice = dlsym(lib, "cuMulticastAddDevice");
	p_cuMemExportToShareableHandle =
	    dlsym(lib, "cuMemExportToShareableHandle");
	p_cuMemImportFromShareableHandle =
	    dlsym(lib, "cuMemImportFromShareableHandle");
	if (!p_cuInit || !p_cuMulticastCreate || !p_cuMulticastAddDevice ||
	    !p_cuMemExportToShareableHandle || !p_cuMemImportFromShareableHandle)
		die("missing driver entry points");
	int rc = p_cuInit(0);
	if (rc != 0) {
		fprintf(stderr, "mcshim-helper: cuInit rc=%d\n", rc);
		_exit(1);
	}

	CUmemGenericAllocationHandle group = 0;
	char buf[256];
	for (;;) {
		int rfd;
		ssize_t r = recv_cmd(s, buf, sizeof(buf), &rfd);
		if (r <= 0)
			_exit(0); /* EOF: parent went away */
		if (!strncmp(buf, "CREATE ", 7)) {
			CUmulticastObjectProp prop;
			memset(&prop, 0, sizeof(prop));
			unsigned long long ht = 0, fl = 0;
			if (sscanf(buf + 7, "%u %zu %llu %llu",
			           &prop.numDevices, &prop.size, &ht,
			           &fl) != 4) {
				reply_err(s, -1);
				continue;
			}
			prop.handleTypes = ht;
			prop.flags = fl;
			rc = p_cuMulticastCreate(&group, &prop);
			if (rc != 0) {
				reply_err(s, rc);
				continue;
			}
			int fd = -1;
			rc = p_cuMemExportToShareableHandle(
			    &fd, group, CU_MEM_HANDLE_TYPE_POSIX_FD, 0);
			if (rc != 0) {
				reply_err(s, rc);
				continue;
			}
			send_reply(s, "OK\n", fd);
			close(fd);
		} else if (!strncmp(buf, "IMPORT", 6)) {
			if (rfd < 0) {
				reply_err(s, -1);
				continue;
			}
			rc = p_cuMemImportFromShareableHandle(
			    &group, (void *)(long)rfd,
			    CU_MEM_HANDLE_TYPE_POSIX_FD);
			close(rfd);
			rfd = -1;
			if (rc != 0) {
				reply_err(s, rc);
				continue;
			}
			send_reply(s, "OK\n", -1);
		} else if (!strncmp(buf, "ADDDEV ", 7)) {
			int dev = atoi(buf + 7);
			rc = p_cuMulticastAddDevice(group, dev);
			if (rc != 0) {
				reply_err(s, rc);
				continue;
			}
			send_reply(s, "OK\n", -1);
		} else if (!strncmp(buf, "EXIT", 4)) {
			send_reply(s, "OK\n", -1);
			_exit(0);
		} else {
			reply_err(s, -1);
		}
		if (rfd >= 0)
			close(rfd);
	}
}
