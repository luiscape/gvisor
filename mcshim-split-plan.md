# Moving the multicast interposer (mcshim) to its own repository

| | |
| :--- | :--- |
| Status | Draft for review |
| Date | 2026-10-13 |
| Branch | `luis/multicast-xgpu-upstream` (github.com/luiscape/gvisor) |
| Related | `mcshim-merge-plan.md` (decision 3, section 9) |

## 1. Summary

`tools/mcshim` (the `LD_PRELOAD` interposer, about 7.5k lines with tests)
moves to its own repository. gVisor keeps the sentry, nvproxy and runsc sides,
and gains two things it lacks today: a written specification of the protocol
it defines, and CPU tests of its own side of that protocol against a stub
interposer.

The model is `cuda-checkpoint`: an operator-provided, out-of-tree component
that runsc finds by path and checks for compatibility. The difference, which
the plan addresses directly, is that gVisor *defines* this protocol rather
than consuming it, and that runsc injects the interposer into application
processes.

The split is safe to do now because the protocol is versioned (N22): a
mismatched pair refuses the checkpoint, or fails the restore, instead of
misbehaving.

## 2. Goals and non-goals

Goals:

- gVisor builds and tests with no dependency on the interposer's sources.
- The interface between the two is written down, versioned, and tested on
  both sides without a GPU.
- The interposer repository has its own build, CPU CI, releases and a
  compatibility table with runsc.
- Nothing changes for workloads: same flags, same files, same protocol
  version (1).

Non-goals:

- Changing the protocol or the interposer's behavior.
- Running GPU tests in either repository's CI (they stay manual, as today).

## 3. What crosses the boundary

The contract, which becomes the specification (phase 1):

| Interface | Defined by | Used by |
| :--- | :--- | :--- |
| `/tmp/mcshim`: marker names (`gate`, `suspend`), acks (`present`, `gated`, `suspended`, `resumed`, `error` `.<pid>`), and the order of transitions | sentry | interposer |
| `mcshim protocol N` in `present.<pid>` (N22) | both | both |
| `nvproxy_exported_object:\tclient=0x... object=0x... class=0x...` in `/proc/<pid>/fdinfo/<fd>` | nvproxy | interposer |
| `MCSHIM_DISABLE=1` in `cuda-checkpoint`'s environment | sentry | interposer |
| `LD_PRELOAD` and `/etc/ld.so.preload`; the `cuda-checkpoint --launch-job` wrap | runsc | interposer's assumptions |

Interposer-internal (stays out of the spec): the published fd files
(`fd-<client>-<object>`), the log, the tracking tables.

The only build dependency today is `//pkg/sentry/control:control_test`
reading `mcshim.c` for `TestCudaShimProtocolMatchesInterposer`. Everything
else in gVisor that names `tools/mcshim` is a comment or a doc link.

## 4. Decisions needed first

| # | Decision | Recommendation |
| :--- | :--- | :--- |
| D1 | Repository owner and name | A Modal organization repository, so it outlives any one person. |
| D2 | License and headers | Keep Apache-2.0, so the code can move back into gVisor if upstream asks. Copy gVisor's `LICENSE`. Decide whether headers keep "The gVisor Authors" or name Modal; ask Modal's legal contact. |
| D3 | Keep history | Yes, with `git filter-repo`: it preserves the reasoning in commit messages, and makes a move back cheap. |
| D4 | Upstream's view | Ask a gVisor maintainer before phase 3, ideally in the PR1 design review: "out-of-tree like `cuda-checkpoint`, with an in-tree spec and stub tests." If they want it in-tree, phases 1 and 2 still pay off and phase 3 is skipped. |

## 5. Phases

### Phase 1: gVisor preparation (lands either way)

Worth doing even if the interposer stays in-tree.

1. **Protocol specification**, `g3doc/proposals/cuda_multigpu_checkpoint.md`
   or a new `g3doc/architecture_guide/` page: every row of section 3 — file
   names, who creates and removes each, the transition order, the ack
   timeout, error reporting, the version rule ("bump with any change the
   other side would misread"), and the fdinfo line's format and stability
   (stable within a protocol version).
2. **Make the sentry's protocol code testable.** Each function in
   `state_cuda_shim.go` takes `*kernel.Kernel` and `*kernel.ThreadGroup`.
   Refactor the driver logic (arm gate, suspend, resume, unwind, wait for
   acks, check protocol) over a small interface — open the control
   directory for a process, list processes, process id — the way
   `checkpointLocked` was made generic over `cudaActions`. Production code
   passes the kernel-backed implementation; behavior is unchanged.
3. **Stub interposer and tests.** A Go fake in the test (a goroutine per
   fake process, on the tmpfs root `TestCudaShimFS` already uses) that
   follows the specification and can be told to refuse, fail a suspend or
   resume, never ack, or announce another version. Cases: a full
   gate/suspend/resume/release cycle; refusal at the gate (unwound);
   suspend failure (`errCudaShimTornDown`, not unwound); ack timeout;
   protocol mismatch before a checkpoint and before a rebuild; a process
   that never announced itself is not waited on; stale acks are cleared.
4. **fdinfo contract test.** `exported_objects_test.go` already checks the
   line; add a comment that it is part of the specification, and a case for
   its exact format.

Acceptance: control and nvproxy tests and nogo pass; the stub tests fail if
the transition order or ack handling changes.

### Phase 2: create the interposer repository

1. **Extract** from the branch head:
   ```sh
   pip install git-filter-repo
   git clone --no-local /path/to/gvisor mcshim && cd mcshim
   git checkout luis/multicast-xgpu-upstream
   git filter-repo --path tools/mcshim/ --path-rename tools/mcshim/:
   ```
   Keep only that branch; push it as `main`.
2. **Build without Bazel.** A `Makefile` that does what `build.sh` does
   (glibc stubs, `-O2 -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=2`, the
   `--no-as-needed` link) and checks the GLIBC_2.17 floor with `objdump`.
   Remove `BUILD` and `glibc_floor_test.go`: the floor check and
   `run.sh glibc` cover the same ground, and `glibc_floor_test.go`'s
   NEEDED check moves into the Makefile's check. Keep `build.sh` for the
   pinned-container build.
3. **Rewrite references.** README and source comments that name
   `pkg/sentry/...` or `runsc/...` become links to gVisor at a pinned
   commit or tag, plus a link to the specification.
4. **CI** (GitHub Actions, CPU only):
   - gcc and clang builds with `-Wall -Wextra -Werror`, normal and
     `-DMCSHIM_TABLE_CHECKS`;
   - the glibc floor and NEEDED check;
   - `run.sh preload noeffect tables` (no Docker needed);
   - `run.sh glibc ctlsec` (Docker on the runner);
   - the `-O0` negative check: a shim built without optimization must fail
     `preload_test.sh` (today's `//tools/mcshim:preload_test` guarantee);
   - **protocol check**: fetch `cudaShimProtocol` from gVisor at the ref
     listed in `COMPATIBILITY.md` and compare it with `MCSHIM_PROTOCOL`.
     This replaces `TestCudaShimProtocolMatchesInterposer`.
5. **Releases.** Tag `vX.Y.Z`; CI attaches `mcshim-<version>-<arch>.so`
   (x86_64 first; arm64 when validated) and its sha256. `COMPATIBILITY.md`
   lists, per release: protocol version, the runsc releases it pairs with,
   driver releases validated.
6. **GPU harness.** Unchanged: `run.sh -r RUNSC` runs the stub and runsc
   tests against a runsc binary built from gVisor or released.

Acceptance: CI green; the released `.so` is byte-for-byte a build of the
same sources the branch tests; `run.sh` passes against a runsc from the
gVisor branch.

### Phase 3: remove the interposer from gVisor

On the branch (and so from the PRs cut from it):

1. Delete `tools/mcshim`, the `make mcshim` target and the `.gitignore`
   line.
2. `pkg/sentry/control/BUILD`: drop `data = ["//tools/mcshim:mcshim.c"]`.
   Delete `TestCudaShimProtocolMatchesInterposer`; the stub tests (phase 1)
   use `cudaShimProtocol`, and the interposer's CI checks the other side.
3. Point references to the new repository: `runsc/config/flags.go` and
   `config.go` ("built from tools/mcshim"), the user guide's build
   instructions (release downloads instead of `build.sh` or the Bazel
   target), the proposal, `exported_objects_test.go`, and the comments in
   `state_cuda_shim.go` that name `mcshim.c`.
4. The user guide gains a short compatibility note: use the interposer
   release paired with your runsc release (`COMPATIBILITY.md`).

Acceptance: `git grep tools/mcshim` is empty outside dev files; `//runsc`
builds; control, boot, config and nvproxy tests and nogo pass.

### Phase 4: Modal's deployment

1. Build `mcshim.so` from the new repository's release, not from the gVisor
   tree; install it at a versioned path (`/opt/mcshim/<version>/mcshim.so`
   or by sha256), and keep every version existing checkpoints use.
2. Roll out runsc and the interposer together: the interposer deployed
   today predates N22, writes `ok`, and a runsc with N22 refuses it.
3. A checkpoint taken with protocol N can only be restored by a runsc that
   speaks N; record the protocol version alongside each checkpoint in
   Modal's metadata so the scheduler restores onto a compatible runsc.

### Phase 5: update the merge plan

- PR3 leaves the upstream plan, with R1-R6, the README part of R23, the
  interposer half of R24, and the `tables` fix.
- PR1 (design) states the out-of-tree model and links the specification.
- PR8 (interposer protocol) gains the phase 1 refactor, the stub tests and
  the specification.
- PR9 (runsc) docs point to the repository.
- Decision 3 is resolved; section 9 becomes the chosen design.

## 6. Order and effort

```
Phase 1 (gVisor prep) ──┬──> Phase 3 (remove from gVisor) ──> Phase 5 (merge plan)
D1-D4 ──> Phase 2 ──────┘                                  └─> Phase 4 (Modal)
```

Phases 1 and 2 are independent and can run in parallel. Phase 3 waits for
both, and for D4.

| Phase | Rough effort |
| :--- | :--- |
| 1. Spec, refactor, stub tests | 2-3 days |
| 2. Repository, build, CI, releases | 1.5-2 days |
| 3. Removal from gVisor | half a day |
| 4. Modal deployment changes | depends on Modal's tooling |
| 5. Merge plan | an hour |

## 7. Risks

| Risk | Mitigation |
| :--- | :--- |
| Upstream prefers the interposer in-tree, or rejects runsc features whose only client is out-of-tree. | Ask first (D4). The spec and stub tests answer the main objection. History and license make a move back cheap. |
| The fdinfo line becomes a cross-repository ABI. | The spec declares it stable within a protocol version; changing it bumps the version, which both sides check. |
| The two drift apart. | N22 at runtime; the interposer's CI compares versions with gVisor; `COMPATIBILITY.md`. |
| Operators run a mismatched pair. | The checkpoint is refused before anything is torn down; a restore fails rather than rebuild wrongly. Documented in the user guide. |
| Restores need old interposer builds. | Versioned install paths; releases are immutable and keep their sha256. |
| The GPU suite runs less often once it lives elsewhere. | Run it per interposer release and per runsc release that touches `pkg/sentry/control` or nvproxy; list the last validated pair in `COMPATIBILITY.md`. |
