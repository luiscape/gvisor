load("//tools:defs.bzl", "cc_binary")

package(
    default_applicable_licenses = ["//:license"],
    licenses = ["notice"],
)

# The multicast suspend/resume interposer, LD_PRELOADed into CUDA processes
# (see README.md). Toolkit-free: CUDA types are declared locally, so no
# cuda.h/nvcc is required.
#
# The glibc floor of the result tracks the build environment's libc. Build
# via `make` (which runs Bazel inside the canonical ubuntu-based build
# container) so the library loads in container images with older glibc than
# an arbitrary host; build.sh remains for standalone, out-of-Bazel builds
# and pins the same floor by building in an ubuntu:22.04 container.
cc_binary(
    name = "mcshim",
    srcs = ["mcshim.c"],
    # This library is preloaded into binaries from arbitrary container
    # images, whose dynamic linkers may not support RELR relocations
    # (glibc < 2.36). Same rationale as //tools/ioctl_sniffer:ioctl_hook.
    features = ["-enable_relr"],
    linkopts = [
        "-ldl",
        "-lpthread",
    ],
    linkshared = True,
    visibility = ["//runsc/mcshimbin:__pkg__"],
)

# The create/attach proxy the interposer forks to rebuild multicast groups
# on resume. It must live next to the interposer in the container (the
# MCSHIM_HELPER environment variable's default).
cc_binary(
    name = "mcshim_helper",
    srcs = ["mcshim_helper.c"],
    linkopts = ["-ldl"],
    visibility = ["//runsc/mcshimbin:__pkg__"],
)
