load("//tools:defs.bzl", "cc_binary")

package(
    default_applicable_licenses = ["//:license"],
    licenses = ["notice"],
)

# Link-time stubs of libdl.so.2 and libpthread.so.0 (see glibc_stubs.sh):
# mcshim.so takes dlopen and pthread_create from them at their original
# symbol versions, so its glibc floor does not follow the build environment's
# glibc (2.34 re-versioned those entry points), and the image's dynamic
# linker finds them in libdl/libpthread (glibc < 2.34) or libc (>= 2.34).
genrule(
    name = "glibc_stubs",
    outs = [
        "glibc_stubs/libdl.so.2",
        "glibc_stubs/libpthread.so.0",
    ],
    cmd = "CC=$(CC) $(location glibc_stubs.sh) $(RULEDIR)/glibc_stubs",
    toolchains = ["@bazel_tools//tools/cpp:current_cc_toolchain"],
    tools = ["glibc_stubs.sh"],
)

filegroup(
    name = "libdl_stub",
    srcs = ["glibc_stubs/libdl.so.2"],
)

filegroup(
    name = "libpthread_stub",
    srcs = ["glibc_stubs/libpthread.so.0"],
)

# The multicast suspend/resume interposer, LD_PRELOADed into CUDA processes
# (see README.md). Toolkit-free: CUDA types are declared locally, so no
# cuda.h/nvcc is required.
#
# The result is preloaded into every process of container images whose glibc
# may be older than the build environment's: its glibc floor is GLIBC_2.17
# (see README.md), which //runsc/mcshimbin:mcshimbin_test checks on the
# embedded copy and tools/mcshim/test/run.sh glibc checks in old images.
cc_binary(
    name = "mcshim",
    srcs = ["mcshim.c"],
    additional_linker_inputs = [
        ":libdl_stub",
        ":libpthread_stub",
    ],
    # The dlsym interposer must forward with a tail call (see dlsym in
    # mcshim.c), which GCC only emits when optimizing; fastbuild is -O0.
    copts = ["-O2"],
    # This library is preloaded into binaries from arbitrary container
    # images, whose dynamic linkers may not support RELR relocations
    # (glibc < 2.36). Same rationale as //tools/ioctl_sniffer:ioctl_hook.
    features = ["-enable_relr"],
    # The stubs must stay NEEDED even though glibc >= 2.34 defines all their
    # symbols in libc, hence --no-as-needed around them; the toolchain's
    # trailing -lstdc++ -lm are left to --as-needed, which drops them.
    linkopts = [
        "-Wl,--no-as-needed",
        "$(location :libdl_stub)",
        "$(location :libpthread_stub)",
        "-Wl,--as-needed",
    ],
    linkshared = True,
    visibility = ["//runsc/mcshimbin:__pkg__"],
)
