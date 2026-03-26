load("//tools:defs.bzl", "cc_binary")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:sandbox"],
    licenses = ["notice"],
)

cc_binary(
    name = "ioctl_overhead_bench",
    testonly = 1,
    srcs = ["ioctl_overhead_bench.c"],
    copts = [
        "-O2",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
    ],
    linkopts = ["-lm"],
    static = True,
)
