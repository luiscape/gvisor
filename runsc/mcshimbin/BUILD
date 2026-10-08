load("//tools:defs.bzl", "go_library", "go_test")

package(
    default_applicable_licenses = ["//:license"],
    licenses = ["notice"],
)

# go:embed cannot reference files outside this package, and the cc_binary
# output carries a build-system name (libmcshim.so); copy it here under the
# name the loader materializes it as.
genrule(
    name = "mcshim_so",
    srcs = ["//tools/mcshim"],
    outs = ["mcshim.so"],
    cmd = "cat < $(SRCS) > $(OUTS)",
)

go_library(
    name = "mcshimbin",
    srcs = ["mcshimbin.go"],
    embedsrcs = ["mcshim.so"],
    visibility = ["//runsc:__subpackages__"],
)

# The embedded interposer is preloaded into every process of a container
# image, so it must not need a glibc newer than the image's (see
# tools/mcshim/README.md).
go_test(
    name = "mcshimbin_test",
    size = "small",
    srcs = ["mcshimbin_test.go"],
    library = ":mcshimbin",
)
