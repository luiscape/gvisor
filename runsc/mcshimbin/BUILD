load("//tools:defs.bzl", "go_library")

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
