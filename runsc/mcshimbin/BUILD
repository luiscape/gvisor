load("//tools:defs.bzl", "go_library")

package(
    default_applicable_licenses = ["//:license"],
    licenses = ["notice"],
)

# go:embed cannot reference files outside this package, and the cc_binary
# outputs carry build-system names (libmcshim.so, mcshim_helper); copy them
# here under the names the loader materializes them as.
genrule(
    name = "mcshim_so",
    srcs = ["//tools/mcshim"],
    outs = ["mcshim.so"],
    cmd = "cat < $(SRCS) > $(OUTS)",
)

genrule(
    name = "mcshim_helper_bin",
    srcs = ["//tools/mcshim:mcshim_helper"],
    outs = ["mcshim-helper.bin"],
    cmd = "cat < $(SRCS) > $(OUTS)",
)

go_library(
    name = "mcshimbin",
    srcs = ["mcshimbin.go"],
    embedsrcs = [
        "mcshim.so",
        "mcshim-helper.bin",
    ],
    visibility = ["//runsc:__subpackages__"],
)
