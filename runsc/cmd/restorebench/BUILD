load("//tools:defs.bzl", "go_binary")

package(
    default_applicable_licenses = ["//:license"],
    licenses = ["notice"],
)

go_binary(
    name = "restorebench",
    srcs = ["main.go"],
    deps = [
        "//pkg/state/statefile",
        "//runsc/config",
        "//runsc/container",
        "//runsc/flag",
        "//runsc/sandbox",
        "//runsc/specutils",
        "@com_github_opencontainers_runtime_spec//specs-go:go_default_library",
    ],
)
