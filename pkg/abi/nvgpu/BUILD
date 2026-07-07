load("//tools:defs.bzl", "go_library", "go_test")

package(default_applicable_licenses = ["//:license"])

licenses(["notice"])

go_library(
    name = "nvgpu",
    srcs = [
        "classes.go",
        "ctrl.go",
        "frontend.go",
        "frontend_unsafe.go",
        "nvfs.go",
        "nvgpu.go",
        "status.go",
        "uvm.go",
    ],
    hostlayout = True,
    marshal = True,
    visibility = [
        "//pkg/sentry:internal",
        "//tools:__subpackages__",
    ],
    deps = ["//pkg/marshal"],
)

go_test(
    name = "nvgpu_test",
    size = "small",
    srcs = ["nvfs_test.go"],
    library = ":nvgpu",
)
