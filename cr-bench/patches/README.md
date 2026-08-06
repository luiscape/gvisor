# Optional hardening patches (not upstreamed)

These patches make nvproxy device FDs track their application mappings and
implement `memmap.Mappable.InvalidateUnsavable`, dropping unsavable device
pmas at checkpoint time instead of panicking with:

    Can't save pma with non-MemoryFile of type *nvproxy.frontendFDMemmapFile

They were developed while debugging SGLang checkpoint failures, then
**ablated and found not load-bearing** once the CUDA process enumeration
race fix (`pkg/sentry/control/state_cuda.go`, the actual PR) was in place:

- All processes observed holding `/dev/nvidia*` mappings at save time were
  late CUDA initializers (mid/post-cuInit), which the race fix catches and
  suspends via cuda-checkpoint (suspension removes their device mappings).
- NVML-only processes create no device mappings at all.
- Full matrix (vLLM + SGLang, single/multi GPU, CUDA graphs + torch.compile,
  deep quiesce) passes with the race fix alone.

They remain structurally useful as defense-in-depth: a process early in
cuInit (device mapped, CUDA driver thread not yet spawned) can evade the
race fix's paused verification heuristic, and any non-libcuda user of the
RM device files would panic the save. No reproducer is known for either.

## Applying

- `nvproxy-invalidate-unsavable-frontend.patch` — frontendFD (/dev/nvidia#,
  /dev/nvidiactl). Additionally requires the mutex declaration below in
  `pkg/sentry/devices/nvproxy/BUILD` (not captured in the patch):

  ```bzl
  declare_mutex(
      name = "maps_mutex",
      out = "maps_mutex.go",
      package = "nvproxy",
      prefix = "maps",
  )
  ```

  and `"maps_mutex.go"` added to the `go_library` srcs.

- `nvproxy-invalidate-unsavable-uvm.patch` — uvmFD (/dev/nvidia-uvm), same
  pattern; requires the frontend patch (references `mapsMutex` and the
  comment on `frontendFD.mappings`).

Both follow the existing pattern in
`pkg/sentry/devices/tpuproxy/vfio/pci_device_fd_mmap.go`.
