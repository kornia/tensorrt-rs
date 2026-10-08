# kornia-tensorrt-sys

Raw FFI bindings for NVIDIA TensorRT, via a pure-C shim over the TensorRT C++
API (bindgen never sees C++ headers). Targets the TensorRT that ships with
JetPack on Jetson Orin (10.3.x, aarch64). The library name is `tensorrt_sys`.
The safe layer on top is [`kornia-tensorrt`](https://github.com/kornia/tensorrt-rs/tree/main/crates/tensorrt) in this workspace.

- Compiles small C++ shims (`logger_shim`, `trt_bridge`, and `builder_shim`
  under the `builder` feature) with `cc`, then generates `btrt_*` bindings with
  bindgen.
- Links `nvinfer`, `nvinfer_plugin`, `cudart` (+ `nvonnxparser` with `builder`).
- Exports `TENSORRT_VERSION` (parsed from `NvInferVersion.h`) — used downstream
  for engine-cache keys. Warns at build time if the installed TRT is outside the
  tested 10.3.x range.
- Declares `links = "nvinfer"` and exports the paths it linked against as
  `cargo::metadata`, so direct dependents' build scripts can read
  `DEP_NVINFER_INCLUDE`, `DEP_NVINFER_LIB`, `DEP_NVINFER_CUDA_INCLUDE`,
  `DEP_NVINFER_CUDA_LIB`, `DEP_NVINFER_BRIDGE_INCLUDE` and
  `DEP_NVINFER_VERSION` (or `DEP_NVINFER_STUB=1` in stub mode).

**Off-Jetson:** set `TRT_STUB=1` (or build on docs.rs) to skip the native
compile/link and use committed pregenerated bindings — `cargo check`/`clippy`/
`doc`/`test` work with no CUDA/TensorRT installed. Every `btrt_*` function
panics if called in a stub build; anything that actually runs TensorRT needs a
real install.

License: Apache-2.0
