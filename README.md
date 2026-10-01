# tensorrt-rs

Rust bindings for [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt).

TensorRT has no C API, so the bindings go through a small hand-written pure-C
shim over the TensorRT C++ API; bindgen only ever sees that C header. Targets
the TensorRT that ships with JetPack on Jetson Orin (10.3.x, aarch64).

## Crates

| Crate | Role |
|-------|------|
| [`crates/tensorrt-rs`](crates/tensorrt-rs) | Raw FFI: `btrt_*` C shim over TensorRT (logger, runtime, engine, execution context, pinned host memory) + optional ONNX → engine builder |

`tensorrt-rs` was extracted from [`kornia/vision-rt`](https://github.com/kornia/vision-rt)
(where it was `crates/trt-sys`), whose `vrt` crate is the safe layer built on top of it.

## Usage

```toml
[dependencies]
tensorrt-rs = { git = "https://github.com/kornia/tensorrt-rs", branch = "main" }
```

`tensorrt-rs` declares `links = "nvinfer"`, so a dependency graph can hold only one
copy of it. Cargo keys a git source on the exact spec string — every crate in
the graph must name this repo the same way.

## Building

The build compiles the C++ shims and runs bindgen, so it needs a C++17 compiler,
libclang, and the TensorRT + CUDA headers and libraries. The defaults are the
JetPack paths; override them elsewhere:

| Variable | Default | Purpose |
|----------|---------|---------|
| `TRT_INCLUDE_DIR` | `/usr/include/aarch64-linux-gnu` | Directory holding `NvInfer.h` |
| `TRT_LIB_DIR` | `/usr/lib/aarch64-linux-gnu` | Directory holding `libnvinfer.so` |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA toolkit root (`include/`, `lib64/`) |
| `TRT_STUB` | unset | Set to skip the native compile/link (see below) |

The `builder` feature adds the in-process ONNX → engine builder and links
`libnvonnxparser`.

### Without TensorRT (CI, docs, laptops)

```bash
TRT_STUB=1 cargo clippy --workspace --all-targets -- -D warnings
```

`TRT_STUB=1` (or a docs.rs build) uses the committed
`src/pregenerated_bindings.rs` instead of compiling the shims and running
bindgen, so `cargo check` / `clippy` / `doc` / `test` work with no CUDA or
TensorRT installed. Every `btrt_*` function panics if called in a stub build.

## Updating for a new TensorRT

See [UPDATING.md](UPDATING.md).

## License

Apache-2.0
