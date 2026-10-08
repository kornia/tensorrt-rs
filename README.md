# tensorrt-rs

Rust bindings for [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt).

TensorRT has no C API, so the bindings go through a small hand-written pure-C
shim over the TensorRT C++ API; bindgen only ever sees that C header. Targets
the TensorRT that ships with JetPack on Jetson Orin (10.3.x, aarch64).

## Crates

| Package (lib) | Path | Role |
|---------------|------|------|
| `kornia-tensorrt-sys` (`tensorrt_sys`) | [`crates/tensorrt-sys`](crates/tensorrt-sys) | Raw FFI: `btrt_*` C shim over TensorRT (logger, runtime, engine, execution context, pinned host memory) + optional ONNX → engine builder. Owns `build.rs` and `links = "nvinfer"`. |
| `kornia-tensorrt` (`tensorrt`) | [`crates/tensorrt`](crates/tensorrt) | Safe API: `Logger` → `Runtime` → `Engine` → `Session` (`Arc` chain), `TensorSpec`, typed `TrtError`, cudarc-backed `DeviceBuffer` / `PinnedBuffer` / `Stream`, user-managed activation memory, `builder` feature. No `kornia` dependency. |

Both were extracted from [`kornia/vision-rt`](https://github.com/kornia/vision-rt):
the `-sys` crate was its `crates/trt-sys`, and the safe crate is the TensorRT part
of its `vrt` crate. The crates.io names `tensorrt-rs`, `tensorrt-sys` and
`tensorrt` belong to unrelated projects, hence the `kornia-` package prefix; the
lib names stay short.

0.2.0 is a breaking release: the 0.1 package `tensorrt-rs` (lib `tensorrt_rs`)
is now `kornia-tensorrt-sys` (lib `tensorrt_sys`), with the same API.

## Usage

```toml
[dependencies]
# Safe API (most users):
kornia-tensorrt = { git = "https://github.com/kornia/tensorrt-rs", branch = "main" }
# Or the raw FFI only:
# kornia-tensorrt-sys = { git = "https://github.com/kornia/tensorrt-rs", branch = "main" }
```

```rust
use tensorrt::{logger::Severity, Engine, Logger, Runtime, Session, Stream};

let runtime = Runtime::new(Logger::new(Severity::Warning)?)?;
let engine = Engine::from_file(runtime, "model.engine")?;
let stream = Stream::new_standalone()?;
let mut session = Session::with_stream(engine, stream.cuda_stream().clone())?;
```

`kornia-tensorrt-sys` declares `links = "nvinfer"`, so a dependency graph can hold
only one copy of it. Cargo keys a git source on the exact spec string (URL plus
`branch`/`tag`/`rev`), so spell the git dependency **once per graph**: every crate
in the graph must name this repo the same way, or Cargo sees two packages that
both link `nvinfer` and refuses to resolve. Its build script exports the TensorRT
and CUDA paths it linked as `DEP_NVINFER_*` for direct dependents.

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

The `builder` feature (`kornia-tensorrt/builder`, which forwards to
`kornia-tensorrt-sys/builder`) adds the in-process ONNX → engine builder and
links `libnvonnxparser`.

### Without TensorRT (CI, docs, laptops)

```bash
TRT_STUB=1 cargo clippy --workspace --all-targets -- -D warnings
```

`TRT_STUB=1` (or a docs.rs build) uses the committed
`crates/tensorrt-sys/src/pregenerated_bindings.rs` instead of compiling the shims and running
bindgen, so `cargo check` / `clippy` / `doc` / `test` work with no CUDA or
TensorRT installed. Every `btrt_*` function panics if called in a stub build.

## Updating for a new TensorRT

See [UPDATING.md](UPDATING.md).

## License

Apache-2.0
