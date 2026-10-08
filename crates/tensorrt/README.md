# kornia-tensorrt

Safe Rust API for NVIDIA TensorRT 10.x on top of
[`kornia-tensorrt-sys`](../tensorrt-sys). The library name is `tensorrt`.

- `Logger` → `Runtime` → `Engine` → `Session`, each holding an `Arc` to its
  parent so TensorRT objects are destroyed in the required order.
- `Logger`, `Runtime`, `Engine`: `Send + Sync`. `Session` (one
  `IExecutionContext`): `Send`, not `Sync` — one session per thread from a
  shared `Arc<Engine>`.
- `TensorSpec` (name, mode, dtype, dims) copied out of the engine once at load;
  unmodelled TensorRT data types fail the load instead of being misread.
- `DeviceBuffer`, `PinnedBuffer`, `Stream` over [cudarc](https://crates.io/crates/cudarc).
- User-managed activation memory: `Engine::device_memory_size`,
  `Session::with_stream_user_memory`, `Session::set_device_memory`, so engines
  that run one after another can share one buffer.
- `builder` feature: in-process ONNX → serialized engine (`EngineBuilder`).

Ported from `vision-rt`'s `vrt` crate (`logger`, `runtime`, `engine`,
`session`, `builder`, `buffer`, `error`, `dtype`) with no behaviour change.

License: Apache-2.0
