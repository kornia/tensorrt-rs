# Updating for a new TensorRT version (Jetson / JetPack)

TensorRT ships as part of **JetPack** on Jetson Orin, so a TRT bump usually means
a JetPack upgrade. Engines are machine-locked to the exact TRT runtime + GPU arch
(SM87), so **every `.engine` must be rebuilt on-device** after the bump — see the
checklist. Off-Jetson, `TRT_STUB=1` lets `cargo check`/`clippy` run against the
committed bindings without any of this.

---

## Files to change in `crates/tensorrt-rs`

### `src/trt_bridge.cpp`

This is the C++ bridge that wraps TRT's abstract C++ API and exposes a flat C surface (`btrt_*` functions). Each function has a comment referencing the exact TRT header and method it wraps.

When TRT renames or changes a method, the C++ compiler reports an error here on `cargo build -p tensorrt-rs`. Fix the error, then proceed.

### `src/logger_shim.cpp`

The ONLY hand-written C++ that cannot be replaced by code generation: `ShimLogger : public nvinfer1::ILogger`. Only change this file if TRT changes the `ILogger::log()` virtual signature.

### `include/trt_bridge.h`

The pure-C header that `bindgen` uses to generate `OUT_DIR/bridge_bindings.rs`. Only change this if the C API surface changes (new `btrt_*` function, changed return type, etc.). `bindgen` regenerates `bridge_bindings.rs` automatically on every build — never edit it by hand.

### `build.rs`

Nothing required — `TENSORRT_VERSION` is parsed from `NvInferVersion.h` at build time and feeds the engine-cache keys. If the new release is outside the tested 10.3.x range, the build emits a `cargo:warning` (it does not fail); bump the supported major.minor set here once the new version is validated.

---

## TRT 8 → TRT 10 migration table

| Old API (TRT 8)                              | New API (TRT 10)                              |
|----------------------------------------------|-----------------------------------------------|
| `obj->destroy()`                             | `delete obj` (standard C++ RAII)              |
| `Dims` with `int32_t` values                 | `Dims64` with `int64_t` values                |
| `builder->setMaxWorkspaceSize(n)`            | `config->setMemoryPoolLimit(kWORKSPACE, n)`   |
| `context->enqueueV2(bindings, stream, null)` | `context->enqueueV3(stream)` (named tensors)  |
| `kEXPLICIT_BATCH` flag in `createNetworkV2` | Removed — explicit batch always assumed       |

TRT 10.x minor releases: the named-tensor I/O API (`setTensorAddress`, `getIOTensorName`) is stable across 10.x. Check TRT release notes for deprecated symbols.

---

## Update checklist

1. Confirm the TRT headers the new JetPack installed (Jetson paths are `aarch64`):
   ```
   dpkg -l | grep -i tensorrt
   grep NV_TENSORRT /usr/include/aarch64-linux-gnu/NvInferVersion.h
   ```

2. Build `tensorrt-rs` — the C++ compiler catches API breakage:
   ```
   cargo build -p tensorrt-rs
   ```
   Fix any errors in `trt_bridge.cpp` (and rarely `logger_shim.cpp`).

3. Rebuild all `.engine` files on-device — engines are tied to the exact TRT
   runtime version + SM87, so anything serialized by the old runtime is stale:
   ```
   /usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.fp16.engine --fp16
   ```
   Downstream caches keyed on `TENSORRT_VERSION` (e.g. `vision-rt`'s `vrt-hub`
   `EngineCache`) pick up the new version on the next build and ignore old
   engines; see `vision-rt`'s `UPDATING.md` for its side of the bump.
