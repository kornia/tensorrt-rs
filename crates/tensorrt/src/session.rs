use std::collections::HashMap;
use std::ffi::CString;
use std::sync::Arc;

use crate::{
    buffer::{DeviceBuffer, Stream},
    dtype::DType,
    engine::{DataType, Engine, TensorMode},
    error::{last_trt_error, Result, TrtError},
};
use cudarc::driver::CudaStream;
use std::ffi::c_void;
use tensorrt_sys::*;

/// Map an engine I/O [`DataType`] to a tensor [`DType`].
///
/// Int8/Bool fall back to `U8` (same 1-byte width); our models never emit them,
/// and `OutputView::f32_ptr` rejects any mismatched read regardless.  BF16 keeps its
/// own tag for exactly that reason: an engine built with `BuilderFlag::kBF16` that
/// also *exposes* bf16 I/O must be rejected at `f32_ptr`, not silently read as f32.
fn dtype_of(d: DataType) -> DType {
    match d {
        DataType::Float32 => DType::F32,
        DataType::Float16 => DType::F16,
        DataType::Bf16 => DType::BF16,
        DataType::Int32 => DType::I32,
        DataType::Int8 | DataType::UInt8 | DataType::Bool => DType::U8,
    }
}

/// Borrowed device-side view of a TRT output: device pointer + resolved
/// shape/dtype/byte-length.
///
/// Aliases Session-owned output memory and is valid only until the next `run_*`
/// call or `Session` drop. Decode reads it **on-device** (the session reuses these
/// buffers, so outputs stay raw device views rather than owned tensors).
pub struct OutputView {
    ptr: *mut c_void,
    shape: Vec<usize>,
    dtype: DType,
}

// SAFETY: a device pointer is a stable address; the holder serializes access via
// the per-frame stream sync.
unsafe impl Send for OutputView {}

impl OutputView {
    /// Shape as `i64` (TRT / decode convention).
    pub fn shape_i64(&self) -> Vec<i64> {
        self.shape.iter().map(|&d| d as i64).collect()
    }
    /// Device pointer as `*const f32`, checked against the output dtype — an
    /// `--fp16`-output engine fails loudly here instead of being misread.
    pub fn f32_ptr(&self) -> Result<*const f32> {
        if self.dtype != DType::F32 {
            return Err(TrtError::Shape(format!(
                "output is {:?}, not F32",
                self.dtype
            )));
        }
        Ok(self.ptr as *const f32)
    }
    /// Device pointer as `*const i32`, checked against the output dtype.
    ///
    /// Index-valued outputs (match tables, label ids, argmax results) come back as
    /// integers. TensorRT's `kINT64` is deliberately rejected at engine load
    /// (see `engine.rs`), so exports must cast such outputs to int32 — this is the
    /// accessor for them.
    pub fn i32_ptr(&self) -> Result<*const i32> {
        if self.dtype != DType::I32 {
            return Err(TrtError::Shape(format!(
                "output is {:?}, not I32",
                self.dtype
            )));
        }
        Ok(self.ptr as *const i32)
    }
}

/// Per-tensor device buffer state for one inference session.
struct TensorState {
    buf: DeviceBuffer,
    shape: Vec<i64>,
    dtype: DataType,
}

/// An inference session: one `IExecutionContext` + owned device buffers + stream.
///
/// # Thread safety
/// `Session` is `Send` but **not `Sync`** — `IExecutionContext` is not thread-safe.
/// For concurrent inference create multiple sessions from one `Arc<Engine>`.
pub struct Session {
    ctx: *mut btrt_context_t,
    _engine: Arc<Engine>,
    stream: Stream,
    // Only OUTPUT buffers are session-owned; inputs are the caller's device
    // pointers, bound per call in `run_device_inputs_on_device`.
    outputs: HashMap<String, TensorState>,
    // Outputs the caller has taken over (see `bind_output`): TensorRT writes straight
    // into these instead of the session-owned buffer above.
    bound: HashMap<String, BoundOutput>,
    // How the context's activation (scratch) memory is provided.
    device_memory: DeviceMemory,
    _not_sync: std::marker::PhantomData<std::cell::UnsafeCell<()>>,
}

/// Who provides the execution context's activation memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeviceMemory {
    /// TensorRT allocated it with the context (`kSTATIC`, the default).
    Owned,
    /// `kUSER_MANAGED`: the caller supplies it via `set_device_memory`; `set`
    /// records whether that has happened, since enqueueing without it is an error.
    UserManaged { set: bool },
}

/// The alignment `set_device_memory` checks at run time. TensorRT's real rule is
/// the device's CUDA memory alignment property, which may be stricter; this only
/// catches the common misuse (an arbitrary offset into a larger allocation).
const MIN_DEVICE_MEMORY_ALIGN: usize = 256;

impl DeviceMemory {
    /// Validate `set_device_memory(ptr, bytes)` against this state.
    ///
    /// `needed` (the engine's `device_memory_size`) is only queried once the state
    /// allows a bind. Returns the state to record once the bind succeeds and
    /// `bytes` as the `i64` the bridge takes. Mirrors the bridge's own checks so a
    /// misuse fails here with a specific error.
    fn check_bind(
        self,
        ptr: usize,
        bytes: usize,
        needed: impl FnOnce() -> Result<usize>,
    ) -> Result<(Self, i64)> {
        if self == Self::Owned {
            return Err(TrtError::DeviceMemory(
                "set_device_memory: this session owns its device memory; create it with \
                 Session::with_stream_user_memory"
                    .into(),
            ));
        }
        let needed = needed()?;
        if ptr == 0 {
            // TensorRT: "Setting memory to nullptr is acceptable if the reported
            // size is 0", and the bridge accepts NULL only together with 0 bytes.
            if needed != 0 || bytes != 0 {
                return Err(TrtError::DeviceMemory(format!(
                    "set_device_memory: null device pointer with {bytes} bytes \
                     (the engine needs {needed}); null is only valid as (null, 0) \
                     for an engine that needs 0 bytes"
                )));
            }
        } else if !ptr.is_multiple_of(MIN_DEVICE_MEMORY_ALIGN) {
            return Err(TrtError::DeviceMemory(format!(
                "set_device_memory: device pointer {ptr:#x} is not aligned to \
                 {MIN_DEVICE_MEMORY_ALIGN} bytes"
            )));
        }
        if bytes < needed {
            return Err(TrtError::DeviceMemory(format!(
                "set_device_memory: buffer is {bytes} bytes but the engine needs {needed}"
            )));
        }
        let len = i64::try_from(bytes).map_err(|_| {
            TrtError::DeviceMemory(format!("set_device_memory: {bytes} bytes overflows i64"))
        })?;
        Ok((Self::UserManaged { set: true }, len))
    }

    /// A user-managed context has no activation memory until the caller gives it
    /// some; enqueueing then would hand TensorRT no scratch space at all.
    fn check_run(self) -> Result<()> {
        if self == (Self::UserManaged { set: false }) {
            return Err(TrtError::DeviceMemory(
                "session was created with user-managed device memory; call \
                 set_device_memory before running"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// A caller-supplied device buffer standing in for one session-owned output buffer.
#[derive(Clone, Copy)]
struct BoundOutput {
    ptr: u64,
    bytes: usize,
}

unsafe impl Send for Session {}

impl Session {
    /// Create a session that shares `cuda_stream` with the rest of the app.
    ///
    /// All device-buffer allocations and TRT enqueue calls use the provided
    /// stream; the caller owns syncing it.
    pub fn with_stream(engine: Arc<Engine>, cuda_stream: Arc<CudaStream>) -> Result<Self> {
        Self::init(engine, Stream::from_cuda_stream(cuda_stream), false)
    }

    /// Like [`with_stream`](Self::with_stream), but the execution context is
    /// created **without** its own activation memory
    /// (`ExecutionContextAllocationStrategy::kUSER_MANAGED`).
    ///
    /// The caller must provide that memory with
    /// [`set_device_memory`](Self::set_device_memory) before the first run; a run
    /// before then fails with an error instead of enqueueing. This lets engines
    /// that always run one after another share one activation buffer sized to the
    /// largest [`Engine::device_memory_size`], instead of each context holding its
    /// own (measured on an 8 GB Orin, pi0.5 with 9 engines: 3765 MB per-context vs
    /// 619 MB shared).
    pub fn with_stream_user_memory(
        engine: Arc<Engine>,
        cuda_stream: Arc<CudaStream>,
    ) -> Result<Self> {
        Self::init(engine, Stream::from_cuda_stream(cuda_stream), true)
    }

    fn init(engine: Arc<Engine>, stream: Stream, user_memory: bool) -> Result<Self> {
        // Guard the raw context so it is destroyed on every early-exit path
        // (e.g. a buffer allocation failure below).
        struct CtxGuard(*mut btrt_context_t);
        impl Drop for CtxGuard {
            fn drop(&mut self) {
                if !self.0.is_null() {
                    unsafe { btrt_context_destroy(self.0) }
                }
            }
        }

        let ctx = unsafe {
            if user_memory {
                btrt_context_create_user_memory(engine.as_ptr())
            } else {
                btrt_context_create(engine.as_ptr())
            }
        };
        if ctx.is_null() {
            return Err(TrtError::Create("ExecutionContext"));
        }
        let mut guard = CtxGuard(ctx);

        // Allocate device buffers for OUTPUT tensors only; inputs are supplied by
        // the caller as device pointers at run time.
        let mut outputs = HashMap::new();
        for spec in engine.specs() {
            if spec.mode != TensorMode::Output {
                continue;
            }
            let buf = DeviceBuffer::alloc_with_stream(
                stream.cuda_stream(),
                buf_bytes(&spec.dims, spec.dtype),
            )?;
            outputs.insert(
                spec.name.clone(),
                TensorState {
                    buf,
                    shape: spec.dims.clone(),
                    dtype: spec.dtype,
                },
            );
        }

        guard.0 = std::ptr::null_mut(); // ownership transfers to Session::drop
        Ok(Self {
            ctx,
            _engine: engine,
            stream,
            outputs,
            bound: HashMap::new(),
            device_memory: if user_memory {
                DeviceMemory::UserManaged { set: false }
            } else {
                DeviceMemory::Owned
            },
            _not_sync: std::marker::PhantomData,
        })
    }

    /// Give a session created by [`with_stream_user_memory`](Self::with_stream_user_memory)
    /// the device memory TensorRT uses for activations (`setDeviceMemoryV2`).
    ///
    /// May be called again to switch buffers between runs. Fails (without touching
    /// the context) on a session that owns its memory, on a null `ptr` (unless
    /// the engine needs 0 bytes: `(null, 0)` is then valid), on a `ptr` not
    /// aligned to 256 bytes, or when `bytes` is smaller than
    /// [`Engine::device_memory_size`], which is re-queried on every call.
    ///
    /// `Ok` means the buffer passed these checks and was handed to TensorRT.
    /// TensorRT's `setDeviceMemoryV2` returns nothing, so if TensorRT itself
    /// rejects the buffer it reports that only through the [`Logger`](crate::Logger),
    /// and the next run fails.
    ///
    /// # Safety
    /// - `ptr` must be a CUDA device allocation in the same CUDA context as the
    ///   engine, at least `bytes` long, and aligned to the device's CUDA memory
    ///   alignment property (TensorRT's rule; see `cudaGetDeviceProperties`). A
    ///   base pointer straight from `cudaMalloc` / `cuMemAlloc` / cudarc meets it.
    ///   A sub-range carved out of a larger allocation must keep that alignment,
    ///   which may be stricter than the 256 bytes checked here.
    /// - **The buffer must outlive every run that uses it.** It must stay allocated
    ///   and unmoved from this call until this session is dropped or given another
    ///   buffer, *and* until all work those runs enqueued has completed (a sync of
    ///   the session's stream). `run_*` returns before the GPU is done, so freeing
    ///   the buffer right after a run returns is a use-after-free on the device.
    /// - **Sessions sharing the buffer must never run concurrently.** TensorRT uses
    ///   it as scratch for the whole of each enqueued run, so the GPU work of any two
    ///   runs on sessions given the same (or an overlapping) buffer must not
    ///   overlap: enqueue them on one CUDA stream, or order them across streams with
    ///   a sync or event. Nothing else (another engine's I/O, a kernel) may use the
    ///   buffer while such a run is in flight.
    /// - The contents are scratch: TensorRT does not preserve them between runs,
    ///   and the caller must not rely on them.
    pub unsafe fn set_device_memory(&mut self, ptr: *mut c_void, bytes: usize) -> Result<()> {
        let engine = &self._engine;
        let (state, len) = self
            .device_memory
            .check_bind(ptr as usize, bytes, || engine.device_memory_size())?;
        // The bridge repeats these checks (and the user-managed one) and sets
        // btrt_last_error on every rejection.
        let rc = unsafe { btrt_context_set_device_memory(self.ctx, ptr, len) };
        if rc != 0 {
            return Err(TrtError::DeviceMemory(last_trt_error()));
        }
        self.device_memory = state;
        Ok(())
    }

    /// Have TensorRT write output `name` **directly into a caller-owned device buffer**
    /// instead of the session-owned one.
    ///
    /// Without this, every result that must outlive the next `run_*` has to be copied
    /// out of session memory, because the session reuses its output buffers on every
    /// run. A pipeline holding two results at once (extract two frames, then match
    /// them) therefore pays a device-to-device copy per output per frame purely to
    /// dodge the aliasing. Binding removes both the copy and the aliasing: the buffer
    /// TensorRT fills *is* the caller's.
    ///
    /// **A binding lasts exactly one run** — including a run that *fails* — so bind
    /// before every `run_*`. That is deliberate: a persistent binding would let a caller
    /// drop the result buffer and have the next run write into freed device memory. With
    /// per-run bindings, forgetting to bind sends the output to the session's own buffer
    /// — the wrong destination, but memory-safe and obvious. Failed runs are covered
    /// because a recoverable error (a frame outside the engine's shape profile, say) is
    /// exactly what a caller catches, drops the result, and retries from. There is
    /// deliberately no `unbind`: a binding never outlives the run it was made for.
    ///
    /// Two outputs may not share one buffer; the second bind is rejected.
    ///
    /// The buffer must be large enough for the output at its **resolved** shape, which
    /// for a dynamic engine is only known once input shapes are set — so the size is
    /// checked on every run, not here, and a too-small buffer fails the run rather than
    /// overflowing.
    ///
    /// # Safety
    /// `ptr` must be a CUDA device allocation of at least `bytes`, and must stay alive,
    /// unmoved, and not aliased by any other binding until the caller's next stream
    /// synchronize — the GPU writes to it during that window, after `run_*` returns.
    /// The one-run lifetime above bounds how long that must hold, but it does not
    /// remove the requirement: the buffer must outlive the sync, not just the call.
    pub unsafe fn bind_output(&mut self, name: &str, ptr: u64, bytes: usize) -> Result<()> {
        if !self.outputs.contains_key(name) {
            return Err(TrtError::UnknownTensor(name.into()));
        }
        if ptr == 0 {
            return Err(TrtError::Shape(format!(
                "output '{name}': cannot bind a null device pointer"
            )));
        }
        // Two outputs sharing one buffer would have TensorRT write both into the same
        // memory, and whichever landed second would win — silently, with no error and
        // plausible-looking data. Cheap to catch here (the map holds a handful of
        // entries) and impossible to debug later.
        if let Some((other, _)) = self
            .bound
            .iter()
            .find(|(other, b)| b.ptr == ptr && other.as_str() != name)
        {
            return Err(TrtError::Shape(format!(
                "output '{name}': device pointer {ptr:#x} is already bound to output \
                 '{other}'; each bound output needs its own buffer"
            )));
        }
        self.bound
            .insert(name.to_string(), BoundOutput { ptr, bytes });
        Ok(())
    }

    /// Run inference with inputs **already in CUDA device memory**, leaving the
    /// outputs in GPU memory — no H2D/D2H copies, no sync.
    ///
    /// `device_inputs`: `(tensor_name, cuda_device_ptr, shape)`. Returns a
    /// borrowed [`OutputView`] per output tensor: device pointer plus the
    /// resolved shape, dtype, and byte length. The views remain valid until the
    /// next `run_*` call or `Session` drop.
    ///
    /// **Caller must sync the shared stream before reading the outputs.**
    ///
    /// # Safety
    /// This does NOT sync — it enqueues async work and returns. The GPU reads the
    /// bound input device pointers during the caller's later stream sync, so
    /// every input buffer must stay valid until that sync (not merely until this
    /// call returns). Additionally the returned views alias Session-owned device
    /// memory — do not outlive the Session or hold them across a subsequent
    /// `run_*` call.
    pub unsafe fn run_device_inputs_on_device(
        &mut self,
        device_inputs: &[(&str, *mut std::ffi::c_void, &[i64])],
    ) -> Result<HashMap<String, OutputView>> {
        // A binding lasts exactly one run — including a run that FAILS. The clear has to
        // wrap every exit path, not just the successful one: `setInputShape`,
        // `setTensorAddress`, and the bound-buffer size check can all bail out, and a
        // caller who handles that error by dropping the result would otherwise leave a
        // live binding pointing at freed device memory. Recoverable errors are exactly
        // the ones a caller retries from, so this path is not hypothetical.
        let result = unsafe { self.run_bound_inputs(device_inputs) };
        self.bound.clear();
        result
    }

    /// The fallible body of [`run_device_inputs_on_device`](Self::run_device_inputs_on_device).
    ///
    /// # Safety
    /// Same contract as the caller: every input pointer must stay valid until the
    /// caller's next stream synchronize.
    unsafe fn run_bound_inputs(
        &mut self,
        device_inputs: &[(&str, *mut std::ffi::c_void, &[i64])],
    ) -> Result<HashMap<String, OutputView>> {
        self.device_memory.check_run()?;
        for (name, dev_ptr, shape) in device_inputs {
            let c_name =
                CString::new(*name).map_err(|_| TrtError::UnknownTensor((*name).into()))?;
            let rc = btrt_context_set_input_shape(
                self.ctx,
                c_name.as_ptr(),
                shape.as_ptr(),
                shape.len() as i32,
            );
            if rc != 0 {
                return Err(TrtError::Trt(last_trt_error()));
            }
            let rc = btrt_context_set_tensor_address(self.ctx, c_name.as_ptr(), *dev_ptr);
            if rc != 0 {
                return Err(TrtError::Trt(last_trt_error()));
            }
        }
        self.resize_output_buffers()?;
        self.enqueue_outputs_only()
    }

    fn enqueue_outputs_only(&mut self) -> Result<HashMap<String, OutputView>> {
        for (name, state) in &self.outputs {
            let c_name = CString::new(name.as_str()).unwrap();
            // A bound output redirects TensorRT at the caller's buffer, so the result
            // lands where it is needed and no copy-out is required.
            let dev_ptr = match self.bound.get(name) {
                Some(b) => b.ptr as usize as *mut std::ffi::c_void,
                None => state.buf.as_device_ptr(&self.stream),
            };
            let code =
                unsafe { btrt_context_set_tensor_address(self.ctx, c_name.as_ptr(), dev_ptr) };
            if code != 0 {
                return Err(TrtError::Trt(last_trt_error()));
            }
        }

        let code = unsafe { btrt_context_enqueue_v3(self.ctx, self.stream.as_raw()) };
        if code != 0 {
            return Err(TrtError::Trt(last_trt_error()));
        }

        let mut result = HashMap::new();
        for (name, state) in &self.outputs {
            // Borrows a Session-owned output buffer; the validity window
            // (until next run_* / Session drop) is the documented caller contract.
            // A bound output instead points at the caller's own buffer, which they
            // already own and which outlives the next run.
            let view = OutputView {
                ptr: match self.bound.get(name) {
                    Some(b) => b.ptr as usize as *mut std::ffi::c_void,
                    None => state.buf.as_device_ptr(&self.stream),
                },
                shape: state.shape.iter().map(|&d| d as usize).collect(),
                dtype: dtype_of(state.dtype),
            };
            result.insert(name.clone(), view);
        }
        // NOTE: bindings are cleared by `run_device_inputs_on_device`, which wraps this
        // whole call so a failed run drops them too. Clearing here would only cover the
        // success path. Safe to clear after this returns: the addresses are already set
        // in the execution context and the views built above already captured the
        // caller's pointers, so the in-flight run still writes where it was told.
        Ok(result)
    }

    fn resolved_output_shape(&self, name: &str) -> Result<Vec<i64>> {
        let c_name = CString::new(name).unwrap();
        let mut dims = [0i64; 8];
        let mut ndims = 0i32;
        let code = unsafe {
            btrt_context_get_tensor_shape(self.ctx, c_name.as_ptr(), dims.as_mut_ptr(), &mut ndims)
        };
        if code != 0 {
            return Err(TrtError::UnknownTensor(name.into()));
        }
        Ok(dims[..ndims as usize].to_vec())
    }

    fn resize_output_buffers(&mut self) -> Result<()> {
        let names: Vec<String> = self.outputs.keys().cloned().collect();
        for name in names {
            let shape = self.resolved_output_shape(&name)?;
            let dtype = self.outputs[&name].dtype;
            let new_len = buf_bytes(&shape, dtype);
            if let Some(b) = self.bound.get(&name) {
                // Caller owns this buffer: never reallocate it, but the resolved shape
                // is only known now, so this is the one place the size can be checked.
                // Failing here beats letting TensorRT overrun a short buffer.
                if b.bytes < new_len {
                    return Err(TrtError::Shape(format!(
                        "output '{name}': bound buffer is {} bytes but the resolved \
                         shape {shape:?} needs {new_len}",
                        b.bytes
                    )));
                }
            } else if self.outputs[&name].buf.len_bytes != new_len {
                self.outputs.get_mut(&name).unwrap().buf =
                    DeviceBuffer::alloc_with_stream(self.stream.cuda_stream(), new_len)?;
            }
            // Shape can change without the byte length changing (e.g. a
            // transposed dynamic profile) — always record the resolved shape.
            self.outputs.get_mut(&name).unwrap().shape = shape;
        }
        Ok(())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe {
                btrt_context_destroy(self.ctx);
            }
        }
    }
}

fn dtype_bytes(dtype: DataType) -> usize {
    match dtype {
        DataType::Float32 | DataType::Int32 => 4,
        DataType::Float16 | DataType::Bf16 => 2,
        DataType::Int8 | DataType::UInt8 | DataType::Bool => 1,
    }
}

/// Byte size of a tensor: product of positive dims (dynamic `-1`/`0` → 1) × dtype.
fn buf_bytes(dims: &[i64], dtype: DataType) -> usize {
    let n: i64 = dims.iter().filter(|&&d| d > 0).product::<i64>().max(1);
    n as usize * dtype_bytes(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_dtypes_map_to_io_tags() {
        assert_eq!(dtype_of(DataType::Float32), DType::F32);
        assert_eq!(dtype_of(DataType::Float16), DType::F16);
        // BF16 keeps its own tag: it must never be read as F16 or F32.
        assert_eq!(dtype_of(DataType::Bf16), DType::BF16);
        assert_eq!(dtype_of(DataType::Int32), DType::I32);
        assert_eq!(dtype_of(DataType::Int8), DType::U8);
        assert_eq!(dtype_of(DataType::UInt8), DType::U8);
        assert_eq!(dtype_of(DataType::Bool), DType::U8);
    }

    #[test]
    fn element_widths() {
        assert_eq!(dtype_bytes(DataType::Float32), 4);
        assert_eq!(dtype_bytes(DataType::Int32), 4);
        assert_eq!(dtype_bytes(DataType::Float16), 2);
        assert_eq!(dtype_bytes(DataType::Bf16), 2);
        assert_eq!(dtype_bytes(DataType::Int8), 1);
        assert_eq!(dtype_bytes(DataType::UInt8), 1);
        assert_eq!(dtype_bytes(DataType::Bool), 1);
    }

    #[test]
    fn buffer_size_is_product_of_static_dims_times_width() {
        assert_eq!(buf_bytes(&[1, 3, 4, 5], DataType::Float32), 240);
        assert_eq!(buf_bytes(&[2, 8], DataType::Float16), 32);
        // Dynamic (-1) and zero dims count as 1 until the shape is resolved.
        assert_eq!(buf_bytes(&[-1, 3, 4, 5], DataType::Float32), 240);
        assert_eq!(buf_bytes(&[0, 7], DataType::UInt8), 7);
        // Scalars and fully-dynamic tensors still get one element.
        assert_eq!(buf_bytes(&[], DataType::Int32), 4);
        assert_eq!(buf_bytes(&[-1, -1], DataType::Bf16), 2);
    }

    fn view(dtype: DType) -> OutputView {
        OutputView {
            ptr: 0x1000 as *mut c_void,
            shape: vec![1, 2, 3],
            dtype,
        }
    }

    #[test]
    fn output_view_typed_pointers_check_dtype() {
        assert_eq!(view(DType::F32).f32_ptr().unwrap() as usize, 0x1000);
        assert_eq!(view(DType::I32).i32_ptr().unwrap() as usize, 0x1000);
        for wrong in [DType::F16, DType::BF16, DType::U8, DType::I32] {
            assert!(matches!(view(wrong).f32_ptr(), Err(TrtError::Shape(_))));
        }
        for wrong in [DType::F32, DType::F16, DType::BF16, DType::U8] {
            assert!(matches!(view(wrong).i32_ptr(), Err(TrtError::Shape(_))));
        }
        assert_eq!(view(DType::F32).shape_i64(), vec![1i64, 2, 3]);
    }

    const UNSET: DeviceMemory = DeviceMemory::UserManaged { set: false };
    const SET: DeviceMemory = DeviceMemory::UserManaged { set: true };
    const PTR: usize = 0x7f00_0000; // 256-aligned

    fn needs(n: usize) -> impl FnOnce() -> Result<usize> {
        move || Ok(n)
    }

    fn is_dev_mem_err<T>(r: Result<T>) -> bool {
        matches!(r, Err(TrtError::DeviceMemory(_)))
    }

    #[test]
    fn run_requires_device_memory_on_user_managed_sessions() {
        assert!(is_dev_mem_err(UNSET.check_run()));
        assert!(SET.check_run().is_ok());
        assert!(DeviceMemory::Owned.check_run().is_ok());
    }

    #[test]
    fn bind_is_rejected_on_owned_sessions_without_querying_the_engine() {
        let r = DeviceMemory::Owned.check_bind(PTR, 1024, || {
            panic!("device_memory_size must not be queried for an owned session")
        });
        assert!(is_dev_mem_err(r));
    }

    #[test]
    fn bind_accepts_an_exact_or_larger_buffer_and_makes_the_session_runnable() {
        for state in [UNSET, SET] {
            let (next, len) = state.check_bind(PTR, 1024, needs(1024)).unwrap();
            assert_eq!(len, 1024);
            assert!(next.check_run().is_ok());
            assert_eq!(state.check_bind(PTR, 4096, needs(1024)).unwrap().1, 4096);
        }
    }

    #[test]
    fn bind_rejects_a_short_buffer() {
        assert!(is_dev_mem_err(UNSET.check_bind(PTR, 1023, needs(1024))));
        assert!(is_dev_mem_err(UNSET.check_bind(PTR, 0, needs(1))));
    }

    #[test]
    fn bind_rejects_null_unless_the_engine_needs_zero_bytes() {
        assert!(is_dev_mem_err(UNSET.check_bind(0, 1024, needs(1024))));
        assert!(is_dev_mem_err(UNSET.check_bind(0, 0, needs(1024))));
        // The bridge accepts NULL only with 0 bytes.
        assert!(is_dev_mem_err(UNSET.check_bind(0, 64, needs(0))));
        // (null, 0) for an engine that needs nothing: valid, and runnable after.
        let (next, len) = UNSET.check_bind(0, 0, needs(0)).unwrap();
        assert_eq!(len, 0);
        assert!(next.check_run().is_ok());
    }

    #[test]
    fn bind_rejects_a_misaligned_pointer() {
        assert!(is_dev_mem_err(UNSET.check_bind(
            PTR + 16,
            1024,
            needs(1024)
        )));
        assert!(UNSET.check_bind(PTR + 256, 1024, needs(1024)).is_ok());
    }

    #[test]
    fn bind_propagates_a_failed_size_query() {
        let r = UNSET.check_bind(PTR, 1024, || Err(TrtError::Trt("boom".into())));
        assert!(matches!(r, Err(TrtError::Trt(_))));
    }

    #[test]
    fn bind_rejects_sizes_that_overflow_i64() {
        let big = i64::MAX as usize + 1;
        assert!(is_dev_mem_err(UNSET.check_bind(PTR, big, needs(0))));
    }
}
