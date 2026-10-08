//! Real-TensorRT test for user-managed context device memory
//! (btrt_context_create_user_memory / btrt_engine_device_memory_size /
//! btrt_context_set_device_memory).
//!
//! Needs a GPU, TensorRT and the `builder` feature, so it is `#[ignore]`d and
//! compiled out in stub mode. Run on the Jetson with:
//!
//! ```text
//! cargo test -p tensorrt-rs --features builder --test user_device_memory -- --ignored
//! ```
//!
//! The model defaults to the TensorRT sample MNIST net; override with
//! `BTRT_TEST_ONNX` (static-shape, one f32 input, one f32 output).
#![cfg(all(feature = "builder", not(trt_stub)))]

use std::ffi::{c_void, CStr, CString};
use std::ptr;
use tensorrt_rs::*;

extern "C" {
    fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> i32;
    fn cudaFree(p: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
    fn cudaMemset(p: *mut c_void, value: i32, bytes: usize) -> i32;
    fn cudaStreamCreate(s: *mut *mut c_void) -> i32;
    fn cudaStreamSynchronize(s: *mut c_void) -> i32;
    fn cudaStreamDestroy(s: *mut c_void) -> i32;
}
const H2D: i32 = 1;
const D2H: i32 = 2;

fn last_error() -> String {
    unsafe { CStr::from_ptr(btrt_last_error()) }
        .to_string_lossy()
        .into_owned()
}

fn dev_alloc(bytes: usize) -> *mut c_void {
    let mut p = ptr::null_mut();
    assert_eq!(unsafe { cudaMalloc(&mut p, bytes.max(1)) }, 0, "cudaMalloc");
    p
}

struct Io {
    input: String,
    output: String,
    in_elems: usize,
    out_elems: usize,
}

unsafe fn io_of(engine: *mut btrt_engine_t) -> Io {
    let (mut input, mut output) = (None, None);
    for i in 0..btrt_engine_num_io_tensors(engine) {
        let name = CStr::from_ptr(btrt_engine_io_tensor_name(engine, i));
        let mut dims = [0i64; 8];
        let mut nd = 0i32;
        assert_eq!(
            btrt_engine_tensor_shape(engine, name.as_ptr(), dims.as_mut_ptr(), &mut nd),
            0
        );
        let elems: i64 = dims[..nd as usize].iter().product();
        assert!(elems > 0, "test model must have static shapes");
        assert_eq!(
            btrt_engine_tensor_dtype(engine, name.as_ptr()),
            0,
            "f32 I/O only"
        );
        let entry = (name.to_string_lossy().into_owned(), elems as usize);
        match btrt_engine_tensor_io_mode(engine, name.as_ptr()) {
            1 => input = Some(entry),
            2 => output = Some(entry),
            m => panic!("unexpected io mode {m}"),
        }
    }
    let (input, in_elems) = input.expect("one input");
    let (output, out_elems) = output.expect("one output");
    Io {
        input,
        output,
        in_elems,
        out_elems,
    }
}

/// Bind I/O, enqueue, sync, return the output. `Err` carries the enqueue error.
unsafe fn run(
    ctx: *mut btrt_context_t,
    io: &Io,
    d_in: *mut c_void,
    stream: *mut c_void,
) -> Result<Vec<f32>, String> {
    let d_out = dev_alloc(io.out_elems * 4);
    assert_eq!(cudaMemset(d_out, 0, io.out_elems * 4), 0);
    let cin = CString::new(io.input.as_str()).unwrap();
    let cout = CString::new(io.output.as_str()).unwrap();
    assert_eq!(btrt_context_set_tensor_address(ctx, cin.as_ptr(), d_in), 0);
    assert_eq!(
        btrt_context_set_tensor_address(ctx, cout.as_ptr(), d_out),
        0
    );
    let rc = btrt_context_enqueue_v3(ctx, stream);
    assert_eq!(cudaStreamSynchronize(stream), 0);
    let res = if rc == 0 {
        let mut out = vec![0f32; io.out_elems];
        assert_eq!(
            cudaMemcpy(out.as_mut_ptr().cast(), d_out, io.out_elems * 4, D2H),
            0
        );
        Ok(out)
    } else {
        Err(last_error())
    };
    cudaFree(d_out);
    res
}

#[test]
#[ignore = "needs a GPU + TensorRT (run on the Jetson with --ignored)"]
fn user_managed_memory_matches_default_context() {
    let onnx = std::env::var("BTRT_TEST_ONNX")
        .unwrap_or_else(|_| "/usr/src/tensorrt/data/mnist/mnist.onnx".into());
    unsafe {
        let logger = btrt_logger_create(2); // kWARNING
        assert!(!logger.is_null());

        let path = CString::new(onnx.clone()).unwrap();
        let (mut blob, mut len) = (ptr::null_mut(), 0usize);
        let rc = btrt_build_engine_from_onnx(
            logger,
            path.as_ptr(),
            0,
            0,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            0,
            64 << 20,
            &mut blob,
            &mut len,
        );
        assert_eq!(rc, 0, "build {onnx}: {}", last_error());

        let rt = btrt_runtime_create(logger);
        assert!(!rt.is_null(), "{}", last_error());
        let engine = btrt_engine_deserialize(rt, blob.cast(), len);
        btrt_blob_free(blob);
        assert!(!engine.is_null(), "{}", last_error());
        let io = io_of(engine);

        let host_in: Vec<f32> = (0..io.in_elems).map(|i| (i % 17) as f32 / 17.0).collect();
        let d_in = dev_alloc(io.in_elems * 4);
        assert_eq!(
            cudaMemcpy(d_in, host_in.as_ptr().cast(), io.in_elems * 4, H2D),
            0
        );
        let mut stream = ptr::null_mut();
        assert_eq!(cudaStreamCreate(&mut stream), 0);

        let need = btrt_engine_device_memory_size(engine);
        assert!(need >= 0, "{}", last_error());

        // Reference: context that allocates its own memory.
        let ctx_default = btrt_context_create(engine);
        assert!(!ctx_default.is_null(), "{}", last_error());
        let expected = run(ctx_default, &io, d_in, stream).expect("default enqueue");

        // A default (static) context must refuse user memory.
        let scratch = dev_alloc(need as usize);
        assert_eq!(
            btrt_context_set_device_memory(ctx_default, scratch, need),
            -1
        );
        assert!(
            last_error().contains("not created with"),
            "{}",
            last_error()
        );

        let ctx_user = btrt_context_create_user_memory(engine);
        assert!(!ctx_user.is_null(), "{}", last_error());

        // The header promises enqueue fails while no buffer is bound.
        if need > 0 {
            assert!(
                run(ctx_user, &io, d_in, stream).is_err(),
                "enqueue on an unbound user-memory context must fail"
            );
        }

        // Argument validation, each with its own message.
        assert_eq!(btrt_context_set_device_memory(ctx_user, scratch, -1), -1);
        assert!(last_error().contains("negative"), "{}", last_error());
        assert_eq!(
            btrt_context_set_device_memory(ctx_user, ptr::null_mut(), 1),
            -1
        );
        assert!(last_error().contains("null ptr"), "{}", last_error());
        if need > 0 {
            assert_eq!(
                btrt_context_set_device_memory(ctx_user, scratch, need - 1),
                -1,
                "undersized buffer must be rejected"
            );
            assert!(last_error().contains("smaller"), "{}", last_error());
            assert_eq!(
                btrt_context_set_device_memory(ctx_user, ptr::null_mut(), 0),
                -1,
                "null/0 must be rejected when the engine needs memory"
            );
        } else {
            assert_eq!(
                btrt_context_set_device_memory(ctx_user, ptr::null_mut(), 0),
                0,
                "null/0 is valid when the engine needs 0 bytes: {}",
                last_error()
            );
        }

        // Exactly getDeviceMemorySizeV2 bytes: binds, enqueues, same result.
        assert_eq!(
            btrt_context_set_device_memory(ctx_user, scratch, need),
            0,
            "{}",
            last_error()
        );
        assert!(last_error().is_empty(), "success must clear the error");
        let got = run(ctx_user, &io, d_in, stream).expect("user-memory enqueue");
        assert_eq!(got, expected, "user-managed memory changed the result");

        cudaStreamDestroy(stream);
        cudaFree(scratch);
        cudaFree(d_in);
        btrt_context_destroy(ctx_user);
        btrt_context_destroy(ctx_default);
        btrt_engine_destroy(engine);
        btrt_runtime_destroy(rt);
        btrt_logger_destroy(logger);
    }
}
