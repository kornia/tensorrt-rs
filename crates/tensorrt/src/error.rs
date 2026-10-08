use thiserror::Error;

#[derive(Debug, Error)]
pub enum TrtError {
    #[error("failed to create {0}")]
    Create(&'static str),
    #[error("engine deserialization failed (TRT version/arch mismatch? engine must be built on this exact Jetson with the same TRT version): {0}")]
    Deserialize(String),
    #[error("tensor '{0}' not found in engine")]
    UnknownTensor(String),
    #[error("shape/dtype mismatch: {0}")]
    Shape(String),
    #[error("CUDA error code {code}: {msg}")]
    Cuda { code: i32, msg: &'static str },
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("TensorRT error: {0}")]
    Trt(String),
}

pub type Result<T> = std::result::Result<T, TrtError>;

// Helper: pull the thread-local last error from the shim.
pub(crate) fn last_trt_error() -> String {
    unsafe {
        let ptr = tensorrt_sys::btrt_last_error();
        if ptr.is_null() {
            return String::new();
        }
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages() {
        assert_eq!(
            TrtError::Create("Runtime").to_string(),
            "failed to create Runtime"
        );
        assert_eq!(
            TrtError::UnknownTensor("images".into()).to_string(),
            "tensor 'images' not found in engine"
        );
        assert_eq!(
            TrtError::Shape("bad rank".into()).to_string(),
            "shape/dtype mismatch: bad rank"
        );
        assert_eq!(
            TrtError::Cuda {
                code: 2,
                msg: "cudaMalloc"
            }
            .to_string(),
            "CUDA error code 2: cudaMalloc"
        );
        assert_eq!(
            TrtError::Trt("boom".into()).to_string(),
            "TensorRT error: boom"
        );
        // A stale engine cache must be distinguishable from a bad shape, and the
        // message must point at the usual cause.
        let e = TrtError::Deserialize("magic mismatch".into()).to_string();
        assert!(e.starts_with("engine deserialization failed"), "{e}");
        assert!(e.ends_with(": magic mismatch"), "{e}");
    }

    #[test]
    fn errors_are_send_sync_static() {
        fn assert_bounds<T: std::error::Error + Send + Sync + 'static>() {}
        assert_bounds::<TrtError>();
    }
}
