//! Safe, idiomatic Rust wrapper for TensorRT 10.x, built on
//! [`tensorrt_sys`] (package `kornia-tensorrt-sys`).
//!
//! # Usage
//! Load an engine, then run inference through a [`Session`] on a shared CUDA
//! stream (one async enqueue per call; the caller syncs the stream):
//! ```no_run
//! use tensorrt::{Engine, Logger, Runtime, Session, Stream};
//! use tensorrt::logger::Severity;
//!
//! let logger  = Logger::new(Severity::Warning)?;
//! let runtime = Runtime::new(logger)?;
//! let engine  = Engine::from_file(runtime, "model.fp16.engine")?;
//! let stream  = Stream::new_standalone()?;
//! let mut session = Session::with_stream(engine, stream.cuda_stream().clone())?;
//! // let outputs = unsafe { session.run_device_inputs_on_device(&[("input", dev_ptr, &shape)]) }?;
//! // stream.sync()?;
//! # Ok::<(), tensorrt::error::TrtError>(())
//! ```
//!
//! # Ownership
//! `Logger ← Arc ← Runtime ← Arc ← Engine ← Arc ← Session`: each child holds an
//! `Arc` to its parent, so TensorRT objects are destroyed in the order NVIDIA
//! requires by construction.
//!
//! # Thread safety
//! - `Logger`, `Runtime` and `Engine` are `Send + Sync` — safe to share across threads.
//! - `Session` is `Send` but **not `Sync`** — IExecutionContext is not thread-safe.
//!   Create one `Session` per thread from a shared `Arc<Engine>`.
//!
//! # Shared activation memory
//! Engines that always run one after another can share one activation buffer
//! instead of each context owning its own: size it with
//! [`Engine::device_memory_size`], create the sessions with
//! [`Session::with_stream_user_memory`] and hand each the same buffer through
//! [`Session::set_device_memory`].

pub mod buffer;
#[cfg(feature = "builder")]
pub mod builder;
pub mod dtype;
pub mod engine;
pub mod error;
pub mod logger;
pub mod runtime;
pub mod session;

/// Boxed, thread-safe error — the convenient return type for callers that
/// aggregate several error kinds.
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

pub use buffer::{DeviceBuffer, PinnedBuffer, Stream};
pub use cudarc;
pub use cudarc::driver::CudaStream;
pub use dtype::{DType, Precision};
pub use engine::{DataType, Engine, TensorMode, TensorSpec};
pub use error::{Result, TrtError};
pub use logger::Logger;
pub use runtime::Runtime;
pub use session::{OutputView, Session};
pub use tensorrt_sys as sys;
pub use tensorrt_sys::TENSORRT_VERSION;

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time "does NOT implement" check (the `static_assertions`
    /// `assert_not_impl_any!` trick): if `T: Sync`, the call below is
    /// ambiguous between the two impls and the crate fails to compile.
    macro_rules! assert_not_sync {
        ($t:ty) => {
            const _: fn() = || {
                trait AmbiguousIfSync<A> {
                    fn some_item() {}
                }
                impl<T: ?Sized> AmbiguousIfSync<()> for T {}
                #[allow(dead_code)]
                struct Invalid;
                impl<T: ?Sized + Sync> AmbiguousIfSync<Invalid> for T {}
                let _ = <$t as AmbiguousIfSync<_>>::some_item;
            };
        };
    }

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_not_sync!(Session);

    #[test]
    fn thread_safety_matches_the_tensorrt_contract() {
        assert_send::<Logger>();
        assert_sync::<Logger>();
        assert_send::<Runtime>();
        assert_sync::<Runtime>();
        assert_send::<Engine>();
        assert_sync::<Engine>();
        assert_send::<Session>();
        assert_send::<Stream>();
        assert_send::<DeviceBuffer>();
        assert_send::<PinnedBuffer<f32>>();
        assert_send::<OutputView>();
        // Session: !Sync is checked at compile time by `assert_not_sync!` above.
    }

    #[test]
    fn tensorrt_version_is_exported() {
        assert!(!TENSORRT_VERSION.is_empty());
        assert_eq!(TENSORRT_VERSION.split('.').count(), 4);
    }
}
