pub mod streaming_mqa;
pub mod mqa;
pub mod streaming_ext;

// Re-export common types for convenience
pub use streaming_mqa::*;
pub use mqa::*;
pub use streaming_ext::*;

// Convenience: re-export AttnWindow from burn-core
pub use burn::nn::attention::AttnWindow;
pub use burn::nn::attention::StreamingMhaCache;

