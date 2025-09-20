use crate::backend::Backend;
use crate::tensor::ops::{BoolTensor, Device, FloatTensor, IntTensor};
use crate::TensorData;

/// Optional batch-creation hook that backends can implement to create many tensors
/// in a single allocation/write sequence. Backends that don't implement this trait
/// can still work by falling back to per-tensor creation.
pub trait BatchTensorOps: Backend {
    /// Create many float tensors at once.
    fn float_batch_from_data(items: Vec<(TensorData, Device<Self>)>) -> Vec<FloatTensor<Self>>;
    /// Create many int tensors at once.
    fn int_batch_from_data(items: Vec<(TensorData, Device<Self>)>) -> Vec<IntTensor<Self>>;
    /// Create many bool tensors at once.
    fn bool_batch_from_data(items: Vec<(TensorData, Device<Self>)>) -> Vec<BoolTensor<Self>>;
}

