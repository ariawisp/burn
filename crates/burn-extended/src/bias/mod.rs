use burn_core as burn;

use burn::tensor::{Int, Tensor, backend::Backend};

/// Build sinks bias from a flat per-head vector shaped [n_heads] into [kv_heads, groups].
pub fn sinks_from_per_head<B: Backend>(per_head: Tensor<B, 1>, kv_heads: usize, groups: usize) -> Tensor<B, 2> {
    per_head.reshape([kv_heads, groups])
}

/// Create ALiBi additive attention bias of shape [B, n_heads, q_len, k_len].
/// Slopes can be provided; if None, a simple decreasing linear set is used.
pub fn alibi_bias<B: Backend>(
    batch: usize,
    n_heads: usize,
    q_len: usize,
    k_len: usize,
    slopes: Option<&[f32]>,
    device: &B::Device,
) -> Tensor<B, 4> {
    let s = if let Some(s) = slopes { s.to_vec() } else { default_slopes(n_heads) };
    let slopes = Tensor::<B, 1>::from_floats(s, device).reshape([n_heads, 1, 1]);
    let q = Tensor::<B, 1, Int>::arange(0..q_len as i64, device).float().reshape([q_len, 1]);
    let k = Tensor::<B, 1, Int>::arange(0..k_len as i64, device).float().reshape([1, k_len]);
    let dist = q - k; // [q_len, k_len]
    let dist = dist.reshape([1, 1, q_len, k_len]).repeat_dim(0, batch).repeat_dim(1, n_heads);
    slopes.unsqueeze().repeat_dim(0, batch).mul(dist)
}

fn default_slopes(n_heads: usize) -> Vec<f32> {
    // Simple monotonic decreasing slopes; for exact ALiBi use known formulas.
    let mut out = Vec::with_capacity(n_heads);
    for i in 0..n_heads { out.push(1.0 / (1.0 + i as f32)); }
    out
}

