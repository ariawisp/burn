use crate as burn;

use crate::module::{Content, DisplaySettings, Module, ModuleDisplay};
use crate::nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig};
use crate::{
    config::Config,
    tensor::{Tensor, backend::Backend},
};

use super::AttnWindow;
use burn_tensor::activation::{quiet_softmax, softmax};

/// Configuration for streaming multi-query attention (MQA/GQA).
#[derive(Config, Debug)]
pub struct StreamingMultiQueryAttentionConfig {
    /// Model dimension per token (input/output features).
    pub d_model: usize,
    /// Total number of attention heads for Q/O.
    pub n_heads: usize,
    /// Number of shared key/value heads.
    pub num_key_value_heads: usize,
    /// Dropout probability on attention logits.
    #[config(default = 0.0)]
    pub dropout: f64,
    /// Use quiet softmax instead of regular softmax.
    #[config(default = false)]
    pub quiet_softmax: bool,
    /// Parameter initializer for linear layers.
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Streaming multi-query attention with KV cache and sliding window.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct StreamingMultiQueryAttention<B: Backend> {
    /// Query projection: `[d_model -> d_model]`.
    pub query: Linear<B>,
    /// Key projection: `[d_model -> num_key_value_heads * head_dim]`.
    pub key: Linear<B>,
    /// Value projection: `[d_model -> num_key_value_heads * head_dim]`.
    pub value: Linear<B>,
    /// Output projection: `[d_model -> d_model]`.
    pub output: Linear<B>,
    /// Dropout on attention logits.
    pub dropout: Dropout,
    /// Model dimension per token (input/output features).
    pub d_model: usize,
    /// Total number of Q heads.
    pub n_heads: usize,
    /// Number of K/V heads.
    pub kv_heads: usize,
    /// Head dimension (d_model / n_heads).
    pub d_k: usize,
    /// Use quiet softmax.
    pub quiet_softmax: bool,
}

impl<B: Backend> ModuleDisplay for StreamingMultiQueryAttention<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("d_model", &self.d_model)
            .add("n_heads", &self.n_heads)
            .add("kv_heads", &self.kv_heads)
            .add("d_k", &self.d_k)
            .add("dropout", &self.dropout.prob)
            .add("quiet_softmax", &self.quiet_softmax)
            .optional()
    }
}

impl StreamingMultiQueryAttentionConfig {
    /// Initialize a new streaming MQA module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> StreamingMultiQueryAttention<B> {
        assert!(
            self.d_model % self.n_heads == 0,
            "d_model must be divisible by n_heads"
        );
        assert!(
            self.n_heads % self.num_key_value_heads == 0,
            "n_heads must be divisible by num_key_value_heads"
        );

        let d_k = self.d_model / self.n_heads;

        let linear = |in_features, out_features| {
            LinearConfig::new(in_features, out_features)
                .with_initializer(self.initializer.clone())
                .init(device)
        };

        StreamingMultiQueryAttention {
            query: linear(self.d_model, self.d_model),
            key: linear(self.d_model, self.num_key_value_heads * d_k),
            value: linear(self.d_model, self.num_key_value_heads * d_k),
            output: linear(self.d_model, self.d_model),
            dropout: DropoutConfig::new(self.dropout).init(),
            d_model: self.d_model,
            n_heads: self.n_heads,
            kv_heads: self.num_key_value_heads,
            d_k,
            quiet_softmax: self.quiet_softmax,
        }
    }
}

/// Streaming KV cache for MQA.
pub struct StreamingMqaCache<B: Backend> {
    /// Key buffer: `[batch, cache_len, kv_heads, d_k]`.
    pub k: Tensor<B, 4>,
    /// Value buffer: `[batch, cache_len, kv_heads, d_k]`.
    pub v: Tensor<B, 4>,
    /// Absolute position after the latest write (exclusive).
    pub global_end_index: usize,
    /// Local buffer end index (exclusive).
    pub local_end_index: usize,
    /// Number of sink tokens preserved at the beginning of the buffer.
    pub sink_tokens: usize,
    /// Maximum capacity of the rolling buffer.
    pub cache_len: usize,
}

impl<B: Backend> StreamingMqaCache<B> {
    /// Create an empty cache with given capacity.
    pub fn new(
        device: &B::Device,
        batch: usize,
        cache_len: usize,
        kv_heads: usize,
        head_dim: usize,
        sink_tokens: usize,
    ) -> Self {
        let zeros_k = Tensor::<B, 4>::zeros([batch, cache_len, kv_heads, head_dim], device);
        let zeros_v = Tensor::<B, 4>::zeros([batch, cache_len, kv_heads, head_dim], device);

        Self {
            k: zeros_k,
            v: zeros_v,
            global_end_index: 0,
            local_end_index: 0,
            sink_tokens,
            cache_len,
        }
    }

    /// Resets indices while keeping allocated buffers.
    pub fn reset(&mut self) {
        self.global_end_index = 0;
        self.local_end_index = 0;
    }

    /// Current number of valid tokens stored in the cache.
    pub fn len(&self) -> usize {
        self.local_end_index
    }

    /// Whether the cache currently holds no tokens.
    pub fn is_empty(&self) -> bool {
        self.local_end_index == 0
    }

    /// Cache capacity (in tokens).
    pub fn capacity(&self) -> usize {
        self.cache_len
    }

    /// Whether the cache is full.
    pub fn is_full(&self) -> bool {
        self.local_end_index >= self.cache_len
    }

    /// Zero-out K and V buffers in place and reset indices.
    pub fn clear(&mut self) {
        let device = self.k.device();
        let [b, cap, h, d] = self.k.dims();
        self.k = Tensor::<B, 4>::zeros([b, cap, h, d], &device);
        self.v = Tensor::<B, 4>::zeros([b, cap, h, d], &device);
        self.reset();
    }
}

/// Parameters for streaming MQA forward.
pub struct StreamingMqaParams<'a, B: Backend> {
    /// Optional rotary encoding to apply to Q and K with an absolute start offset.
    pub rope: Option<&'a crate::nn::rope_encoding::RotaryEncoding<B>>,
    /// Absolute position of the first token in the current chunk.
    pub start_pos: usize,
    /// Window selection policy.
    pub window: AttnWindow,
    /// Optional sinks logits per (kv_head, group) pair, shape `[kv_heads, groups]`.
    pub sinks: Option<&'a Tensor<B, 2>>,
    /// Optional additive attention bias with shape `[B, n_heads, q_len, k_len]`,
    /// where `k_len` must match the active window length.
    pub attn_bias: Option<&'a Tensor<B, 4>>,
}

impl<B: Backend> StreamingMultiQueryAttention<B> {
    /// Forward with streaming KV cache and optional windowing and sinks bias.
    ///
    /// Inputs are self-attention by default (query=key=value).
    pub fn forward_streaming(
        &self,
        x: Tensor<B, 3>,
        cache: &mut StreamingMqaCache<B>,
        params: StreamingMqaParams<B>,
    ) -> Tensor<B, 3> {
        debug_assert!(
            cache.sink_tokens <= cache.cache_len,
            "sink tokens exceed cache capacity"
        );
        let [batch_size, seq_len, _] = x.dims();
        let groups = self.n_heads / self.kv_heads;

        // Projections
        // Q -> [B, nH, T, d_k]
        let q = self.attention_linear_q(x.clone(), &self.query);
        // K,V -> [B, kvH, T, d_k] (after swap)
        let k = self.attention_linear_kv(x.clone(), &self.key);
        let v = self.attention_linear_kv(x, &self.value);

        // Optionally apply RoPE with offset to Q and K (last two dims are [heads, d_k]).
        let (q, k) = if let Some(rope) = params.rope {
            debug_assert!(self.d_k % 2 == 0, "RoPE requires even head_dim");
            // reshape to [B, T, heads, d_k] for rope apply along sequence
            let q_rs = q.swap_dims(1, 2); // [B, T, nH, d_k]
            let k_rs = k.swap_dims(1, 2); // [B, T, kvH, d_k]
            let q_ro = rope.apply(q_rs, params.start_pos);
            let k_ro = rope.apply(k_rs, params.start_pos);
            (q_ro.swap_dims(1, 2), k_ro.swap_dims(1, 2))
        } else {
            (q, k)
        };

        // Update rolling cache with new K/V tokens
        let num_new = seq_len;
        let current_end = params.start_pos + num_new;
        debug_assert!(
            current_end >= cache.global_end_index,
            "start_pos must be non-decreasing"
        );
        let sink = cache.sink_tokens;
        let cap = cache.cache_len;
        let delta = current_end.saturating_sub(cache.global_end_index);
        let need = cache.local_end_index + delta;

        if need > cap {
            let num_evicted = need - cap;
            let num_rolled = cache.local_end_index.saturating_sub(num_evicted + sink);
            if num_rolled > 0 {
                let src_start = sink + num_evicted;
                let src_end = sink + num_evicted + num_rolled;
                // Roll K
                let rolled = cache.k.clone().slice([
                    0..batch_size,
                    src_start..src_end,
                    0..self.kv_heads,
                    0..self.d_k,
                ]);
                cache.k.inplace(|t| {
                    t.slice_assign(
                        [
                            0..batch_size,
                            sink..sink + num_rolled,
                            0..self.kv_heads,
                            0..self.d_k,
                        ],
                        rolled,
                    )
                });
                // Roll V
                let rolled_v = cache.v.clone().slice([
                    0..batch_size,
                    src_start..src_end,
                    0..self.kv_heads,
                    0..self.d_k,
                ]);
                cache.v.inplace(|t| {
                    t.slice_assign(
                        [
                            0..batch_size,
                            sink..sink + num_rolled,
                            0..self.kv_heads,
                            0..self.d_k,
                        ],
                        rolled_v,
                    )
                });
            }
            cache.local_end_index = cache.local_end_index + delta - num_evicted;
        } else {
            cache.local_end_index += delta;
        }

        // Write new K,V at the end
        let local_end = cache.local_end_index;
        let local_start = local_end - num_new;
        let k_rs = k.swap_dims(1, 2); // [B, T, kvH, d_k]
        let v_rs = v.swap_dims(1, 2);
        cache.k.inplace(|t| {
            t.slice_assign(
                [
                    0..batch_size,
                    local_start..local_end,
                    0..self.kv_heads,
                    0..self.d_k,
                ],
                k_rs,
            )
        });
        cache.v.inplace(|t| {
            t.slice_assign(
                [
                    0..batch_size,
                    local_start..local_end,
                    0..self.kv_heads,
                    0..self.d_k,
                ],
                v_rs,
            )
        });
        cache.global_end_index = current_end;

        // Determine the active window
        let active_len = match params.window {
            AttnWindow::Full => local_end,
            AttnWindow::Window(w) => {
                debug_assert!(
                    cache.sink_tokens + w <= cache.cache_len,
                    "window+sink exceeds cache capacity"
                );
                sink + w.min(local_end.saturating_sub(sink))
            }
        };
        let start = local_end.saturating_sub(active_len);

        // Gather window K,V from cache: [B, kvH, Tk, d_k]
        let k_win = cache
            .k
            .clone()
            .slice([
                0..batch_size,
                start..local_end,
                0..self.kv_heads,
                0..self.d_k,
            ])
            .swap_dims(1, 2);
        let v_win = cache
            .v
            .clone()
            .slice([
                0..batch_size,
                start..local_end,
                0..self.kv_heads,
                0..self.d_k,
            ])
            .swap_dims(1, 2);

        // Expand KV across groups to match Q heads: [B, nH, Tk/d_k]
        let k_exp = k_win
            .unsqueeze_dim::<5>(2) // [B, kvH, 1, Tk, d_k]
            .repeat_dim(2, groups) // [B, kvH, groups, Tk, d_k]
            .reshape([batch_size, self.n_heads, active_len, self.d_k]);
        let v_exp = v_win
            .unsqueeze_dim::<5>(2)
            .repeat_dim(2, groups)
            .reshape([batch_size, self.n_heads, active_len, self.d_k]);

        // Attention
        let mut attn_scores = q
            .matmul(k_exp.transpose())
            .div_scalar((self.d_k as f32).sqrt());
        attn_scores = self.dropout.forward(attn_scores);

        // Additive attention bias (already shaped to window)
        if let Some(bias) = params.attn_bias {
            attn_scores = attn_scores + bias.clone();
        }

        // Optional sinks bias: append as a sentinel column and then discard after softmax.
        let weights = if let Some(sinks) = params.sinks {
            // sinks: [kvH, groups] -> [nH]
            let sinks = sinks.clone().reshape([self.kv_heads * groups]);
            // [1, nH, 1, 1] -> [B, nH, Tq, 1]
            let mut s = sinks.reshape([1, self.n_heads, 1, 1]);
            s = s.repeat_dim(0, batch_size);
            s = s.repeat_dim(2, seq_len);
            // concat on last dim
            let attn_scores_cat = Tensor::cat(vec![attn_scores.clone(), s], 3);
            let w_all = if self.quiet_softmax {
                quiet_softmax(attn_scores_cat, 3)
            } else {
                softmax(attn_scores_cat, 3)
            };
            // discard sink column
            w_all.slice([0..batch_size, 0..self.n_heads, 0..seq_len, 0..active_len])
        } else if self.quiet_softmax {
            quiet_softmax(attn_scores, 3)
        } else {
            softmax(attn_scores, 3)
        };

        let context = weights.matmul(v_exp);
        // [B, nH, Tq, d_k] -> [B, Tq, nH, d_k] -> [B, Tq, d_model]
        let context = context
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, self.d_model]);
        self.output.forward(context)
    }

    fn attention_linear_q(&self, x: Tensor<B, 3>, linear: &Linear<B>) -> Tensor<B, 4> {
        let [batch_size, seq_length, _d_model] = x.dims();
        linear
            .forward(x)
            .reshape([batch_size, seq_length, self.n_heads, self.d_k])
            .swap_dims(1, 2)
    }

    fn attention_linear_kv(&self, x: Tensor<B, 3>, linear: &Linear<B>) -> Tensor<B, 4> {
        let [batch_size, seq_length, _d_model] = x.dims();
        linear
            .forward(x)
            .reshape([batch_size, seq_length, self.kv_heads, self.d_k])
            .swap_dims(1, 2)
    }
}

// Tests for this module are provided under crates/burn-core/tests/attention_mqa_streaming.rs
