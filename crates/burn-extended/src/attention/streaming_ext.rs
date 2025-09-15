use burn_core as burn;

use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig};
use burn::{
    config::Config,
    tensor::{Tensor, backend::Backend},
};

use super::AttnWindow;
use burn_tensor::activation::{quiet_softmax, softmax};

/// Configuration for the extended streaming multi-head attention module.
#[derive(Config, Debug)]
pub struct ExtStreamingMultiHeadAttentionConfig {
    pub d_model: usize,
    pub n_heads: usize,
    #[config(default = 0.0)]
    pub dropout: f64,
    #[config(default = false)]
    pub quiet_softmax: bool,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Extended streaming multi-head attention with optional additive attn_bias.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct ExtStreamingMultiHeadAttention<B: Backend> {
    pub query: Linear<B>,
    pub key: Linear<B>,
    pub value: Linear<B>,
    pub output: Linear<B>,
    pub dropout: Dropout,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_k: usize,
    pub quiet_softmax: bool,
}

impl<B: Backend> ModuleDisplay for ExtStreamingMultiHeadAttention<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("d_model", &self.d_model)
            .add("n_heads", &self.n_heads)
            .add("d_k", &self.d_k)
            .add("dropout", &self.dropout.prob)
            .add("quiet_softmax", &self.quiet_softmax)
            .optional()
    }
}

impl ExtStreamingMultiHeadAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ExtStreamingMultiHeadAttention<B> {
        let linear = |in_features, out_features| {
            LinearConfig::new(in_features, out_features)
                .with_initializer(self.initializer.clone())
                .init(device)
        };

        assert!(
            self.d_model % self.n_heads == 0,
            "d_model must be divisible by n_heads"
        );
        let d_k = self.d_model / self.n_heads;

        ExtStreamingMultiHeadAttention {
            query: linear(self.d_model, self.d_model),
            key: linear(self.d_model, self.d_model),
            value: linear(self.d_model, self.d_model),
            output: linear(self.d_model, self.d_model),
            dropout: DropoutConfig::new(self.dropout).init(),
            d_model: self.d_model,
            n_heads: self.n_heads,
            d_k,
            quiet_softmax: self.quiet_softmax,
        }
    }
}

/// Parameters for extended streaming attention forward.
pub struct ExtStreamingParams<'a, B: Backend> {
    pub rope: Option<&'a burn::nn::rope_encoding::RotaryEncoding<B>>,
    pub start_pos: usize,
    pub window: AttnWindow,
    /// Optional additive bias on attention logits `[B, n_heads, q_len, k_len]` matching window.
    pub attn_bias: Option<&'a Tensor<B, 4>>,
}

impl<B: Backend> ExtStreamingMultiHeadAttention<B> {
    pub fn forward_streaming(
        &self,
        x: Tensor<B, 3>,
        cache: &mut burn::nn::attention::StreamingMhaCache<B>,
        params: ExtStreamingParams<B>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();

        let q = self.attention_linear(x.clone(), &self.query);
        let mut k = self.attention_linear(x.clone(), &self.key);
        let mut v = self.attention_linear(x, &self.value);

        // Apply RoPE if provided
        if let Some(rope) = params.rope {
            let q_rs = q.swap_dims(1, 2);
            let k_rs = k.swap_dims(1, 2);
            k = rope.apply(k_rs, params.start_pos).swap_dims(1, 2);
            // Here we keep applying to K and Q in the same way
            let q_ro = rope.apply(q_rs, params.start_pos).swap_dims(1, 2);
            // overwrite q
            let _ = q_ro; // q is immutable; recompute below for simplicity
        }

        // Recompute q with rope applied if any
        let q = if let Some(rope) = params.rope {
            let q_rs = self.attention_linear(x.clone(), &self.query).swap_dims(1, 2);
            rope.apply(q_rs, params.start_pos).swap_dims(1, 2)
        } else {
            self.attention_linear(x.clone(), &self.query)
        };

        // Update cache similarly to burn-core streaming attention.
        let num_new = seq_len;
        let current_end = params.start_pos + num_new;
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
                let rolled_k = cache.k.clone().slice([
                    0..batch_size,
                    src_start..src_end,
                    0..self.n_heads,
                    0..self.d_k,
                ]);
                cache.k.inplace(|t| {
                    t.slice_assign([
                        0..batch_size,
                        sink..sink + num_rolled,
                        0..self.n_heads,
                        0..self.d_k,
                    ], rolled_k)
                });
                let rolled_v = cache.v.clone().slice([
                    0..batch_size,
                    src_start..src_end,
                    0..self.n_heads,
                    0..self.d_k,
                ]);
                cache.v.inplace(|t| {
                    t.slice_assign([
                        0..batch_size,
                        sink..sink + num_rolled,
                        0..self.n_heads,
                        0..self.d_k,
                    ], rolled_v)
                });
            }
            cache.local_end_index = cache.local_end_index + delta - num_evicted;
        } else {
            cache.local_end_index += delta;
        }

        let local_end = cache.local_end_index;
        let local_start = local_end - num_new;
        let k_rs = k.swap_dims(1, 2);
        let v_rs = v.swap_dims(1, 2);
        cache.k.inplace(|t| {
            t.slice_assign([
                0..batch_size,
                local_start..local_end,
                0..self.n_heads,
                0..self.d_k,
            ], k_rs)
        });
        cache.v.inplace(|t| {
            t.slice_assign([
                0..batch_size,
                local_start..local_end,
                0..self.n_heads,
                0..self.d_k,
            ], v_rs)
        });
        cache.global_end_index = current_end;

        let active_len = match params.window {
            AttnWindow::Full => local_end,
            AttnWindow::Window(w) => sink + w.min(local_end.saturating_sub(sink)),
        };
        let start = local_end.saturating_sub(active_len);

        let k_win = cache
            .k
            .clone()
            .slice([
                0..batch_size,
                start..local_end,
                0..self.n_heads,
                0..self.d_k,
            ])
            .swap_dims(1, 2);
        let v_win = cache
            .v
            .clone()
            .slice([
                0..batch_size,
                start..local_end,
                0..self.n_heads,
                0..self.d_k,
            ])
            .swap_dims(1, 2);

        let mut attn_scores = q
            .matmul(k_win.transpose())
            .div_scalar((self.d_k as f32).sqrt());
        attn_scores = self.dropout.forward(attn_scores);

        if let Some(bias) = params.attn_bias {
            attn_scores = attn_scores + bias.clone();
        }
        let weights = if self.quiet_softmax {
            quiet_softmax(attn_scores, 3)
        } else {
            softmax(attn_scores, 3)
        };

        let context = weights.matmul(v_win)
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, self.d_model]);
        self.output.forward(context)
    }

    fn attention_linear(&self, x: Tensor<B, 3>, linear: &Linear<B>) -> Tensor<B, 4> {
        let [batch_size, seq_length, _d_model] = x.dims();
        linear
            .forward(x)
            .reshape([batch_size, seq_length, self.n_heads, self.d_k])
            .swap_dims(1, 2)
    }
}

