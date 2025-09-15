use burn_core as burn;

use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig};
use burn::{
    config::Config,
    tensor::{Bool, Tensor, backend::Backend},
};

use burn_tensor::activation::{quiet_softmax, softmax};

/// Configuration to create a Multi-Query/Grouped-Query Attention layer.
#[derive(Config, Debug)]
pub struct MultiQueryAttentionConfig {
    /// The size of each linear layer (model dimension).
    pub d_model: usize,
    /// The number of query heads.
    pub n_heads: usize,
    /// The number of key/value heads (must divide `n_heads`).
    pub num_key_value_heads: usize,
    /// Dropout probability (on attention logits).
    #[config(default = 0.1)]
    pub dropout: f64,
    /// Minimum float used to mask attention scores before softmax.
    #[config(default = -1.0e4)]
    pub min_float: f64,
    /// Use quiet softmax instead of regular softmax.
    #[config(default = false)]
    pub quiet_softmax: bool,
    /// Parameter initializer for the linear layers.
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Multi-Query/Grouped-Query Attention (training/non-streaming inference, with masks).
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct MultiQueryAttention<B: Backend> {
    pub query: Linear<B>,
    pub key: Linear<B>,
    pub value: Linear<B>,
    pub output: Linear<B>,
    pub dropout: Dropout,
    pub d_model: usize,
    pub n_heads: usize,
    pub kv_heads: usize,
    pub d_k: usize,
    pub min_float: f64,
    pub quiet_softmax: bool,
}

impl<B: Backend> ModuleDisplay for MultiQueryAttention<B> {
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
            .add("min_float", &self.min_float)
            .add("quiet_softmax", &self.quiet_softmax)
            .optional()
    }
}

impl MultiQueryAttentionConfig {
    /// Initialize a new MQA/GQA module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiQueryAttention<B> {
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

        MultiQueryAttention {
            query: linear(self.d_model, self.d_model),
            key: linear(self.d_model, self.num_key_value_heads * d_k),
            value: linear(self.d_model, self.num_key_value_heads * d_k),
            output: linear(self.d_model, self.d_model),
            dropout: DropoutConfig::new(self.dropout).init(),
            d_model: self.d_model,
            n_heads: self.n_heads,
            kv_heads: self.num_key_value_heads,
            d_k,
            min_float: self.min_float,
            quiet_softmax: self.quiet_softmax,
        }
    }
}

/// Input for MQA forward.
#[derive(Debug, Clone)]
pub struct MqaInput<B: Backend> {
    pub query: Tensor<B, 3>,
    pub key: Tensor<B, 3>,
    pub value: Tensor<B, 3>,
    pub mask_pad: Option<Tensor<B, 2, Bool>>,
    pub mask_attn: Option<Tensor<B, 3, Bool>>,
    /// Optional additive bias on attention logits, shape `[B, n_heads, q_len, k_len]`.
    pub attn_bias: Option<Tensor<B, 4>>, 
}

impl<B: Backend> MqaInput<B> {
    pub fn self_attn(tensor: Tensor<B, 3>) -> Self {
        Self {
            query: tensor.clone(),
            key: tensor.clone(),
            value: tensor,
            mask_pad: None,
            mask_attn: None,
            attn_bias: None,
        }
    }
    pub fn new(query: Tensor<B, 3>, key: Tensor<B, 3>, value: Tensor<B, 3>) -> Self {
        Self { query, key, value, mask_pad: None, mask_attn: None, attn_bias: None }
    }
    pub fn mask_pad(mut self, mask_pad: Tensor<B, 2, Bool>) -> Self { self.mask_pad = Some(mask_pad); self }
    pub fn mask_attn(mut self, mask_attn: Tensor<B, 3, Bool>) -> Self { self.mask_attn = Some(mask_attn); self }
    pub fn attn_bias(mut self, attn_bias: Tensor<B, 4>) -> Self { self.attn_bias = Some(attn_bias); self }
}

#[derive(Debug, Clone)]
pub struct MqaOutput<B: Backend> {
    pub weights: Tensor<B, 4>,
    pub context: Tensor<B, 3>,
}

impl<B: Backend> MultiQueryAttention<B> {
    /// Forward pass with masks and optional additive attention bias.
    pub fn forward(&self, input: MqaInput<B>) -> MqaOutput<B> {
        let [batch_size, seq_q, d_model] = input.query.dims();
        let seq_k = input.key.dims()[1];
        let groups = self.n_heads / self.kv_heads;

        // Projections
        let q = self.attention_linear_q(input.query, &self.query);
        let k = self.attention_linear_kv(input.key, &self.key);
        let v = self.attention_linear_kv(input.value, &self.value);

        // Expand KV across groups to match Q heads.
        let k_exp = k
            .unsqueeze_dim::<5>(2) // [B, kvH, 1, Tk, d_k]
            .repeat_dim(2, groups)
            .reshape([batch_size, self.n_heads, seq_k, self.d_k]);
        let v_exp = v
            .unsqueeze_dim::<5>(2)
            .repeat_dim(2, groups)
            .reshape([batch_size, self.n_heads, seq_k, self.d_k]);

        // Compute attention scores
        let mut attn_scores = q
            .matmul(k_exp.transpose())
            .div_scalar((self.d_k as f32).sqrt());

        // Apply masks
        if let Some(mask_pad) = input.mask_pad {
            let [b, sl] = mask_pad.dims();
            attn_scores = attn_scores.mask_fill(mask_pad.reshape([b, 1, 1, sl]), self.min_float);
        }
        if let Some(mask_attn) = input.mask_attn {
            let [b, s1, s2] = mask_attn.dims();
            attn_scores = attn_scores.mask_fill(mask_attn.reshape([b, 1, s1, s2]), self.min_float);
        }

        // Additive bias
        if let Some(attn_bias) = input.attn_bias {
            attn_scores = attn_scores + attn_bias;
        }

        // Softmax
        let weights = if self.quiet_softmax {
            quiet_softmax(attn_scores, 3)
        } else {
            softmax(attn_scores, 3)
        };

        let context = weights.clone().matmul(v_exp)
            .swap_dims(1, 2)
            .reshape([batch_size, seq_q, d_model]);
        let context = self.output.forward(context);
        MqaOutput { weights, context }
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

