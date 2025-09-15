use burn_core as burn;

use burn::nn::rope_encoding::RotaryEncoding;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// Initialize RoPE with NTK/YaRN-style scaling and concentration without modifying burn-core.
///
/// - Scales inverse frequencies by parts between two thresholds (alpha/beta) to stretch context.
/// - Applies a multiplicative concentration factor to the cosine/sine table.
pub fn init_ntk_yarn<B: Backend>(
    max_sequence_length: usize,
    head_dim: usize,
    device: &B::Device,
    scaling_factor: f32,
    initial_context_length: f32,
    ntk_alpha: f32,
    ntk_beta: f32,
) -> RotaryEncoding<B> {
    let config = burn::nn::rope_encoding::RotaryEncodingConfig::new(max_sequence_length, head_dim);
    if scaling_factor <= 1.0 {
        return config.init::<B>(device);
    }

    let base = config.theta; // default 10000.0
    let d_model = head_dim as f32 * 2.0;
    let d_half = d_model / 2.0;
    let log_base = base.ln();
    let two_pi = core::f32::consts::PI * 2.0;
    let low = d_half * (initial_context_length / (ntk_beta * two_pi)).ln() / log_base;
    let high = d_half * (initial_context_length / (ntk_alpha * two_pi)).ln() / log_base;
    assert!(low > 0.0 && high > 0.0 && low < high && high < d_half - 1.0);

    let scaling = move |inv_freq: Tensor<B, 1>| {
        // inv_freq = base^(-2i/d_model)  =>  i = -(d_model/2) * ln(inv_freq)/ln(base)
        let i = inv_freq
            .clone()
            .log()
            .mul_scalar(-d_model / 2.0 / log_base);

        let ramp = i
            .clone()
            .sub_scalar(low)
            .div_scalar(high - low)
            .clamp(0.0, 1.0);
        let one_minus = ramp.clone().neg().add_scalar(1.0);
        let interp = inv_freq.clone().div_scalar(scaling_factor);
        interp.mul(one_minus).add(inv_freq.mul(ramp))
    };

    let mut rope = config.init_with_frequency_scaling::<B>(scaling, device);
    let concentration = 0.1 * scaling_factor.ln() + 1.0;
    rope.freq_complex = rope.freq_complex.mul_scalar(concentration);
    rope
}

