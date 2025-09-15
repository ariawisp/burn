use burn_core as burn;

use burn::tensor::{Int, Tensor, backend::Backend};

/// Basic logits processing configuration for sampling.
#[derive(Clone, Copy, Debug, Default)]
pub struct SamplerConfig {
    pub temperature: f32,             // <= 0.0 means greedy
    pub top_k: Option<usize>,         // keep top-k logits, mask others
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}

/// Apply temperature scaling to logits. No-op if `temperature <= 0` or `temperature == 1`.
pub fn apply_temperature<B: Backend>(logits: Tensor<B, 2>, temperature: f32) -> Tensor<B, 2> {
    if temperature > 0.0 && (temperature - 1.0).abs() > core::f32::EPSILON {
        logits / Tensor::from_floats([temperature], &logits.device()).unsqueeze()
    } else {
        logits
    }
}

/// Mask logits outside the top-k per row by setting them to -inf.
/// Note: This is an O(V*logV) CPU fallback using host sorting for portability.
pub fn apply_top_k_cpu<B: Backend>(logits: Tensor<B, 2>, k: usize) -> Tensor<B, 2> {
    if k == 0 { return logits; }
    let device = logits.device();
    let [b, v] = logits.dims();
    let data = logits.into_data().convert::<f32>();
    let mut out: Vec<f32> = vec![f32::NEG_INFINITY; b * v];
    for bi in 0..b {
        let row = &data.value[bi * v..(bi + 1) * v];
        // Indices sorted by logit descending
        let mut idx: Vec<usize> = (0..v).collect();
        idx.sort_unstable_by(|&i, &j| row[j].partial_cmp(&row[i]).unwrap_or(core::cmp::Ordering::Equal));
        for &col in idx.iter().take(k.min(v)) {
            out[bi * v + col] = row[col];
        }
    }
    Tensor::<B, 2>::from_floats(out, &device).reshape([b, v])
}

/// Apply repetition, frequency and presence penalties using token history.
/// This operates on CPU for simplicity and portability.
pub fn apply_penalties_cpu<B: Backend>(
    logits: Tensor<B, 2>,
    tokens_history: Option<&[Vec<usize>]>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
) -> Tensor<B, 2> {
    if tokens_history.is_none() {
        return logits;
    }
    let [b, v] = logits.dims();
    let device = logits.device();
    let mut out = logits.into_data().convert::<f32>().value;
    let rep = repetition_penalty.unwrap_or(1.0);
    let freq = frequency_penalty.unwrap_or(0.0);
    let pres = presence_penalty.unwrap_or(0.0);

    for bi in 0..b.min(tokens_history.unwrap().len()) {
        let history = &tokens_history.unwrap()[bi];
        if history.is_empty() { continue; }
        // Count occurrences
        let mut counts = vec![0usize; v];
        for &tok in history.iter() {
            if tok < v { counts[tok] += 1; }
        }
        for tok in 0..v {
            let c = counts[tok] as f32;
            if c > 0.0 {
                // Repetition penalty: decrease positive logits more, increase negative less
                if rep != 1.0 {
                    let idx = bi * v + tok;
                    let l = out[idx];
                    out[idx] = if l > 0.0 { l / rep } else { l * rep };
                }
                // Frequency/presence penalties: subtract proportional biases
                let idx = bi * v + tok;
                out[idx] -= freq * c + pres;
            }
        }
    }
    Tensor::<B, 2>::from_floats(out, &device).reshape([b, v])
}

/// Greedy sampling (argmax).
pub fn sample_greedy<B: Backend>(logits: Tensor<B, 2>) -> Tensor<B, 1, Int> {
    // argmax along vocab dim
    let [b, _v] = logits.dims();
    let indices = logits.argmax(1).reshape([b]);
    indices.int()
}

/// Simple CPU multinomial sampling after softmax. Suitable for demos.
pub fn sample_multinomial_cpu<B: Backend>(logits: Tensor<B, 2>) -> Tensor<B, 1, Int> {
    use rand::distributions::WeightedIndex;
    use rand::prelude::*;
    let [b, v] = logits.dims();
    let device = logits.device();
    let mut out = vec![0i64; b];
    let probs = logits.softmax(1).into_data().convert::<f32>().value;
    let mut rng = thread_rng();
    for bi in 0..b {
        let row = &probs[bi * v..(bi + 1) * v];
        let dist = WeightedIndex::new(row.iter().map(|&p| if p.is_finite() && p > 0.0 { p } else { 0.0 })).unwrap();
        let sample = dist.sample(&mut rng) as i64;
        out[bi] = sample;
    }
    Tensor::<B, 1, Int>::from_ints(out, &device)
}

/// Apply typical processors to logits and sample next tokens.
pub fn process_and_sample<B: Backend>(
    logits: Tensor<B, 2>,
    tokens_history: Option<&[Vec<usize>]>,
    cfg: SamplerConfig,
    greedy_when_temp_zero: bool,
) -> Tensor<B, 1, Int> {
    let mut l = logits;
    if let Some(k) = cfg.top_k { l = apply_top_k_cpu(l, k); }
    l = apply_penalties_cpu(l, tokens_history, cfg.repetition_penalty, cfg.frequency_penalty, cfg.presence_penalty);
    l = apply_temperature(l, cfg.temperature);
    if cfg.temperature <= 0.0 && greedy_when_temp_zero {
        sample_greedy(l)
    } else {
        sample_multinomial_cpu(l)
    }
}

