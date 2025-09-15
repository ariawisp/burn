use burn_core as burn;

use alloc::vec::Vec;
use burn::tensor::{Int, Tensor, backend::Backend};

use crate::attention::AttnWindow;
use crate::sampling::{process_and_sample, SamplerConfig};

/// Trait for an autoregressive model that can produce logits from token IDs with a streaming cache.
pub trait AutoregressiveModel<B: Backend> {
    type Cache;
    fn init_cache(&self, batch: usize, device: &B::Device) -> Self::Cache;
    /// Forward pass that returns next-token logits for the last position of the provided sequence.
    fn forward_logits(
        &self,
        tokens: Tensor<B, 2, Int>, // [B, T]
        cache: &mut Self::Cache,
        start_pos: usize,
        window: AttnWindow,
    ) -> Tensor<B, 2>; // [B, vocab]
}

/// Configuration for text generation.
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub eos_token: Option<usize>,
    pub sampler: SamplerConfig,
    pub window: AttnWindow,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            eos_token: None,
            sampler: SamplerConfig { temperature: 1.0, top_k: Some(50), repetition_penalty: None, frequency_penalty: None, presence_penalty: None },
            window: AttnWindow::Full,
        }
    }
}

/// Generate tokens for a batch of prompts using an autoregressive model.
pub fn generate<B: Backend, M: AutoregressiveModel<B>>(
    model: &M,
    device: &B::Device,
    prompts: &[Vec<usize>],
    cfg: GenerationConfig,
) -> Vec<Vec<usize>> {
    let b = prompts.len();
    let mut tokens: Vec<Vec<usize>> = prompts.to_vec();
    let mut cache = model.init_cache(b, device);
    let mut start_pos = tokens[0].len();

    for _step in 0..cfg.max_new_tokens {
        // Build tokens tensor [B, T]
        let max_t = tokens.iter().map(|t| t.len()).max().unwrap_or(0);
        let mut flat: Vec<i64> = Vec::with_capacity(b * max_t);
        for row in 0..b {
            let row_t = &tokens[row];
            for &tok in row_t.iter() { flat.push(tok as i64); }
            for _ in row_t.len()..max_t { flat.push(0); } // pad with 0; model should ignore via pad mask
        }
        let input = Tensor::<B, 2, Int>::from_ints(flat, device).reshape([b, max_t]);

        let logits = model.forward_logits(input, &mut cache, start_pos.saturating_sub(1), cfg.window);

        // CPU-side history for penalties
        let history: Vec<Vec<usize>> = tokens.clone();
        let next = process_and_sample::<B>(logits, Some(&history), cfg.sampler, /*greedy_when_temp_zero*/ true);
        let next_data = next.into_data().convert::<i64>().value;

        // Append and check EOS
        let mut all_eos = true;
        for i in 0..b {
            let tok = next_data[i] as usize;
            tokens[i].push(tok);
            if let Some(eos) = cfg.eos_token {
                if tok != eos { all_eos = false; }
            } else {
                all_eos = false;
            }
        }
        start_pos += 1;
        if all_eos { break; }
    }
    tokens
}

