use burn_core as burn;

use burn::tensor::backend::Backend;
use burn::nn::attention::StreamingMhaCache;

use crate::attention::StreamingMqaCache;
use crate::attention::AttnWindow;

/// Policy to determine per-layer attention windows.
#[derive(Clone, Copy, Debug)]
pub enum WindowPolicy {
    Full,
    Fixed(usize),
    EveryOther { window: usize, full_on_even: bool },
}

impl WindowPolicy {
    pub fn window_for(&self, layer_idx: usize) -> AttnWindow {
        match *self {
            WindowPolicy::Full => AttnWindow::Full,
            WindowPolicy::Fixed(w) => AttnWindow::Window(w),
            WindowPolicy::EveryOther { window, full_on_even } => {
                let even = layer_idx % 2 == 0;
                if (even && full_on_even) || (!even && !full_on_even) {
                    AttnWindow::Full
                } else {
                    AttnWindow::Window(window)
                }
            }
        }
    }
}

/// Manages a set of streaming MQA caches (one per layer).
pub struct MqaCacheManager<B: Backend> {
    pub caches: alloc::vec::Vec<StreamingMqaCache<B>>,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub cache_len: usize,
    pub sink_tokens: usize,
}

impl<B: Backend> MqaCacheManager<B> {
    pub fn new(device: &B::Device, num_layers: usize, kv_heads: usize, head_dim: usize, cache_len: usize, sink_tokens: usize, batch: usize) -> Self {
        let mut caches = alloc::vec::Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            caches.push(StreamingMqaCache::new(device, batch, cache_len, kv_heads, head_dim, sink_tokens));
        }
        Self { caches, kv_heads, head_dim, cache_len, sink_tokens }
    }
    pub fn cache_mut(&mut self, layer: usize) -> &mut StreamingMqaCache<B> { &mut self.caches[layer] }
}

/// Manages a set of streaming MHA caches (one per layer).
pub struct MhaCacheManager<B: Backend> {
    pub caches: alloc::vec::Vec<StreamingMhaCache<B>>,
    pub n_heads: usize,
    pub head_dim: usize,
    pub cache_len: usize,
    pub sink_tokens: usize,
}

impl<B: Backend> MhaCacheManager<B> {
    pub fn new(device: &B::Device, num_layers: usize, n_heads: usize, head_dim: usize, cache_len: usize, sink_tokens: usize, batch: usize) -> Self {
        let mut caches = alloc::vec::Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            caches.push(StreamingMhaCache::new(device, batch, cache_len, n_heads, head_dim, sink_tokens));
        }
        Self { caches, n_heads, head_dim, cache_len, sink_tokens }
    }
    pub fn cache_mut(&mut self, layer: usize) -> &mut StreamingMhaCache<B> { &mut self.caches[layer] }
}

