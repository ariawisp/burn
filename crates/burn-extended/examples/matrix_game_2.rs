#![recursion_limit = "256"]

use burn::backend::wgpu::{self, Wgpu as B, WgpuDevice};
use burn::tensor::{Distribution, Tensor};
use burn_extended::attention::{AttnWindow, ExtStreamingMultiHeadAttentionConfig, ExtStreamingParams};

fn main() {
    let device = WgpuDevice::default();
    wgpu::init_setup::<wgpu::graphics::Metal>(&device, Default::default());

    let b = 1usize;
    let t = 32usize;
    let d_model = 96usize;
    let n_heads = 3usize;
    let head_dim = d_model / n_heads;
    let chunk = 8usize;
    let cache_len = 64usize;
    let sink_tokens = 4usize;

    let smha = ExtStreamingMultiHeadAttentionConfig::new(d_model, n_heads)
        .with_dropout(0.0)
        .init::<B>(&device);

    let mut cache = burn::nn::attention::StreamingMhaCache::new(&device, b, cache_len, n_heads, head_dim, sink_tokens);
    let x = Tensor::<B, 3>::random([b, t, d_model], Distribution::Default, &device);

    let mut outputs = Vec::new();
    for i in 0..(t / chunk) {
        let start = i * chunk;
        let y = smha.forward_streaming(
            x.clone().slice([0..b, start..start + chunk, 0..d_model]),
            &mut cache,
            ExtStreamingParams { rope: None, start_pos: start, window: AttnWindow::Window(16), attn_bias: None },
        );
        outputs.push(y);
    }
    let y = Tensor::cat(outputs, 1);
    println!("matrix-game-2 example output shape: {:?}", y.dims());
}

