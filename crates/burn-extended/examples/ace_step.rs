#![recursion_limit = "256"]

use burn::backend::wgpu::{self, Wgpu as B, WgpuDevice};
use burn::nn::RotaryEncodingConfig;
use burn::tensor::{Distribution, Tensor};
use burn_extended::attention::{
    AttnWindow, ExtStreamingMultiHeadAttentionConfig, ExtStreamingParams,
};

fn main() {
    let device = WgpuDevice::default();
    wgpu::init_setup::<wgpu::graphics::Metal>(&device, Default::default());

    let b = 2usize;
    let t = 48usize;
    let d_model = 128usize;
    let n_heads = 4usize;
    let head_dim = d_model / n_heads;
    let chunk = 12usize;
    let cache_len = 128usize;

    let smha = ExtStreamingMultiHeadAttentionConfig::new(d_model, n_heads)
        .with_dropout(0.0)
        .init::<B>(&device);

    let rope = RotaryEncodingConfig::new(4096, head_dim).init::<B>(&device);

    let mut cache = burn::nn::attention::StreamingMhaCache::new(
        &device, b, cache_len, n_heads, head_dim, /*sink*/ 0,
    );
    let x = Tensor::<B, 3>::random([b, t, d_model], Distribution::Default, &device);

    let mut outputs = Vec::new();
    for i in 0..(t / chunk) {
        let start = i * chunk;
        let q_len = chunk;
        let win = 32usize;
        let k_len = win.min(start + chunk);
        let bias = Tensor::<B, 4>::zeros([b, n_heads, q_len, k_len], &device);

        let y = smha.forward_streaming(
            x.clone().slice([0..b, start..start + chunk, 0..d_model]),
            &mut cache,
            ExtStreamingParams { rope: Some(&rope), start_pos: start, window: AttnWindow::Window(win), attn_bias: Some(&bias) },
        );
        outputs.push(y);
    }
    let y = Tensor::cat(outputs, 1);
    println!("ace-step example output shape: {:?}", y.dims());
}

