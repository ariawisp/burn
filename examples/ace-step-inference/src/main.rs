#![recursion_limit = "256"]

use burn::nn::attention::{AttnWindow, StreamingMultiHeadAttentionConfig, StreamingMhaCache, StreamingParams};
use burn::nn::RotaryEncodingConfig;
use burn::tensor::{backend::Backend, Distribution, Tensor};
use burn::backend::wgpu::{self, Wgpu as B, WgpuDevice};

fn main() {
    let device = WgpuDevice::default();
    wgpu::init_setup::<wgpu::graphics::Metal>(&device, Default::default());

    // ACE‑Step inspired: streaming MHA with sliding window and optional attn_bias.
    let b = 2usize; // batch
    let t = 48usize; // total tokens
    let d_model = 128usize;
    let n_heads = 4usize;
    let head_dim = d_model / n_heads;
    let chunk = 12usize;
    let cache_len = 128usize;

    let smha = StreamingMultiHeadAttentionConfig::new(d_model, n_heads)
        .with_dropout(0.0)
        .init::<B>(&device);

    let rope = RotaryEncodingConfig::new(4096, head_dim).init::<B>(&device);

    let mut cache = StreamingMhaCache::new(&device, b, cache_len, n_heads, head_dim, /*sink*/ 0);
    let x = Tensor::<B, 3>::random([b, t, d_model], Distribution::Default, &device);

    // Example additive bias: encourage locality (Gaussian around diagonal) within the active window.
    let mut outputs = Vec::new();
    for i in 0..(t / chunk) {
        let start = i * chunk;
        let q_len = chunk;
        let win = 32usize;
        // Active key length is min(sink + win, start+chunk)
        let k_len = win.min(start + chunk);
        let mut bias = Tensor::<B, 4>::zeros([b, n_heads, q_len, k_len], &device);
        // here we could fill bias with a negative distance penalty; keep zeros for simplicity

        let y = smha.forward_streaming(
            x.clone().slice([0..b, start..start + chunk, 0..d_model]),
            &mut cache,
            StreamingParams {
                rope: Some(&rope),
                start_pos: start,
                window: AttnWindow::Window(win),
                attn_bias: Some(&bias),
            },
        );
        outputs.push(y);
    }
    let y = Tensor::cat(outputs, 1);
    println!("ACE-Step streaming MHA output shape: {:?}", y.dims());
}
