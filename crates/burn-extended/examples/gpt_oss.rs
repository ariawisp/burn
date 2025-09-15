#![recursion_limit = "256"]

use burn::backend::wgpu::{self, Wgpu as B, WgpuDevice};
use burn::nn::RotaryEncodingConfig;
use burn::tensor::{Distribution, Tensor};

use burn_extended::attention::{
    AttnWindow, StreamingMqaCache, StreamingMqaParams, StreamingMultiQueryAttentionConfig,
};
use burn_extended::rope::init_ntk_yarn;

fn main() {
    let device = WgpuDevice::default();
    wgpu::init_setup::<wgpu::graphics::Metal>(&device, Default::default());

    let b = 1usize;
    let t = 64usize;
    let d_model = 256usize;
    let n_heads = 8usize;
    let kv_heads = 2usize;
    let head_dim = d_model / n_heads;
    let chunk = 16usize;
    let cache_len = 256usize;

    let attn = StreamingMultiQueryAttentionConfig::new(d_model, n_heads, kv_heads)
        .with_dropout(0.0)
        .init::<B>(&device);

    let rope = init_ntk_yarn::<B>(8192, head_dim, &device, 32.0, 4096.0, 1.0, 32.0);
    let groups = n_heads / kv_heads;
    let sinks = Tensor::<B, 2>::random([kv_heads, groups], Distribution::Default, &device);

    let mut cache = StreamingMqaCache::new(&device, b, cache_len, kv_heads, head_dim, 0);
    let x = Tensor::<B, 3>::random([b, t, d_model], Distribution::Default, &device);

    let mut outputs = Vec::new();
    for i in 0..(t / chunk) {
        let start = i * chunk;
        let x_i = x.clone().slice([0..b, start..start + chunk, 0..d_model]);
        let params = StreamingMqaParams {
            rope: Some(&rope),
            start_pos: start,
            window: AttnWindow::Window(128),
            sinks: Some(&sinks),
            attn_bias: None,
        };
        let y = attn.forward_streaming(x_i, &mut cache, params);
        outputs.push(y);
    }
    let y = Tensor::cat(outputs, 1);
    println!("gpt-oss example output shape: {:?}", y.dims());
}

