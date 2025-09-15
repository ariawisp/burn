use burn::nn::attention::{
    AttnWindow, StreamingMqaCache, StreamingMultiQueryAttentionConfig, StreamingMqaParams,
};
use burn::nn::RotaryEncodingConfig;
use burn::tensor::{backend::Backend, Distribution, Tensor};
use burn::backend::ndarray::NdArray as B;

fn main() {
    let device = <B as Backend>::Device::default();

    // Simple GPT‑OSS‑style block demo: Streaming MQA + RoPE NTK/YaRN + sinks.
    let b = 1usize; // batch
    let t = 64usize; // total tokens
    let d_model = 256usize;
    let n_heads = 8usize;
    let kv_heads = 2usize; // MQA/GQA
    let head_dim = d_model / n_heads;
    let chunk = 16usize; // stream in chunks
    let cache_len = 256usize;
    let sink_tokens = 0usize;

    // Attention module
    let attn = StreamingMultiQueryAttentionConfig::new(d_model, n_heads, kv_heads)
        .with_dropout(0.0)
        .init::<B>(&device);

    // RoPE with NTK/YaRN convenience
    let rope = RotaryEncodingConfig::new(8192, head_dim)
        .init_ntk_yarn::<B>(&device, /*scaling_factor*/ 32.0, /*init_ctx*/ 4096.0, /*alpha*/ 1.0, /*beta*/ 32.0);

    // Simulated learned sinks logits per (kv_head, group)
    let groups = n_heads / kv_heads;
    let sinks = Tensor::<B, 2>::random([kv_heads, groups], Distribution::Default, &device);

    // Streaming cache
    let mut cache = StreamingMqaCache::new(&device, b, cache_len, kv_heads, head_dim, sink_tokens);

    // Dummy hidden states to run through attention
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
    let dims = y.dims();
    println!("GPT-OSS streaming attention output shape: {:?}", dims);
}
