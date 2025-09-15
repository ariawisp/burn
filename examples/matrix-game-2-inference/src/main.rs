use burn::nn::attention::{AttnWindow, StreamingMhaCache, StreamingMultiHeadAttentionConfig, StreamingParams};
use burn::tensor::{backend::Backend, Distribution, Tensor};
use burn::backend::ndarray::NdArray as B;

fn main() {
    let device = <B as Backend>::Device::default();

    // Matrix‑Game‑2 inspired: demonstrate sink tokens for persistent memory anchors.
    let b = 1usize;
    let t = 32usize;
    let d_model = 96usize;
    let n_heads = 3usize;
    let head_dim = d_model / n_heads;
    let chunk = 8usize;
    let cache_len = 64usize;
    let sink_tokens = 4usize; // keep first 4 tokens always attendable

    let smha = StreamingMultiHeadAttentionConfig::new(d_model, n_heads)
        .with_dropout(0.0)
        .init::<B>(&device);

    let mut cache = StreamingMhaCache::new(&device, b, cache_len, n_heads, head_dim, sink_tokens);
    let x = Tensor::<B, 3>::random([b, t, d_model], Distribution::Default, &device);

    let mut outputs = Vec::new();
    for i in 0..(t / chunk) {
        let start = i * chunk;
        let y = smha.forward_streaming(
            x.clone().slice([0..b, start..start + chunk, 0..d_model]),
            &mut cache,
            StreamingParams {
                rope: None,
                start_pos: start,
                window: AttnWindow::Window(16), // sliding window w/ sink preservation
                attn_bias: None,
            },
        );
        outputs.push(y);
    }
    let y = Tensor::cat(outputs, 1);
    println!("Matrix-Game-2 streaming w/ sinks output shape: {:?}", y.dims());
}
