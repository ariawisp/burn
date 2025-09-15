use burn_core::nn::RotaryEncodingConfig;
use burn_core::nn::attention::{
    AttnWindow, StreamingMqaCache, StreamingMultiQueryAttentionConfig, StreamingMqaParams,
};
use burn_core::tensor::{Distribution, Shape, Tensor};
type TB = burn_ndarray::NdArray<f32>;
use burn_tensor::Tolerance;
use burn_tensor::ops::FloatElem;

#[test]
fn streaming_mqa_no_window_vs_full_window_equal() {
    let device = Default::default();
    let b = 2;
    let t = 12;
    let d_model = 32;
    let n_heads = 8;
    let kv_heads = 2;

    let x = Tensor::<TB, 3>::random([b, t, d_model], Distribution::Default, &device);

    let smqa = StreamingMultiQueryAttentionConfig::new(d_model, n_heads, kv_heads)
        .with_dropout(0.0)
        .init::<TB>(&device);
    let mut cache1 = StreamingMqaCache::new(
        &device,
        b,
        /*cache_len*/ 64,
        kv_heads,
        d_model / n_heads,
        /*sink*/ 0,
    );
    let out1 = smqa.forward_streaming(
        x.clone(),
        &mut cache1,
        StreamingMqaParams {
            rope: None,
            start_pos: 0,
            window: AttnWindow::Full,
            sinks: None,
            attn_bias: None,
        },
    );

    let mut cache2 = StreamingMqaCache::new(
        &device,
        b,
        /*cache_len*/ 64,
        kv_heads,
        d_model / n_heads,
        /*sink*/ 0,
    );
    let out2 = smqa.forward_streaming(
        x,
        &mut cache2,
        StreamingMqaParams {
            rope: None,
            start_pos: 0,
            window: AttnWindow::Window(t),
            sinks: None,
            attn_bias: None,
        },
    );

    assert_eq!(out1.shape(), Shape::new([b, t, d_model]));
    out1.into_data()
        .assert_approx_eq::<FloatElem<TB>>(&out2.into_data(), Tolerance::default());
}

#[test]
fn streaming_mqa_chunked_with_rope_matches_full_call() {
    let device = Default::default();
    let b = 1;
    let t = 16;
    let d_model = 32;
    let n_heads = 8;
    let kv_heads = 2;

    let x = Tensor::<TB, 3>::random([b, t, d_model], Distribution::Default, &device);
    let head_dim = d_model / n_heads;
    let rope = RotaryEncodingConfig::new(512, head_dim).init::<TB>(&device);
    let smqa = StreamingMultiQueryAttentionConfig::new(d_model, n_heads, kv_heads)
        .with_dropout(0.0)
        .init::<TB>(&device);

    // Full single-shot (one chunk)
    let mut cache_full = StreamingMqaCache::new(
        &device, b, /*cache_len*/ 64, kv_heads, head_dim, /*sink*/ 0,
    );
    let out_full = smqa.forward_streaming(
        x.clone(),
        &mut cache_full,
        StreamingMqaParams {
            rope: Some(&rope),
            start_pos: 0,
            window: AttnWindow::Window(t),
            sinks: None,
            attn_bias: None,
        },
    );

    // Chunked
    let mut cache_chunked = StreamingMqaCache::new(
        &device, b, /*cache_len*/ 64, kv_heads, head_dim, /*sink*/ 0,
    );
    let mut outputs = Vec::new();
    let chunk = 4;
    for i in 0..(t / chunk) {
        let start = i * chunk;
        let x_i = x.clone().slice([0..b, start..start + chunk, 0..d_model]);
        let params = StreamingMqaParams {
            rope: Some(&rope),
            start_pos: start,
            window: AttnWindow::Window(t),
            sinks: None,
            attn_bias: None,
        };
        let y = smqa.forward_streaming(x_i, &mut cache_chunked, params);
        outputs.push(y);
    }
    let out_chunked = Tensor::cat(outputs, 1);

    assert_eq!(out_full.shape(), Shape::new([b, t, d_model]));
    out_full
        .into_data()
        .assert_approx_eq::<FloatElem<TB>>(&out_chunked.into_data(), Tolerance::rel_abs(0.5, 0.2));
}

#[test]
fn streaming_mqa_with_sinks_neg_infty_equivalence() {
    // When sinks logits are large negative, outputs should match no-sinks path.
    let device = Default::default();
    let b = 1;
    let t = 8;
    let d_model = 32;
    let n_heads = 8;
    let kv_heads = 2;
    let groups = n_heads / kv_heads;

    let x = Tensor::<TB, 3>::random([b, t, d_model], Distribution::Default, &device);

    let smqa = StreamingMultiQueryAttentionConfig::new(d_model, n_heads, kv_heads)
        .with_dropout(0.0)
        .init::<TB>(&device);

    let mut cache1 = StreamingMqaCache::new(&device, b, 64, kv_heads, d_model / n_heads, 0);
    let out1 = smqa.forward_streaming(
        x.clone(),
        &mut cache1,
        StreamingMqaParams { rope: None, start_pos: 0, window: AttnWindow::Full, sinks: None, attn_bias: None },
    );

    let mut cache2 = StreamingMqaCache::new(&device, b, 64, kv_heads, d_model / n_heads, 0);
    let sinks = Tensor::<TB, 2>::full([kv_heads, groups], -1.0e9, &device);
    let out2 = smqa.forward_streaming(
        x,
        &mut cache2,
        StreamingMqaParams { rope: None, start_pos: 0, window: AttnWindow::Full, sinks: Some(&sinks), attn_bias: None },
    );

    out1
        .into_data()
        .assert_approx_eq::<FloatElem<TB>>(&out2.into_data(), Tolerance::default());
}
