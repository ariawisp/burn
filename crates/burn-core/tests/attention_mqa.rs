use burn_core::nn::attention::{MultiQueryAttentionConfig, MqaInput};
use burn_core::tensor::{Distribution, Shape, Tensor};
type TB = burn_ndarray::NdArray<f32>;

#[test]
fn mqa_shapes_and_masks() {
    let device = Default::default();
    let [b, tq, tk, d_model, n_heads, kv_heads] = [2, 5, 7, 32, 8, 2];

    let mqa = MultiQueryAttentionConfig::new(d_model, n_heads, kv_heads)
        .with_dropout(0.0)
        .init::<TB>(&device);

    let q = Tensor::<TB, 3>::random([b, tq, d_model], Distribution::Default, &device);
    let k = Tensor::<TB, 3>::random([b, tk, d_model], Distribution::Default, &device);
    let v = Tensor::<TB, 3>::random([b, tk, d_model], Distribution::Default, &device);

    // Padding mask: mask the last key positions
    let pad = 3usize;
    let mut mask_pad = Tensor::<TB, 2, burn_core::tensor::Bool>::full([b, tk], false, &device);
    let trues = Tensor::<TB, 2, burn_core::tensor::Bool>::full([b, pad], true, &device);
    mask_pad = mask_pad.slice_assign([0..b, tk - pad..tk], trues);

    let input = MqaInput::new(q, k, v).mask_pad(mask_pad);
    let out = mqa.forward(input);
    assert_eq!(out.context.shape(), Shape::new([b, tq, d_model]));
    assert_eq!(out.weights.shape(), Shape::new([b, n_heads, tq, tk]));
}
