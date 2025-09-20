#![recursion_limit = "256"]

use burn_core::module::Module;
use burn_core as burn;
use burn_core::nn;
use burn_core::prelude::*;
use burn_store::ModuleSnapshot;
use divan::{AllocProfiler, Bencher};
use std::fs;
use std::path::PathBuf;
use tempfile::tempdir;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

#[cfg(feature = "batched")]
use burn_tensor::BatchTensorOps;

type TestBackend = burn_wgpu::Metal;

// Simple model for basic benchmarks
#[derive(Module, Debug)]
struct SimpleModel<B: Backend> {
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
}

impl<B: Backend> SimpleModel<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            linear1: nn::LinearConfig::new(256, 512).init(device),
            linear2: nn::LinearConfig::new(512, 1024).init(device),
        }
    }
}

// Medium model with various layer types
#[derive(Module, Debug)]
struct MediumModel<B: Backend> {
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
    linear3: nn::Linear<B>,
}

impl<B: Backend> MediumModel<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            linear1: nn::LinearConfig::new(512, 1024).init(device),
            linear2: nn::LinearConfig::new(1024, 2048).init(device),
            linear3: nn::LinearConfig::new(2048, 4096).init(device),
        }
    }
}

// Large model to test scalability
#[derive(Module, Debug)]
struct LargeModel<B: Backend> {
    layers: Vec<nn::Linear<B>>,
}

impl<B: Backend> LargeModel<B> {
    fn new(device: &B::Device) -> Self {
        let mut layers = Vec::new();
        for i in 0..20 {
            let in_size = if i == 0 { 1024 } else { 2048 };
            layers.push(nn::LinearConfig::new(in_size, 2048).init(device));
        }
        Self { layers }
    }
}

fn create_test_file<B: Backend, M: Module<B>>(path: &PathBuf, model: M) {
    let mut store = burn_store::safetensors::SafetensorsStore::from_file(path.clone());
    model.collect_to(&mut store).expect("Failed to save model");
}

#[cfg(feature = "batched")]
fn load_batched<B: Backend + BatchTensorOps, M: Module<B>>(module: &mut M, path: PathBuf) {
    use burn_store::safetensors::apply_batched;
    let mut store = burn_store::safetensors::SafetensorsStore::from_file(path);
    apply_batched::<B, _>(&mut store, module).expect("batched apply failed");
}

#[cfg(not(feature = "batched"))]
fn load_default<B: Backend, M: Module<B>>(module: &mut M, path: PathBuf) {
    let mut store = burn_store::safetensors::SafetensorsStore::from_file(path);
    let _ = module.apply_from(&mut store);
}

fn main() {
    println!("Benchmarking safetensors apply: {}", if cfg!(feature = "batched") {"batched"} else {"default"});
    divan::main();
}

#[divan::bench_group(sample_count = 20)]
mod medium_model {
    use super::*;
    type B = TestBackend;
    type D = <B as Backend>::Device;

    fn setup() -> (tempfile::TempDir, PathBuf, u64) {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("medium_model.safetensors");
        let device: D = Default::default();
        let model = MediumModel::<B>::new(&device);
        create_test_file(&file_path, model);
        let file_size = fs::metadata(&file_path).unwrap().len();
        (temp_dir, file_path, file_size)
    }

    #[divan::bench(name = "medium new_store")] 
    fn medium_new_store(bencher: Bencher) {
        let (_temp_dir, file_path, file_size) = setup();
        bencher.counter(divan::counter::BytesCount::new(file_size)).bench(|| {
            let device: D = Default::default();
            let mut model = MediumModel::<B>::new(&device);
            #[cfg(feature = "batched")]
            { load_batched::<B, _>(&mut model, file_path.clone()); }
            #[cfg(not(feature = "batched"))]
            { load_default::<B, _>(&mut model, file_path.clone()); }
        });
    }
}

#[divan::bench_group(sample_count = 10)]
mod large_model {
    use super::*;
    type B = TestBackend;
    type D = <B as Backend>::Device;

    fn setup() -> (tempfile::TempDir, PathBuf, u64) {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("large_model.safetensors");
        let device: D = Default::default();
        let model = LargeModel::<B>::new(&device);
        create_test_file(&file_path, model);
        let file_size = fs::metadata(&file_path).unwrap().len();
        (temp_dir, file_path, file_size)
    }

    #[divan::bench(name = "large new_store")] 
    fn large_new_store(bencher: Bencher) {
        let (_temp_dir, file_path, file_size) = setup();
        bencher.counter(divan::counter::BytesCount::new(file_size)).bench(|| {
            let device: D = Default::default();
            let mut model = LargeModel::<B>::new(&device);
            #[cfg(feature = "batched")]
            { load_batched::<B, _>(&mut model, file_path.clone()); }
            #[cfg(not(feature = "batched"))]
            { load_default::<B, _>(&mut model, file_path.clone()); }
        });
    }
}
