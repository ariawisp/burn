#![recursion_limit = "256"]

use burn_core::module::Module;
use burn_core::nn;
use burn_core::prelude::*;
use burn_core::record::{FullPrecisionSettings, Recorder};
use burn_import::safetensors::SafetensorsFileRecorder;
use burn_store::ModuleSnapshot;

#[cfg(feature = "metal")]
mod cube_batched_store {
    use super::*;
    use hashbrown::HashMap;
    use burn_store::safetensors::{SafetensorsStore, store::SafetensorsError};
    use burn_store::{ModuleSnapshoter, TensorSnapshot, ApplyResult};
    use burn_tensor::{backend::Backend, DType, TensorData, BatchTensorOps};
    use burn_core::module::{ModuleMapper, ParamId};

    pub struct CubeBatchedStore(pub SafetensorsStore);
    impl CubeBatchedStore {
        pub fn from_file(path: impl Into<std::path::PathBuf>) -> Self { Self(SafetensorsStore::from_file(path)) }
    }
    impl ModuleSnapshoter for CubeBatchedStore {
        type Error = SafetensorsError;
        fn collect_from<B: Backend, M: ModuleSnapshot<B>>(&mut self, module: &M) -> Result<(), Self::Error> { self.0.collect_from(module) }
        fn apply_to<B: Backend, M: ModuleSnapshot<B>>(&mut self, module: &mut M) -> Result<ApplyResult, Self::Error> {
            apply_batched::<B, M>(&mut self.0, module)
        }
    }

    fn apply_batched<B: Backend + BatchTensorOps, M: ModuleSnapshot<B>>(store: &mut SafetensorsStore, module: &mut M) -> Result<ApplyResult, SafetensorsError> {
        // snapshots
        let mut snapshots = match store {
            #[cfg(feature = "std")]
            SafetensorsStore::File(p) => burn_store::safetensors::store::safetensors_to_snapshots_lazy_file(&p.path)?,
            SafetensorsStore::Memory(p) => {
                let data_arc = p.data.clone().ok_or_else(|| SafetensorsError::Other("No data loaded".to_string()))?;
                burn_store::safetensors::store::safetensors_to_snapshots_lazy(data_arc)?
            }
        };
        let from_adapter = store.get_from_adapter();
        snapshots = snapshots.into_iter().filter_map(|s| from_adapter.adapt_tensor(&s)).collect();

        let mut snap_by_path: HashMap<String, TensorSnapshot> = HashMap::new();
        for s in snapshots { snap_by_path.insert(s.full_path(), s); }

        enum Kind { Float, Int, Bool }
        struct Expect<B: Backend> { list: Vec<(String, Kind, Vec<usize>, DType, B::Device)>, path: Vec<String> }
        impl<B: Backend> Expect<B> { fn new()->Self{Self{list:Vec::new(),path:Vec::new()}} fn cur(&self)->String{self.path.join(".")} }
        impl<B: Backend> ModuleMapper<B> for Expect<B> {
            fn enter_module(&mut self, n:&str,_:&str){ self.path.push(n.to_string()); }
            fn exit_module(&mut self,_:&str,_:&str){ self.path.pop(); }
            fn map_float<const D:usize>(&mut self,_:ParamId,t:burn_tensor::Tensor<B,D>)->burn_tensor::Tensor<B,D>{ self.list.push((self.cur(),Kind::Float,t.shape().dims.to_vec(),t.dtype(),t.device())); t }
            fn map_int<const D:usize>(&mut self,_:ParamId,t:burn_tensor::Tensor<B,D,burn_tensor::Int>)->burn_tensor::Tensor<B,D,burn_tensor::Int>{ self.list.push((self.cur(),Kind::Int,t.shape().dims.to_vec(),t.dtype(),t.device())); t }
            fn map_bool<const D:usize>(&mut self,_:ParamId,t:burn_tensor::Tensor<B,D,burn_tensor::Bool>)->burn_tensor::Tensor<B,D,burn_tensor::Bool>{ self.list.push((self.cur(),Kind::Bool,t.shape().dims.to_vec(),t.dtype(),t.device())); t }
        }
        let mut probe = Expect::<B>::new();
        let _ = module.clone().map(&mut probe);

        let mut float_items: Vec<(TensorData, B::Device, String)> = Vec::new();
        let mut int_items: Vec<(TensorData, B::Device, String)> = Vec::new();
        let mut bool_items: Vec<(TensorData, B::Device, String)> = Vec::new();
        let mut applied: Vec<String> = Vec::new();
        let mut missing: Vec<String> = Vec::new();
        let mut errors: Vec<String> = Vec::new();
        for (path, kind, shape, dtype, dev) in probe.list.into_iter() {
            match snap_by_path.remove(&path) {
                None => missing.push(path),
                Some(view) => {
                    let data0 = view.to_data();
                    if data0.shape != shape { errors.push(format!("Shape mismatch for '{}': expected {:?}, found {:?}", path, shape, data0.shape)); continue; }
                    if data0.dtype != dtype { errors.push(format!("Type mismatch for '{}': expected {:?}, found {:?}", path, dtype, data0.dtype)); continue; }
                    match kind {
                        Kind::Float => float_items.push((data0, dev.clone(), path.clone())),
                        Kind::Int => int_items.push((data0, dev.clone(), path.clone())),
                        Kind::Bool => bool_items.push((data0, dev.clone(), path.clone())),
                    }
                    applied.push(path);
                }
            }
        }
        let floats = <B as BatchTensorOps>::float_batch_from_data(float_items.iter().map(|(d,dev,_)|(d.clone(),dev.clone())).collect());
        let ints = <B as BatchTensorOps>::int_batch_from_data(int_items.iter().map(|(d,dev,_)|(d.clone(),dev.clone())).collect());
        let bools = <B as BatchTensorOps>::bool_batch_from_data(bool_items.iter().map(|(d,dev,_)|(d.clone(),dev.clone())).collect());

        use hashbrown::HashMap as StdHashMap;
        let mut fmap: StdHashMap<String, burn_tensor::ops::FloatTensor<B>> = StdHashMap::new();
        let mut imap: StdHashMap<String, burn_tensor::ops::IntTensor<B>> = StdHashMap::new();
        let mut bmap: StdHashMap<String, burn_tensor::ops::BoolTensor<B>> = StdHashMap::new();
        for ((_,_,p),t) in float_items.into_iter().zip(floats.into_iter()) { fmap.insert(p,t); }
        for ((_,_,p),t) in int_items.into_iter().zip(ints.into_iter()) { imap.insert(p,t); }
        for ((_,_,p),t) in bool_items.into_iter().zip(bools.into_iter()) { bmap.insert(p,t); }

        struct Install<'a,B:Backend>{f:&'a mut StdHashMap<String,burn_tensor::ops::FloatTensor<B>>, i:&'a mut StdHashMap<String,burn_tensor::ops::IntTensor<B>>, b:&'a mut StdHashMap<String,burn_tensor::ops::BoolTensor<B>>, path:Vec<String>}
        impl<'a,B:Backend> Install<'a,B>{ fn new(f:&'a mut _, i:&'a mut _, b:&'a mut _)->Self{ Self{f,i,b,path:Vec::new()} } fn cur(&self)->String{ self.path.join(".") } }
        impl<'a,B:Backend> ModuleMapper<B> for Install<'a,B>{
            fn enter_module(&mut self,n:&str,_:&str){ self.path.push(n.to_string()); }
            fn exit_module(&mut self,_:&str,_:&str){ self.path.pop(); }
            fn map_float<const D:usize>(&mut self,_:ParamId,_:burn_tensor::Tensor<B,D>)->burn_tensor::Tensor<B,D>{ let k=self.cur(); let prim=self.f.remove(&k).unwrap(); burn_tensor::Tensor::from_primitive(burn_tensor::TensorPrimitive::Float(prim)) }
            fn map_int<const D:usize>(&mut self,_:ParamId,_:burn_tensor::Tensor<B,D,burn_tensor::Int>)->burn_tensor::Tensor<B,D,burn_tensor::Int>{ let k=self.cur(); let prim=self.i.remove(&k).unwrap(); burn_tensor::Tensor::from_primitive(prim) }
            fn map_bool<const D:usize>(&mut self,_:ParamId,_:burn_tensor::Tensor<B,D,burn_tensor::Bool>)->burn_tensor::Tensor<B,D,burn_tensor::Bool>{ let k=self.cur(); let prim=self.b.remove(&k).unwrap(); burn_tensor::Tensor::from_primitive(prim) }
        }
        let mut installer = Install::<B>::new(&mut fmap, &mut imap, &mut bmap);
        *module = module.clone().map(&mut installer);
        let unused: Vec<String> = snap_by_path.keys().cloned().collect();
        let result = ApplyResult { applied, skipped: Vec::new(), missing, unused, errors };
        if store.get_validate() && !result.errors.is_empty() { return Err(SafetensorsError::ValidationFailed(format!("Import errors: {:?}", result.errors))); }
        if !store.get_allow_partial() && !result.missing.is_empty() { return Err(SafetensorsError::TensorNotFound(format!("Missing tensors: {:?}", result.missing))); }
        Ok(result)
    }
}

#[cfg(feature = "metal")]
use cube_batched_store::CubeBatchedStore as Store;
#[cfg(not(feature = "metal"))]
type Store = burn_store::safetensors::SafetensorsStore;
use burn_store::safetensors::SafetensorsStore;
use divan::{AllocProfiler, Bencher};
use std::fs;
use std::path::PathBuf;
use tempfile::tempdir;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

// Backend type aliases for easy switching
type NdArrayBackend = burn_ndarray::NdArray<f32>;

#[cfg(feature = "wgpu")]
type WgpuBackend = burn_wgpu::Wgpu;

#[cfg(feature = "cuda")]
type CudaBackend = burn_cuda::Cuda<f32, i32>;

#[cfg(feature = "candle")]
type CandleBackend = burn_candle::Candle<f32, i64>;

#[cfg(feature = "tch")]
type TchBackend = burn_tch::LibTorch<f32>;

#[cfg(feature = "metal")]
type MetalBackend = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;

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
    conv1: nn::conv::Conv2d<B>,
    conv2: nn::conv::Conv2d<B>,
}

impl<B: Backend> MediumModel<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            linear1: nn::LinearConfig::new(512, 1024).init(device),
            linear2: nn::LinearConfig::new(1024, 2048).init(device),
            linear3: nn::LinearConfig::new(2048, 4096).init(device),
            conv1: nn::conv::Conv2dConfig::new([3, 64], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            conv2: nn::conv::Conv2dConfig::new([64, 128], [5, 5])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
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
        // Create a model with 20 layers
        for i in 0..20 {
            let in_size = if i == 0 { 1024 } else { 2048 };
            layers.push(nn::LinearConfig::new(in_size, 2048).init(device));
        }
        Self { layers }
    }
}

fn create_test_file<B: Backend, M: Module<B>>(path: &PathBuf, model: M) {
    let mut store = SafetensorsStore::from_file(path.clone());
    model.collect_to(&mut store).expect("Failed to save model");
}

fn main() {
    println!("Available backends:");
    println!("  - NdArray (CPU)");
    #[cfg(feature = "wgpu")]
    println!("  - WGPU (GPU)");
    #[cfg(feature = "cuda")]
    println!("  - CUDA (NVIDIA GPU)");
    #[cfg(feature = "candle")]
    println!("  - Candle");
    #[cfg(feature = "tch")]
    println!("  - LibTorch");
    #[cfg(feature = "metal")]
    println!("  - Metal (Apple GPU)");
    println!();

    divan::main();
}

// Macro to generate benchmarks for each backend
macro_rules! bench_backend {
    ($backend:ty, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            type TestBackend = $backend;
            type TestDevice = <TestBackend as Backend>::Device;

            #[divan::bench_group(sample_count = 30)]
            mod small_model {
                use super::*;

                fn setup_simple_model_file() -> (tempfile::TempDir, PathBuf, u64) {
                    let temp_dir = tempdir().unwrap();
                    let file_path = temp_dir.path().join("simple_model.safetensors");
                    let device: TestDevice = Default::default();
                    let model = SimpleModel::<TestBackend>::new(&device);
                    create_test_file(&file_path, model);
                    let file_size = fs::metadata(&file_path).unwrap().len();
                    (temp_dir, file_path, file_size)
                }

                #[divan::bench(name = "old_recorder")]
                fn old_recorder_simple(bencher: Bencher) {
                    let (_temp_dir, file_path, file_size) = setup_simple_model_file();

                    bencher
                        .counter(divan::counter::BytesCount::new(file_size))
                        .bench(|| {
                            let device: TestDevice = Default::default();
                            let recorder =
                                SafetensorsFileRecorder::<FullPrecisionSettings>::default();
                            let record = recorder
                                .load(file_path.clone().into(), &device)
                                .expect("Failed to load");
                            let _model =
                                SimpleModel::<TestBackend>::new(&device).load_record(record);
                        });
                }

                #[divan::bench(name = "new_store")]
                fn new_store_simple(bencher: Bencher) {
                    let (_temp_dir, file_path, file_size) = setup_simple_model_file();

                    bencher
                        .counter(divan::counter::BytesCount::new(file_size))
                        .bench(|| {
                            let device: TestDevice = Default::default();
                            let mut model = SimpleModel::<TestBackend>::new(&device);
    let mut store = Store::from_file(file_path.clone());
                            model.apply_from(&mut store).expect("Failed to load");
                        });
                }
            }

            #[divan::bench_group(sample_count = 20)]
            mod medium_model {
                use super::*;

                fn setup_medium_model_file() -> (tempfile::TempDir, PathBuf, u64) {
                    let temp_dir = tempdir().unwrap();
                    let file_path = temp_dir.path().join("medium_model.safetensors");
                    let device: TestDevice = Default::default();
                    let model = MediumModel::<TestBackend>::new(&device);
                    create_test_file(&file_path, model);
                    let file_size = fs::metadata(&file_path).unwrap().len();
                    (temp_dir, file_path, file_size)
                }

                #[divan::bench(name = "old_recorder")]
                fn old_recorder_medium(bencher: Bencher) {
                    let (_temp_dir, file_path, file_size) = setup_medium_model_file();

                    bencher
                        .counter(divan::counter::BytesCount::new(file_size))
                        .bench(|| {
                            let device: TestDevice = Default::default();
                            let recorder =
                                SafetensorsFileRecorder::<FullPrecisionSettings>::default();
                            let record = recorder
                                .load(file_path.clone().into(), &device)
                                .expect("Failed to load");
                            let _model =
                                MediumModel::<TestBackend>::new(&device).load_record(record);
                        });
                }

                #[divan::bench(name = "new_store")]
                fn new_store_medium(bencher: Bencher) {
                    let (_temp_dir, file_path, file_size) = setup_medium_model_file();

                    bencher
                        .counter(divan::counter::BytesCount::new(file_size))
                        .bench(|| {
                            let device: TestDevice = Default::default();
                            let mut model = MediumModel::<TestBackend>::new(&device);
    let mut store = Store::from_file(file_path.clone());
                            model.apply_from(&mut store).expect("Failed to load");
                        });
                }
            }

            #[divan::bench_group(sample_count = 10)]
            mod large_model {
                use super::*;

                fn setup_large_model_file() -> (tempfile::TempDir, PathBuf, u64) {
                    let temp_dir = tempdir().unwrap();
                    let file_path = temp_dir.path().join("large_model.safetensors");
                    let device: TestDevice = Default::default();
                    let model = LargeModel::<TestBackend>::new(&device);
                    create_test_file(&file_path, model);
                    let file_size = fs::metadata(&file_path).unwrap().len();
                    (temp_dir, file_path, file_size)
                }

                #[divan::bench(name = "old_recorder")]
                fn old_recorder_large(bencher: Bencher) {
                    let (_temp_dir, file_path, file_size) = setup_large_model_file();

                    bencher
                        .counter(divan::counter::BytesCount::new(file_size))
                        .bench(|| {
                            let device: TestDevice = Default::default();
                            let recorder =
                                SafetensorsFileRecorder::<FullPrecisionSettings>::default();
                            let record = recorder
                                .load(file_path.clone().into(), &device)
                                .expect("Failed to load");
                            let _model =
                                LargeModel::<TestBackend>::new(&device).load_record(record);
                        });
                }

                #[divan::bench(name = "new_store")]
                fn new_store_large(bencher: Bencher) {
                    let (_temp_dir, file_path, file_size) = setup_large_model_file();

                    bencher
                        .counter(divan::counter::BytesCount::new(file_size))
                        .bench(|| {
                            let device: TestDevice = Default::default();
                            let mut model = LargeModel::<TestBackend>::new(&device);
                            let mut store = SafetensorsStore::from_file(file_path.clone());
                            model.apply_from(&mut store).expect("Failed to load");
                        });
                }
            }
        }
    };
}

// Generate benchmarks for NdArray backend (always available)
bench_backend!(NdArrayBackend, ndarray_backend, "NdArray Backend (CPU)");

// Generate benchmarks for WGPU backend (if feature enabled)
#[cfg(feature = "wgpu")]
bench_backend!(WgpuBackend, wgpu_backend, "WGPU Backend (GPU)");

// Generate benchmarks for CUDA backend (if feature enabled)
#[cfg(feature = "cuda")]
bench_backend!(CudaBackend, cuda_backend, "CUDA Backend (NVIDIA GPU)");

// Generate benchmarks for Candle backend (if feature enabled)
#[cfg(feature = "candle")]
bench_backend!(CandleBackend, candle_backend, "Candle Backend");

// Generate benchmarks for LibTorch backend (if feature enabled)
#[cfg(feature = "tch")]
bench_backend!(TchBackend, tch_backend, "LibTorch Backend");

// Generate benchmarks for Metal backend (if feature enabled on macOS)
#[cfg(feature = "metal")]
bench_backend!(MetalBackend, metal_backend, "Metal Backend (Apple GPU)");
