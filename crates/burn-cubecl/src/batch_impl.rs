use crate::{CubeBackend, CubeRuntime};
use crate::tensor::CubeTensor;
use alloc::vec::Vec;
use burn_tensor::backend::Backend;
use burn_tensor::tensor::batch::BatchTensorOps;
use burn_tensor::TensorData;
use cubecl::server::AllocationDescriptor;

impl<R, F, I, BT> BatchTensorOps for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: crate::FloatElement,
    I: crate::IntElement,
    BT: crate::element::BoolElement,
{
    fn float_batch_from_data(
        items: Vec<(TensorData, <Self as Backend>::Device)>,
    ) -> Vec<burn_tensor::ops::FloatTensor<Self>> {
        batch_create::<R>(items)
    }

    fn int_batch_from_data(
        items: Vec<(TensorData, <Self as Backend>::Device)>,
    ) -> Vec<burn_tensor::ops::IntTensor<Self>> {
        batch_create::<R>(items)
    }

    fn bool_batch_from_data(
        items: Vec<(TensorData, <Self as Backend>::Device)>,
    ) -> Vec<burn_tensor::ops::BoolTensor<Self>> {
        batch_create::<R>(items)
    }
}

fn batch_create<R: CubeRuntime>(
    mut items: Vec<(TensorData, R::Device)>,
) -> Vec<CubeTensor<R>> {
    // Group by device instance; do simple O(n^2) grouping to avoid trait bounds on Device.
    let mut result: Vec<CubeTensor<R>> = Vec::with_capacity(items.len());

    while !items.is_empty() {
        let (first_data, first_dev) = items.remove(0);
        let mut group: Vec<(TensorData, R::Device)> = Vec::new();
        group.push((first_data, first_dev.clone()));

        let mut i = 0;
        while i < items.len() {
            if items[i].1 == first_dev {
                let entry = items.remove(i);
                group.push(entry);
            } else {
                i += 1;
            }
        }

        // Perform a single create_tensors for this device.
        let client = R::client(&first_dev);
        let mut descs: Vec<(AllocationDescriptor<'_>, &[u8])> = Vec::with_capacity(group.len());
        let mut shapes: Vec<Vec<usize>> = Vec::with_capacity(group.len());
        let mut dtypes: Vec<burn_tensor::DType> = Vec::with_capacity(group.len());
        for (data, _) in group.iter() {
            let shape = data.shape.clone();
            let elem_size = data.dtype.size();
            shapes.push(shape);
            dtypes.push(data.dtype);
            descs.push((AllocationDescriptor::optimized(&group.last().unwrap().0.shape, elem_size), data.as_bytes()));
        }

        // Fix descriptors to use the corresponding shapes; the above used last() incorrectly for lifetime.
        descs.clear();
        for (idx, (data, _)) in group.iter().enumerate() {
            let elem_size = data.dtype.size();
            descs.push((AllocationDescriptor::optimized(&shapes[idx], elem_size), data.as_bytes()));
        }

        let allocations = client.create_tensors(descs);

        for (idx, alloc) in allocations.into_iter().enumerate() {
            let shape = cubecl::prelude::Shape::from(shapes[idx].clone());
            let tensor = CubeTensor::new_contiguous(
                client.clone(),
                first_dev.clone(),
                shape,
                alloc.handle,
                dtypes[idx],
            );
            result.push(tensor);
        }
    }

    result
}
