use resnet_burn::model::{imagenet, resnet::ResNet};
use std::time::Instant;

use burn::{
    backend::NdArray,
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::{backend::Backend, Device, Element, Shape, Tensor, TensorData},
};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

const MODEL_PATH: &str = "resnet18-ImageNet1k";
const HEIGHT: usize = 224;
const WIDTH: usize = 224;

fn to_tensor<B: Backend, T: Element>(
    data: Vec<T>,
    shape: [usize; 3],
    device: &Device<B>,
) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(
        TensorData::new(data, Shape::new(shape)).convert::<f32>(),
        device,
    )
    .swap_dims(2, 1)
    .swap_dims(1, 0)
        / 255
}

fn main() {
    println!("{} BURN INFERENCE {}", "=".repeat(25), "=".repeat(25));
    let torch_weights = std::env::args().nth(2).expect("No image path provided");

    let device = Default::default();
    println!("Device is: {:?}", device);
    let model: ResNet<NdArray, _> = ResNet::resnet18(1000, &device);

    let load_args = LoadArgs::new(torch_weights.into())
        .with_key_remap("(.+)\\.downsample\\.0\\.(.+)", "$1.downsample.conv.$2")
        .with_key_remap("(.+)\\.downsample\\.1\\.(.+)", "$1.downsample.bn.$2")
        .with_key_remap("(layer[1-4])\\.([0-9]+)\\.(.+)", "$1.blocks.$2.$3");
    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new()
        .load(load_args, &device)
        .expect("Should load PyTorch model weights");

    let model = model.load_record(record);

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(MODEL_PATH, &recorder)
        .expect("Should be able to save weights to file");
    let model = model
        .load_file(MODEL_PATH, &recorder, &device)
        .expect("Should be able to load weights from file");

    let img_path = std::env::args().nth(1).expect("No image path provided");
    let img = image::open(img_path).expect("Should be able to load image");

    let resized_img = img.resize_exact(
        WIDTH as u32,
        HEIGHT as u32,
        image::imageops::FilterType::Triangle,
    );

    let img_tensor: Tensor<NdArray, 4> = to_tensor(
        resized_img.into_rgb8().into_raw(),
        [HEIGHT, WIDTH, 3],
        &device,
    )
    .unsqueeze::<4>();

    let x = imagenet::Normalizer::new(&device).normalize(img_tensor);
    println!("Input shape: {:?}", x.shape().dims::<4>());
    println!("Input dtype: {:?}", x.dtype());

    let start = Instant::now();
    let out = model.forward(x);
    let duration = start.elapsed();
    println!("Inference time: {:?}", duration);

    let (_score, idx) = out.max_dim_with_indices(1);
    let idx = idx.into_scalar() as usize;

    println!("Predicted: {}", imagenet::CLASSES[idx],);
}
