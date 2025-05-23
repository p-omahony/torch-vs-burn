#!/usr/bin/env python

import argparse
import time
from PIL import Image
import torch
from torchvision import models, transforms

def main():
    print(25*"=", "TORCH INFERENCE", 25*"=")
    parser = argparse.ArgumentParser(description="Run ResNet-18 inference on an input image.")
    parser.add_argument("image_path", type=str, help="Path to the input image (e.g., ../data/dog.jpg)")
    parser.add_argument("weights_path", type=str, help="Path to the model weights")
    args = parser.parse_args()

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(num_classes=1000, weights=None)
    ckpt = torch.load(args.weights_path, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

    img = Image.open(args.image_path).convert("RGB")
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)

    devices = set(p.device for p in model.parameters())
    devices.update(b.device for b in model.buffers())
    print(f"Device is: {list(devices)[0]}")
    start = time.time()
    with torch.no_grad():
        print(f"Input shape: {input_batch.shape}")
        print(f"Input dtype: {input_batch.dtype}")
        output = model(input_batch)
    duration = (time.time() - start)*1000

    print(f"Inference time: {duration:.2f}ms")
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    class_id = torch.argmax(probabilities).item()
    class_label = weights.meta["categories"][class_id]
    print(f"Predicted: {class_label}")


if __name__ == "__main__":
    main()

