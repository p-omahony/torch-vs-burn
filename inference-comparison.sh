#!/bin/bash

IMG="../data/labrador.jpg"
MODEL="../data/resnet18-f37072fd.pth"

# Python Inference
(
  cd resnet-torchvision || exit
  source venv/bin/activate
  python infer.py "$IMG" "$MODEL"
)

# Rust Inference
(
  cd resnet-burn || exit
  ./target/release/resnet_burn "$IMG" "$MODEL"
)

