# PyTorch or Burn ????

Inference comparison between PyTorch and Burn for a ResNet on a single image. Only the forward pass is timed.      
To build Rust: `cd resnet-burn && cargo build --release && cd ..`.    
To run it: `./inference-comparison.sh`.      

Result for ResNet18 on cpu:
```
========================= TORCH INFERENCE =========================
Device is: cpu
Input shape: torch.Size([1, 3, 224, 224])
Input dtype: torch.float32
Inference time: 18.08ms
Predicted: Labrador retriever
========================= BURN INFERENCE =========================
Device is: Cpu
Input shape: [1, 3, 224, 224]
Input dtype: F32
Inference time: 167.744042ms
Predicted: Labrador retriever
```

Am I missing something ??? Need investigation...
