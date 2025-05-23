cd resnet-torchvision
source venv/bin/activate
python infer.py ../data/labrador.jpg ../data/resnet18-f37072fd.pth

cd ../resnet-burn
cargo run --quiet --release ../data/labrador.jpg ../data/resnet18-f37072fd.pth
