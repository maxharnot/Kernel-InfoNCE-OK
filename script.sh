# CIFAR-10

# SimCLR (repro.)
python simclr_module.py \
  --batch_size=256 \
  --dataset="cifar10" \
  --fast_dev_run=0 \
  --gamma=1 \
  --jitter_strength=0.5 \
  --loss_type="spectral" \
  --max_epochs=100 \
  --optimizer='lars'

# Laplacian Kernel
python simclr_module.py
  --dataset="cifar10"
  --fast_dev_run=0
  --gamma=1
  --jitter_strength=0.5
  --loss_type="origin"
  --max_epochs=100
  --optimizer='lars'

# Exponential Kernel (gamma = 0.5)
python simclr_module.py
  --dataset="cifar10"
  --fast_dev_run=0
  --gamma=0.5
  --jitter_strength=0.5
  --loss_type="origin"
  --max_epochs=100
  --optimizer='lars'

# Simple Sum Kernel
python simclr_module.py
  --dataset="cifar10"
  --fast_dev_run=0
  --gamma=1
  --jitter_strength=0.5
  --loss_type="sum"
  --max_epochs=100
  --optimizer='lars'

# Concatenation Sum Kernel
python simclr_module.py
  --dataset="cifar10"
  --fast_dev_run=0
  --feat_dim=256  # default 128
  --gamma=1
  --jitter_strength=0.5
  --loss_type="product"
  --max_epochs=100
  --optimizer='lars'