model_name="laplacian_kernel"
dataset="cifar100"
ckpt_dir="/content/drive/MyDrive/_simclr_ok/${model_name}/${dataset}/checkpoints/"
ckpt_path=$(find $ckpt_dir -name "epoch*")

echo python simclr_finetune.py \
  --ckpt_path=$ckpt_path
  --dataset=${dataset}
  --batch_size=256
  --gpus=1
  --num_epochs=50
  --gamma=1
