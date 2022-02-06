## Train:

Trained on one NVIDIA-A6000 GPU.

####GTA5 -> Cityscapes:
`python ./train_mscgan.py --gpus X --scale_factor=0.5 --num_scales=2 --num_workers=4 --batch_size_list 32 16 2 --epochs_per_scale=40 --lambda_cyclic=1 --lambda_adversarial=1 --lambda_labels=3 --lambda_style=10 --warmup_epochs=0 --groups_num=8 --base_channels_list 64 64 64 --lr_g=0.0001 --lr_d=0.0001 --lr_semseg=0.0001`

####Synthia -> Cityscapes:
`python ./train_mscgan.py --source=synthia --src_data_dir=/home/shahaf/data/synthia --src_data_list=./dataset/synthia_list/ --gpus X --scale_factor=0.5 --num_scales=2 --num_workers=4 --batch_size_list 32 16 2 --epochs_per_scale=40 --lambda_cyclic=1 --lambda_adversarial=1 --lambda_labels=3 --lambda_style=10 --warmup_epochs=0 --groups_num=8 --base_channels_list 32 64 64 --lr_g=0.0001 --lr_d=0.0001 --lr_semseg=0.0001`
