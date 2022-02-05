## Train:

Trained on one NVIDIA-A6000 GPU.

`python ./train_mscgan.py --gpus 2 --scale_factor=0.5 --num_scales=2 --num_workers=4 --batch_size_list 32 16 1 --epochs_per_scale=40 --lambda_cyclic=1 --lambda_adversarial=1 --lambda_labels=3 --lambda_style=10 --warmup_epochs=0 --groups_num=8 --base_channels_list 64 64 64 --lr_g=0.0001 --lr_d=0.0001 --lr_semseg=0.0001`