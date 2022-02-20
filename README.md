## Train:

Trained on one NVIDIA-A6000 GPU.

####GTA5 -> Cityscapes:
`python ./train_mscgan.py --gpus X --batch_size_list 32 16 2 --epochs_per_scale=40 --lambda_cyclic=1 --lambda_adversarial=1 --lambda_labels=3 --lambda_style=10 --warmup_epochs=0 --groups_num=8 --base_channels_list 64 64 64 --lr_g=0.0001 --lr_d=0.0001 --lr_semseg=0.0001`

Create GTA5 Source In Target Dataset From Trained Model:
` python ./create_sit_dataset.py --trained_msc_model_path=path-to-pretrained-sit-model --sit_dataset_path=path-to-save-output-images`

####Synthia -> Cityscapes:
`python ./train_mscgan.py --source=synthia --src_data_dir=/home/shahaf/data/synthia --src_data_list=./dataset/synthia_list/ --gpus X --batch_size_list 32 16 2 --epochs_per_scale=40 --lambda_cyclic=1 --lambda_adversarial=1 --lambda_labels=3 --lambda_style=10 --warmup_epochs=0 --groups_num=8 --base_channels_list 32 64 64 --lr_g=0.0001 --lr_d=0.0001 --lr_semseg=0.0001`

Create Synthia Source In Target Dataset From Trained Model:
` python ./create_sit_dataset.py --source=synthia --src_data_dir=/home/shahaf/data/synthia --src_data_list=./dataset/synthia_list/ --gpus X --batch_size=X --trained_msc_model_path=path-to-pretrained-sit-model --sit_dataset_path=path-to-save-output-images`
