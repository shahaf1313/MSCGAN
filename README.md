
Running MSC-GAN Model:
All configurations can be found at config.py file. All configuration parameters has default value and help paragraph. Change them to match your environment (paths to datasets/lists/etc.)

To train the MSC-GAN model, use main_train.py. Example for a running command:

python3 ./main_train.py --scale_factor=0.5 --num_scales=5 --num_workers=16 --batch_size_list 20 20 20 20 8 2 --gpus=0 --nfc 16 --epochs_per_scale=40  --Gsteps 1 --Dsteps 1 --lambda_cyclic 0.85 --lambda_adversarial 1 --num_layer 7

To train a semantic segmentation net after MSC-GAN model was trained, use train_semseg_pyramid.py. Exmaple for running command: 

python3 ./train_semseg_pyramid.py --gpus=4 --multiscale_model_path=./GoldenModels/ConvGenerator/Gst.pth --batch_size=2 --scale_factor=0.5 --num_steps=200000 --model=VGG --tb_logs_dir=/home/shahaf/SinGAN/TrainSemseg/tb_runs .

Enjoy!
Shahaf