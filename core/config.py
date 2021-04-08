import argparse
import torch
import datetime
import random
import numpy as np
import os
import sys

def get_arguments():
    parser = argparse.ArgumentParser()
    # workspace:
    parser.add_argument("--gpus", type=int, nargs='+', help="String that contains available GPUs to use", default=[0])
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)

    # load, input, save configurations:
    parser.add_argument('--manualSeed', default=1337, type=int, help='manual seed')
    parser.add_argument('--continue_train_from_path', type=str, help='Path to folder that contains all networks and continues to train from there', default='')
    parser.add_argument('--train_last_scale_more', default=False, action='store_true', help='Continues to train last generator and critic.')
    parser.add_argument('--nc_z', type=int, help='noise # channels', default=3)
    parser.add_argument('--nc_im', type=int, help='image # channels', default=3)
    parser.add_argument('--out', help='output folder', default='Output')

    # Dataset parameters:
    parser.add_argument("--source", type=str, default='gta5', help="source dataset : gta5 or synthia")
    parser.add_argument("--target", type=str, default='cityscapes', help="target dataset : cityscapes")
    parser.add_argument("--src_data_dir", type=str, default='/home/shahaf/data/GTA5', help="Path to the directory containing the source dataset.")
    parser.add_argument("--src_data_list", type=str, default='./dataset/gta5_list/', help="Path to folder that contains a file with a list of images from the source dataset. File named set.txt, where set is train/val/test.")
    parser.add_argument("--trg_data_dir", type=str, default='/home/shahaf/data/cityscapes', help="Path to the directory containing the target dataset.")
    parser.add_argument("--trg_data_list", type=str, default='./dataset/cityscapes_list/', help="Path to folder that contains a file with a list of images from the target dataset. File named set.txt, where set is train/val/test.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of threads for each worker")

    # networks parameters:
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--batch_size_list', type=int, nargs='+', help="batch size in each one of the scales", default=[0])
    parser.add_argument('--norm_type', type=str, default='batch_norm')
    parser.add_argument('--use_unet_generator', default=False, action='store_true', help='Uses U-Net as a generator from large enough scale')
    parser.add_argument('--use_downscale_discriminator', default=False, action='store_true', help='Uses Downscaled discriminator')
    parser.add_argument('--nfc', type=int, default=8)
    parser.add_argument('--ker_size', type=int, help='kernel size', default=3)
    parser.add_argument('--num_layer', type=int, help='number of layers', default=5)
    parser.add_argument('--stride', help='stride', default=1)
    parser.add_argument('--padd_size', type=int, help='net pad size', default=0)  # math.floor(opt.ker_size/2)

    # pyramid parameters:
    parser.add_argument('--scale_factor', type=float, help='pyramid scale factor', default=0.75)  # pow(0.5,1/6))
    parser.add_argument('--min_size', type=int, help='image minimal size at the coarser scale', default=None)
    parser.add_argument('--max_size', type=int, help='image maximal size at the largest scale', default=None)
    parser.add_argument('--num_scales', type=int, help='number of scales in the pyramid', default=None)

    # optimization hyper parameters:
    parser.add_argument('--epochs_per_scale', type=int, default=12, help='number of epochs to train per scale')
    parser.add_argument('--gamma', type=float, help='scheduler gamma', default=0.1)
    parser.add_argument('--lr_g', type=float, default=0.00025, help='learning rate, default=0.00025')
    parser.add_argument('--lr_d', type=float, default=0.00025, help='learning rate, default=0.00025')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--Gsteps', type=int, help='Generator inner steps', default=3)
    parser.add_argument('--Dsteps', type=int, help='Discriminator inner steps', default=3)
    parser.add_argument('--identity_loss_calc_rate', type=int, help='rate for identity loss calculation', default=0)
    parser.add_argument('--cyclic_loss_calc_rate', type=int, help='rate for cyclic loss calculation', default=1)
    parser.add_argument('--lambda_grad', type=float, help='gradient penelty weight', default=0.1)
    parser.add_argument('--lambda_adversarial', type=float, help='adversarial loss weight', default=1)
    parser.add_argument('--lambda_cyclic', type=float, help='cyclic loss weight', default=1)

    # Semseg network parameters:
    parser.add_argument("--model", type=str, required=False, default='DeepLab', help="available options : DeepLab and VGG")
    parser.add_argument('--epochs_semseg', type=int, default=12, help='number of epochs to train semseg model')
    parser.add_argument("--multiscale_model_path", type=str, default='', help="path to Generators from source to target domain and vice versa.")
    parser.add_argument("--semseg_model_path", type=str, default='', help="path to folder that contains classifier and feature extractor weights.")
    parser.add_argument("--semseg_model_epoch_to_resume", type=int, default=-1, help='Epoch that checkpoint to semseg net saved from')
    parser.add_argument('--load_only_gta_weights', default=False, action='store_true', help='Loads only source (GTA5) semeseg weigths.')
    parser.add_argument('--lr_semseg', type=float, default=0.00025, help='learning rate, default=0.00025')
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Regularisation parameter for L2-loss.")
    parser.add_argument("--ita", type=float, default=2.0, help="ita for robust entropy")
    parser.add_argument("--entW", type=float, default=0.005, help="weight for entropy")
    parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate (only for deeplab).")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")


    # Miscellaneous parameters:
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument("--tb_logs_dir", type=str, required=False, default='./runs', help="Path to Tensorboard logs dir.")
    parser.add_argument('--debug_run', default=False, action='store_true')
    parser.add_argument("--checkpoints_dir", type=str, required=False, default='./TrainedModels', help="Where to save snapshots of the model.")
    parser.add_argument("--print_rate", type=int, required=False, default=100, help="Print progress to screen every x iterations")
    parser.add_argument("--save_pics_rate", type=int, required=False, default=1000, help="Save images to tb every x iterations")

    return parser


class Logger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        self.log.flush()

def post_config(opt):
        # init fixed parameters
        if opt.debug_run:
            opt.tb_logs_dir = './debug_runs'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus)[1:-1].strip(' ').replace(" ", "")
        opt.device = torch.device('cpu' if opt.not_cuda else 'cuda')
        opt.out_ = 'TrainedModels/%s' % datetime.datetime.now().strftime('%d-%m-%Y::%H:%M:%S')
        try:
            os.makedirs(opt.out_)
        except OSError:
            pass
        opt.logger = Logger(os.path.join(opt.out_, 'log.txt'))
        sys.stdout = opt.logger
        opt.nfc_init = opt.nfc
        opt.min_nfc = opt.nfc
        opt.min_nfc_init = opt.min_nfc
        opt.scale_factor_init = opt.scale_factor

        if opt.manualSeed is None:
            opt.manualSeed = random.randint(1, 10000)
        print("Random Seed: ", opt.manualSeed)
        # torch.set_deterministic(True)
        # torch.backends.cudnn.deterministic = True
        random.seed(opt.manualSeed)
        torch.manual_seed(opt.manualSeed)
        torch.cuda.manual_seed(opt.manualSeed)
        np.random.RandomState(opt.manualSeed)
        np.random.seed(opt.manualSeed)

        if torch.cuda.is_available() and opt.not_cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        return opt