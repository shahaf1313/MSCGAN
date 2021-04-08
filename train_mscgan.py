from constants import H,W
from data import CreateSrcDataLoader
from data import CreateTrgDataLoader
from config import get_arguments, post_config
from SinGAN.functions import adjust_scales2image
from SinGAN.training import train
import numpy as np

if __name__ == '__main__':
    parser = get_arguments()
    opt = parser.parse_args()
    opt = post_config(opt)

    opt.batch_size = 1
    opt.curr_scale = 0
    source_loader, target_loader = CreateSrcDataLoader(opt), CreateTrgDataLoader(opt, get_scales_pyramid=True)
    opt.epoch_size = np.maximum(len(target_loader.dataset), len(source_loader.dataset))

    source_loaders, target_loaders, num_epochs =[], [], []
    for i in range(opt.num_scales+1):
        opt.batch_size = opt.batch_size_list[i]
        opt.curr_scale = i
        source_loader, target_loader = CreateSrcDataLoader(opt), CreateTrgDataLoader(opt, get_scales_pyramid=True)
        source_loader.dataset.SetEpochSize(opt.epoch_size)
        target_loader.dataset.SetEpochSize(opt.epoch_size)
        source_loaders.append(source_loader)
        target_loaders.append(target_loader)

    opt.source_loaders = source_loaders
    opt.target_loaders = target_loaders

    adjust_scales2image(H, W, opt)
    print('########################### MLCGAN Configuration ##############################')
    for arg in vars(opt):
        print(arg + ': ' + str(getattr(opt, arg)))
    print('##################################################################################')
    train(opt)
    print('Finished Training.')
