from torch.utils import data
from data.gta5_dataset import GTA5DataSet
import os.path as osp
import numpy as np
from data.cityscapes_dataset import cityscapesDataSet
from data.synthia_dataset import SYNDataSet
from data.im2im_dataset import Im2imDataset


def CreateSrcDataLoader(opt, set='train', get_image_label=False):
    if opt.source == 'gta5':
        source_dataset = GTA5DataSet(opt.src_data_dir,
                                     opt.src_data_list,
                                     opt.scale_factor,
                                     opt.num_scales,
                                     opt.curr_scale,
                                     set,
                                     get_image_label=get_image_label)
    # elif args.source == 'synthia':
    #     source_dataset = SYNDataSet(args.data_dir, args.data_list, crop_size=image_sizes['cityscapes'], resize=image_sizes['synthia'], mean=IMG_MEAN)
    else:
        raise ValueError('The source dataset mush be either gta5 or synthia')

    source_dataloader = data.DataLoader(source_dataset,
                                        batch_size=opt.batch_size,
                                        shuffle=True,
                                        num_workers=opt.num_workers,
                                        pin_memory=True)
    return source_dataloader


def CreateTrgDataLoader(opt, set='train', get_image_label=False, get_scales_pyramid=False):
    target_dataset = cityscapesDataSet(opt.trg_data_dir,
                                       opt.trg_data_list,
                                       opt.scale_factor,
                                       opt.num_scales,
                                       opt.curr_scale,
                                       set,
                                       get_image_label=get_image_label,
                                       get_scales_pyramid=get_scales_pyramid)

    if set == 'train':
        target_dataloader = data.DataLoader(target_dataset,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            num_workers=opt.num_workers,
                                            pin_memory=True)
    elif set == 'val' or set == 'test':
        target_dataloader = data.DataLoader(target_dataset,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            num_workers=opt.num_workers,
                                            pin_memory=True)
    else:
        raise Exception("Argument set has not entered properly. Options are train or eval.")

    return target_dataloader

def CreateIm2ImDataLoader(opt, set='train'):

    domain_a_dataset = Im2imDataset( osp.join(opt.im2im_data_dir, 'trainA'),
                                     opt.scale_factor,
                                     opt.num_scales,
                                     opt.curr_scale,
                                     set)


    domain_b_dataset = Im2imDataset( osp.join(opt.im2im_data_dir, 'trainB'),
                                     opt.scale_factor,
                                     opt.num_scales,
                                     opt.curr_scale,
                                     set)
    opt.epoch_size = np.maximum(len(domain_a_dataset),len(domain_b_dataset))
    if len(domain_a_dataset) != len(domain_b_dataset):
        if opt.epoch_size > len(domain_a_dataset): #domain B is larger
            domain_a_dataset.SetEpochSize(len(domain_b_dataset))
        else: #domain A is larger
            domain_b_dataset.SetEpochSize(len(domain_a_dataset))

    domain_a_dataloader = data.DataLoader(domain_a_dataset,
                                      batch_size=opt.batch_size,
                                      shuffle=True,
                                      num_workers=opt.num_workers,
                                      pin_memory=True)

    domain_b_dataloader = data.DataLoader(domain_b_dataset,
                                   batch_size=opt.batch_size,
                                   shuffle=True,
                                   num_workers=opt.num_workers,
                                   pin_memory=True)


    return domain_a_dataloader, domain_b_dataloader
