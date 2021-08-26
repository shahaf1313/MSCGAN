from torch.utils import data
from data.gta5_dataset import GTA5DataSet
import os.path as osp
import numpy as np
from data.cityscapes_dataset import cityscapesDataSet
from data.synthia_dataset import SYNDataSet
from data.im2im_dataset import Im2imDataset


def CreateSrcDataLoader(opt, Gs, set='train', get_image_label=False, get_image_label_pyramid=False):
    if opt.source == 'gta5':
        source_dataset = GTA5DataSet(opt.src_data_dir,
                                     opt.src_data_list,
                                     opt.scale_factor,
                                     opt.num_scales,
                                     opt.curr_scale,
                                     set,
                                     Gs,
                                     get_image_label=get_image_label,
                                     get_image_label_pyramid=get_image_label_pyramid)
    # elif args.source == 'synthia':
    #     source_dataset = SYNDataSet(args.data_dir, args.data_list, crop_size=image_sizes['cityscapes'], resize=image_sizes['synthia'], mean=IMG_MEAN)
    else:
        raise ValueError('The source dataset mush be either gta5 or synthia')

    source_dataloader = data.DataLoader(source_dataset,
                                        batch_size=opt.batch_size,
                                        shuffle=True,
                                        num_workers=opt.num_workers,
                                        pin_memory=True,
                                        drop_last=True)
    return source_dataloader


def CreateTrgDataLoader(opt, Gs, set='train', get_image_label=False, generate_prev_image=False):
    target_dataset = cityscapesDataSet(opt.trg_data_dir,
                                       opt.trg_data_list,
                                       opt.scale_factor,
                                       opt.num_scales,
                                       opt.curr_scale,
                                       set,
                                       Gs,
                                       get_image_label=get_image_label,
                                       generate_prev_image=generate_prev_image)

    if set == 'train':
        target_dataloader = data.DataLoader(target_dataset,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            num_workers=opt.num_workers,
                                            pin_memory=True,
                                            drop_last=True)
    elif set == 'val' or set == 'test':
        target_dataloader = data.DataLoader(target_dataset,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            num_workers=opt.num_workers,
                                            pin_memory=True,
                                            drop_last=True)
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



def create_scale_dataloader(opt, Gst, Gts):
    opt.batch_size = opt.batch_size_list[opt.curr_scale]
    source_loader, target_loader = CreateSrcDataLoader(opt, Gst, get_image_label=True), CreateTrgDataLoader(opt, Gts, generate_prev_image=True)
    opt.epoch_size = np.maximum(len(source_loader.dataset), len(target_loader.dataset))
    source_loader.dataset.SetEpochSize(opt.epoch_size)
    target_loader.dataset.SetEpochSize(opt.epoch_size)
    opt.source_loaders.append(source_loader)
    opt.target_loaders.append(target_loader)
    # if opt.last_scale:
    opt.target_validation_loader = CreateTrgDataLoader(opt, None, set='val', get_image_label=True, generate_prev_image=False)
