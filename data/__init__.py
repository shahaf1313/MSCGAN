from torch.utils import data
from data.gta5_dataset import GTA5DataSet
from data.cityscapes_dataset import cityscapesDataSet
from data.synthia_dataset import SYNDataSet


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
                                            shuffle=False,
                                            num_workers=opt.num_workers,
                                            pin_memory=True)
    else:
        raise Exception("Argument set has not entered properly. Options are train or eval.")

    return target_dataloader
