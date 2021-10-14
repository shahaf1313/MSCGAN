def main(opt):
    if opt.train_mode == 'im2im':
        opt.image_full_size = IMG_CROP_SIZE_IM2IM
        opt.source_loaders, opt.target_loaders = im2im_train(opt)
    elif opt.train_mode == 'im_synth':
        opt.source_loaders = image_synthesis_train(opt)
    elif opt.train_mode == 'semseg':
        opt.image_full_size = IMG_CROP_SIZE_SEMSEG
        opt.source_loaders, opt.target_loaders = segmentation_train(opt)
        opt.target_validation_loader = CreateTrgDataLoader(opt, set='val', get_image_label=True, get_scales_pyramid=False)
    else:
        raise NotImplementedError('Train mode entered is not implemented. Try again with implemented mode. For help use --help.')

    print('########################### MSCGAN Configuration ##############################')
    for arg in vars(opt):
        print(arg + ': ' + str(getattr(opt, arg)))
    print('##################################################################################')
    train(opt)
    print('Finished Training.')

def segmentation_train(opt):
    opt.batch_size = 1
    opt.curr_scale = 0
    source_loader, target_loader = CreateSrcDataLoader(opt), CreateTrgDataLoader(opt)
    opt.epoch_size = np.maximum(len(target_loader.dataset), len(source_loader.dataset))

    source_loaders, target_loaders, num_epochs =[], [], []
    for i in range(opt.num_scales+1):
        opt.batch_size = opt.batch_size_list[i]
        opt.curr_scale = i
        source_loader, target_loader = CreateSrcDataLoader(opt, get_image_label=True), CreateTrgDataLoader(opt, get_scales_pyramid=True)
        source_loader.dataset.SetEpochSize(opt.epoch_size)
        target_loader.dataset.SetEpochSize(opt.epoch_size)
        source_loaders.append(source_loader)
        target_loaders.append(target_loader)

    return source_loaders, target_loaders

def im2im_train(opt):
    opt.batch_size = 1
    opt.curr_scale = 0
    domain_a_dataloader, domain_b_dataloader = CreateIm2ImDataLoader(opt)

    domain_a_dataloaders, domain_b_dataloaders = [], []
    for i in range(opt.num_scales+1):
        opt.batch_size = opt.batch_size_list[i]
        opt.curr_scale = i
        domain_a_dataloader, domain_b_dataloader = CreateIm2ImDataLoader(opt)
        domain_a_dataloaders.append(domain_a_dataloader)
        domain_b_dataloaders.append(domain_b_dataloader)
    return domain_a_dataloaders, domain_b_dataloaders

def image_synthesis_train(opt):
    opt.batch_size = 1
    opt.curr_scale = 0
    source_loader = CreateSrcDataLoader(opt)
    opt.epoch_size = len(source_loader.dataset)
    source_loaders, num_epochs =[], []
    for i in range(opt.num_scales+1):
        opt.batch_size = opt.batch_size_list[i]
        opt.curr_scale = i
        source_loader = CreateSrcDataLoader(opt, get_image_label=True)
        source_loader.dataset.SetEpochSize(opt.epoch_size)
        source_loaders.append(source_loader)
    return source_loaders

if __name__ == '__main__':
    from core.config import get_arguments, post_config
    parser = get_arguments()
    opt = parser.parse_args()
    opt = post_config(opt)
    from core.constants import IMG_CROP_SIZE_IM2IM, IMG_CROP_SIZE_SEMSEG
    from data import CreateSrcDataLoader
    from data import CreateTrgDataLoader
    from data import CreateIm2ImDataLoader
    from core.training import train
    import numpy as np
    main(opt)

