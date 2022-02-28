def main(opt):
    model = torch.load(opt.trained_msc_model_path)
    for i in range(len(model)):
        model[i].eval()
        model[i] = torch.nn.DataParallel(model[i])
        model[i].to(opt.device)
    opt.num_scales = opt.curr_scale = len(model)-1
    resize = IMG_SIZE_SOURCE[opt.source][::-1]
    source_train_loader = CreateSrcDataLoader(opt, get_filename=True, get_original_image=True)
    if opt.skip_created_files:
        already_created = next(os.walk(opt.sit_dataset_path))[2]
        for f in already_created:
            if f in source_train_loader.dataset.img_ids:
                source_train_loader.dataset.img_ids.remove(f)
    print('Number of images to convert: %d' % len(source_train_loader.dataset.img_ids))
    for source_scales, filenames in tqdm(source_train_loader):
        for i in range(len(source_scales)):
            source_scales[i] = source_scales[i].to(opt.device)
        sit_batch = concat_pyramid_eval(model, source_scales, opt)
        sit_batch = torch.nn.functional.interpolate(sit_batch, size=resize, mode='bicubic')
        for i, filename in enumerate(filenames):
            save_image(norm_image(sit_batch[i]), os.path.join(opt.sit_dataset_path, filename))
    print('Finished Creating SIT Dataset.')


if __name__ == "__main__":
    from core.config import get_arguments, post_config
    parser = get_arguments()
    opt = parser.parse_args()
    opt = post_config(opt)
    from tqdm import tqdm
    from data import CreateSrcDataLoader
    import torch
    from core.config import get_arguments, post_config
    from core.functions import norm_image
    from core.training import concat_pyramid_eval
    import os
    from torchvision.utils import save_image
    from core.constants import IMG_SIZE_SOURCE
    main(opt)

