def main(opt):
    opt.curr_scale = opt.semseg_train_scale
    model = torch.load(opt.trained_msc_model_path)
    images_path = os.path.join(opt.sit_dataset_path, 'images')
    labels_path = os.path.join(opt.sit_dataset_path, 'labels')
    Path(images_path).mkdir(parents=True, exist_ok=True)
    Path(labels_path).mkdir(parents=True, exist_ok=True)
    for i in range(len(model)):
        model[i].eval()
        model[i] = torch.nn.DataParallel(model[i])
        model[i].to(opt.device)
    opt.curr_scale = len(model)-1
    source_train_loader = create_sit_dataloader(opt)

    for source_scales, label, filenames in tqdm(source_train_loader):
        for i in range(len(source_scales)):
            source_scales[i] = source_scales[i].to(opt.device)
        sit_batch = concat_pyramid_eval(model, source_scales, opt)
        if opt.save_sit_full_scale:
            sit_batch = torch.nn.functional.interpolate(sit_batch, size=SOURCE_IMAGE_SIZE[::-1], mode='bicubic')
            label = torch.nn.functional.interpolate(label.unsqueeze(1), size=SOURCE_IMAGE_SIZE[::-1], mode='nearest').squeeze(1)
        for i, filename in enumerate(filenames):
            # output_sit_image = denorm(sit_batch[i]).detach().cpu().numpy()
            # output_sit_image = np.round(np.moveaxis(output_sit_image, 0, -1) * 255).astype(np.uint8)
            # im = Image.fromarray(output_sit_image)
            save_image(norm_image(sit_batch[i]),    os.path.join(images_path, filename))
            save_segmentation_map(label[i], os.path.join(labels_path, filename))
            # save_image(torch.from_numpy(colorize_mask(label[i]))/255.,     os.path.join(labels_path, filename))
    print('Finished Creating SIT Dataset.')


if __name__ == "__main__":
    from core.config import get_arguments, post_config
    parser = get_arguments()
    opt = parser.parse_args()
    opt = post_config(opt)
    from tqdm import tqdm
    from data.generate_sit_dataset import create_sit_dataloader
    import torch
    from core.config import get_arguments, post_config
    from core.functions import norm_image, colorize_mask, save_segmentation_map
    from core.training import concat_pyramid_eval
    from core.constants import SOURCE_IMAGE_SIZE
    import os
    from pathlib import Path
    from torchvision.utils import save_image
    main(opt)

