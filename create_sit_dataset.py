SYNTHIA_SIZE=(760,1280)
def main(opt):
    model = torch.load(opt.trained_msc_model_path)
    for i in range(len(model)):
        model[i].eval()
        model[i] = torch.nn.DataParallel(model[i])
        model[i].to(opt.device)
    opt.curr_scale = len(model)-1
    source_train_loader = CreateSrcDataLoader(opt, get_filename=True)

    for source_scales, filenames in tqdm(source_train_loader):
        for i in range(len(source_scales)):
            source_scales[i] = source_scales[i].to(opt.device)
        sit_batch = concat_pyramid_eval(model, source_scales, opt)
        sit_batch = torch.nn.functional.interpolate(sit_batch, size=SYNTHIA_SIZE)
        for i, filename in enumerate(filenames):
            # output_sit_image = denorm(sit_batch[i]).detach().cpu().numpy()
            # output_sit_image = np.round(np.moveaxis(output_sit_image, 0, -1) * 255).astype(np.uint8)
            # im = Image.fromarray(output_sit_image)
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
    from PIL import Image
    from core.functions import norm_image
    from core.training import concat_pyramid_eval
    import os
    from torchvision.utils import save_image
    import numpy as np
    main(opt)

