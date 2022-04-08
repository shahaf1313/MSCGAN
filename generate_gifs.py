def main(opt):
    opt.curr_scale = opt.num_scales = 2
    gta_loader = CreateSrcDataLoader(opt)
    cs_loader = CreateTrgDataLoader(opt, get_scales_pyramid=True)
    opt.source='synthia'
    opt.src_data_dir=r'/home/shahaf/data/synthia'
    opt.src_data_list=r'./dataset/synthia_list/'
    syn_loader = CreateSrcDataLoader(opt)
    opt.continue_train_from_path = opt.gta_models_path
    Gst_gta, Gts_gta, _, _, _ = load_trained_networks(opt)
    opt.continue_train_from_path = opt.syn_models_path
    Gst_syn, Gts_syn, _, _, _ = load_trained_networks(opt)
    for iter, (gta_scales, syn_scales, cs_scales) in tqdm(enumerate(zip(gta_loader, syn_loader, cs_loader))):
        if iter >= 1000:
            break
        for i in range(len(gta_scales)):
            gta_scales[i] = gta_scales[i].to(opt.device)
            syn_scales[i] = syn_scales[i].to(opt.device)
            cs_scales[i] = cs_scales[i].to(opt.device)
        sit_gta = norm_image(concat_pyramid_eval(Gst_gta, gta_scales, opt)).detach().cpu()
        tis_gta = norm_image(concat_pyramid_eval(Gts_gta, cs_scales, opt)).detach().cpu()
        sit_syn = norm_image(concat_pyramid_eval(Gst_syn, syn_scales, opt)).detach().cpu()
        tis_syn = norm_image(concat_pyramid_eval(Gts_syn, cs_scales, opt)).detach().cpu()

        orig_path = ops.join(opt.gifs_out_path, "{:03d}_1.png".format(iter))
        save_image(make_grid(torch.cat((norm_image(gta_scales[-1]),
                                        norm_image(cs_scales[-1]),
                                        norm_image(syn_scales[-1]),
                                        norm_image(cs_scales[-1])), 0), nrow=2).detach().cpu(), orig_path)
        sit_path = ops.join(opt.gifs_out_path, "{:03d}_2.png".format(iter))
        save_image(make_grid(torch.cat((sit_gta,
                                        tis_gta,
                                        sit_syn,
                                        tis_syn), 0), nrow=2), sit_path)
    print('Finished Creating GIFs.')


if __name__ == "__main__":
    from core.config import get_arguments, post_config
    parser = get_arguments()
    opt = parser.parse_args()
    opt = post_config(opt)
    from tqdm import tqdm
    from core.training import load_trained_networks
    from data import CreateSrcDataLoader
    from data import CreateTrgDataLoader
    import torch
    from core.config import get_arguments, post_config
    from core.functions import norm_image
    from core.training import concat_pyramid_eval
    from torchvision.utils import make_grid
    import os.path as ops
    from torchvision.utils import save_image
    main(opt)

