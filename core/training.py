import core.functions as functions
import core.models as models
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from functools import partial
from core.constants import MAX_CHANNELS_PER_LAYER
import numpy as np
import time
from core.constants import H, W
from core.functions import imresize_torch
from torch.utils.tensorboard import SummaryWriter
import signal, os, sys
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def demo_basic(model, rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = model.to(rank)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

# (Shahaf) It might be that to use DDP correctly all you need to do is to run the following line:
# run_demo(demo_basic, num_of_gpus)

def train(opt):
    if opt.continue_train_from_path != '':
        Gst, Gts, Dst, Dts = load_trained_networks(opt)
        assert len(Gst) == len(Gts) == len(Dst) == len(Dts)
        scale_num = len(Gst) - 1 if opt.resume_to_epoch > 0 else len(Gst)
        resume_first_iteration = True if opt.resume_to_epoch > 0 else False
        opt.resume_to_epoch = opt.resume_to_epoch if opt.resume_to_epoch > 0 else 1
    else:
        scale_num = 0
        Gst, Gts = [], []
        Dst, Dts = [], []
        resume_first_iteration = False

    opt.tb = SummaryWriter(os.path.join(opt.tb_logs_dir, opt.folder_string))
    graceful_exit = GracefulExit(opt.out_, opt.debug_run)
    with graceful_exit:
        while scale_num < opt.num_scales + 1:
            opt.curr_scale = scale_num
            opt.base_channels = np.minimum(MAX_CHANNELS_PER_LAYER, int(opt.nfc * np.power(2, (np.floor((opt.curr_scale+1)/2)))))
            opt.outf = '%s/%d' % (opt.out_, scale_num)
            try:
                os.makedirs(opt.outf)
            except OSError:
                pass

            if resume_first_iteration:
                curr_nets = []
                for net_list in [Dst, Gst, Dts, Gts]:
                    curr_net = net_list[scale_num].train()
                    curr_nets.append(functions.reset_grads(curr_net, True))
                    net_list.remove(net_list[scale_num])
                Dst_curr, Gst_curr = curr_nets[0], curr_nets[1]
                Dts_curr, Gts_curr = curr_nets[2], curr_nets[3]
            else:
                Dst_curr, Gst_curr = init_models(opt)
                Dts_curr, Gts_curr = init_models(opt)

            #add networks to GracefulExit:
            graceful_exit.Gst.append(Gst_curr)
            graceful_exit.Gts.append(Gts_curr)
            graceful_exit.Dst.append(Dst_curr)
            graceful_exit.Dts.append(Dts_curr)

            # todo: implement DistributedDataParallel using the tutorial form pytorch's website!
            if len(opt.gpus) > 1: #Use data parallel and SyncBatchnorm
                # Dst_curr, Gst_curr = nn.parallel.DistributedDataParallel(nn.SyncBatchNorm.convert_sync_batchnorm(Dst_curr)), nn.parallel.DistributedDataParallel(nn.SyncBatchNorm.convert_sync_batchnorm(Gst_curr))
                # Dts_curr, Gts_curr = nn.parallel.DistributedDataParallel(nn.SyncBatchNorm.convert_sync_batchnorm(Dts_curr)), nn.parallel.DistributedDataParallel(nn.SyncBatchNorm.convert_sync_batchnorm(Gts_curr))
                Dst_curr, Gst_curr = nn.DataParallel(Dst_curr), nn.DataParallel(Gst_curr)
                Dts_curr, Gts_curr = nn.DataParallel(Dts_curr), nn.DataParallel(Gts_curr)


            scale_nets = train_single_scale(Dst_curr, Gst_curr, Dts_curr, Gts_curr, Gst, Gts, Dst, Dts,
                                            opt, resume=resume_first_iteration, epoch_num_to_resume=opt.resume_to_epoch)
            for net in scale_nets:
                net = functions.reset_grads(net, False)
                net.eval()
            Dst_curr, Gst_curr, Dts_curr, Gts_curr = scale_nets

            Gst.append(Gst_curr)
            Gts.append(Gts_curr)
            Dst.append(Dst_curr)
            Dts.append(Dts_curr)

            if not opt.debug_run:
                torch.save(Gst, '%s/Gst.pth' % (opt.out_))
                torch.save(Gts, '%s/Gts.pth' % (opt.out_))
                torch.save(Dst, '%s/Dst.pth' % (opt.out_))
                torch.save(Dts, '%s/Dts.pth' % (opt.out_))

            opt.prev_base_channels = opt.base_channels
            resume_first_iteration = False
            scale_num += 1

        opt.tb.close()
        return

def train_single_scale(netDst, netGst, netDts, netGts, Gst: list, Gts: list, Dst: list, Dts: list, opt, resume=False, epoch_num_to_resume=1):
        # setup optimizers and schedulers:
        optimizerDst = optim.Adam(netDst.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        optimizerGst = optim.Adam(netGst.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        optimizerDts = optim.Adam(netDts.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        optimizerGts = optim.Adam(netGts.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))


        batch_size = opt.source_loaders[opt.curr_scale].batch_size
        opt.save_pics_rate = int(opt.epoch_size * np.maximum(opt.Dsteps, opt.Gsteps) / batch_size / opt.pics_per_epoch)
        total_steps_per_scale = opt.epochs_per_scale * int(opt.epoch_size * np.maximum(opt.Dsteps, opt.Gsteps) / batch_size)
        start = time.time()
        discriminator_steps = 0
        generator_steps = 0
        epoch_num = epoch_num_to_resume if resume else 1
        steps = (epoch_num_to_resume-1)*int(opt.epoch_size * np.maximum(opt.Dsteps, opt.Gsteps) / batch_size) if resume else 0
        checkpoint_int = 1
        print_int = 0 if not resume else int(steps/opt.print_rate)
        save_pics_int = 0
        keep_training = True

        while keep_training:
            print('scale %d: starting epoch %d...' % (opt.curr_scale, epoch_num))
            epoch_num += 1
            for batch_num, (source_scales, target_scales) in enumerate(zip(opt.source_loaders[opt.curr_scale], opt.target_loaders[opt.curr_scale])):
                if steps > total_steps_per_scale:
                    keep_training = False
                    break
                if opt.debug_run and steps > opt.debug_stop_iteration:
                    keep_training = False
                    break

                # Move scale tensors to CUDA:
                for i in range(len(source_scales)):
                    source_scales[i] = source_scales[i].to(opt.device)
                    target_scales[i] = target_scales[i].to(opt.device)

                # Create pyramid concatenation:
                with torch.no_grad():
                    prev_sit = concat_pyramid(Gst, source_scales, opt)
                    prev_tis = concat_pyramid(Gts, target_scales, opt)

                ############################
                # (1) Update D networks: maximize D(x) + D(G(z))
                ###########################
                cyc_images = None
                optimizerDst.zero_grad()
                optimizerDts.zero_grad()
                for j in range(opt.Dsteps):
                    #train discriminator networks between domains (S->T, T->S)
                    # (netD, netG, prev, real_images, from_scales, opt)

                    #S -> T:
                    D_x, D_G_z, errD = adversarial_disciriminative_train(netDst, netGst, prev_sit, target_scales[opt.curr_scale], source_scales, opt)
                    opt.tb.add_scalar('Scale%d/ST/DiscriminatorLoss' % opt.curr_scale, errD.item()/opt.lambda_adversarial, discriminator_steps)
                    opt.tb.add_scalar('Scale%d/ST/DiscriminatorRealImagesLoss' % opt.curr_scale, D_x/opt.lambda_adversarial, discriminator_steps)
                    opt.tb.add_scalar('Scale%d/ST/DiscriminatorFakeImagesLoss' % opt.curr_scale, D_G_z/opt.lambda_adversarial, discriminator_steps)


                    # T -> S:
                    D_x, D_G_z, errD = adversarial_disciriminative_train(netDts, netGts, prev_tis, source_scales[opt.curr_scale], target_scales, opt)
                    opt.tb.add_scalar('Scale%d/TS/DiscriminatorLoss' % opt.curr_scale, errD.item()/opt.lambda_adversarial, discriminator_steps)
                    opt.tb.add_scalar('Scale%d/TS/DiscriminatorRealImagesLoss' % opt.curr_scale, D_x/opt.lambda_adversarial, discriminator_steps)
                    opt.tb.add_scalar('Scale%d/TS/DiscriminatorFakeImagesLoss' % opt.curr_scale, D_G_z/opt.lambda_adversarial, discriminator_steps)

                    discriminator_steps += 1

                optimizerDst.step()
                optimizerDts.step()


                ############################
                # (2) Update G networks: maximize D(G(z)), minimize Gst(Gts(s))-s and vice versa
                ###########################
                optimizerGst.zero_grad()
                optimizerGts.zero_grad()
                for j in range(opt.Gsteps):
                    # train generator networks between domains (S->T, T->S)

                    # S -> T:
                    errG = adversarial_generative_train(netGst, netDst, prev_sit, source_scales, opt)
                    opt.tb.add_scalar('Scale%d/ST/GeneratorAdversarialLoss' % opt.curr_scale, errG.item()/opt.lambda_adversarial, generator_steps)

                    # T -> S:
                    errG = adversarial_generative_train(netGts, netDts, prev_tis, target_scales, opt)
                    opt.tb.add_scalar('Scale%d/TS/GeneratorAdversarialLoss' % opt.curr_scale, errG.item()/opt.lambda_adversarial, generator_steps)

                    if opt.cyclic_loss_calc_rate > 0 and generator_steps % opt.cyclic_loss_calc_rate == 0:
                        # Cycle Consistency Loss:
                        # (netGx, x, x_scales, prev_x,
                        #  netGy, y, y_scales, prev_y,
                        #  m_noise, opt)
                        cyc_loss_x, cyc_loss_y, cyc_loss, cyc_images = cycle_consistency_loss(source_scales[-1], prev_sit, netGst, Gst,
                                                                                              target_scales[-1], prev_tis, netGts, Gts, opt)

                            # cycle_consistency_loss(netGst, source_scales[opt.curr_scale], source_scales, prev_sit,
                            #                                                                   netGts, target_scales[opt.curr_scale], target_scales, prev_tis,
                            #                                                                   opt)
                        opt.tb.add_scalar('Scale%d/Cyclic/LossSTS' % opt.curr_scale, cyc_loss_x.item()/opt.lambda_cyclic, int(generator_steps/opt.cyclic_loss_calc_rate))
                        opt.tb.add_scalar('Scale%d/Cyclic/LossTST' % opt.curr_scale, cyc_loss_y.item()/opt.lambda_cyclic, int(generator_steps/opt.cyclic_loss_calc_rate))
                        opt.tb.add_scalar('Scale%d/Cyclic/Loss' % opt.curr_scale, cyc_loss.item()/opt.lambda_cyclic, int(generator_steps/opt.cyclic_loss_calc_rate))

                    # if opt.identity_loss_calc_rate > 0 and generator_steps % opt.identity_loss_calc_rate == 0:
                    #     # Identity Loss:
                    #     # (netGx, x, x_scales, prev_x,
                    #     #  netGy, y, y_scales, prev_y,
                    #     #  m_noise, opt)
                    #     idt_loss_x, idt_loss_y, idt_loss, _ = identity_loss(netGst, source_scales[opt.curr_scale], source_scales, prev_sit,
                    #                                                         netGts, target_scales[opt.curr_scale], target_scales, prev_tis,
                    #                                                         opt)
                    #     opt.tb.add_scalar('Scale%d/Identity/LossTT' % opt.curr_scale, idt_loss_x.item(), int(generator_steps/opt.identity_loss_calc_rate))
                    #     opt.tb.add_scalar('Scale%d/Identity/LossSS' % opt.curr_scale, idt_loss_y.item(), int(generator_steps/opt.identity_loss_calc_rate))
                    #     opt.tb.add_scalar('Scale%d/Identity/Loss' % opt.curr_scale, idt_loss.item(), int(generator_steps/opt.identity_loss_calc_rate))

                    generator_steps += 1

                optimizerGst.step()
                optimizerGts.step()

                if int(steps/opt.print_rate) >= print_int or steps == 0:
                    elapsed = time.time() - start
                    print('scale %d:[step %d/%d] ; elapsed time = %.2f secs per step, %.2f secs per image' %
                          (opt.curr_scale, print_int*opt.print_rate, total_steps_per_scale, elapsed/opt.print_rate, elapsed/opt.print_rate/batch_size))
                    start = time.time()
                    print_int += 1

                if int(steps/opt.save_pics_rate) >= save_pics_int or steps == 0:
                    s     = norm_image(source_scales[opt.curr_scale][0])
                    t     = norm_image(target_scales[opt.curr_scale][0])
                    if cyc_images is None:
                        _, _, _, cyc_images = cycle_consistency_loss(source_scales[-1], prev_sit, netGst, Gst,
                                                                     target_scales[-1], prev_tis, netGts, Gts, opt)
                    sit   = norm_image(cyc_images[0][0])
                    sitis = norm_image(cyc_images[1][0])
                    tis   = norm_image(cyc_images[2][0])
                    tisit = norm_image(cyc_images[3][0])
                    opt.tb.add_image('Scale%d/source' % opt.curr_scale, s, save_pics_int*opt.save_pics_rate)
                    opt.tb.add_image('Scale%d/source_in_traget' % opt.curr_scale, sit, save_pics_int*opt.save_pics_rate)
                    opt.tb.add_image('Scale%d/source_in_traget_in_source' % opt.curr_scale, sitis, save_pics_int*opt.save_pics_rate)
                    opt.tb.add_image('Scale%d/target' % opt.curr_scale, t, save_pics_int*opt.save_pics_rate)
                    opt.tb.add_image('Scale%d/target_in_source' % opt.curr_scale, tis, save_pics_int*opt.save_pics_rate)
                    opt.tb.add_image('Scale%d/target_in_source_in_target' % opt.curr_scale, tisit, save_pics_int*opt.save_pics_rate)

                    save_pics_int += 1

                # Save network checkpoint every 2k steps:
                if steps > checkpoint_int * 2000:
                    print('scale %d: saving networks after %d steps...' % (opt.curr_scale, steps))
                    if (len(opt.gpus) > 1):
                        functions.save_networks(netDst.module, netGst.module, netDts.module, netGts.module, Gst, Gts, Dst, Dts, opt)
                    else:
                        functions.save_networks(netDst, netGst, netDts, netGts, Gst, Gts, Dst, Dts, opt)
                    checkpoint_int += 1
                steps = np.minimum(generator_steps, discriminator_steps)

        if (len(opt.gpus) > 1):
            functions.save_networks(netDst.module, netGst.module, netDts.module, netGts.module, Gst, Gts, Dst, Dts, opt)
            return netDst.module, netGst.module, netDts.module, netGts.module
        else:
            functions.save_networks(netDst, netGst, netDts, netGts, Gst, Gts, Dst, Dts, opt)
            return netDst, netGst, netDts, netGts

def adversarial_disciriminative_train(netD, netG, prev, real_images, from_scales, opt, real_segmaps=None):
    # train with real image
    # output = netD(real_images).to(opt.device)
    output = netD(real_images)
    errD_real = -1 * opt.lambda_adversarial * output.mean()
    errD_real.backward(retain_graph=True)
    D_x = errD_real.item()

    # train with fake
    curr = from_scales[opt.curr_scale]
    with torch.no_grad():
        fake_images = netG(curr, prev, real_segmaps) if opt.use_conditinal_generator else netG(curr, prev)
    output = netD(fake_images.detach())
    errD_fake =  opt.lambda_adversarial * output.mean()
    errD_fake.backward(retain_graph=True)
    D_G_z = errD_fake.item()

    gradient_penalty =  opt.lambda_adversarial * functions.calc_gradient_penalty(netD, real_images, fake_images, opt.lambda_grad, opt.device)
    gradient_penalty.backward()

    errD = errD_real + errD_fake + gradient_penalty

    return D_x, D_G_z, errD


def adversarial_generative_train(netG, netD, prev, from_scales, opt, seg_maps=None):

    ##todo: I added!
    # train with fake
    curr = from_scales[opt.curr_scale]
    fake = netG(curr, prev, seg_maps) if opt.use_conditinal_generator else netG(curr, prev)
    ##end

    output = netD(fake)
    errG = -1 * opt.lambda_adversarial * output.mean()
    errG.backward()
    return errG

# def cycle_consistency_loss(currGst, source_batch, source_scales, prev_sit, currGts, target_batch, target_scales, prev_tis, opt):
def cycle_consistency_loss(source_batch, prev_sit, currGst, Gst_pyramid,
                           target_batch, prev_tis, currGts, Gts_pyramid, opt,
                           segmap_source=None, segmap_target=None, net_seg_target=None, net_seg_source=None):
    criterion_sts = nn.L1Loss()
    criterion_tst = nn.L1Loss()

    #source in target:
    sit_image = currGst(source_batch, prev_sit, segmap_source) if opt.use_conditinal_generator else currGst(source_batch, prev_sit)
    # todo: make sure that segmentation network is passed correctly:
    # created_segmap_target = net_seg_target(sit_image)
    created_segmap_target = None
    with torch.no_grad():
        generated_pyramid_sit = functions.GeneratePyramid(sit_image, opt.num_scales, opt.curr_scale, opt.scale_factor, opt.image_full_size)
        prev_sit_generated = concat_pyramid(Gts_pyramid, generated_pyramid_sit, opt)
    #source in target in source:
    sitis_image = currGts(sit_image, prev_sit_generated, created_segmap_target) if opt.use_conditinal_generator else currGts(sit_image, prev_sit_generated)
    loss_sts = opt.lambda_cyclic * criterion_sts(sitis_image, source_batch)
    loss_sts.backward()


    #source in target:
    tis_image = currGts(target_batch, prev_tis)
    with torch.no_grad():
        generated_pyramid_tis = functions.GeneratePyramid(tis_image, opt.num_scales, opt.curr_scale, opt.scale_factor, opt.image_full_size)
        prev_tis_generated = concat_pyramid(Gst_pyramid, generated_pyramid_tis, opt)
    #source in target in source:
    tisit_image = currGst(tis_image, prev_tis_generated)
    loss_tst = opt.lambda_cyclic * criterion_tst(tisit_image, target_batch)
    loss_tst.backward()

    loss = loss_sts + loss_tst

    return loss_sts, loss_tst, loss, (sit_image, sitis_image, tis_image, tisit_image)

# def identity_loss(netGx, x, x_scales, prev_x, netGy, y, y_scales, prev_y, opt):
#     criterion_x = nn.L1Loss()
#     criterion_y = nn.L1Loss()
#
#     #Gy(y):
#     curr_y = y_scales[opt.curr_scale]
#     Gy_y = netGx(curr_y, prev_y)
#     loss_y = criterion_y(Gy_y, y)
#     loss_y.backward()
#
#     #Gx(x):
#     curr_x = x_scales[opt.curr_scale]
#     Gx_x = netGy(curr_x, prev_x)
#     loss_x = criterion_x(Gx_x, x)
#     loss_x.backward()
#
#     loss = loss_x + loss_y
#     return loss_x, loss_y, loss, (Gx_x, Gy_y)

def concat_pyramid(Gs, sources, opt):
    if len(Gs) == 0:
        return torch.zeros_like(sources[0])

    G_z = sources[0]
    count = 0
    for G, source_curr, source_next in zip(Gs, sources, sources[1:]):
        G_z = G_z[:, :, 0:source_curr.shape[2], 0:source_curr.shape[3]]
        G_z = G(source_curr, G_z.detach())
        # G_z = imresize(G_z, 1 / opt.scale_factor, opt)
        G_z = imresize_torch(G_z, 1 / opt.scale_factor)
        G_z = G_z[:, :, 0:source_next.shape[2], 0:source_next.shape[3]]
        count += 1
    return G_z

def init_models(opt):
    use_four_level_net = np.power(opt.scale_factor, opt.num_scales - opt.curr_scale) * np.minimum(H,W) / 16 > opt.ker_size
    # generator initialization:
    if opt.use_unet_generator:
        # use_four_level_unet = np.power(opt.scale_factor, opt.num_scales - opt.curr_scale) * np.minimum(H,W) / 16 > opt.ker_size
        if use_four_level_net:
            print('Generating 4 layers UNET model')
            netG = models.UNetGeneratorFourLayers(opt).to(opt.device)
        else:
            print('Generating 2 layers UNET model')
            netG = models.UNetGeneratorTwoLayers(opt).to(opt.device)
    elif opt.use_conditinal_generator:
        netG = models.LabelConditionedGenerator(opt).to(opt.device)
    else:
        netG = models.ConvGenerator(opt).to(opt.device)
    netG.apply(models.weights_init)
    print(netG)


    # discriminator initialization:
    if opt.use_downscale_discriminator:
        netD = models.WDiscriminatorDownscale(opt, use_four_level_net).to(opt.device)
    else:
        netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    print(netD)

    return netD, netG

def load_trained_networks(opt):
    Gst = torch.load(os.path.join(opt.continue_train_from_path, 'Gst.pth'))
    Gts = torch.load(os.path.join(opt.continue_train_from_path, 'Gts.pth'))
    Dst = torch.load(os.path.join(opt.continue_train_from_path, 'Dst.pth'))
    Dts = torch.load(os.path.join(opt.continue_train_from_path, 'Dts.pth'))
    for m1, m2, m3, m4 in zip(Gts, Gts, Dst, Dts):
        m1.eval().to(opt.device)
        m2.eval().to(opt.device)
        m3.eval().to(opt.device)
        m4.eval().to(opt.device)
    return Gst, Gts, Dst, Dts

def norm_image(im):
    out = (im + 1)/2
    assert torch.max(out) <= 1 and torch.min(out) >= 0
    return out

def handler(nets_dict, signum, frame):
    if not nets_dict['debug']:
        torch.save(nets_dict['Gst'], '%s/Gst.pth' % (nets_dict['out_path']))
        torch.save(nets_dict['Gts'], '%s/Gts.pth' % (nets_dict['out_path']))
        torch.save(nets_dict['Dst'], '%s/Dst.pth' % (nets_dict['out_path']))
        torch.save(nets_dict['Dts'], '%s/Dts.pth' % (nets_dict['out_path']))
        print('\nExited unexpectedly. Pyramids has been saved successfully.')
        print('Length of pyramid list:', len(nets_dict['Gst']))
        print('Exiting. Bye!')
    sys.exit(0)

class GracefulExit:
    def __init__(self, path_to_save_networks, is_debug):
        self.path_to_save_networks = path_to_save_networks
        self.Gst = []
        self.Gts = []
        self.Dst = []
        self.Dts = []
        self.net_dict = {'Gts' : self.Gts,
                         'Gst' : self.Gst,
                         'Dts' : self.Dts,
                         'Dst' : self.Dst,
                         'out_path' : path_to_save_networks,
                         'debug' : is_debug}

    def __enter__(self):
        signal.signal(signal.SIGTERM,  partial(handler, self.net_dict))
        signal.signal(signal.SIGINT,   partial(handler, self.net_dict))

    def __exit__(self, type, value, traceback):
        pass


