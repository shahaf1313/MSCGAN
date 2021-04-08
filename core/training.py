import core.functions as functions
import core.models as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import numpy as np
import time
from core.constants import H, W
from core.functions import imresize_torch
import datetime
from torch.utils.tensorboard import SummaryWriter


def train(opt):
    if opt.continue_train_from_path != '':
        Gst, Gts, Dst, Dts = load_trained_networks(opt)
        assert len(Gst) == len(Gts) == len(Dst) == len(Dts)
        scale_num = len(Gst) - 1 if opt.train_last_scale_more else 0
        resume_first_iteration = True
    else:
        scale_num = 0
        Gst, Gts = [], []
        Dst, Dts = [], []
        resume_first_iteration = False

    opt.tb = SummaryWriter(os.path.join(opt.tb_logs_dir, '%sGPU%d/' % (datetime.datetime.now().strftime('%d-%m-%Y::%H:%M:%S'), opt.gpus[0])))

    while scale_num < opt.stop_scale + 1:
        opt.curr_scale = scale_num
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

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
            resume_first_iteration = False
        else:
            Dst_curr, Gst_curr = init_models(opt)
            Dts_curr, Gts_curr = init_models(opt)

        if len(opt.gpus) > 1:
            # Dst_curr, Gst_curr = nn.DataParallel(Dst_curr, device_ids=opt.gpus), nn.DataParallel(Gst_curr, device_ids=opt.gpus)
            # Dts_curr, Gts_curr = nn.DataParallel(Dts_curr, device_ids=opt.gpus), nn.DataParallel(Gts_curr, device_ids=opt.gpus) [i for i in range(len(opt.gpus))]

            Dst_curr, Gst_curr = nn.DataParallel(Dst_curr), nn.DataParallel(Gst_curr)
            Dts_curr, Gts_curr = nn.DataParallel(Dts_curr), nn.DataParallel(Gts_curr)


        scale_nets = train_single_scale(Dst_curr, Gst_curr, Dts_curr, Gts_curr, Gst, Gts, Dst, Dts, opt)
        for net in scale_nets:
            net = functions.reset_grads(net, False)
            net.eval()
        Dst_curr, Gst_curr, Dts_curr, Gts_curr = scale_nets

        Gst.append(Gst_curr)
        Gts.append(Gts_curr)
        Dst.append(Dst_curr)
        Dts.append(Dts_curr)

        torch.save(Gst, '%s/Gst.pth' % (opt.out_))
        torch.save(Gts, '%s/Gts.pth' % (opt.out_))
        torch.save(Dst, '%s/Dst.pth' % (opt.out_))
        torch.save(Dts, '%s/Dts.pth' % (opt.out_))

        scale_num += 1
        nfc_prev = opt.nfc
        del Dst_curr, Gst_curr, Dts_curr, Gts_curr
    opt.tb.close()
    return

def train_single_scale(netDst, netGst, netDts, netGts, Gst: list, Gts: list, Dst: list, Dts: list, opt):
        # setup optimizers and schedulers:
        optimizerDst = optim.Adam(netDst.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        optimizerGst = optim.Adam(netGst.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        optimizerDts = optim.Adam(netDts.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        optimizerGts = optim.Adam(netGts.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))

        discriminator_steps = 0
        generator_steps = 0
        steps = 0
        checkpoint_int = 1
        print_int = 0
        save_pics_int = 0
        epoch_num = 1
        batch_size = opt.source_loaders[opt.curr_scale].batch_size
        PICS_PER_EPOCH = 10
        opt.save_pics_rate = int(opt.epoch_size * np.maximum(opt.Dsteps, opt.Gsteps) / batch_size / PICS_PER_EPOCH)
        total_steps_per_scale = opt.epochs_per_scale * int(opt.epoch_size * np.maximum(opt.Dsteps, opt.Gsteps) / batch_size)
        start = time.time()
        keep_training = True

        while keep_training:
            print('scale %d: starting epoch %d...' % (opt.curr_scale, epoch_num))
            epoch_num += 1
            for batch_num, (source_scales, target_scales) in enumerate(zip(opt.source_loaders[opt.curr_scale], opt.target_loaders[opt.curr_scale])):
                if steps > total_steps_per_scale:
                    keep_training = False
                    break
                if opt.debug_run and steps > 20:
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
                        cyc_loss_x, cyc_loss_y, cyc_loss, cyc_images = cycle_consistency_loss(netGst, source_scales[opt.curr_scale], source_scales, prev_sit,
                                                                                              netGts, target_scales[opt.curr_scale], target_scales, prev_tis,
                                                                                              opt)
                        opt.tb.add_scalar('Scale%d/Cyclic/LossSTS' % opt.curr_scale, cyc_loss_x.item()/opt.lambda_cyclic, int(generator_steps/opt.cyclic_loss_calc_rate))
                        opt.tb.add_scalar('Scale%d/Cyclic/LossTST' % opt.curr_scale, cyc_loss_y.item()/opt.lambda_cyclic, int(generator_steps/opt.cyclic_loss_calc_rate))
                        opt.tb.add_scalar('Scale%d/Cyclic/Loss' % opt.curr_scale, cyc_loss.item()/opt.lambda_cyclic, int(generator_steps/opt.cyclic_loss_calc_rate))

                    if opt.identity_loss_calc_rate > 0 and generator_steps % opt.identity_loss_calc_rate == 0:
                        # Identity Loss:
                        # (netGx, x, x_scales, prev_x,
                        #  netGy, y, y_scales, prev_y,
                        #  m_noise, opt)
                        idt_loss_x, idt_loss_y, idt_loss, _ = identity_loss(netGst, source_scales[opt.curr_scale], source_scales, prev_sit,
                                                                            netGts, target_scales[opt.curr_scale], target_scales, prev_tis,
                                                                            opt)
                        opt.tb.add_scalar('Scale%d/Identity/LossTT' % opt.curr_scale, idt_loss_x.item(), int(generator_steps/opt.identity_loss_calc_rate))
                        opt.tb.add_scalar('Scale%d/Identity/LossSS' % opt.curr_scale, idt_loss_y.item(), int(generator_steps/opt.identity_loss_calc_rate))
                        opt.tb.add_scalar('Scale%d/Identity/Loss' % opt.curr_scale, idt_loss.item(), int(generator_steps/opt.identity_loss_calc_rate))

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
                        _, _, _, cyc_images = cycle_consistency_loss(netGst, source_scales[opt.curr_scale], source_scales, prev_sit,
                                                                     netGts, target_scales[opt.curr_scale], target_scales, prev_tis,
                                                                     opt)
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

                # Save network checkpoint every 20k steps:
                if steps > checkpoint_int * 20000:
                    print('scale %d: saving networks after %d steps...' % (opt.curr_scale, steps))
                    if (len(opt.gpus) > 1):
                        functions.save_networks(netDst.module, netGst.module, netDts.module, netGts.module, opt)
                    else:
                        functions.save_networks(netDst, netGst, netDts, netGts, opt)
                    checkpoint_int += 1
                steps = np.minimum(generator_steps, discriminator_steps)

        if (len(opt.gpus) > 1):
            functions.save_networks(netDst.module, netGst.module, netDts.module, netGts.module, opt)
            return netDst.module, netGst.module, netDts.module, netGts.module
        else:
            functions.save_networks(netDst, netGst, netDts, netGts, opt)
            return netDst, netGst, netDts, netGts

def adversarial_disciriminative_train(netD, netG, prev, real_images, from_scales, opt):
    # train with real image
    output = netD(real_images).to(opt.device)
    errD_real = -1 * opt.lambda_adversarial * output.mean()
    errD_real.backward(retain_graph=True)
    D_x = errD_real.item()

    # train with fake
    curr = from_scales[opt.curr_scale]
    with torch.no_grad():
        fake_images = netG(curr, prev)
    output = netD(fake_images.detach())
    errD_fake =  opt.lambda_adversarial * output.mean()
    errD_fake.backward(retain_graph=True)
    D_G_z = errD_fake.item()

    gradient_penalty =  opt.lambda_adversarial * functions.calc_gradient_penalty(netD, real_images, fake_images, opt.lambda_grad, opt.device)
    gradient_penalty.backward()

    errD = errD_real + errD_fake + gradient_penalty

    return D_x, D_G_z, errD


def adversarial_generative_train(netG, netD, prev, from_scales, opt):

    ##todo: I added!
    # train with fake
    curr = from_scales[opt.curr_scale]
    fake = netG(curr, prev)
    ##end

    output = netD(fake)
    errG = -1 * opt.lambda_adversarial * output.mean()
    errG.backward()

    # reconstruction loss (as appers in singan, doesn't work for my settings):
    # loss = nn.MSELoss()
    # prev_rec = concat_pyramid(Gs, source_scales, target_scales, source_scale, m_noise, m_image, opt)
    # rec_loss = loss(prev_rec.detach(), curr_scale)
    # rec_loss.backward(retain_graph=True)
    # rec_loss = rec_loss.detach()

    return errG

def cycle_consistency_loss(netGx, x, x_scales, prev_x, netGy, y, y_scales, prev_y, opt):
    criterion_xx = nn.L1Loss()
    criterion_yy = nn.L1Loss()

    #Gy(x):
    curr_x = x_scales[opt.curr_scale]
    Gy_x = netGx(curr_x, prev_x)

    #Gx(Gy(x)):
    curr_yx = Gy_x
    Gx_Gy_x = netGy(curr_yx, prev_y)
    loss_x = opt.lambda_cyclic * criterion_xx(Gx_Gy_x, x)
    loss_x.backward()

    # Gx(y):
    curr_y = y_scales[opt.curr_scale]
    Gx_y = netGy(curr_y, prev_y)

    # Gy(Gx(y)):
    curr_xy = Gx_y
    Gy_Gx_y = netGx(curr_xy, prev_x)
    loss_y = opt.lambda_cyclic * criterion_yy(Gy_Gx_y, y)
    loss_y.backward()

    loss = loss_x + loss_y

    return loss_x, loss_y, loss, (Gy_x, Gx_Gy_x, Gx_y, Gy_Gx_y)

def identity_loss(netGx, x, x_scales, prev_x, netGy, y, y_scales, prev_y, opt):
    criterion_x = nn.L1Loss()
    criterion_y = nn.L1Loss()

    #Gy(y):
    curr_y = y_scales[opt.curr_scale]
    Gy_y = netGx(curr_y, prev_y)
    loss_y = criterion_y(Gy_y, y)
    loss_y.backward()

    #Gx(x):
    curr_x = x_scales[opt.curr_scale]
    Gx_x = netGy(curr_x, prev_x)
    loss_x = criterion_x(Gx_x, x)
    loss_x.backward()

    loss = loss_x + loss_y
    return loss_x, loss_y, loss, (Gx_x, Gy_y)

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
    # generator initialization:
    if opt.use_unet_generator:
        use_four_level_unet = np.power(opt.scale_factor, opt.num_scales - opt.curr_scale) * np.minimum(H,W) / 16 > opt.ker_size
        if use_four_level_unet:
            print('Generating 4 layers UNET model')
            netG = models.UNetGeneratorFourLayers(opt).to(opt.device)
        else:
            print('Generating 2 layers UNET model')
            netG = models.UNetGeneratorTwoLayers(opt).to(opt.device)
    else:
        netG = models.ConvGenerator(opt).to(opt.device)
    netG.apply(models.weights_init)
    print(netG)


    # discriminator initialization:
    if opt.use_downscale_discriminator:
        netD = models.WDiscriminatorDownscale(opt).to(opt.device)
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

