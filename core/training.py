import core.models as models
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from data.labels_info import trainId2label
from core.constants import MAX_CHANNELS_PER_LAYER, NUM_CLASSES, IGNORE_LABEL, BEST_MIOU
from core.sync_batchnorm import convert_model
from semseg_models import CreateSemsegModel
import numpy as np
import time
from core.constants import H, W
from core.functions import imresize_torch, colorize_mask, reset_grads, save_networks, calc_gradient_penalty, GeneratePyramid, one_hot_encoder, compute_iou_torch, compute_cm_batch_torch, runningScore, encode_semseg_out
from torch.utils.tensorboard import SummaryWriter
import os
# from functools import partial
# import signal, os, sys


def train(opt):
    opt.best_miou = BEST_MIOU
    semseg_cs = None
    if opt.continue_train_from_path != '':
        Gst, Gts, Dst, Dts = load_trained_networks(opt)
        # todo: add loading semseg network
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
    # graceful_exit = GracefulExit(opt.out_, opt.debug_run)
    # with graceful_exit:
    while scale_num < opt.num_scales + 1:
        opt.curr_scale = scale_num
        opt.last_scale = opt.curr_scale == opt.num_scales
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
                curr_nets.append(reset_grads(curr_net, True))
                net_list.remove(net_list[scale_num])
            Dst_curr, Gst_curr = curr_nets[0], curr_nets[1]
            Dts_curr, Gts_curr = curr_nets[2], curr_nets[3]
        else:
            Dst_curr, Gst_curr = init_models(opt)
            Dts_curr, Gts_curr = init_models(opt)
            if opt.last_scale: #Last scale, add semseg network:
                semseg_cs, _ = CreateSemsegModel(opt)
            else:
                semseg_cs = None

        # #add networks to GracefulExit:
        # graceful_exit.Gst.append(Gst_curr)
        # graceful_exit.Gts.append(Gts_curr)
        # graceful_exit.Dst.append(Dst_curr)
        # graceful_exit.Dts.append(Dts_curr)
        # if opt.last_scale: #Last scale, save semseg network:
        #     graceful_exit.semseg_cs = semseg_cs

        # todo: implement DistributedDataParallel using the tutorial form pytorch's website!
        if len(opt.gpus) > 1: #Use data parallel and SyncBatchNorm
            Dst_curr, Gst_curr = convert_model(nn.DataParallel(Dst_curr)).to(opt.device), convert_model(nn.DataParallel(Gst_curr)).to(opt.device)
            Dts_curr, Gts_curr = convert_model(nn.DataParallel(Dts_curr)).to(opt.device), convert_model(nn.DataParallel(Gts_curr)).to(opt.device)
            if opt.last_scale: #Last scale, convert also the semseg network to DP+SBN:
                semseg_cs = convert_model(nn.DataParallel(semseg_cs)).to(opt.device)

            # Dst_curr, Gst_curr = convert_model(nn.DataParallel(Dst_curr, device_ids=opt.gpus)).to(opt.device), convert_model(nn.DataParallel(Gst_curr, device_ids=opt.gpus)).to(opt.device)
            # Dts_curr, Gts_curr = convert_model(nn.DataParallel(Dts_curr, device_ids=opt.gpus)).to(opt.device), convert_model(nn.DataParallel(Gts_curr, device_ids=opt.gpus)).to(opt.device)
            # if opt.last_scale: #Last scale, convert also the semseg network to DP+SBN:
            #     semseg_cs = convert_model(nn.DataParallel(semseg_cs)).to(opt.device)

            # Dst_curr, Gst_curr = nn.DataParallel(Dst_curr, device_ids=opt.gpus), nn.DataParallel(Gst_curr, device_ids=opt.gpus)
            # Dts_curr, Gts_curr = nn.DataParallel(Dts_curr, device_ids=opt.gpus), nn.DataParallel(Gts_curr, device_ids=opt.gpus)
            # if opt.last_scale: #Last scale, convert also the semseg network to DP+SBN:
            #     semseg_cs = nn.DataParallel(semseg_cs, device_ids=opt.gpus)

        print(Dst_curr), print(Gst_curr), print(Dts_curr), print(Gts_curr)
        if opt.last_scale: #Last scale, print semseg network:
            print(semseg_cs)
        scale_nets = train_single_scale(Dst_curr, Gst_curr, Dts_curr, Gts_curr, Gst, Gts, Dst, Dts,
                                        opt, resume=resume_first_iteration, epoch_num_to_resume=opt.resume_to_epoch,
                                        semseg_cs=semseg_cs)
        for net in scale_nets:
            net = reset_grads(net, False)
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

def train_single_scale(netDst, netGst, netDts, netGts, Gst: list, Gts: list, Dst: list, Dts: list,
                       opt, resume=False, epoch_num_to_resume=1, semseg_cs=None):
        optimizerDst = optim.Adam(netDst.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        optimizerGst = optim.Adam(netGst.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        optimizerDts = optim.Adam(netDts.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        optimizerGts = optim.Adam(netGts.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        if opt.last_scale:
            optimizerSemseg = optim.SGD(semseg_cs.module.optim_parameters(opt) if (len(opt.gpus) > 1) else semseg_cs.optim_parameters(opt), lr=opt.lr_semseg, momentum=opt.momentum, weight_decay=opt.weight_decay)
            optimizerSemsegGen = optim.SGD(semseg_cs.module.optim_parameters(opt) if (len(opt.gpus) > 1) else semseg_cs.optim_parameters(opt), lr=opt.lr_semseg/10, momentum=opt.momentum, weight_decay=opt.weight_decay)
        else:
            optimizerSemseg = None

        batch_size = opt.source_loaders[opt.curr_scale].batch_size
        opt.save_pics_rate = np.maximum(2, int(opt.epoch_size * np.minimum(opt.Dsteps, opt.Gsteps) / batch_size / opt.pics_per_epoch))
        total_steps_per_scale = opt.epochs_per_scale * int(opt.epoch_size * np.minimum(opt.Dsteps, opt.Gsteps) / batch_size)
        start = time.time()
        discriminator_steps = 0
        generator_steps = 0
        semseg_steps = 0
        epoch_num = epoch_num_to_resume if resume else 1
        steps = (epoch_num_to_resume-1)*int(opt.epoch_size * np.maximum(opt.Dsteps, opt.Gsteps) / batch_size) if resume else 0
        checkpoint_int = 1
        print_int = 0 if not resume else int(steps/opt.print_rate)
        save_pics_int = 0
        keep_training = True

        while keep_training:
            print('scale %d: starting epoch [%d/%d]' % (opt.curr_scale, epoch_num, opt.epochs_per_scale))
            opt.warmup = epoch_num <= opt.warmup_epochs
            if opt.last_scale and opt.warmup:
                print('scale %d: warmup epoch [%d/%d]' % (opt.curr_scale, epoch_num, opt.warmup_epochs))

            for batch_num, ((source_scales, source_labels), target_scales) in enumerate(zip(opt.source_loaders[opt.curr_scale], opt.target_loaders[opt.curr_scale])):
                if steps > total_steps_per_scale:
                    keep_training = False
                    break
                if opt.debug_run and steps > epoch_num*opt.debug_stop_iteration:
                    if opt.debug_stop_epoch <= epoch_num:
                        keep_training = False
                    break

                netDst.train()
                netGst.train()
                netDts.train()
                netGts.train()
                if opt.last_scale:
                    semseg_cs.train()

                # Move scale and label tensors to CUDA:
                # source_labels = source_labels.to(opt.device)
                for i in range(len(source_scales)):
                    source_scales[i] = source_scales[i].to(opt.device)
                    source_labels[i] = source_labels[i].to(opt.device)
                    target_scales[i] = target_scales[i].to(opt.device)

                # Resize scale and label tensors if needed:
                if opt.use_half_image_size:
                    for i in range(len(source_scales)):
                        source_labels[i] = (nn.functional.interpolate(source_labels[i].unsqueeze(1), scale_factor=[0.5,0.5], mode='nearest')).squeeze()
                        source_scales[i] = torch.clamp(nn.functional.interpolate(source_scales[i], scale_factor=[0.5,0.5], mode='bicubic'), -1, 1)
                        target_scales[i] = torch.clamp(nn.functional.interpolate(target_scales[i], scale_factor=[0.5,0.5], mode='bicubic'), -1, 1)

                # Create segmentation maps if needed:
                source_segmaps = [one_hot_encoder(source_label) for source_label in source_labels] #if opt.last_scale and not opt.warmup else None
                # source_segmap = one_hot_encoder(source_label) if opt.last_scale else None
                target_segmap = encode_semseg_out(semseg_cs(source_scales[-1]), opt.ignore_threshold) if opt.last_scale and not opt.warmup else None

                # # Create pyramid concatenation:
                # with torch.no_grad():
                #     prev_sit = concat_pyramid(Gst, source_scales, opt, source_segmap)
                #     prev_tis = concat_pyramid(Gts, target_scales, opt, target_segmap)


                ############################
                # (1) Update D networks: maximize D(x) + D(G(z))
                ###########################
                cyc_images, semseg_labels, semseg_softs = None, None, None
                for j in range(opt.Dsteps):
                    #train discriminator networks between domains (S->T, T->S)
                    optimizerDst.zero_grad()
                    optimizerDts.zero_grad()
                    # if opt.last_scale and not opt.warmup:
                    #     optimizerSemseg.zero_grad()

                    #S -> T:
                    D_x, D_G_z, errD = adversarial_disciriminative_train(netDst, netGst, Gst, target_scales[opt.curr_scale], source_scales, opt, real_segmap=source_segmaps)
                    opt.tb.add_scalar('Scale%d/ST/DiscriminatorLoss' % opt.curr_scale, errD.item()/opt.lambda_adversarial, discriminator_steps)
                    opt.tb.add_scalar('Scale%d/ST/DiscriminatorRealImagesLoss' % opt.curr_scale, D_x/opt.lambda_adversarial, discriminator_steps)
                    opt.tb.add_scalar('Scale%d/ST/DiscriminatorFakeImagesLoss' % opt.curr_scale, D_G_z/opt.lambda_adversarial, discriminator_steps)


                    # T -> S:
                    D_x, D_G_z, errD = adversarial_disciriminative_train(netDts, netGts, Gts, source_scales[opt.curr_scale], target_scales, opt, real_segmap=target_segmap)
                    opt.tb.add_scalar('Scale%d/TS/DiscriminatorLoss' % opt.curr_scale, errD.item()/opt.lambda_adversarial, discriminator_steps)
                    opt.tb.add_scalar('Scale%d/TS/DiscriminatorRealImagesLoss' % opt.curr_scale, D_x/opt.lambda_adversarial, discriminator_steps)
                    opt.tb.add_scalar('Scale%d/TS/DiscriminatorFakeImagesLoss' % opt.curr_scale, D_G_z/opt.lambda_adversarial, discriminator_steps)

                    optimizerDst.step()
                    optimizerDts.step()
                    # if opt.last_scale and not opt.warmup:
                    #     optimizerSemseg.step()

                    discriminator_steps += 1


                ############################
                # (2) Update G networks: maximize D(G(z)), minimize Gst(Gts(s))-s and vice versa
                ###########################

                for j in range(opt.Gsteps):
                    # train generator networks between domains (S->T, T->S)
                    optimizerGst.zero_grad()
                    optimizerGts.zero_grad()
                    if opt.last_scale and not opt.warmup:
                        optimizerSemsegGen.zero_grad()

                    # S -> T:
                    errG = adversarial_generative_train(netGst, netDst, Gst, source_scales, opt, real_segmap=source_segmaps)
                    opt.tb.add_scalar('Scale%d/ST/GeneratorAdversarialLoss' % opt.curr_scale, errG.item()/opt.lambda_adversarial, generator_steps)

                    # T -> S:
                    target_segmap = encode_semseg_out(semseg_cs(source_scales[-1]), opt.ignore_threshold) if opt.last_scale and not opt.warmup else None
                    errG = adversarial_generative_train(netGts, netDts, Gts, target_scales, opt, real_segmap=target_segmap)
                    opt.tb.add_scalar('Scale%d/TS/GeneratorAdversarialLoss' % opt.curr_scale, errG.item()/opt.lambda_adversarial, generator_steps)

                    if opt.cyclic_loss_calc_rate > 0 and generator_steps % opt.cyclic_loss_calc_rate == 0:
                        # Cycle Consistency Loss:
                        # (netGx, x, x_scales, prev_x,
                        #  netGy, y, y_scales, prev_y,
                        #  m_noise, opt)
                        cyc_loss_labels, cyc_loss_x, cyc_loss_y, cyc_loss, cyc_images = cycle_consistency_loss( source_scales, netGst, Gst,
                                                                                                                target_scales, netGts, Gts, opt,
                                                                                                                source_segmaps, semseg_cs)
                        opt.tb.add_scalar('Scale%d/Cyclic/LossSTS' % opt.curr_scale, cyc_loss_x.item()/opt.lambda_cyclic, int(generator_steps/opt.cyclic_loss_calc_rate))
                        opt.tb.add_scalar('Scale%d/Cyclic/LossTST' % opt.curr_scale, cyc_loss_y.item()/opt.lambda_cyclic, int(generator_steps/opt.cyclic_loss_calc_rate))
                        opt.tb.add_scalar('Scale%d/Cyclic/Loss' % opt.curr_scale, cyc_loss.item()/opt.lambda_cyclic, int(generator_steps/opt.cyclic_loss_calc_rate))
                        if cyc_loss_labels != None:
                            opt.tb.add_scalar('Scale%d/Cyclic/LossLabels' % opt.curr_scale, cyc_loss_labels.item()/opt.lambda_cyclic, int(generator_steps/opt.cyclic_loss_calc_rate))

                    optimizerGst.step()
                    optimizerGts.step()
                    if opt.last_scale and not opt.warmup:
                        optimizerSemsegGen.step()

                    generator_steps += 1



                ############################
                # (3) Update semantic segmentation network: minimize CE Loss on converted images (Use GT of source domain):
                ###########################
                if opt.last_scale:
                    optimizerSemseg.zero_grad()
                    optimizerGst.zero_grad()
                    semseg_softs, semseg_labels, semseg_loss = semantic_segmentation_loss(source_scales, Gst, netGst, semseg_cs, source_labels, opt)
                    opt.tb.add_scalar('Semseg/SemsegCsLoss', semseg_loss.item(), semseg_steps)
                    optimizerSemseg.step()
                    optimizerGst.step()
                    semseg_steps += 1

                if int(steps/opt.print_rate) >= print_int or steps == 0:
                    elapsed = time.time() - start
                    print('scale %d:[step %d/%d] ; elapsed time = %.2f secs per step, %.2f secs per image' %
                          (opt.curr_scale, print_int*opt.print_rate, total_steps_per_scale, elapsed/opt.print_rate, elapsed/opt.print_rate/batch_size))
                    start = time.time()
                    print_int += 1

                if int(steps/opt.save_pics_rate) >= save_pics_int or steps == 0:
                    s       = norm_image(source_scales[opt.curr_scale][0])
                    t       = norm_image(target_scales[opt.curr_scale][0])
                    s_lbl   = colorize_mask(source_labels[-1][0])
                    if cyc_images is None:
                        _, _, _, _, cyc_images = cycle_consistency_loss(source_scales, netGst, Gst,
                                                                        target_scales, netGts, Gts, opt,
                                                                        source_segmaps, semseg_cs)
                    sit   = norm_image(cyc_images[0][0])
                    sitis = norm_image(cyc_images[1][0])
                    tis   = norm_image(cyc_images[2][0])
                    tisit = norm_image(cyc_images[3][0])
                    opt.tb.add_image('Scale%d/source' % opt.curr_scale, s, save_pics_int*opt.save_pics_rate)
                    opt.tb.add_image('Scale%d/source_label' % opt.curr_scale, s_lbl, save_pics_int*opt.save_pics_rate)
                    opt.tb.add_image('Scale%d/source_in_traget' % opt.curr_scale, sit, save_pics_int*opt.save_pics_rate)
                    opt.tb.add_image('Scale%d/source_in_traget_in_source' % opt.curr_scale, sitis, save_pics_int*opt.save_pics_rate)
                    opt.tb.add_image('Scale%d/target' % opt.curr_scale, t, save_pics_int*opt.save_pics_rate)
                    opt.tb.add_image('Scale%d/target_in_source' % opt.curr_scale, tis, save_pics_int*opt.save_pics_rate)
                    opt.tb.add_image('Scale%d/target_in_source_in_target' % opt.curr_scale, tisit, save_pics_int*opt.save_pics_rate)

                    if opt.last_scale:
                        sit_label = colorize_mask(semseg_labels[0])
                        softs_max = torch.nn.functional.softmax(semseg_softs, dim=1)
                        hist_values = softs_max.max(dim=1)[0][0]
                        s_label = colorize_mask(source_labels[0])
                        opt.tb.add_image('Scale%d/source_in_target_label' % opt.curr_scale, sit_label, save_pics_int*opt.save_pics_rate)
                        opt.tb.add_image('Scale%d/source_in_target_values' % opt.curr_scale, hist_values, save_pics_int*opt.save_pics_rate, dataformats='HW')
                        opt.tb.add_histogram('Scale%d/source_in_target_histogram' % opt.curr_scale, hist_values, save_pics_int*opt.save_pics_rate, bins='auto')
                        opt.tb.add_image('Scale%d/source_label' % opt.curr_scale, s_label, save_pics_int*opt.save_pics_rate)
                        if not opt.warmup:
                            target_segmap = one_hot_encoder(encode_semseg_out(semseg_cs(target_scales[-1]), opt.ignore_threshold), per_class_encode=False, generated_label=True)
                            t_label = colorize_mask(target_segmap[0])
                            opt.tb.add_image('Scale%d/target_label' % opt.curr_scale, t_label, save_pics_int*opt.save_pics_rate)

                    save_pics_int += 1

                # Save network checkpoint every 1k steps:
                if steps > checkpoint_int * 1000:
                    print('scale %d: saving networks after %d steps...' % (opt.curr_scale, steps))
                    save_networks(opt.outf, netDst, netGst, netDts, netGts, Gst, Gts, Dst, Dts, opt, semseg_cs)
                    checkpoint_int += 1

                steps = np.minimum(generator_steps, discriminator_steps)

            ############################
            # (5) Validate performance after each epoch if we are at the last scale:
            ############################
            if opt.last_scale:
                iou, miou, cm = calculte_validation_accuracy(semseg_cs, opt.target_validation_loader, epoch_num, opt)
                export_epoch_accuracy(opt, iou, miou, cm, epoch_num)
                if miou > opt.best_miou:
                    opt.best_miou = miou
                    save_networks(os.path.join(opt.checkpoints_dir, '%.2f_mIoU_model' % (miou)), netDst, netGst, netDts, netGts, Gst, Gts, Dst, Dts, opt, semseg_cs)

            epoch_num += 1

        save_networks(opt.outf, netDst, netGst, netDts, netGts, Gst, Gts, Dst, Dts, opt, semseg_cs)
        return (netDst, netGst, netDts, netGts) if len(opt.gpus) == 1 else (netDst.module, netGst.module, netDts.module, netGts.module)

def adversarial_disciriminative_train(netD, netG, Gs, real_images, from_scales, opt, real_segmap=None):
    # train with real image
    # output = netD(real_images).to(opt.device)
    output = netD(real_images)
    errD_real = -1 * opt.lambda_adversarial * output.mean()
    errD_real.backward(retain_graph=True)
    D_x = errD_real.item()

    # train with fake
    with torch.no_grad():
        curr = from_scales[opt.curr_scale]
        prev = concat_pyramid(Gs, from_scales, opt, real_segmap)
        fake_images = netG(curr, prev, real_segmap[-1] if type(real_segmap)==list else real_segmap)
    output = netD(fake_images.detach())
    errD_fake = opt.lambda_adversarial * output.mean()
    errD_fake.backward(retain_graph=True)
    D_G_z = errD_fake.item()

    gradient_penalty = opt.lambda_adversarial * calc_gradient_penalty(netD, real_images, fake_images, opt.lambda_grad, opt.device)
    gradient_penalty.backward()

    errD = errD_real + errD_fake + gradient_penalty

    return D_x, D_G_z, errD


def adversarial_generative_train(netG, netD, Gs, from_scales, opt, real_segmap=None):

    # train with fake
    curr = from_scales[opt.curr_scale]
    prev = concat_pyramid(Gs, from_scales, opt, real_segmap)
    fake = netG(curr, prev, real_segmap[-1] if type(real_segmap)==list else real_segmap)
    output = netD(fake)
    errG = -1 * opt.lambda_adversarial * output.mean()
    errG.backward()
    return errG

def cycle_consistency_loss(source_scales, currGst, Gst_pyramid,
                           target_scales, currGts, Gts_pyramid, opt,
                           segmaps_source=None, semseg_net=None):
    criterion_sts = nn.L1Loss()
    criterion_tst = nn.L1Loss()
    criterion_labels = nn.L1Loss()
    source_batch  = source_scales[-1]
    target_batch =  target_scales[-1]
    loss_labels = None

    #source in target:
    with torch.no_grad():
        prev_sit = concat_pyramid(Gst_pyramid, source_scales, opt, segmaps_source)
    sit_image = currGst(source_batch, prev_sit, segmaps_source[-1] if type(segmaps_source)==list else segmaps_source)
    with torch.no_grad():
        generated_pyramid_sit = GeneratePyramid(sit_image, opt.num_scales, opt.curr_scale, opt.scale_factor, opt.image_full_size)
        prev_sit_generated = concat_pyramid(Gts_pyramid, generated_pyramid_sit, opt, segmaps_source)
    #source in target in source:
    sitis_image = currGts(sit_image, prev_sit_generated, segmaps_source[-1] if type(segmaps_source)==list else segmaps_source)
    loss_sts = opt.lambda_cyclic * criterion_sts(sitis_image, source_batch)
    loss_sts.backward()

    #traget in source:
    segmap_target = encode_semseg_out(semseg_net(target_batch), opt.ignore_threshold) if opt.last_scale and not opt.warmup else None
    with torch.no_grad():
        prev_tis = concat_pyramid(Gts_pyramid, target_scales, opt, segmap_target)
    tis_image = currGts(target_batch, prev_tis, segmap_target)
    with torch.no_grad():
        generated_pyramid_tis = GeneratePyramid(tis_image, opt.num_scales, opt.curr_scale, opt.scale_factor, opt.image_full_size)
        prev_tis_generated = concat_pyramid(Gst_pyramid, generated_pyramid_tis, opt, segmap_target)
    #target in source in target:
    tisit_image = currGst(tis_image, prev_tis_generated, segmap_target)
    loss_tst = opt.lambda_cyclic * criterion_tst(tisit_image, target_batch)
    loss_tst.backward(retain_graph = opt.last_scale and not opt.warmup)
    # Label cyclic loss:
    if opt.last_scale and not opt.warmup:
        tisit_segmap = encode_semseg_out(semseg_net(tisit_image), opt.ignore_threshold)
        loss_labels = opt.lambda_cyclic * criterion_labels(tisit_segmap[:,:NUM_CLASSES,:,:], segmap_target[:,:NUM_CLASSES,:,:])
        loss_labels.backward()

    loss = loss_sts + loss_tst

    return loss_labels, loss_sts, loss_tst, loss, (sit_image, sitis_image, tis_image, tisit_image)

def semantic_segmentation_loss(input_pyramid, Gs, currG, semseg_net, input_label, opt):
    prev_converted_image = concat_pyramid(Gs, input_pyramid, opt, input_label)
    converted_image = currG(input_pyramid[-1], prev_converted_image, one_hot_encoder(input_label) if not opt.warmup else None)
    output_softs, semseg_loss = semseg_net(converted_image, input_label)
    semseg_loss = semseg_loss.mean()
    output_label = output_softs.argmax(1)
    semseg_loss.backward()
    return output_softs, output_label, semseg_loss

def concat_pyramid(Gs, sources, opt, labels=None):
    if len(Gs) == 0:
        return torch.zeros_like(sources[0])
    with torch.no_grad:
        G_z = sources[0]
        labels = [None] * len(sources) if labels==None else labels
        for G, source_curr, source_next, label_curr in zip(Gs, sources, sources[1:], labels):
            G_z = G_z[:, :, 0:source_curr.shape[2], 0:source_curr.shape[3]]
            G_z = G(source_curr, G_z.detach(), label_curr)
            # G_z = imresize(G_z, 1 / opt.scale_factor, opt)
            G_z = imresize_torch(G_z, 1 / opt.scale_factor, mode='bicubic')
            G_z = G_z[:, :, 0:source_next.shape[2], 0:source_next.shape[3]]
    return G_z.detach()

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
    else: #Conditial Generator(!):
        netG = models.LabelConditionedGenerator(opt).to(opt.device)
        # if opt.curr_scale == opt.num_scales: # last scale, initialize label conditioning:
        #     netG = models.LabelConditionedGenerator(opt).to(opt.device)
        # else: # not last scale, use regular generator:
        #     netG = models.ConvGenerator(opt).to(opt.device)
    netG.apply(models.weights_init)


    # discriminator initialization:
    if opt.use_downscale_discriminator:
        netD = models.WDiscriminatorDownscale(opt, use_four_level_net).to(opt.device)
    else:
        netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)

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

def norm_image(im, norm_type='tanh_norm'):
    if norm_type=='tanh_norm':
        out = (im + 1)/2
    elif norm_type=='general_norm':
        out = (im-im.min())
        out = out/out.max()
    else:
        raise NotImplemented()
    assert torch.max(out) <= 1 and torch.min(out) >= 0
    return out

def export_epoch_accuracy(opt, iou, miou, cm, epoch):
    for i in range(NUM_CLASSES):
        opt.tb.add_scalar('Semseg/%s class accuracy' % (trainId2label[i].name), iou[i], epoch)
    opt.tb.add_scalar('Semseg/Accuracy History [mIoU]', miou, epoch)
    print('================Model Acuuracy Summery================')
    for i in range(NUM_CLASSES):
        print('%s class accuracy: = %.2f%%' % (trainId2label[i].name, iou[i]*100))
    print('Average accuracy of test set on target domain: mIoU = %2f%%' % (miou*100))
    print('======================================================')

def calculte_validation_accuracy(semseg_net, target_val_loader, epoch_num, opt):
    semseg_net.eval()
    with torch.no_grad():
        running_metrics_val = runningScore(NUM_CLASSES)
        cm = torch.zeros((NUM_CLASSES, NUM_CLASSES)).cuda()
        for val_batch_num, (target_images, target_labels) in enumerate(target_val_loader):
            if opt.debug_run and val_batch_num > opt.debug_stop_iteration:
                break
            target_images = target_images.to(opt.device)
            target_labels = target_labels.to(opt.device)
            with torch.no_grad():
                pred_softs = semseg_net(target_images)
                pred_labels = torch.argmax(pred_softs, dim=1)
                cm += compute_cm_batch_torch(pred_labels, target_labels, IGNORE_LABEL, NUM_CLASSES)
                running_metrics_val.update(target_labels.cpu().numpy(), pred_labels.cpu().numpy())
                if val_batch_num == 0:
                    t        = norm_image(target_images[0])
                    t_lbl    = colorize_mask(target_labels[0])
                    pred_lbl = colorize_mask(pred_labels[0])
                    opt.tb.add_image('Semseg/Validtaion/target', t, epoch_num)
                    opt.tb.add_image('Semseg/Validtaion/target_label', t_lbl, epoch_num)
                    opt.tb.add_image('Semseg/Validtaion/prediction_label', pred_lbl, epoch_num)
        iou, miou = compute_iou_torch(cm)

        # proda's calc:
        score, class_iou = running_metrics_val.get_scores()
        for k, v in score.items():
            print(k, v)

        for k, v in class_iou.items():
            print(k, v)

        running_metrics_val.reset()
    return iou, miou, cm


# def handler(nets_dict, signum, frame):
#     if not nets_dict['debug']:
#         torch.save(nets_dict['Gst'], '%s/Gst.pth' % (nets_dict['out_path']))
#         torch.save(nets_dict['Gts'], '%s/Gts.pth' % (nets_dict['out_path']))
#         torch.save(nets_dict['Dst'], '%s/Dst.pth' % (nets_dict['out_path']))
#         torch.save(nets_dict['Dts'], '%s/Dts.pth' % (nets_dict['out_path']))
#         if nets_dict['semseg_cs'] != None:
#             torch.save(nets_dict['semseg_cs'], '%s/semseg_cs.pth' % (nets_dict['out_path']))
#         print('\nExited unexpectedly. Pyramids & semseg has been saved successfully.')
#         print('Length of pyramid list:', len(nets_dict['Gst']))
#         print('Exiting. Bye!')
#     sys.exit(0)
#
# class GracefulExit:
#     def __init__(self, path_to_save_networks, is_debug):
#         self.path_to_save_networks = path_to_save_networks
#         self.Gst = []
#         self.Gts = []
#         self.Dst = []
#         self.Dts = []
#         self.semseg_cs = None
#         self.net_dict = {'Gts' : self.Gts,
#                          'Gst' : self.Gst,
#                          'Dts' : self.Dts,
#                          'Dst' : self.Dst,
#                          'semseg_cs' : self.semseg_cs,
#                          'out_path' : path_to_save_networks,
#                          'debug' : is_debug}
#
#     def __enter__(self):
#         signal.signal(signal.SIGTERM,  partial(handler, self.net_dict))
#         signal.signal(signal.SIGINT,   partial(handler, self.net_dict))
#
#     def __exit__(self, type, value, traceback):
#         pass
#
#
