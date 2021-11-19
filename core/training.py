import core.models as models
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from data.labels_info import trainId2label
from core.constants import NUM_CLASSES, IGNORE_LABEL, BEST_MIOU
from core.sync_batchnorm import convert_model
from semseg_models import CreateSemsegModel
from core.style_transfer import StyleTransferLoss
import numpy as np
from core.models import VGGLoss
import time
import copy
from core.constants import H, W
from core.functions import imresize_torch, colorize_mask, reset_grads, save_networks, calc_gradient_penalty, GeneratePyramid, one_hot_encoder, compute_iou_torch, \
    compute_cm_batch_torch, runningScore, norm_image, encode_semseg_out
from torch.utils.tensorboard import SummaryWriter
import os


def train(opt):
    opt.best_miou = BEST_MIOU
    semseg_cs = None
    if opt.continue_train_from_path != '':
        Gst, Gts, Dst, Dts, semseg_cs, Dcs, Dgta = load_trained_networks(opt)
        # todo: add loading semseg network
        assert len(Gst) == len(Gts) == len(Dst) == len(Dts) #== len(Dcs) == len(Dgta)
        scale_num = len(Gst) - 1 if opt.resume_to_epoch > 0 else len(Gst)
        resume_first_iteration = True if opt.resume_to_epoch > 0 else False
        opt.resume_to_epoch = opt.resume_to_epoch if opt.resume_to_epoch > 0 else 1
        #todo: delete after first full training!!!
        if Dcs == None and Dgta == None:
            opt.base_channels = opt.base_channels_list[0]
            _, _, dcs = init_models(opt)
            _, _, dgta = init_models(opt)
            Dcs = [None, None, dcs]
            Dgta = [None, None, dgta]
    else:
        scale_num = 0
        Gst, Gts = [], []
        Dst, Dts = [], []
        Dcs, Dgta = [], []
        resume_first_iteration = False

    opt.tb = SummaryWriter(os.path.join(opt.tb_logs_dir, opt.folder_string))
    while scale_num < opt.num_scales + 1:
        opt.curr_scale = scale_num
        opt.last_scale = opt.curr_scale == opt.num_scales
        opt.base_channels = opt.base_channels_list[0]  if len(opt.base_channels_list) == 1 else opt.base_channels_list[opt.curr_scale]
        opt.outf = '%s/%d' % (opt.out_folder, scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
            pass

        if resume_first_iteration:
            curr_nets = []
            for net_list in [Dst, Gst, Dts, Gts, Dcs, Dgta]:
                curr_net = net_list[scale_num].train()
                curr_nets.append(reset_grads(curr_net, True))
                net_list.remove(net_list[scale_num])
            Dst_curr, Gst_curr = curr_nets[0], curr_nets[1]
            Dts_curr, Gts_curr = curr_nets[2], curr_nets[3]
            Dcs_curr, Dgta_curr = curr_nets[4], curr_nets[5]
        else:
            Dst_curr, Gst_curr, Dcs_curr = init_models(opt)
            Dts_curr, Gts_curr, Dgta_curr = init_models(opt)
            semseg_cs, _ = CreateSemsegModel(opt)

        if len(opt.gpus) > 1:
            # todo: This code section initializes a SyncBN model. It utilizes more memory but gives better results:
            # Use data parallel and SyncBatchNorm
            # Dst_curr, Gst_curr = convert_model(nn.DataParallel(Dst_curr)).to(opt.device), convert_model(nn.DataParallel(Gst_curr)).to(opt.device)
            # Dts_curr, Gts_curr = convert_model(nn.DataParallel(Dts_curr)).to(opt.device), convert_model(nn.DataParallel(Gts_curr)).to(opt.device)
            # if opt.last_scale:  # Last scale, convert also the semseg network to DP+SBN:
            #     semseg_cs = convert_model(nn.DataParallel(semseg_cs)).to(opt.device)

            # todo: for now, decided to use GroupNorm instead of BN, so not using sync BN!
            Dst_curr, Gst_curr = nn.DataParallel(Dst_curr), nn.DataParallel(Gst_curr)
            Dts_curr, Gts_curr = nn.DataParallel(Dts_curr), nn.DataParallel(Gts_curr)
            Dcs_curr, Dgta_curr = nn.DataParallel(Dcs_curr), nn.DataParallel(Dgta_curr)
            if opt.last_scale: #Last scale, convert also the semseg network to DP+SBN:
                semseg_cs = nn.DataParallel(semseg_cs)

        print(Dst_curr), print(Gst_curr), print(Dts_curr), print(Gts_curr), print(Dcs_curr), print(Dgta_curr)
        if opt.last_scale:  # Last scale, print semseg network:
            print(semseg_cs)
        scale_nets = train_single_scale(Dst_curr, Gst_curr, Dts_curr, Gts_curr, Dcs_curr, Dgta_curr, Gst, Gts, Dst, Dts, Dcs, Dgta,
                                        opt, resume=resume_first_iteration, epoch_num_to_resume=opt.resume_to_epoch,
                                        semseg_cs=semseg_cs)
        for net in scale_nets:
            net = reset_grads(net, False)
            net.eval()
        Dst_curr, Gst_curr, Dts_curr, Gts_curr, Dcs_curr, Dgta_curr = scale_nets

        Gst.append(Gst_curr)
        Gts.append(Gts_curr)
        Dst.append(Dst_curr)
        Dts.append(Dts_curr)
        Dcs.append(Dcs_curr)
        Dgta.append(Dgta_curr)

        if not opt.debug_run:
            torch.save(Gst, '%s/Gst.pth' % (opt.out_folder))
            torch.save(Gts, '%s/Gts.pth' % (opt.out_folder))
            torch.save(Dst, '%s/Dst.pth' % (opt.out_folder))
            torch.save(Dts, '%s/Dts.pth' % (opt.out_folder))
            torch.save(Dcs, '%s/Dcs.pth' % (opt.out_folder))
            torch.save(Dgta, '%s/Dgta.pth' % (opt.out_folder))

        opt.prev_base_channels = opt.base_channels
        resume_first_iteration = False
        scale_num += 1

    opt.tb.close()
    return


def train_single_scale(netDst, netGst, netDts, netGts, netDcs, netDgta,
                       Gst: list, Gts: list, Dst: list, Dts: list, Dcs: list, Dgta: list,
                       opt, resume=False, epoch_num_to_resume=1, semseg_cs=None):
    gst_params = []
    for net in Gst:
        gst_params += list(net.parameters())
    gst_params += list(netGst.parameters())

    gts_params = []
    for net in Gts:
        gts_params += list(net.parameters())
    gts_params += list(netGts.parameters())

    optimizerDst = optim.Adam(netDst.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerGst = optim.Adam(gst_params, lr=opt.lr_g, betas=(opt.beta1, 0.999))
    optimizerDts = optim.Adam(netDts.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerGts = optim.Adam(gts_params, lr=opt.lr_g, betas=(opt.beta1, 0.999))
    optimizerDcs = optim.Adam(netDcs.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerDgta = optim.Adam(netDgta.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    if opt.last_scale:
        optimizerSemsegCS = optim.SGD(semseg_cs.module.optim_parameters(opt) if (len(opt.gpus) > 1) else semseg_cs.optim_parameters(opt), lr=opt.lr_semseg, momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)
        optimizerSemsegGen = optim.SGD(semseg_cs.module.optim_parameters(opt) if (len(opt.gpus) > 1) else semseg_cs.optim_parameters(opt), lr=opt.lr_semseg / 4,
                                       momentum=opt.momentum, weight_decay=opt.weight_decay)
        semseg_gta = torch.load(opt.pretrained_deeplabv2_on_gta_miou_70)
        # optimizerSemsegGTA = optim.SGD(semseg_gta.module.optim_parameters(opt) if (len(opt.gpus) > 1) else semseg_gta.optim_parameters(opt), lr=opt.lr_semseg, momentum=opt.momentum,
        #                               weight_decay=opt.weight_decay)
    else:
        optimizerSemsegCS, optimizerSemsegGen, semseg_gta, optimizerSemsegGTA = None, None, None, None

    batch_size = opt.source_loaders[opt.curr_scale].batch_size
    opt.save_pics_rate = set_pics_save_rate(opt.pics_per_epoch, batch_size, opt)
    opt.style_transfer_loss = StyleTransferLoss(opt)
    total_steps_per_scale = opt.epochs_per_scale * int(opt.epoch_size * np.minimum(opt.Dsteps, opt.Gsteps) / batch_size)
    start = time.time()
    epoch_num = epoch_num_to_resume if resume else 1
    steps = (epoch_num_to_resume - 1) * int(opt.epoch_size * np.maximum(opt.Dsteps, opt.Gsteps) / batch_size) if resume else 0
    discriminator_steps = 0
    generator_steps = 0
    semseg_steps    = 0
    checkpoint_int  = 1 if not resume else int(steps / opt.save_checkpoint_rate) + 1
    print_int       = 0 if not resume else int(steps / opt.print_rate)
    save_pics_int   = 0 if not resume else int(steps / opt.save_pics_rate)
    opt.debug_init_steps = 0 if not resume else steps
    keep_training = True

    while keep_training:
        print('scale %d: starting epoch [%d/%d]' % (opt.curr_scale, epoch_num, opt.epochs_per_scale))
        opt.warmup = epoch_num <= opt.warmup_epochs
        if opt.last_scale and opt.warmup:
            print('scale %d: warmup epoch [%d/%d]' % (opt.curr_scale, epoch_num, opt.warmup_epochs))

        for batch_num, ((source_scales, source_label), target_scales) in enumerate(zip(opt.source_loaders[opt.curr_scale], opt.target_loaders[opt.curr_scale])):
            if steps > total_steps_per_scale:
                keep_training = False
                break
            if opt.debug_run and steps > opt.debug_init_steps + (epoch_num-resume*(epoch_num_to_resume-1)) * opt.debug_stop_iteration:
                if opt.debug_stop_epoch <= epoch_num:
                    keep_training = False
                break

            netDst.train()
            netGst.train()
            netDts.train()
            netGts.train()
            netDcs.train()
            netDgta.train()
            for scale_Gst, scale_Gts in zip(Gst, Gts):
                scale_Gst.train()
                scale_Gts.train()
            if opt.last_scale:
                semseg_cs.train()

            # Move scale and label tensors to CUDA:
            source_label = source_label.to(opt.device)
            for i in range(len(source_scales)):
                source_scales[i] = source_scales[i].to(opt.device)
                target_scales[i] = target_scales[i].to(opt.device)

            # Resize scale and label tensors if needed:
            if opt.use_half_image_size:
                source_label = (nn.functional.interpolate(source_label.unsqueeze(1), scale_factor=[0.5, 0.5], mode='nearest')).squeeze(1)
                for i in range(len(source_scales)):
                    source_scales[i] = torch.clamp(nn.functional.interpolate(source_scales[i], scale_factor=[0.5, 0.5], mode='bicubic'), -1, 1)
                    target_scales[i] = torch.clamp(nn.functional.interpolate(target_scales[i], scale_factor=[0.5, 0.5], mode='bicubic'), -1, 1)

            # Create segmentation maps if needed:
            source_label = source_label
            source_segmap = one_hot_encoder(source_label)
            # target_softs = (semseg_cs(target_scales[-1])).detach() if opt.last_scale and not opt.warmup else None
            # target_segmap = encode_semseg_out(target_softs, opt.ignore_threshold) if opt.last_scale and not opt.warmup else None


            ############################
            # (1) Update D networks: maximize D(x) + D(G(z))
            ###########################
            cyc_images, semseg_labels, semseg_softs, trusted_labels_target = None, None, None, None
            for j in range(opt.Dsteps):
                # train discriminator networks between domains (S->T, T->S)
                optimizerDst.zero_grad()
                optimizerDts.zero_grad()

                # S -> T:
                discriminator_losses = adversarial_discriminative_train(netDst, netGst, Gst, target_scales[opt.curr_scale], source_scales, opt, real_segmap=source_segmap)
                for k,v in discriminator_losses.items():
                    opt.tb.add_scalar('Scale%d/ST/Discriminator/%s' % (opt.curr_scale, k), v, discriminator_steps)

                # T -> S:
                discriminator_losses = adversarial_discriminative_train(netDts, netGts, Gts, source_scales[opt.curr_scale], target_scales, opt, real_segmap=None)
                for k,v in discriminator_losses.items():
                    opt.tb.add_scalar('Scale%d/TS/Discriminator/%s' % (opt.curr_scale, k), v, discriminator_steps)

                optimizerDst.step()
                optimizerDts.step()

                discriminator_steps += 1

            ############################
            # (2) Update G networks: maximize D(G(z)), minimize Gst(Gts(s))-s and vice versa
            ###########################

            # Extract features from source and target images:
            # content_features_source, style_features_source = opt.style_transfer_loss.extract_features(source_scales[-1])
            # content_features_target, style_features_target = opt.style_transfer_loss.extract_features(target_scales[-1])
            # content_features_source, style_features_source = [c.detach() for c in content_features_source], [s.detach() for s in style_features_source]
            # content_features_target, style_features_target = [c.detach() for c in content_features_target], [s.detach() for s in style_features_target]
            for j in range(opt.Gsteps):
                # train generator networks between domains (S->T, T->S)
                optimizerGst.zero_grad()
                optimizerGts.zero_grad()
                if opt.last_scale and not opt.warmup:
                    optimizerSemsegGen.zero_grad()

                # S -> T:
                generator_losses = adversarial_generative_train(netGst, netDst, Gst, source_scales, opt,
                                                      # source_content_features=content_features_source,
                                                      # target_style_features=style_features_target,
                                                      source_segmap=source_segmap)
                for k,v in generator_losses.items():
                    opt.tb.add_scalar('Scale%d/ST/Generator/%s' % (opt.curr_scale,k), v, generator_steps)

                # T -> S:
                generator_losses = adversarial_generative_train(netGts, netDts, Gts, target_scales, opt,
                                                      # source_content_features=content_features_target,
                                                      # target_style_features=style_features_source,
                                                      source_segmap=None,
                                                      retain_graph=False)
                for k,v in generator_losses.items():
                    opt.tb.add_scalar('Scale%d/TS/Generator/%s' % (opt.curr_scale,k), v, generator_steps)

                cyc_losses, cyc_images = cycle_consistency_loss(source_scales, netGst, Gst,
                                                                target_scales, netGts, Gts, opt,
                                                                source_label, source_segmap,
                                                                semseg_cs, semseg_gta)
                for k,v in cyc_losses.items():
                    opt.tb.add_scalar('Scale%d/Cyclic/%s' % (opt.curr_scale,k), v, generator_steps)


                optimizerGst.step()
                optimizerGts.step()
                if opt.last_scale and not opt.warmup:
                    optimizerSemsegGen.step()

                generator_steps += 1

            ############################
            # (3) Update semantic segmentation network: minimize CE Loss on converted images (Use GT of source domain):
            ###########################
            if opt.last_scale:
                optimizerDcs.zero_grad()
                optimizerDgta.zero_grad()       
                # S -> T:
                discriminator_losses = adversarial_discriminative_train(netDcs, netGst, Gst, target_scales[opt.curr_scale], source_scales, opt, semseg=semseg_cs)
                for k,v in discriminator_losses.items():
                    opt.tb.add_scalar('Scale%d/SemsegCS/Discriminator/%s' % (opt.curr_scale, k), v, semseg_steps)
                # T -> S:
                discriminator_losses = adversarial_discriminative_train(netDgta, netGts, Gts, source_scales[opt.curr_scale], target_scales, opt, semseg=semseg_gta)
                for k,v in discriminator_losses.items():
                    opt.tb.add_scalar('Scale%d/SemsegGTA/Discriminator/%s' % (opt.curr_scale, k), v, semseg_steps)
                optimizerDcs.step()
                optimizerDgta.step()

                optimizerSemsegCS.zero_grad()
                if not opt.warmup:
                    optimizerGst.zero_grad()
                    optimizerGts.zero_grad()

                # Train semseg on GTA5 image converted to CS, using GTA5 labels:
                prev = concat_pyramid(Gst, source_scales, opt)
                fake_image = netGst(source_scales[-1], prev, source_segmap)
                semseg_softs, semseg_loss = semseg_cs(fake_image, source_label)
                semseg_loss = semseg_loss.mean()
                semseg_labels = semseg_softs.argmax(1)
                semseg_loss.backward()
                opt.tb.add_scalar('Semseg/SemsegLoss', semseg_loss.item(), semseg_steps)


                # S -> T:
                generator_losses = adversarial_generative_train(netGst, netDcs, Gst, source_scales, opt, semseg_net=semseg_cs)
                for k,v in generator_losses.items():
                    opt.tb.add_scalar('Scale%d/SemsegCS/Generator/%s' % (opt.curr_scale,k), v, semseg_steps)
                # T -> S:
                generator_losses = adversarial_generative_train(netGts, netDgta, Gts, target_scales, opt, semseg_net=semseg_gta)
                for k,v in generator_losses.items():
                    opt.tb.add_scalar('Scale%d/SemsegGTA/Generator/%s' % (opt.curr_scale,k), v, semseg_steps)

                optimizerSemsegCS.step()
                if not opt.warmup:
                    optimizerGst.step()
                    optimizerGts.step()
                
                semseg_steps += 1

            if int(steps / opt.print_rate) >= print_int or steps == 0:
                elapsed = time.time() - start
                print('scale %d:[step %d/%d] ; elapsed time = %.2f secs per step, %.2f secs per image' %
                      (opt.curr_scale, print_int * opt.print_rate, total_steps_per_scale, elapsed / opt.print_rate, elapsed / opt.print_rate / batch_size))
                start = time.time()
                print_int += 1

            if int(steps / opt.save_pics_rate) >= save_pics_int or steps == 0:
                s = norm_image(source_scales[opt.curr_scale][0])
                t = norm_image(target_scales[opt.curr_scale][0])
                sit = norm_image(cyc_images['sit'][0])
                sitis = norm_image(cyc_images['sitis'][0])
                tis = norm_image(cyc_images['tis'][0])
                tisit = norm_image(cyc_images['tisit'][0])
                opt.tb.add_image('Scale%d/Cyclic/source' % opt.curr_scale, s, save_pics_int * opt.save_pics_rate)
                opt.tb.add_image('Scale%d/Cyclic/source_in_traget' % opt.curr_scale, sit, save_pics_int * opt.save_pics_rate)
                opt.tb.add_image('Scale%d/Cyclic/source_in_traget_in_source' % opt.curr_scale, sitis, save_pics_int * opt.save_pics_rate)
                opt.tb.add_image('Scale%d/Cyclic/target' % opt.curr_scale, t, save_pics_int * opt.save_pics_rate)
                opt.tb.add_image('Scale%d/Cyclic/target_in_source' % opt.curr_scale, tis, save_pics_int * opt.save_pics_rate)
                opt.tb.add_image('Scale%d/Cyclic/target_in_source_in_target' % opt.curr_scale, tisit, save_pics_int * opt.save_pics_rate)

                if opt.last_scale:
                    sit_label = colorize_mask(semseg_labels[0])
                    softs_max = torch.nn.functional.softmax(semseg_softs, dim=1)
                    hist_values = softs_max.max(dim=1)[0][0]
                    s_label = colorize_mask(source_label[0])
                    opt.tb.add_image('Scale%d/SemsegCS/source_in_target_label' % opt.curr_scale, sit_label, save_pics_int * opt.save_pics_rate)
                    opt.tb.add_image('Scale%d/SemsegCS/source_in_target_values' % opt.curr_scale, hist_values, save_pics_int * opt.save_pics_rate, dataformats='HW')
                    opt.tb.add_histogram('Scale%d/SemsegCS/source_in_target_histogram' % opt.curr_scale, hist_values, save_pics_int * opt.save_pics_rate, bins='auto')
                    opt.tb.add_image('Scale%d/SemsegCS/source_label' % opt.curr_scale, s_label, save_pics_int * opt.save_pics_rate)
                    opt.tb.add_image('Scale%d/LabelsCyclic/source_label' % opt.curr_scale, s_label, save_pics_int * opt.save_pics_rate)
                    if not opt.warmup:
                        sitis_label = colorize_mask(cyc_images['sitis_softs'].argmax(1)[0])
                        opt.tb.add_image('Scale%d/LabelsCyclic/sitis_label' % opt.curr_scale, sitis_label, save_pics_int * opt.save_pics_rate)
                        if opt.use_target_label_loss:
                            softs_max_target = torch.nn.functional.softmax(cyc_images['t_softs'], dim=1)
                            hist_values_target = softs_max_target.max(dim=1)[0][0]
                            t_label = colorize_mask(cyc_images['t_softs'].argmax(1)[0])
                            tisit_label = colorize_mask(cyc_images['tisit_softs'].argmax(1)[0])
                            opt.tb.add_image('Scale%d/target_values' % opt.curr_scale, hist_values_target, save_pics_int * opt.save_pics_rate, dataformats='HW')
                            opt.tb.add_histogram('Scale%d/target_histogram' % opt.curr_scale, hist_values_target, save_pics_int * opt.save_pics_rate, bins='auto')
                            opt.tb.add_image('Scale%d/LabelsCyclic/target_label' % opt.curr_scale, t_label, save_pics_int * opt.save_pics_rate)
                            opt.tb.add_image('Scale%d/LabelsCyclic/tisit_label' % opt.curr_scale, tisit_label, save_pics_int * opt.save_pics_rate)

                save_pics_int += 1

            # Save network checkpoint every opt.save_checkpoint_rate steps:
            if steps > checkpoint_int * opt.save_checkpoint_rate:
                print('scale %d: saving networks after %d steps...' % (opt.curr_scale, steps))
                save_networks(opt.outf, netDst, netGst, netDts, netGts, netDcs, netDgta, Gst, Gts, Dst, Dts, Dcs, Dgta, opt, semseg_cs)
                checkpoint_int += 1

            steps += np.minimum(opt.Gsteps, opt.Dsteps)

        ############################
        # (5) Validate performance after each epoch if we are at the last scale:
        ############################
        if opt.last_scale:
            iou, miou, cm = calculte_validation_accuracy(semseg_cs, opt.target_validation_loader, epoch_num, opt)
            export_epoch_accuracy(opt, iou, miou, cm, epoch_num)
            if miou > opt.best_miou:
                opt.best_miou = miou
                save_networks(os.path.join(opt.checkpoints_dir, '%.2f_mIoU_model' % (miou)),
                              netDst, netGst, netDts, netGts, netDcs, netDgta,
                              Gst, Gts, Dst, Dts, Dcs, Dgta, opt, semseg_cs)
            # Static Focal Loss: set weight vector for next epoch
            # Only after training has passed 40% of the epochs in the last scale.
            if opt.use_focal_static_loss and epoch_num >= int(opt.epochs_per_scale*0.4):
                semseg_cs.ce_loss.weight = (1-miou)**2
        epoch_num += 1

    save_networks(opt.outf, netDst, netGst, netDts, netGts, netDcs, netDgta, Gst, Gts, Dst, Dts, Dcs, Dgta, opt, semseg_cs)
    return (netDst, netGst, netDts, netGts, netDcs, netDgta) if len(opt.gpus) == 1 else (netDst.module, netGst.module, netDts.module, netGts.module, netDcs.module, netDgta.module)


def adversarial_discriminative_train(netD, netG, Gs, real_images, from_scales, opt, semseg=None, real_segmap=None):
    losses = {}
    # train with real image
    real_images = real_images if semseg==None else semseg(real_images).detach()
    errD_real = -1 * netD(real_images).mean()
    losses['RealImagesLoss'] = errD_real.item()
    errD_real *=  opt.lambda_adversarial
    errD_real.backward(retain_graph=True)

    # train with fake
    # fake_images = generate_image(netG, from_scales[opt.curr_scale], Gs, from_scales, real_segmap, opt, grad_on_generator=False)
    with torch.no_grad():
        if semseg==None:
            curr = from_scales[opt.curr_scale]
            prev = concat_pyramid(Gs, from_scales, opt)
            fake_images = netG(curr, prev, real_segmap)
        else:
            fake_images = semseg(from_scales[opt.curr_scale])

    errD_fake = netD(fake_images.detach()).mean()
    losses['FakeImagesLoss'] = errD_fake.item()
    errD_fake *= opt.lambda_adversarial
    errD_fake.backward(retain_graph=True)

    gradient_penalty = calc_gradient_penalty(netD, real_images, fake_images, opt.lambda_grad, opt.device)
    losses['LossGP'] = gradient_penalty.item()
    gradient_penalty *= opt.lambda_adversarial
    gradient_penalty.backward()

    losses['LossAdversarial'] = losses['FakeImagesLoss'] + losses['RealImagesLoss'] + losses['LossGP']
    return losses

def adversarial_generative_train(netG, netD, Gs, source_scales, opt,
                                 semseg_net=None,
                                 source_content_features=None, target_style_features=None, source_segmap=None, retain_graph=True):
    losses = {}
    # train with fake
    fake_target_image = generate_image(netG, source_scales[opt.curr_scale], Gs, source_scales, source_segmap, opt)
    # fake_content_features, fake_style_features = opt.style_transfer_loss.extract_features(fake_target_image)
    if semseg_net != None:
        fake_target_image = semseg_net(fake_target_image)
    adv_loss = -1 * netD(fake_target_image).mean()
    losses['LossAdversarial'] = adv_loss.item()
    adv_loss *=  opt.lambda_adversarial
    adv_loss.backward(retain_graph=True)

    # if source_content_features != None and target_style_features != None:
    #     total_style_loss, style_losses = opt.style_transfer_loss(fake_target_image, fake_content_features, fake_style_features,
    #                                       source_content_features, target_style_features)
    #     total_style_loss *=  opt.lambda_style
    #     total_style_loss.backward(retain_graph=retain_graph)
    #     losses.update(style_losses)
    return losses


def cycle_consistency_loss(source_scales, currGst, Gst_pyramid,
                           target_scales, currGts, Gts_pyramid, opt,
                           source_label, source_segmap,
                           semseg_cs=None, semseg_gta=None):
    losses = {}
    images = {}
    criterion_sts = nn.L1Loss()
    criterion_tst = nn.L1Loss()
    criterion_target_labels = nn.L1Loss()
    source_batch = source_scales[-1]
    target_batch = target_scales[-1]

    # source in target:
    # with torch.no_grad():
    prev_sit = concat_pyramid(Gst_pyramid, source_scales, opt)
    sit_image = currGst(source_batch, prev_sit, source_segmap)
    images['sit'] = sit_image
    with torch.no_grad():
        generated_pyramid_sit = GeneratePyramid(sit_image, opt.num_scales, opt.curr_scale, opt.scale_factor, opt.image_full_size)
    prev_sit_generated = concat_pyramid(Gts_pyramid, generated_pyramid_sit, opt)
    # source in target in source:
    sitis_image = currGts(sit_image, prev_sit_generated, source_segmap)
    images['sitis'] = sitis_image
    loss_sts = criterion_sts(sitis_image, source_batch)
    losses['LossSTS'] = loss_sts.item()
    loss_sts *= opt.lambda_cyclic
    loss_sts.backward(retain_graph=opt.last_scale)

    # Source Cyclic Label Loss:
    if opt.last_scale and not opt.warmup:
        softs_source_labels, loss_source_labels = semseg_gta(sitis_image, source_label)
        losses['SourceLabelLoss'] = loss_source_labels.item()
        images['sitis_softs'] = softs_source_labels
        loss_source_labels *= opt.lambda_labels
        loss_source_labels.backward()


    # traget in source:
    # with torch.no_grad():
    prev_tis = concat_pyramid(Gts_pyramid, target_scales, opt)
    tis_image = currGts(target_batch, prev_tis, None)
    images['tis'] = tis_image
    with torch.no_grad():
        generated_pyramid_tis = GeneratePyramid(tis_image, opt.num_scales, opt.curr_scale, opt.scale_factor, opt.image_full_size)
    prev_tis_generated = concat_pyramid(Gst_pyramid, generated_pyramid_tis, opt)
    # target in source in target:
    tisit_image = currGst(tis_image, prev_tis_generated, None)
    images['tisit'] = tisit_image
    loss_tst = criterion_tst(tisit_image, target_batch)
    losses['LossTST'] = loss_tst.item()
    loss_tst *= opt.lambda_cyclic
    loss_tst.backward(retain_graph=opt.last_scale and not opt.warmup)

    # Target Label Cyclic Loss:
    if opt.use_target_label_loss and opt.last_scale and not opt.warmup:
        target_softs = semseg_cs(target_batch)
        images['t_softs'] = target_softs
        tisit_softs = semseg_cs(tisit_image)
        images['tisit_softs'] = tisit_softs
        loss_labels_target = criterion_target_labels(target_softs, tisit_softs)
        losses['TargetLabelLoss'] = loss_labels_target.item()
        loss_labels_target *= opt.lambda_labels
        loss_labels_target.backward()
    losses['TotalLoss'] = losses['LossTST'] + losses['LossSTS']

    return losses, images


def synthetic_semantic_segmentation_loss(input_pyramid, Gs, currG, semseg_net, input_label, opt):
    prev_converted_image = concat_pyramid(Gs, input_pyramid, opt)
    converted_image = currG(input_pyramid[-1], prev_converted_image, one_hot_encoder(input_label) if not opt.warmup else None)
    output_softs, semseg_loss = semseg_net(converted_image, input_label)
    semseg_loss = semseg_loss.mean()
    output_label = output_softs.argmax(1)
    semseg_loss.backward()
    return output_softs, output_label, semseg_loss

def real_semantic_segmentation_loss(input_target_image, trusted_label, semseg_net, opt):
    output_softs, semseg_loss = semseg_net(input_target_image, trusted_label)
    semseg_loss = semseg_loss.mean()
    output_label = output_softs.argmax(1)
    semseg_loss.backward()
    return output_softs, output_label, semseg_loss


def concat_pyramid(Gs, sources, opt):
    if len(Gs) == 0:
        return torch.zeros_like(sources[0])
    # with torch.no_grad():
    G_z = sources[0]
    # labels = [None] * len(sources) if labels == None else labels
    for G, source_curr, source_next in zip(Gs, sources, sources[1:]):
        G_z = G_z[:, :, 0:source_curr.shape[2], 0:source_curr.shape[3]]
        G_z = G(source_curr, G_z.detach())
        # G_z = imresize(G_z, 1 / opt.scale_factor, opt)
        G_z = imresize_torch(G_z, 1 / opt.scale_factor, mode='bicubic')
        G_z = G_z[:, :, 0:source_next.shape[2], 0:source_next.shape[3]]
    return G_z.detach()

def generate_image(netG, curr_images, Gs, scales, cond_image, opt):
    # with torch.no_grad():
    prevs = concat_pyramid(Gs, scales, opt)
    fake_images = netG(curr_images, prevs, cond_image)

    return fake_images

def init_models(opt):
    use_four_level_net = np.power(opt.scale_factor, opt.num_scales - opt.curr_scale) * np.minimum(H, W) / 16 > opt.ker_size

    # generator initialization:
    if opt.use_unet_generator:
        # use_four_level_unet = np.power(opt.scale_factor, opt.num_scales - opt.curr_scale) * np.minimum(H,W) / 16 > opt.ker_size
        if use_four_level_net:
            print('Generating 4 layers UNET model')
            netG = models.UNetGeneratorFourLayers(opt).to(opt.device)
        else:
            print('Generating 2 layers UNET model')
            netG = models.UNetGeneratorTwoLayers(opt).to(opt.device)
    elif opt.use_fcc or opt.use_fcc_g:
        netG = models.FCCGenerator(opt).to(opt.device)
    else:  # Conditial Generator(!):
        netG = models.LabelConditionedGenerator(opt).to(opt.device)
        # if opt.curr_scale == opt.num_scales: # last scale, initialize label conditioning:
        #     netG = models.LabelConditionedGenerator(opt).to(opt.device)
        # else: # not last scale, use regular generator:
        #     netG = models.ConvGenerator(opt).to(opt.device)
    netG.apply(models.weights_init)

    # discriminator initialization:
    if opt.use_downscale_discriminator:
        netD = models.WDiscriminatorDownscale(opt, use_four_level_net).to(opt.device)
    elif opt.use_fcc or opt.use_fcc_d:
        netD = models.FCCDiscriminator(opt).to(opt.device)
    else:
        netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)

    # discriminator for semseg initialization:
    netD_SS = models.WDiscriminator(opt, is_semseg_discriminator=True).to(opt.device)
    netD_SS.apply(models.weights_init)

    return netD, netG, netD_SS


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

    if os.path.isfile(os.path.join(opt.continue_train_from_path, 'semseg_cs.pth')):
        semseg_cs = torch.load(os.path.join(opt.continue_train_from_path, 'semseg_cs.pth'))
        if not hasattr(semseg_cs, 'ce_loss'):
            semseg_cs.ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    else:
        semseg_cs = None

    if os.path.isfile(os.path.join(opt.continue_train_from_path, 'Dcs.pth')):
        Dcs = torch.load(os.path.join(opt.continue_train_from_path, 'Dcs.pth'))
    else:
        Dcs = None

    if os.path.isfile(os.path.join(opt.continue_train_from_path, 'Dgta.pth')):
        Dgta = torch.load(os.path.join(opt.continue_train_from_path, 'Dgta.pth'))
    else:
        Dgta = None

    return Gst, Gts, Dst, Dts, semseg_cs, Dcs, Dgta


def export_epoch_accuracy(opt, iou, miou, cm, epoch):
    for i in range(NUM_CLASSES):
        opt.tb.add_scalar('Semseg/%s class accuracy' % (trainId2label[i].name), iou[i], epoch)
    opt.tb.add_scalar('Semseg/Accuracy History [mIoU]', miou, epoch)
    print('================Model Acuuracy Summery================')
    for i in range(NUM_CLASSES):
        print('%s class accuracy: = %.2f%%' % (trainId2label[i].name, iou[i] * 100))
    print('Average accuracy of test set on target domain: mIoU = %2f%%' % (miou * 100))
    print('======================================================')


def calculte_validation_accuracy(semseg_net, target_val_loader, epoch_num, opt):
    semseg_net.eval()
    with torch.no_grad():
        running_metrics_val = runningScore(NUM_CLASSES)
        cm = torch.zeros((NUM_CLASSES, NUM_CLASSES)).cuda()
        for val_batch_num, (target_images, target_labels) in enumerate(target_val_loader):
            if opt.use_half_image_size:
                target_labels = (nn.functional.interpolate(target_labels.unsqueeze(1), scale_factor=[0.5, 0.5], mode='nearest')).squeeze(1)
                target_images = torch.clamp(nn.functional.interpolate(target_images, scale_factor=[0.5, 0.5], mode='bicubic'), -1, 1)

            if opt.debug_run and val_batch_num > opt.debug_stop_iteration:
                break
            target_images = target_images.to(opt.device)
            target_labels = target_labels.to(opt.device)
            pred_softs = semseg_net(target_images)
            pred_labels = torch.argmax(pred_softs, dim=1)
            cm += compute_cm_batch_torch(pred_labels, target_labels, IGNORE_LABEL, NUM_CLASSES)
            running_metrics_val.update(target_labels.cpu().numpy(), pred_labels.cpu().numpy())
            if val_batch_num == 0:
                t = norm_image(target_images[0])
                t_lbl = colorize_mask(target_labels[0])
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

def set_pics_save_rate(pics_per_epoch, batch_size, opt):
    return np.maximum(2, int(opt.epoch_size * np.minimum(opt.Dsteps, opt.Gsteps) / batch_size / pics_per_epoch))
