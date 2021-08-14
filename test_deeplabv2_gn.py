def main(opt):
    opt.curr_scale=0
    opt.num_scales=0
    opt.num_steps=1e6
    source_train_loader = CreateSrcDataLoader(opt, 'train_semseg_net', get_image_label=True)
    source_val_loader = CreateSrcDataLoader(opt, 'val_semseg_net', get_image_label=True)
    opt.epoch_size = len(source_train_loader.dataset)
    opt.save_pics_rate = set_pics_save_rate(opt.pics_per_epoch, opt.batch_size, opt)
    opt.force_bn_in_deeplab = not opt.use_gn_in_semseg
    opt.force_gn_in_deeplab = opt.use_gn_in_semseg
    opt.norm_type = 'GN' if opt.use_gn_in_semseg else 'BN'
    # opt.force_bn_in_deeplab = False
    # semseg_net_gn, semseg_optimizer_gn = CreateSemsegModel(opt)
    opt.force_bn_in_deeplab = True
    semseg_net, semseg_optimizer = CreateSemsegModel(opt)

    print('Architecture of %s Semantic Segmentation network:\n' % (opt.norm_type) + str(semseg_net))
    # print('Architecture of GN Semantic Segmentation network:\n' + str(semseg_net_gn))
    opt.tb = SummaryWriter(os.path.join(opt.tb_logs_dir, '%sGPU%d' % (datetime.datetime.now().strftime('%d-%m-%Y::%H:%M:%S'), opt.gpus[0])))

    steps = 0
    print_int = 0
    save_pics_int = 0
    epoch_num = 1
    start = time.time()
    keep_training = True

    while keep_training:
        print('semeg train: starting epoch %d...' % (epoch_num))
        semseg_net.train()
        # semseg_net_gn.train()

        for batch_num, (source_scales, source_label) in enumerate(source_train_loader):
            if steps > opt.num_steps:
                keep_training = False
                break

            semseg_optimizer.zero_grad()
            # semseg_optimizer_gn.zero_grad()
            # BN:
            source_image = source_scales[-1].to(opt.device)
            source_label = source_label.to(opt.device)
            output_softs, semseg_loss = semseg_net(source_image, source_label)
            semseg_loss = semseg_loss.mean()
            output_label = output_softs.argmax(1)
            opt.tb.add_scalar('TrainSemseg/%s/loss' % (opt.norm_type), semseg_loss.item(), steps)
            semseg_loss.backward()
            # GN:
            # output_softs_gn, semseg_loss_gn = semseg_net_gn(source_image, source_label)
            # semseg_loss_gn = semseg_loss_gn.mean()
            # output_label_gn = output_softs_gn.argmax(1)
            # opt.tb.add_scalar('TrainSemsegGN/loss', semseg_loss_gn.item(), steps)
            # semseg_loss_gn.backward()
            # Backward:
            semseg_optimizer.step()
            # semseg_optimizer_gn.step()


            if int(steps/opt.print_rate) >= print_int or steps == 0:
                elapsed = time.time() - start
                print('train semseg:[%d/%d] ; elapsed time = %.2f secs per step' %
                      (print_int*opt.print_rate, opt.num_steps, elapsed/opt.print_rate))
                start = time.time()
                print_int += 1

            if int(steps/opt.save_pics_rate) >= save_pics_int or steps == 0:
                s       = denorm(source_image[0])
                s_lbl   = colorize_mask(source_label[0])
                pred_lbl_bn = colorize_mask(output_label[0])
                # pred_lbl_gn = colorize_mask(output_label_gn[0])
                opt.tb.add_image('TrainSemseg/%s/source'%opt.norm_type, s, save_pics_int*opt.save_pics_rate)
                opt.tb.add_image('TrainSemseg/%s/source_label'%opt.norm_type, s_lbl, save_pics_int*opt.save_pics_rate)
                opt.tb.add_image('TrainSemseg/%s/pred_label'%opt.norm_type, pred_lbl_bn, save_pics_int*opt.save_pics_rate)
                # opt.tb.add_image('TrainSemseg/pred_label_gn', pred_lbl_gn, save_pics_int*opt.save_pics_rate)
                save_pics_int += 1

            steps += 1
        # Update LR:
        # semseg_scheduler.step()
        #Validation:
        print('train semseg: starting validation after epoch %d.' % epoch_num)
        iou_bn, miou_bn, cm_bn = calculte_validation_accuracy(semseg_net, opt.norm_type, source_val_loader, opt, epoch_num)
        # iou_gn, miou_gn, cm_gn = calculte_validation_accuracy(semseg_net_gn, 'GN', source_val_loader, opt, epoch_num)
        save_epoch_accuracy(opt.tb, 'Validtaion%s'%opt.norm_type, iou_bn, miou_bn, epoch_num)
        # save_epoch_accuracy(opt.tb, 'ValidtaionBN', iou_gn, miou_gn, epoch_num)
        if epoch_num > 15:
            torch.save(semseg_net, '%s/semseg_net_%s_epoch%d.pth' % (opt.norm_type.lower(), opt.out, epoch_num))
            # torch.save(semseg_net_gn, '%s/semseg_net_gn_epoch%d.pth' % (opt.out, epoch_num))
        epoch_num += 1

    #Save final network:
    torch.save(semseg_net, '%s/semseg_net_%s_epoch%d.pth' % (opt.norm_type.lower(), opt.out, epoch_num))
    # torch.save(semseg_net_gn, '%s/semseg_net_gn_epoch%d.pth' % (opt.out, epoch_num))

    opt.tb.close()
    print('Finished training.')

def save_epoch_accuracy(tb, set, iou, miou, epoch):
    for i in range(NUM_CLASSES):
        tb.add_scalar('%sAccuracy/%s class accuracy' % (set, trainId2label[i].name), iou[i], epoch)
    tb.add_scalar('%sAccuracy/Accuracy History [mIoU]' % set, miou, epoch)


def calculte_validation_accuracy(semseg_net, batch_type, val_loader, opt, epoch_num):
    semseg_net.eval()
    rand_samp_inds = np.random.randint(0, len(val_loader.dataset), 5)
    rand_batchs = np.floor(rand_samp_inds/opt.batch_size).astype(np.int)
    cm = torch.zeros((NUM_CLASSES, NUM_CLASSES)).cuda()
    for batch_num, (images, labels) in enumerate(val_loader):
        images = images[-1].to(opt.device)
        labels = labels.to(opt.device)
        with torch.no_grad():
            pred_softs = semseg_net(images)
            pred_labels = torch.argmax(pred_softs, dim=1)
            cm += compute_cm_batch_torch(pred_labels, labels, IGNORE_LABEL, NUM_CLASSES)
            if batch_num in rand_batchs:
                t        = denorm(images[0])
                t_lbl    = colorize_mask(labels[0])
                pred_lbl = colorize_mask(pred_labels[0])
                opt.tb.add_image('Validtaion%s/Epoch%d/target' % (batch_type, epoch_num), t, batch_num)
                opt.tb.add_image('Validtaion%s/Epoch%d/target_label' % (batch_type, epoch_num), t_lbl, batch_num)
                opt.tb.add_image('Validtaion%s/Epoch%d/prediction_label' % (batch_type, epoch_num), pred_lbl, batch_num)
    iou, miou = compute_iou_torch(cm)
    return iou, miou, cm

def set_pics_save_rate(pics_per_epoch, batch_size, opt):
    return np.maximum(2, int(opt.epoch_size * np.minimum(opt.Dsteps, opt.Gsteps) / batch_size / pics_per_epoch))

if __name__ == "__main__":
    from core.config import get_arguments, post_config
    parser = get_arguments()
    parser.add_argument('--use_gn_in_semseg', default=False, action='store_true')
    opt = parser.parse_args()
    opt = post_config(opt)
    from semseg_models import CreateSemsegModel
    from core.constants import NUM_CLASSES, IGNORE_LABEL, trainId2label
    from core.functions import compute_cm_batch_torch, compute_iou_torch
    from data import CreateSrcDataLoader
    import torch
    from core.config import get_arguments, post_config
    from core.functions import denorm, colorize_mask
    import numpy as np
    import time
    import os
    from torch.utils.tensorboard import SummaryWriter
    import datetime
    main(opt)

