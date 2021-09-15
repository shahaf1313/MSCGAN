def main(opt):
    opt.curr_scale=0
    opt.num_scales=0
    source_train_loader = CreateSrcDataLoader(opt, 'train_semseg_net', get_image_label=True)
    source_val_loader = CreateSrcDataLoader(opt, 'val_semseg_net', get_image_label=True)
    opt.epoch_size = len(source_train_loader.dataset)
    opt.save_pics_rate = set_pics_save_rate(opt.pics_per_epoch, opt.batch_size, opt)
    total_steps_per_scale = opt.epochs_per_scale * int(opt.epoch_size * np.minimum(opt.Dsteps, opt.Gsteps) / opt.batch_size)
    semseg_net, semseg_optimizer = CreateSemsegModel(opt)

    print('Architecture of DeepLavV2 Semantic Segmentation network:\n' + str(semseg_net))
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

        for batch_num, (source_scales, source_label) in enumerate(source_train_loader):
            if steps > total_steps_per_scale:
                keep_training = False
                break
            if opt.debug_run and steps > opt.debug_stop_iteration:
                if opt.debug_stop_epoch <= epoch_num:
                    keep_training = False
                break

            semseg_optimizer.zero_grad()
            # BN:
            source_image = source_scales[-1].to(opt.device)
            source_label = source_label.to(opt.device)
            output_softs, semseg_loss = semseg_net(source_image, source_label)
            semseg_loss = semseg_loss.mean()
            output_label = output_softs.argmax(1)
            opt.tb.add_scalar('TrainSemseg/loss', semseg_loss.item(), steps)
            semseg_loss.backward()
            # Backward:
            semseg_optimizer.step()

            if int(steps/opt.print_rate) >= print_int or steps == 0:
                elapsed = time.time() - start
                print('scale %d:[step %d/%d] ; elapsed time = %.2f secs per step, %.2f secs per image' %
                      (opt.curr_scale, print_int * opt.print_rate,
                       total_steps_per_scale,
                       elapsed / opt.print_rate,
                       elapsed / opt.print_rate / opt.batch_size))
                start = time.time()
                print_int += 1

            if int(steps/opt.save_pics_rate) >= save_pics_int or steps == 0:
                s       = denorm(source_image[0])
                s_lbl   = colorize_mask(source_label[0])
                pred_lbl = colorize_mask(output_label[0])
                opt.tb.add_image('TrainSemseg/source', s, save_pics_int*opt.save_pics_rate)
                opt.tb.add_image('TrainSemseg/source_label', s_lbl, save_pics_int*opt.save_pics_rate)
                opt.tb.add_image('TrainSemseg/pred_label', pred_lbl, save_pics_int*opt.save_pics_rate)
                save_pics_int += 1

            steps += 1
        # Update LR:
        # semseg_scheduler.step()
        #Validation:
        print('train semseg: starting validation after epoch %d.' % epoch_num)
        iou_bn, miou_bn, cm_bn = calculte_validation_accuracy(semseg_net, source_val_loader, opt, epoch_num)
        save_epoch_accuracy(opt.tb, 'Validtaion', iou_bn, miou_bn, epoch_num)
        if epoch_num > 15:
            print('Saving network after %d epochs...' % epoch_num)
            torch.save(semseg_net, '%s/semseg_net.pth' % (opt.out_))
        epoch_num += 1

    opt.tb.close()
    print('Finished training.')

def save_epoch_accuracy(tb, set, iou, miou, epoch):
    for i in range(NUM_CLASSES):
        tb.add_scalar('%sAccuracy/%s class accuracy' % (set, trainId2label[i].name), iou[i], epoch)
    tb.add_scalar('%sAccuracy/Accuracy History [mIoU]' % set, miou, epoch)


def calculte_validation_accuracy(semseg_net, val_loader, opt, epoch_num):
    semseg_net.eval()
    rand_samp_inds = np.random.randint(0, len(val_loader.dataset), 5)
    rand_batchs = np.floor(rand_samp_inds/opt.batch_size).astype(np.int)
    cm = torch.zeros((NUM_CLASSES, NUM_CLASSES)).cuda()
    for batch_num, (images, labels) in enumerate(val_loader):
        if opt.debug_run and batch_num > 15:
            break
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
                opt.tb.add_image('Validtaion/Epoch%d/target' % (epoch_num), t, batch_num)
                opt.tb.add_image('Validtaion/Epoch%d/target_label' % (epoch_num), t_lbl, batch_num)
                opt.tb.add_image('Validtaion/Epoch%d/prediction_label' % (epoch_num), pred_lbl, batch_num)
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

