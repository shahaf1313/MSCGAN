def main(opt):
    best_miou = 0
    opt.num_steps=1e6
    opt.curr_scale = opt.semseg_train_scale
    opt.num_epochs_to_adjust = 400
    source_train_loader = CreateSrcDataLoader(opt, 'train_semseg_net', get_image_label=True)
    source_val_loader = CreateSrcDataLoader(opt, 'val_semseg_net', get_image_label=True)
    opt.epoch_size = len(source_train_loader.dataset)
    opt.save_pics_rate = set_pics_save_rate(opt.pics_per_epoch, opt.batch_size, opt)
    semseg_net, semseg_optimizer = CreateSemsegModel(opt)
    semseg_net = torch.nn.DataParallel(semseg_net)

    print('Architecture of Semantic Segmentation network:\n' + str(semseg_net.module))
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
            if steps > opt.num_steps:
                keep_training = False
                break

            semseg_optimizer.zero_grad()
            source_image = source_scales[opt.curr_scale].to(opt.device)
            source_label = torch.nn.functional.interpolate(source_label.to(opt.device).unsqueeze(1),
                                                           scale_factor=opt.scale_factor**(opt.num_scales-opt.curr_scale),
                                                           mode='nearest').squeeze(1)
            output_softs, semseg_loss = semseg_net(source_image, source_label)
            semseg_loss = semseg_loss.mean()
            output_label = output_softs.argmax(1)
            opt.tb.add_scalar('TrainSemseg/loss', semseg_loss.item(), steps)
            semseg_loss.backward()
            semseg_optimizer.step()


            if int(steps/opt.print_rate) >= print_int or steps == 0:
                elapsed = time.time() - start
                print('train semseg:[%d/%d] ; elapsed time = %.2f secs per step' %
                      (print_int*opt.print_rate, opt.num_steps, elapsed/opt.print_rate))
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
        semseg_net.module.adjust_learning_rate(opt, semseg_optimizer, epoch_num)
        #Validation:
        print('train semseg: starting validation after epoch %d.' % epoch_num)
        iou, miou, cm = calculte_validation_accuracy(semseg_net, source_val_loader, opt, epoch_num)
        save_epoch_accuracy(opt.tb, 'Validtaion', iou, miou, epoch_num)
        if miou > best_miou and epoch_num > 0:
            torch.save(semseg_net.module, '%s/semseg_trained_on_gta.pth' % (opt.out_folder))
            with open('%s/miou_results.txt'%(opt.out_folder), 'w') as f:
                f.write('best iou: ' + str(iou) + '\n')
                f.write('best miou: ' + str(miou) + '\n')

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
        images = images[opt.curr_scale].to(opt.device)
        labels = torch.nn.functional.interpolate(labels.to(opt.device).unsqueeze(1),
                                        scale_factor=opt.scale_factor**(opt.num_scales-opt.curr_scale),
                                        mode='nearest').squeeze(1)
        with torch.no_grad():
            pred_softs = semseg_net(images)
            pred_labels = torch.argmax(pred_softs, dim=1)
            cm += compute_cm_batch_torch(pred_labels, labels, IGNORE_LABEL, NUM_CLASSES)
            if batch_num in rand_batchs:
                t        = denorm(images[0])
                t_lbl    = colorize_mask(labels[0])
                pred_lbl = colorize_mask(pred_labels[0])
                opt.tb.add_image('Validtaion/target', t, epoch_num)
                opt.tb.add_image('Validtaion/target_label', t_lbl, epoch_num)
                opt.tb.add_image('Validtaion/prediction_label', pred_lbl, epoch_num)
    iou, miou = compute_iou_torch(cm)
    return iou, miou, cm

def set_pics_save_rate(pics_per_epoch, batch_size, opt):
    return np.maximum(2, int(opt.epoch_size * np.minimum(opt.Dsteps, opt.Gsteps) / batch_size / pics_per_epoch))

if __name__ == "__main__":
    from core.config import get_arguments, post_config
    parser = get_arguments()
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

