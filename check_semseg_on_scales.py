def main(opt):
    opt.curr_scale=2
    opt.num_scales=2
    opt.scale_factor=0.5
    source_val_loader = CreateSrcDataLoader(opt, 'val_semseg_net', get_image_label_pyramid=True)
    semseg_gta = torch.nn.DataParallel(torch.load(opt.pretrained_deeplabv2_on_gta_miou_70)) if (len(opt.gpus) > 1) else torch.load(opt.pretrained_deeplabv2_on_gta_miou_70)

    print('Architecture of Semantic Segmentation network:\n' + str(semseg_gta))
    opt.tb = SummaryWriter(os.path.join(opt.tb_logs_dir, '%sGPU%d' % (datetime.datetime.now().strftime('%d-%m-%Y::%H:%M:%S'), opt.gpus[0])))
    semseg_gta.eval()

    #Validation:
    for s in range(3):
        print('starting validation on scale %d.' % s)
        iou, miou, cm = calculte_validation_accuracy(semseg_gta, source_val_loader, opt, s)
        save_epoch_accuracy(opt.tb, 'Validtaion', iou, miou, s)
        print('Scale %s: mIoU = %.2f' % (s, miou))

    opt.tb.close()
    print('Finished Eval Mode.')

def save_epoch_accuracy(tb, set, iou, miou, epoch):
    for i in range(NUM_CLASSES):
        tb.add_scalar('%sAccuracy/%s class accuracy' % (set, trainId2label[i].name), iou[i], epoch)
    tb.add_scalar('%sAccuracy/Accuracy History [mIoU]' % set, miou, epoch)


def calculte_validation_accuracy(semseg_net, val_loader, opt, s):
    semseg_net.eval()
    cm = torch.zeros((NUM_CLASSES, NUM_CLASSES)).cuda()
    for batch_num, (images, labels) in enumerate(val_loader):
        images = images[s].to(opt.device)
        labels = labels[s].to(opt.device)
        assert images.shape[2:] == labels.shape[1:]
        with torch.no_grad():
            pred_softs = semseg_net(images)
            pred_labels = torch.argmax(pred_softs, dim=1)
            cm += compute_cm_batch_torch(pred_labels, labels, IGNORE_LABEL, NUM_CLASSES)
            if batch_num == 0:
                t        = denorm(images[0])
                t_lbl    = colorize_mask(labels[0])
                pred_lbl = colorize_mask(pred_labels[0])
                opt.tb.add_image('Validtaion/scale%d/target' % (s), t, batch_num)
                opt.tb.add_image('Validtaion/scale%d/target_label' % (s), t_lbl, batch_num)
                opt.tb.add_image('Validtaion/scale%d/prediction_label' % (s), pred_lbl, batch_num)
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

