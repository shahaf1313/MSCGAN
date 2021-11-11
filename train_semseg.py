from semseg_models import CreateSemsegModel
from core.constants import NUM_CLASSES, IGNORE_LABEL, trainId2label
from core.functions import compute_cm_batch_torch, compute_iou_torch, imresize_torch
from data import CreateSrcDataLoader, CreateTrgDataLoader
import torch
from core.config import get_arguments, post_config
from core.functions import denorm, colorize_mask
import numpy as np
import time
import os
from torch.utils.tensorboard import SummaryWriter
import datetime

def main():
    parser = get_arguments()
    opt = parser.parse_args()
    opt = post_config(opt)

    multiscale_model = torch.load(opt.multiscale_model_path)
    opt.curr_scale = len(multiscale_model)
    opt.num_scales = len(multiscale_model)
    for scale in multiscale_model:
        scale.eval()
        scale.to(opt.device)

    source_train_loader = CreateSrcDataLoader(opt, 'train', get_image_label=True)
    opt.epoch_size = len(source_train_loader.dataset)
    target_val_loader = CreateTrgDataLoader(opt, 'val', get_scales_pyramid=True)

    semseg_net, semseg_optimizer = CreateSemsegModel(opt)
    # semseg_scheduler = torch.optim.lr_scheduler.MultiStepLR(semseg_optimizer, milestones=np.arange(0, opt.num_epochs, 10), gamma=0.9)

    print('######### Network created #########')
    print('Architecture of Semantic Segmentation network:\n' + str(semseg_net))
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

            # Move scale tensors to CUDA:
            for i in range(len(source_scales)):
                source_scales[i] = source_scales[i].to(opt.device)
            source_label = source_label.to(opt.device)
            semseg_optimizer.zero_grad()

            with torch.no_grad():
                source_in_target = create_target_from_source(multiscale_model, source_scales, opt)

            predicted, loss_seg, loss_ent = semseg_net(source_in_target, lbl=source_label)
            pred_label = torch.argmax(predicted, dim=1)
            loss = torch.mean(loss_seg + opt.entW*loss_ent)
            loss.backward()
            opt.tb.add_scalar('TrainSemseg/loss', loss.item(), steps)
            opt.tb.add_scalar('TrainSemseg/loss_seg', loss_seg.item(), steps)
            opt.tb.add_scalar('TrainSemseg/loss_ent', loss_ent.item(), steps)
            semseg_optimizer.step()


            if int(steps/opt.print_rate) >= print_int or steps == 0:
                elapsed = time.time() - start
                print('train semseg:[%d/%d] ; elapsed time = %.2f secs per step' %
                      (print_int*opt.print_rate, opt.num_steps, elapsed/opt.print_rate))
                start = time.time()
                print_int += 1

            if int(steps/opt.save_pics_rate) >= save_pics_int or steps == 0:
                s       = denorm(source_scales[-1][0])
                sit     = denorm(source_in_target[0])
                s_lbl   = colorize_mask(source_label[0])
                sit_lbl = colorize_mask(pred_label[0])
                opt.tb.add_image('TrainSemseg/source', s, save_pics_int*opt.save_pics_rate)
                opt.tb.add_image('TrainSemseg/source_in_target', sit, save_pics_int*opt.save_pics_rate)
                opt.tb.add_image('TrainSemseg/source_label', s_lbl, save_pics_int*opt.save_pics_rate)
                opt.tb.add_image('TrainSemseg/source_in_target_label', sit_lbl, save_pics_int*opt.save_pics_rate)
                save_pics_int += 1

            steps += 1
        # Update LR:
        # semseg_scheduler.step()
        #Validation:
        print('train semseg: starting validation after epoch %d.' % epoch_num)
        iou, miou, cm = calculte_validation_accuracy(semseg_net, target_val_loader, opt, epoch_num)
        save_epoch_accuracy(opt.tb, 'Validtaion', iou, miou, epoch_num)
        print('train semseg: average accuracy of epoch #%d on target domain: mIoU = %2f' % (epoch_num, miou))
        if epoch_num % 4 == 0:
            torch.save(semseg_net, '%s/%s_AdaptedToTarget_Epoch%d.pth' % (opt.out_folder, opt.model, epoch_num))
        epoch_num += 1

    #Save final network:
    torch.save(semseg_net, '%s/%s_AdaptedToTarget_Final.pth' % (opt.out_folder, opt.model))

    #Test:
    print('train semseg: starting final accuracy calculation...')
    iou, miou, cm = calculte_validation_accuracy(semseg_net, target_val_loader, opt, epoch_num)
    save_epoch_accuracy(opt.tb, 'Test', iou, miou, epoch_num)
    opt.tb.close()
    print('Finished training.')

def create_target_from_source(Gs, sources, opt):
    G_n = torch.empty(1)
    for G, source_curr, source_next in zip(Gs, sources, sources[1:]):
        G_n = G(source_curr, G_n.detach())
        G_n = imresize_torch(G_n, 1 / opt.scale_factor)
        G_n = G_n[:, :, 0:source_next.shape[2], 0:source_next.shape[3]]
    # Last scale:
    G_n = Gs[-1](sources[-1], G_n.detach())
    return G_n

def save_epoch_accuracy(tb, set, iou, miou, epoch):
    if set == 'Validtaion':
        for i in range(NUM_CLASSES):
            tb.add_scalar('%sAccuracy/%s class accuracy' % (set, trainId2label[i].name), iou[i], epoch)
        tb.add_scalar('%sAccuracy/Accuracy History [mIoU]' % set, miou, epoch)
    elif set == 'Test':
        print('================Model Acuuracy Summery================')
        for i in range(NUM_CLASSES):
            print('%s class accuracy: = %.2f' % (trainId2label[i].name, iou[i]))
        print('Average accuracy of test set on target domain: mIoU = %2f' % miou)
        print('======================================================')

def calculte_validation_accuracy(semseg_net, target_val_loader, opt, epoch_num):
    semseg_net.eval()
    rand_samp_inds = np.random.randint(0, len(target_val_loader.dataset), 5)
    rand_batchs = np.floor(rand_samp_inds/opt.batch_size).astype(np.int)
    cm = torch.zeros((NUM_CLASSES, NUM_CLASSES)).cuda()
    for val_batch_num, (target_images, target_labels) in enumerate(target_val_loader):
        target_images = target_images.to(opt.device)
        target_labels = target_labels.to(opt.device)
        with torch.no_grad():
            pred_softs = semseg_net(target_images)
            pred_labels = torch.argmax(pred_softs, dim=1)
            cm += compute_cm_batch_torch(pred_labels, target_labels, IGNORE_LABEL, NUM_CLASSES)
            if val_batch_num in rand_batchs:
                t        = denorm(target_images[0])
                t_lbl    = colorize_mask(target_labels[0])
                pred_lbl = colorize_mask(pred_labels[0])
                opt.tb.add_image('ValidtaionEpoch%d/target' % epoch_num, t, val_batch_num)
                opt.tb.add_image('ValidtaionEpoch%d/target_label' % epoch_num, t_lbl, val_batch_num)
                opt.tb.add_image('ValidtaionEpoch%d/prediction_label' % epoch_num, pred_lbl, val_batch_num)
    iou, miou = compute_iou_torch(cm)
    return iou, miou, cm

if __name__ == "__main__":
    main()

