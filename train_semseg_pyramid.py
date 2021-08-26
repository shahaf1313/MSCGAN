from core.config import get_arguments, post_config
import datetime
from torch import nn
from data import CreateSrcDataLoader, CreateTrgDataLoader
from core.functions import denorm, colorize_mask
import numpy as np
import time
from core.sync_batchnorm import convert_model
from core.constants import NUM_CLASSES, IGNORE_LABEL, trainId2label
from core.functions import compute_cm_batch_torch, compute_iou_torch, imresize_torch
import torch
import os
from torch.utils.tensorboard import SummaryWriter

def main():
    parser = get_arguments()
    opt = parser.parse_args()
    opt = post_config(opt)
    from semseg_models import CreateSemsegPyramidModel
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    Gst = torch.load(os.path.join(opt.multiscale_model_path, 'Gst.pth'), map_location='cpu')
    Gts = torch.load(os.path.join(opt.multiscale_model_path, 'Gst.pth'), map_location='cpu')
    opt.curr_scale = len(Gst)
    opt.num_scales = len(Gst)
    for scaleGst, scaleGsts in zip(Gst,Gts):
        scaleGst.eval()
        scaleGst.to(opt.device)
        scaleGsts.eval()
        scaleGsts.to(opt.device)

    source_train_loader = CreateSrcDataLoader(opt, 'train_semseg_net', get_image_label=True)
    source_val_loader = CreateSrcDataLoader(opt, 'val_semseg_net', get_image_label=True)
    opt.epoch_size = len(source_train_loader.dataset)
    target_val_loader = CreateTrgDataLoader(opt, 'val', get_image_label=True, generate_prev_image=True)

    #Semseg To Cityscapes dataset:
    feature_extractor_cs, classifier_cs, optimizer_fea_cs, optimizer_cls_cs = CreateSemsegPyramidModel(opt, 'CS')
    scheduler_fea_cs = torch.optim.lr_scheduler.StepLR(optimizer_fea_cs, step_size=5,gamma=0.9)
    scheduler_cls_cs = torch.optim.lr_scheduler.StepLR(optimizer_cls_cs, step_size=5, gamma=0.9)

    #Semseg To GTA5 dataset:
    feature_extractor_gta, classifier_gta, optimizer_fea_gta, optimizer_cls_gta = CreateSemsegPyramidModel(opt, 'GTA')
    scheduler_fea_gta = torch.optim.lr_scheduler.StepLR(optimizer_fea_gta, step_size=5,gamma=0.9)
    scheduler_cls_gta = torch.optim.lr_scheduler.StepLR(optimizer_cls_gta, step_size=5, gamma=0.9)

    # Convert to DataPatallel object if needed:
    if len(opt.gpus) > 1:
        # for scale in range(len(Gst)):
        #     Gst[scale] = nn.DataParallel(Gst[scale])
        #     Gts[scale] = nn.DataParallel(Gts[scale])
        feature_extractor_cs, classifier_cs   = convert_model(nn.DataParallel(feature_extractor_cs)).to(opt.device),  convert_model(nn.DataParallel(classifier_cs)).to(opt.device)
        feature_extractor_gta, classifier_gta = convert_model(nn.DataParallel(feature_extractor_gta)).to(opt.device), convert_model(nn.DataParallel(classifier_gta)).to(opt.device)

    print('######### Network created #########')
    print('Architecture of Semantic Segmentation network:\n' + str(classifier_cs) + str(feature_extractor_cs))
    opt.tb = SummaryWriter(os.path.join(opt.tb_logs_dir, '%sGPU%d' % (datetime.datetime.now().strftime('%d-%m-%Y::%H:%M:%S'), opt.gpus[0])))

    steps = 0
    print_int = 0
    save_pics_int = 0
    epoch_num = 1 if opt.semseg_model_epoch_to_resume < 0 else opt.semseg_model_epoch_to_resume + 1
    start = time.time()
    keep_training = True
    opt.save_pics_rate = int(opt.epoch_size * np.maximum(opt.Dsteps, opt.Gsteps) / opt.batch_size / opt.pics_per_epoch)
    total_steps = opt.epochs_semseg * int(opt.epoch_size / opt.batch_size)

    while keep_training:
        print('semeg train: starting epoch %d...' % (epoch_num))
        feature_extractor_cs.train()
        classifier_cs.train()
        feature_extractor_gta.train()
        classifier_gta.train()

        for batch_num, (source_scales, source_label) in enumerate(source_train_loader):
            if steps > total_steps:
                keep_training = False
                break
            if opt.debug_run and steps > 20*epoch_num:
                break

            # Move scale tensors to CUDA:
            for i in range(len(source_scales)):
                source_scales[i] = source_scales[i].to(opt.device)
            source_label = source_label.type(torch.long)
            source_label = source_label.to(opt.device)

            #Train Semseg of CS:
            optimizer_fea_cs.zero_grad()
            optimizer_cls_cs.zero_grad()
            with torch.no_grad():
                source_in_target_cs = create_target_from_source(Gst, source_scales, opt)
            size = source_label.shape[-2:]
            pred_softs_cs = classifier_cs(feature_extractor_cs(source_in_target_cs), size)
            pred_labels_cs = torch.argmax(pred_softs_cs, dim=1)
            loss_cs = criterion(pred_softs_cs, source_label)
            loss_cs.backward()
            optimizer_fea_cs.step()
            optimizer_cls_cs.step()
            opt.tb.add_scalar('TrainSemsegCityscapes/loss', loss_cs.item(), steps)

            #Train Semseg of GTA:
            optimizer_fea_gta.zero_grad()
            optimizer_cls_gta.zero_grad()
            source_in_target_gta = source_scales[-1]
            size = source_label.shape[-2:]
            pred_softs_gta = classifier_gta(feature_extractor_gta(source_in_target_gta), size)
            pred_labels_gta = torch.argmax(pred_softs_gta, dim=1)
            loss_gta = criterion(pred_softs_gta, source_label)
            loss_gta.backward()
            optimizer_fea_gta.step()
            optimizer_cls_gta.step()
            opt.tb.add_scalar('TrainSemsegGTA5/loss', loss_gta.item(), steps)

            if int(steps/opt.print_rate) >= print_int or steps == 0:
                elapsed = time.time() - start
                print('train semseg:[%d/%d] ; elapsed time = %.2f secs per step' %
                      (print_int*opt.print_rate, total_steps, elapsed/opt.print_rate))
                start = time.time()
                print_int += 1
            if int(steps/opt.save_pics_rate) >= save_pics_int or steps == 0:
                s       = denorm(source_scales[-1][0])
                s_lbl   = colorize_mask(source_label[0])
                sit_cs  = denorm(source_in_target_cs[0])
                sit_lbl_cs = colorize_mask(pred_labels_cs[0])
                sit_lbl_gta = colorize_mask(pred_labels_gta[0])
                opt.tb.add_image('TrainSemseg/source', s, save_pics_int*opt.save_pics_rate)
                opt.tb.add_image('TrainSemseg/source_in_target', sit_cs, save_pics_int*opt.save_pics_rate)
                opt.tb.add_image('TrainSemseg/source_gt_label', s_lbl, save_pics_int*opt.save_pics_rate)
                opt.tb.add_image('TrainSemseg/source_in_target_semseg_label_cs', sit_lbl_cs, save_pics_int*opt.save_pics_rate)
                opt.tb.add_image('TrainSemseg/source_semseg_label_gta', sit_lbl_gta, save_pics_int*opt.save_pics_rate)
                save_pics_int += 1

            steps += 1

        if opt.debug_run and epoch_num > 10:
            keep_training = False

        # Update LR:
        scheduler_fea_cs.step()
        scheduler_cls_cs.step()
        scheduler_fea_gta.step()
        scheduler_cls_gta.step()

        #Validation:
        # Cityscapes dataset:
        print('train semseg: starting validation after epoch %d.' % epoch_num)
        cs_results, gta_results, voting_results = calculte_validation_accuracy_cs(feature_extractor_cs, classifier_cs, feature_extractor_gta, classifier_gta, Gts, target_val_loader, opt, epoch_num)
        save_epoch_accuracy(opt.tb, 'CitsyscapesValidationOnlyCityscapes', cs_results[0], cs_results[1], epoch_num)
        save_epoch_accuracy(opt.tb, 'CitsyscapesValidationOnlyGTA5', gta_results[0], gta_results[1], epoch_num)
        save_epoch_accuracy(opt.tb, 'CitsyscapesValidationVoting', voting_results[0], voting_results[1], epoch_num)
        print('train semseg: average accuracy of epoch #%d on target domain using Cityscape Semseg only: mIoU = %2f' % (epoch_num, cs_results[1]))
        print('train semseg: average accuracy of epoch #%d on target domain using GTA5 Semseg only: mIoU = %2f' % (epoch_num, gta_results[1]))
        print('train semseg: average accuracy of epoch #%d on target domain using voting: mIoU = %2f' % (epoch_num, voting_results[1]))

        #GTA5 dataset:
        iou_gta_val, miou_gta_val, cm_gta_val = calculte_validation_accuracy_gta(feature_extractor_gta, classifier_gta, source_val_loader, opt, epoch_num)
        save_epoch_accuracy(opt.tb, 'GTA5ValidationTrainedOnGTA5', iou_gta_val, miou_gta_val, epoch_num)
        print('train semseg: average accuracy of epoch #%d on source domain trained with source images: mIoU = %2f' % (epoch_num, miou_gta_val))

        opt.tb.add_scalars('Epoch Acuuracy Summery', {'mIoU Cityscaps only': cs_results[1],
                                                      'mIoU GTA5 only': gta_results[1],
                                                      'mIoU Voting only': voting_results[1],
                                                      'mIoU On Source (GTA5 images on GTA5 semseg)': miou_gta_val}, epoch_num)

        # Save checkpoint:
        save_checkpoint(feature_extractor_cs, classifier_cs, feature_extractor_gta, classifier_gta, epoch_num, opt)
        epoch_num += 1

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
    for i in range(NUM_CLASSES):
        tb.add_scalar('%s/%s class accuracy' % (set, trainId2label[i].name), iou[i], epoch)
    tb.add_scalar('%s/Accuracy History [mIoU]' % set, miou, epoch)
    print('================Epoch Acuuracy Summery================')
    for i in range(NUM_CLASSES):
        print('%s class accuracy: = %.2f' % (trainId2label[i].name, iou[i]))
    print('Average accuracy of test set on target domain: mIoU = %2f' % miou)
    print('======================================================')

def calculte_validation_accuracy_cs(feature_extractor_cs, classifier_cs, feature_extractor_gta, classifier_gta, Gts, target_val_loader, opt, epoch_num):
    feature_extractor_cs.eval()
    classifier_cs.eval()
    feature_extractor_gta.eval()
    classifier_gta.eval()
    rand_samp_inds = np.random.randint(0, len(target_val_loader.dataset), 5)
    rand_batchs = np.floor(rand_samp_inds/opt.batch_size).astype(np.int)
    cm_cs, cm_gta, cm_voting = torch.zeros((NUM_CLASSES, NUM_CLASSES)).cuda(), torch.zeros((NUM_CLASSES, NUM_CLASSES)).cuda(), torch.zeros((NUM_CLASSES, NUM_CLASSES)).cuda()
    for val_batch_num, (target_scales, target_labels) in enumerate(target_val_loader):
        if opt.debug_run and val_batch_num > 15:
            break
        # Move scale tensors to CUDA:
        for i in range(len(target_scales)):
            target_scales[i] = target_scales[i].to(opt.device)
        target_labels = target_labels.to(opt.device)

        with torch.no_grad():
            size = target_labels.shape[-2:]
            pred_softs_cs = get_pred_softs_cs(target_scales[-1], feature_extractor_cs, classifier_cs, size)
            pred_softs_gta = get_pred_softs_gta(target_scales, feature_extractor_gta, classifier_gta, size, Gts, opt)
            pred_softs_voting = 0.5*(pred_softs_gta + pred_softs_cs)

            pred_labels_cs     = torch.argmax(pred_softs_cs, dim=1)
            pred_labels_gta    = torch.argmax(pred_softs_gta, dim=1)
            pred_labels_voting = torch.argmax(pred_softs_voting, dim=1)

            cm_cs += compute_cm_batch_torch(pred_labels_cs, target_labels, IGNORE_LABEL, NUM_CLASSES)
            cm_gta += compute_cm_batch_torch(pred_labels_gta, target_labels, IGNORE_LABEL, NUM_CLASSES)
            cm_voting += compute_cm_batch_torch(pred_labels_voting, target_labels, IGNORE_LABEL, NUM_CLASSES)

            if val_batch_num in rand_batchs or val_batch_num==0:
                t               = denorm(target_scales[-1][0])
                t_lbl           = colorize_mask(target_labels[0])
                pred_lbl_cs     = colorize_mask(pred_labels_cs[0])
                pred_lbl_gta    = colorize_mask(pred_labels_gta[0])
                pred_lbl_voting = colorize_mask(pred_labels_voting[0])
                opt.tb.add_image('ValidtaionCityscapesEpoch%d/target' % epoch_num, t, val_batch_num)
                opt.tb.add_image('ValidtaionCityscapesEpoch%d/target_label' % epoch_num, t_lbl, val_batch_num)
                opt.tb.add_image('ValidtaionCityscapesEpoch%d/prediction_label_cs' % epoch_num, pred_lbl_cs, val_batch_num)
                opt.tb.add_image('ValidtaionCityscapesEpoch%d/prediction_label_gta' % epoch_num, pred_lbl_gta, val_batch_num)
                opt.tb.add_image('ValidtaionCityscapesEpoch%d/prediction_label_voting' % epoch_num, pred_lbl_voting, val_batch_num)
    iou_cs, miou_cs = compute_iou_torch(cm_cs)
    iou_gta, miou_gta = compute_iou_torch(cm_gta)
    iou_voting, miou_voting = compute_iou_torch(cm_voting)
    return (iou_cs, miou_cs, cm_cs), (iou_gta, miou_gta, cm_gta), (iou_voting, miou_voting, cm_voting)

def calculte_validation_accuracy_gta(feature_extractor_gta, classifier_gta, source_val_loader, opt, epoch_num):
    feature_extractor_gta.eval()
    classifier_gta.eval()
    rand_samp_inds = np.random.randint(0, len(source_val_loader.dataset), 5)
    rand_batchs = np.floor(rand_samp_inds/opt.batch_size).astype(np.int)
    cm = torch.zeros((NUM_CLASSES, NUM_CLASSES)).cuda()
    # todo: add validation on target images trained on gta5 semseg!!!
    for val_batch_num, (source_images, source_labels) in enumerate(source_val_loader):
        if opt.debug_run and val_batch_num > 15:
            break
        # Move scale tensors to CUDA:
        for i in range(len(source_images)):
            source_images[i] = source_images[i].to(opt.device)
        source_labels = source_labels.to(opt.device)
        with torch.no_grad():
            size = source_labels.shape[-2:]
            pred_softs = classifier_gta(feature_extractor_gta(source_images[-1]), size)
            pred_labels = torch.argmax(pred_softs, dim=1)
            cm += compute_cm_batch_torch(pred_labels, source_labels, IGNORE_LABEL, NUM_CLASSES)
            if val_batch_num in rand_batchs or val_batch_num==0:
                s               = denorm(source_images[-1][0])
                s_lbl           = colorize_mask(source_labels[0])
                pred_lbl     = colorize_mask(pred_labels[0])
                opt.tb.add_image('ValidtaionGTA5Epoch%d/source' % epoch_num, s, val_batch_num)
                opt.tb.add_image('ValidtaionGTA5Epoch%d/source_label' % epoch_num, s_lbl, val_batch_num)
                opt.tb.add_image('ValidtaionGTA5Epoch%d/prediction_label_gta' % epoch_num, pred_lbl, val_batch_num)
    iou, miou = compute_iou_torch(cm)
    return iou, miou, cm

def get_pred_softs_cs(target_images, feature_extractor, classifier, size):
    pred_softs = classifier(feature_extractor(target_images), size)
    return pred_softs

def get_pred_softs_gta(target_scales, feature_extractor, classifier, size, Gts, opt):
    pred_softs = classifier(feature_extractor(create_target_from_source(Gts, target_scales, opt)), size)
    return pred_softs


def save_checkpoint(feature_extractor_cs, classifier_cs, feature_extractor_gta, classifier_gta, epoch_num, opt):
    if len(opt.gpus) > 1:
        torch.save(feature_extractor_cs.module, '%s/%s_%s_on_%s_Epoch%d.pth' % (opt.out_,opt.model, 'featureExtractor', 'CS', epoch_num))
        torch.save(classifier_cs.module, '%s/%s_%s_on_%s_Epoch%d.pth' % (opt.out_,opt.model, 'classifier', 'CS', epoch_num))
        torch.save(feature_extractor_gta.module, '%s/%s_%s_on_%s_Epoch%d.pth' % (opt.out_,opt.model, 'featureExtractor', 'GTA', epoch_num))
        torch.save(classifier_gta.module, '%s/%s_%s_on_%s_Epoch%d.pth' % (opt.out_,opt.model, 'classifier', 'GTA', epoch_num))

    else:
        torch.save(feature_extractor_cs, '%s/%s_%s_on_%s_Epoch%d.pth' % (opt.out_,opt.model, 'featureExtractor', 'CS', epoch_num))
        torch.save(classifier_cs, '%s/%s_%s_on_%s_Epoch%d.pth' % (opt.out_,opt.model, 'classifier', 'CS', epoch_num))
        torch.save(feature_extractor_gta, '%s/%s_%s_on_%s_Epoch%d.pth' % (opt.out_,opt.model, 'featureExtractor', 'GTA', epoch_num))
        torch.save(classifier_gta, '%s/%s_%s_on_%s_Epoch%d.pth' % (opt.out_,opt.model, 'classifier', 'GTA', epoch_num))

if __name__ == "__main__":
    main()

