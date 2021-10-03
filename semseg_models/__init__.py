from semseg_models.deeplab import Deeplab
from semseg_models.deeplabv2 import DeeplabV2
from semseg_models.fcn8s import VGG16_FCN8s
import torch.optim as optim
import os
from core.constants import NUM_CLASSES
from semseg_models.build import build_feature_extractor, build_classifier, ASPP_Classifier_V2
import torch

def CreateSemsegModel(args):
    model, optimizer = None, None
    if args.model == 'DeepLab':
        model = Deeplab(num_classes=NUM_CLASSES)
        optimizer = optim.SGD(model.optim_parameters(args),
                              lr=args.lr_semseg, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer.zero_grad()

    if args.model == 'DeepLabV2':
        if args.load_pretrained_semseg_on_gta:
            model = torch.load(args.pretrained_deeplabv2_on_gta_miou_70)
        elif args.images_per_gpu[args.curr_scale] > 16 or args.force_bn_in_deeplab:
            model = DeeplabV2(BatchNorm=True, num_classes=NUM_CLASSES)
        elif args.images_per_gpu[args.curr_scale] <= 16 or args.force_gn_in_deeplab:
            model = DeeplabV2(BatchNorm=False, num_classes=NUM_CLASSES)

        optimizer = optim.SGD(model.optim_parameters(args),
                              lr=args.lr_semseg, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer.zero_grad()

    if args.model == 'VGG':
        model = VGG16_FCN8s(num_classes=NUM_CLASSES)
        optimizer = optim.Adam(
        [
            {'params': model.get_parameters(bias=False)},
            {'params': model.get_parameters(bias=True),
             'lr': args.lr_semseg * 2}
        ],
        lr=args.lr_semseg,
        betas=(0.9, 0.99))
        optimizer.zero_grad()
    # if args.semseg_model_path != '':
    #     model=torch.load(args.pretrained_semseg_path)
    model.to(args.device)
    return model, optimizer

def CreateSemsegPyramidModel(args, dataset_name=None):
    if args.semseg_model_path != '' and not(dataset_name!='GTA' and args.load_only_gta_weights):
        feature_extractor = torch.load(os.path.join(args.semseg_model_path,
                                                    '%s_%s_on_%s_Epoch%d.pth'
                                                    % (args.model,
                                                       'featureExtractor',
                                                       dataset_name,
                                                       args.semseg_model_epoch_to_resume)))
        feature_extractor.to(args.device)
        classifier       = torch.load(os.path.join(args.semseg_model_path,
                                                    '%s_%s_on_%s_Epoch%d.pth'
                                                    % (args.model,
                                                       'classifier',
                                                       dataset_name,
                                                       args.semseg_model_epoch_to_resume)))
        classifier.to(args.device)

    elif args.model == 'DeepLab':
        model_name = 'deeplab_resnet101'
        feature_extractor = build_feature_extractor(model_name)
        feature_extractor.to(args.device)
        classifier = build_classifier(model_name)
        classifier.to(args.device)

    elif args.model == 'VGG':
        model_name = 'deeplab_vgg16'
        feature_extractor = build_feature_extractor(model_name)
        feature_extractor.to(args.device)
        classifier = build_classifier(model_name)
        classifier.to(args.device)
    else:
        raise NotImplementedError

    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=args.lr_semseg, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_fea.zero_grad()
    optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=args.lr_semseg, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_cls.zero_grad()

    return feature_extractor, classifier, optimizer_fea, optimizer_cls
