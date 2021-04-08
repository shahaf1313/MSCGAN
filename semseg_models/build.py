from SemsegNetworks.layers import FrozenBatchNorm2d
from SemsegNetworks import vgg, resnet
from torch import nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from constants import NUM_CLASSES

def build_feature_extractor(model_name):
    _, backbone_name = model_name.split('_')
    if backbone_name.startswith('resnet'):
        backbone = resnet_feature_extractor(backbone_name, pretrained_weights="https://download.pytorch.org/models/resnet101-5d3b4d8f.pth", aux=False, pretrained_backbone=True, freeze_bn=True)
    elif backbone_name.startswith('vgg'):
        backbone = vgg_feature_extractor(backbone_name, pretrained_weights="https://web.eecs.umich.edu/~justincj/models/vgg16-00b39a1b.pth", aux=False, pretrained_backbone=True, freeze_bn=False)
    else:
        raise NotImplementedError
    return backbone

def build_classifier(model_name):
    _, backbone_name = model_name.split('_')
    if backbone_name.startswith('vgg'):
        classifier = ASPP_Classifier_V2(1024, [6, 12, 18, 24], [6, 12, 18, 24], NUM_CLASSES)
    elif backbone_name.startswith('resnet'):
        classifier = ASPP_Classifier_V2(2048, [6, 12, 18, 24], [6, 12, 18, 24], NUM_CLASSES)
    else:
        raise NotImplementedError
    return classifier

class ASPP_Classifier_V2(nn.Module):
    def __init__(self, in_channels, dilation_series, padding_series, num_classes):
        super(ASPP_Classifier_V2, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(
                    in_channels,
                    num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True,
                )
            )

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x, size=None):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        if size is not None:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return out



class vgg_feature_extractor(nn.Module):
    def __init__(self, backbone_name, pretrained_weights=None, aux=False, pretrained_backbone=True, freeze_bn=False):
        super(vgg_feature_extractor, self).__init__()
        backbone = vgg.__dict__[backbone_name](
            pretrained=pretrained_backbone, pretrained_weights=pretrained_weights)
        features, _ = list(backbone.features.children()), list(backbone.classifier.children())
        features = nn.Sequential(*(features[i] for i in list(range(23))+list(range(24,30))))
        for i in [23,25,27]:
            features[i].dilation = (2, 2)
            features[i].padding = (2, 2)
        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

        backbone = nn.Sequential(*([features[i] for i in range(len(features))] + [fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))
        return_layers = {'4': 'low_fea', '32': 'out'}

        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, x):
        feas = self.backbone(x)
        out = feas['out']
        return out

class resnet_feature_extractor(nn.Module):
    def __init__(self, backbone_name, pretrained_weights=None, aux=False, pretrained_backbone=True, freeze_bn=False):
        super(resnet_feature_extractor, self).__init__()
        bn_layer = nn.BatchNorm2d
        if freeze_bn:
            bn_layer = FrozenBatchNorm2d
        backbone = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=[False, True, True], pretrained_weights=pretrained_weights, norm_layer=bn_layer)
        return_layers = {'layer4': 'out'}
        if aux:
            return_layers['layer3'] = 'aux'
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, x):
        out = self.backbone(x)['out']
        return out