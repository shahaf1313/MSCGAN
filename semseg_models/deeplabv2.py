import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch
from core.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from core.constants import IGNORE_LABEL
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
affine_par = True

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out

class SEBlock(nn.Module):
    def __init__(self, inplanes, r = 16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.se = nn.Sequential(
            nn.Linear(inplanes, inplanes//r),
            nn.ReLU(inplace=True),
            nn.Linear(inplanes//r, inplanes),
            nn.Sigmoid()
        )
    def forward(self, x):
        xx = self.global_pool(x)
        xx = xx.view(xx.size(0), xx.size(1))
        se_weight = self.se(xx).unsqueeze(-1).unsqueeze(-1)
        return x.mul(se_weight)

class Classifier_Module2(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes, droprate = 0.1, use_se = True):
        super(Classifier_Module2, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.conv2d_list.append(
            nn.Sequential(*[
                nn.Conv2d(inplanes, 256, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.GroupNorm(num_groups=32, num_channels=256, affine = True),
                nn.ReLU(inplace=True) ]))

        for dilation, padding in zip(dilation_series, padding_series):
            #self.conv2d_list.append(
            #    nn.BatchNorm2d(inplanes))
            self.conv2d_list.append(
                nn.Sequential(*[
                    #nn.ReflectionPad2d(padding),
                    nn.Conv2d(inplanes, 256, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True),
                    nn.GroupNorm(num_groups=32, num_channels=256, affine = True),
                    nn.ReLU(inplace=True) ]))

        if use_se:
            self.bottleneck = nn.Sequential(*[SEBlock(256 * (len(dilation_series) + 1)),
                                              nn.Conv2d(256 * (len(dilation_series) + 1), 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) ,
                                              nn.GroupNorm(num_groups=32, num_channels=256, affine = True) ])
        else:
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(256 * (len(dilation_series) + 1), 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) ,
                nn.GroupNorm(num_groups=32, num_channels=256, affine = True) ])

        self.head = nn.Sequential(*[nn.Dropout2d(droprate),
                                    nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=False) ])

        ##########init#######
        for m in self.conv2d_list:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.bottleneck:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.head:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)

    def forward(self, x, get_feat=False):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out = torch.cat( (out, self.conv2d_list[i+1](x)), 1)
        out = self.bottleneck(out)
        if get_feat:
            out_dict = {}
            out = self.head[0](out)
            out_dict['feat'] = out
            out = self.head[1](out)
            out_dict['out'] = out
            return out_dict
        else:
            out = self.head(out)
            return out

class ResNet(nn.Module):
    def __init__(
            self,
            name: str,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.layer5 = self._make_pred_layer(Classifier_Module2, 512*block.expansion, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, SynchronizedBatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def _forward_impl(self, x: Tensor, lbl: Tensor = None):
        # See note [TorchScript super()]
        _, _, h, w = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.layer5(x, get_feat=False)
        out = F.interpolate(out, size=(h,w), mode='bilinear', align_corners=True)

        if lbl is not None:
            loss = self.CrossEntropy2d(out, lbl)
            return out, loss
        return out

    def get_1x_lr_params(self):
        b = []
        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        b = []
        b.append(self.layer5.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params(), 'lr': args.lr_semseg},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.lr_semseg}]

    def adjust_learning_rate(self, args, optimizer, i):
        lr = args.lr_semseg * ((1 - float(i) / args.num_steps) ** (args.power))
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10

    def CrossEntropy2d(self, predict, target):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        # todo: Shahaf added
        # target = F.one_hot(target).permute(0,3,1,2)
        loss = self.ce_loss(predict, target.type(torch.long))
        # todo: end
        # original code:
        # n, c, h, w = predict.size()
        # target_mask = (target >= 0) * (target != IGNORE_LABEL)
        # target = target[target_mask]
        # predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        # predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        # loss = F.cross_entropy(predict, target, weight=weight, size_average=size_average)
        return loss


    def forward(self, x: Tensor, lbl: Tensor = None) :
        return self._forward_impl(x, lbl)

#
# class ResNet101(nn.Module):
#     def __init__(self, block, layers, num_classes, BatchNorm, bn_clr=False):
#         self.inplanes = 64
#         self.bn_clr = bn_clr
#         super(ResNet101, self).__init__()
#         self.ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = BatchNorm(64, affine=affine_par)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
#         self.layer1 = self._make_layer(block, 64, layers[0], BatchNorm=BatchNorm)
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2, BatchNorm=BatchNorm)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, BatchNorm=BatchNorm)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, BatchNorm=BatchNorm)
#         #self.layer5 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
#         self.layer5 = self._make_pred_layer(Classifier_Module2, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
#         if self.bn_clr:
#             self.bn_pretrain = BatchNorm(2048, affine=affine_par)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, 0.01)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, SynchronizedBatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 BatchNorm(planes * block.expansion, affine=affine_par))
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, BatchNorm=BatchNorm))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))
#
#         return nn.Sequential(*layers)
#
    # def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
    #     return block(inplanes, dilation_series, padding_series, num_classes)
#
#     def forward(self, x, lbl=None):
#         _, _, h, w = x.size()
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         if self.bn_clr:
#             x = self.bn_pretrain(x)
#
#         out = self.layer5(x, get_feat=False)
#         out = F.interpolate(out, size=(h,w), mode='bilinear', align_corners=True)
#
#         if lbl is not None:
#             loss = self.CrossEntropy2d(out, lbl)
#             return out, loss
#         return out
#
#     def get_1x_lr_params(self):
#
#         b = []
#
#         b.append(self.conv1)
#         b.append(self.bn1)
#         b.append(self.layer1)
#         b.append(self.layer2)
#         b.append(self.layer3)
#         b.append(self.layer4)
#
#         for i in range(len(b)):
#             for j in b[i].modules():
#                 jj = 0
#                 for k in j.parameters():
#                     jj += 1
#                     if k.requires_grad:
#                         yield k
#
#     def get_10x_lr_params(self):
#
#         b = []
#         if self.bn_clr:
#             b.append(self.bn_pretrain.parameters())
#         b.append(self.layer5.parameters())
#
#         for j in range(len(b)):
#             for i in b[j]:
#                 yield i
#
#     def optim_parameters(self, args):
#         return [{'params': self.get_1x_lr_params(), 'lr': args.lr_semseg},
#                 {'params': self.get_10x_lr_params(), 'lr': 10 * args.lr_semseg}]
#
#     def adjust_learning_rate(self, args, optimizer, i):
#         lr = args.lr_semseg * ((1 - float(i) / args.num_steps) ** (args.power))
#         optimizer.param_groups[0]['lr'] = lr
#         if len(optimizer.param_groups) > 1:
#             optimizer.param_groups[1]['lr'] = lr * 10
#
#     def CrossEntropy2d(self, predict, target):
#         assert not target.requires_grad
#         assert predict.dim() == 4
#         assert target.dim() == 3
#         assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
#         assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
#         assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
#         # todo: Shahaf added
#         # target = F.one_hot(target).permute(0,3,1,2)
#         loss = self.ce_loss(predict, target.type(torch.long))
#         # todo: end
#         # original code:
#         # n, c, h, w = predict.size()
#         # target_mask = (target >= 0) * (target != IGNORE_LABEL)
#         # target = target[target_mask]
#         # predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
#         # predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
#         # loss = F.cross_entropy(predict, target, weight=weight, size_average=size_average)
#         return loss

def freeze_bn_func(m):
    if m.__class__.__name__.find('BatchNorm') != -1 or isinstance(m, SynchronizedBatchNorm2d) \
            or isinstance(m, nn.BatchNorm2d):
        m.weight.requires_grad = False
        m.bias.requires_grad = False

def DeeplabV2(backbone, BatchNorm, num_classes=21, freeze_bn=False):
    print('Architecture of backbone: ' + backbone)
    if backbone == 'resnet101':
        model = ResNet(backbone, Bottleneck, [3, 4, 23, 3], num_classes, BatchNorm)
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
    elif backbone == 'resnet50':
        model = ResNet(backbone, Bottleneck, [3, 4, 6, 3], num_classes, BatchNorm)
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
    elif backbone == 'resnet18':
        model = ResNet(backbone, BasicBlock, [2, 2, 2, 2], num_classes, BatchNorm)
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
    else:
        raise NotImplemented()

    if freeze_bn:
        model.apply(freeze_bn_func)

    model_dict = {}
    state_dict = model.state_dict()
    for k, v in pretrain_dict.items():
        if k in state_dict:
            model_dict[k] = v
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)

    return model
