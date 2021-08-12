import torch
import torch.nn as nn
from core.constants import NUM_CLASSES, H, W
import numpy as np
import torchvision


# Blocks:
class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride, norm_type):
        super(ConvBlock,self).__init__()
        if norm_type == 'batch_norm':
            self.add_module('norm',nn.BatchNorm2d(in_channel))
        elif norm_type == 'instance_norm':
            self.add_module('norm',nn.InstanceNorm2d(in_channel,affine=True))
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd))

class CondConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride, norm_type, use_rad, activation='lrelu'):
        super(CondConvBlock, self).__init__()
        # Normalization:
        self.bn = nn.BatchNorm2d(in_channel)
        if use_rad:
            self.cond_norm = RAD(ker_size, in_channel, label_nc=3)
        else:
            self.cond_norm = SPADE(norm_type, ker_size, in_channel, label_nc=3)
        # Activation:
        if activation=='lrelu':
            self.actvn = nn.LeakyReLU(0.2)
        elif activation=='tanh':
            self.actvn = nn.Tanh()
        # Convolution:
        self.conv = nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)

    def forward(self, input_image, cond_image=None):
            if cond_image==None:
                z = self.conv(self.actvn(self.bn(input_image)))
            else:
                z = self.conv(self.actvn(self.cond_norm(input_image, cond_image)))
            return z

class SPADE(nn.Module):
    # # Creates SPADE normalization layer based on the given configuration
    # SPADE consists of two steps. First, it normalizes the activations using
    # your favorite normalization method, such as Batch Norm or Instance Norm.
    # Second, it applies scale and bias to the normalized output, conditioned on
    # the segmentation map.
    # The format of |config_text| is spade(norm)(ks), where
    # (norm) specifies the type of parameter-free normalization.
    #       (e.g. syncbatch, batch, instance)
    # (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
    # Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
    # Also, the other arguments are
    # |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
    # |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
    def __init__(self, param_free_norm_type, kernel_size, norm_nc, label_nc):
        super().__init__()
        if param_free_norm_type == 'instance_norm':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch_norm':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        # nhidden = 64

        pw = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=kernel_size, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = nn.functional.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        # out = x * (1 + gamma) + beta

        return out

class RAD(nn.Module):
    def __init__(self, kernel_size, norm_nc, label_nc):
        super(RAD, self).__init__()
        num_groups = int(1 if norm_nc<10 else norm_nc/4)
        self.param_free_norm = nn.GroupNorm(num_groups=num_groups, num_channels=norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        # nhidden = 128
        nhidden = 32

        pw = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            ResBlockRAD(label_nc, nhidden, kernel_size, pw),
            # ResBlockRAD(nhidden, nhidden, kernel_size, pw),
            # ResBlockRAD(nhidden, nhidden, kernel_size, pw)
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = nn.functional.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        # out = x * (1 + gamma) + beta

        return out

class ResBlockRAD(nn.Module):
    def __init__(self, in_channels, out_channels, ker_size, padd, activation='relu'):
        super(ResBlockRAD, self).__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Sequential(nn.Conv2d(in_channels, out_channels, ker_size, padding=padd),
                                    nn.LeakyReLU(0.2),
                                    nn.Conv2d(out_channels, out_channels, ker_size, padding=padd))
        self.activate = self.activation_func(activation)
        self.shortcut = nn.Conv2d(in_channels, out_channels, ker_size, padding=padd)

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

    def activation_func(self, activation):
        return  nn.ModuleDict([
            ['relu', nn.ReLU(inplace=True)],
            ['leaky_relu', nn.LeakyReLU(negative_slope=0.2, inplace=True)],
            ['selu', nn.SELU(inplace=True)],
            ['none', nn.Identity()]
        ])[activation]

class LocalNet(nn.Module):

    def forward(self, x_in):
        """Double convolutional block

        :param x_in: image features
        :returns: image features
        :rtype: Tensor

        """
        x = self.lrelu(self.conv1(self.refpad(x_in)))
        x = self.lrelu(self.conv2(self.refpad(x)))

        return x

    def __init__(self, in_channels=16, out_channels=64):
        """Double convolutional block

        :param in_channels:  number of input channels
        :param out_channels: number of output channels
        :returns: N/A
        :rtype: N/A

        """
        super(LocalNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 0, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 0, 1)
        self.lrelu = nn.LeakyReLU()
        self.refpad = nn.ReflectionPad2d(1)


# Generators:
class ConvGenerator(nn.Module):
    def __init__(self, opt):
        super(ConvGenerator, self).__init__()
        self.is_initial_scale = opt.curr_scale == 0
        alpha = 1 if self.is_initial_scale else 0
        self.head = ConvBlock((2-alpha)*opt.nc_im, opt.base_channels, opt.ker_size, padd=1, stride=1, norm_type=opt.norm_type)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            block = ConvBlock(opt.base_channels, opt.base_channels, opt.ker_size, padd=1, stride=1,  norm_type=opt.norm_type)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(
            nn.Conv2d(opt.base_channels, opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=1),
            nn.BatchNorm2d(opt.nc_im), #todo: I added, see what is happening
            nn.Tanh()
        )
    def forward(self, curr_scale, prev_scale, label=None):
        if self.is_initial_scale:
            z = curr_scale
        else:
            z = torch.cat((curr_scale, prev_scale), 1)
        z = self.head(z)
        z = self.body(z)
        z = self.tail(z)
        # ind = int((prev_scale.shape[2]-curr_scale.shape[2])/2)
        # z = z[:,:,ind:(prev_scale.shape[2]-ind),ind:(curr_scale.shape[3]-ind)]
        return z

class ConditionedGenerator(nn.Module):
    def __init__(self, opt):
        super(ConditionedGenerator, self).__init__()
        self.is_initial_scale = opt.curr_scale == 0
        self.use_extended_generator = opt.curr_scale >= opt.num_scales - 1
        self.head =     CondConvBlock((2 - self.is_initial_scale) * opt.nc_im, opt.base_channels, opt.ker_size, use_rad=opt.use_rad, padd=1, stride=1, norm_type=opt.norm_type)
        self.body_1 =   CondConvBlock(opt.base_channels, opt.base_channels, opt.ker_size, use_rad=opt.use_rad, padd=1, stride=1, norm_type=opt.norm_type)
        self.body_2 =   CondConvBlock(opt.base_channels, opt.base_channels, opt.ker_size, use_rad=opt.use_rad, padd=1, stride=1, norm_type=opt.norm_type)
        self.body_3 =   CondConvBlock(opt.base_channels, opt.base_channels, opt.ker_size, use_rad=opt.use_rad, padd=1, stride=1, norm_type=opt.norm_type)
        self.body_4 =   CondConvBlock(opt.base_channels, opt.base_channels, opt.ker_size, use_rad=opt.use_rad, padd=1, stride=1, norm_type=opt.norm_type)
        self.body_5 =   CondConvBlock(opt.base_channels, opt.base_channels, opt.ker_size, use_rad=opt.use_rad, padd=1, stride=1, norm_type=opt.norm_type)
        self.tail =     CondConvBlock(opt.base_channels, opt.nc_im, opt.ker_size, use_rad=opt.use_rad, padd=1, stride=1, norm_type=opt.norm_type)
        self.tail_actvn = nn.Tanh()
    def forward(self, source, source_prev, target):
        if self.is_initial_scale:
            z = source
        else:
            z = torch.cat((source, source_prev), 1)

        z = self.head(z, target)
        z = self.body_1(z, target)
        z = self.body_2(z, target)
        z = self.body_3(z, target)
        if self.use_extended_generator:
            z = self.body_4(z, target)
            z = self.body_5(z, target)
        z = self.tail(z, target)
        z = self.tail_actvn(z)
        return z

class UNetGeneratorFourLayers(nn.Module):

    def __init__(self, opt):

        super().__init__()
        self.is_initial_scale = opt.curr_scale == 0

        self.dconv_down1 = LocalNet(3 if self.is_initial_scale else 6, 16)
        self.dconv_down2 = LocalNet(16, 32)
        self.dconv_down3 = LocalNet(32, 64)
        self.dconv_down4 = LocalNet(64, 128)
        self.dconv_down5 = LocalNet(128, 128)

        self.maxpool = nn.MaxPool2d(2, padding=0)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.up_conv1x1_1 = nn.Conv2d(128, 128, 1)
        self.up_conv1x1_2 = nn.Conv2d(128, 128, 1)
        self.up_conv1x1_3 = nn.Conv2d(64, 64, 1)
        self.up_conv1x1_4 = nn.Conv2d(32, 32, 1)

        self.dconv_up4 = LocalNet(256, 128)
        self.dconv_up3 = LocalNet(192, 64)
        self.dconv_up2 = LocalNet(96, 32)
        self.dconv_up1 = LocalNet(48, 16)

        self.conv_last = LocalNet(16, 3)

    def forward(self, curr_scale, prev_scale):
        if self.is_initial_scale:
            x = curr_scale
        else:
            x = torch.cat((curr_scale, prev_scale), 1)

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        x = self.dconv_down5(x)

        x = self.up_conv1x1_1(self.upsample(x))

        if x.shape[3] != conv4.shape[3] and x.shape[2] != conv4.shape[2]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 1))
        elif x.shape[2] != conv4.shape[2]:
            x = torch.nn.functional.pad(x, (0, 0, 0, 1))
        elif x.shape[3] != conv4.shape[3]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 0))

        x = torch.cat([x, conv4], dim=1)

        x = self.dconv_up4(x)
        x = self.up_conv1x1_2(self.upsample(x))

        if x.shape[3] != conv3.shape[3] and x.shape[2] != conv3.shape[2]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 1))
        elif x.shape[2] != conv3.shape[2]:
            x = torch.nn.functional.pad(x, (0, 0, 0, 1))
        elif x.shape[3] != conv3.shape[3]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 0))

        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.up_conv1x1_3(self.upsample(x))

        del conv3

        if x.shape[3] != conv2.shape[3] and x.shape[2] != conv2.shape[2]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 1))
        elif x.shape[2] != conv2.shape[2]:
            x = torch.nn.functional.pad(x, (0, 0, 0, 1))
        elif x.shape[3] != conv2.shape[3]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 0))

        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.up_conv1x1_4(self.upsample(x))

        del conv2

        if x.shape[3] != conv1.shape[3] and x.shape[2] != conv1.shape[2]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 1))
        elif x.shape[2] != conv1.shape[2]:
            x = torch.nn.functional.pad(x, (0, 0, 0, 1))
        elif x.shape[3] != conv1.shape[3]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 0))

        x = torch.cat([x, conv1], dim=1)
        del conv1

        x = self.dconv_up1(x)

        out = self.conv_last(x)
        out = torch.tanh(out)

        return out

class UNetGeneratorTwoLayers(nn.Module):

    def __init__(self, opt):

        super().__init__()
        self.is_initial_scale = opt.curr_scale == 0

        self.dconv_down1 = LocalNet(3 if self.is_initial_scale else 6, 16)
        self.dconv_down2 = LocalNet(16, 32)
        self.dconv_down3 = LocalNet(32, 32)

        self.maxpool = nn.MaxPool2d(2, padding=0)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.up_conv1x1_1 = nn.Conv2d(32, 32, 1)
        self.up_conv1x1_2 = nn.Conv2d(32, 32, 1)

        self.dconv_up2 = LocalNet(64, 32)
        self.dconv_up1 = LocalNet(48, 16)

        self.conv_last = LocalNet(16, 3)

    def forward(self, curr_scale, prev_scale):

        if self.is_initial_scale:
            x = curr_scale
        else:
            x = torch.cat((curr_scale, prev_scale), 1)

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        x = self.dconv_down3(x)

        x = self.up_conv1x1_1(self.upsample(x))

        if x.shape[3] != conv2.shape[3] and x.shape[2] != conv2.shape[2]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 1))
        elif x.shape[2] != conv2.shape[2]:
            x = torch.nn.functional.pad(x, (0, 0, 0, 1))
        elif x.shape[3] != conv2.shape[3]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 0))

        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.up_conv1x1_2(self.upsample(x))

        if x.shape[3] != conv1.shape[3] and x.shape[2] != conv1.shape[2]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 1))
        elif x.shape[2] != conv1.shape[2]:
            x = torch.nn.functional.pad(x, (0, 0, 0, 1))
        elif x.shape[3] != conv1.shape[3]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 0))

        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        del conv1

        out = self.conv_last(x)
        out = torch.tanh(out)

        return out

class FCCGenerator(nn.Module):
    def __init__(self, opt):
        super(FCCGenerator, self).__init__()
        self.is_initial_scale = opt.curr_scale == 0
        self.embedding_features_size = [8, 16]
        self.embedding_channels_size = 64
        self.embedding_size = self.embedding_features_size[0] * self.embedding_features_size[1] * self.embedding_channels_size
        self.scale_size = [int(opt.scale_factor**(opt.num_scales-opt.curr_scale) * H),
                           int(opt.scale_factor**(opt.num_scales-opt.curr_scale) * W)]
        assert H == W/2
        self.scale_factor = np.sqrt(self.embedding_features_size[0] / self.scale_size[0])

        # Down Conv:
        self.conv_1_down = CondConvBlock(opt.nc_im,
                                         self.embedding_channels_size // 4,
                                         opt.ker_size, opt.padd_size, stride=1, norm_type=opt.norm_type)

        self.conv_2_down = CondConvBlock(self.embedding_channels_size // 4,
                                         self.embedding_channels_size // 2,
                                         opt.ker_size, opt.padd_size, stride=1, norm_type=opt.norm_type)

        self.conv_3_down = CondConvBlock(self.embedding_channels_size // 2,
                                         self.embedding_channels_size,
                                         opt.ker_size, opt.padd_size, stride=1, norm_type=opt.norm_type)

        # Up Conv:
        self.conv_1_up = CondConvBlock(self.embedding_channels_size,
                                       self.embedding_channels_size // 2,
                                       opt.ker_size, opt.padd_size, stride=1, norm_type=opt.norm_type)

        self.conv_2_up = CondConvBlock(self.embedding_channels_size // 2,
                                       self.embedding_channels_size // 4,
                                       opt.ker_size, opt.padd_size, stride=1, norm_type=opt.norm_type)

        self.conv_3_up = CondConvBlock(self.embedding_channels_size // 4,
                                       opt.nc_im,
                                       opt.ker_size, opt.padd_size, stride=1, norm_type=opt.norm_type)

        # Down FC:
        self.fc_1_down = nn.Sequential(nn.BatchNorm1d(self.embedding_size),
                                  nn.LeakyReLU(0.2),
                                  nn.Linear(self.embedding_size, self.embedding_size//2**4, bias=True))

        self.fc_2_down = nn.Sequential(nn.BatchNorm1d(self.embedding_size//2**4),
                                  nn.LeakyReLU(0.2),
                                  nn.Linear(self.embedding_size//2**4, self.embedding_size//2**8, bias=True))

        # Up FC:
        self.fc_1_up = nn.Sequential(nn.BatchNorm1d(self.embedding_size//2**8),
                                  nn.LeakyReLU(0.2),
                                  nn.Linear(self.embedding_size//2**8, self.embedding_size//2**4, bias=True))

        self.fc_2_up = nn.Sequential(nn.BatchNorm1d(self.embedding_size//2**4),
                                  nn.LeakyReLU(0.2),
                                  nn.Linear(self.embedding_size//2**4, self.embedding_size, bias=True))

        # Final activation:
        self.final_actvn = nn.Tanh()
    def forward(self, curr_scale, prev_scale, seg_map=None):
        if self.is_initial_scale:
            x = curr_scale
        else:
            x = torch.cat((curr_scale, prev_scale), 1)
        x = self.conv_1_down(x)
        x = self.conv_2_down(x)
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.conv_3_down(x)
        x = nn.functional.interpolate(x, size=self.embedding_features_size, mode='nearest')
        x = x.view((x.shape[0],-1))
        x = self.fc_1_down(x)
        x = self.fc_2_down(x)
        x = self.fc_1_up(x)
        x = self.fc_2_up(x)
        x = x.view((x.shape[0], self.embedding_channels_size, self.embedding_features_size[0], self.embedding_features_size[1] ))
        x = self.conv_1_up(x)
        x = nn.functional.interpolate(x, scale_factor=1./self.scale_factor, mode='nearest')
        x = self.conv_2_up(x)
        x = nn.functional.interpolate(x, size=self.scale_size, mode='nearest')
        x = self.conv_3_up(x)
        x = self.final_actvn(x)
        return x


# Discriminators:
class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        num_layer_in_body = opt.num_layer if opt.curr_scale >= opt.num_scales - 1 else opt.num_layer-2
        self.head = ConvBlock(opt.nc_im, opt.base_channels, opt.ker_size, padd=1, stride=1, norm_type=opt.norm_type)
        self.body = nn.Sequential()
        for i in range(num_layer_in_body):
            block = ConvBlock(opt.base_channels, opt.base_channels, opt.ker_size, padd=1, stride=1, norm_type=opt.norm_type)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(nn.Conv2d(opt.base_channels,1,kernel_size=opt.ker_size,stride=1,padding=1),
                                  nn.BatchNorm2d(1), #todo: I added, see what is happening
                                  nn.LeakyReLU(0.2))

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

class WDiscriminatorDownscale(nn.Module):
    def __init__(self, opt, four_level_discriminator):
        super(WDiscriminatorDownscale, self).__init__()
        self.four_level = four_level_discriminator
        self.l0 = nn.Sequential(
            nn.Conv2d(opt.nc_im, opt.base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(opt.base_channels),
            nn.LeakyReLU()
        )
        self.l1 = nn.Sequential(
            nn.Conv2d(opt.base_channels, 2*opt.base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*opt.base_channels),
            nn.LeakyReLU()
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(2*opt.base_channels, 2*opt.base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*opt.base_channels),
            nn.LeakyReLU()
        )
        if self.four_level:
            self.l3 = nn.Sequential(
                nn.Conv2d(2*opt.base_channels, 4*opt.base_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(4*opt.base_channels),
                nn.LeakyReLU()
            )
            self.l4 = nn.Sequential(
                nn.Conv2d(4*opt.base_channels, 4*opt.base_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(4*opt.base_channels),
                nn.LeakyReLU()
            )
        # self.l5 = nn.Sequential(
        #     nn.Conv2d(opt.base_channels, opt.base_channels, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(opt.base_channels),
        #     nn.LeakyReLU()
        # )

    def forward(self, x):
        x0 = self.l0(x)
        x1 = self.l1(x0)
        x2 = self.l2(x1)
        if self.four_level:
            x3 = self.l3(x2)
            x4 = self.l4(x3)
            # x5 = self.l5(x4)
            return x4
        return  x2

class FCCDiscriminator(nn.Module):
    def __init__(self, opt):
        super(FCCDiscriminator, self).__init__()
        self.embedding_features_size = [8, 16]
        self.embedding_channels_size = 64
        self.embedding_size = self.embedding_features_size[0] * self.embedding_features_size[1] * self.embedding_channels_size
        self.scale_size = [opt.scale_factor**(opt.num_scales-opt.curr_scale) * H,
                           opt.scale_factor**(opt.num_scales-opt.curr_scale) * W]
        assert H == W/2
        self.scale_factor = np.sqrt(self.embedding_features_size[0] / self.scale_size[0])

        self.conv_1 = CondConvBlock(opt.nc_im,
                                    self.embedding_channels_size // 4,
                                    opt.ker_size, opt.padd_size, stride=1, norm_type=opt.norm_type)

        self.conv_2 = CondConvBlock(self.embedding_channels_size // 4,
                                    self.embedding_channels_size // 2,
                                    opt.ker_size, opt.padd_size, stride=1, norm_type=opt.norm_type)

        self.conv_3 = CondConvBlock(self.embedding_channels_size // 2,
                                    self.embedding_channels_size,
                                    opt.ker_size, opt.padd_size, stride=1, norm_type=opt.norm_type)

        self.fc_1 = nn.Sequential(nn.BatchNorm1d(self.embedding_size),
                                  nn.LeakyReLU(0.2),
                                  nn.Linear(self.embedding_size, self.embedding_size//2**4, bias=True))

        self.fc_2 = nn.Sequential(nn.BatchNorm1d(self.embedding_size//2**4),
                                  nn.LeakyReLU(0.2),
                                  nn.Linear(self.embedding_size//2**4, self.embedding_size//2**8, bias=True))

        self.fc_3 = nn.Sequential(nn.BatchNorm1d(self.embedding_size//2**8),
                                  nn.LeakyReLU(0.2),
                                  nn.Linear(self.embedding_size//2**8, 1, bias=True))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.conv_3(x)
        x = nn.functional.interpolate(x, size=self.embedding_features_size, mode='nearest')
        x = x.view((x.shape[0],-1))
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        return x


# Miscellaneous:
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        if m.affine == True:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

# VGG architecture, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i])
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
