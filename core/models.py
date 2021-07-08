import torch
import torch.nn as nn
from collections import OrderedDict
from core.constants import NUM_CLASSES


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride, norm_type):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd))
        if norm_type == 'batch_norm':
            self.add_module('norm',nn.BatchNorm2d(out_channel))
        elif norm_type == 'instance_norm':
            self.add_module('norm',nn.InstanceNorm2d(out_channel,affine=True))
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

class ConvBlockSpade(nn.Module):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride, norm_type, activation='lrelu'):
        super(ConvBlockSpade, self).__init__()
        # self._warmup = True
        self.spade = SPADE(norm_type, ker_size, in_channel, label_nc=NUM_CLASSES+1) #+1 for don't care label
        self.bn = nn.BatchNorm2d(in_channel)
        self.conv = nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)
        if activation=='lrelu':
            self.actvn = nn.LeakyReLU(0.2)
        elif activation=='tanh':
            self.actvn = nn.Tanh()

    def forward(self, x, seg_map=None):
        if seg_map==None:
            z = self.conv(self.actvn(self.bn(x)))
        else:
            z = self.conv(self.actvn(self.spade(x, seg_map)))
        return z

    # @property
    # def warmup(self):
    #     return self._warmup
    #
    # @warmup.setter
    # def warmup(self, val):
    #     self._warmup = val

class LabelConditionedGenerator(nn.Module):
    def __init__(self, opt):
        super(LabelConditionedGenerator, self).__init__()
        self.is_initial_scale = opt.curr_scale == 0
        # self._warmup = True
        alpha = 1 if self.is_initial_scale else 0
        # self.head = nn.Sequential(OrderedDict([('head_block',ConvBlockSpade((2-alpha)*opt.nc_im, opt.base_channels, opt.ker_size, padd=1, stride=1, norm_type=opt.norm_type))]))
        self.head = ConvBlockSpade((2-alpha)*opt.nc_im, opt.base_channels, opt.ker_size, padd=1, stride=1, norm_type=opt.norm_type)
        self.body_1 = ConvBlockSpade(opt.base_channels, opt.base_channels, opt.ker_size, padd=1, stride=1,  norm_type=opt.norm_type)
        self.body_2 = ConvBlockSpade(opt.base_channels, opt.base_channels, opt.ker_size, padd=1, stride=1,  norm_type=opt.norm_type)
        self.body_3 = ConvBlockSpade(opt.base_channels, opt.base_channels, opt.ker_size, padd=1, stride=1,  norm_type=opt.norm_type)

        # for i in range(opt.num_layer-2):
        #     block = ConvBlockSpade(opt.base_channels, opt.base_channels, opt.ker_size, padd=1, stride=1,  norm_type=opt.norm_type)
        #     self.body.append(block)
        self.tail = ConvBlockSpade(opt.base_channels, opt.nc_im, opt.ker_size, padd=1, stride=1,  norm_type=opt.norm_type)
        self.tail_actvn = nn.Tanh()
    def forward(self, curr_scale, prev_scale, seg_map=None):
        if self.is_initial_scale:
            z = curr_scale
        else:
            z = torch.cat((curr_scale, prev_scale), 1)

        z = self.head(z, seg_map)
        # for block in self.body:
        #     z = block(z, seg_map)
        z = self.body_1(z, seg_map)
        z = self.body_2(z, seg_map)
        z = self.body_3(z, seg_map)
        z = self.tail(z, seg_map)
        z = self.tail_actvn(z)
        return z

    # @property
    # def warmup(self):
    #     return self._warmup
    #
    # @warmup.setter
    # def warmup(self, val):
    #     self.head.warmup = val
    #     for block in self.body:
    #         block.warmup = val
    #     self.tail.warmup = val

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
    def forward(self, curr_scale, prev_scale):
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

class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.head = ConvBlock(opt.nc_im, opt.base_channels, opt.ker_size, padd=1, stride=1, norm_type=opt.norm_type)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            block = ConvBlock(opt.base_channels, opt.base_channels, opt.ker_size, padd=1, stride=1, norm_type=opt.norm_type)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(nn.Conv2d(opt.base_channels,1,kernel_size=opt.ker_size,stride=1,padding=1),
                                  nn.BatchNorm2d(1), #todo: I added, see what is happening
                                  nn.LeakyReLU(0.2))

    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        # #todo: I Added! see if neccessairy!
        # x = torch.tanh(x)
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        if m.affine == True:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

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
class SPADE(nn.Module):
    def __init__(self, param_free_norm_type, kernel_size, norm_nc, label_nc):
        super().__init__()

        if param_free_norm_type == 'instance_norm':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        # todo: add syncbatch after I will implement DDP:
        # elif param_free_norm_type == 'syncbatch':
        #     self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch_norm':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

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

        return out
