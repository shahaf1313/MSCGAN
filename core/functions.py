import torch
import numpy as np
import torch.nn as nn
import math
from skimage import io as img
from skimage import color
from PIL import Image
from core.constants import palette, NUM_CLASSES, IGNORE_LABEL


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)

def reset_grads(model, require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model


def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t


def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t

def imresize_torch(image_batch, scale):
    new_size = np.ceil(scale * np.array([image_batch.shape[2], image_batch.shape[3]])).astype(np.int)
    return nn.functional.interpolate(image_batch, size=(new_size[0], new_size[1]), mode='bicubic')

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    # disc_interpolates.size()),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    # LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def np2torch(x, opt):
    if opt.nc_im == 3:
        x = x[:, :, :, None]
        x = x.transpose((3, 2, 0, 1)) / 255
    else:
        x = color.rgb2gray(x)
        x = x[:, :, None, None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if not (opt.not_cuda):
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not (opt.not_cuda) else x.type(torch.FloatTensor)
    # x = x.type(torch.FloatTensor)
    x = norm(x)
    return x


def torch2uint8(x):
    x = x[0, :, :, :]
    x = x.permute((1, 2, 0))
    x = 255 * denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x


def read_image2np(opt):
    x = img.imread('%s/%s' % (opt.input_dir, opt.input_name))
    x = x[:, :, 0:3]
    return x


def save_networks(netDst, netGst, netDts, netGts, Gst, Gts, Dst, Dts, opt):
    if not opt.debug_run:
        torch.save(Dst + [netDst], '%s/Dst.pth' % (opt.outf))
        torch.save(Gst + [netGst], '%s/Gst.pth' % (opt.outf))
        torch.save(Dts + [netDts], '%s/Dts.pth' % (opt.outf))
        torch.save(Gts + [netGts], '%s/Gts.pth' % (opt.outf))

def colorize_mask(mask):
    # mask: tensor of the mask
    # returns: numpy array of the colorized mask
    new_mask = Image.fromarray(mask.cpu().numpy().astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    new_mask = np.array(new_mask.convert('RGB')).transpose((2, 0, 1))
    return new_mask

def nanmean_torch(x):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum()
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum()
    return value / num

def confusion_matrix_torch(y_pred, y_true, num_classes):
    N = num_classes
    y = (N * y_true + y_pred).type(torch.long)
    y = torch.bincount(y)
    if len(y) < N * N:
        y = torch.cat((y, torch.zeros(N * N - len(y), dtype=torch.long).cuda()))
    y = y.reshape(N, N)
    return y

def compute_cm_batch_torch(y_pred, y_true, ignore_label, classes):
    batch_size = y_pred.shape[0]
    confusion_matrix = torch.zeros((classes, classes)).cuda()
    for i in range(batch_size):
        y_pred_curr = y_pred[i, :, :]
        y_true_curr = y_true[i, :, :]
        inds_to_calc = y_true_curr != ignore_label
        y_pred_curr = y_pred_curr[inds_to_calc]
        y_true_curr = y_true_curr[inds_to_calc]
        assert y_pred_curr.shape == y_true_curr.shape
        confusion_matrix += confusion_matrix_torch(y_pred_curr, y_true_curr, classes)
    return confusion_matrix


def compute_iou_torch(confusion_matrix):
    intersection = torch.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(dim=1)
    predicted_set = confusion_matrix.sum(dim=0)
    union = ground_truth_set + predicted_set - intersection
    iou = intersection / union.type(torch.float32)
    miou = nanmean_torch(iou)
    return iou, miou

def GeneratePyramid(image, num_scales, curr_scale, scale_factor, crop_size):
    scales_pyramid = []
    if isinstance(image, Image.Image):
        for i in range(0, curr_scale + 1, 1):
            scale = math.pow(scale_factor, num_scales - i)
            curr_size = (np.ceil(scale * np.array(crop_size))).astype(np.int)
            curr_scale_image = image.resize(curr_size, Image.BICUBIC)
            curr_scale_image = RGBImageToNumpy(curr_scale_image)
            scales_pyramid.append(curr_scale_image)
    elif isinstance(image, torch.Tensor):
        for i in range(0, curr_scale + 1, 1):
            scale = math.pow(scale_factor, curr_scale - i)
            curr_scale_image = imresize_torch(image, scale)
            scales_pyramid.append(curr_scale_image)
    return scales_pyramid



def RGBImageToNumpy(im):
    im = np.asarray(im, np.float32)
    im = np.transpose(im, (2, 0, 1))
    im = (im - 128.) / 128  # change from 0..255 to -1..1
    return im

def one_hot_encoder(input, num_classes=NUM_CLASSES, ignore_label=IGNORE_LABEL):
    z = input.clone()
    z[z==ignore_label] = num_classes
    output = nn.functional.one_hot(z.type(torch.int64), num_classes+1).permute(0,3,1,2).type(torch.float32)
    return output