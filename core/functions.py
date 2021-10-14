import torch
import numpy as np
import torch.nn as nn
import math
from skimage import io as img
from skimage import color
from PIL import Image
import os
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

def imresize_torch(image_batch, scale, mode):
    new_size = np.ceil(scale * np.array([image_batch.shape[2], image_batch.shape[3]])).astype(np.int)
    if mode=='bicubic':
        return nn.functional.interpolate(image_batch, size=(new_size[0], new_size[1]), mode=mode, align_corners=True)
    else:
        return nn.functional.interpolate(image_batch, size=(new_size[0], new_size[1]), mode=mode)

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates
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


def save_networks(path, netDst, netGst, netGts, Gst, Gts, Dst, opt, semseg_cs=None):
    # if not opt.debug_run:
        try:
            os.makedirs(path)
        except OSError:
            pass
        if len(opt.gpus) > 1:
            torch.save(Dst + [netDst.module], '%s/Dst.pth' % (path))
            torch.save(Gst + [netGst.module], '%s/Gst.pth' % (path))
            torch.save(Gts + [netGts.module], '%s/Gts.pth' % (path))
            if semseg_cs != None:
                torch.save(semseg_cs.module, '%s/semseg_cs.pth' % (path))
        else:
            torch.save(Dst + [netDst], '%s/Dst.pth' % (path))
            torch.save(Gst + [netGst], '%s/Gst.pth' % (path))
            torch.save(Gts + [netGts], '%s/Gts.pth' % (path))
            if semseg_cs != None:
                torch.save(semseg_cs, '%s/semseg_cs.pth' % (path))

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

def GeneratePyramid(image, num_scales, curr_scale, scale_factor, crop_size, is_label=False):
    scales_pyramid = []
    if isinstance(image, Image.Image):
        for i in range(0, curr_scale + 1, 1):
            scale = math.pow(scale_factor, num_scales - i)
            curr_size = (np.ceil(scale * np.array(crop_size))).astype(np.int)
            curr_scale_image = image.resize(curr_size, Image.BICUBIC if not is_label else Image.NEAREST)
            curr_scale_image = RGBImageToNumpy(curr_scale_image) if not is_label else ImageToNumpy(curr_scale_image)
            scales_pyramid.append(curr_scale_image)
    elif isinstance(image, torch.Tensor):
        for i in range(0, curr_scale + 1, 1):
            if is_label:
                scale = math.pow(scale_factor, num_scales - i)
            else:
                scale = math.pow(scale_factor, curr_scale - i)
            curr_scale_image = imresize_torch(image, scale, mode='nearest' if is_label else 'bicubic')
            curr_scale_image = curr_scale_image.squeeze(0) if is_label else curr_scale_image
            scales_pyramid.append(curr_scale_image)
    return scales_pyramid

def RGBImageToNumpy(im):
    im = ImageToNumpy(im)
    im = (im - 128.) / 128  # change from 0..255 to -1..1
    return im

def ImageToNumpy(im):
    im = np.asarray(im, np.float32)
    if len(im.shape) == 3:
        im = np.transpose(im, (2, 0, 1))
    return im

def encode_semseg_out(input):
    max_classes = (input == input.max(dim=1, keepdim=True)[0]) * input
    return max_classes

def encode_semseg_out_with_ignore_label(input, threshold):
    softmax_input = nn.functional.softmax(input, dim=1)
    non_ignore_classes = (softmax_input == softmax_input.max(dim=1, keepdim=True)[0]) * input
    max_soft_values = softmax_input.max(dim=1)[0]
    ignore_class = torch.zeros_like(max_soft_values)
    ignore_class[max_soft_values < threshold] = 1
    non_ignore_classes[softmax_input < threshold] = 0
    return torch.cat((non_ignore_classes, ignore_class.unsqueeze(1)), dim=1)

def strongly_trusted_labels(input, threshold, ignore_label=IGNORE_LABEL):
    softmax_input = nn.functional.softmax(input, dim=1)
    softmax_input[softmax_input < threshold] = 0.
    softmax_max_values = softmax_input.max(dim=1)[0]
    strong_labels = softmax_input.argmax(dim=1)
    strong_labels[softmax_max_values == 0.] = ignore_label
    return strong_labels

def one_hot_encoder_ignore_class(input, num_classes=NUM_CLASSES, ignore_label=IGNORE_LABEL):
    z = input.clone()
    z[z==ignore_label] = num_classes
    output = nn.functional.one_hot(z.type(torch.int64), num_classes+1).permute(0,3,1,2).type(torch.float32)
    return output

def one_hot_encoder(input, num_classes=NUM_CLASSES, ignore_label=IGNORE_LABEL):
    z = input.clone()
    z[input == ignore_label] = num_classes
    z = z.squeeze(1)
    output = nn.functional.one_hot(z.type(torch.int64), num_classes+1).permute(0,3,1,2).type(torch.float32)
    return output[:,:num_classes,:,:]

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
            ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

