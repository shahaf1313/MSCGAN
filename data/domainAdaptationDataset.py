import numpy as np
from constants import IGNORE_LABEL, IMG_CROP_SIZE
from PIL import Image
import math
import os.path as osp
from torch.utils import data
from torchvision import transforms

class domainAdaptationDataSet(data.Dataset):
    def __init__(self, root, list_path, scale_factor, num_scales, curr_scale, set, get_image_label=False):
        self.root = root
        if list_path != None:
            self.list_path = osp.join(list_path, '%s.txt' % set)
            self.img_ids = [i_id.strip() for i_id in open(self.list_path)]
        self.scale_factor = scale_factor
        self.num_scales = num_scales
        self. curr_scale = curr_scale
        self.set = set
        self.trans = transforms.ToTensor()
        self.crop_size = IMG_CROP_SIZE
        self.ignore_label = IGNORE_LABEL
        self.get_image_label = get_image_label
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

    def __len__(self):
        return len(self.img_ids)

    def SetEpochSize(self, epoch_size):
        if (epoch_size > len(self.img_ids)):
            self.img_ids = self.img_ids * int(np.ceil(float(epoch_size) / len(self.img_ids)))
        self.img_ids = self.img_ids[:epoch_size]

    def RGBImageToNumpy(self, im):
        im = np.asarray(im, np.float32)
        im = np.transpose(im, (2, 0, 1))
        im = (im - 128.) / 128  # change from 0..255 to -1..1
        return im

    def GeneratePyramid(self, image: Image):
        scales_pyramid = []
        for i in range(0, self.curr_scale + 1, 1):
            scale = math.pow(self.scale_factor, self.num_scales - i)
            curr_size = (np.ceil(scale * np.array(self.crop_size))).astype(np.int)
            curr_scale = image.resize(curr_size, Image.BICUBIC)
            curr_scale = self.RGBImageToNumpy(curr_scale)
            scales_pyramid.append(curr_scale)

        return scales_pyramid

