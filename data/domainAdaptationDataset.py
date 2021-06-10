import numpy as np
from core.constants import IGNORE_LABEL, IMG_CROP_SIZE_SEMSEG
from PIL import Image
from core.functions import GeneratePyramid
import os.path as osp
from torch.utils import data
from torchvision import transforms

class domainAdaptationDataSet(data.Dataset):
    def __init__(self, root, images_list_path, scale_factor, num_scales, curr_scale, set, get_image_label=False):
        self.root = root
        if images_list_path != None:
            self.images_list_file = osp.join(images_list_path, '%s.txt' % set)
            self.img_ids = [image_id.strip() for image_id in open(self.images_list_file)]
        self.scale_factor = scale_factor
        self.num_scales = num_scales
        self. curr_scale = curr_scale
        self.set = set
        self.trans = transforms.ToTensor()
        self.crop_size = IMG_CROP_SIZE_SEMSEG
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


    def GeneratePyramid(self, image: Image):
        scales_pyramid = GeneratePyramid(image, self.num_scales, self.curr_scale, self.scale_factor, self.crop_size)

        return scales_pyramid

