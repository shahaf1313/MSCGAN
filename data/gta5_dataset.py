import os.path as osp
from data.domainAdaptationDataset import domainAdaptationDataSet
from PIL import Image
import numpy as np
from core.functions import concat_pyramid, RGBImageToNumpy, ImageToNumpy
import torch

from core.constants import IMG_RESIZE

class GTA5DataSet(domainAdaptationDataSet):
    def __init__(self, root, images_list_path, scale_factor, num_scales, curr_scale, set, pyramid_generators, get_image_label=False, get_image_label_pyramid=False):
        super(GTA5DataSet, self).__init__(root, images_list_path, scale_factor, num_scales, curr_scale, set, pyramid_generators, get_image_label=get_image_label)
        self.resize = IMG_RESIZE
        self.get_image_label_pyramid = get_image_label_pyramid
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]
        image = Image.open(osp.join(self.root, "images/%s" % name)).convert('RGB')
        image = image.resize(self.resize, Image.BICUBIC)
        left = self.resize[0]-self.crop_size[0]
        upper= self.resize[1]-self.crop_size[1]
        left = np.random.randint(0, high=left)
        upper= np.random.randint(0, high=upper)
        right= left + self.crop_size[0]
        lower= upper+ self.crop_size[1]
        image = image.crop((left, upper, right, lower))

        label, label_copy, labels_pyramid = None, None, None
        if self.get_image_label: # or self.get_image_label_pyramid:
            label = Image.open(osp.join(self.root, "labels/%s" % name))
            label = label.resize(self.resize, Image.NEAREST)
            label = label.crop((left, upper, right, lower))
            # if self.get_image_label_pyramid:
            #     labels_pyramid =  self.GeneratePyramid(label, is_label=True)
            #     labels_pyramid = [self.convert_to_class_ids(label_scale) for label_scale in labels_pyramid]
            # else:
            label = self.convert_to_class_ids(label)


        scales_pyramid = self.GeneratePyramid(image)
        curr_image = scales_pyramid[-1]
        prev_image = concat_pyramid(self.Gs, [s.unsqueeze(0) for s in scales_pyramid], self.scale_factor)
        prev_image = prev_image.squeeze(0)
        if self.get_image_label:
            label = torch.tensor(ImageToNumpy(label))
            return curr_image, prev_image, label
        # elif self.get_image_label_pyramid:
        #     return scales_pyramid, labels_pyramid
        else:
            return curr_image, prev_image

    def convert_to_class_ids(self, label_image):
        label = np.asarray(label_image, np.float32)
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy
