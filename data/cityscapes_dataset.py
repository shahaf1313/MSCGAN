import numpy as np
from data.domainAdaptationDataset import domainAdaptationDataSet
from core.functions import RGBImageToNumpy, ImageToNumpy
import os.path as osp
from PIL import Image
from core.functions import concat_pyramid
import torch


class cityscapesDataSet(domainAdaptationDataSet):
    def __init__(self, root, images_list_path, scale_factor, num_scales, curr_scale, set, pyramid_generators, get_image_label=False, generate_prev_image=False):
        super(cityscapesDataSet, self).__init__(root, images_list_path, scale_factor, num_scales, curr_scale, set, pyramid_generators, get_image_label=get_image_label)
        self.generate_prev_image= generate_prev_image

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]
        image = Image.open(osp.join(   self.root, "leftImg8bit/%s/%s" % (self.set, name)   )).convert('RGB')
        image = image.resize(self.crop_size, Image.BICUBIC)

        scales_pyramid, label, label_copy = None, None, None
        if self.get_image_label:
            lbname = name.replace("leftImg8bit", "gtFine_labelIds")
            label = Image.open(osp.join(self.root, "gtFine/%s/%s" % (self.set, lbname)))
            label = label.resize( self.crop_size, Image.NEAREST )
            assert image.size == label.size
            label = np.asarray(label, np.float32)
            label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
            label_copy = torch.tensor(ImageToNumpy(label_copy))

        scales_pyramid, prev_image = None, None
        if self.generate_prev_image:
            scales_pyramid = self.GeneratePyramid(image)
            curr_image = scales_pyramid[-1]
            prev_image = concat_pyramid(self.Gs, [s.unsqueeze(0) for s in scales_pyramid], self.scale_factor)
            prev_image = prev_image.squeeze(0)
        full_image = torch.tensor(RGBImageToNumpy(image))

        if self.generate_prev_image and self.get_image_label:
            return curr_image, prev_image, label_copy
        elif self.generate_prev_image and not self.get_image_label:
            return curr_image, prev_image
        elif not self.generate_prev_image and self.get_image_label:
            return full_image, label_copy
        elif not self.generate_prev_image and not self.get_image_label:
            return full_image




