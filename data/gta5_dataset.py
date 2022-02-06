import os.path as osp
from data.domainAdaptationDataset import domainAdaptationDataSet
from PIL import Image
import numpy as np
from core.constants import IMG_RESIZE

class GTA5DataSet(domainAdaptationDataSet):
    def __init__(self, root, images_list_path, scale_factor, num_scales, curr_scale, set, get_image_label=False, get_image_label_pyramid=False, get_filename=False):
        super(GTA5DataSet, self).__init__(root, images_list_path, scale_factor, num_scales, curr_scale, set, get_image_label=get_image_label)
        self.resize = IMG_RESIZE
        self.get_image_label_pyramid = get_image_label_pyramid
        self.get_filename = get_filename
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
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
        if self.get_image_label or self.get_image_label_pyramid:
            label = Image.open(osp.join(self.root, "labels/%s" % name))
            label = label.resize(self.resize, Image.NEAREST)
            label = label.crop((left, upper, right, lower))
            if self.get_image_label_pyramid:
                labels_pyramid =  self.GeneratePyramid(label, is_label=True)
                labels_pyramid = [self.convert_to_class_ids(label_scale) for label_scale in labels_pyramid]
            else:
                label = self.convert_to_class_ids(label)

        scales_pyramid = self.GeneratePyramid(image)
        if self.get_image_label:
            return scales_pyramid, label
        elif self.get_image_label_pyramid:
            return scales_pyramid, labels_pyramid
        else:
            return scales_pyramid if not self.get_filename else scales_pyramid, self.img_ids[index]
