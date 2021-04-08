import os.path as osp
from data.domainAdaptationDataset import domainAdaptationDataSet
from PIL import Image
import numpy as np
from constants import IMG_RESIZE

class GTA5DataSet(domainAdaptationDataSet):
    def __init__(self, root, list_path, scale_factor, num_scales, curr_scale, set, get_image_label=False):
        super(GTA5DataSet, self).__init__(root, list_path, scale_factor, num_scales, curr_scale, set, get_image_label=get_image_label)
        self.resize = IMG_RESIZE
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

        label, label_copy = None, None
        if self.get_image_label:
            label = Image.open(osp.join(self.root, "labels/%s" % name))
            label = label.resize(self.resize, Image.NEAREST)
            label = label.crop((left, upper, right, lower))
            label = np.asarray(label, np.float32)
            label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
            label_copy = label_copy.copy()

        scales_pyramid = self.GeneratePyramid(image)
        if self.get_image_label:
            return scales_pyramid, label_copy
        else:
            return scales_pyramid

    def SetEpochSize(self, epoch_size):
        if (epoch_size > len(self.img_ids)):
            self.img_ids = self.img_ids * int(np.ceil(float(epoch_size) / len(self.img_ids)))
        self.img_ids = self.img_ids[:epoch_size]
