import numpy as np
from data.domainAdaptationDataset import domainAdaptationDataSet
from core.functions import RGBImageToNumpy
import os.path as osp
from PIL import Image
from core.constants import DATASETS_IMG_SIZE

class cityscapesDataSet(domainAdaptationDataSet):
    def __init__(self, root, images_list_path, scale_factor, num_scales, curr_scale, set, get_image_label=False, get_scales_pyramid=False):
        super(cityscapesDataSet, self).__init__(root, images_list_path, scale_factor, num_scales, curr_scale, set, get_image_label=get_image_label)
        self.get_scales_pyramid= get_scales_pyramid
        self.source_image_size = DATASETS_IMG_SIZE['cityscapes']
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
    def __getitem__(self, index):
        name = self.img_ids[index]
        image = Image.open(osp.join(   self.root, "leftImg8bit/%s/%s" % (self.set, name)   )).convert('RGB')
        # image = image.resize(self.crop_size, Image.BICUBIC)
        if self.set == 'val' or self.set == 'test':
            pass
        else:
            left = self.source_image_size[0]-self.crop_size[0]
            upper= self.source_image_size[1]-self.crop_size[1]
            left = np.random.randint(0, high=left)
            upper= np.random.randint(0, high=upper)
            right= left + self.crop_size[0]
            lower= upper+ self.crop_size[1]
            image = image.crop((left, upper, right, lower))

        scales_pyramid, label, label_copy = None, None, None
        if self.get_image_label:
            lbname = name.replace("leftImg8bit", "gtFine_labelIds")
            label = Image.open(osp.join(   self.root, "gtFine/%s/%s" % (self.set, lbname)   ))
            # label = label.resize( self.crop_size, Image.NEAREST )
            if self.set == 'val' or self.set == 'test':
                pass
            else:
                label = label.crop((left, upper, right, lower))
            assert image.size == label.size
            label = np.asarray(label, np.float32)
            label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v

        scales_pyramid = None
        if self.get_scales_pyramid:
            scales_pyramid = self.GeneratePyramid(image)
        else:
            image = RGBImageToNumpy(image)

        if self.get_scales_pyramid and self.get_image_label:
            return scales_pyramid, label_copy.copy()
        elif self.get_scales_pyramid and not self.get_image_label:
            return scales_pyramid
        elif not self.get_scales_pyramid and self.get_image_label:
            return image.copy(), label_copy.copy()
        elif not self.get_scales_pyramid and not self.get_image_label:
            return image.copy()




