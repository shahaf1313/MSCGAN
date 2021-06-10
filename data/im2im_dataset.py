import os.path as osp
from data.domainAdaptationDataset import domainAdaptationDataSet
from PIL import Image
from core.constants import IMG_CROP_SIZE_IM2IM
from os import walk


class Im2imDataset(domainAdaptationDataSet):
    def __init__(self, root, scale_factor, num_scales, curr_scale, set):
        super(Im2imDataset, self).__init__(root, None, scale_factor, num_scales, curr_scale, set, get_image_label=False)
        _, _, filenames = next(walk(self.root))
        self.img_ids = filenames
        self.crop_size = IMG_CROP_SIZE_IM2IM

    def __getitem__(self, index):
        image = Image.open(osp.join(self.root, self.img_ids[index])).convert('RGB')
        scales_pyramid = self.GeneratePyramid(image)
        return scales_pyramid

