import os.path as osp
from data.domainAdaptationDataset import domainAdaptationDataSet
from PIL import Image
import numpy as np
from core.constants import IMG_RESIZE
from torch.utils.data import DataLoader

class sit_dataset(domainAdaptationDataSet):
    def __init__(self, root, images_list_path, scale_factor, num_scales, curr_scale, set, random_crop=False):
        super(sit_dataset, self).__init__(root, images_list_path, scale_factor, num_scales, curr_scale, set)
        self.resize = IMG_RESIZE
        self.random_crop = random_crop

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]
        image = Image.open(osp.join(self.root, "images/%s" % name)).convert('RGB')
        if self.random_crop:
            image = image.resize(self.resize, Image.BICUBIC)
            left = self.resize[0]-self.crop_size[0]
            upper= self.resize[1]-self.crop_size[1]
            left = np.random.randint(0, high=left)
            upper= np.random.randint(0, high=upper)
            right= left + self.crop_size[0]
            lower= upper+ self.crop_size[1]
            image = image.crop((left, upper, right, lower))
        else:
            image = image.resize(self.crop_size, Image.BICUBIC)

        label = Image.open(osp.join(self.root, "labels/%s" % name))
        if self.random_crop:
            label = label.resize(self.resize, Image.NEAREST)
            label = label.crop((left, upper, right, lower))
        else:
            label = label.resize(self.crop_size, Image.NEAREST)
        label = self.convert_to_class_ids(label)
        scales_pyramid = self.GeneratePyramid(image)

        return scales_pyramid, label, name

    def convert_to_class_ids(self, label_image):
        label = np.asarray(label_image, np.float32)
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy

def create_sit_dataloader(opt, set='train'):
    _sit_dataset = sit_dataset(opt.src_data_dir,
                                opt.src_data_list,
                                opt.scale_factor,
                                opt.num_scales,
                                opt.curr_scale,
                                set)

    sit_dataloader =     DataLoader(_sit_dataset,
                                    batch_size=opt.batch_size,
                                    shuffle=True,
                                    num_workers=opt.num_workers,
                                    pin_memory=True,
                                    drop_last=False)
    return sit_dataloader

    # def SetEpochSize(self, epoch_size):
    #     if (epoch_size > len(self.img_ids)):
    #         self.img_ids = self.img_ids * int(np.ceil(float(epoch_size) / len(self.img_ids)))
    #     self.img_ids = self.img_ids[:epoch_size]
