import numpy as np
from data.domainAdaptationDataset import domainAdaptationDataSet
from core.functions import RGBImageToNumpy
import os.path as osp
import os
from PIL import Image

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)  #os.walk: traversal all files in rootdir and its subfolders
        for filename in filenames
        if filename.endswith(suffix)
    ]
class cityscapesDataSet(domainAdaptationDataSet):
    def __init__(self, root, images_list_path, scale_factor, num_scales, curr_scale, set, get_image_label=False, get_scales_pyramid=False, get_pseudo_label=False, pseudo_root=''):
        super(cityscapesDataSet, self).__init__(root, images_list_path, scale_factor, num_scales, curr_scale, set, get_image_label=get_image_label)
        self.get_scales_pyramid= get_scales_pyramid
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        self.get_pseudo_label = get_pseudo_label
        self.pseudo_root = pseudo_root
        # if self.get_pseudo_label:
        #     self.pseudo_root = pseudo_root
        #     print('len before removal: ', len(self.img_ids))
        #     available_pseudos = sorted(recursive_glob(rootdir=self.pseudo_root, suffix=".png"))
        #     available_pseudos = [p.split('/')[-1] for p in available_pseudos]
        #     img_ids_to_remove = []
        #     for im_id in self.img_ids:
        #         if im_id.split('/')[-1] not in available_pseudos:
        #             img_ids_to_remove.append(im_id)
        #     for i in range(len(img_ids_to_remove)):
        #         self.img_ids.remove(img_ids_to_remove[i])
        #     print('len after removal: ', len(self.img_ids))
    def __getitem__(self, index):
        name = self.img_ids[index]
        image = Image.open(osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))).convert('RGB')
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
        elif self.get_pseudo_label:
            file_name = name.split('/')[1]
            label = Image.open(osp.join(self.pseudo_root, file_name))
            # label = label.resize( self.crop_size, Image.NEAREST )
            # assert image.size == label.size
            label = np.asarray(label, np.float32)
            label_copy = label
            # label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
            # for k, v in self.id_to_trainid.items():
            #     label_copy[label == k] = v


        scales_pyramid = None
        if self.get_scales_pyramid:
            scales_pyramid = self.GeneratePyramid(image)
        else:
            image = RGBImageToNumpy(image)

        if self.get_scales_pyramid and (self.get_image_label or self.get_pseudo_label):
            return scales_pyramid, label_copy.copy()
        elif self.get_scales_pyramid and not (self.get_image_label or self.get_pseudo_label):
            return scales_pyramid
        elif not self.get_scales_pyramid and (self.get_image_label or self.get_pseudo_label):
            return image.copy(), label_copy.copy()
        elif not self.get_scales_pyramid and not (self.get_image_label or self.get_pseudo_label):
            return image.copy()




