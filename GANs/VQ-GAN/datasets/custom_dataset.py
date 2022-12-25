import os
from os.path import join as opj
from glob import glob

from PIL import Image
import numpy as np

from .base_dataset import BaseDataset

IMG_SUFFIX = [".png", ".jpg", ".JPG", ".PNG", ".JPEG"]
is_img = lambda path: os.path.splitext(path)[1] in IMG_SUFFIX

class ImageNetDataset(BaseDataset):
    def __init__(self, args, transform, is_train=True):
        super().__init__(args)
        self.trainval = "train" if is_train else "val"
        self.img_paths = self.get_paths()
        self.transform = transform
    def get_paths(self):
        root_dir = opj(self.args.data_root_dir, self.args.data_name, self.trainval)
        paths = []
        for root, folders, fns in os.walk(root_dir):
            if len(fns) > 0:
                for fn in fns:
                    path = opj(root, fn)
                    if is_img(path):
                        paths.append(path)
        return paths
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = np.array(img)
        img = self.transform(image=img)["image"]
        return img

class CelebAHQDataset(BaseDataset):
    def __init__(self, args, transform, is_train=True):
        super().__init__(args)
        self.trainval = "train" if is_train else "val"
        self.img_paths = self.get_paths()
        self.transform = transform
    def get_paths(self):
        root_dir = opj(self.args.data_root_dir, self.args.data_name)
        paths = []
        for root, folders, fns in os.walk(root_dir):
            if len(fns) > 0:
                for fn in fns:
                    path = opj(root, fn)
                    if is_img(path):
                        paths.append(path)
        return paths
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = np.array(img)
        img = self.transform(image=img)["image"]
        return img
