from glob import glob 
import os
from os.path import join as opj

from PIL import Image

from .base_dataset import BaseDataset

class SingleBaseDataset(BaseDataset):
    def __init__(self, args, transform):
        super().__init__(args)
        self.img_paths = glob(opj(args.data_root_dir, args.data_name, "*"))
        self.transform = transform
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = self.transform(Image.open(img_path).convert("RGB"))
        return img, img
class SingleDataset(BaseDataset):
    def __init__(self, data_dir, transform):
        super().__init__(None)
        self.img_paths = sorted(glob(opj(data_dir, "*")))[:60]
        self.n_paths = len(self.img_paths)
        self.transform = transform
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = self.transform(Image.open(img_path).convert("RGB"))
        return {"img": img}
        