from glob import glob 
import os
from os.path import join as opj

from PIL import Image

from .base_dataset import BaseDataset

class SingleDataset(BaseDataset):
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