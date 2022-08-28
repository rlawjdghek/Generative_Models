from os.path import join as opj

import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .custom_dataset import SingleDataset

def get_dataloader(args):
    train_transform = T.Compose([
        T.Resize((args.img_size_H, args.img_size_W)),
        T.RandomHorizontalFlip(),
        T.CenterCrop((args.img_size_H, args.img_size_W)),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    if "LSUN" in args.data_name:
        data_name, *data_class = args.data_name.split("_")
        data_class = "_".join(data_class)
        train_dataset = datasets.LSUN(root=opj(args.data_root_dir, data_name), classes=[data_class+"_train"], transform=train_transform)
    elif args.data_name == "CelebA_HQ":
        train_dataset = SingleDataset(args, train_transform)    
    if args.use_DDP:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    train_loader = DataLoader(train_dataset, batch_size=max(args.batch_size//args.n_gpus, 1), shuffle=shuffle, num_workers=args.n_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    return train_loader

        
        
    