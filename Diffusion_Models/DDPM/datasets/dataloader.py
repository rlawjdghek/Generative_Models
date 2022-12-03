from os.path import join as opj

import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .custom_dataset import SingleDataset, SingleBaseDataset

def get_dataloader(args, eval_batch_size=None):
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
    elif args.data_name == "CelebA-HQ-img":
        train_dataset = SingleBaseDataset(args, train_transform)    
    if args.use_DDP:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    if eval_batch_size is not None:
        bs = eval_batch_size
    else:
        bs = max(args.batch_size//args.n_gpus, 1)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=shuffle, num_workers=args.n_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    return train_loader

def get_single_dataloader(data_dir, img_size_H, img_size_W, batch_size, imagenet_normalize=True, drop_last=False, shuffle=True):
    if imagenet_normalize:
        H = 299
        W = 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        H = img_size_H
        W = img_size_W
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    transform = T.Compose([
        T.Resize((img_size_H, img_size_W)),
        T.Resize((H, W)),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    dataset = SingleDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True, drop_last=drop_last)
    return loader