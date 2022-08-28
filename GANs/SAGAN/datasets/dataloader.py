from os.path import join as opj

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data.distributed import DistributedSampler

def get_dataloader(args):
    if args.data_name == "imagenet":
        train_transform = T.Compose([
            T.Resize((args.img_size_H, args.img_size_W)),
            T.CenterCrop(128),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        val_transform = T.Compose([
            T.Resize((args.img_size_H, args.img_size_W)),
            T.ToTensor(),
            T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        train_dataset = ImageFolder(opj(args.data_root_dir, args.data_name, "train"), transform=train_transform)
        val_dataset = ImageFolder(opj(args.data_root_dir, args.data_name, "val"), transform=val_transform)
    elif args.data_name == "celeba":
        train_transform = T.Compose([
            T.Resize((args.img_size_H, args.img_size_W)),
            T.CenterCrop(128),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        val_transform = T.Compose([
            T.Resize((args.img_size_H, args.img_size_W)),
            T.ToTensor(),
            T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        train_dataset = ImageFolder(opj(args.data_root_dir, args.data_name, "train"), transform=train_transform)
        val_dataset = ImageFolder(opj(args.data_root_dir, args.data_name, "val"), transform=val_transform)
    else:
        raise NotImplementedError
    if args.use_DDP:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    train_loader = DataLoader(train_dataset, batch_size=max(args.batch_size//args.n_gpus, 1), shuffle=shuffle, num_workers=args.n_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader
        
