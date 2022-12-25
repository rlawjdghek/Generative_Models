import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_dataloader(args):
    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=max(args.img_size_H, args.img_size_W)),
        A.RandomCrop(height=args.img_size_H, width=args.img_size_W),
        A.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])
    valid_transform = train_transform
    if args.data_name == "Imagenet":
        from .custom_dataset import ImageNetDataset
        train_dataset = ImageNetDataset(args, train_transform, is_train=True)
        valid_dataset = ImageNetDataset(args, valid_transform, is_train=False)
    elif args.data_name == "CelebA-HQ-img":
        from .custom_dataset import CelebAHQDataset
        train_dataset = CelebAHQDataset(args, train_transform, is_train=True)
        valid_dataset = CelebAHQDataset(args, valid_transform, is_train=False)
    if args.use_DDP:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    train_loader = DataLoader(train_dataset, batch_size=max(1, args.batch_size//args.n_gpus), shuffle=shuffle, num_workers=args.n_workers, pin_memory=True, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True)
    return train_loader, valid_loader
