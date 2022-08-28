import os
from os.path import join as opj
import argparse
import datetime
import time
import cv2

import torch
import torch.distributed as dist
from torch.backends import cudnn
torch.autograd.set_detect_anomaly(True)
from utils.util import *
from datasets.dataloader import get_dataloader
from models.SAGAN import SAGAN

def build_args(is_test=False):
    parser = argparse.ArgumentParser()
    #### dataset ####
    parser.add_argument("--data_root_dir", type=str, default="/data")
    parser.add_argument("--data_name", type=str, default="imagenet")
    parser.add_argument("--img_size_H", type=int, default=128)
    parser.add_argument("--img_size_W", type=int, default=128)
    #### model ####

    #### train & test ####
    parser.add_argument("--n_iters", type=int, default=10000000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int ,default=100)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--D_per_G", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--D_lr", type=float, default=0.0004)
    parser.add_argument("--G_lr", type=float, default=0.0001)
    
    #### save & load ####
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--save_root_dir", type=str, default="/data/jeonghokim/GANs/SAGAN")
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--log_save_iter_freq", type=int, default=100)
    parser.add_argument("--model_save_iter_freq", type=int, default=1000)
    parser.add_argument("--img_save_iter_freq", type=int, default=100)
    parser.add_argument("--n_save_imgs", type=int, default=4)

    #### config ####
    parser.add_argument("--use_DDP", action="store_true")
    
    args = parser.parse_args()
    if args.data_name == "imagenet":
        args.n_cls = 1000
    elif args.data_name == "celeba":
        args.n_cls = 1
    args.save_name = f"{datetime.datetime.now().strftime('%Y%m%d')}_" + args.save_name
    args.is_test = is_test
    if args.is_test:
        args.use_DDP = False
        args.no_save = True
    if args.use_DDP:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=72000))
        args.n_gpus = dist.get_world_size()
    else:
        args.local_rank = 0
        args.n_gpus = 1
    args.save_dir = opj(args.save_root_dir, args.save_name)
    args.img_save_dir = opj(args.save_dir, "save_images")
    args.model_save_dir = opj(args.save_dir, "save_models")
    args.eval_save_dir = opj(args.save_dir, "eval_save_images")
    args.log_path = opj(args.save_dir, "log.txt")
    args.config_path = opj(args.save_dir, "config.json")
    if not args.no_save:
        os.makedirs(args.img_save_dir, exist_ok=True)
        os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.eval_save_dir, exist_ok=True)
    return args

def main_worker(args, logger):
    train_loader, val_loader = get_dataloader(args)
    logger.write(f"[Train] # of imgs : {len(train_loader.dataset)}")
    logger.write(f"1 epochs : {len(train_loader)}iters")
    args.iter_per_epoch = len(train_loader)
    save_args(args, args.config_path)
    model = SAGAN(args)
    model.print_n_params(logger)
    cur_iter = 1
    start_time = time.time()
    for epoch in range(1, 10000000):
        loss_D_meter = AverageMeter()
        loss_G_meter = AverageMeter()
        for img, label in train_loader:
            img = img.cuda(args.local_rank)
            label = label.cuda(args.local_rank)
            model.set_input(real_img=img, label=label)
            model.train(cur_iter)
            model.update_moving_avg()
            
            BS = img.shape[0]
            loss_D_meter.update(model.D_loss_val, BS)
            loss_G_meter.update(model.G_loss_val, BS)
            if cur_iter % args.log_save_iter_freq == 0:
                elapsed_time = str(datetime.timedelta(seconds=time.time() - start_time))[:-7]
                msg = f"[Train]_[Elapsed Time - {elapsed_time}]_[Epoch - {epoch}]_[Iter - {cur_iter}/{args.n_iters}]_[D/loss - {model.D_loss_val:.4f}]_[D/mean - {loss_D_meter.avg:.4f}]_[G/loss - {model.G_loss_val:.4f}]_[G/mean - {loss_G_meter.avg:.4f}]"
                logger.write(msg)
                logger.write("="*30)
            if cur_iter % args.model_save_iter_freq == 0:
                to_path = opj(args.model_save_dir, f"{cur_iter}_{args.n_iters}.pth")
                model.save(to_path)
            if cur_iter % args.img_save_iter_freq == 0:
                save_img = None
                for _ in range(args.n_save_imgs):
                    model.inference()
                    gene_img_img = tensor2img(model.gene_img)
                    if save_img is None:
                        save_img = gene_img_img
                    else:
                        save_img = np.concatenate([save_img, gene_img_img], axis=1)
                to_path = opj(args.img_save_dir, f"{cur_iter}_{args.n_iters}.png")
                if args.local_rank == 0:
                    cv2.imwrite(to_path, save_img[:,:,::-1])
            if cur_iter == args.n_iters: break
            cur_iter += 1
if __name__ == "__main__":
    args = build_args()
    logger = Logger(args.local_rank, args.no_save)
    logger.open(args.log_path)
    print_args(args, logger)
    cudnn.benchmark = True
    main_worker(args, logger)
