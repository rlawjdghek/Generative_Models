import os
from os.path import join as opj
import argparse 
import datetime
import time

import numpy as np
import cv2
import torch
import torch.distributed as dist
from torch.backends import cudnn

from utils.util import *
from datasets.dataloader import get_dataloader
from models.VQGAN import VQGAN
torch.autograd.set_detect_anomaly(True)

def build_args(is_test=False):
    parser = argparse.ArgumentParser()

    #### dataset ####
    parser.add_argument("--data_root_dir", type=str, default="/home/data")
    parser.add_argument("--data_name", type=str, default="CelebA-HQ-img")
    parser.add_argument("--img_size_H", type=int, default=256)
    parser.add_argument("--img_size_W", type=int, default=256)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--in_ch", type=int, default=3)

    #### model ####
    parser.add_argument("--ngf", type=int, default=128)
    parser.add_argument("--ngf_mult", default=[1,1,2,2,4])
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--attn_resolutions", default=[16])
    parser.add_argument("--z_dim", type=int, default=256)
    parser.add_argument("--n_embed", type=int, default=1024)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--D_n_layers", type=int, default=3)
    parser.add_argument("--num_res_blks", type=int, default=2)
    parser.add_argument("--double_z", type=bool, default=False)
    parser.add_argument("--D_use_actnorm" ,type=bool, default=False)
    
    #### train & eval ####
    parser.add_argument("--n_iters", type=int, default=50000000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--perceptual_weight", type=float, default=1.0)
    parser.add_argument("--D_weight", type=float, default=0.8)
    parser.add_argument("--D_thres_iter", type=int, default=30001)
    parser.add_argument("--codebook_weight", type=float, default=1.0)
    parser.add_argument("--G_lr", type=float, default=4.5e-6)
    parser.add_argument("--D_lr", type=float, default=4.5e-6)
    parser.add_argument("--betas", default=(0.5, 0.9))
    parser.add_argument("--adv_loss_type", type=str, default="hinge")

    #### save & load ####
    parser.add_argument("--no_save", type=bool, default=False)
    parser.add_argument("--save_root_dir" ,type=str, default="/media/data1/jeonghokim/GANs/VQGAN")
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--log_save_iter_freq", type=int, default=200)
    parser.add_argument("--img_save_iter_freq", type=int, default=5000)
    parser.add_argument("--model_save_iter_freq", type=int, default=30000)
    parser.add_argument("--eval_iter_freq", type=int, default=60000)
    #### config ####
    parser.add_argument("--use_DDP", action="store_true")

    args = parser.parse_args()
    args.save_name = f"{datetime.datetime.now().strftime('%Y%m%d')}_" + args.save_name
    args.is_test = is_test
    if is_test:
        args.use_DDP = False
        args.is_test = True
        args.no_save = True
    if args.use_DDP:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=72000))
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
    train_loader, valid_loader = get_dataloader(args)
    logger.write(f"[Train] # of imgs : {len(train_loader.dataset)}")
    logger.write(f"1 epochs : {len(train_loader)} iters")
    if args.local_rank == 0:
        save_args(args, args.config_path)
    model = VQGAN(args)
    model.print_n_params(logger)
    cur_iter = 1
    start_time = time.time()
    break_flag = False
    for epoch in range(1, 100000):
        loss_G_meter = AverageMeter()
        loss_D_meter = AverageMeter()
        for img in train_loader:
            img = img.cuda(args.local_rank)
            model.set_input(real_img=img)
            model.train(cur_iter)

            BS = img.shape[0]
            loss_G_meter.update(model.G_loss_val*2, BS)
            loss_D_meter.update(model.D_loss_val*2, BS)
            if cur_iter % args.log_save_iter_freq == 0:
                elapsed_time = str(datetime.timedelta(seconds=time.time() - start_time))[:-7]
                msg = f"[Train]_[Elapsed time - {elapsed_time}]_[iter - {cur_iter}/{args.n_iters}]_[Epoch - {epoch}]_[D/loss - {model.D_loss_val:.4f}]_[G/loss - {model.G_loss_val:.4f}]_[D/loss avg - {loss_D_meter.avg:.4f}]_[G/loss avg - {loss_G_meter.avg:.4f}]"
                logger.write(msg)
            if cur_iter % args.model_save_iter_freq == 0:
                to_path = opj(args.model_save_dir, f"{cur_iter}_{args.n_iters}.pth")
                model.save(to_path)
            if cur_iter % args.img_save_iter_freq == 0:
                real_img_img = tensor2img(model.real_img)
                gene_img_img = tensor2img(model.recon_img)
                save_img = np.concatenate([real_img_img, gene_img_img], axis=1)
                to_path = opj(args.img_save_dir, f"{cur_iter}_{args.n_iters}.png")
                if args.local_rank == 0:
                    cv2.imwrite(to_path, save_img[:,:,::-1])
            if cur_iter == args.n_iters:
                break_flag = True
                break
            if args.use_DDP:
                dist.barrier()
            cur_iter += 1

        if break_flag: 
            break
if __name__ == "__main__":
    args = build_args()
    logger = Logger(args.local_rank, no_save=args.no_save)
    logger.open(args.log_path)
    print_args(args, logger)
    cudnn.benchmark = True
    main_worker(args, logger)
    