import os
from os.path import join as opj
import argparse
import datetime
import time

import cv2
import torch
import torch.distributed as dist
from torch.backends import cudnn

from utils.util import *
from datasets.dataloader import get_dataloader
from models.DDPM import DDPM
def build_args(is_test=False):
    parser = argparse.ArgumentParser()

    #### dataset ####
    parser.add_argument("--data_root_dir", type=str, default="/home/data/")
    parser.add_argument("--data_name", type=str, default="CelebA-HQ-img", choices=["LSUN_church_outdoor", "CelebA-HQ-img"])
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--img_size_H", type=int, default=256)
    parser.add_argument("--img_size_W", type=int, default=256)
    parser.add_argument("--in_ch", type=int, default=3)
    parser.add_argument("--out_ch", type=int, default=3)

    #### model ####
    parser.add_argument("--G_name", type=str, default="UNet")
    parser.add_argument("--self_condition", type=bool, default=False, help="모델이 예측한 x0를 추가 입력으로 함.")
    parser.add_argument("--model_objective", type=str, default="pred_noise", choices=["pred_noise", "pred_x0"])
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ngf_mults", default=[1,2,4,8,8,8])
    parser.add_argument("--resnet_blk_group_bn", type=int, default=8)
    
    #### train & test ####
    parser.add_argument("--n_iters", type=int, default=10_000_000)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--n_timesteps", type=int, default=1000, help="total time step T. default T=1000 in official paper.")
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"], help="initialize method for beta")
    parser.add_argument("--sampling_timesteps", type=int, default=None)
    parser.add_argument("--ddim_sampling_eta", type=float, default=1.0)  # 1이면 DDIM, 0이면 DDPM
    parser.add_argument("--p2_loss_weight_k", type=float, default=1)
    parser.add_argument("--p2_loss_weight_gamma", type=float, default=0.0)
    parser.add_argument("--loss_type", type=str, default="l1", choices=["l1", "l2"])
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--betas", default=(0.9, 0.99))
    parser.add_argument("--n_fid_images", type=int, default=30000)
    parser.add_argument("--eval_iter_freq", type=int, default=500000)
    
    #### save & load ####
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--save_root_dir", type=str, default="/media/data1/jeonghokim/GANs/DDPM")
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--log_save_iter_freq", type=int, default=1000)
    parser.add_argument("--model_save_iter_freq", type=int, default=10000)
    parser.add_argument("--img_save_iter_freq", type=int, default=5000)
    parser.add_argument("--n_save_images", type=int, default=8)    
    parser.add_argument("--resume_path", type=str, default="")

    #### config ####
    parser.add_argument("--use_DDP", action="store_true")

    args = parser.parse_args()
    if is_test:
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
    args.save_name = f"{datetime.datetime.now().strftime('%Y%m%d')}_" + args.save_name
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
    train_loader = get_dataloader(args)
    logger.write(f"[Train] # of imgs : {len(train_loader)*args.batch_size}")
    logger.write(f"1 epochs : {len(train_loader)} iters")
    save_args(args, args.config_path)
    model = DDPM(args)
    model.print_n_params(logger)

    cur_iter = 1
    start_time = time.time()
    break_flag = False
    for epoch in range(1, 100000):
        loss_G_meter = AverageMeter()
        if args.use_DDP:
            train_loader.sampler.set_epoch(epoch)
        for img, _ in train_loader:
            img = img.cuda(args.local_rank)
            model.set_input(img)
            model.train()

            BS = img.shape[0]
            loss_G_meter.update(model.loss_val, BS)
            
            if cur_iter % args.log_save_iter_freq == 0:
                elapsed_time = str(datetime.timedelta(seconds=time.time() - start_time))[:-7]
                msg = f"[Train]_[Elapsed Time - {elapsed_time}]_[iter - {cur_iter}/{args.n_iters}]_[Epoch - {epoch}]_[loss - {model.loss_val:.4f}]_[loss mean- {loss_G_meter.avg:.4f}]"
                logger.write(msg)
                logger.write("="*30)
            if cur_iter % args.model_save_iter_freq == 0:
                to_path = opj(args.model_save_dir, f"{cur_iter}_{args.n_iters}.pth")
                model.save(to_path)
            if cur_iter % args.img_save_iter_freq == 0:
                img_shape = [args.n_save_images, model.in_ch, args.img_size_H, args.img_size_W]
                model.p_sample_loop(img_shape)
                gene_img_img = tensor2img(model.gene_img)
                to_path = opj(args.img_save_dir, f"{cur_iter}_{args.n_iters}.png")
                if args.local_rank == 0:
                    cv2.imwrite(to_path, gene_img_img[:,:,::-1])
            if cur_iter % args.eval_iter_freq == 0:
                real_dir = opj(args.data_root_dir, args.data_name)
                fid_dict = model.evaluate(real_dir) 
                msg = f"[FID]_[iter - {cur_iter}/{args.n_iters}]_[FID - {fid_dict['fid']:.4f}]"
                logger.write(msg)            
            if args.use_DDP:
                dist.barrier()  
            if cur_iter >= args.n_iters:
                break_flag = True
                break
            cur_iter += 1
        if break_flag:
            break
        


if __name__=="__main__":
    args = build_args()
    logger = Logger(args.local_rank, args.no_save)
    logger.open(args.log_path)
    print_args(args, logger)
    cudnn.benchmark = True
    main_worker(args, logger)
