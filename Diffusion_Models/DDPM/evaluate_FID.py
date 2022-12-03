import os
from os.path import join as opj
import datetime

from tqdm import tqdm
import cv2
import torch
import torch.distributed as dist
from pytorch_fid.fid_score import calculate_fid_given_paths

from models.DDPM import DDPM
from utils.util import *

#### config ####
real_dir = "/home/data/CelebA-HQ-img"
load_dir = "/media/data1/jeonghokim/GANs/DDPM/20221118_"
save_model_iters = 1250000
batch_size = 20
fid_batch_size = 300
n_timesteps = 1000
use_DDP = True
n_sample_timesteps = 50  ####
eta = 1  ####
################
if use_DDP:
    local_rank = int(os.environ["LOCAL_RANK"]) 
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=72000))
    n_gpus = dist.get_world_size()
else:
    local_rank = 0
    n_gpus = 1

model_load_path = opj(load_dir, f"save_models/{save_model_iters}_10000000.pth")
save_dir = opj(load_dir, f"generated_images/{save_model_iters}_{n_sample_timesteps}")
os.makedirs(save_dir, exist_ok=True)
config_path = opj(load_dir, "config.json")
config = load_args(config_path)
config.n_timesteps = n_timesteps
config.use_DDP = use_DDP
config.local_rank = local_rank

img_shape = [batch_size, config.in_ch, config.img_size_H, config.img_size_W]
model = DDPM(config)
model.load(model_load_path)
model.G.eval()

n_imgs_per_gpu = config.n_fid_images // n_gpus
assert n_imgs_per_gpu % batch_size == 0
for i in tqdm(range(n_imgs_per_gpu // batch_size)):
    gene_img = model.sample(img_shape, n_sample_timesteps=n_sample_timesteps, eta=eta) 
    for j in range(batch_size):
        to_path = opj(save_dir, f"{n_imgs_per_gpu*local_rank + i*batch_size + j}.png")
        gene_img_img = tensor2img(gene_img[j])
        cv2.imwrite(to_path, gene_img_img[:,:,::-1])
if use_DDP:
    dist.barrier()
if local_rank == 0:
    fid_val = calculate_fid_given_paths([real_dir, save_dir], batch_size=fid_batch_size, device=torch.device("cuda"), dims=2048, num_workers=4)
if use_DDP:
    dist.barrier()
if local_rank == 0:
    print(f"fid val : {fid_val}")
        
