import os
from os.path import join as opj
import shutil
import random
import copy

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .base_network import define_G, count_params, moving_average, define_coef
from .base_model import extract, default, define_loss
from utils.util import tensor2img
from metrics.fid import calculate_fid_given_paths


class DDPM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_DDP = args.use_DDP
        self.n_timesteps = args.n_timesteps
        self.sampling_timesteps = default(args.sampling_timesteps, self.n_timesteps)  # 추론할 때 쓰인다. p_sample, ddim_sample
        assert self.sampling_timesteps <= args.n_timesteps
        self.is_ddim_sampling = self.sampling_timesteps < self.n_timesteps
        self.ddim_sampling_eta = args.ddim_sampling_eta

        self.G = define_G(args).cuda(args.local_rank)
        self.G.in_ch = 3
        self.C = define_coef(args).cuda(args.local_rank)
        
        self.in_ch = self.G.in_ch
        self.n_params_G = count_params(self.G)
        if os.path.exists(args.resume_path):
            self.load(args.resume_path)
        if args.use_DDP:
            self.G = DDP(self.G, device_ids=[args.local_rank])

        self.criterion = define_loss(args.loss_type)
        self.optimizer = torch.optim.Adam(self.G.parameters(), lr=args.lr, betas=args.betas)
    def set_input(self, img, noise=None):
        self.x_0 = img
        self.noise = noise
    def q_sample(self, x_start, t, noise):  # 논문의 eq.14에서 x_0, noise를 통하여 입력 x_t를 계산
        noise = default(noise, torch.randn_like(x_start).cuda(self.args.local_rank))
        first_term = extract(self.C.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        second_term = extract(self.C.sqrt_one_minus_alphas_cumprod, t, noise.shape) * noise
        return first_term + second_term
    def predict_start_from_noise(self, x_t, t, noise):  # 논문의 eq.14에서 x_t, noise를 통하여 x_0 계산
        first_term = extract(self.C.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
        second_term = extract(self.C.sqrt_recipm1_alphas_cumprod, t, noise.shape) * noise
        return first_term - second_term
    def predict_noise_from_start(self, x_t, t, x_0):  # 논문의 eq.14에서 x_t, x_0을 통하여 noise 계산
        return (extract(self.C.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_0) / extract(self.C.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    def model_prediction(self, x, t, x_self_cond=None):
        model_output = self.G(x, t, x_self_cond)
        
        if self.args.model_objective == "pred_noise":
            pred_noise = model_output
            pred_x_0 = self.predict_start_from_noise(x, t, pred_noise)
        elif self.args.model_objective == "pred_x0":
            pred_x_0 = model_output
            pred_noise = self.predict_noise_from_start(x, t, pred_x_0)
        return {"pred_noise": pred_noise, "pred_x_0": pred_x_0}
    def train(self):
        BS, C, H, W = self.x_0.shape
        t = torch.randint(0, self.args.n_timesteps, (BS,), device=self.x_0.get_device()).long()
        noise = torch.randn_like(self.x_0) if self.noise is None else self.noise
        x = self.q_sample(x_start=self.x_0, t=t, noise=noise)
        
        x_self_cond = None
        if self.args.self_condition and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_prediction(x, t)["pred_x_0"].detach()
        
        gene_x = self.G(x, t, x_self_cond)

        if self.args.model_objective == "pred_noise":
            target = noise
        elif self.args.model_objective == "pred_x0":
            target = self.x_0
        loss = self.criterion(gene_x, target)
        loss = torch.mean(loss, dim=list(range(loss.ndim))[1:])
        loss = (loss * extract(self.C.p2_loss_weight, t, loss.shape)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.loss_val = loss.item()
    def save(self, save_path):
        state_dict = {}
        if self.args.use_DDP:
            state_dict["G"] = self.G.module.state_dict()
        else:
            state_dict["G"] = self.G.state_dict()
        if self.args.local_rank == 0:
            torch.save(state_dict, save_path)
    def load(self, load_path):
        print(f"load from {load_path}")
        state_dict = torch.load(load_path, map_location={"cuda:0":f"cuda:{self.args.local_rank}"})
        if self.use_DDP:
            self.G.module.load_state_dict(state_dict["G"])
        else:
            self.G.load_state_dict(state_dict["G"])
    def print_n_params(self, logger):
        logger.write(f"# of G parameters : {self.n_params_G}")
    def q_posterior(self, x_0, x_t, t):  # mu, var, log_var 반환
        mu = extract(self.C.posterior_mean_coef1, t, x_t.shape) * x_0 + \
             extract(self.C.posterior_mean_coef2, t, x_t.shape) * x_t
        var = extract(self.C.posterior_variance, t, x_t.shape)
        log_var = extract(self.C.posterior_log_variance_clipped, t, x_t.shape)
        return mu, var, log_var
    def p_mean_variance(self, x_t, t, x_self_cond=None, clip_denoised=True):  # 샘플링에 필요한 mu, log_var와 self_condition을 위한 pred_x_0를 반환
        preds = self.model_prediction(x_t, t, x_self_cond)
        pred_x_0 = preds["pred_x_0"]
        if clip_denoised:
            pred_x_0.clip_(-1.0, 1.0)
        mu, var, log_var = self.q_posterior(pred_x_0, x_t, t)
        return mu, var, log_var, pred_x_0
    @torch.no_grad()
    def p_sample(self, x_t, t, x_self_cond=None, clip_denoised=True):  # 논문 algorithm2 for문 1 step, pred_x_0은 self_condition을 위함
        BS = x_t.shape[0]
        time = torch.full([BS, ], t, dtype=torch.long).cuda(x_t.get_device())
        mu, _, log_var, pred_x_0 = self.p_mean_variance(x_t, time, x_self_cond=x_self_cond, clip_denoised=clip_denoised)
        z = torch.randn_like(x_t).cuda(x_t.get_device()) if t > 0 else 0.0
        x_t_1 = mu + (0.5 * log_var).exp() * z  # 결과적으로는 sigma가 된다. 
        return x_t_1, pred_x_0        
    @torch.no_grad()
    def p_sample_loop(self, shape):  # 논문의 Algorithm 2
        x = torch.randn(shape).cuda(self.args.local_rank)  # x_T
        pred_x_0 = None
        for t in reversed(range(0, self.n_timesteps)):
            x_self_cond = pred_x_0 if self.args.self_condition else None
            x, pred_x_0 = self.p_sample(x_t=x, t=t, x_self_cond=x_self_cond)
        gene_img = x
        return gene_img
    @torch.no_grad()
    def ddim_sample_loop(self, shape, n_sample_timesteps, eta):
        t_lst = torch.linspace(-1, self.n_timesteps-1, steps=n_sample_timesteps+1)
        t_lst = list(reversed(t_lst.int().tolist()))
        t_pairs = list(zip(t_lst[:-1], t_lst[1:]))
        x = torch.randn(shape).cuda(self.args.local_rank)  # x_T
        pred_x_0 = None
        for t, t_1 in t_pairs:
            time = torch.full([x.shape[0], ], t, dtype=torch.long).cuda(x.get_device())
            x_self_cond = pred_x_0 if self.args.self_condition else None
            preds = self.model_prediction(x, time, x_self_cond)
            eps_theta = preds["pred_noise"]
            pred_x_0 = preds["pred_x_0"]
            if t_1 < 0:
                gene_img = pred_x_0
                break
            alpha_t = self.C.alphas_cumprod[t]  # DDIM이랑 DDPM의 notation은 약간 다름.
            alpha_t_1 = self.C.alphas_cumprod[t_1]
            
            sigma = eta * ((1-alpha_t_1) / (1-alpha_t)).sqrt() * (1 - alpha_t/alpha_t_1).sqrt()
            x_t_1 = alpha_t_1.sqrt() * pred_x_0 + (1-alpha_t_1-sigma**2).sqrt() * eps_theta + sigma*torch.randn_like(x)  # DDIM eq 12
            x = x_t_1
        return gene_img
    @torch.no_grad()
    def sample(self, shape, n_sample_timesteps, eta):
        if n_sample_timesteps == self.n_timesteps:
            return self.p_sample_loop(shape)
        else:
            return self.ddim_sample_loop(shape, n_sample_timesteps, eta)
    @torch.no_grad()
    def evaluate(self, real_dir):  # FID
        # generate images
        print("evaluate")
        self.G.eval()
        gene_dir = opj(self.args.eval_save_dir, "generated_images")
        if self.args.local_rank == 0:
            shutil.rmtree(gene_dir, ignore_errors=True)
            os.makedirs(gene_dir)
        if self.args.use_DDP:
            dist.barrier()
        n_imgs_per_gpu = self.args.n_fid_images // self.args.n_gpus
        bs = self.args.eval_batch_size
        for i in range(n_imgs_per_gpu // bs):
            img_shape = [bs, self.in_ch, self.args.img_size_H, self.args.img_size_W]
            gene_img = self.p_sample_loop(img_shape)
            for j in range(bs):
                to_path = opj(gene_dir, f"{n_imgs_per_gpu * self.args.local_rank + i*bs + j}.png")
                gene_img_img = tensor2img(gene_img[j])
                cv2.imwrite(to_path, gene_img_img[:,:,::-1])
        if self.args.use_DDP:
            dist.barrier()
        fid_val = calculate_fid_given_paths([real_dir, gene_dir], img_size_H=self.args.img_size_H, img_size_W=self.args.img_size_W, batch_size=self.args.eval_batch_size)
        self.G.train()
        return {"fid": fid_val}
        
                
                

            


        