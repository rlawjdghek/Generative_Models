import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from .base_network import define_G, count_params, moving_average, define_coef
from .base_model import extract, default, define_loss

class DDPM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_timesteps = args.n_timesteps
        self.sampling_timesteps = default(args.sampling_timesteps, self.n_timesteps)  # 추론할 때 쓰인다. p_sample, ddim_sample
        assert self.sampling_timesteps <= args.n_timesteps
        self.is_ddim_sampling = self.sampling_timesteps < self.n_timesteps
        self.ddim_sampling_eta = args.ddim_sampling_eta

        self.G = define_G(args).cuda(args.local_rank)
        self.C = define_coef(args).cuda(args.local_rank)
        self.in_ch = self.G.in_ch
        if args.local_rank == 0:
            self.G_ema = copy.deepcopy(self.G)
        self.n_params_G = count_params(self.G)
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
        second_term = extract(self.C.sqrt_recipm1_alphas_cumprod, t, noise) * noise
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
    def update_ema(self):
        if self.args.use_DDP:
            moving_average(self.G.module, self.G_ema, beta=0.999)
        else:
            moving_average(self.G, self.G_ema, beta=0.999)
    def save(self, save_path):
        state_dict = {}
        if self.args.use_DDP:
            state_dict["G"] = self.G.module.state_dict()
        else:
            state_dict["G"] = self.G.state_dict()
        if self.args.local_rank == 0:
            state_dict["G_ema"] = self.G_ema.state_dict()
            torch.save(state_dict, save_path)
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
        self.gene_img = gene_img
            
            

