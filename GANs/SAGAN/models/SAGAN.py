import copy

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .base_model import BaseModel, moving_average
from .base_network import define_D, define_G, count_params

class SAGAN(BaseModel):
    def __init__(self, args):
        self.args = args
        self.D = define_D(args).cuda(args.local_rank)
        self.G = define_G(args).cuda(args.local_rank)
        self.G_ema = copy.deepcopy(self.G).eval()
        self.n_params_D = count_params(self.D)
        self.n_params_G = count_params(self.G)
        if args.use_DDP:
            self.D = DDP(self.D, device_ids=[args.local_rank])
            self.G = DDP(self.G, device_ids=[args.local_rank])
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=args.D_lr, betas=(0, 0.9))
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=args.G_lr, betas=(0, 0.9))
    def set_input(self, real_img, label):
        self.real_img = real_img
        self.label = label
    def train(self, cur_iter):
        self.update_D()

        if cur_iter % self.args.D_per_G == 0:
            self.update_G()
    def update_D(self):
        latent_z = torch.randn((self.real_img.shape[0], self.args.latent_dim)).cuda(self.args.local_rank)
        with torch.no_grad():
            gene_img = self.G(latent_z, self.label)
        real_pred = self.D(self.real_img, self.label)
        gene_pred = self.D(gene_img.detach(), self.label)
        D_loss = F.relu(1 + gene_pred).mean() + F.relu(1 - real_pred).mean()
        self.optimizer_D.zero_grad()
        D_loss.backward()
        self.optimizer_D.step()        

        self.D_loss_val = D_loss.item()
    def update_G(self):
        latent_z = torch.randn((self.real_img.shape[0], self.args.latent_dim)).cuda(self.args.local_rank)
        class_idx = torch.multinomial(torch.ones(self.args.n_cls), self.real_img.shape[0], replacement=True).cuda(self.args.local_rank)
        gene_img = self.G(latent_z, class_idx)
        gene_pred = self.D(gene_img, class_idx)
        G_loss = -gene_pred.mean()
        self.optimizer_G.zero_grad()
        G_loss.backward()
        self.optimizer_G.step()
        
        self.G_loss_val = G_loss.item()
    def update_moving_avg(self):
        if self.args.use_DDP:
            moving_average(self.G.module, self.G_ema, beta=0.999)
        else:
            moving_average(self.G, self.G_ema, beta=0.999)
    @torch.no_grad()
    def inference(self):
        latent_z = torch.randn((self.real_img.shape[0], self.args.latent_dim)).cuda(self.args.local_rank)
        class_idx = torch.multinomial(torch.ones(self.args.n_cls), self.real_img.shape[0], replacement=True).cuda(self.args.local_rank)
        self.G.eval()
        self.gene_img = self.G(latent_z, class_idx)
        self.G.train()
        # self.gene_img = self.G_ema(latent_z, class_idx) 이건 안됨. BN 떄문인가?
    @torch.no_grad()
    def ema_custom_inference(self, class_idx):
        latent_z = torch.randn((self.real_img.shape[0], self.args.latent_dim)).cuda(self.args.local_rank)
        self.gene_img = self.G_ema(latent_z, class_idx)        
    def DDP_save_load(self, save_path):
        if self.args.local_rank == 0:
            state_dict = {}
            state_dict["G"] = self.G.state_dict()
            state_dict["D"] = self.D.state_dict()
            torch.save(state_dict, save_path)
        dist.barrier()
        map_loc = {"cuda:0": f"cuda:{self.args.local_rank}"}
        load_state_dict = torch.load(save_path, map_location=map_loc)
        self.G.load_state_dict(load_state_dict["G"])
        self.D.load_state_dict(load_state_dict["D"])
    def save(self, save_path):
        state_dict = {}
        if self.args.use_DDP:
            state_dict["G"] = self.G.module.state_dict()
            state_dict["D"] = self.D.module.state_dict()
        else:
            state_dict["G"] = self.G.state_dict()
            state_dict["D"] = self.D.state_dict()
        if self.args.local_rank == 0:
            torch.save(state_dict, save_path)
    def print_n_params(self, logger):
        logger.write(f"# of D parameters : {self.n_params_D}")
        logger.write(f"# of G parameters : {self.n_params_G}")
