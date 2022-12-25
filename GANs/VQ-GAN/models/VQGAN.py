from itertools import chain

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim

from models.base_model import BaseModel
from models.base_network import define_encoder, define_decoder, define_quantizer, define_D, count_params
from models.lpips import LPIPS

class VQGAN(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.perceptual_weight = args.perceptual_weight
        self.D_weight = args.D_weight
        self.D_thres_iter = args.D_thres_iter
        self.codebook_weight = args.codebook_weight
        self.encoder = define_encoder(args).cuda(args.local_rank)
        self.decoder = define_decoder(args).cuda(args.local_rank)
        self.quantizer = define_quantizer(args).cuda(args.local_rank)
        self.pre_quant_conv = nn.Conv2d(args.z_dim, args.embed_dim, 1, 1, 0).cuda(args.local_rank)
        self.post_quant_conv = nn.Conv2d(args.embed_dim, args.z_dim, 1, 1, 0).cuda(args.local_rank)
        self.D = define_D(args).cuda(args.local_rank)
        if args.use_DDP:
            self.encoder = DDP(self.encoder, device_ids=[args.local_rank])
            self.decoder = DDP(self.decoder, device_ids=[args.local_rank])
            self.quantizer = DDP(self.quantizer, device_ids=[args.local_rank])
            self.pre_quant_conv = DDP(self.pre_quant_conv, device_ids=[args.local_rank])
            self.post_quant_conv = DDP(self.post_quant_conv, device_ids=[args.local_rank])
            self.D = DDP(self.D, device_ids=[args.local_rank], broadcast_buffers=False)
            self.decoder_last_layer = self.decoder.module.conv_out.weight
        else:
            self.decoder_last_layer = self.decoder.conv_out.weight
        self.criterion_LPIPS = LPIPS().cuda(args.local_rank).eval()
        if args.adv_loss_type == "hinge":
            from .base_model import HingeLoss
            self.criterion_adv = HingeLoss()
        elif args.adv_loss_type == "vanilla":
            from .base_model import VanilaLoss
            self.criterion_adv = VanilaLoss()
        self.optimizer_G = optim.Adam(
            chain(
                self.encoder.parameters(), 
                self.decoder.parameters(), 
                self.quantizer.parameters(), 
                self.pre_quant_conv.parameters(), 
                self.post_quant_conv.parameters()
                ), lr=args.G_lr, betas=args.betas)
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=args.D_lr, betas=args.betas)
        self.G_loss_val = 0
        self.D_loss_val = 0
    def set_input(self, real_img):
        self.real_img = real_img
    def train(self, cur_iter):
        # encoding
        h = self.encoder(self.real_img)
        h = self.pre_quant_conv(h)
        z_q, codebook_loss, _ = self.quantizer(h)

        # decoding
        z_q = self.post_quant_conv(z_q)
        recon_img = self.decoder(z_q)
        #### generator ####
        if cur_iter % 2 == 1:
            recon_loss = torch.abs(self.real_img - recon_img)
            if self.perceptual_weight > 0:
                try:
                    perceptual_loss = self.criterion_LPIPS(self.real_img, recon_img)
                except:
                    perceptual_loss = 0
                recon_loss = recon_loss + self.perceptual_weight * perceptual_loss
            else:
                perceptual_loss = torch.tensor([0.0])
            nll_loss = torch.mean(recon_loss)
            logit_gene =  self.D(recon_img)
            adv_loss = -torch.mean(logit_gene)
            try:
                adapt_weight = self.calc_adaptive_weight(nll_loss, adv_loss)
            except:
                print("error in adapt weight")
                adapt_weight = 0
            is_D_valid = self.is_valid_iter(1, cur_iter, self.D_thres_iter)
            G_loss = nll_loss + adapt_weight * is_D_valid * adv_loss + self.codebook_weight * codebook_loss.mean()
            self.optimizer_G.zero_grad()
            G_loss.backward()
            self.optimizer_G.step()

            self.G_loss_val = G_loss.item()
        #### discriminator ####
        else:
            logit_real = self.D(self.real_img)
            logit_gene = self.D(recon_img.detach())
            is_D_valid = self.is_valid_iter(1, cur_iter, self.D_thres_iter)
            D_loss = is_D_valid * self.criterion_adv(logit_real, logit_gene)
            
            self.optimizer_D.zero_grad()
            D_loss.backward()
            self.optimizer_D.step()
            
            self.D_loss_val = D_loss.item()
        self.recon_img = recon_img.detach()
    def calc_adaptive_weight(self, recon_loss, adv_loss):  # 논문 eq. (7)
        nll_grad = torch.autograd.grad(recon_loss, self.decoder_last_layer, retain_graph=True)[0]
        adv_grad = torch.autograd.grad(adv_loss, self.decoder_last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grad) / (torch.norm(adv_grad) + 1e-6)
        d_weight = torch.clamp(d_weight, 0, 1e4).detach()
        d_weight = self.D_weight * d_weight
        return d_weight
    def is_valid_iter(self, weight, cur_iter, threshold=0, val=0.0):
        if cur_iter < threshold:
            weight = val
        return weight
    def print_n_params(self, logger):
        logger.write(f"# of encoder params : {count_params(self.encoder)}")
        logger.write(f"# of decoder params : {count_params(self.decoder)}")
        logger.write(f"# of quantizer params : {count_params(self.quantizer)}")
        logger.write(f"# of pre_quant_conv params : {count_params(self.pre_quant_conv)}")
        logger.write(f"# of post_quant_conv params : {count_params(self.post_quant_conv)}")
        logger.write(f"# of D params : {count_params(self.D)}")
    def save(self, save_path):
        state_dict = {}
        if self.args.use_DDP:
            state_dict["encoder"] = self.encoder.module.state_dict()
            state_dict["decoder"] = self.decoder.module.state_dict()
            state_dict["pre_quant_conv"] = self.pre_quant_conv.module.state_dict()
            state_dict["post_quant_conv"] = self.post_quant_conv.module.state_dict()
            state_dict["D"] = self.D.module.state_dict()
        else:
            state_dict["encoder"] = self.encoder.state_dict()
            state_dict["decoder"] = self.decoder.state_dict()
            state_dict["pre_quant_conv"] = self.pre_quant_conv.state_dict()
            state_dict["post_quant_conv"] = self.post_quant_conv.state_dict()
            state_dict["D"] = self.D.state_dict()
        if self.args.local_rank == 0:
            torch.save(state_dict, save_path)
        

