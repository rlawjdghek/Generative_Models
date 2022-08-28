import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .base_network import BaseNetwork, spectral_init
class CondBN(nn.Module):
    def __init__(self, in_ch, n_cls):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_ch)
        self.embed = nn.Embedding(n_cls, 2*in_ch)
        self.embed.weight.data[:, :in_ch] = 1
        self.embed.weight.data[:, in_ch:] = 0
    def forward(self, x, class_idx):
        x = self.norm(x)
        embed = self.embed(class_idx)
        gamma, beta = embed.chunk(2, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta
        return x
class SelfAttn(nn.Module):
    def __init__(self, in_ch, gain):
        super().__init__()
        self.q_layer = spectral_init(nn.Conv1d(in_ch, in_ch//8, 1), gain=gain)
        self.k_layer = spectral_init(nn.Conv1d(in_ch, in_ch//8, 1), gain=gain)
        self.v_layer = spectral_init(nn.Conv1d(in_ch, in_ch, 1), gain=gain)
        self.gamma = nn.Parameter(torch.tensor(0.0))
    def forward(self, x):
        BS, C, H, W = x.shape
        flatten = x.reshape(BS, C, -1)
        q = self.q_layer(flatten).permute(0, 2, 1)
        k = self.k_layer(flatten)
        v = self.v_layer(flatten)
        attn = F.softmax(q@k, dim=1)  # [BS x HW x HW]
        output = v@attn  # [BS x C x HW]
        output = output.reshape(BS, C, H, W)
        output = self.gamma * output + x
        return output
class ConvBlk(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, bn=True, n_cls=None, self_attn=False, upsample=True, act_layer=nn.ReLU()):
        super().__init__()
        self.conv = spectral_init(nn.Conv2d(in_ch, out_ch, k, s, p, bias=False if bn else True))
        self.act_layer = act_layer
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2)
        else:
            self.upsample = None
        if bn:
            self.norm = CondBN(out_ch, n_cls)
        else:
            self.norm = None
        if self_attn:
            self.attn = SelfAttn(out_ch, 1)
        else:
            self.attn = None

    def forward(self, x, class_idx=None):
        if self.upsample is not None:
            x = self.upsample(x)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x, class_idx)
        x = self.act_layer(x)
        if self.attn is not None:
            x = self.attn(x)
        return x
#### generator ####
class Generator(BaseNetwork):
    def __init__(self, latent_dim=100, n_cls=1000):
        super().__init__()
        self.linear = spectral_init(nn.Linear(latent_dim, 4*4*512))
        self.conv = nn.ModuleList([
            ConvBlk(512, 512, n_cls=n_cls),
            ConvBlk(512, 512, n_cls=n_cls),
            ConvBlk(512, 512, n_cls=n_cls, self_attn=True),
            ConvBlk(512, 256, n_cls=n_cls),
            ConvBlk(256, 128, n_cls=n_cls)
        ])
        self.last_conv = spectral_init(nn.Conv2d(128, 3, 3, 1, 1))
        self.act_layer = nn.ReLU()
    def forward(self, z, class_idx):
        x = self.act_layer(self.linear(z))
        x = x.reshape(-1, 512, 4, 4)
        for m in self.conv:
            x = m(x, class_idx)
        x = self.last_conv(x)
        return torch.tanh(x)
#### discriminator ####
class Discriminator(BaseNetwork):
    def __init__(self, n_cls=1000):
        super().__init__()
        self.conv = nn.ModuleList([
            ConvBlk(3, 128, s=2, bn=False, upsample=False, act_layer=nn.LeakyReLU(0.2)),
            ConvBlk(128, 256, s=2, bn=False, upsample=False, act_layer=nn.LeakyReLU(0.2)),
            ConvBlk(256, 512, s=1, bn=False, upsample=False, self_attn=True, act_layer=nn.LeakyReLU(0.2)),
            ConvBlk(512, 512, s=2, bn=False, upsample=False, act_layer=nn.LeakyReLU(0.2)),
            ConvBlk(512, 512, s=2, bn=False, upsample=False, act_layer=nn.LeakyReLU(0.2)),
            ConvBlk(512, 512, s=2, bn=False, upsample=False, act_layer=nn.LeakyReLU(0.2)),
        ])
        self.linear = spectral_init(nn.Linear(512, 1))
        self.embed = nn.Embedding(n_cls, 512)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.embed = spectral_norm(self.embed)
    def forward(self, x, class_idx):
        for m in self.conv:
            x = m(x)
        BS, C, H, W = x.shape
        x = x.reshape(BS, C, -1)
        x = x.sum(2)  # [BS x 512]
        out_linear = self.linear(x).squeeze(1)
        embed = self.embed(class_idx)
        prod = (x * embed).sum(1)
        return out_linear + prod
        

