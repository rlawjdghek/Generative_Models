import math

import torch
import torch.nn as nn
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    b_1 = scale * 0.0001
    b_T = scale * 0.02
    return torch.linspace(b_1, b_T, timesteps, dtype=torch.float32)
def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
def extract(a ,t, x_shape):  # t는 1~1000중 [BS,] shape으로 값을 가지고, 1000개의 미리 정해진 계수 alpha 또는 beta에서 t를 index로 뽑아야 한다. 또한 이는 4차원 텐서인 x (x_shape) 와 곱해지므로 뽑힌 alpha 또는 beta를 [BS x 1 x 1 x 1]로 reshape한다. 
    BS, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(BS, *((1,)*(len(x_shape)-1)))
def default(x, y):
    if x is not None:
        return x
    return y() if callable(y) else y
def define_loss(loss_type):
    if loss_type == "l1":
        return nn.L1Loss(reduction='none')
    elif loss_type == "l2":
        return nn.MSELoss(reduction="none")
    else:
        raise NotImplementedError  