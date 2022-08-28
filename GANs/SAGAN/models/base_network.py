import torch.nn as nn
from torch.nn import init
from torch.nn.utils import spectral_norm
def define_D(args):
    from .networks import Discriminator
    D = Discriminator(args.n_cls)
    return D
def define_G(args):
    from .networks import Generator
    G = Generator(args.latent_dim, args.n_cls)
    return G
def spectral_init(module, gain=1):
    init.kaiming_uniform_(module.weight, gain)
    if module.bias is not None:
        module.bias.data.zero_()
    return spectral_norm(module)
def count_params(model):
    n_params = 0
    for p in model.parameters():
        n_params += p.numel()
    return n_params
class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()