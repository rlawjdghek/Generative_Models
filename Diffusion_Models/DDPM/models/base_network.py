import torch
import torch.nn as nn

def define_G(args):
    if args.G_name == "UNet":
        from .networks import UNet
        G = UNet(ngf=args.ngf, ngf_mults=args.ngf_mults, in_ch=args.in_ch, out_ch=args.out_ch, self_condition=args.self_condition, resnet_blk_group_bn=args.resnet_blk_group_bn)
    return G
def define_coef(args):
    from .networks import Coef
    C = Coef(beta_schedule=args.beta_schedule, n_timesteps=args.n_timesteps, p2_loss_weight_k=args.p2_loss_weight_k, p2_loss_weight_gamma=args.p2_loss_weight_gamma)
    return C
def count_params(model):
    n_params = 0
    for p in model.parameters():
        n_params += p.numel()
    return n_params
def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)
class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()