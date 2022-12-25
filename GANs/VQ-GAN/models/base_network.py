import torch.nn as nn

def define_encoder(args):
    from .networks import Encoder
    encoder = Encoder(ch=args.ngf, ch_mult=args.ngf_mult, num_res_blocks=args.num_res_blks, in_channels=args.in_ch, resolution=args.resolution, z_channels=args.z_dim, attn_resolutions=args.attn_resolutions, double_z=args.double_z)
    return encoder
def define_decoder(args):
    from .networks import Decoder
    decoder = Decoder(ch=args.ngf, out_ch=args.in_ch, ch_mult=args.ngf_mult, in_channels=args.in_ch, resolution=args.resolution, z_channels=args.z_dim, num_res_blocks=args.num_res_blks, attn_resolutions=args.attn_resolutions)
    return decoder
def define_quantizer(args):
    from .networks import Quantizer
    quantizer = Quantizer(n_embed=args.n_embed, embed_dim=args.embed_dim, beta=args.beta)
    return quantizer
def define_D(args):
    from .networks import NLayerDiscriminator
    D = NLayerDiscriminator(input_nc=args.in_ch, ndf=args.ndf, n_layers=args.D_n_layers, use_actnorm=args.D_use_actnorm)
    D.apply(weights_init)
    return D
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
def count_params(model):
    n_params = 0
    for p in model.parameters():
        n_params += p.numel()
    return n_params
class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()