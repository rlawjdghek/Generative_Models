import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_network import BaseNetwork

class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        )
    def forward(self, x):
        return self.conv(x)        
class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch
        self.conv = nn.Conv2d(in_ch, out_ch, 4, 2, 1)
    def forward(self, x):
        return self.conv(x)
class LayerNorm(nn.Module):
    '''
    자연어에서의 layernorm과 비전에서의 layernorm이 다르다. 둘 다 배치는 상관없지만, 자연어에서는 단어 단위로 정규화 할 때 사용한다. 비전에서는 배치 1개에 대하여 정규화 한다.

    이전에는 자연어의 형태를 따라가기 위해 [BS x C x H x W]를 [BS x (H*W) x C]로 바꾸어 진행했었다. 이 떄에는 nn.LayerNorm(C)를 사용하여서 이 클래스를 정의하지 않았는데, 여기서는 attention에서 [BS x C x H x W]로 가기 때문에 이전과 동일한 역할을 하는 layernorm이 필요하다. 

    따라서 단어 단위를 잘 생각해보면, 비전에서는 하나의 패치가 단어가 되므로 패치 단위, 즉 point wise로 정규화를 진행하면 된다. 

    '''
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones((1,dim,1,1)))
        self.eps = 1e-5
    def forward(self, x):
        '''
        x : [BS x C x H x W]
        '''
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + self.eps).rsqrt() * self.gamma
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn
    def forward(self, x):
        x = self.norm(x)
        x = self.fn(x)
        return x
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)
class LinearAttention(nn.Module):
    '''
    Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention 참고. 아래의 Attention 클래스와 달리 softmax를 q와 k에 따로 따로 취한뒤, k와 v를 먼저 행렬곱하고, 다음에 q를 곱한다. 

    또한 여기서는 nn.Linear를 사용하지 않고 1x1conv로 구현하였다. 이 구현 방식도 알아두면 좋을듯하다. linear로 하게되면 중간중간 upsample이나 downsample 할 때 permute랑 reshape을 많이 사용할 수 밖에없음.
    '''
    def __init__(self, dim, hidden_dim=128, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.qk_scale = self.head_dim ** -0.5
        self.qkv_layer = nn.Conv2d(dim, hidden_dim*3, 1, 1, 0, bias=False)
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            LayerNorm(dim)
        )
    def forward(self, x):
        BS, C, H, W = x.shape
        q,k,v = self.qkv_layer(x).reshape(BS, 3, self.n_heads, self.head_dim, -1).permute(1,0,2,3,4)  # [BS x n_heads x head_dim x (H*W)]
        q = F.softmax(q, dim=-2)
        k = F.softmax(k, dim=-1)
        q = q * self.qk_scale
        v = v / (H*W)
        
        context = k@v.transpose(-2, -1)  # [BS x n_heads x head_dim x head_dim]

        out = context.transpose(-2, -1) @ q  # [BS x n_heads x head_dim x (H*W)]
        out = out.reshape(BS, -1, H, W)
        return self.proj(out)
class Attention(nn.Module):
    def __init__(self, dim, hidden_dim=128, n_heads=4, qk_scale=16):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.qk_scale = qk_scale
        self.qkv_layer = nn.Conv2d(dim, hidden_dim*3, 1, 1, 0, bias=False)
        self.proj = nn.Conv2d(hidden_dim, dim, 1, 1, 0)
    def forward(self, x):
        BS, C, H, W = x.shape
        q,k,v = self.qkv_layer(x).reshape(BS, 3, self.n_heads, self.head_dim, -1).permute(1,0,2,3,4)  # [BS x n_head x head_dim x (H*W)]
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        attn = q.transpose(-2, -1) @ k 
        attn = F.softmax(attn, dim=-1)  # [BS x n_head x (H*W) x (H*W)]
        
        out = attn @ v.transpose(-2, -1)  # [BS x n_head x (H*W) x head_dim]
        out = out.permute(0,1,3,2).contiguous().reshape(BS, -1, H, W)
        return self.proj(out)
class SinusoidalPE(nn.Module):  # 여기에서는 특이하게도 시간 t를 받는다. t는 pos와 같다. 따라서 log를 잘 풀어보면 결구에는 sin(pos/10000**(2*i / d))가 된다. 
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half_dim = self.dim // 2
        embedding = math.log(10000) / (half_dim-1)
        embedding = torch.exp(torch.arange(half_dim) * -embedding).cuda(t.get_device())
        embedding = t[:, None] * embedding[None, :]
        embedding = torch.cat([embedding.sin(), embedding.cos()], dim=-1)
        return embedding
class Blk(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.norm = nn.GroupNorm(groups, out_ch)
        self.act = nn.SiLU()
    def forward(self, x, scale_shift=None):
        x = self.norm(self.conv(x))
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x        
class ResnetBlk(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=None, groups=8):
        super().__init__()
        if time_dim is not None:
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, out_ch*2)
            )
        else:
            self.mlp = None
        self.blk1 = Blk(in_ch, out_ch, groups=groups)
        self.blk2 = Blk(out_ch, out_ch, groups=groups)
        self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1, 1, 0) if in_ch != out_ch else nn.Identity()
    def forward(self, x, time_embedding=None):
        if (self.mlp is not None) and (time_embedding is not None):
            time_embedding = self.mlp(time_embedding).unsqueeze(-1).unsqueeze(-1)
            scale_shift = time_embedding.chunk(2, dim=1)
        h = self.blk1(x, scale_shift)
        h = self.blk2(h)
        return h + self.conv_shortcut(x)
class UNet(BaseNetwork):
    def __init__(self, ngf, ngf_mults=(1,2,4,8), in_ch=3, out_ch=3, self_condition=False, resnet_blk_group_bn=8):
        super().__init__()
        self.self_condition = self_condition
        
        in_ch = in_ch if not self_condition else in_ch*2
        self.in_ch = in_ch
        self.init_conv = nn.Conv2d(in_ch, ngf, 7, 1, 3)
        ngfs = [ngf, *map(lambda m: ngf*m, ngf_mults)]
        in_out_dims = list(zip(ngfs[:-1], ngfs[1:]))
        sinu_pe_layer = SinusoidalPE(ngf)
        time_dim = ngf*4
        self.time_mlp = nn.Sequential(
            sinu_pe_layer,
            nn.Linear(ngf, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.down_blks = nn.ModuleList()
        self.up_blks = nn.ModuleList()
        
        for i, (in_dim, out_dim) in enumerate(in_out_dims):
            is_last = i == (len(in_out_dims) - 1)
            self.down_blks.append(nn.ModuleList([
                ResnetBlk(in_dim, in_dim, time_dim=time_dim, groups=resnet_blk_group_bn),
                ResnetBlk(in_dim, in_dim, time_dim=time_dim, groups=resnet_blk_group_bn),
                Residual(PreNorm(in_dim, LinearAttention(in_dim))),
                Downsample(in_dim, out_dim) if not is_last else nn.Conv2d(in_dim, out_dim, 3, 1, 1)
            ]))
        mid_ch = ngfs[-1]
        self.mid_blk1 = ResnetBlk(mid_ch, mid_ch, time_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_ch, Attention(mid_ch)))
        self.mid_blk2 = ResnetBlk(mid_ch, mid_ch, time_dim=time_dim)

        for i, (out_dim, in_dim) in enumerate(in_out_dims[::-1]):
            is_last = i == (len(in_out_dims) - 1)
            self.up_blks.append(nn.ModuleList([
                ResnetBlk(in_dim+out_dim, in_dim, time_dim=time_dim, groups=resnet_blk_group_bn),
                ResnetBlk(in_dim+out_dim, in_dim, time_dim=time_dim, groups=resnet_blk_group_bn),
                Residual(PreNorm(in_dim, LinearAttention(in_dim))),
                Upsample(in_dim, out_dim) if not is_last else nn.Conv2d(in_dim, out_dim, 3, 1, 1)
            ]))
        last_out_dim, _ = in_out_dims[0]
        self.last_resblk = ResnetBlk(last_out_dim*2, last_out_dim, time_dim=time_dim, groups=resnet_blk_group_bn)
        self.last_conv = nn.Conv2d(last_out_dim, out_ch, 1, 1, 0)
    def forward(self, x, t, x_self_cond=None):
        if self.self_condition:
            x_self_cond = x_self_cond if x_self_cond is not None else torch.zeros_like(x)
            x = torch.cat([x, x_self_cond], dim=1)
        x = self.init_conv(x)
        skip = x.clone()
        t = self.time_mlp(t)

        hiddens = []
        for blk1, blk2, attn, down in self.down_blks:
            x = blk1(x, t)
            hiddens.append(x)

            x = blk2(x, t)
            x = attn(x)
            hiddens.append(x)

            x = down(x)
        x = self.mid_blk1(x, t)
        x = self.mid_attn(x)
        x = self.mid_blk2(x, t)

        for blk1, blk2, attn, up in self.up_blks:
            x = torch.cat([x, hiddens.pop()], dim=1)
            x = blk1(x, t)

            x = torch.cat([x, hiddens.pop()], dim=1)
            x = blk2(x, t)
            x = attn(x)
            
            x = up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.last_resblk(x, t)
        x = self.last_conv(x)
        return x
class Coef(BaseNetwork):
    def __init__(self, beta_schedule, n_timesteps, p2_loss_weight_k, p2_loss_weight_gamma):
        super().__init__()
        # 논문에서 사용되는 계수들 등록
        if beta_schedule == "linear":
            from .base_model import linear_beta_schedule
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == "cosine":
            from .base_model import cosine_beta_schedule
            betas = cosine_beta_schedule(n_timesteps)
        else:
            raise NotImplementedError(f"Unknown beta schedule {beta_schedule}")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value=1.0)
    
        register_buffer = lambda name, val : self.register_buffer(name, val.to(torch.float32))
        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        
        posterior_variance = (1 - alphas_cumprod_prev) / (1 - alphas_cumprod) * betas  # eq. 7
        register_buffer("posterior_variance", posterior_variance)
        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer("posterior_mean_coef1", (torch.sqrt(alphas_cumprod_prev) * betas) / (1.0 - alphas_cumprod))  # eq. 7
        register_buffer("posterior_mean_coef2", (torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev)) / (1.0 - alphas_cumprod))  # eq. 7
        register_buffer("p2_loss_weight", (p2_loss_weight_k + alphas_cumprod / (1.0 - alphas_cumprod)) ** (-p2_loss_weight_gamma))

        # my_print = lambda x: print(f"{x} : {getattr(self, x)[:5]}")
        # my_print("p2_loss_weight")
        # my_print("posterior_mean_coef2")
        # my_print("posterior_mean_coef1")
        # my_print("posterior_log_variance_clipped")
        # my_print("posterior_variance")
        # my_print("sqrt_recipm1_alphas_cumprod")
        # my_print("sqrt_recip_alphas_cumprod")
        # my_print("log_one_minus_alphas_cumprod")
        # my_print("sqrt_one_minus_alphas_cumprod")
        # my_print("sqrt_alphas_cumprod")
        # my_print("alphas_cumprod_prev")
        # my_print("alphas_cumprod")
        # exit()
