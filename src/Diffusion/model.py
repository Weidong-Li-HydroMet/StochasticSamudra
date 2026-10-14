"""
Model definition for the 1-degree conditional diffusion model.

Copied from weidong/Diffusion/DiffusionDate_noDropout.py (the file the 0.25-degree
train_diffusion_025deg.py imports):
  * UNet_1degV1 and its building blocks: unchanged.
  * Diffusion_for_UNet: only the methods used for training and for the conditional
    DDIM sample are kept (noise schedule, noise_images, sample_timesteps,
    predict_start_from_v, i_posterior, sample_ddim_conditional). The inpaint /
    repaint / copaint / SDE samplers are left in the original file.
"""
import torch
import torch.nn as nn
import math
import numpy as np
from einops import rearrange
from functools import partial
from tqdm.auto import tqdm
from torch import nn, einsum, optim
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.checkpoint as checkpoint


Tensor = torch.Tensor


def Downsample_1deg(dim_in, dim_out, scale=2):
    class DownsampleModule(nn.Module):
        def __init__(self):
            super(DownsampleModule, self).__init__()
            self.maxpool = nn.MaxPool2d(scale)
            self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=1)

        def forward(self, x):
            pad_h = (scale - x.shape[2] % scale) % scale
            if pad_h > 0:
                x = F.pad(x, (0, 0, 0, pad_h))
            x = self.maxpool(x)
            x = self.conv(x)
            return x

    return DownsampleModule()


def Upsample_1deg(dim_in, dim_out, scale=2):
    class UpsampleModule(nn.Module):
        def __init__(self):
            super(UpsampleModule, self).__init__()
            self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
            self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)

        def forward(self, x):
            x = self.upsample(x)
            x = self.conv(x)
            return x

    return UpsampleModule()



class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :] * 1000.
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    

class Block(nn.Module):
    def __init__(self, dim_in, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift!=None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x
    

class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, *, time_emb_dim=None,  groups=8):
        super().__init__()
        self.mlp_t = (nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if time_emb_dim!= None else None)
        self.block1 = Block(dim_in, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift_t = None
        if self.mlp_t!=None and time_emb!=None:
            time_emb = self.mlp_t(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift_t = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift_t)
        h = self.block2(h)
        return h + self.res_conv(x)
    

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    

class UNet_1degV1(nn.Module):
    def __init__(self, dim_in, dim_out, c, c_mults=(1, 2, 4, 8), resnet_block_groups=4, scale = [2,2,2]):
        super().__init__()
        self.init_conv = nn.Conv2d(dim_in, c, 1, padding=0)
        dims = [c*x for x in c_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        # time embeddings
        time_dim = c * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        # datetime (day-of-year) embedding for seasonality
        self.date_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (d_in, d_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(d_in, d_in, time_emb_dim=time_dim),
                        block_klass(d_in, d_in, time_emb_dim=time_dim),
                        Downsample_1deg(d_in, d_out, scale = scale[ind])
                        if not is_last
                        else nn.Conv2d(d_in, d_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (d_in, d_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(d_out + d_in, d_out, time_emb_dim=time_dim),
                        block_klass(d_out + d_in, d_out, time_emb_dim=time_dim),
                        Upsample_1deg(d_out, d_in, scale = scale[-ind-1])
                        if not is_last
                        else nn.Conv2d(d_out, d_in, 3, padding=1),
                    ]
                )
            )

        self.final_res_block = block_klass(c * 2, c, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(c, dim_out, 1)

    def forward(self, x, time, date_frac=None):
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        # Add datetime (day-of-year) embedding for seasonality
        if date_frac is not None:
            t = t + self.date_mlp(date_frac)
        h = []
        #for block1, block2, attn, downsample in self.downs:
        # for block1, block2, downsample in self.downs:
        #     x = block1(x, t)
        #     h.append(x)
        #     x = block2(x, t)
        #     h.append(x)
        #     x = downsample(x)
        for block1, block2, downsample in self.downs:
            x = checkpoint.checkpoint(block1, x, t)
            h.append(x)
            x = checkpoint.checkpoint(block2, x, t)
            h.append(x)
            x = downsample(x)

        # x = self.mid_block1(x, t)
        # x = self.mid_block2(x, t)
        x = checkpoint.checkpoint(self.mid_block1, x, t)
        x = checkpoint.checkpoint(self.mid_block2, x, t)

        #for block1, block2, attn, upsample in self.ups:
        # for block1, block2, upsample in self.ups:
        #     x = torch.cat((x, h.pop()), dim=1)
        #     x = block1(x, t)
        #     x = torch.cat((x, h.pop()), dim=1)
        #     x = block2(x, t)
        #     x = upsample(x)
        for block1, block2, upsample in self.ups:
            skip = h.pop()
            x = x[:, :, :skip.shape[2], :skip.shape[3]]
            x = torch.cat((x, skip), dim=1)
            x = checkpoint.checkpoint(block1, x, t)
            skip = h.pop()
            x = x[:, :, :skip.shape[2], :skip.shape[3]]
            x = torch.cat((x, skip), dim=1)
            x = checkpoint.checkpoint(block2, x, t)
            x = upsample(x)

        x = x[:, :, :r.shape[2], :r.shape[3]]
        x = torch.cat((x, r), dim=1)

        #x = self.final_res_block(x, t)
        x = checkpoint.checkpoint(self.final_res_block, x, t)
        return self.final_conv(x)


class Diffusion_for_UNet:
    """Continuous-time v-prediction diffusion with a cosine logSNR schedule.

    u in [0, 1]:  lambda(u) = -2 log tan(a u + b),  lambda in [-lambda_range, +lambda_range]
                  alpha^2 = sigmoid(lambda),  sigma^2 = 1 - alpha^2
    x_t = alpha x + sigma eps,   v = alpha eps - sigma x,   x_0 = alpha x_t - sigma v
    """
    def __init__(self, size, lambda_range = 20, steps = 1000, device='cuda'):
        self.size = size
        self.device = device
        self.lambda_range = lambda_range
        self.steps = steps
        self.b = math.atan(math.exp(-lambda_range/2.))
        self.a = math.atan(math.exp(lambda_range/2))-math.atan(math.exp(-lambda_range/2.))
        self.bound = 0.999

    def u2lambda(self,u):
        return -2*torch.log(torch.tan(self.a*u+self.b)).to(self.device)

    def u2alpha(self,u):
        return torch.sqrt(1./(1+torch.exp(-self.u2lambda(u)))).to(self.device)

    def u2sigma(self,u):
        return torch.sqrt(1.-1./(1+torch.exp(-self.u2lambda(u)))).to(self.device)
    
    def noise_images(self, x, u):
        eps = torch.randn_like(x).to(self.device)
        x_t = self.u2alpha(u)[:,None,None,None]*x + self.u2sigma(u)[:,None,None,None]*eps
        v = self.u2alpha(u)[:,None,None,None]*eps - self.u2sigma(u)[:,None,None,None]*x
        return x_t, eps, v
    
    def sample_timesteps(self, n):
        return torch.rand(n).to(self.device)
    
    def predict_start_from_v(self, x_t, u, v):
        if len(self.size)==3:
            sigma = self.u2sigma(u)[:,None,None,None]
            alpha = self.u2alpha(u)[:,None,None,None]
        elif len(self.size)==4:
            sigma = self.u2sigma(u)[:,None,None,None,None]
            alpha = self.u2alpha(u)[:,None,None,None,None]
        return alpha*x_t-sigma*v

    def i_posterior(self, x_0, x_t, u, u_next):
        #Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        exp = torch.exp((self.u2lambda(u)-self.u2lambda(u_next))/2.)
        posterior_mean_coef1 = (1-exp)*self.u2alpha(u_next)
        posterior_mean_coef2 = exp*self.u2alpha(u_next)/self.u2alpha(u)
        if len(self.size)==3:
            posterior_mean = posterior_mean_coef1[:,None,None,None]*x_0 + posterior_mean_coef2[:,None,None,None]*x_t
        elif len(self.size)==4:
            posterior_mean = posterior_mean_coef1[:,None,None,None,None]*x_0 + posterior_mean_coef2[:,None,None,None,None]*x_t
        return posterior_mean

    @torch.inference_mode()
    def sample_ddim_conditional(self, model, n, cond, mask=None, steps=100, cc=69, visual_bar=True, date_frac=None):
        """Conditional DDIM sampling: P(state | forcing, date).
        Args:
            model: UNet that takes cat(noisy_state, cond) as input.
            n: batch size.
            cond: forcing tensor [n, N_BOUND, H, W].
            mask: ocean mask [1, N_PROG, H, W] or [N_PROG, H, W]. Applied every step.
            steps: number of DDIM steps.
            cc: number of output channels (N_PROG).
            date_frac: day-of-year fraction for date embedding.
        """
        u_seq = torch.arange(self.bound, 0, -1./steps).to(self.device)
        u_seq = torch.cat((u_seq, torch.tensor([0]).to(self.device)))
        x = torch.randn([n, cc] + [self.size[1], self.size[2]]).to(self.device)
        iterator = tqdm(range(len(u_seq)-1), total=len(u_seq)-1) if visual_bar else range(len(u_seq)-1)
        for i in iterator:
            u = torch.ones(n).to(self.device) * u_seq[i]
            u_next = torch.ones(n).to(self.device) * u_seq[i+1]
            if mask is not None:
                x = x * mask
            x_in = torch.cat([x, cond], dim=1)
            predicted_v = model(x_in, u, date_frac=date_frac)
            if mask is not None:
                predicted_v = predicted_v * mask
            x_0 = self.predict_start_from_v(x, u, predicted_v)
            if i < steps * 0.7:
                x_0 = x_0.clamp(-12, 12)
            x = self.i_posterior(x_0, x, u, u_next)
        return x

