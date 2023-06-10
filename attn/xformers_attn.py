from torch import nn, FloatTensor
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional
from xformers.ops import memory_efficient_attention

def l2norm(t):
    return F.normalize(t, dim=-1)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, cross_attend=False, scale=8):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        self.cross_attend = cross_attend
        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, heads, 1, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        typical_scale = dim_head ** -.5
        scale_ratio = scale/typical_scale
        self.q_scale = nn.Parameter(torch.full((dim_head,), scale_ratio))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: FloatTensor, context: Optional[FloatTensor]=None, context_mask=None):
        assert (context is None) != self.cross_attend

        h = self.heads
        x = self.norm(x)

        kv_input = context if self.cross_attend else x

        q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1))

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=h), (q, k, v))

        nk, nv = self.null_kv
        nk, nv = map(lambda t: repeat(t, "h 1 d -> b 1 h d", b=x.shape[0]), (nk, nv))

        k = torch.cat((nk, k), dim=-3)
        v = torch.cat((nv, v), dim=-3)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        if context_mask is None:
            attn_bias = None
        else:
            context_mask = F.pad(context_mask, (1, 0), value=True)
            context_mask = rearrange(context_mask, "b j -> b 1 1 j")
            attn_bias = torch.where(context_mask == True, 0., -10000.)
            attn_bias = attn_bias.expand(-1, h, q.size(1), -1)

        out: FloatTensor = memory_efficient_attention(q, k, v, attn_bias)

        out = rearrange(out, "b n h d -> b n (h d)")
        return self.to_out(out)