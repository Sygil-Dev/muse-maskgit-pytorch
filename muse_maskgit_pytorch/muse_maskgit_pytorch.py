import math
from functools import partial
from os import PathLike
from pathlib import Path
from random import random
from typing import Callable, List, Optional, Union

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from accelerate import Accelerator
from beartype import beartype
from einops import rearrange, repeat
from rich import inspect
from torch import einsum, nn, isnan
from tqdm.auto import tqdm
from transformers import T5EncoderModel, T5Tokenizer
from memory_efficient_attention_pytorch import Attention as MemAttention

from .t5 import DEFAULT_T5_NAME, get_encoded_dim, get_model_and_tokenizer, t5_encode_text
from .vqgan_vae import VQGanVAE
from .vqgan_vae_taming import VQGanVAETaming


# helpers
def exists(val):
    return val is not None


def default(val, d):
    return val if val is not None else d


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def l2norm(t):
    return F.normalize(t, dim=-1)


# tensor helpers
def get_mask_subset_prob(mask, prob, min_mask=0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim=-1, keepdim=True) * prob).clamp(min=min_mask)
    logits = torch.rand((batch, seq), device=device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim=-1).float()

    num_padding = (~mask).sum(dim=-1, keepdim=True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask


# classes
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class GEGLU(nn.Module):
    """https://arxiv.org/abs/2002.05202"""

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return gate * F.gelu(x)


def FeedForward(dim, mult=4):
    """https://arxiv.org/abs/2110.09456"""

    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Linear(inner_dim, dim, bias=False),
    )


class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, cross_attend=False, scale=8):
        super().__init__()
        self.scale = scale
        self.heads = heads
        inner_dim = dim_head * heads

        self.cross_attend = cross_attend
        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, heads, 1, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, context=None, context_mask=None):
        assert not (exists(context) ^ self.cross_attend)

        h = self.heads
        x = self.norm(x)

        kv_input = context if self.cross_attend else x

        q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1))

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        nk, nv = self.null_kv
        nk, nv = map(lambda t: repeat(t, "h 1 d -> b h 1 d", b=x.shape[0]), (nk, nv))

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        if exists(context_mask):
            context_mask = rearrange(context_mask, "b j -> b 1 1 j")
            context_mask = F.pad(context_mask, (1, 0), value=True)

            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~context_mask, mask_value)

        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerBlocks(nn.Module):
    def __init__(self, *, dim, depth, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, dim_head=dim_head, heads=heads),
                        Attention(dim=dim, dim_head=dim_head, heads=heads, cross_attend=True),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = LayerNorm(dim)

    def forward(self, x, context=None, context_mask=None):
        for attn, cross_attn, ff in self.layers:
            x = attn(x) + x

            x = cross_attn(x, context=context, context_mask=context_mask) + x

            x = ff(x) + x

        return self.norm(x)


class MemoryEfficientTransformerBlocks(nn.Module):
    def __init__(self, *, dim, depth, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        MemAttention(dim=dim, dim_head=dim_head, heads=heads),
                        MemAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = LayerNorm(dim)

    def forward(self, x, context=None, mask=None):
        for attn, cross_attn, ff in self.layers:
            x = attn(x) + x

            x = cross_attn(x, context=context, mask=mask) + x

            x = ff(x) + x

        return self.norm(x)


# transformer - it's all we need
class Transformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens: int,
        dim: int,
        seq_len: int,
        dim_out: Optional[int] = None,
        t5_name: str = DEFAULT_T5_NAME,
        self_cond: bool = False,
        add_mask_id: bool = False,
        cache_path: PathLike = None,
        memory_efficient: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.mask_id = num_tokens if add_mask_id else None

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens + int(add_mask_id), dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.seq_len = seq_len
        self.memory_efficient = memory_efficient

        if memory_efficient:
            self.transformer_blocks = MemoryEfficientTransformerBlocks(dim=dim, **kwargs)
        else:
            self.transformer_blocks = TransformerBlocks(dim=dim, **kwargs)
        self.norm = LayerNorm(dim)

        self.dim_out = default(dim_out, num_tokens)
        self.to_logits = nn.Linear(dim, self.dim_out, bias=False)

        # text conditioning
        t5, tokenizer = get_model_and_tokenizer(t5_name, cache_path)
        self.t5: T5EncoderModel = t5
        self.tokenizer: T5Tokenizer = tokenizer

        self.t5.eval()

        text_embed_dim = get_encoded_dim(t5_name)

        self.text_embed_proj = (
            nn.Linear(text_embed_dim, dim, bias=False) if text_embed_dim != dim else nn.Identity()
        )

        # optional self conditioning
        self.self_cond = self_cond
        self.self_cond_to_init_embed = FeedForward(dim)

    def encode_text(self, *args, **kwargs):
        kwargs.update(tokenizer=self.tokenizer, t5=self.t5)
        return t5_encode_text(*args, **kwargs)

    def forward_with_cond_scale(self, *args, cond_scale=3.0, return_embed=False, **kwargs):
        if cond_scale == 1:
            return self.forward(*args, return_embed=return_embed, cond_drop_prob=0.0, **kwargs)

        logits, embed = self.forward(*args, return_embed=True, cond_drop_prob=0.0, **kwargs)
        null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        return (scaled_logits, embed) if return_embed else scaled_logits

    def forward_with_neg_prompt(
        self,
        *args,
        text_embed: torch.Tensor,
        neg_text_embed: torch.Tensor,
        cond_scale=3.0,
        return_embed=False,
        **kwargs,
    ):
        neg_logits = self.forward(*args, neg_text_embed=neg_text_embed, cond_drop_prob=0.0, **kwargs)
        pos_logits, embed = self.forward(
            *args, return_embed=True, text_embed=text_embed, cond_drop_prob=0.0, **kwargs
        )

        scaled_logits = neg_logits + (pos_logits - neg_logits) * cond_scale

        if return_embed:
            return scaled_logits, embed

        return scaled_logits

    def forward(
        self,
        x,
        return_embed=False,
        return_logits=False,
        labels=None,
        ignore_index=0,
        self_cond_embed=None,
        cond_drop_prob=0.0,
        conditioning_token_ids: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[torch.Tensor] = None,
    ):
        device, b, n = x.device, *x.shape
        assert n <= self.seq_len

        # prepare texts

        if texts is not None and text_embeds is not None:
            raise ValueError("only one of texts or text_embeds should be passed in")

        if texts is not None:
            text_embeds = self.encode_text(texts)

        context = self.text_embed_proj(text_embeds)

        context_mask = (text_embeds != 0).any(dim=-1)

        # classifier free guidance

        if self.training and cond_drop_prob > 0.0:
            mask = prob_mask_like((b, 1), 1.0 - cond_drop_prob, device)
            context_mask = context_mask & mask

        # concat conditioning image token ids if needed

        if exists(conditioning_token_ids):
            conditioning_token_ids = rearrange(conditioning_token_ids, "b ... -> b (...)")
            cond_token_emb = self.token_emb(conditioning_token_ids)
            context = torch.cat((context, cond_token_emb), dim=-2)
            context_mask = F.pad(context_mask, (0, conditioning_token_ids.shape[-1]), value=True)

        # embed tokens

        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(n, device=device))

        if self.self_cond:
            if not exists(self_cond_embed):
                self_cond_embed = torch.zeros_like(x)
            x = x + self.self_cond_to_init_embed(self_cond_embed)

        if self.memory_efficient:
            embed = self.transformer_blocks(x, context=context, mask=context_mask)
        else:
            embed = self.transformer_blocks(x, context=context, context_mask=context_mask)

        logits = self.to_logits(embed)

        if return_embed:
            return logits, embed

        if not exists(labels):
            return logits

        if self.dim_out == 1:
            loss = F.binary_cross_entropy_with_logits(rearrange(logits, "... 1 -> ..."), labels)
        else:
            loss = F.cross_entropy(rearrange(logits, "b n c -> b c n"), labels, ignore_index=ignore_index)

        if not return_logits:
            return loss

        return loss, logits


# self critic wrapper
class SelfCritic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.to_pred = nn.Linear(net.dim, 1)

    def forward_with_cond_scale(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_cond_scale(x, *args, return_embed=True, **kwargs)
        return self.to_pred(embeds)

    def forward_with_neg_prompt(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_neg_prompt(x, *args, return_embed=True, **kwargs)
        return self.to_pred(embeds)

    def forward(self, x, *args, labels=None, **kwargs):
        _, embeds = self.net(x, *args, return_embed=True, **kwargs)
        logits = self.to_pred(embeds)

        if not exists(labels):
            return logits

        logits = rearrange(logits, "... 1 -> ...")
        return F.binary_cross_entropy_with_logits(logits, labels)


# specialized transformers
class MaskGitTransformer(Transformer):
    def __init__(self, *args, **kwargs):
        if kwargs.pop("add_mask_id", True) is not True:
            raise ValueError("MaskGitTransformer does not accept add_mask_id argument")
        super().__init__(*args, add_mask_id=True, **kwargs)


class TokenCritic(Transformer):
    def __init__(self, *args, **kwargs):
        if kwargs.pop("dim_out", 1) != 1:
            raise ValueError("TokenCritic does not accept dim_out argument")
        super().__init__(*args, dim_out=1, **kwargs)


# classifier free guidance functions
def uniform(shape, min=0, max=1, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def prob_mask_like(shape, prob, device=None):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob


# sampling helpers


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(2, ind, val)
    return probs


# noise schedules


def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


# main maskgit classes


@beartype
class MaskGit(nn.Module):
    def __init__(
        self,
        image_size,
        transformer: MaskGitTransformer,
        accelerator: Optional[Accelerator] = None,
        noise_schedule: Callable = cosine_schedule,
        token_critic: Optional[TokenCritic] = None,
        self_token_critic: bool = False,
        vae: Optional[Union[VQGanVAE, VQGanVAETaming]] = None,
        cond_vae: Optional[Union[VQGanVAE, VQGanVAETaming]] = None,
        cond_image_size: Optional[int] = None,
        cond_drop_prob: float = 0.5,
        self_cond_prob: float = 0.9,
        no_mask_token_prob: float = 0.0,
        critic_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.accelerator = accelerator

        self.vae = vae.copy_for_eval() if vae is not None else None

        if cond_vae is not None:
            if cond_image_size is None:
                raise ValueError("cond_image_size must be specified if conditioning")
            self.cond_vae = cond_vae.eval()
        else:
            self.cond_vae = self.vae

        self.image_size = image_size
        self.cond_image_size = cond_image_size
        self.resize_image_for_cond_image = exists(cond_image_size)
        self.cond_drop_prob = cond_drop_prob

        self.transformer = transformer
        self.self_cond = transformer.self_cond
        if not self.vae.codebook_size == self.cond_vae.codebook_size == transformer.num_tokens:
            raise ValueError("transformer num_tokens must be set to be equal to the vae codebook size")

        self.mask_id = transformer.mask_id
        self.noise_schedule = noise_schedule

        if token_critic and self_token_critic:
            raise ValueError("cannot have both self_token_critic and token_critic")
        self.token_critic = SelfCritic(transformer) if self_token_critic else token_critic
        self.critic_loss_weight = critic_loss_weight

        # self conditioning
        self.self_cond_prob = self_cond_prob

        # percentage of tokens to be [mask]ed to remain the same token, so that transformer produces better embeddings across all tokens as done in original BERT paper
        # may be needed for self conditioning
        self.no_mask_token_prob = no_mask_token_prob

    @property
    def device(self):
        return self.accelerator.device if self.accelerator else next(self.parameters()).device

    def save(self, path):
        if self.accelerator:
            self.accelerator.save(self.state_dict(), path)
        else:
            torch.save(self.state_dict(), path)

    def load(self, path):
        path = Path(path)
        if not path.exists() and path.is_file():
            raise ValueError(f"cannot find file {path} (does not exist or is not a file)")
        state_dict = torch.load(str(path), map_location="cpu")
        self.load_state_dict(state_dict)

    def print(self, *args, **kwargs):
        return self.accelerator.print(*args, **kwargs) if self.accelerator else print(*args, **kwargs)

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        texts: List[str],
        negative_texts: Optional[List[str]] = None,
        cond_images: Optional[torch.Tensor] = None,
        fmap_size=None,
        temperature=1.0,
        topk_filter_thres=0.9,
        can_remask_prev_masked=False,
        force_not_use_token_critic=False,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        cond_scale=3,
        critic_noise_scale=1,
    ):
        fmap_size = default(fmap_size, self.vae.get_encoded_fmap_size(self.image_size))

        # begin with all image token ids masked

        device = next(self.parameters()).device

        seq_len = fmap_size**2

        batch_size = len(texts)

        shape = (batch_size, seq_len)

        ids = torch.full(shape, self.mask_id, dtype=torch.long, device=device)
        scores = torch.zeros(shape, dtype=torch.float32, device=device)

        starting_temperature = temperature

        cond_ids = None

        text_embeds = self.transformer.encode_text(texts)

        demask_fn = self.transformer.forward_with_cond_scale

        # whether to use token critic for scores

        use_token_critic = exists(self.token_critic) and not force_not_use_token_critic

        if use_token_critic:
            token_critic_fn = self.token_critic.forward_with_cond_scale

        # negative prompting, as in paper

        neg_text_embeds = None
        if exists(negative_texts):
            assert len(texts) == len(negative_texts)

            neg_text_embeds = self.transformer.encode_text(negative_texts)
            demask_fn = partial(
                self.transformer.forward_with_neg_prompt,
                neg_text_embeds=neg_text_embeds,
            )

            if use_token_critic:
                token_critic_fn = partial(
                    self.token_critic.forward_with_neg_prompt,
                    neg_text_embeds=neg_text_embeds,
                )

        if self.resize_image_for_cond_image:
            if cond_images is None:
                raise ValueError("conditioning image must be passed in to generate for super res maskgit")
            with torch.no_grad():
                _, cond_ids, _ = self.cond_vae.encode(cond_images)

        self_cond_embed = None

        for timestep, steps_until_x0 in tqdm(
            zip(
                torch.linspace(0, 1, timesteps, device=device),
                reversed(range(timesteps)),
            ),
            total=timesteps,
        ):
            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = max(int((rand_mask_prob * seq_len).item()), 1)

            masked_indices = scores.topk(num_token_masked, dim=-1).indices

            ids = ids.scatter(1, masked_indices, self.mask_id)

            logits, embed = demask_fn(
                ids,
                text_embeds=text_embeds,
                self_cond_embed=self_cond_embed,
                conditioning_token_ids=cond_ids,
                cond_scale=cond_scale,
                return_embed=True,
            )

            self_cond_embed = embed if self.self_cond else None

            filtered_logits = top_k(logits, topk_filter_thres)

            temperature = starting_temperature * (steps_until_x0 / timesteps)  # temperature is annealed

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            is_mask = ids == self.mask_id

            ids = torch.where(is_mask, pred_ids, ids)

            if use_token_critic:
                scores = token_critic_fn(
                    ids,
                    text_embeds=text_embeds,
                    conditioning_token_ids=cond_ids,
                    cond_scale=cond_scale,
                )

                scores = rearrange(scores, "... 1 -> ...")

                scores = scores + (uniform(scores.shape, device=device) - 0.5) * critic_noise_scale * (
                    steps_until_x0 / timesteps
                )

            else:
                probs_without_temperature = logits.softmax(dim=-1)

                scores = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
                scores = rearrange(scores, "... 1 -> ...")

                if not can_remask_prev_masked:
                    scores = scores.masked_fill(~is_mask, -1e5)
                else:
                    assert (
                        self.no_mask_token_prob > 0.0
                    ), "without training with some of the non-masked tokens forced to predict, not sure if the logits will be meaningful for these token"

        # get ids

        ids = rearrange(ids, "b (i j) -> b i j", i=fmap_size, j=fmap_size)

        if not exists(self.vae):
            return ids

        images = self.vae.decode_from_ids(ids)
        return images

    def forward(
        self,
        images_or_ids: torch.Tensor,
        ignore_index=-1,
        cond_images: Optional[torch.Tensor] = None,
        cond_token_ids: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[torch.Tensor] = None,
        cond_drop_prob=None,
        train_only_generator=False,
        sample_temperature=None,
    ):
        # tokenize if needed
        if images_or_ids.dtype == torch.float:
            if self.vae is None:
                raise ValueError("you must pass in a vae if you want to train from raw images")

            if not all([height_or_width == self.image_size for height_or_width in images_or_ids.shape[-2:]]):
                raise ValueError("the image you passed in is not of the correct dimensions")

            with torch.no_grad():
                _, ids, _ = self.vae.encode(images_or_ids)
        elif self.resize_image_for_cond_image is True:
            raise ValueError(
                "you cannot pass in raw image token ids if you want autoresizing of images for conditioning"
            )
        else:
            ids = images_or_ids

        # validate text embedding arguments
        if text_embeds is not None and texts is not None:
            raise ValueError("cannot pass in both text and text embeddings")
        elif text_embeds is None and texts is None:
            raise ValueError("must pass in either text or text embeddings")

        # get some basic variables
        ids = rearrange(ids, "b ... -> b (...)")
        batch, seq_len, device, cond_drop_prob = (
            *ids.shape,
            ids.device,
            default(cond_drop_prob, self.cond_drop_prob),
        )

        # take care of creating conditioning image if required
        if self.resize_image_for_cond_image:
            cond_images = F.interpolate(images_or_ids, self.cond_image_size, mode="nearest")

        # tokenize conditional images if needed
        if cond_images is not None:
            if cond_token_ids is not None:
                raise ValueError(
                    "if conditioning on low resolution, cannot pass in both images and token ids"
                )
            if self.cond_vae is None:
                raise ValueError(
                    "you must pass in a cond vae if you want to condition on low resolution images"
                )

            assert all(
                [height_or_width == self.cond_image_size for height_or_width in cond_images.shape[-2:]]
            )

            with torch.no_grad():
                _, cond_token_ids, _ = self.cond_vae.encode(cond_images)

        # prepare mask
        rand_time = uniform((batch,), device=self.device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (seq_len * rand_mask_probs).round().clamp(min=1)

        mask_id = self.mask_id
        batch_randperm = torch.rand((batch, seq_len), device=self.device).argsort(dim=-1)
        mask = batch_randperm < rearrange(num_token_masked, "b -> b 1")

        mask_id = self.transformer.mask_id
        labels = torch.where(mask, ids, ignore_index)

        if self.no_mask_token_prob > 0.0:
            no_mask_mask = get_mask_subset_prob(mask, self.no_mask_token_prob)
            mask &= ~no_mask_mask

        x: torch.Tensor = torch.where(mask, mask_id, ids)

        # encode text if needed
        if text_embeds is None and texts is not None:
            text_embeds = self.transformer.encode_text(texts)

        # make sure we have text embeddings now
        if text_embeds is None:
            raise ValueError("No text embeddings found, if text was passed it did not encode correctly")

        # self conditioning
        self_cond_embed = None
        if self.transformer.self_cond and random() < self.self_cond_prob:
            with torch.no_grad():
                _, self_cond_embed = self.transformer(
                    x,
                    text_embeds=text_embeds,
                    conditioning_token_ids=cond_token_ids,
                    cond_drop_prob=0.0,
                    return_embed=True,
                )

                self_cond_embed.detach_()
        # get loss
        ce_loss, logits = self.transformer(
            x,
            text_embeds=text_embeds,
            self_cond_embed=self_cond_embed,
            conditioning_token_ids=cond_token_ids,
            labels=labels,
            cond_drop_prob=cond_drop_prob,
            ignore_index=ignore_index,
            return_logits=True,
        )
        if isnan(ce_loss):
            self.print(f"ERROR: found NaN loss: {ce_loss}")
            raise ValueError("NaN loss")

        if not exists(self.token_critic) or train_only_generator:
            return ce_loss

        # token critic loss

        sampled_ids = gumbel_sample(logits, temperature=default(sample_temperature, random()))

        critic_input = torch.where(mask, sampled_ids, x)
        critic_labels = (ids != critic_input).float()

        bce_loss = self.token_critic(
            critic_input,
            text_embeds=text_embeds,
            conditioning_token_ids=cond_token_ids,
            labels=critic_labels,
            cond_drop_prob=cond_drop_prob,
        )

        return ce_loss + self.critic_loss_weight * bce_loss


# final Muse class


@beartype
class Muse(nn.Module):
    def __init__(self, base: MaskGit, superres: MaskGit):
        super().__init__()
        self.base_maskgit = base.eval()

        assert superres.resize_image_for_cond_image
        self.superres_maskgit = superres.eval()

    @torch.no_grad()
    def forward(
        self,
        texts: List[str],
        cond_scale=3.0,
        temperature=1.0,
        timesteps=18,
        superres_timesteps=None,
        return_lowres=False,
        return_pil_images=True,
    ):
        lowres_image = self.base_maskgit.generate(
            texts=texts,
            cond_scale=cond_scale,
            temperature=temperature,
            timesteps=timesteps,
        )

        superres_image = self.superres_maskgit.generate(
            texts=texts,
            cond_scale=cond_scale,
            cond_images=lowres_image,
            temperature=temperature,
            timesteps=default(superres_timesteps, timesteps),
        )

        if return_pil_images:
            lowres_image = list(map(T.ToPILImage(), lowres_image))
            superres_image = list(map(T.ToPILImage(), superres_image))

        if not return_lowres:
            return superres_image

        return superres_image, lowres_image
