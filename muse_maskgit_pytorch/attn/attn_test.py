import torch
from torch import BoolTensor, FloatTensor, allclose, arange, manual_seed, no_grad, randn
from torch.nn.functional import pad

from muse_maskgit_pytorch.attn.ein_attn import Attention as EinAttn
from muse_maskgit_pytorch.attn.xformers_attn import Attention as XformersAttn

device = torch.device("cuda")
dtype = torch.float32
seed = 42

# realistically this would be 320 in stable-diffusion, but I'm going smaller during testing
vision_dim = 64

attn_init_params = {
    "dim": vision_dim,
    "dim_head": 64,
    # realistically this would be at least 5
    "heads": 2,
    "cross_attend": True,
    "scale": 8,
}

with no_grad():
    # seed RNG before we initialize any layers, so that both will end up with same params
    manual_seed(seed)
    ein_attn = EinAttn(**attn_init_params).to(device, dtype).eval()
    # commented-out scaled dot product attention because it didn't support flash attn, so we'll try with xformers instead.
    # manual_seed(seed)
    # sdp_attn = SDPAttn(**attn_init_params).to(device, dtype).eval()
    manual_seed(seed)
    xfo_attn = XformersAttn(**attn_init_params).to(device, dtype).eval()

    batch_size = 2

    # realistically this would be 64**2 in stable-diffusion
    vision_tokens = 32**2  # 1024

    # generate rand on-CPU for cross-platform determinism of results
    x: FloatTensor = randn(batch_size, vision_tokens, vision_dim, dtype=dtype).to(device)

    # I've said text here simply as an example of something you could cross-attend to
    text_tokens = 16  # CLIP would be 77
    # for a *general* cross-attention Module:
    # kv_in_dim could differ from q_in_dim, but this attention Module requires x and context to have same dim.
    text_dim = vision_dim
    context: FloatTensor = randn(batch_size, text_tokens, text_dim, dtype=dtype).to(device)

    # attend to just the first two tokens in each text condition (e.g. if both were uncond, so [BOS, EOS] followed by PAD tokens)
    context_mask: BoolTensor = (arange(text_tokens, device=device) < 2).expand(batch_size, -1).contiguous()

    # for xformers cutlassF kernel: masks are only supported for keys whose lengths are multiples of 8:
    # https://gist.github.com/Birch-san/0c36d228e1d4b881a06d1c6e5289d569
    # so, we add whatever we feel like to the end of the key to extend it to a multiple of 8,
    # and add "discard" tokens to the mask to get rid of the excess
    # note: muse will add an extra "null" token to our context, so we'll account for that in advance
    mask_length = context_mask.shape[-1] + 1
    extra_tokens_needed = 8 - (mask_length % 8)
    # 0-pad mask to multiple of 8 tokens
    xfo_context_mask = pad(context_mask, (0, extra_tokens_needed))
    # replicate-pad embedding to multiple of 8 tokens (mask will hide the extra tokens)
    xfo_context = pad(
        context,
        (
            0,
            0,
            0,
            extra_tokens_needed,
        ),
        "replicate",
    )

    ein_result: FloatTensor = ein_attn.forward(x, context, context_mask)
    # sdp attn works, but only supports flash attn when context_mask is None.
    # with sdp_kernel(enable_math=False):
    #     sdp_result: FloatTensor = sdp_attn.forward(x, context, context_mask)
    xfo_attn: FloatTensor = xfo_attn.forward(x, xfo_context, xfo_context_mask)

    # default rtol
    rtol = 1e-5
    # atol would normally be 1e-8
    atol = 5e-7
    # assert allclose(ein_result, sdp_result, rtol=rtol, atol=atol), f"looks like attention implementations weren't equivalent, to tolerance rtol={rtol}, atol={atol}"
    assert allclose(
        ein_result, xfo_attn, rtol=rtol, atol=atol
    ), f"looks like attention implementations weren't equivalent, to tolerance rtol={rtol}, atol={atol}"
    print(f"attention implementations returned equivalent result, to tolerance rtol={rtol}, atol={atol}")
