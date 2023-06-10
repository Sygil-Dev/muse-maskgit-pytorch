from attn.ein_attn import Attention as EinAttn
from attn.sdp_attn import Attention as SDPAttn
import torch
from torch import FloatTensor, BoolTensor, manual_seed, randn, arange, allclose, no_grad

device = torch.device('cuda')
dtype = torch.float32
seed = 42

# realistically this would be 320 in stable-diffusion, but I'm going smaller during testing
vision_dim = 64

attn_init_params = {
    'dim': vision_dim,
    'dim_head': 64,
    # realistically this would be at least 5
    'heads': 2,
    'cross_attend': True,
    'scale': 8,
}

with no_grad():
    # seed RNG before we initialize any layers, so that both will end up with same params
    manual_seed(seed)
    ein_attn = EinAttn(**attn_init_params).to(device, dtype).eval()
    manual_seed(seed)
    sdp_attn = SDPAttn(**attn_init_params).to(device, dtype).eval()

    batch_size = 2

    # realistically this would be 64**2 in stable-diffusion
    vision_tokens = 32**2 # 1024

    # generate rand on-CPU for cross-platform determinism of results
    x: FloatTensor = randn(batch_size, vision_tokens, vision_dim, dtype=dtype).to(device)

    text_tokens = 16 # CLIP would be 77
    # there's no reason why these would **have** to be the same (in stable-diffusion text_dim is 768)
    # but lucid didn't expose any separate param for customizing the cross attention input dim.
    # easily fixed, but whatever I'll work with what's there.
    text_dim = vision_dim
    context: FloatTensor = randn(batch_size, text_tokens, text_dim, dtype=dtype).to(device)

    # attend to just the first two tokens in each text condition (e.g. if both were uncond, so [BOS, EOS] followed by PAD tokens)
    context_mask: BoolTensor = (arange(text_tokens, device=device) < 2).expand(batch_size, -1)
    # context_mask = None

    ein_result: FloatTensor = ein_attn.forward(x, context, context_mask)
    sdp_result: FloatTensor = sdp_attn.forward(x, context, context_mask)

    # default relative and absolute tolerance
    rtol=1e-5
    atol=5e-7
    assert allclose(ein_result, sdp_result, rtol=rtol, atol=atol), f"looks like attention implementations weren't equivalent, to tolerance rtol={rtol}, atol={atol}"
    print(f'attention implementations returned equivalent result, to tolerance rtol={rtol}, atol={atol}')