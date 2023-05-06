import copy
import importlib
from urllib.parse import urlparse
from math import log, sqrt
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from einops import rearrange
from omegaconf import OmegaConf, DictConfig

from taming.models.vqgan import VQModel  # , GumbelVQ
from torch import nn
from tqdm_loggable.auto import tqdm

# constants
CACHE_PATH = Path.home().joinpath(".cache/taming")

VQGAN_VAE_PATH = "https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1"
VQGAN_VAE_CONFIG_PATH = "https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1"

# helpers methods


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def download(url, filename=None, root=CACHE_PATH, chunk_size=1024):
    filename = default(filename, urlparse(url).path.split("/")[-1])
    root_dir = Path(root)

    target_path = root_dir.joinpath(filename)
    if target_path.exists():
        if target_path.isfile():
            return str(target_path)
        raise RuntimeError(f"{target_path} exists and is not a regular file")

    target_tmp = target_path.with_name(f".{target_path.name}.tmp")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    filesize = int(resp.headers.get("content-length", 0))
    with target_tmp.open("wb") as f:
        for data in tqdm(
            resp.iter_content(chunk_size=chunk_size),
            desc=filename,
            total=filesize,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ):
            f.write(data)
    target_tmp.rename(target_path)
    return target_path


# VQGAN from Taming Transformers paper
# https://arxiv.org/abs/2012.09841


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class VQGanVAETaming(nn.Module):
    def __init__(self, vqgan_model_path=None, vqgan_config_path=None, accelerator: Accelerator = None):
        super().__init__()
        if accelerator is None:
            accelerator = Accelerator()

        # Download model if needed
        if vqgan_model_path is None:
            CACHE_PATH.mkdir(parents=True, exist_ok=True)
            model_filename = "vqgan.1024.model.ckpt"
            config_filename = "vqgan.1024.config.yml"
            with accelerator.local_main_process_first():
                config_path = download(VQGAN_VAE_CONFIG_PATH, config_filename)
                model_path = download(VQGAN_VAE_PATH, model_filename)
        else:
            config_path = Path(vqgan_config_path)
            model_path = Path(vqgan_model_path)

        with accelerator.local_main_process_first():
            config: DictConfig = OmegaConf.load(config_path)
            model: VQModel = instantiate_from_config(config["model"])
            state = torch.load(model_path, map_location="cpu")["state_dict"]
            model.load_state_dict(state, strict=False)

        print(f"Loaded VQGAN from {model_path} and {config_path}")
        self.model = model
        # f as used in https://github.com/CompVis/taming-transformers#overview-of-pretrained-models

        f = config.model.params.ddconfig.resolution / config.model.params.ddconfig.attn_resolutions[0]
        self.num_layers = int(log(f) / log(2))
        self.channels = 3
        self.image_size = 256
        self.num_tokens = config.model.params.n_embed
        self.is_gumbel = False  # isinstance(self.model, GumbelVQ)
        self.codebook_size = config["model"]["params"]["n_embed"]

    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        _, _, [_, _, indices] = self.model.encode(img)
        if self.is_gumbel:
            return rearrange(indices, "b h w -> b (h w)", b=b)
        return rearrange(indices, "(b n) -> b n", b=b)

    def get_encoded_fmap_size(self, image_size):
        return image_size // (2**self.num_layers)

    def decode_from_ids(self, img_seq):
        img_seq = rearrange(img_seq, "b h w -> b (h w)")
        b, n = img_seq.shape
        one_hot_indices = F.one_hot(img_seq, num_classes=self.num_tokens).float()
        z = (
            one_hot_indices @ self.model.quantize.embed.weight
            if self.is_gumbel
            else (one_hot_indices @ self.model.quantize.embedding.weight)
        )

        z = rearrange(z, "b (h w) c -> b c h w", h=int(sqrt(n)))
        img = self.model.decode(z)

        img = (img.clamp(-1.0, 1.0) + 1) * 0.5
        return img

    def encode(self, im_seq):
        # encode output
        # fmap, loss, (perplexity, min_encodings, min_encodings_indices) = self.model.encode(im_seq)
        fmap, loss, (_, _, min_encodings_indices) = self.model.encode(im_seq)

        b, _, h, w = fmap.shape
        min_encodings_indices = rearrange(min_encodings_indices, "(b h w) 1 -> b h w", h=h, w=w, b=b)
        return fmap, min_encodings_indices, loss

    def decode_ids(self, ids):
        return self.model.decode_code(ids)

    def copy_for_eval(self):
        device = next(self.parameters()).device
        vae_copy = copy.deepcopy(self.cpu())

        vae_copy.eval()
        return vae_copy.to(device)

    def forward(self, img):
        raise NotImplementedError("Forward not implemented for Taming VAE")
