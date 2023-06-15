import logging

import torch
import torch.nn as nn
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config

from .layers import Decoder, Encoder
from .quantize import VectorQuantize

logger = logging.getLogger(__name__)


class VQVAE(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, n_embed, embed_dim, beta, enc, dec, **kwargs):
        super().__init__()
        self.encoder = Encoder(**enc)
        self.decoder = Decoder(**dec)

        self.prev_quant = nn.Linear(enc["dim"], embed_dim)
        self.quantizer = VectorQuantize(n_embed, embed_dim, beta)
        self.post_quant = nn.Linear(embed_dim, dec["dim"])

    def freeze(self):
        self.eval()
        self.requires_grad_(False)

    def encode(self, x):
        x = self.encoder(x)
        x = self.prev_quant(x)
        x, loss, indices = self.quantizer(x)
        return x, loss, indices

    def decode(self, x):
        x = self.post_quant(x)
        x = self.decoder(x)
        return x.clamp(-1.0, 1.0)

    def forward(self, inputs: torch.FloatTensor):
        z, loss, _ = self.encode(inputs)
        rec = self.decode(z)
        return rec, loss

    def encode_to_ids(self, inputs):
        _, _, indices = self.encode(inputs)
        return indices

    def decode_from_ids(self, indice):
        z_q = self.quantizer.decode_ids(indice)
        img = self.decode(z_q)
        return img

    def __call__(self, inputs: torch.FloatTensor):
        return self.forward(inputs)
