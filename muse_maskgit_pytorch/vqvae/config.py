from pydantic import BaseModel, Field


class EncoderConfig(BaseModel):
    image_size: int = Field(...)
    patch_size: int = Field(...)
    dim: int = Field(...)
    depth: int = Field(...)
    num_head: int = Field(...)
    mlp_dim: int = Field(...)
    in_channels: int = Field(...)
    dim_head: int = Field(...)
    dropout: float = Field(...)


class DecoderConfig(BaseModel):
    image_size: int = Field(...)
    patch_size: int = Field(...)
    dim: int = Field(...)
    depth: int = Field(...)
    num_head: int = Field(...)
    mlp_dim: int = Field(...)
    out_channels: int = Field(...)
    dim_head: int = Field(...)
    dropout: float = Field(...)


class VQVAEConfig(BaseModel):
    n_embed: int = Field(...)
    embed_dim: int = Field(...)
    beta: float = Field(...)
    enc: EncoderConfig = Field(...)
    dec: DecoderConfig = Field(...)


VIT_S_CONFIG = VQVAEConfig(
    n_embed=8192,
    embed_dim=32,
    beta=0.25,
    enc=EncoderConfig(
        image_size=256,
        patch_size=8,
        dim=512,
        depth=8,
        num_head=8,
        mlp_dim=2048,
        in_channels=3,
        dim_head=64,
        dropout=0.0,
    ),
    dec=DecoderConfig(
        image_size=256,
        patch_size=8,
        dim=512,
        depth=8,
        num_head=8,
        mlp_dim=2048,
        out_channels=3,
        dim_head=64,
        dropout=0.0,
    ),
)
