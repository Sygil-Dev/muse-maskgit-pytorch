from dataclasses import dataclass, field
from functools import cached_property
from os import PathLike
from typing import List, Optional, Tuple, Union, Dict

import torch
from beartype import beartype
from torch import Tensor
from transformers import T5Config, T5EncoderModel, T5Tokenizer


# dataclass for T5 model info
@dataclass
class T5ModelInfo:
    name: str
    cache_dir: Optional[PathLike] = None
    dtype: Optional[torch.dtype] = torch.float32
    config: T5Config = field(init=False)

    def __post_init__(self):
        self.config = T5Config.from_pretrained(self.name, cache_dir=self.cache_dir)
        self._model = None
        self._tokenizer = None

    # Using cached_property to avoid loading the model/tokenizer until needed
    @cached_property
    def model(self) -> T5EncoderModel:
        if not self._model:
            self._model = T5EncoderModel.from_pretrained(
                self.name, cache_dir=self.cache_dir, torch_dtype=self.dtype
            )
        return self._model

    @cached_property
    def tokenizer(self) -> T5Tokenizer:
        if not self._tokenizer:
            self._tokenizer = T5Tokenizer.from_pretrained(
                self.name, cache_dir=self.cache_dir, torch_dtype=self.dtype
            )
        return self._tokenizer


# config
MAX_LENGTH = 256
DEFAULT_T5_NAME = "google/t5-v1_1-base"
T5_OBJECTS: Dict[str, T5ModelInfo] = {}


def get_model_and_tokenizer(
    name: str, cache_path: Optional[PathLike] = None, dtype: torch.dtype = torch.float32
) -> Tuple[T5EncoderModel, T5Tokenizer]:
    global T5_OBJECTS
    if name not in T5_OBJECTS.keys():
        T5_OBJECTS[name] = T5ModelInfo(name=name, cache_dir=cache_path, dtype=dtype)
    return T5_OBJECTS[name].model, T5_OBJECTS[name].tokenizer


def get_encoded_dim(
    name: str, cache_path: Optional[PathLike] = None, dtype: torch.dtype = torch.float32
) -> int:
    global T5_OBJECTS
    if name not in T5_OBJECTS.keys():
        T5_OBJECTS[name] = T5ModelInfo(name=name, cache_dir=cache_path, dtype=dtype)
    return T5_OBJECTS[name].config.d_model


# encoding text
@beartype
def t5_encode_text_from_encoded(
    input_ids: Tensor,
    attn_mask: Tensor,
    t5: T5EncoderModel,
    output_device: Optional[Union[torch.device, str]] = None,
) -> Tensor:
    device = t5.device
    input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
    with torch.no_grad():
        output = t5(input_ids=input_ids, attention_mask=attn_mask)
        encoded_text = output.last_hidden_state.detach()

    attn_mask = attn_mask.bool()
    encoded_text: Tensor = encoded_text.masked_fill(attn_mask[..., None], 0.0)
    return encoded_text if output_device is None else encoded_text.to(output_device)


@beartype
def t5_encode_text(
    texts: Union[str, List[str]],
    tokenizer: T5Tokenizer,
    t5: T5EncoderModel,
    output_device: Optional[Union[torch.device, str]] = None,
) -> Tensor:
    if isinstance(texts, str):
        texts = [texts]

    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
    )
    return t5_encode_text_from_encoded(encoded["input_ids"], encoded["attention_mask"], t5, output_device)
