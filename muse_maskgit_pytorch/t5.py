from os import PathLike
from typing import List, Optional, Tuple, Union

import torch
from beartype import beartype
from torch import Tensor
from transformers import T5Config, T5EncoderModel, T5Tokenizer

# config
MAX_LENGTH = 256
DEFAULT_T5_NAME = "google/t5-v1_1-base"
T5_CONFIGS = {}


# singleton globals
def get_tokenizer(name: str, cache_path: Optional[PathLike] = None) -> T5Tokenizer:
    if cache_path is not None:
        tokenizer = T5Tokenizer.from_pretrained(name, cache_dir=cache_path)
    else:
        tokenizer = T5Tokenizer.from_pretrained(name)
    return tokenizer


def get_model(name: str, cache_path: Optional[PathLike] = None) -> T5EncoderModel:
    if cache_path is not None:
        model = T5EncoderModel.from_pretrained(name, cache_dir=cache_path)
    else:
        model = T5EncoderModel.from_pretrained(name)
    return model


def get_model_and_tokenizer(name: str, cache_path: Optional[PathLike] = None) -> Tuple[T5Config, T5Tokenizer]:
    global T5_CONFIGS
    if name not in T5_CONFIGS:
        T5_CONFIGS[name] = {
            "model": get_model(name, cache_path),
            "tokenizer": get_tokenizer(name, cache_path),
        }
    else:
        if "model" not in T5_CONFIGS[name].keys():
            T5_CONFIGS[name]["model"] = get_model(name, cache_path)
        if "tokenizer" not in T5_CONFIGS[name].keys():
            T5_CONFIGS[name]["tokenizer"] = get_tokenizer(name, cache_path)
    return T5_CONFIGS[name]["model"], T5_CONFIGS[name]["tokenizer"]


def get_encoded_dim(name: str) -> int:
    global T5_CONFIGS
    if name not in T5_CONFIGS:
        # avoids loading the model if we only want to get the dim
        config: T5Config = T5Config.from_pretrained(name)
        T5_CONFIGS[name] = dict(config=config)
    elif "config" in T5_CONFIGS[name]:
        config: T5Config = T5_CONFIGS[name]["config"]
    elif "model" in T5_CONFIGS[name]:
        config: T5Config = T5_CONFIGS[name]["model"].config
    else:
        raise ValueError("Could not find config for T5 model")
    return config.d_model


# encoding text
@beartype
def t5_encode_text_from_encoded(
    input_ids: Tensor,
    attn_mask: Tensor,
    t5: T5EncoderModel,
    output_device: str = None,
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
