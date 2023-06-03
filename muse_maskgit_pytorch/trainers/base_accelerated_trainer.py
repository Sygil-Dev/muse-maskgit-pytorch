from os import PathLike
from pathlib import Path
from shutil import rmtree
from typing import Dict, Optional, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType
from beartype import beartype
from datasets import Dataset
from lion_pytorch import Lion
from torch import nn
from torch.optim import Adam, AdamW, Optimizer
from transformers.optimization import Adafactor
from torch.utils.data import DataLoader, random_split

try:
    from accelerate.data_loader import MpDeviceLoaderWrapper
except ImportError:
    MpDeviceLoaderWrapper = DataLoader
    pass

try:
    from bitsandbytes.optim import Adam8bit, AdamW8bit, Lion8bit
except ImportError:
    Adam8bit = AdamW8bit = Lion8bit = None

try:
    import wandb
except ImportError:
    wandb = None

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)


def noop(*args, **kwargs):
    pass


# helper functions


def identity(t, *args, **kwargs):
    return t


def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)


def yes_or_no(question):
    answer = input(f"{question} (y/n) ")
    return answer.lower() in ("yes", "y")


def pair(val):
    return val if isinstance(val, tuple) else (val, val)


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# image related helpers fnuctions and dataset


def get_accelerator(*args, **kwargs):
    kwargs_handlers = kwargs.get("kwargs_handlers", [])
    if ddp_kwargs not in kwargs_handlers:
        kwargs_handlers.append(ddp_kwargs)
        kwargs.update(kwargs_handlers=kwargs_handlers)
    accelerator = Accelerator(*args, **kwargs)
    return accelerator


def split_dataset(dataset: Dataset, valid_frac: float, accelerator: Accelerator, seed: int = 42):
    if valid_frac > 0:
        train_size = int((1 - valid_frac) * len(dataset))
        valid_size = len(dataset) - train_size
        ds, valid_ds = random_split(
            dataset,
            [train_size, valid_size],
            generator=torch.Generator().manual_seed(seed),
        )
        accelerator.print(
            f"training with dataset of {len(ds)} samples and validating with randomly splitted {len(valid_ds)} samples"
        )
    else:
        valid_ds = ds
        accelerator.print(f"training with shared training and valid dataset of {len(ds)} samples")
    return ds, valid_ds


# main trainer class


def get_optimizer(
    use_8bit_adam: bool,
    optimizer: str,
    parameters: dict,
    lr: float,
    weight_decay: float,
    optimizer_kwargs: dict = {},
):
    if use_8bit_adam is True and Adam8bit is None:
        print(
            "Please install bitsandbytes to use 8-bit optimizers. You can do so by running `pip install "
            "bitsandbytes` | Defaulting to non 8-bit equivalent..."
        )

    bnb_supported_optims = ["Adam", "AdamW", "Lion"]
    if use_8bit_adam and optimizer not in bnb_supported_optims:
        print(f"8bit is not supported by the {optimizer} optimizer, Using standard {optimizer} instead.")

    # optimizers
    if optimizer == "Adam":
        return (
            Adam8bit(parameters, lr=lr, weight_decay=weight_decay, **optimizer_kwargs)
            if use_8bit_adam and Adam8bit is not None
            else Adam(parameters, lr=lr, weight_decay=weight_decay, **optimizer_kwargs)
        )
    elif optimizer == "AdamW":
        return (
            AdamW8bit(parameters, lr=lr, weight_decay=weight_decay, **optimizer_kwargs)
            if use_8bit_adam and AdamW8bit is not None
            else AdamW(parameters, lr=lr, weight_decay=weight_decay, **optimizer_kwargs)
        )
    elif optimizer == "Lion":
        # Reckless reuse of the use_8bit_adam flag
        return (
            Lion8bit(parameters, lr=lr, weight_decay=weight_decay, **optimizer_kwargs)
            if use_8bit_adam and Lion8bit is not None
            else Lion(parameters, lr=lr, weight_decay=weight_decay, **optimizer_kwargs)
        )
    elif optimizer == "Adafactor":
        return Adafactor(parameters, lr=lr, weight_decay=weight_decay, relative_step=False, scale_parameter=False,  **optimizer_kwargs)
    else:
        raise NotImplementedError(f"{optimizer} optimizer not supported yet.")


@beartype
class BaseAcceleratedTrainer(nn.Module):
    def __init__(
        self,
        dataloader: Union[DataLoader, MpDeviceLoaderWrapper],
        valid_dataloader: Union[DataLoader, MpDeviceLoaderWrapper],
        accelerator: Accelerator,
        *,
        current_step: int,
        num_train_steps: int,
        max_grad_norm: Optional[int] = None,
        save_results_every: int = 100,
        save_model_every: int = 1000,
        results_dir: Union[str, PathLike] = Path.cwd().joinpath("results"),
        logging_dir: Union[str, PathLike] = Path.cwd().joinpath("results/logs"),
        apply_grad_penalty_every: int = 4,
        gradient_accumulation_steps: int = 1,
        clear_previous_experiments: bool = False,
        validation_image_scale: float = 1.0,
        only_save_last_checkpoint: bool = False,
    ):
        super().__init__()
        self.model: nn.Module = None
        # instantiate accelerator
        self.gradient_accumulation_steps: int = gradient_accumulation_steps
        self.accelerator: Accelerator = accelerator
        self.logging_dir: Path = Path(logging_dir) if not isinstance(logging_dir, Path) else logging_dir
        self.results_dir: Path = Path(results_dir) if not isinstance(results_dir, Path) else results_dir

        # training params
        self.only_save_last_checkpoint: bool = only_save_last_checkpoint
        self.validation_image_scale: float = validation_image_scale
        self.register_buffer("steps", torch.Tensor([current_step]))
        self.num_train_steps: int = num_train_steps
        self.max_grad_norm: Optional[Union[int, float]] = max_grad_norm

        self.dl = dataloader
        self.valid_dl = valid_dataloader
        self.dl_iter = iter(self.dl)
        self.valid_dl_iter = iter(self.valid_dl)

        self.save_model_every: int = save_model_every
        self.save_results_every: int = save_results_every
        self.apply_grad_penalty_every: int = apply_grad_penalty_every

        # Clear previous experiment data if requested
        if clear_previous_experiments is True and self.accelerator.is_local_main_process:
            if self.results_dir.exists():
                rmtree(self.results_dir, ignore_errors=True)
        # Make sure logging and results directories exist
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.optim: Optimizer = None

        self.print = self.accelerator.print
        self.log = self.accelerator.log

    def save(self, path):
        if not self.accelerator.is_main_process:
            return

        pkg = dict(
            model=self.accelerator.get_state_dict(self.model),
            optim=self.optim.state_dict(),
        )
        self.accelerator.save(pkg, path)

    def load(self, path: Union[str, PathLike]):
        if not isinstance(path, Path):
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file {path} does not exist.")

        pkg = torch.load(path, map_location="cpu")
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(pkg["model"])

        self.optim.load_state_dict(pkg["optim"])
        return pkg

    def log_validation_images(self, images, step, prompts=None):
        if prompts:
            self.print(f"Logging with prompts: {prompts}")
        if self.validation_image_scale != 1:
            # Feel free to make pr for better solution!
            output_size = (
                int(images[0].size[0] * self.validation_image_scale),
                int(images[0].size[1] * self.validation_image_scale),
            )
            for i in range(len(images)):
                images[i] = images[i].resize(output_size)
        if self.accelerator.is_main_process:
            for tracker in self.accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("validation", np_images, step, dataformats="NHWC")
                elif tracker.name == "wandb":
                    tracker.log(
                        {
                            "validation": [
                                wandb.Image(image, caption="" if not prompts else prompts[i])
                                for i, image in enumerate(images)
                            ]
                        }
                    )

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return (
            False
            if self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1
            else True
        )

    @property
    def is_main_process(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main_process(self):
        return self.accelerator.is_local_main_process

    def train_step(self):
        raise NotImplementedError("You are calling train_step on the base trainer with no models")

    def train(self, log_fn=noop):
        self.model.train()
        while self.steps < self.num_train_steps:
            with self.accelerator.autocast():
                logs = self.train_step()
            log_fn(logs)
        self.print("training complete")
