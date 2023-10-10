import argparse
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Optional, Union

import accelerate
import bz2file as bz2
import datasets
import diffusers
import torch
import transformers
from accelerate.utils import ProjectConfiguration
from datasets import concatenate_datasets, load_dataset
from diffusers.optimization import SchedulerType, get_scheduler
from omegaconf import OmegaConf
from rich import inspect
from torch.optim import Optimizer
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel

import wandb
from muse_maskgit_pytorch.t5 import t5_encode_text_from_encoded
from muse_maskgit_pytorch.utils import (
    get_latest_checkpoints,
)

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ImportError:
    print("TPU support has been disabled, please install torch_xla and train on an XLA device")
    torch_xla = None
    xm = None

from muse_maskgit_pytorch import (
    MaskGit,
    MaskGitTrainer,
    MaskGitTransformer,
    VQGanVAE,
    VQGanVAETaming,
    get_accelerator,
)
from muse_maskgit_pytorch.dataset import (
    ImageTextDataset,
    LocalTextImageDataset,
    URLTextDataset,
    get_dataset_from_dataroot,
    split_dataset_into_dataloaders,
)
from muse_maskgit_pytorch.trainers.base_accelerated_trainer import get_optimizer


def compressed_pickle(title, data):
    with bz2.BZ2File(title, "w") as f:
        pickle.dump(data, f)


def decompress_pickle(file):
    data = bz2.BZ2File(file, "rb")
    data = pickle.load(data)
    return data


# remove some unnecessary errors from transformer shown on the console.
transformers.logging.set_verbosity_error()

# disable bitsandbytes welcome message.
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--only_save_last_checkpoint",
    action="store_true",
    help="Only save last checkpoint.",
)
parser.add_argument(
    "--validation_image_scale",
    default=1.0,
    type=float,
    help="Factor by which to scale the validation images.",
)
parser.add_argument(
    "--no_center_crop",
    action="store_true",
    help="Don't do center crop.",
)
parser.add_argument(
    "--random_crop",
    action="store_true",
    help="Crop the images at random locations instead of cropping from the center.",
)
parser.add_argument(
    "--no_flip",
    action="store_true",
    help="Don't flip image.",
)
parser.add_argument(
    "--dataset_save_path",
    type=str,
    default="dataset",
    help="Path to save the dataset if you are making one from a directory",
)
parser.add_argument(
    "--clear_previous_experiments",
    action="store_true",
    help="Whether to clear previous experiments.",
)
parser.add_argument(
    "--num_tokens",
    type=int,
    default=256,
    help="Number of tokens. Must be same as codebook size above",
)
parser.add_argument(
    "--seq_len",
    type=int,
    default=1024,
    help="The sequence length. Must be equivalent to fmap_size ** 2 in vae",
)
parser.add_argument("--depth", type=int, default=2, help="The depth of model")
parser.add_argument("--dim_head", type=int, default=64, help="Attention head dimension")
parser.add_argument("--heads", type=int, default=8, help="Attention heads")
parser.add_argument("--ff_mult", type=int, default=4, help="Feed forward expansion factor")
parser.add_argument("--t5_name", type=str, default="t5-small", help="Name of your t5 model")
parser.add_argument(
    "--use_clip",
    action="store_true",
    default=False,
    help="whether to use MetaClip instead of a T5",
)
parser.add_argument("--cond_image_size", type=int, default=None, help="Conditional image size.")
parser.add_argument(
    "--validation_prompt",
    type=str,
    default="A photo of a dog",
    help="Validation prompt(s), pipe | separated",
)
parser.add_argument(
    "--timesteps",
    type=int,
    default=18,
    help="Number of steps to use when generating the validation image. Defautl: 18",
)
parser.add_argument("--max_grad_norm", type=float, default=None, help="Max gradient norm.")
parser.add_argument("--seed", type=int, default=42, help="Training seed.")
parser.add_argument(
    "--valid_frac", type=float, default=0.05, help="Fraction of dataset to use for validation."
)
parser.add_argument("--use_ema", action="store_true", help="Whether to use ema.")
parser.add_argument("--ema_beta", type=float, default=0.995, help="Ema beta.")
parser.add_argument("--ema_update_after_step", type=int, default=1, help="Ema update after step.")
parser.add_argument(
    "--ema_update_every",
    type=int,
    default=1,
    help="Ema update every this number of steps.",
)
parser.add_argument(
    "--apply_grad_penalty_every",
    type=int,
    default=4,
    help="Apply gradient penalty every this number of steps.",
)
parser.add_argument(
    "--image_column",
    type=str,
    default="image",
    help="The column of the dataset containing an image.",
)
parser.add_argument(
    "--caption_column",
    type=str,
    default="caption",
    help="The column of the dataset containing a caption or a list of captions.",
)
parser.add_argument(
    "--log_with",
    type=str,
    default="wandb",
    help=(
        'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
        ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
    ),
)
parser.add_argument(
    "--project_name",
    type=str,
    default="muse_maskgit",
    help=("Name to use for the project to identify it when saved to a tracker such as wandb or tensorboard."),
)
parser.add_argument(
    "--run_name",
    type=str,
    default=None,
    help=(
        "Name to use for the run to identify it when saved to a tracker such"
        " as wandb or tensorboard. If not specified a random one will be generated."
    ),
)
parser.add_argument(
    "--wandb_user",
    type=str,
    default=None,
    help=(
        "Specify the name for the user or the organization in which the project will be saved when using wand."
    ),
)
parser.add_argument(
    "--mixed_precision",
    type=str,
    default="no",
    choices=["no", "fp16", "bf16"],
    help="Precision to train on.",
)
parser.add_argument(
    "--use_8bit_adam",
    action="store_true",
    help="Whether to use the 8bit adam optimiser",
)
parser.add_argument(
    "--results_dir",
    type=str,
    default="results",
    help="Path to save the training samples and checkpoints",
)
parser.add_argument(
    "--logging_dir",
    type=str,
    default=None,
    help="Path to log the losses and LR",
)

# vae_trainer args
parser.add_argument(
    "--vae_path",
    type=str,
    default=None,
    help="Path to the vae model. eg. 'results/vae.steps.pt'",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default=None,
    help="ID of HuggingFace dataset to use (cannot be used with --train_data_dir, use multiple by splitting with '|', "
    "they must have the same image column and text column)",
)
parser.add_argument(
    "--hf_split_name",
    type=str,
    default="train",
    help="Subset or split to use from the dataset when using a dataset form HuggingFace.",
)
parser.add_argument(
    "--streaming",
    action="store_true",
    help="If using a HuggingFace dataset, whether to stream it or not.",
)
parser.add_argument(
    "--train_data_dir",
    type=str,
    default=None,
    help="Local dataset folder to use (cannot be used with --dataset_name)",
)
parser.add_argument(
    "--num_train_steps",
    type=int,
    default=-1,
    help="Total number of steps to train for. eg. 50000. | Use only if you want to stop training early",
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=5,
    help="Total number of epochs to train for. eg. 5.",
)
parser.add_argument(
    "--dim",
    type=int,
    default=128,
    help="Model dimension.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=512,
    help="Batch Size.",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-4,
    help="Learning Rate.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Gradient Accumulation Steps",
)
parser.add_argument(
    "--save_results_every",
    type=int,
    default=100,
    help="Save results every N steps.",
)
parser.add_argument(
    "--save_model_every",
    type=int,
    default=500,
    help="Save the model every N steps.",
)
parser.add_argument(
    "--checkpoint_limit",
    type=int,
    default=None,
    help="Keep only X number of checkpoints and delete the older ones.",
)
parser.add_argument(
    "--vq_codebook_size",
    type=int,
    default=256,
    help="Image Size.",
)
parser.add_argument("--vq_codebook_dim", type=int, default=256, help="VQ Codebook dimensions.")
parser.add_argument(
    "--channels", type=int, default=3, help="Number of channels for the VAE. Use 3 for RGB or 4 for RGBA."
)
parser.add_argument("--layers", type=int, default=4, help="Number of layers for the VAE.")
parser.add_argument("--discr_layers", type=int, default=4, help="Number of layers for the VAE discriminator.")
parser.add_argument(
    "--cond_drop_prob",
    type=float,
    default=0.5,
    help="Conditional dropout, for classifier free guidance.",
)
parser.add_argument(
    "--image_size",
    type=int,
    default=256,
    help="Image size. You may want to start with small images, and then curriculum learn to larger ones, but because the vae is all convolution, it should generalize to 512 (as in paper) without training on it",
)
parser.add_argument(
    "--lr_scheduler",
    type=str,
    default="constant",
    help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
)
parser.add_argument(
    "--scheduler_power",
    type=float,
    default=1.0,
    help="Controls the power of the polynomial decay schedule used by the CosineScheduleWithWarmup scheduler. "
    "It determines the rate at which the learning rate decreases during the schedule.",
)
parser.add_argument(
    "--lr_warmup_steps",
    type=int,
    default=0,
    help="Number of steps for the warmup in the lr scheduler.",
)
parser.add_argument(
    "--num_cycles",
    type=int,
    default=1,
    help="Number of cycles for the lr scheduler.",
)
parser.add_argument(
    "--resume_path",
    type=str,
    default=None,
    help="Path to the last saved checkpoint. 'results/maskgit.steps.pt'",
)
parser.add_argument(
    "--taming_model_path",
    type=str,
    default=None,
    help="path to your trained VQGAN weights. This should be a .ckpt file. (only valid when taming option is enabled)",
)
parser.add_argument(
    "--taming_config_path",
    type=str,
    default=None,
    help="path to your trained VQGAN config. This should be a .yaml file. (only valid when taming option is enabled)",
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="Adam",
    help="Optimizer to use. Choose between: ['Adam', 'AdamW','Lion', 'Adafactor', "
    "'AdaBound', 'AdaMod', 'AccSGD', 'AdamP', 'AggMo', 'DiffGrad', 'Lamb', "
    "'NovoGrad', 'PID', 'QHAdam', 'QHM', 'RAdam', 'SGDP', 'SGDW', 'Shampoo', "
    "'SWATS', 'Yogi']. Default: Adam",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.0,
    help="Optimizer weight_decay to use. Default: 0.0",
)
parser.add_argument(
    "--cache_path",
    type=str,
    default=None,
    help="The path to cache huggingface models",
)
parser.add_argument(
    "--no_cache",
    action="store_true",
    help="Do not save the dataset pyarrow cache/files to disk to save disk space and reduce the time it takes to launch the training.",
)
parser.add_argument(
    "--link",
    action="store_true",
    help="whether to load a dataset with links instead of image (image column becomes URL column)",
)
parser.add_argument(
    "--latest_checkpoint",
    action="store_true",
    help="Automatically find and use the latest checkpoint in the folder.",
)
parser.add_argument(
    "--do_not_save_config",
    action="store_true",
    default=False,
    help="Generate example YAML configuration file",
)
parser.add_argument(
    "--use_l2_recon_loss",
    action="store_true",
    help="Use F.mse_loss instead of F.l1_loss.",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="debug logging on",
)
parser.add_argument(
    "--config_path",
    type=str,
    default=None,
    help="debug logging on",
)
parser.add_argument(
    "--attention_type",
    type=str,
    default="flash",
    help="what type of attention to use [ein, flash, xformers] | Default: flash",
)
parser.add_argument(
    "--precompute",
    action="store_true",
    default=False,
    help="whether to precompute text embeds (only use if we wan to compute, not load)",
)
parser.add_argument(
    "--precompute_path",
    type=str,
    default="",
    help="The path to save or load embeds",
)


@dataclass
class Arguments:
    total_params: Optional[int] = None
    image_size: int = 256
    num_tokens: int = 256
    num_train_steps: int = -1
    num_epochs: int = 5
    dim: int = 128
    channels: int = 3
    batch_size: int = 1
    lr: float = 1e-4
    gradient_accumulation_steps: int = 1
    save_results_every: int = 100
    save_model_every: int = 500
    vq_codebook_size: int = 256
    vq_codebook_dim: int = 256
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    seq_len: int = 1024
    depth: int = 2
    dim_head: int = 64
    heads: int = 8
    ff_mult: int = 4
    t5_name: str = "t5-small"
    use_clip: bool = False
    mixed_precision: str = "no"
    cond_image_size: Optional[int] = None
    validation_prompt: str = "A photo of a dog"
    timesteps: int = 18
    optimizer: str = "Lion"
    only_save_last_checkpoint: bool = False
    validation_image_scale: float = 1.0
    no_center_crop: bool = False
    no_flip: bool = False
    dataset_save_path: Optional[str] = None
    clear_previous_experiments: bool = False
    max_grad_norm: Optional[float] = None
    seed: int = 42
    valid_frac: float = 0.05
    use_ema: bool = False
    ema_beta: float = 0.995
    ema_update_after_step: int = 1
    ema_update_every: int = 1
    apply_grad_penalty_every: int = 4
    image_column: str = "image"
    caption_column: str = "caption"
    log_with: str = "wandb"
    use_8bit_adam: bool = False
    results_dir: str = "results"
    logging_dir: Optional[str] = None
    vae_path: Optional[str] = None
    dataset_name: Optional[str] = None
    hf_split_name: Optional[str] = None
    streaming: bool = False
    train_data_dir: Optional[str] = None
    checkpoint_limit: Union[int, str] = None
    cond_drop_prob: float = 0.5
    scheduler_power: float = 1.0
    num_cycles: int = 1
    resume_path: Optional[str] = None
    taming_model_path: Optional[str] = None
    taming_config_path: Optional[str] = None
    weight_decay: float = 0.0
    cache_path: Optional[str] = None
    no_cache: bool = False
    link: bool = False
    latest_checkpoint: bool = False
    do_not_save_config: bool = False
    use_l2_recon_loss: bool = False
    debug: bool = False
    config_path: Optional[str] = None
    attention_type: str = "flash"
    precompute: bool = False
    precompute_path: str = ""


def main():
    args = parser.parse_args(namespace=Arguments())

    if accelerate.utils.is_rich_available():
        from rich import print
        from rich.traceback import install as traceback_install

        traceback_install(show_locals=args.debug, width=120, word_wrap=True)

    if args.config_path:
        print("Using config file and ignoring CLI args")

        try:
            conf = OmegaConf.load(args.config_path)
            conf_keys = conf.keys()
            args_to_convert = vars(args)

            for key in conf_keys:
                try:
                    args_to_convert[key] = conf[key]
                except KeyError:
                    print(f"Error parsing config - {key}: {conf[key]} | Using default or parsed")

        except FileNotFoundError:
            print("Could not find config, using default and parsed values...")

    # Set up debug logging as early as possible
    if args.debug is True:
        logging.basicConfig(level=logging.DEBUG)
        transformers.logging.set_verbosity_debug()
        datasets.logging.set_verbosity_debug()
        diffusers.logging.set_verbosity_debug()
    else:
        logging.basicConfig(level=logging.INFO)

    project_config = ProjectConfiguration(
        project_dir=args.logging_dir if args.logging_dir else os.path.join(args.results_dir, "logs"),
        total_limit=args.checkpoint_limit,
        automatic_checkpoint_naming=True,
    )

    accelerator: accelerate.Accelerator = get_accelerator(
        log_with=args.log_with,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=project_config,
        even_batches=True,
    )

    # Get these errors out of the way early
    if args.vae_path and args.taming_model_path:
        raise ValueError("Can't pass both vae_path and taming_model_path at the same time!")
    if args.train_data_dir and args.dataset_name:
        raise ValueError("Can't pass both train_data_dir and dataset_name at the same time!")

    if accelerator.is_main_process:
        accelerator.print(f"Preparing MaskGit for training on {accelerator.device.type}")
        if args.debug:
            inspect(args, docs=False)

        accelerate.utils.set_seed(args.seed)

    # Load the dataset (main process first to download, rest will load from cache)
    with accelerator.main_process_first():
        if args.train_data_dir is not None:
            if args.no_cache:
                pass
            else:
                dataset = get_dataset_from_dataroot(
                    args.train_data_dir,
                    image_column=args.image_column,
                    caption_column=args.caption_column,
                    save_path=args.dataset_save_path,
                )
        elif args.dataset_name is not None:
            if "|" in args.dataset_name:
                loaded_datasets = []
                for name in args.dataset_name.split("|"):
                    accelerator.print(f"Loading {name}")
                    data_to_add = load_dataset(
                        name,
                        streaming=args.streaming,
                        cache_dir=args.cache_path,
                        save_infos=True,
                        split="train",
                    )

                    data_to_add.remove_columns(
                        [
                            col
                            for col in data_to_add.column_names
                            if col != args.caption_column or col != args.image_column
                        ]
                    )

                    loaded_datasets.append(data_to_add)

                try:
                    dataset = concatenate_datasets(loaded_datasets)
                except ValueError:
                    raise UserWarning("Failed concatenating dataset... Make sure they use the same columns!")

            else:
                dataset = load_dataset(
                    args.dataset_name,
                    streaming=args.streaming,
                    cache_dir=args.cache_path,
                    save_infos=True,
                    split="train",
                )

                if args.streaming:
                    if args.cache_path:
                        dataset = load_dataset(args.dataset_name, cache_dir=args.cache_path)[
                            args.hf_split_name
                        ]
                    else:
                        dataset = load_dataset(args.dataset_name)[args.hf_split_name]
        else:
            raise ValueError("You must pass either train_data_dir or dataset_name (but not both)")

    # Load the VAE
    with accelerator.main_process_first():
        if args.vae_path:
            accelerator.print("Loading Muse VQGanVAE")

            if args.latest_checkpoint:
                args.vae_path, ema_model_path = get_latest_checkpoints(args.vae_path, use_ema=args.use_ema)
                if ema_model_path:
                    ema_vae = VQGanVAE(
                        dim=args.dim,
                        vq_codebook_dim=args.vq_codebook_dim,
                        vq_codebook_size=args.vq_codebook_size,
                        l2_recon_loss=args.use_l2_recon_loss,
                        channels=args.channels,
                        layers=args.layers,
                        discr_layers=args.discr_layers,
                        accelerator=accelerator,
                    )
                    print(f"Resuming EMA VAE from latest checkpoint: {ema_model_path}")

                    ema_vae.load(ema_model_path, map="cpu")
                else:
                    ema_vae = None

                print(f"Resuming VAE from latest checkpoint: {args.vae_path}")
            else:
                accelerator.print("Resuming VAE from: ", args.vae_path)
                ema_vae = None

            # use config next to checkpoint if there is one and merge the cli arguments to it
            # the cli arguments will take priority so we can use it to override any value we want.
            # if os.path.exists(f"{args.vae_path}.yaml"):
            # print("Config file found, reusing config from it. Use cli arguments to override any desired value.")
            # conf = OmegaConf.load(f"{args.vae_path}.yaml")
            # cli_conf = OmegaConf.from_cli()
            ## merge the config file and the cli arguments.
            # conf = OmegaConf.merge(conf, cli_conf)

            vae = VQGanVAE(
                dim=args.dim,
                vq_codebook_dim=args.vq_codebook_dim,
                vq_codebook_size=args.vq_codebook_size,
                l2_recon_loss=args.use_l2_recon_loss,
                channels=args.channels,
                layers=args.layers,
                discr_layers=args.discr_layers,
            ).to(accelerator.device)

            vae.load(args.vae_path, map="cpu")

        elif args.taming_model_path is not None and args.taming_config_path is not None:
            accelerator.print(f"Using Taming VQGanVAE, loading from {args.taming_model_path}")
            vae = VQGanVAETaming(
                vqgan_model_path=args.taming_model_path,
                vqgan_config_path=args.taming_config_path,
                accelerator=accelerator,
            )
            args.num_tokens = vae.codebook_size
            args.seq_len = vae.get_encoded_fmap_size(args.image_size) ** 2
        else:
            raise ValueError(
                "You must pass either vae_path or taming_model_path + taming_config_path (but not both)"
            )

    # freeze VAE before parsing to transformer
    vae.requires_grad_(False)

    # then you plug the vae and transformer into your MaskGit like so:

    # (1) create your transformer / attention network
    if args.attention_type == "flash":
        xformers = False
        flash = True
    elif args.attention_type == "xformers":
        xformers = True
        flash = True
    elif args.attention_type == "ein":
        xformers = False
        flash = False
    else:
        raise NotImplementedError(f'Attention of type "{args.attention_type}" does not exist')

    transformer: MaskGitTransformer = MaskGitTransformer(
        # num_tokens must be same as codebook size above
        num_tokens=args.num_tokens if args.num_tokens else args.vq_codebook_size,
        # seq_len must be equivalent to fmap_size ** 2 in vae
        seq_len=args.seq_len,
        dim=args.dim,
        depth=args.depth,
        dim_head=args.dim_head,
        heads=args.heads,
        # feedforward expansion factor
        ff_mult=args.ff_mult,
        # name of your T5 model configuration
        t5_name=args.t5_name,
        cache_path=args.cache_path,
        flash=flash,
        xformers=xformers,
        use_clip=args.use_clip,
    )

    if args.use_clip:
        model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=args.cache_path).to(
            accelerator.device
        )
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", cache_dir=args.cache_path)

        clip = (model, tokenizer)
    else:
        model = None
        tokenizer = None
        clip = None

    # (2) pass your trained VAE and the base transformer to MaskGit
    maskgit = MaskGit(
        vae=vae,  # vqgan vae
        transformer=transformer,  # transformer
        clip=model,
        clip_tokenizer=tokenizer,
        accelerator=accelerator,  # accelerator
        image_size=args.image_size,  # image size
        cond_drop_prob=args.cond_drop_prob,  # conditional dropout, for classifier free guidance
        cond_image_size=args.cond_image_size,
    )

    # load the maskgit transformer from disk if we have previously trained one
    with accelerator.main_process_first():
        if args.resume_path is not None and len(args.resume_path) > 1:
            load = True

            accelerator.print("Loading Muse MaskGit...")

            if args.latest_checkpoint:
                try:
                    args.resume_path, ema_model_path = get_latest_checkpoints(
                        args.resume_path,
                        use_ema=args.use_ema,
                        model_type="maskgit",
                        cond_image_size=args.cond_image_size,
                    )
                    print(f"Resuming MaskGit from latest checkpoint: {args.resume_path}")
                    # if args.use_ema:
                    #    print(f"Resuming EMA MaskGit from latest checkpoint: {ema_model_path}")

                except ValueError:
                    load = False

            else:
                accelerator.print("Resuming MaskGit from: ", args.resume_path)

            if load:
                maskgit.load(args.resume_path)

                resume_from_parts = args.resume_path.split(".")
                for i in range(len(resume_from_parts) - 1, -1, -1):
                    if resume_from_parts[i].isdigit():
                        current_step = int(resume_from_parts[i])
                        accelerator.print(f"Found step {current_step} for the MaskGit model.")
                        break
                if current_step == 0:
                    accelerator.print("No step found for the MaskGit model.")
            else:
                current_step = 0
        else:
            accelerator.print("Initialized new empty MaskGit model.")
            current_step = 0

    # Use the parameters() method to get an iterator over all the learnable parameters of the model
    total_params = sum(p.numel() for p in maskgit.parameters())
    args.total_params = total_params

    accelerator.print(f"Total number of parameters: {format(total_params, ',d')}")

    if args.precompute_path and not args.precompute:
        embeds = decompress_pickle(args.precompute_path)
    else:
        embeds = []

    if args.use_clip:
        transformer.tokenizer = None

    # Create the dataset objects
    with accelerator.main_process_first():
        if args.no_cache and args.train_data_dir:
            dataset = LocalTextImageDataset(
                args.train_data_dir,
                args.image_size,
                tokenizer=transformer.tokenizer,
                center_crop=False if args.no_center_crop else True,
                flip=False if args.no_flip else True,
                using_taming=False if not args.taming_model_path else True,
                random_crop=args.random_crop if args.random_crop else False,
                alpha_channel=False if args.channels == 3 else True,
                embeds=embeds,
            )
        elif args.link:
            if not args.dataset_name:
                raise AssertionError("You can only use links in huggingface datasets")

            dataset = URLTextDataset(
                dataset,
                args.image_size,
                transformer.tokenizer,
                image_column=args.image_column,
                caption_column=args.caption_column,
                center_crop=False if args.no_center_crop else True,
                flip=False if args.no_flip else True,
                using_taming=False if not args.taming_model_path else True,
                embeds=embeds,
            )
        else:
            dataset = ImageTextDataset(
                dataset,
                args.image_size,
                transformer.tokenizer,
                image_column=args.image_column,
                caption_column=args.caption_column,
                center_crop=False if args.no_center_crop else True,
                flip=False if args.no_flip else True,
                stream=args.streaming,
                using_taming=False if not args.taming_model_path else True,
                embeds=embeds,
            )

    # Create the dataloaders
    dataloader, validation_dataloader = split_dataset_into_dataloaders(
        dataset,
        args.valid_frac if not args.streaming else 0,
        args.seed,
        args.batch_size,
    )

    # Create the optimizer and scheduler, wrap optimizer in scheduler
    optimizer: Optimizer = get_optimizer(
        args.use_8bit_adam, args.optimizer, set(transformer.parameters()), args.lr, args.weight_decay
    )

    if args.num_train_steps > 0:
        num_lr_steps = args.num_train_steps * args.gradient_accumulation_steps
    else:
        num_lr_steps = args.num_epochs * len(dataloader)

    scheduler: SchedulerType = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=num_lr_steps,
        num_cycles=args.num_cycles,
        power=args.scheduler_power,
    )

    # Prepare the model, optimizer, and dataloaders for distributed training
    maskgit, optimizer, dataloader, validation_dataloader, scheduler = accelerator.prepare(
        maskgit, optimizer, dataloader, validation_dataloader, scheduler
    )

    # Wait for everyone to catch up, then print some info and initialize the trackers
    accelerator.wait_for_everyone()
    accelerator.print(f"[{accelerator.process_index}] Ready to create trainer!")
    if accelerator.distributed_type == accelerate.DistributedType.TPU:
        proc_idx = accelerator.process_index + 1
        n_procs = accelerator.num_processes
        local_proc_idx = accelerator.local_process_index + 1
        xm_ord = xm.get_ordinal() + 1
        xm_world = xm.xrt_world_size()
        xm_local_ord = xm.get_local_ordinal() + 1
        xm_master_ord = xm.is_master_ordinal()
        is_main = accelerator.is_main_process
        is_local_main = accelerator.is_local_main_process

        with accelerator.local_main_process_first():
            accelerator.print(
                f"[P{proc_idx:03d}]: PID {proc_idx}/{n_procs}, local #{local_proc_idx}, ",
                f"XLA ord={xm_ord}/{xm_world}, local={xm_local_ord}, master={xm_master_ord} ",
                f"Accelerate: main={is_main}, local main={is_local_main} ",
            )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            args.project_name,
            config=vars(args),
            init_kwargs={
                "wandb": {
                    "entity": f"{args.wandb_user or wandb.api.default_entity}",
                    "name": args.run_name,
                },
            },
        )

    embeds = []
    if args.precompute:
        accelerator.print("Beginning pre-computation of embeddings using T5...")
        maskgit.transformer.t5.requires_grad_(False)
        for imgs, input_ids, attn_mask, _ in tqdm(iter(dataloader)):
            with torch.no_grad():
                embedding = t5_encode_text_from_encoded(input_ids, attn_mask, maskgit.transformer.t5, "cpu")
                embeds.append(embedding)

        if args.precompute_path:
            compressed_pickle(args.precompute_path, embeds)

        with accelerator.main_process_first():
            if args.no_cache and args.train_data_dir:
                dataset = LocalTextImageDataset(
                    args.train_data_dir,
                    args.image_size,
                    tokenizer=transformer.tokenizer,
                    center_crop=False if args.no_center_crop else True,
                    flip=False if args.no_flip else True,
                    using_taming=False if not args.taming_model_path else True,
                    random_crop=args.random_crop if args.random_crop else False,
                    alpha_channel=False if args.channels == 3 else True,
                    embeds=embeds,
                )
            elif args.link:
                if not args.dataset_name:
                    raise AssertionError("You can only use links in huggingface datasets")

                dataset = URLTextDataset(
                    dataset,
                    args.image_size,
                    transformer.tokenizer,
                    image_column=args.image_column,
                    caption_column=args.caption_column,
                    center_crop=False if args.no_center_crop else True,
                    flip=False if args.no_flip else True,
                    using_taming=False if not args.taming_model_path else True,
                    embeds=embeds,
                )
            else:
                dataset = ImageTextDataset(
                    dataset,
                    args.image_size,
                    transformer.tokenizer,
                    image_column=args.image_column,
                    caption_column=args.caption_column,
                    center_crop=False if args.no_center_crop else True,
                    flip=False if args.no_flip else True,
                    stream=args.streaming,
                    using_taming=False if not args.taming_model_path else True,
                    embeds=embeds,
                )

        accelerator.print("Embeddings pre-computed!")

        # Create the dataloaders
        dataloader, validation_dataloader = split_dataset_into_dataloaders(
            dataset,
            args.valid_frac if not args.streaming else 0,
            args.seed,
            args.batch_size,
        )

    # Create the trainer
    accelerator.wait_for_everyone()
    trainer = MaskGitTrainer(
        maskgit=maskgit,
        dataloader=dataloader,
        valid_dataloader=validation_dataloader,
        accelerator=accelerator,
        optimizer=optimizer,
        scheduler=scheduler,
        current_step=current_step + 1 if current_step != 0 else current_step,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        max_grad_norm=args.max_grad_norm,
        save_results_every=args.save_results_every,
        save_model_every=args.save_model_every,
        results_dir=args.results_dir,
        logging_dir=args.logging_dir if args.logging_dir else os.path.join(args.results_dir, "logs"),
        use_ema=args.use_ema,
        ema_vae=ema_vae,
        ema_update_after_step=args.ema_update_after_step,
        ema_update_every=args.ema_update_every,
        apply_grad_penalty_every=args.apply_grad_penalty_every,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        validation_prompts=args.validation_prompt.split("|"),
        timesteps=args.timesteps,
        clear_previous_experiments=args.clear_previous_experiments,
        validation_image_scale=args.validation_image_scale,
        only_save_last_checkpoint=args.only_save_last_checkpoint,
        num_epochs=args.num_epochs,
        args=args,
        clip=clip,
    )

    # Prepare the trainer for distributed training
    accelerator.print("MaskGit Trainer initialized, preparing for training...")
    trainer = accelerator.prepare(trainer)

    # Train the model!
    accelerator.print("Starting training!")
    trainer.train()

    # Clean up and wait for other processes to finish (loggers etc.)
    if accelerator.is_main_process:
        accelerator.print("Training complete!")
        accelerator.end_training()


if __name__ == "__main__":
    main()
