import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Optional, Union

import wandb
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from omegaconf import OmegaConf

from muse_maskgit_pytorch import (
    VQGanVAE,
    VQGanVAETaming,
    VQGanVAETrainer,
    get_accelerator,
)
from muse_maskgit_pytorch.dataset import (
    ImageDataset,
    get_dataset_from_dataroot,
    split_dataset_into_dataloaders,
)

# disable bitsandbytes welcome message.
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--webdataset", type=str, default=None, help="Path to webdataset if using one.")
parser.add_argument(
    "--only_save_last_checkpoint",
    action="store_true",
    help="Only save last checkpoint.",
)
parser.add_argument(
    "--validation_image_scale",
    default=1,
    type=float,
    help="Factor by which to scale the validation images.",
)
parser.add_argument(
    "--no_center_crop",
    action="store_true",
    help="Don't do center crop.",
)
parser.add_argument(
    "--no_flip",
    action="store_true",
    help="Don't flip image.",
)
parser.add_argument(
    "--random_crop",
    action="store_true",
    help="Crop the images at random locations instead of cropping from the center.",
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
parser.add_argument("--max_grad_norm", type=float, default=None, help="Max gradient norm.")
parser.add_argument(
    "--discr_max_grad_norm",
    type=float,
    default=None,
    help="Max gradient norm for discriminator.",
)
parser.add_argument("--seed", type=int, default=42, help="Seed.")
parser.add_argument("--valid_frac", type=float, default=0.05, help="validation fraction.")
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
    default="muse_vae",
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
    choices=["no", "fp8", "fp16", "bf16"],
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
    "--dataset_name",
    type=str,
    default=None,
    help="Name of the huggingface dataset used.",
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
    help="Whether to stream the huggingface dataset",
)
parser.add_argument(
    "--train_data_dir",
    type=str,
    default=None,
    help="Dataset folder where your input images for training are.",
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
parser.add_argument("--dim", type=int, default=128, help="Model dimension.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch Size.")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning Rate.")
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Gradient Accumulation.",
)
parser.add_argument(
    "--save_results_every",
    type=int,
    default=100,
    help="Save results every this number of steps.",
)
parser.add_argument(
    "--save_model_every",
    type=int,
    default=500,
    help="Save the model every this number of steps.",
)
parser.add_argument(
    "--checkpoint_limit",
    type=int,
    default=None,
    help="Keep only X number of checkpoints and delete the older ones.",
)
parser.add_argument("--vq_codebook_size", type=int, default=256, help="Image Size.")
parser.add_argument("--vq_codebook_dim", type=int, default=256, help="VQ Codebook dimensions.")
parser.add_argument(
    "--channels", type=int, default=3, help="Number of channels for the VAE. Use 3 for RGB or 4 for RGBA."
)
parser.add_argument("--layers", type=int, default=4, help="Number of layers for the VAE.")
parser.add_argument("--discr_layers", type=int, default=4, help="Number of layers for the VAE discriminator.")
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
    help="Path to the last saved checkpoint. 'results/vae.steps.pt'",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.0,
    help="Optimizer weight_decay to use. Default: 0.0",
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
    default="Lion",
    help="Optimizer to use. Choose between: ['Adam', 'AdamW','Lion']. Default: Lion",
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
    "--latest_checkpoint",
    action="store_true",
    help="Whether to use the latest checkpoint",
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


@dataclass
class Arguments:
    only_save_last_checkpoint: bool = False
    validation_image_scale: float = 1.0
    no_center_crop: bool = False
    no_flip: bool = False
    random_crop: bool = False
    dataset_save_path: Optional[str] = None
    clear_previous_experiments: bool = False
    max_grad_norm: Optional[float] = None
    discr_max_grad_norm: Optional[float] = None
    num_tokens: int = 256
    seq_len: int = 1024
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
    mixed_precision: str = "no"
    use_8bit_adam: bool = False
    results_dir: str = "results"
    logging_dir: Optional[str] = None
    resume_path: Optional[str] = None
    dataset_name: Optional[str] = None
    streaming: bool = False
    train_data_dir: Optional[str] = None
    num_train_steps: int = -1
    num_epochs: int = 5
    dim: int = 128
    batch_size: int = 512
    lr: float = 1e-5
    gradient_accumulation_steps: int = 1
    save_results_every: int = 100
    save_model_every: int = 500
    checkpoint_limit: Union[int, str] = None
    vq_codebook_size: int = 256
    vq_codebook_dim: int = 256
    cond_drop_prob: float = 0.5
    image_size: int = 256
    lr_scheduler: str = "constant"
    scheduler_power: float = 1.0
    lr_warmup_steps: int = 0
    num_cycles: int = 1
    taming_model_path: Optional[str] = None
    taming_config_path: Optional[str] = None
    optimizer: str = "Lion"
    weight_decay: float = 0.0
    cache_path: Optional[str] = None
    no_cache: bool = False
    latest_checkpoint: bool = False
    do_not_save_config: bool = False
    use_l2_recon_loss: bool = False
    debug: bool = False
    config_path: Optional[str] = None


def preprocess_webdataset(args, image):
    return {args.image_column: image}


def main():
    args = parser.parse_args(namespace=Arguments())

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

    project_config = ProjectConfiguration(
        project_dir=args.logging_dir if args.logging_dir else os.path.join(args.results_dir, "logs"),
        total_limit=args.checkpoint_limit,
        automatic_checkpoint_naming=True,
    )

    accelerator = get_accelerator(
        log_with=args.log_with,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=project_config,
        even_batches=True,
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

    if args.webdataset is not None:
        import webdataset as wds

        dataset = wds.WebDataset(args.webdataset).shuffle(1000).decode("rgb").to_tuple("png")
        dataset = dataset.map(lambda image: preprocess_webdataset(args, image))
    elif args.train_data_dir:
        dataset = get_dataset_from_dataroot(
            args.train_data_dir,
            image_column=args.image_column,
            caption_column=args.caption_column,
            save_path=args.dataset_save_path,
            save=not args.no_cache,
        )
    elif args.dataset_name:
        if args.cache_path:
            dataset = load_dataset(args.dataset_name, streaming=args.streaming, cache_dir=args.cache_path)[
                "train"
            ]
        else:
            dataset = load_dataset(args.dataset_name, streaming=args.streaming, cache_dir=args.cache_path)[
                "train"
            ]
        if args.streaming:
            if dataset.info.dataset_size is None:
                print("Dataset doesn't support streaming, disabling streaming")
                args.streaming = False
                if args.cache_path:
                    dataset = load_dataset(args.dataset_name, cache_dir=args.cache_path)[args.hf_split_name]
                else:
                    dataset = load_dataset(args.dataset_name)[args.hf_split_name]

    if args.resume_path is not None:
        load = True
        accelerator.print(f"Using Muse VQGanVAE, loading from {args.resume_path}")
        vae = VQGanVAE(
            dim=args.dim,
            vq_codebook_dim=args.vq_codebook_dim,
            vq_codebook_size=args.vq_codebook_size,
            l2_recon_loss=args.use_l2_recon_loss,
            channels=args.channels,
            layers=args.layers,
            discr_layers=args.discr_layers,
            accelerator=accelerator,
        )

        if args.latest_checkpoint:
            accelerator.print("Finding latest checkpoint...")
            orig_vae_path = args.resume_path

            if os.path.isfile(args.resume_path) or ".pt" in args.resume_path:
                # If args.vae_path is a file, split it into directory and filename
                args.resume_path, _ = os.path.split(args.resume_path)

            checkpoint_files = glob.glob(os.path.join(args.resume_path, "vae.*.pt"))
            if checkpoint_files:
                latest_checkpoint_file = max(
                    checkpoint_files, key=lambda x: int(re.search(r"vae\.(\d+)\.pt", x).group(1))
                )

                # Check if latest checkpoint is empty or unreadable
                if os.path.getsize(latest_checkpoint_file) == 0 or not os.access(
                    latest_checkpoint_file, os.R_OK
                ):
                    accelerator.print(
                        f"Warning: latest checkpoint {latest_checkpoint_file} is empty or unreadable."
                    )
                    if len(checkpoint_files) > 1:
                        # Use the second last checkpoint as a fallback
                        latest_checkpoint_file = max(
                            checkpoint_files[:-1], key=lambda x: int(re.search(r"vae\.(\d+)\.pt", x).group(1))
                        )
                        accelerator.print("Using second last checkpoint: ", latest_checkpoint_file)
                    else:
                        accelerator.print("No usable checkpoint found.")
                        load = False
                elif latest_checkpoint_file != orig_vae_path:
                    accelerator.print("Resuming VAE from latest checkpoint: ", latest_checkpoint_file)
                else:
                    accelerator.print("Using checkpoint specified in vae_path: ", orig_vae_path)

                args.resume_path = latest_checkpoint_file
            else:
                accelerator.print("No checkpoints found in directory: ", args.resume_path)
                load = False
        else:
            accelerator.print("Resuming VAE from: ", args.resume_path)

        if load:
            vae.load(args.resume_path, map="cpu")

            resume_from_parts = args.resume_path.split(".")
            for i in range(len(resume_from_parts) - 1, -1, -1):
                if resume_from_parts[i].isdigit():
                    current_step = int(resume_from_parts[i])
                    accelerator.print(f"Found step {current_step} for the VAE model.")
                    break
            if current_step == 0:
                accelerator.print("No step found for the VAE model.")
        else:
            accelerator.print("No step found for the VAE model.")
            current_step = 0

    elif args.taming_model_path is not None and args.taming_config_path is not None:
        print(f"Using Taming VQGanVAE, loading from {args.taming_model_path}")
        vae = VQGanVAETaming(
            vqgan_model_path=args.taming_model_path,
            vqgan_config_path=args.taming_config_path,
            accelerator=accelerator,
        )
        args.num_tokens = vae.codebook_size
        args.seq_len = vae.get_encoded_fmap_size(args.image_size) ** 2
    else:
        print("Initialising empty VAE")
        vae = VQGanVAE(
            dim=args.dim,
            vq_codebook_dim=args.vq_codebook_dim,
            vq_codebook_size=args.vq_codebook_size,
            channels=args.channels,
            layers=args.layers,
            discr_layers=args.discr_layers,
            accelerator=accelerator,
        )

        current_step = 0

    # Use the parameters() method to get an iterator over all the learnable parameters of the model
    total_params = sum(p.numel() for p in vae.parameters())

    print(f"Total number of parameters: {format(total_params, ',d')}")

    dataset = ImageDataset(
        dataset,
        args.image_size,
        image_column=args.image_column,
        center_crop=not args.no_center_crop,
        flip=not args.no_flip,
        stream=args.streaming,
        random_crop=args.random_crop,
        alpha_channel=False if args.channels == 3 else True,
    )
    # dataloader

    dataloader, validation_dataloader = split_dataset_into_dataloaders(
        dataset, args.valid_frac, args.seed, args.batch_size
    )
    trainer = VQGanVAETrainer(
        vae,
        dataloader,
        validation_dataloader,
        accelerator,
        current_step=current_step + 1 if current_step != 0 else current_step,
        num_train_steps=args.num_train_steps,
        lr=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        max_grad_norm=args.max_grad_norm,
        discr_max_grad_norm=args.discr_max_grad_norm,
        save_results_every=args.save_results_every,
        save_model_every=args.save_model_every,
        results_dir=args.results_dir,
        logging_dir=args.logging_dir if args.logging_dir else os.path.join(args.results_dir, "logs"),
        use_ema=args.use_ema,
        ema_beta=args.ema_beta,
        ema_update_after_step=args.ema_update_after_step,
        ema_update_every=args.ema_update_every,
        apply_grad_penalty_every=args.apply_grad_penalty_every,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clear_previous_experiments=args.clear_previous_experiments,
        validation_image_scale=args.validation_image_scale,
        only_save_last_checkpoint=args.only_save_last_checkpoint,
        optimizer=args.optimizer,
        use_8bit_adam=args.use_8bit_adam,
        num_cycles=args.num_cycles,
        scheduler_power=args.scheduler_power,
        num_epochs=args.num_epochs,
        args=args,
    )

    trainer.train()


if __name__ == "__main__":
    main()
