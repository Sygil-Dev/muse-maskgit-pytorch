import argparse
import glob
import hashlib
import os
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import accelerate
import PIL
import torch
from accelerate.utils import ProjectConfiguration
from datasets import Dataset, Image, load_dataset
from torchvision.utils import save_image
from tqdm import tqdm

from muse_maskgit_pytorch import (
    VQGanVAE,
    VQGanVAETaming,
    get_accelerator,
)
from muse_maskgit_pytorch.dataset import (
    ImageDataset,
    get_dataset_from_dataroot,
)
from muse_maskgit_pytorch.utils import (
    get_latest_checkpoints,
)

from muse_maskgit_pytorch.vqvae import VQVAE

# Create the parser
parser = argparse.ArgumentParser()
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
    "--random_image",
    action="store_true",
    help="Get a random image from the dataset to use for the reconstruction.",
)
parser.add_argument(
    "--dataset_save_path",
    type=str,
    default="dataset",
    help="Path to save the dataset if you are making one from a directory",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Seed for reproducibility. If set to -1 a random seed will be generated.",
)
parser.add_argument("--valid_frac", type=float, default=0.05, help="validation fraction.")
parser.add_argument(
    "--image_column",
    type=str,
    default="image",
    help="The column of the dataset containing an image.",
)
parser.add_argument(
    "--mixed_precision",
    type=str,
    default="no",
    choices=["no", "fp16", "bf16"],
    help="Precision to train on.",
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
    help="Name of the huggingface dataset used.",
)
parser.add_argument(
    "--train_data_dir",
    type=str,
    default=None,
    help="Dataset folder where your input images for training are.",
)
parser.add_argument("--dim", type=int, default=128, help="Model dimension.")
parser.add_argument("--batch_size", type=int, default=512, help="Batch Size.")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate.")
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
    "--chunk_size",
    type=int,
    default=256,
    help="This is used to split big images into smaller chunks so we can still reconstruct them no matter the size.",
)
parser.add_argument(
    "--min_chunk_size",
    type=int,
    default=8,
    help="We use a minimum chunk size to ensure that the image is always reconstructed correctly.",
)
parser.add_argument(
    "--overlap_size",
    type=int,
    default=256,
    help="The overlap size used with --chunk_size to overlap the chunks and make sure the whole image is reconstructe as well as make sure we remove artifacts caused by doing the reconstrucion in chunks.",
)
parser.add_argument(
    "--min_overlap_size",
    type=int,
    default=1,
    help="We use a minimum overlap size to ensure that the image is always reconstructed correctly.",
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
    "--input_image",
    type=str,
    default=None,
    help="Path to an image to use as input for reconstruction instead of using one from the dataset.",
)
parser.add_argument(
    "--input_folder",
    type=str,
    default=None,
    help="Path to a folder with images to use as input for creating a dataset for reconstructing all the imgaes in it instead of just one image.",
)
parser.add_argument(
    "--exclude_folders",
    type=str,
    default=None,
    help="List of folders we want to exclude when doing reconstructions from an input folder.",
)
parser.add_argument(
    "--gpu",
    type=int,
    default=0,
    help="GPU to use in case we want to use a specific GPU for inference.",
)
parser.add_argument(
    "--cpu",
    action="store_true",
    help="Use the CPU instead of the GPU, this will be really slow but can be useful for testing or if you dont have a good GPU.",
)
parser.add_argument(
    "--max_retries",
    type=int,
    default=30,
    help="Max number of times to retry in case the reconstruction fails due to OOM or any other error.",
)
parser.add_argument(
    "--latest_checkpoint",
    action="store_true",
    help="Use the latest checkpoint using the vae_path folder instead of using just a specific vae_path.",
)
parser.add_argument(
    "--use_paintmind",
    action="store_true",
    help="Use PaintMind VAE..",
)
parser.add_argument(
    "--save_originals",
    action="store_true",
    help="Save the original input.png and output.png images to a subfolder instead of deleting them after the grid is made.",
)
parser.add_argument("--use_ema", action="store_true", help="Whether to use ema.")
parser.add_argument("--ema_beta", type=float, default=0.995, help="Ema beta.")


@dataclass
class Arguments:
    only_save_last_checkpoint: bool = False
    validation_image_scale: float = 1.0
    no_center_crop: bool = False
    no_flip: bool = False
    random_crop: bool = False
    random_image: bool = False
    dataset_save_path: Optional[str] = None
    clear_previous_experiments: bool = False
    max_grad_norm: Optional[float] = None
    discr_max_grad_norm: Optional[float] = None
    num_tokens: int = 256
    seq_len: int = 1024
    channels: int = 3
    layers: int = 4
    discr_layers: int = 4
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
    generate_config: bool = False


def seed_to_int(s):
    if type(s) is int:
        return s
    if s is None or s == "":
        return random.randint(0, 2**32 - 1)

    if "," in s:
        s = s.split(",")

    if type(s) is list:
        seed_list = []
        for seed in s:
            if seed is None or seed == "":
                seed_list.append(random.randint(0, 2**32 - 1))
            else:
                seed_list = s

        return seed_list

    n = abs(int(s) if s.isdigit() else random.Random(s).randint(0, 2**32 - 1))
    while n >= 2**32:
        n = n >> 32
    return n


def main():
    args = parser.parse_args(namespace=Arguments())

    project_config = ProjectConfiguration(
        project_dir=args.logging_dir if args.logging_dir else os.path.join(args.results_dir, "logs"),
        automatic_checkpoint_naming=True,
    )

    accelerator: accelerate.Accelerator = get_accelerator(
        log_with=args.log_with,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=project_config,
        even_batches=True,
    )

    # set pytorch seed for reproducibility
    torch.manual_seed(seed_to_int(args.seed))

    if args.train_data_dir and not args.input_image and not args.input_folder:
        dataset = get_dataset_from_dataroot(
            args.train_data_dir,
            image_column=args.image_column,
            save_path=args.dataset_save_path,
        )
    elif args.dataset_name and not args.input_image and not args.input_folder:
        dataset = load_dataset(args.dataset_name)["train"]

    elif args.input_image and not args.input_folder:
        # Create dataset from single input image
        dataset = Dataset.from_dict({"image": [args.input_image]}).cast_column("image", Image())

    if args.input_folder:
        # Create dataset from input folder
        extensions = ["jpg", "jpeg", "png", "webp"]
        exclude_folders = args.exclude_folders.split(",") if args.exclude_folders else []

        filepaths = []
        for root, dirs, files in os.walk(args.input_folder, followlinks=True):
            # Resolve symbolic link to actual path and exclude based on actual path
            resolved_root = os.path.realpath(root)
            for exclude_folder in exclude_folders:
                if exclude_folder in resolved_root:
                    dirs[:] = []
                    break
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    filepaths.append(os.path.join(root, file))

        if not filepaths:
            print(f"No images with extensions {extensions} found in {args.input_folder}.")
            exit(1)

        dataset = Dataset.from_dict({"image": filepaths}).cast_column("image", Image())

    if args.vae_path and args.taming_model_path:
        raise Exception("You can't pass vae_path and taming args at the same time.")

    if args.vae_path and not args.use_paintmind:
        accelerator.print("Loading Muse VQGanVAE")
        vae = VQGanVAE(
            dim=args.dim,
            vq_codebook_size=args.vq_codebook_size,
            vq_codebook_dim=args.vq_codebook_dim,
            channels=args.channels,
            layers=args.layers,
            discr_layers=args.discr_layers,
        ).to("cpu" if args.cpu else accelerator.device if args.gpu == 0 else f"cuda:{args.gpu}")

        if args.latest_checkpoint:
            args.vae_path, ema_model_path = get_latest_checkpoints(args.vae_path, use_ema=args.use_ema)
            print(f"Resuming VAE from latest checkpoint: {args.vae_path if  not args.use_ema else ema_model_path}")
            #if args.use_ema:
            #    print(f"Resuming EMA VAE from latest checkpoint: {ema_model_path}")
        else:
            accelerator.print("Resuming VAE from: ", args.vae_path)

        vae.load(args.vae_path if not args.use_ema or not ema_model_path else ema_model_path, map="cpu")

    if args.use_paintmind:
        # load VAE
        accelerator.print("Loading VQVAE from 'neggles/vaedump/vit-s-vqgan-f4' ...")
        vae: VQVAE = VQVAE.from_pretrained("neggles/vaedump", subfolder="vit-s-vqgan-f4")

    elif args.taming_model_path:
        print("Loading Taming VQGanVAE")
        vae = VQGanVAETaming(
            vqgan_model_path=args.taming_model_path,
            vqgan_config_path=args.taming_config_path,
        )
        args.num_tokens = vae.codebook_size
        args.seq_len = vae.get_encoded_fmap_size(args.image_size) ** 2

    # move vae to device
    vae = vae.to("cpu" if args.cpu else accelerator.device if args.gpu == 0 else f"cuda:{args.gpu}")

    # Use the parameters() method to get an iterator over all the learnable parameters of the model
    total_params = sum(p.numel() for p in vae.parameters())

    print(f"Total number of parameters: {format(total_params, ',d')}")

    # then you plug the vae and transformer into your MaskGit as so

    dataset = ImageDataset(
        dataset,
        args.image_size,
        image_column=args.image_column,
        center_crop=True if not args.no_center_crop and not args.random_crop else False,
        flip=not args.no_flip,
        random_crop=args.random_crop if args.random_crop else False,
        alpha_channel=False if args.channels == 3 else True,
    )

    if args.input_image and not args.input_folder:
        image_id = 0 if not args.random_image else random.randint(0, len(dataset))

        os.makedirs(f"{args.results_dir}/outputs", exist_ok=True)

        save_image(
            dataset[image_id],
            f"{args.results_dir}/outputs/input.{str(args.input_image).split('.')[-1]}",
            format="PNG",
        )

        _, ids, _ = vae.encode(
            dataset[image_id][None].to(
                "cpu" if args.cpu else accelerator.device if args.gpu == 0 else f"cuda:{args.gpu}"
            )
        )
        recon = vae.decode_from_ids(ids)
        save_image(recon, f"{args.results_dir}/outputs/output.{str(args.input_image).split('.')[-1]}")

    if not args.input_image and not args.input_folder:
        image_id = 0 if not args.random_image else random.randint(0, len(dataset))

        os.makedirs(f"{args.results_dir}/outputs", exist_ok=True)

        save_image(dataset[image_id], f"{args.results_dir}/outputs/input.png")

        _, ids, _ = vae.encode(
            dataset[image_id][None].to(
                "cpu" if args.cpu else accelerator.device if args.gpu == 0 else f"cuda:{args.gpu}"
            )
        )
        recon = vae.decode_from_ids(ids)
        save_image(recon, f"{args.results_dir}/outputs/output.png")

    if args.input_folder:
        # Create output directory and save input images and reconstructions as grids
        output_dir = os.path.join(args.results_dir, "outputs", os.path.basename(args.input_folder))
        os.makedirs(output_dir, exist_ok=True)

        for i in tqdm(range(len(dataset))):
            retries = 0
            while True:
                try:
                    save_image(dataset[i], f"{output_dir}/input.png")

                    if not args.use_paintmind:
                        # encode
                        _, ids, _ = vae.encode(
                            dataset[i][None].to(
                                "cpu"
                                if args.cpu
                                else accelerator.device
                                if args.gpu == 0
                                else f"cuda:{args.gpu}"
                            )
                        )
                        # decode
                        recon = vae.decode_from_ids(ids)
                        # print (recon.shape) # torch.Size([1, 3, 512, 1136])
                        save_image(recon, f"{output_dir}/output.png")
                    else:
                        # encode
                        encoded, _, _ = vae.encode(
                            dataset[i][None].to(
                                "cpu"
                                if args.cpu
                                else accelerator.device
                                if args.gpu == 0
                                else f"cuda:{args.gpu}"
                            )
                        )

                        # decode
                        recon = vae.decode(encoded).squeeze(0)
                        recon = torch.clamp(recon, -1.0, 1.0)
                        save_image(recon, f"{output_dir}/output.png")

                    # Load input and output images
                    input_image = PIL.Image.open(f"{output_dir}/input.png")
                    output_image = PIL.Image.open(f"{output_dir}/output.png")

                    # Create horizontal grid with input and output images
                    grid_image = PIL.Image.new(
                        "RGB" if args.channels == 3 else "RGBA",
                        (input_image.width + output_image.width, input_image.height),
                    )
                    grid_image.paste(input_image, (0, 0))
                    grid_image.paste(output_image, (input_image.width, 0))

                    # Save grid
                    now = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
                    hash = hashlib.sha1(input_image.tobytes()).hexdigest()

                    filename = f"{hash}_{now}-{os.path.basename(args.vae_path)}.png"
                    grid_image.save(f"{output_dir}/{filename}", format="PNG")

                    if not args.save_originals:
                        # Remove input and output images after the grid was made.
                        os.remove(f"{output_dir}/input.png")
                        os.remove(f"{output_dir}/output.png")
                    else:
                        os.makedirs(os.path.join(output_dir, "originals"), exist_ok=True)
                        shutil.move(
                            f"{output_dir}/input.png",
                            f"{os.path.join(output_dir, 'originals')}/input_{now}.png",
                        )
                        shutil.move(
                            f"{output_dir}/output.png",
                            f"{os.path.join(output_dir, 'originals')}/output_{now}.png",
                        )

                    del _
                    del ids
                    del recon

                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                    break  # Exit the retry loop if there were no errors

                except RuntimeError as e:
                    if "out of memory" in str(e) and retries < args.max_retries:
                        retries += 1
                        # print(f"Out of Memory. Retry #{retries}")
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        continue  # Retry the loop

                    else:
                        if "out of memory" not in str(e):
                            print(f"\n{e}")
                        else:
                            print(f"Skipping image {i} after {retries} retries due to out of memory error")
                        break  # Exit the retry loop after too many retries


if __name__ == "__main__":
    main()
