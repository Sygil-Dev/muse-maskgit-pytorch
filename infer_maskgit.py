import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from muse_maskgit_pytorch import (
    # MaskGitTrainer,
    MaskGit,
    MaskGitTransformer,
    VQGanVAE,
    VQGanVAETaming,
    get_accelerator,
)
from muse_maskgit_pytorch.utils import (
    get_latest_checkpoints,
)

# Create the parser
parser = argparse.ArgumentParser()
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
parser.add_argument("--t5_name", type=str, default="t5-large", help="Name of your t5 model")
parser.add_argument("--cond_image_size", type=int, default=None, help="Conditional image size.")
parser.add_argument(
    "--prompt",
    type=str,
    default="A photo of a dog",
    help="Prompt to use for generation, you can use multiple prompts separated by |.",
)
parser.add_argument(
    "--timesteps", type=int, default=18, help="Number of steps to use for generating the image. Default: 18"
)
parser.add_argument(
    "--cond_scale",
    type=float,
    default=3.0,
    help="Conditional Scale to use for generating the image. Default: 3.0",
)
parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate. Default: 1")
parser.add_argument("--max_grad_norm", type=float, default=None, help="Max gradient norm.")
parser.add_argument("--seed", type=int, default=42, help="Seed.")
parser.add_argument("--valid_frac", type=float, default=0.05, help="validation fraction.")
parser.add_argument("--use_ema", action="store_true", help="Whether to use ema.")
parser.add_argument("--ema_beta", type=float, default=0.995, help="Ema beta.")
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
# vae_trainer args
parser.add_argument(
    "--vae_path",
    type=str,
    default=None,
    help="Path to the vae model. eg. 'results/vae.steps.pt'",
)
parser.add_argument("--dim", type=int, default=128, help="Model dimension.")
parser.add_argument("--batch_size", type=int, default=512, help="Batch Size.")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate.")
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Gradient Accumulation.",
)
parser.add_argument("--vq_codebook_size", type=int, default=256, help="Image Size.")
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
    "--validation_image_scale",
    default=1,
    type=float,
    help="Factor by which to scale the validation images.",
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
    "--latest_checkpoint",
    action="store_true",
    help="Automatically find and use the latest checkpoint in the folder.",
)


@dataclass
class Arguments:
    validation_image_scale: float = 1.0
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
    mixed_precision: str = "no"
    use_8bit_adam: bool = False
    results_dir: str = "results"
    resume_path: Optional[str] = None
    dim: int = 128
    batch_size: int = 512
    lr: float = 1e-5
    gradient_accumulation_steps: int = 1
    vq_codebook_size: int = 256
    vq_codebook_dim: int = 256
    cond_drop_prob: float = 0.5
    image_size: int = 256
    taming_model_path: Optional[str] = None
    taming_config_path: Optional[str] = None
    optimizer: str = "Lion"
    weight_decay: float = 0.0
    latest_checkpoint: bool = False
    do_not_save_config: bool = False
    use_l2_recon_loss: bool = False
    debug: bool = False
    config_path: Optional[str] = None
    generate_config: bool = False


def main():
    args = parser.parse_args(namespace=Arguments())

    accelerator = get_accelerator(
        mixed_precision=args.mixed_precision,
    )

    # Load the VAE
    with accelerator.main_process_first():
        if args.vae_path:
            print("Loading Muse VQGanVAE")

            if args.latest_checkpoint:
                args.vae_path, ema_model_path = get_latest_checkpoints(args.vae_path, use_ema=args.use_ema, model_type="vae")
                print(f"Resuming VAE from latest checkpoint: {args.resume_path}")
                #if args.use_ema:
                #    print(f"Resuming EMA VAE from latest checkpoint: {ema_model_path}")
            else:
                print("Resuming VAE from: ", args.vae_path)

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
            ).to("cpu" if args.cpu else accelerator.device if args.gpu == 0 else f"cuda:{args.gpu}")

            vae.load(args.vae_path)

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
            raise ValueError(
                "You must pass either vae_path or taming_model_path + taming_config_path (but not both)"
            )

    # freeze VAE before parsing to transformer
    vae.requires_grad_(False)

    # then you plug the vae and transformer into your MaskGit like so:

    ## (1) create your transformer / attention network
    # if args.attention_type == "flash":
    # xformers = False
    # flash = True
    # elif args.attention_type == "xformers":
    # xformers = True
    # flash = True
    # elif args.attention_type == "ein":
    # xformers = False
    # flash = False
    # else:
    # raise NotImplementedError(f'Attention of type "{args.attention_type}" does not exist')

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
        # cache_path=args.cache_path,
        # flash=flash,
        # xformers=xformers,
    )

    # (2) pass your trained VAE and the base transformer to MaskGit
    maskgit = MaskGit(
        vae=vae,  # vqgan vae
        transformer=transformer,  # transformer
        accelerator=accelerator,  # accelerator
        image_size=args.image_size,  # image size
        cond_drop_prob=args.cond_drop_prob,  # conditional dropout, for classifier free guidance
        cond_image_size=args.cond_image_size,
    )

    maskgit.to("cpu" if args.cpu else accelerator.device if args.gpu == 0 else f"cuda:{args.gpu}")

    # load the maskgit transformer from disk if we have previously trained one
    if args.resume_path is not None and len(args.resume_path) > 1:
        load = True

        accelerator.print("Loading Muse MaskGit...")

        if args.latest_checkpoint:
            args.resume_path, ema_model_path = get_latest_checkpoints(args.resume_path, use_ema=args.use_ema, model_type="maskgit", cond_image_size=args.cond_image_size)
            print(f"Resuming MaskGit from latest checkpoint: {args.resume_path}")
            #if args.use_ema:
            #    print(f"Resuming EMA MaskGit from latest checkpoint: {ema_model_path}")
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
        accelerator.print(
            "We need a MaskGit model to do inference with. Please provide a path to a checkpoint.."
        )

    # Use the parameters() method to get an iterator over all the learnable parameters of the model
    total_params = sum(p.numel() for p in maskgit.parameters())
    args.total_params = total_params

    print(f"Total number of parameters: {format(total_params, ',d')}")

    texts = [args.prompt] if "|" not in args.prompt else str(args.prompt).split("|")
    print(f"Prompt: {texts}")

    for i in tqdm(range(args.num_images), total=args.num_images):
        # ready your training text and images
        images = maskgit.generate(
            texts=texts,
            # cond_images=F.interpolate(torch.randn(1, 3, 512, 512).to('cpu' if args.cpu else accelerator.device if args.gpu == 0 else f"cuda:{args.gpu}"), 256),
            cond_scale=args.cond_scale,  # conditioning scale for classifier free guidance
            timesteps=args.timesteps,
            # fmap_size=None,
            # temperature=1.0,
            # topk_filter_thres=0.9,
            # can_remask_prev_masked=False,
            # force_not_use_token_critic=False,
            # critic_noise_scale=1,
        )

        # print(images.shape) # (3, 3, 256, 256)

        now = datetime.now().strftime("%m-%d-%Y_%H-%M-%S.%f")

        # save image to disk
        save_path = str(f"{args.results_dir}/outputs/validation/maskgit/{now}-{current_step}.png")
        os.makedirs(str(f"{args.results_dir}/outputs/validation/maskgit/"), exist_ok=True)

        save_image(images, save_path)

        del images
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
