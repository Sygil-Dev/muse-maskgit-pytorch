from __future__ import print_function

import glob
import shutil
import os
import re
import PIL
import torch
import hashlib
from tqdm import tqdm
from torchvision.utils import save_image
from datetime import datetime

def get_latest_checkpoints(resume_path, use_ema=False, model_type="vae", cond_image_size=False):
    """Gets the latest checkpoint paths for both the non-ema and ema VAEs.

    Args:
        resume_path: The path to the directory containing the VAE checkpoints.

    Returns:
        A tuple containing the paths to the latest non-ema and ema VAE checkpoints, respectively.
    """

    vae_path, _ = os.path.split(resume_path)
    if cond_image_size:
        checkpoint_files = glob.glob(os.path.join(vae_path, "maskgit_superres.*.pt"))
    else:
        checkpoint_files = glob.glob(
            os.path.join(vae_path, "vae.*.pt" if model_type == "vae" else "maskgit.*.pt")
        )
    # print(checkpoint_files)

    print(f"Finding latest {'VAE' if model_type == 'vae' else 'MaskGit'} checkpoint...")

    # Get the latest non-ema VAE checkpoint path
    if cond_image_size:
        latest_non_ema_checkpoint_file = max(
            checkpoint_files,
            key=lambda x: int(re.search(r"maskgit_superres\.(\d+)\.pt", x).group(1)),
        )
    else:
        latest_non_ema_checkpoint_file = max(
            checkpoint_files,
            key=lambda x: int(
                re.search(r"vae\.(\d+)\.pt$" if model_type == "vae" else r"maskgit\.(\d+)\.pt$", x).group(1)
            )
            if not x.endswith("ema.pt")
            else -1,
        )

    # Check if the latest checkpoints are empty or unreadable
    if os.path.getsize(latest_non_ema_checkpoint_file) == 0 or not os.access(
        latest_non_ema_checkpoint_file, os.R_OK
    ):
        print(f"Warning: latest checkpoint {latest_non_ema_checkpoint_file} is empty or unreadable.")
        if len(checkpoint_files) > 1:
            # Use the second last checkpoint as a fallback
            if cond_image_size:
                latest_non_ema_checkpoint_file = max(
                    checkpoint_files[:-1],
                    key=lambda x: int(re.search(r"maskgit_superres\.(\d+)\.pt", x).group(1)),
                )
            else:
                latest_non_ema_checkpoint_file = max(
                    checkpoint_files[:-1],
                    key=lambda x: int(
                        re.search(
                            r"vae\.(\d+)\.pt$" if model_type == "vae" else r"maskgit\.(\d+)\.pt$", x
                        ).group(1)
                    )
                    if not x.endswith("ema.pt")
                    else -1,
                )
            print("Using second last checkpoint: ", latest_non_ema_checkpoint_file)
        else:
            print("No usable checkpoint found.")

    if use_ema:
        # Get the latest ema VAE checkpoint path
        if cond_image_size:
            latest_ema_checkpoint_file = max(
                checkpoint_files,
                key=lambda x: int(re.search(r"maskgit_superres\.(\d+)\.ema\.pt$", x).group(1))
                if x.endswith("ema.pt")
                else -1,
            )
        else:
            latest_ema_checkpoint_file = max(
                checkpoint_files,
                key=lambda x: int(
                    re.search(
                        r"vae\.(\d+)\.ema\.pt$" if model_type == "vae" else r"maskgit\.(\d+)\.ema\.pt$", x
                    ).group(1)
                )
                if x.endswith("ema.pt")
                else -1,
            )

        if os.path.getsize(latest_ema_checkpoint_file) == 0 or not os.access(
            latest_ema_checkpoint_file, os.R_OK
        ):
            print(f"Warning: latest EMA checkpoint {latest_ema_checkpoint_file} is empty or unreadable.")
            if len(checkpoint_files) > 1:
                # Use the second last checkpoint as a fallback
                if cond_image_size:
                    latest_ema_checkpoint_file = max(
                        checkpoint_files[:-1],
                        key=lambda x: int(re.search(r"maskgit_superres\.(\d+)\.ema\.pt$", x).group(1))
                        if x.endswith("ema.pt")
                        else -1,
                    )
                else:
                    latest_ema_checkpoint_file = max(
                        checkpoint_files[:-1],
                        key=lambda x: int(
                            re.search(
                                r"vae\.(\d+)\.ema\.pt$"
                                if model_type == "vae"
                                else r"maskgit\.(\d+)\.ema\.pt$",
                                x,
                            ).group(1)
                        )
                        if x.endswith("ema.pt")
                        else -1,
                    )
                print("Using second last EMA checkpoint: ", latest_ema_checkpoint_file)
    else:
        latest_ema_checkpoint_file = None

    return latest_non_ema_checkpoint_file, latest_ema_checkpoint_file


def remove_duplicate_weights(ema_state_dict, non_ema_state_dict):
    """Removes duplicate weights from the ema state dictionary.

    Args:
      ema_state_dict: The state dictionary of the ema model.
      non_ema_state_dict: The state dictionary of the non-ema model.

    Returns:
      The ema state dictionary with duplicate weights removed.
    """

    ema_state_dict_copy = ema_state_dict.copy()
    for key, value in ema_state_dict.items():
        if key in non_ema_state_dict and torch.equal(ema_state_dict[key], non_ema_state_dict[key]):
            del ema_state_dict_copy[key]
    return ema_state_dict_copy

def vae_folder_validation(accelerator, vae, dataset, args=None):

    # Create output directory and save input images and reconstructions as grids
    output_dir = os.path.join(args.results_dir, "outputs", os.path.basename(args.input_folder))
    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        retries = 0
        while True:
            try:
                save_image(dataset[i], f"{output_dir}/input.png")

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