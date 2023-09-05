from __future__ import print_function
import re, glob, os, torch

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
        checkpoint_files = glob.glob(os.path.join(vae_path, "vae.*.pt" if model_type == "vae" else "maskgit.*.pt"))
    #print(checkpoint_files)

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
            key=lambda x: int(re.search(r"vae\.(\d+)\.pt$" if model_type == "vae" else r"maskgit\.(\d+)\.pt$", x).group(1))
            if not x.endswith("ema.pt")
            else -1,
        )

    # Check if the latest checkpoints are empty or unreadable
    if os.path.getsize(latest_non_ema_checkpoint_file) == 0 or not os.access(
        latest_non_ema_checkpoint_file, os.R_OK
    ):
        print(
            f"Warning: latest checkpoint {latest_non_ema_checkpoint_file} is empty or unreadable."
        )
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
                    key=lambda x: int(re.search(r"vae\.(\d+)\.pt$" if model_type == "vae" else r"maskgit\.(\d+)\.pt$", x).group(1))
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
                key=lambda x: int(re.search(r"vae\.(\d+)\.ema\.pt$" if model_type == "vae" else r"maskgit\.(\d+)\.ema\.pt$", x).group(1))
                if x.endswith("ema.pt")
                else -1,
            )

        if os.path.getsize(latest_ema_checkpoint_file) == 0 or not os.access(
            latest_ema_checkpoint_file, os.R_OK
        ):
            print(
                f"Warning: latest EMA checkpoint {latest_ema_checkpoint_file} is empty or unreadable."
            )
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
                        key=lambda x: int(re.search(r"vae\.(\d+)\.ema\.pt$" if model_type == "vae" else r"maskgit\.(\d+)\.ema\.pt$", x).group(1))
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