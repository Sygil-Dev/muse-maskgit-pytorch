import logging
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image

from muse_maskgit_pytorch.vqvae import VQVAE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# where to find the model and the test images
model_repo = "neggles/vaedump"
model_subdir = "vit-s-vqgan-f4"
test_images = ["testimg_1.png", "testimg_2.png"]

# where to save the preprocessed and reconstructed images
image_dir = Path.cwd().joinpath("temp")
image_dir.mkdir(exist_ok=True, parents=True)

# image transforms for the VQVAE
transform_enc = T.Compose([T.Resize(512), T.RandomCrop(256), T.ToTensor()])
transform_dec = T.Compose([T.ConvertImageDtype(torch.uint8), T.ToPILImage()])


def get_save_path(path: Path, append: str) -> Path:
    # append a string to the filename before the extension
    # n.b. only keeps the final suffix, e.g. "foo.xyz.png" -> "foo-prepro.png"
    return path.with_name(f"{path.stem}-{append}{path.suffix}")


def main():
    torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32

    # load VAE
    logger.info(f"Loading VQVAE from {model_repo}/{model_subdir}...")
    vae: VQVAE = VQVAE.from_pretrained(model_repo, subfolder=model_subdir, torch_dtype=dtype)
    vae = vae.to(torch_device)
    logger.info(f"Loaded VQVAE from {model_repo} to {vae.device} with dtype {vae.dtype}")

    # download and process images
    for image in test_images:
        image_path = hf_hub_download(model_repo, subfolder="images", filename=image, local_dir=image_dir)
        image_path = Path(image_path)
        logger.info(f"Downloaded {image_path}, size {image_path.stat().st_size} bytes")

        # preprocess
        image_obj = Image.open(image_path).convert("RGB")
        image_tensor: torch.Tensor = transform_enc(image_obj)
        save_path = get_save_path(image_path, "prepro")
        save_image(image_tensor, save_path, normalize=True, range=(-1.0, 1.0))
        logger.info(f"Saved preprocessed image to {save_path}")

        # encode
        encoded, _, _ = vae.encode(image_tensor.unsqueeze(0).to(vae.device))

        # decode
        reconstructed = vae.decode(encoded).squeeze(0)
        reconstructed = torch.clamp(reconstructed, -1.0, 1.0)

        # save
        save_path = get_save_path(image_path, "recon")
        save_image(reconstructed, save_path, normalize=True, range=(-1.0, 1.0))
        logger.info(f"Saved reconstructed image to {save_path}")

        # compare
        image_prepro = transform_dec(image_tensor)
        image_recon = transform_dec(reconstructed)
        canvas = Image.new("RGB", (512 + 12, 256 + 8), (248, 248, 242))
        canvas.paste(image_prepro, (4, 4))
        canvas.paste(image_recon, (256 + 8, 4))
        save_path = get_save_path(image_path, "compare")
        canvas.save(save_path)
        logger.info(f"Saved comparison image to {save_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
