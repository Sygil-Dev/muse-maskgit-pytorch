import os
import random
import sys
import time
from pathlib import Path
from threading import Thread

import datasets
import torch
from datasets import Image
from PIL import Image as pImage
from PIL import ImageFile
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as T
from tqdm_loggable.auto import tqdm
from transformers import T5Tokenizer

from muse_maskgit_pytorch.t5 import MAX_LENGTH

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):
    def __init__(
        self,
        dataset,
        image_size,
        image_column="image",
        flip=True,
        center_crop=True,
        stream=False,
    ):
        super().__init__()
        self.dataset = dataset
        self.image_column = image_column
        self.stream = stream
        transform_list = [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(image_size),
        ]
        if flip:
            transform_list.append(T.RandomHorizontalFlip())
        if center_crop:
            transform_list.append(T.CenterCrop(image_size))
        transform_list.append(T.ToTensor())
        self.transform = T.Compose(transform_list)

    def __len__(self):
        if not self.stream:
            return len(self.dataset)
        else:
            print("Using streaming, fetching length...")
            return int(self.dataset.info.dataset_size)

    def __getitem__(self, index):
        image = self.dataset[index][self.image_column]
        return self.transform(image) - 0.5


class ImageTextDataset(ImageDataset):
    def __init__(
        self,
        dataset,
        image_size: int,
        tokenizer: T5Tokenizer,
        image_column="image",
        caption_column="caption",
        flip=True,
        center_crop=True,
        stream=False,
    ):
        super().__init__(
            dataset,
            image_size=image_size,
            image_column=image_column,
            flip=flip,
            center_crop=center_crop,
            stream=stream,
        )
        self.caption_column: str = caption_column
        self.tokenizer: T5Tokenizer = tokenizer

    def __getitem__(self, index):
        image = self.dataset[index][self.image_column]
        descriptions = self.dataset[index][self.caption_column]
        if self.caption_column is None or descriptions is None:
            text = ""
        elif isinstance(descriptions, list):
            if len(descriptions) == 0:
                text = ""
            else:
                text = random.choice(descriptions)
        else:
            text = descriptions
        # max length from the paper
        encoded = self.tokenizer.batch_encode_plus(
            [str(text)],
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
        )

        input_ids = encoded.input_ids
        attn_mask = encoded.attention_mask
        return self.transform(image), input_ids[0], attn_mask[0]


class LocalTextImageDataset(Dataset):
    def __init__(self, path, image_size, tokenizer, flip=True, center_crop=True):
        super().__init__()
        self.tokenizer = tokenizer

        print("Building dataset...")

        extensions = ["jpg", "jpeg", "png", "webp"]
        self.image_paths = []
        self.caption_pair = []
        self.images = []

        for ext in extensions:
            self.image_paths.extend(list(Path(path).rglob(f"*.{ext}")))

        random.shuffle(self.image_paths)
        for image_path in tqdm(self.image_paths):
            # check image size and ignore images with 0 byte.
            if os.path.getsize(image_path) == 0:
                continue
            caption_path = image_path.with_suffix(".txt")
            if os.path.exists(str(caption_path)):
                captions = caption_path.read_text(encoding="utf-8").split("\n")
                captions = list(filter(lambda t: len(t) > 0, captions))
            else:
                captions = []
            self.images.append(image_path)
            self.caption_pair.append(captions)

        transform_list = [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(image_size),
        ]
        if flip:
            transform_list.append(T.RandomHorizontalFlip())
        if center_crop:
            transform_list.append(T.CenterCrop(image_size))
        transform_list.append(T.ToTensor())
        self.transform = T.Compose(transform_list)

    def __len__(self):
        return len(self.caption_pair)

    def __getitem__(self, index):
        image = self.images[index]
        image = pImage.open(image)
        descriptions = self.caption_pair[index]
        if descriptions is None:
            text = ""
        elif isinstance(descriptions, list):
            if len(descriptions) == 0:
                text = ""
            else:
                text = random.choice(descriptions)
        else:
            text = descriptions
        # max length from the paper
        encoded = self.tokenizer.batch_encode_plus(
            [str(text)],
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
        )

        input_ids = encoded.input_ids
        attn_mask = encoded.attention_mask
        return self.transform(image), input_ids[0], attn_mask[0]


def get_directory_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def save_dataset_with_progress(dataset, save_path):
    # Estimate the total size of the dataset in bytes
    total_size = sys.getsizeof(dataset)

    # Start saving the dataset in a separate thread
    save_thread = Thread(target=dataset.save_to_disk, args=(save_path,))
    save_thread.start()

    # Create a tqdm progress bar and update it periodically
    with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
        while save_thread.is_alive():
            if os.path.exists(save_path):
                size = get_directory_size(save_path)
                # Update the progress bar based on the current size of the saved file
                pbar.update(size - pbar.n)  # Update by the difference between current and previous size
            time.sleep(1)


def get_dataset_from_dataroot(data_root, image_column="image", caption_column="caption", save_path="dataset"):
    # Check if data_root is a symlink and resolve it to its target location if it is
    if os.path.islink(data_root):
        data_root = os.path.realpath(data_root)

    extensions = ["jpg", "jpeg", "png", "webp"]
    image_paths = []

    for ext in extensions:
        image_paths.extend(list(Path(data_root).rglob(f"*.{ext}")))

    random.shuffle(image_paths)
    data_dict = {image_column: [], caption_column: []}
    for image_path in tqdm(image_paths):
        # check image size and ignore images with 0 byte.
        if os.path.getsize(image_path) == 0:
            continue
        caption_path = image_path.with_suffix(".txt")
        if os.path.exists(str(caption_path)):
            captions = caption_path.read_text(encoding="utf-8").split("\n")
            captions = list(filter(lambda t: len(t) > 0, captions))
        else:
            captions = []
        image_path = str(image_path)
        data_dict[image_column].append(image_path)
        data_dict[caption_column].append(captions)
    dataset = datasets.Dataset.from_dict(data_dict)
    dataset = dataset.cast_column(image_column, Image())
    # dataset.save_to_disk(save_path)
    save_dataset_with_progress(dataset, save_path)
    return dataset


def split_dataset_into_dataloaders(dataset, valid_frac=0.05, seed=42, batch_size=1):
    print(f"Dataset length: {len(dataset)} samples")
    if valid_frac > 0:
        train_size = int((1 - valid_frac) * len(dataset))
        valid_size = len(dataset) - train_size
        print(f"Splitting dataset into {train_size} training samples and {valid_size} validation samples")
        split_generator = torch.Generator().manual_seed(seed)
        train_dataset, validation_dataset = random_split(
            dataset,
            [train_size, valid_size],
            generator=split_generator,
        )
    else:
        print("Using shared dataset for training and validation")
        train_dataset = dataset
        validation_dataset = dataset

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return dataloader, validation_dataloader
