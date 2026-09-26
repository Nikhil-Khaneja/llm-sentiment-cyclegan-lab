"""Unpaired Monet (domain A) / Photo (domain B) image data.

Direction names used everywhere: A2B = Monet -> Photo, B2A = Photo -> Monet (the Kaggle direction).
"""
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

EXTS = {".jpg", ".jpeg", ".png"}


def list_images(folder: Path):
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        raise FileNotFoundError(f"no images in {folder} - see task3_gan/data/README.md")
    return files


def split_domain(files, holdout, seed):
    """Fixed hold-out split: `holdout` images never seen in training (used for evaluation)."""
    files = list(files)
    random.Random(seed).shuffle(files)
    return files[holdout:], files[:holdout]


def train_transform(size, load_size):
    return T.Compose([
        T.Resize(load_size, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomCrop(size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1] to match the generator's tanh
    ])


def eval_transform(size):
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])


class Unpaired(Dataset):
    """Each item pairs A[i] with a *random* B image, so no fixed pairing is ever learned.
    Epoch length = size of the larger domain."""

    def __init__(self, files_a, files_b, transform, epoch_size=None):
        self.a, self.b, self.tf = files_a, files_b, transform
        self.n = epoch_size or max(len(files_a), len(files_b))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        a = Image.open(self.a[i % len(self.a)]).convert("RGB")
        b = Image.open(random.choice(self.b)).convert("RGB")
        return self.tf(a), self.tf(b)


class Folder(Dataset):
    def __init__(self, files, transform):
        self.files, self.tf = files, transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert("RGB")), i


def to_uint8(x: torch.Tensor):
    """[-1, 1] float image batch -> uint8 HWC numpy batch."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
