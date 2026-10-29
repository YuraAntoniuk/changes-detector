import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.transforms import random_flip_rotate


class LEVIRCDDataset(Dataset):
    """LEVIR-CD pairs served as patches.

    On first use each (A, B, label) PNG triplet is decoded once into an
    uncompressed uint8 `.npy` array [7,H,W] (RGB t1, RGB t2, binary label)
    under `<root>/_cache/<split>/`. Patches are then read through a memory map,
    so a 128px crop touches only its own rows instead of decoding three full
    1024px PNGs.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        patch_size: int = 128,
        train: bool = False,
        crops_per_image: int = 1,
    ):
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.train = train

        dir_a = self.root / split / "A"
        self.names = sorted(p.name for p in dir_a.glob("*.png"))
        if not self.names:
            raise FileNotFoundError(f"No images found under {dir_a}")

        self.cache_dir = self.root / "_cache" / split
        self._build_cache()

        self.image_size = np.load(self._cache_path(self.names[0]), mmap_mode="r").shape[-1]
        assert self.image_size % patch_size == 0, (
            f"image_size={self.image_size} must be divisible by patch_size={patch_size}"
        )
        self.grid = self.image_size // patch_size

        if train:
            # several random crops per image per epoch, so one epoch covers a
            # meaningful fraction of each 1024px scene, not a single 128px window
            self.index = [i for i in range(len(self.names)) for _ in range(crops_per_image)]
        else:
            self.index = [
                (img_idx, gy, gx)
                for img_idx in range(len(self.names))
                for gy in range(self.grid)
                for gx in range(self.grid)
            ]

    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{Path(name).stem}.npy"

    def _build_cache(self) -> None:
        missing = [n for n in self.names if not self._cache_path(n).exists()]
        if not missing:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        split_dir = self.root / self.split
        for name in tqdm(missing, desc=f"caching {self.split}"):
            img1 = np.array(Image.open(split_dir / "A" / name).convert("RGB"))
            img2 = np.array(Image.open(split_dir / "B" / name).convert("RGB"))
            label = np.array(Image.open(split_dir / "label" / name).convert("L"))
            arr = np.concatenate(
                [img1.transpose(2, 0, 1), img2.transpose(2, 0, 1), (label > 127)[None].astype(np.uint8)]
            )
            # write-then-rename so an interrupted run never leaves a truncated cache file
            path = self._cache_path(name)
            tmp = path.with_name(path.stem + ".tmp.npy")
            np.save(tmp, arr)
            tmp.replace(path)

    def __len__(self) -> int:
        return len(self.index)

    def _load_patch(self, name: str, y: int, x: int):
        p = self.patch_size
        arr = np.load(self._cache_path(name), mmap_mode="r")
        patch = torch.from_numpy(np.array(arr[:, y:y + p, x:x + p]))
        img1 = patch[0:3].float() / 255.0
        img2 = patch[3:6].float() / 255.0
        label = patch[6:7].float()
        return img1, img2, label

    def __getitem__(self, idx: int):
        p = self.patch_size
        if self.train:
            name = self.names[self.index[idx]]
            y = random.randint(0, self.image_size - p)
            x = random.randint(0, self.image_size - p)
            img1, img2, label = self._load_patch(name, y, x)
            img1, img2, label = random_flip_rotate(img1, img2, label)
        else:
            img_idx, gy, gx = self.index[idx]
            img1, img2, label = self._load_patch(self.names[img_idx], gy * p, gx * p)

        return img1.contiguous(), img2.contiguous(), label.contiguous()
