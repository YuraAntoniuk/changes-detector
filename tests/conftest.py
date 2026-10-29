from pathlib import Path

import numpy as np
import pytest
from PIL import Image

IMAGE_SIZE = 64
PATCH_SIZE = 32


def write_split(root: Path, split: str, n: int, rng: np.random.Generator) -> None:
    for d in ("A", "B", "label"):
        (root / split / d).mkdir(parents=True)
    for i in range(n):
        a = rng.integers(40, 160, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        b = a.copy()
        label = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
        y, x = rng.integers(0, IMAGE_SIZE - 16, 2)
        b[y:y + 16, x:x + 16] = 250  # a new "building"
        label[y:y + 16, x:x + 16] = 255
        name = f"{split}_{i}.png"
        Image.fromarray(a).save(root / split / "A" / name)
        Image.fromarray(b).save(root / split / "B" / name)
        Image.fromarray(label).save(root / split / "label" / name)


@pytest.fixture(scope="session")
def levir_root(tmp_path_factory) -> Path:
    """Tiny LEVIR-CD look-alike: same folder layout, 64px images."""
    root = tmp_path_factory.mktemp("levir")
    rng = np.random.default_rng(0)
    for split, n in (("train", 4), ("val", 2), ("test", 2)):
        write_split(root, split, n, rng)
    return root
