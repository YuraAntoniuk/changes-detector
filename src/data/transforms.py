import random

import torch


def random_flip_rotate(img1: torch.Tensor, img2: torch.Tensor, label: torch.Tensor):
    if random.random() < 0.5:
        img1, img2, label = img1.flip(-1), img2.flip(-1), label.flip(-1)
    if random.random() < 0.5:
        img1, img2, label = img1.flip(-2), img2.flip(-2), label.flip(-2)
    k = random.randint(0, 3)
    if k:
        img1, img2, label = (
            torch.rot90(img1, k, dims=(-2, -1)),
            torch.rot90(img2, k, dims=(-2, -1)),
            torch.rot90(label, k, dims=(-2, -1)),
        )
    return img1, img2, label
