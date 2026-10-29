from typing import Literal

import torch
import snntorch.spikegen as spikegen

from src.config import EncodingConfig

# Sentinel-2 band -> role, for spectral-index coding once OSCD is ingested.
SENTINEL2_BANDS = {
    "red": "B04",
    "green": "B03",
    "nir": "B08",
    "swir1": "B11",
}


def rate_encode(img: torch.Tensor, time_steps: int, gain: float = 1.0) -> torch.Tensor:
    """img: [B,C,H,W] in [0,1] -> [T,B,C,H,W] spikes in {0,1}."""
    return spikegen.rate(img, num_steps=time_steps, gain=gain)


def difference_encode(
    img_t1: torch.Tensor,
    img_t2: torch.Tensor,
    time_steps: int,
    threshold: float,
    reduce: Literal["max", "mean"] = "max",
) -> torch.Tensor:
    """img_t1, img_t2: [B,C,H,W] in [0,1] -> [T,B,1,H,W] spikes in {0,1}.

    Single event stream: magnitude of |t2-t1| above `threshold`, rate-coded so
    sub-threshold pixels never fire while above-threshold pixels get a
    magnitude-modulated spike train (not an identical frame repeated T times).
    """
    diff = (img_t2 - img_t1).abs()
    mag = diff.amax(dim=1, keepdim=True) if reduce == "max" else diff.mean(dim=1, keepdim=True)
    mag = (mag - threshold).clamp(min=0) / max(1e-6, 1 - threshold)
    return spikegen.rate(mag, num_steps=time_steps)


def compute_spectral_indices(
    bands: dict[str, torch.Tensor],
    indices: tuple[str, ...] = ("ndvi", "ndwi", "ndbi"),
    eps: float = 1e-6,
) -> torch.Tensor:
    """bands: {"red","nir","green","swir1"} -> [B, len(indices), H, W] each rescaled to [0,1].

    Requires multispectral input (e.g. OSCD/Sentinel-2); not usable on LEVIR-CD (RGB only).
    """
    out = []
    for idx in indices:
        if idx == "ndvi":
            num, den = bands["nir"] - bands["red"], bands["nir"] + bands["red"]
        elif idx == "ndwi":
            num, den = bands["green"] - bands["nir"], bands["green"] + bands["nir"]
        elif idx == "ndbi":
            num, den = bands["swir1"] - bands["nir"], bands["swir1"] + bands["nir"]
        else:
            raise ValueError(f"unknown spectral index {idx!r}")
        raw = num / (den + eps)
        out.append((raw + 1) / 2)
    return torch.cat(out, dim=1)


def encode_batch(
    img_t1: torch.Tensor, img_t2: torch.Tensor, cfg: EncodingConfig
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Dispatch on cfg.kind. Return shape signals the model family to use:
    tuple -> two-branch Siamese model, single Tensor -> single-stream model.
    """
    if cfg.kind == "rate":
        return rate_encode(img_t1, cfg.time_steps), rate_encode(img_t2, cfg.time_steps)
    if cfg.kind == "difference":
        return difference_encode(img_t1, img_t2, cfg.time_steps, cfg.diff_threshold)
    if cfg.kind == "spectral":
        raise NotImplementedError(
            "spectral-index encoding needs multispectral bands (NIR/SWIR); "
            "not available for LEVIR-CD (RGB only) -- pending OSCD ingestion"
        )
    raise ValueError(f"unknown encoding kind {cfg.kind!r}")
