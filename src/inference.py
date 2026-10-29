from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.config import TrainConfig
from src.metrics import OpCounter, summarize_spike_rates
from src.pipeline import build_model, forward_pass, load_checkpoint


@dataclass
class ScenePrediction:
    logits: np.ndarray  # [H,W] float32, same size as the input scene
    spike_rates: dict[str, float] | None  # SNN only, per neuron per time step
    efficiency: dict | None  # OpCounter.summary(), per patch

    @property
    def probs(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logits))


def load_model(checkpoint: str | Path | None, device: torch.device) -> tuple[torch.nn.Module, TrainConfig, dict]:
    """checkpoint=None builds a randomly initialized model from the default
    config: useful to check the pipeline end to end before any training."""
    if checkpoint is None:
        cfg, meta = TrainConfig(), {}
        model = build_model(cfg)
    else:
        ckpt = load_checkpoint(checkpoint, map_location=device)
        cfg = ckpt.pop("config")
        model = build_model(cfg)
        model.load_state_dict(ckpt.pop("model_state"))
        meta = {k: ckpt[k] for k in ("epoch", "global_step", "best_f1", "val_metrics") if k in ckpt}
    return model.to(device).eval(), cfg, meta


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """[H,W,3] uint8 -> [3,H,W] float in [0,1]."""
    return torch.from_numpy(np.ascontiguousarray(img[..., :3])).permute(2, 0, 1).float() / 255.0


@torch.no_grad()
def predict_scene(
    model: torch.nn.Module,
    cfg: TrainConfig,
    img1: torch.Tensor,
    img2: torch.Tensor,
    device: torch.device,
    batch_size: int = 8,
    seed: int = 0,
    progress: Callable[[float], None] | None = None,
) -> ScenePrediction:
    """Tiled inference on a whole scene of any size.

    img1, img2: [3,H,W] in [0,1]. The scene is zero-padded (identically in
    both dates, so the padding reads as "no change") to a multiple of the
    training patch size, cut into non-overlapping patches like the val/test
    split, and the logits are stitched back and cropped to [H,W].
    """
    _, h, w = img1.shape
    p = cfg.data.patch_size
    ph, pw = -h % p, -w % p
    pair = torch.nn.functional.pad(torch.stack([img1, img2]), (0, pw, 0, ph))
    gh, gw = pair.shape[-2] // p, pair.shape[-1] // p
    # [2,3,gh*p,gw*p] -> [gh*gw,2,3,p,p], row-major over the grid
    patches = pair.unfold(2, p, p).unfold(3, p, p).permute(2, 3, 0, 1, 4, 5).reshape(-1, 2, 3, p, p)

    is_snn = hasattr(model, "spike_rate_report")
    if is_snn:
        model.reset_spike_stats()
    ops = OpCounter(model)
    out = torch.empty(len(patches), 1, p, p)
    devices = [device] if device.type == "cuda" else []
    try:
        # rate/difference encoding draws random spikes: a fixed seed makes a
        # rerun on the same scene give the same mask
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for i in range(0, len(patches), batch_size):
                batch = patches[i:i + batch_size].to(device)
                out[i:i + batch_size] = forward_pass(model, batch[:, 0], batch[:, 1], cfg).float().cpu()
                if progress is not None:
                    progress(min(1.0, (i + batch_size) / len(patches)))
    finally:
        ops.remove()

    logits = out.reshape(gh, gw, p, p).permute(0, 2, 1, 3).reshape(gh * p, gw * p)[:h, :w]
    time_steps = cfg.encoding.time_steps if is_snn else 1
    return ScenePrediction(
        logits=logits.numpy(),
        spike_rates=summarize_spike_rates(model.spike_rate_report()) if is_snn else None,
        efficiency=ops.summary(len(patches), time_steps=time_steps),
    )
