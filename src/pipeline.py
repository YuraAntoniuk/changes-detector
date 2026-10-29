import pathlib
import pickle
import types
from pathlib import Path

import torch

from src.config import TrainConfig
from src.encoding.encoders import encode_batch
from src.models.baseline_cnn import FCEF, FCSiamDiff
from src.models.snn_unet import SiameseSpikingUNet, SpikingUNet

# encoding.kind -> (siamese model builder, single-stream model builder)
_MODEL_BUILDERS = {
    "snn": (SiameseSpikingUNet, SpikingUNet),
    "baseline": (FCSiamDiff, FCEF),
}


def build_model(cfg: TrainConfig) -> torch.nn.Module:
    siamese_cls, single_stream_cls = _MODEL_BUILDERS[cfg.model]
    mc = cfg.model_cfg
    kwargs = {"channels": mc.channels, "bottleneck": mc.bottleneck}
    if cfg.model == "snn":
        kwargs["beta"] = mc.beta
        kwargs["use_checkpointing"] = cfg.use_checkpointing
    if cfg.encoding.kind == "difference":
        if cfg.model == "snn":
            kwargs["in_ch"] = 1
        return single_stream_cls(**kwargs)
    return siamese_cls(fusion_mode=mc.fusion_mode, **kwargs)


def forward_pass(model: torch.nn.Module, img1: torch.Tensor, img2: torch.Tensor, cfg: TrainConfig):
    # Only the SNN models consume spike-encoded input; the ANN baselines
    # (FCSiamDiff / FCEF) always take raw pixel tensors directly.
    if cfg.model != "snn":
        return model(img1, img2)
    encoded = encode_batch(img1, img2, cfg.encoding)
    if isinstance(encoded, tuple):
        return model(*encoded)
    return model(encoded)


class _PortableUnpickler(pickle.Unpickler):
    # Checkpoints written before DataConfig.root became a str hold a concrete
    # WindowsPath/PosixPath, which only unpickles on the OS that wrote it.
    # The pure variants load anywhere; load_checkpoint turns them into str.
    def find_class(self, module, name):
        if module.startswith("pathlib") and name in ("WindowsPath", "PosixPath"):
            return getattr(pathlib, "Pure" + name)
        return super().find_class(module, name)


_portable_pickle = types.SimpleNamespace(
    Unpickler=_PortableUnpickler, load=lambda f, **kw: _PortableUnpickler(f, **kw).load(), __name__="pickle"
)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    """torch.load for our own checkpoints (full pickle, trusted files only),
    readable regardless of which OS wrote them."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False, pickle_module=_portable_pickle)
    cfg = ckpt.get("config")
    if cfg is not None:
        cfg.data.root = str(cfg.data.root)
    return ckpt
