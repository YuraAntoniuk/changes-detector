import io
import os
import pathlib

import numpy as np
import torch
import torch.nn as nn

from src.config import TrainConfig
from src.inference import load_model, predict_scene, to_tensor
from src.pipeline import load_checkpoint


class PixelDiffModel(nn.Module):
    """Stand-in 'baseline' whose logit is a pure per-pixel function of the
    input, so tiled output can be compared against the untiled formula."""

    def forward(self, img1, img2):
        return (img2 - img1).sum(dim=1, keepdim=True) * 10


def test_predict_scene_stitches_tiles_back_in_place():
    cfg = TrainConfig(model="baseline")
    cfg.data.patch_size = 32
    img1, img2 = torch.rand(2, 3, 70, 100)  # not a multiple of the patch size
    pred = predict_scene(PixelDiffModel(), cfg, img1, img2, torch.device("cpu"), batch_size=3)
    expected = ((img2 - img1).sum(dim=0) * 10).numpy()
    assert pred.logits.shape == (70, 100)
    np.testing.assert_allclose(pred.logits, expected, rtol=1e-5, atol=1e-5)
    assert pred.spike_rates is None


def test_untrained_snn_reports_spikes_and_is_reproducible_with_seed():
    model, cfg, meta = load_model(None, torch.device("cpu"))
    cfg.encoding.time_steps = 2
    cfg.data.patch_size = 32
    img = to_tensor(np.random.default_rng(0).integers(0, 255, (48, 48, 3), dtype=np.uint8))
    a = predict_scene(model, cfg, img, img.flip(-1), torch.device("cpu"), seed=1)
    b = predict_scene(model, cfg, img, img.flip(-1), torch.device("cpu"), seed=1)
    assert meta == {}
    np.testing.assert_array_equal(a.logits, b.logits)
    assert 0 <= a.spike_rates["overall"] <= 1
    assert a.efficiency["dense_macs_per_pass"] > 0


def test_checkpoint_from_another_os_loads():
    class ForeignPath:
        # unpickles as a PosixPath on Windows / WindowsPath elsewhere
        def __reduce__(self):
            cls = pathlib.PosixPath if os.name == "nt" else pathlib.WindowsPath
            return (cls, ("some", "data", "root"))

    cfg = TrainConfig()
    cfg.data.root = ForeignPath()
    buf = io.BytesIO()
    torch.save({"config": cfg}, buf)
    loaded = load_checkpoint(io.BytesIO(buf.getvalue()))
    assert isinstance(loaded["config"].data.root, str)
    assert loaded["config"].data.root.replace("\\", "/") == "some/data/root"


def test_checkpoint_saved_before_upsample_option_keeps_transpose_decoder():
    old_cfg = TrainConfig()
    del vars(old_cfg.model_cfg)["upsample"]  # as pickled before the field existed
    new_cfg = TrainConfig()
    for cfg, expected in ((old_cfg, "transpose"), (new_cfg, "nearest")):
        buf = io.BytesIO()
        torch.save({"config": cfg}, buf)
        assert load_checkpoint(io.BytesIO(buf.getvalue()))["config"].model_cfg.upsample == expected
