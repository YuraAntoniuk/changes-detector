import pytest
import torch
import torch.nn as nn

from src.config import TrainConfig
from src.data.dataset import LEVIRCDDataset
from src.encoding.encoders import difference_encode, encode_batch, rate_encode
from src.losses import BCEDiceLoss
from src.metrics import OpCounter, SegMetricsAccumulator
from src.pipeline import build_model, forward_pass
from tests.conftest import IMAGE_SIZE, PATCH_SIZE


def test_rate_encode_is_binary_with_rate_tracking_intensity():
    img = torch.tensor([0.0, 0.5, 1.0]).view(1, 3, 1, 1).expand(1, 3, 20, 20)
    torch.manual_seed(0)
    spikes = rate_encode(img, time_steps=200)
    assert spikes.shape == (200, 1, 3, 20, 20)
    assert set(spikes.unique().tolist()) <= {0.0, 1.0}
    rates = spikes.mean(dim=(0, 1, 3, 4))
    assert rates[0] == 0 and rates[2] == 1 and abs(rates[1] - 0.5) < 0.02


def test_difference_encode_is_silent_where_nothing_changed():
    img1 = torch.rand(2, 3, 16, 16)
    img2 = img1.clone()
    img2[:, :, :8] = (img2[:, :, :8] + 0.6).clamp(max=1.0)
    spikes = difference_encode(img1, img2, time_steps=8, threshold=0.1)
    assert spikes.shape == (8, 2, 1, 16, 16)
    assert spikes[:, :, :, 8:].sum() == 0
    assert spikes[:, :, :, :8].sum() > 0


def test_encode_batch_dispatch_and_spectral_guard():
    cfg = TrainConfig().encoding
    img = torch.rand(1, 3, 8, 8)
    assert isinstance(encode_batch(img, img, cfg), tuple)
    cfg.kind = "difference"
    assert isinstance(encode_batch(img, img, cfg), torch.Tensor)
    cfg.kind = "spectral"
    with pytest.raises(NotImplementedError):
        encode_batch(img, img, cfg)


@pytest.mark.parametrize("model_kind,encoding", [
    ("snn", "rate"), ("snn", "difference"), ("baseline", "rate"), ("baseline", "difference"),
])
def test_every_model_runs_forward_and_backward(model_kind, encoding):
    cfg = TrainConfig(model=model_kind)
    cfg.encoding.kind = encoding
    cfg.encoding.time_steps = 2
    model = build_model(cfg)
    img1, img2 = torch.rand(2, 2, 3, 32, 32)
    label = (torch.rand(2, 1, 32, 32) > 0.9).float()
    logits = forward_pass(model, img1, img2, cfg)
    assert logits.shape == (2, 1, 32, 32)
    BCEDiceLoss()(logits, label).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_segmentation_metrics_on_known_counts():
    target = torch.tensor([1.0, 1.0, 0.0, 0.0]).view(1, 1, 2, 2)
    logits = torch.tensor([5.0, -5.0, 5.0, -5.0]).view(1, 1, 2, 2)  # 1 TP, 1 FN, 1 FP, 1 TN
    acc = SegMetricsAccumulator()
    acc.update(logits, target)
    m = acc.compute()
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["iou"] == pytest.approx(1 / 3)
    assert m["overall_accuracy"] == pytest.approx(0.5)


def test_op_counter_counts_spikes_as_accumulates_and_reals_as_macs():
    conv = nn.Conv2d(1, 4, 3, padding=1)
    ops = OpCounter(conv)
    x = torch.zeros(1, 1, 4, 4)
    x[0, 0, 0, :3] = 1.0  # 3 spikes
    x[0, 0, 1, 0] = 0.5  # 1 real value
    conv(x)
    ops.remove()
    s = ops.stats[""]
    fanout = 4 * 9  # out_channels * kernel area
    assert s["ac"] == 3 * fanout
    assert s["mac"] == 1 * fanout
    assert s["dense_macs"] == 16 * fanout


def test_dataset_tiles_eval_split_and_crops_train_split(levir_root):
    val = LEVIRCDDataset(levir_root, "val", patch_size=PATCH_SIZE)
    grid = IMAGE_SIZE // PATCH_SIZE
    assert len(val) == 2 * grid * grid
    img1, img2, label = val[0]
    assert img1.shape == img2.shape == (3, PATCH_SIZE, PATCH_SIZE)
    assert label.shape == (1, PATCH_SIZE, PATCH_SIZE)
    assert 0 <= img1.min() and img1.max() <= 1 and set(label.unique().tolist()) <= {0.0, 1.0}
    # the tiles of one image must hold exactly its label pixels (16x16 building)
    assert sum(val[i][2].sum().item() for i in range(grid * grid)) == 16 * 16

    train = LEVIRCDDataset(levir_root, "train", patch_size=PATCH_SIZE, train=True, crops_per_image=3)
    assert len(train) == 4 * 3
    assert train[0][0].shape == (3, PATCH_SIZE, PATCH_SIZE)
