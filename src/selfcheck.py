"""Pipeline health checks that need no trained model and no real training.

    python -m src.selfcheck            # all checks
    python -m src.selfcheck --quick    # skip the overfit checks

"Passing" means the system is wired correctly (data loads, every model /
encoding combination runs forward and backward, gradients reach all
parameters, the model can fit a toy task, checkpoints load) -- not that the
trained model is accurate.
"""
import argparse
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from src.config import CHECKPOINT_DIR, DataConfig, TrainConfig
from src.losses import BCEDiceLoss
from src.pipeline import build_model, forward_pass, load_checkpoint

SPLITS = ("train", "val", "test")
COMBOS = (("snn", "rate"), ("snn", "difference"), ("baseline", "rate"), ("baseline", "difference"))


@dataclass
class CheckResult:
    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    detail: str
    seconds: float


def _cfg(model: str, encoding: str, time_steps: int = 2) -> TrainConfig:
    cfg = TrainConfig(model=model)
    cfg.encoding.kind = encoding
    cfg.encoding.time_steps = time_steps
    return cfg


def toy_batch(batch_size: int = 4, size: int = 64, seed: int = 0):
    """Learnable synthetic pairs: t2 is t1 plus one bright square, label marks it."""
    g = torch.Generator().manual_seed(seed)
    img1 = 0.2 + 0.3 * torch.rand(batch_size, 3, size, size, generator=g)
    img2 = (img1 + 0.05 * torch.randn(batch_size, 3, size, size, generator=g)).clamp(0, 1)
    label = torch.zeros(batch_size, 1, size, size)
    for b in range(batch_size):
        s = size // 4
        y, x = torch.randint(0, size - s, (2,), generator=g).tolist()
        img2[b, :, y:y + s, x:x + s] = 1.0
        label[b, :, y:y + s, x:x + s] = 1.0
    return img1, img2, label


def check_environment(device: torch.device) -> tuple[str, str]:
    detail = f"torch {torch.__version__}, device {device}"
    if device.type == "cuda":
        detail += f" ({torch.cuda.get_device_name(device)})"
        return "PASS", detail
    return "WARN", detail + " -- no CUDA: training would be very slow"


def check_dataset(root: Path, split: str) -> tuple[str, str]:
    dirs = [root / split / d for d in ("A", "B", "label")]
    missing = [str(d) for d in dirs if not d.is_dir()]
    if missing:
        return "FAIL", "missing " + ", ".join(missing)
    names = [{p.name for p in d.glob("*.png")} for d in dirs]
    if not names[0]:
        return "FAIL", f"no PNGs in {dirs[0]}"
    if not names[0] == names[1] == names[2]:
        return "FAIL", f"A/B/label file names differ (A={len(names[0])}, B={len(names[1])}, label={len(names[2])})"
    return "PASS", f"{len(names[0])} image pairs"


def check_forward_backward(model_kind: str, encoding: str, device: torch.device) -> tuple[str, str]:
    cfg = _cfg(model_kind, encoding)
    model = build_model(cfg).to(device).train()
    img1, img2, label = (t.to(device) for t in toy_batch(batch_size=2))
    logits = forward_pass(model, img1, img2, cfg)
    if logits.shape != label.shape:
        return "FAIL", f"output shape {tuple(logits.shape)}, expected {tuple(label.shape)}"
    loss = BCEDiceLoss().to(device)(logits, label)
    loss.backward()
    if not torch.isfinite(loss):
        return "FAIL", f"loss is {loss.item()}"
    no_grad = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    if no_grad:
        return "FAIL", f"{len(no_grad)} parameters got no/non-finite gradient, e.g. {no_grad[0]}"
    n_params = sum(p.numel() for p in model.parameters())
    return "PASS", f"{type(model).__name__}, {n_params / 1e6:.2f}M params, loss {loss.item():.3f}"


def check_overfit(model_kind: str, device: torch.device, steps: int = 60) -> tuple[str, str]:
    """Can the model fit one fixed toy batch? Catches broken gradients /
    detached graphs / dead neurons that a single backward pass can miss."""
    torch.manual_seed(0)
    cfg = _cfg(model_kind, "rate", time_steps=4)
    model = build_model(cfg).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = BCEDiceLoss().to(device)
    img1, img2, label = (t.to(device) for t in toy_batch())
    losses = []
    for _ in range(steps):
        loss = loss_fn(forward_pass(model, img1, img2, cfg), label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    detail = f"loss {first:.3f} -> {last:.3f} in {steps} steps"
    return ("PASS" if last < 0.5 * first else "FAIL"), detail


def check_checkpoint(path: Path, device: torch.device) -> tuple[str, str]:
    ckpt = load_checkpoint(path, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    p = cfg.data.patch_size
    with torch.no_grad():
        logits = forward_pass(model, torch.rand(1, 3, p, p, device=device), torch.rand(1, 3, p, p, device=device), cfg)
    if not torch.isfinite(logits).all():
        return "FAIL", "non-finite output"
    f1 = ckpt.get("val_metrics", {}).get("f1", float("nan"))
    return "PASS", f"{cfg.model}/{cfg.encoding.kind}, epoch {ckpt.get('epoch')}, val F1 {f1:.3f}"


def run_checks(
    data_root: str | Path | None = None,
    checkpoint_dir: str | Path = CHECKPOINT_DIR,
    device: torch.device | None = None,
    overfit: bool = True,
    progress: Callable[[float, str], None] | None = None,
) -> list[CheckResult]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(data_root or DataConfig().root)
    checks: list[tuple[str, Callable[[], tuple[str, str]]]] = [("environment", lambda: check_environment(device))]
    checks += [(f"dataset: {s}", lambda s=s: check_dataset(root, s)) for s in SPLITS]
    checks += [
        (f"forward+backward: {m}/{e}", lambda m=m, e=e: check_forward_backward(m, e, device)) for m, e in COMBOS
    ]
    if overfit:
        # budgets for the default (parameter-free, nearest) upsampling; the
        # surrogate-gradient SNN fits the toy batch markedly slower than the CNN
        overfit_steps = {"snn": 150, "baseline": 150}
        checks += [
            (f"overfit toy batch: {m}", lambda m=m, n=n: check_overfit(m, device, steps=n))
            for m, n in overfit_steps.items()
        ]
    checks += [
        (f"checkpoint: {p.relative_to(checkpoint_dir).as_posix()}", lambda p=p: check_checkpoint(p, device))
        for p in sorted(Path(checkpoint_dir).glob("*/*.pt"))
    ]

    results = []
    for i, (name, fn) in enumerate(checks):
        if progress is not None:
            progress(i / len(checks), name)
        t0 = time.perf_counter()
        try:
            status, detail = fn()
        except Exception as e:  # a check crashing is itself the finding
            status, detail = "FAIL", f"{type(e).__name__}: {e}"
            traceback.print_exc()
        results.append(CheckResult(name, status, detail, time.perf_counter() - t0))
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--quick", action="store_true", help="skip the overfit checks")
    args = p.parse_args()
    results = run_checks(args.data_root, overfit=not args.quick)
    for r in results:
        print(f"[{r.status}] {r.name:42s} {r.seconds:6.1f}s  {r.detail}")
    failed = sum(r.status == "FAIL" for r in results)
    print(f"\n{len(results) - failed}/{len(results)} checks without failure")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
