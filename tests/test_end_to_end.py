"""The full workflow on the synthetic dataset: train -> resume -> evaluate ->
UI inference -> self-check. Small enough to run on CPU in seconds."""
import sys

import pytest

import app
import evaluate
import train
from src.pipeline import load_checkpoint
from src.selfcheck import run_checks
from tests.conftest import PATCH_SIZE


@pytest.fixture
def out_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(train, "RUNS_DIR", tmp_path / "runs")
    return tmp_path


def run_cli(module, monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", [module.__name__, *map(str, args)])
    module.main()


@pytest.mark.parametrize("model_kind", ["snn", "baseline"])
def test_train_resume_evaluate(model_kind, levir_root, out_dirs, monkeypatch, capsys):
    common = ["--data-root", levir_root, "--num-workers", 0]
    run_cli(
        train, monkeypatch, "--model", model_kind, "--patch-size", PATCH_SIZE, "--crops-per-image", 2,
        "--batch-size", 2, "--epochs", 2, "--time-steps", 2, "--max-steps-per-epoch", 2,
        "--run-name", "smoke", *common,
    )
    last = out_dirs / "checkpoints" / "smoke" / "last.pt"
    assert (out_dirs / "checkpoints" / "smoke" / "best.pt").exists()
    assert any((out_dirs / "runs" / "smoke").iterdir()), "no TensorBoard events written"
    ckpt = load_checkpoint(last)
    assert ckpt["epoch"] == 1 and ckpt["config"].model == model_kind
    assert isinstance(ckpt["config"].data.root, str)

    # resuming a finished run is a no-op that must still load cleanly
    run_cli(train, monkeypatch, "--resume", last, *common)
    assert "resumed smoke from epoch 1" in capsys.readouterr().out

    run_cli(evaluate, monkeypatch, "--checkpoint", last, "--split", "test", "--batch-size", 2, *common)
    out = capsys.readouterr().out
    assert "[test] segmentation metrics" in out
    assert ("spike rates" in out) == (model_kind == "snn")


def test_ui_detection_and_selfcheck(levir_root, out_dirs, monkeypatch):
    run_cli(
        train, monkeypatch, "--patch-size", PATCH_SIZE, "--crops-per-image", 1, "--batch-size", 2,
        "--epochs", 1, "--time-steps", 2, "--max-steps-per-epoch", 1, "--run-name", "ui",
        "--data-root", levir_root, "--num-workers", 0,
    )
    monkeypatch.setattr(app, "DATA_ROOT", levir_root)
    monkeypatch.setattr(app, "PROJECT_ROOT", out_dirs)
    monkeypatch.setattr(app, "CHECKPOINT_DIR", out_dirs / "checkpoints")
    monkeypatch.setattr(app, "RUNS_DIR", out_dirs / "runs")
    app.cached_model.cache_clear()

    choices = app.list_checkpoints()
    assert choices == ["checkpoints/ui/best.pt", "checkpoints/ui/last.pt", app.UNTRAINED]
    assert "epoch 0" in app.checkpoint_markdown(choices[0])

    no_progress = lambda *a, **k: None
    a, b, gt, prob, over, metrics, spikes, eff, state = app.run_detection(
        choices[0], "Dataset sample", "test", 1, None, None, None, "Full image", 0, 0.5, progress=no_progress
    )
    assert a.shape == prob.shape == over.shape == (64, 64, 3)
    assert "| f1 |" in metrics and "energy ratio" in eff and spikes is not None
    # moving the threshold re-renders without re-running the model
    over2, metrics2 = app.rethreshold(state, 0.9)
    assert over2.shape == over.shape and "| f1 |" in metrics2

    # uploaded pair without ground truth
    *_, metrics3, _, _, _ = app.run_detection(
        app.UNTRAINED, "Upload images", "test", 0, a, b, None, "Full image", 0, 0.5, progress=no_progress
    )
    assert "No ground truth" in metrics3

    fig, summary = app.training_curves("ui")
    assert fig is not None and "best val F1" in summary

    results = run_checks(levir_root, checkpoint_dir=out_dirs / "checkpoints", overfit=False)
    failed = [r for r in results if r.status == "FAIL"]
    assert not failed, failed
