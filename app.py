"""Local web UI: run change detection on LEVIR-CD pairs or uploaded images,
browse training curves, run the pipeline self-check.

    python app.py                      # http://127.0.0.1:7860
    python app.py --host 0.0.0.0       # reachable from other machines (e.g. a rented server)
"""
import argparse
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.config import CHECKPOINT_DIR, PROJECT_ROOT, RUNS_DIR, DataConfig
from src.inference import ScenePrediction, load_model, predict_scene, to_tensor
from src.metrics import SegMetricsAccumulator
from src.selfcheck import run_checks

UNTRAINED = "(untrained: random weights)"
DATA_ROOT = Path(DataConfig().root)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REGIONS = {"Full image": None, "512 px center crop": 512, "256 px center crop": 256}

# overlay colors: correct change, false alarm, missed change, prediction without ground truth
TP_RGB, FP_RGB, FN_RGB, PRED_RGB = (0, 200, 70), (235, 50, 50), (40, 130, 255), (255, 190, 0)


# ---------- data / model helpers ----------

def list_checkpoints() -> list[str]:
    found = sorted(p.relative_to(PROJECT_ROOT).as_posix() for p in CHECKPOINT_DIR.glob("*/*.pt"))
    return found + [UNTRAINED]


def list_samples(split: str) -> list[str]:
    return sorted(p.name for p in (DATA_ROOT / split / "A").glob("*.png"))


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_sample(split: str, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    names = list_samples(split)
    if not names:
        raise gr.Error(f"No images under {DATA_ROOT / split / 'A'}")
    name = names[int(index)]
    d = DATA_ROOT / split
    label = np.array(Image.open(d / "label" / name).convert("L")) > 127
    return read_rgb(d / "A" / name), read_rgb(d / "B" / name), label, name


@lru_cache(maxsize=2)
def cached_model(choice: str):
    return load_model(None if choice == UNTRAINED else PROJECT_ROOT / choice, DEVICE)


def center_crop(arr: np.ndarray | None, size: int | None) -> np.ndarray | None:
    if arr is None or size is None:
        return arr
    h, w = arr.shape[:2]
    y, x = max(0, (h - size) // 2), max(0, (w - size) // 2)
    return arr[y:y + size, x:x + size]


# ---------- rendering ----------

def prob_heatmap(probs: np.ndarray) -> np.ndarray:
    return (matplotlib.colormaps["magma"](probs)[..., :3] * 255).astype(np.uint8)


def overlay(img: np.ndarray, pred: np.ndarray, label: np.ndarray | None) -> np.ndarray:
    out = img.astype(np.float32) * 0.55
    if label is None:
        layers = [(pred, PRED_RGB)]
    else:
        layers = [(pred & label, TP_RGB), (pred & ~label, FP_RGB), (~pred & label, FN_RGB)]
    for mask, rgb in layers:
        out[mask] = 0.25 * out[mask] + 0.75 * np.array(rgb, np.float32)
    return out.clip(0, 255).astype(np.uint8)


def metrics_markdown(logits: np.ndarray, label: np.ndarray | None, threshold: float) -> str:
    pred_frac = float((logits > np.log(threshold / (1 - threshold))).mean())
    if label is None:
        return f"No ground truth for this pair. Predicted changed area: **{pred_frac:.2%}** of pixels."
    acc = SegMetricsAccumulator(threshold)
    acc.update(torch.from_numpy(logits), torch.from_numpy(label))
    m = acc.compute()
    rows = "\n".join(f"| {k.replace('_', ' ')} | {v:.4f} |" for k, v in m.items())
    return (
        f"| metric (change class) | value |\n|---|---|\n{rows}\n\n"
        f"Changed pixels: ground truth **{label.mean():.2%}**, predicted **{pred_frac:.2%}**."
    )


def spike_plot(rates: dict[str, float] | None):
    if not rates:
        return None
    layers = [k for k in rates if k != "overall"]
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(layers) + 1))
    ax.barh([n.replace("_stages", "").replace(".block", "") for n in layers], [rates[n] for n in layers],
            color="#4c78a8")
    ax.axvline(rates["overall"], color="#e45756", ls="--", label=f"overall {rates['overall']:.3f}")
    ax.invert_yaxis()
    ax.set_xlabel("spikes per neuron per time step")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def efficiency_markdown(eff: dict | None, is_snn: bool, patch: int) -> str:
    if eff is None:
        return ""
    lines = [
        f"Per {patch}x{patch} patch (estimated, 45 nm energy constants):",
        "",
        "| | value |",
        "|---|---|",
        f"| dense MACs, ANN-equivalent | {eff['dense_macs_per_pass']:.3e} |",
    ]
    if is_snn:
        lines += [
            f"| SNN accumulates (AC) | {eff['ac_per_sample']:.3e} |",
            f"| SNN multiply-accumulates (MAC) | {eff['mac_per_sample']:.3e} |",
            f"| SNN energy | {eff['energy_j_per_sample']:.3e} J |",
            f"| ANN-equivalent energy | {eff['ann_equivalent_energy_j']:.3e} J |",
            f"| **SNN / ANN energy ratio** | **{eff['energy_ratio_vs_ann']:.3f}** (< 1 means the SNN is cheaper) |",
        ]
    else:
        lines.append(f"| energy (dense MACs) | {eff['ann_equivalent_energy_j']:.3e} J |")
    return "\n".join(lines)


def checkpoint_markdown(choice: str) -> str:
    _, cfg, meta = cached_model(choice)
    head = (
        f"**{cfg.model} / {cfg.encoding.kind} encoding**, T={cfg.encoding.time_steps}, "
        f"patch {cfg.data.patch_size}px, channels {cfg.model_cfg.channels}"
    )
    if not meta:
        return head + "  \nRandom weights: output is meaningless, but every stage of the pipeline runs."
    vm = meta.get("val_metrics", {})
    return head + (
        f"  \nepoch {meta.get('epoch')}, val F1 {vm.get('f1', float('nan')):.3f}, "
        f"val IoU {vm.get('iou', float('nan')):.3f}"
    )


# ---------- event handlers ----------

def preview(split: str, index: int):
    a, b, label, name = load_sample(split, index)
    return a, b, (label * 255).astype(np.uint8), f"`{split}/{name}` ({a.shape[1]}x{a.shape[0]})"


def on_split(split: str):
    n = len(list_samples(split))
    return gr.Slider(minimum=0, maximum=max(0, n - 1), value=0, step=1)


def on_source(source: str):
    dataset = source == "Dataset sample"
    return gr.Column(visible=dataset), gr.Column(visible=not dataset)


def run_detection(choice, source, split, index, up_a, up_b, up_label, region, seed, threshold,
                  progress=gr.Progress()):
    if source == "Dataset sample":
        a, b, label, _ = load_sample(split, index)
    else:
        if up_a is None or up_b is None:
            raise gr.Error("Upload both the 'before' and the 'after' image.")
        a, b = up_a[..., :3], up_b[..., :3]
        if a.shape != b.shape:
            raise gr.Error(f"Image sizes differ: {a.shape[:2]} vs {b.shape[:2]} (they must be co-registered).")
        label = None
        if up_label is not None:
            label = (up_label if up_label.ndim == 2 else up_label[..., 0]) > 127
            if label.shape != a.shape[:2]:
                raise gr.Error("The ground-truth mask must have the same size as the images.")
    size = REGIONS[region]
    a, b, label = center_crop(a, size), center_crop(b, size), center_crop(label, size)

    progress(0, desc="loading model")
    model, cfg, _ = cached_model(choice)
    pred: ScenePrediction = predict_scene(
        model, cfg, to_tensor(a), to_tensor(b), DEVICE, seed=int(seed),
        progress=lambda f: progress(f, desc=f"inference on {DEVICE}"),
    )
    state = {"logits": pred.logits, "label": label, "after": b}
    over, metrics = rethreshold(state, threshold)
    is_snn = pred.spike_rates is not None
    eff_md = efficiency_markdown(pred.efficiency, is_snn, cfg.data.patch_size)
    if not is_snn:
        eff_md = "CNN baseline: no spikes.\n\n" + eff_md
    gt = None if label is None else (label * 255).astype(np.uint8)
    return a, b, gt, prob_heatmap(pred.probs), over, metrics, spike_plot(pred.spike_rates), eff_md, state


def rethreshold(state, threshold):
    if not state:
        return None, ""
    logits = state["logits"]
    pred = logits > np.log(threshold / (1 - threshold))
    return overlay(state["after"], pred, state["label"]), metrics_markdown(logits, state["label"], threshold)


def list_runs() -> list[str]:
    return sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir()) if RUNS_DIR.is_dir() else []


def training_curves(run: str | None):
    if not run:
        return None, "No TensorBoard runs found under `runs/`."
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(RUNS_DIR / run), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])

    def series(tag):
        ev = ea.Scalars(tag) if tag in tags else []
        return np.array([e.step for e in ev]), np.array([e.value for e in ev])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    steps, loss = series("train/loss")
    ax = axes[0, 0]
    if len(loss):
        k = max(1, len(loss) // 100)
        smooth = np.convolve(loss, np.ones(k) / k, mode="valid")
        ax.plot(steps, loss, color="#4c78a8", alpha=0.2, lw=0.5)
        ax.plot(steps[k - 1:], smooth, color="#4c78a8", label=f"train (moving avg, {k} steps)")
        ax.legend()
    ax.set(title="training loss", xlabel="step")
    for ax, tags_, title in (
        (axes[0, 1], ("val/loss",), "validation loss"),
        (axes[1, 0], ("val/f1", "val/iou"), "validation F1 / IoU"),
        (axes[1, 1], ("val/precision", "val/recall"), "validation precision / recall"),
    ):
        for tag in tags_:
            x, y = series(tag)
            if len(y):
                ax.plot(x, y, marker="o", ms=3, label=tag.split("/")[1])
        ax.set(title=title, xlabel="epoch")
        ax.legend()
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout()

    _, f1 = series("val/f1")
    if len(f1):
        best = int(np.argmax(f1))
        summary = f"{len(f1)} epochs logged, best val F1 **{f1[best]:.4f}** at epoch {best}."
    else:
        summary = f"No validation metrics logged ({len(loss)} training steps)."
    return fig, summary


def system_check(quick: bool, progress=gr.Progress()):
    results = run_checks(overfit=not quick, progress=lambda f, name: progress(f, desc=name))
    rows = [[r.status, r.name, r.detail, f"{r.seconds:.1f}"] for r in results]
    failed = sum(r.status == "FAIL" for r in results)
    warned = sum(r.status == "WARN" for r in results)
    verdict = "all checks passed" if not failed else f"**{failed} check(s) failed** (see the console for tracebacks)"
    return rows, f"{len(results)} checks: {verdict}" + (f", {warned} warning(s)" if warned else "") + "."


# ---------- layout ----------

def build_ui() -> gr.Blocks:
    checkpoints = list_checkpoints()
    test_count = len(list_samples("test"))
    runs = list_runs()

    with gr.Blocks(title="SNN Change Detection") as demo:
        gr.Markdown(
            "# SNN change detection\n"
            f"Spiking Siamese U-Net on LEVIR-CD. Device: **{DEVICE}**"
            + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else " (inference will be slow)")
        )
        with gr.Tab("Change detection"):
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    ckpt = gr.Dropdown(checkpoints, value=checkpoints[0], label="Model")
                    ckpt_info = gr.Markdown(checkpoint_markdown(checkpoints[0]))
                    source = gr.Radio(["Dataset sample", "Upload images"], value="Dataset sample", label="Input")
                    with gr.Column(visible=True) as dataset_col:
                        split = gr.Dropdown(["test", "val", "train"], value="test", label="Split")
                        index = gr.Slider(0, max(0, test_count - 1), value=0, step=1, label="Image index")
                        sample_name = gr.Markdown()
                    with gr.Column(visible=False) as upload_col:
                        up_a = gr.Image(type="numpy", label="Before (t1)", height=180)
                        up_b = gr.Image(type="numpy", label="After (t2)", height=180)
                        up_label = gr.Image(type="numpy", image_mode="L", label="Ground-truth mask (optional)",
                                            height=180)
                    region = gr.Dropdown(list(REGIONS), value="Full image", label="Region",
                                         info="crop for faster inference on CPU")
                    seed = gr.Number(0, precision=0, label="Spike-encoding seed")
                    run_btn = gr.Button("Detect changes", variant="primary")
                with gr.Column(scale=3):
                    with gr.Row():
                        img_a = gr.Image(label="Before (t1)", interactive=False)
                        img_b = gr.Image(label="After (t2)", interactive=False)
                        img_gt = gr.Image(label="Ground truth", interactive=False)
                    with gr.Row():
                        img_prob = gr.Image(label="Change probability", interactive=False)
                        img_over = gr.Image(label="Prediction overlay", interactive=False)
                    threshold = gr.Slider(0.01, 0.99, value=0.5, step=0.01, label="Decision threshold")
                    gr.Markdown(
                        "Overlay: <span style='color:rgb(0,200,70)'>■</span> correctly detected change · "
                        "<span style='color:rgb(235,50,50)'>■</span> false alarm · "
                        "<span style='color:rgb(40,130,255)'>■</span> missed change · "
                        "<span style='color:rgb(255,190,0)'>■</span> predicted change (no ground truth)"
                    )
                    with gr.Row():
                        metrics_md = gr.Markdown()
                        eff_md = gr.Markdown()
                    spikes = gr.Plot(label="Spike rate per layer")
            state = gr.State(None)

            ckpt.change(checkpoint_markdown, ckpt, ckpt_info)
            source.change(on_source, source, [dataset_col, upload_col])
            split.change(on_split, split, index)
            for trigger in (split.change, index.release, demo.load):
                trigger(preview, [split, index], [img_a, img_b, img_gt, sample_name])
            run_btn.click(
                run_detection,
                [ckpt, source, split, index, up_a, up_b, up_label, region, seed, threshold],
                [img_a, img_b, img_gt, img_prob, img_over, metrics_md, spikes, eff_md, state],
            )
            threshold.release(rethreshold, [state, threshold], [img_over, metrics_md])

        with gr.Tab("Training runs"):
            with gr.Row():
                run_dd = gr.Dropdown(runs, value=runs[-1] if runs else None, label="Run (runs/)")
                refresh = gr.Button("Refresh", scale=0)
            run_summary = gr.Markdown()
            curves = gr.Plot()
            run_dd.change(training_curves, run_dd, [curves, run_summary])
            refresh.click(lambda: gr.Dropdown(choices=list_runs()), None, run_dd)
            demo.load(training_curves, run_dd, [curves, run_summary])

        with gr.Tab("System check"):
            gr.Markdown(
                "Checks that the pipeline is wired correctly without real training: data folders, "
                "forward + backward pass of every model/encoding combination, fitting a synthetic toy "
                "batch, loading every checkpoint. Same as `python -m src.selfcheck`."
            )
            with gr.Row():
                quick = gr.Checkbox(False, label="Quick: skip the toy-batch fitting (the slowest check)")
                check_btn = gr.Button("Run system check", variant="primary", scale=0)
            check_summary = gr.Markdown()
            check_table = gr.Dataframe(headers=["status", "check", "detail", "seconds"], interactive=False,
                                       wrap=True)
            check_btn.click(system_check, quick, [check_table, check_summary])
    return demo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="temporary public gradio.live link")
    args = p.parse_args()
    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
