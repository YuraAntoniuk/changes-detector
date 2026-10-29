# SNN Change Detection

Master's thesis project: detecting changes of terrain objects in satellite imagery with spiking neural networks (SNN).
A Siamese spiking U-Net (LIF neurons, surrogate gradients, snnTorch) compares two co-registered images and outputs a per-pixel change mask. It is benchmarked against a conventional CNN of the same structure (FC-Siam-diff) on accuracy and on estimated energy.

The method, datasets, compute plan and roadmap are in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Setup

Python 3.12 with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
uv pip install -r requirements.txt --index-strategy unsafe-best-match
python check_env.py
```

Data: LEVIR-CD unpacked to `data/levir-cd/{train,val,test}/{A,B,label}/*.png`. On first use, each split is decoded once into `data/levir-cd/_cache/`.

## Is the system working?

Three checks, none of which needs a trained model:

| command | what it shows | time |
|---|---|---|
| `python -m pytest` | automated tests on a synthetic 64px dataset: encoders, metrics, op counting, dataset tiling, tiled inference, cross-OS checkpoints, and the full train → resume → evaluate → UI workflow | ~10 s |
| `python -m src.selfcheck` | on the real setup: data folders, forward + backward of every model/encoding, fitting a toy batch, loading every checkpoint | ~20 s on GPU |
| `python app.py` | the web UI below; its *System check* tab runs the same self-check | |

"Passing" means the pipeline is wired correctly, not that a model is accurate. Accuracy comes from training.

## Web UI

```bash
python app.py                 # opens on http://127.0.0.1:7860
python app.py --host 0.0.0.0  # on a server; or tunnel: ssh -L 7860:localhost:7860 <server>
```

- **Change detection:** choose a checkpoint (or an untrained model, to see that the pipeline runs) and a LEVIR-CD pair, or upload your own before/after images and an optional mask. It shows the change-probability map and an overlay that marks correct detections, false alarms and missed changes, along with precision/recall/F1/IoU, spike rate per layer and the SNN-vs-ANN energy estimate. The threshold slider re-renders without rerunning the model.
- **Training runs:** loss and validation-metric curves from `runs/` (TensorBoard logs).
- **System check:** runs `src.selfcheck` and shows the results as a table.

A full 1024×1024 scene takes about 3 s on an RTX 4050. On CPU, use the 256/512 px crop option.

## Training and evaluation

```bash
python train.py --model snn --encoding rate --run-name snn_rate          # Siamese spiking U-Net
python train.py --model baseline --run-name baseline                     # FC-Siam-diff CNN
python train.py --model snn --encoding difference --run-name snn_diff    # event-style encoding, single stream
python train.py --model snn --upsample transpose --run-name snn_tconv     # ablation: learnable (non-spiking) upsampling
python train.py --resume checkpoints/snn_rate/last.pt                    # continue an interrupted run

python evaluate.py --checkpoint checkpoints/snn_rate/best.pt --split test --per-layer
tensorboard --logdir runs
```

A checkpoint is saved every epoch (`last.pt`, plus `best.pt` by validation F1). The writes are atomic, so a killed instance never leaves a half-written file. Checkpoints load on both Windows and Linux. `--data-root` points to the dataset on another machine.

## Layout

```
app.py              web UI (Gradio)
train.py            training loop: TensorBoard logging, per-epoch checkpoints, resume
evaluate.py         test metrics, spike rates, operation counts and energy estimate
src/config.py       all hyperparameters (dataclasses)
src/data/           LEVIR-CD dataset (patch tiling, memory-mapped cache), augmentation
src/encoding/       spike encoders: rate, difference (event-style), spectral-index (needs multispectral data)
src/models/         spiking U-Nets, CNN baselines, shared building blocks
src/pipeline.py     model factory, forward pass, portable checkpoint loading
src/inference.py    tiled whole-scene inference
src/selfcheck.py    pipeline health checks
src/losses.py       BCE + Dice
src/metrics.py      segmentation metrics, synaptic-operation counter, energy constants
tests/              pytest suite on synthetic data
```

## Status and known limitations

- **First test run** (`snn` + rate coding, T=8, 128 px patches, 25 epochs, RTX 4050): test F1 0.44, IoU 0.28. That's far below typical LEVIR-CD CNN results (F1 ≈ 0.85+), and no CNN baseline has been trained here yet. Tuning the decision threshold doesn't help: even the best threshold chosen on the test set itself only reaches F1 0.445.
- **Decoder upsampling is now spike-preserving** (`--upsample nearest`, the default for both the SNN and the baseline). The first run used `ConvTranspose2d`, whose real-valued output made every decoder convolution a multiply-accumulate at each time step, so the SNN came out ≈3.7× *more* expensive than the ANN. With nearest upsampling the SNN performs no multiply-accumulates at all, and its energy ratio is ≈ 1.57 × (average input spike rate) at T=8. The first run's spike rates would give ≈0.3–0.4, but this needs confirming with a trained model. The trade-off is that the model is smaller (0.98M parameters instead of 1.33M) and fits a toy task more slowly. Old checkpoints still load with `transpose`, and `--upsample transpose` reproduces the old architecture.
- **Predictions show 128 px tile seams,** because the model only sees one patch of context at a time. Overlapping tiles or larger patches would help.
- **Spectral-index encoding** is implemented but needs multispectral data (OSCD / Sentinel-2). LEVIR-CD is RGB only.
