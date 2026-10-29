# SNN Change Detection — Project Plan

Master's thesis: **a method for detecting changes of terrain objects in satellite imagery using spiking neural networks (SNN).**

This document collects the overall method concept, data and compute resources, local environment setup (Windows), project structure, and practical nuances.

---

## Contents

1. [Method overview](#1-method-overview)
2. [Datasets](#2-datasets)
3. [Compute resources](#3-compute-resources)
4. [Environment setup (Windows)](#4-environment-setup-windows)
5. [Project structure](#5-project-structure)
6. [Coding conventions and Windows nuances](#6-coding-conventions-and-windows-nuances)
7. [Reproducing the environment on a rented server](#7-reproducing-the-environment-on-a-rented-server)
8. [Roadmap](#8-roadmap)

---

## 1. Method overview

The method is a pipeline of stages. Each stage maps to a subsection of the thesis and is a place to show an original contribution.

### 1.1 Input data

A pair (or time series) of images of the same area taken at different times (t₁, t₂).

- Images must be **co-registered** (pixel-aligned).
- Images must be **radiometrically normalized**, so that differences in illumination or atmosphere are not detected as changes.
- Benchmark datasets are already preprocessed; own imagery has to be processed manually.

### 1.2 Spike encoding (key stage, main source of novelty)

| Encoding | Idea | Notes |
|---|---|---|
| **Rate coding** | Pixel value → spike probability at each time step | Simple, stable baseline |
| **Difference (event-based) coding** | Spike is emitted where the value changed between t₁ and t₂ above a threshold (emulates an event camera) | "React only to changes" is built into the encoding — strong candidate for the author's contribution |
| **Spectral-index coding** | Encode NDVI (vegetation), NDWI (water), NDBI (built-up) instead of raw bands | Physically meaningful features, links the method to the GIS domain |

### 1.3 Architecture

- **Siamese network**: two branches with shared weights process t₁ and t₂; features are compared; a **U-Net-like decoder** produces a per-pixel change mask.
- All nonlinearities are **LIF (Leaky Integrate-and-Fire)** neurons.
- **Hybrid option**: first layer is a conventional convolutional encoder, the rest are spiking layers. Often improves accuracy and is easy to justify.

### 1.4 Training

- **Surrogate gradients + BPTT** over T time steps.
- Loss: **BCE + Dice** (binary segmentation).
- Strong class imbalance ("no change" pixels dominate) → weight the "change" class.

### 1.5 Evaluation

- **Accuracy**: F1, IoU, Precision / Recall for the "change" class, Overall Accuracy.
- **Efficiency** (mandatory for SNN work):
  - synaptic operations (SOPs) vs. MAC operations of an equivalent ANN;
  - spike rate per layer;
  - estimated energy consumption on neuromorphic hardware.
- **Baselines**:
  - conventional Siamese CNN of the same structure (FC-Siam-diff or similar);
  - preferably also an ANN→SNN converted model.

### 1.6 GIS post-processing

1. Save the change mask as a georeferenced **GeoTIFF**.
2. Vectorize into polygons.
3. Compute changed areas and statistics.
4. Visualize in **QGIS**.

This step turns the "model" into a "method for detecting changes of terrain objects", as stated in the thesis topic.

---

## 2. Datasets

### Primary

- **OSCD (Onera Satellite Change Detection)** — Sentinel-2, 13 spectral bands, urban changes. Closest match to "satellite imagery". Small (~24 image pairs) → requires patch extraction and augmentation.

### Secondary (second experiment / generalization check)

- **LEVIR-CD** — 637 pairs, 0.5 m RGB, building changes. Large, well-annotated, the most popular benchmark.
- **S2Looking** — off-nadir satellite images, harder.
- **WHU-CD**, **SYSU-CD** — additional options; SYSU-CD includes various change types (vegetation, roads).
- **SECOND** — semantic change detection (what changed into what), if the scope is extended.

> Check licenses up front — most are permitted for academic use.

### Own imagery (demonstration only, no labels)

- **Copernicus Data Space** — Sentinel-1/2, free.
- **Google Earth Engine**.

Useful for demonstrating the method on a region of interest (e.g. Ukraine), not for training.

---

## 3. Compute resources

### Local machine

Laptop with **RTX 4050 Laptop (6 GB VRAM)** — prototyping, debugging, small runs with small patches.

Main SNN limitation is memory: BPTT stores the network state for every time step, so memory grows roughly linearly with T.

### Rented GPU (for full training runs)

Recommended card: **RTX 4090 (24 GB)**. A100 80 GB is most likely unnecessary.

| Provider | RTX 4090 price (Oct 2026, approx.) | Notes |
|---|---|---|
| Vast.ai | from ~$0.20/h, median ~$0.48/h | Marketplace, host-set prices, reliability varies |
| RunPod | ~$0.34/h Community, ~$0.74/h Secure | Predictable, per-second billing |

**Estimated budget**: 50–150 GPU-hours for all experiments (hyperparameter search, encoding variants, baselines, ablations) → roughly **$20–80** on an RTX 4090. Prices change — verify before renting.

### Rules for rented servers

- Debug locally; start the server only for full runs.
- Save a checkpoint **every epoch** — cheap instances can be interrupted.
- Keep datasets and checkpoints on persistent storage (network volume) or sync them back regularly; don't pay for idle GPU time.
- Log experiments with **TensorBoard** or **Weights & Biases** — the plots are needed for the thesis text.

---

## 4. Environment setup (Windows)

Development is done on Windows (no WSL). Thesis text is written on Windows as well.

**Stack**: NVIDIA driver · Git · uv · Python 3.12 · PyTorch (CUDA) · snnTorch · rasterio · geopandas · PyCharm · QGIS.

**Why not Anaconda**: `rasterio` / `geopandas` / `pyproj` / `shapely` now ship pip wheels with GDAL bundled; PyTorch no longer publishes conda packages; mixing conda and pip causes conflicts. `uv` is fast, manages Python versions itself, and the environment is reproducible from a single file.

### 4.1 NVIDIA driver

Install the latest driver (nvidia.com or NVIDIA App), then verify:

```powershell
nvidia-smi
```

Do **not** install the CUDA Toolkit separately — PyTorch wheels bundle the required CUDA libraries.

### 4.2 Git

Install from <https://git-scm.com> with defaults, then:

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global core.autocrlf true
```

### 4.3 uv

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart PowerShell. If venv activation is blocked, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 4.4 Project and virtual environment

Use a short path, Latin characters only, no spaces (GDAL and some other libraries break on non-Latin paths):

```powershell
mkdir C:\projects\snn-cd
cd C:\projects\snn-cd
git init
uv venv --python 3.12
.venv\Scripts\activate
```

### 4.5 PyTorch with CUDA

Open <https://pytorch.org/get-started/locally/>, select **Windows → Pip → Python → latest CUDA**, and run the suggested command through uv:

```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu12X
```

Replace `cu12X` with the value from the selector.

### 4.6 Project libraries

```powershell
uv pip install snntorch rasterio geopandas shapely pyproj numpy matplotlib scikit-learn tqdm tensorboard jupyter ipykernel
```

SpikingJelly (optional, later — the PyPI release lags behind the repository):

```powershell
uv pip install git+https://github.com/fangwei123456/spikingjelly.git
```

### 4.7 Environment check

`check_env.py`:

```python
import torch, snntorch, rasterio, geopandas

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000, device="cuda")
    print("GPU test:", (x @ x).sum().item() != 0)
print("snnTorch:", snntorch.__version__)
print("rasterio:", rasterio.__version__, "| GDAL:", rasterio.__gdal_version__)
print("geopandas:", geopandas.__version__)
```

```powershell
python check_env.py
```

Expected: `CUDA available: True` and the GPU name.

### 4.8 PyCharm

1. Install via **JetBrains Toolbox**; activate the free **JetBrains student license**.
2. **File → Open** → `C:\projects\snn-cd`.
3. **Settings → Project → Python Interpreter → Add Interpreter → Add Local Interpreter → Select existing** → `C:\projects\snn-cd\.venv\Scripts\python.exe`.
4. Run `check_env.py` from PyCharm to confirm the interpreter.
5. `.ipynb` notebooks open directly in PyCharm with the same interpreter.

> Install packages with `uv pip install ...` in the PyCharm terminal, not via PyCharm's package manager, so uv stays in control of the environment.

### 4.9 QGIS

Install the **LTR** version from <https://qgis.org>. QGIS has its own bundled Python and is **not** connected to `.venv` — it is only used to open and style results (GeoTIFF, GeoPackage) and to produce maps for the thesis.

---

## 5. Project structure

```
snn-cd/
├── data/            # datasets (not in git)
├── checkpoints/     # model weights (not in git)
├── runs/            # TensorBoard logs (not in git)
├── notebooks/       # exploratory experiments
├── src/             # dataset loading, encoding, models, training, evaluation
├── results/         # figures and tables for the thesis
├── check_env.py
├── requirements.txt
└── PROJECT_PLAN.md
```

`.gitignore`:

```
.venv/
data/
checkpoints/
runs/
__pycache__/
.idea/
```

---

## 6. Coding conventions and Windows nuances

- **Always guard the entry point** — otherwise `DataLoader` with `num_workers > 0` crashes on Windows:

  ```python
  if __name__ == "__main__":
      main()
  ```

- **Use `pathlib` for paths**, never hardcoded `\` or `/` strings, so the same code runs on the Linux server:

  ```python
  from pathlib import Path
  DATA_DIR = Path("data") / "oscd"
  ```

- DataLoader workers start slower on Windows than on Linux — keep `num_workers` moderate locally.
- `torch.compile` may be limited on Windows; not required for this project.
- Never commit datasets, checkpoints, or logs.

---

## 7. Reproducing the environment on a rented server

Freeze the environment locally after changes. `uv pip freeze` drops two things that a Linux install needs, so the first two lines and the Windows-only marker are re-added:

```powershell
$freeze = uv pip freeze
@("# torch/torchvision below are CUDA 13.2 builds from this index; see PROJECT_PLAN.md section 7",
  "--extra-index-url https://download.pytorch.org/whl/cu132") +
  ($freeze -replace '^(pywinpty==.*)$', "`$1 ; sys_platform == 'win32'") | Set-Content -Encoding utf8 requirements.txt
git add .
git commit -m "Update environment"
git push
```

On the server (Linux):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
git clone <repo-url> snn-cd && cd snn-cd
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt --index-strategy unsafe-best-match
python check_env.py
python -m src.selfcheck   # data, every model/encoding, toy-batch fitting, checkpoints
python -m pytest          # automated tests on synthetic data, ~10 s
```

> `requirements.txt` pins the CUDA 13.2 build of PyTorch and names its index on the `--extra-index-url` line. `--index-strategy unsafe-best-match` is needed because uv otherwise takes every package only from the first index that has it, and the PyTorch index also mirrors packages such as `certifi` at older versions. If the server's driver needs a different CUDA build, reinstall `torch`/`torchvision` with the matching `--index-url` before installing the rest.

---

## 8. Roadmap

- [ ] Set up the local environment (section 4) and pass `check_env.py`
- [ ] Create a private Git repository and push the initial structure
- [ ] Download OSCD into `data/`, inspect structure and size; optionally LEVIR-CD
- [ ] Literature review:
  - change detection baselines: FC-Siam-diff, BIT, ChangeFormer
  - SNN for segmentation and remote sensing (Google Scholar: "spiking neural network change detection", "spiking remote sensing")
- [ ] Agree with the supervisor on the original contribution (candidate: difference- or index-based spike encoding + energy-efficiency analysis)
- [ ] Data pipeline: reading, normalization, patch extraction, augmentation
- [ ] Spike encoders: rate, difference, spectral-index
- [ ] Baseline Siamese CNN (FC-Siam-diff)
- [ ] Spiking Siamese U-Net (LIF, surrogate gradients)
- [ ] Training and evaluation: F1, IoU, Precision/Recall, OA, SOPs, spike rate, energy estimate
- [ ] Ablations: encoding type, number of time steps T, hybrid vs. fully spiking
- [ ] Optional: ANN→SNN conversion baseline
- [ ] GIS post-processing: GeoTIFF export, vectorization, area statistics, QGIS maps
- [ ] Demonstration on own Sentinel-2 imagery of a region of interest
- [ ] Thesis text, figures, and tables
