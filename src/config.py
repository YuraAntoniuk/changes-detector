from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"

LEVIR_CD_DIR = DATA_DIR / "levir-cd"


@dataclass
class DataConfig:
    # str, not Path: the config is pickled into every checkpoint, and a pickled
    # WindowsPath/PosixPath cannot be unpickled on the other OS
    root: str = str(LEVIR_CD_DIR)
    # 128 keeps a 4-level spiking Siamese U-Net (T=8) within ~6GB VRAM;
    # bump to 256 only on a bigger GPU (see PROJECT_PLAN.md section 3).
    patch_size: int = 128
    # random training crops drawn from each 1024px image per epoch
    # (16 x 128^2 = 1/4 of the scene area per epoch)
    crops_per_image: int = 16
    num_workers: int = 2


@dataclass
class EncodingConfig:
    # "rate" | "difference" | "spectral" (spectral needs multispectral bands, not available for LEVIR-CD)
    kind: str = "rate"
    time_steps: int = 8
    # threshold on normalized (0-1) absolute pixel difference, used by "difference" encoding
    diff_threshold: float = 0.1


@dataclass
class ModelConfig:
    channels: tuple[int, ...] = (16, 32, 64, 128)
    bottleneck: int = 256
    beta: float = 0.9  # LIF membrane decay (SNN models only)
    fusion_mode: str = "absdiff"  # "absdiff" | "concat" | "both" -- see FeatureFusion
    # decoder upsampling: "nearest" keeps spike maps binary, so every SNN decoder
    # conv is driven by spikes (accumulates only); "transpose" (learnable
    # ConvTranspose2d) emits real values that cost a full MAC at every time step
    upsample: str = "nearest"


@dataclass
class TrainConfig:
    # "baseline" | "snn"
    model: str = "snn"
    batch_size: int = 4
    epochs: int = 30
    lr: float = 1e-3
    min_lr: float = 1e-5  # cosine annealing floor
    change_class_weight: float = 3.0
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    seed: int = 0
    run_name: str = ""
    max_steps_per_epoch: int = 0  # 0 = full epoch
    # per-time-step gradient checkpointing escape hatch if VRAM is tight; see PROJECT_PLAN.md section 3
    use_checkpointing: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
