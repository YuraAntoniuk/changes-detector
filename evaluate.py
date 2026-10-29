import argparse

import torch
from torch.utils.data import DataLoader

from src.metrics import OpCounter, SegMetricsAccumulator, summarize_spike_rates
from src.config import DataConfig, TrainConfig
from src.data.dataset import LEVIRCDDataset
from src.pipeline import build_model, forward_pass, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-root", type=str, default=DataConfig().root)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-layer", action="store_true", help="print per-layer operation counts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    cfg: TrainConfig = ckpt["config"]
    # the saved root is the training machine's path; use this machine's data
    cfg.data.root = args.data_root
    # rate/difference encoding samples spikes at random: seed for repeatable numbers
    torch.manual_seed(args.seed)

    ds = LEVIRCDDataset(cfg.data.root, args.split, patch_size=cfg.data.patch_size, train=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    if hasattr(model, "reset_spike_stats"):
        model.reset_spike_stats()
    ops = OpCounter(model)

    metrics = SegMetricsAccumulator()
    num_samples = 0
    with torch.no_grad():
        for img1, img2, label in dl:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            logits = forward_pass(model, img1, img2, cfg)
            metrics.update(logits, label)
            num_samples += img1.shape[0]
    ops.remove()

    print(f"[{args.split}] segmentation metrics:", metrics.compute())

    is_snn = hasattr(model, "spike_rate_report")
    if is_snn:
        print("spike rates (output, per neuron per step):", summarize_spike_rates(model.spike_rate_report()))

    time_steps = cfg.encoding.time_steps if is_snn else 1
    eff = ops.summary(num_samples, time_steps=time_steps)
    p = cfg.data.patch_size
    print(f"efficiency per {p}x{p} sample:")
    print(f"  dense MACs per pass (ANN-equivalent): {eff['dense_macs_per_pass']:.3e}")
    if is_snn:
        print(f"  SNN ops over T={time_steps}: AC={eff['ac_per_sample']:.3e} MAC={eff['mac_per_sample']:.3e}")
        print(
            f"  energy: SNN={eff['energy_j_per_sample']:.3e} J  ANN-equivalent={eff['ann_equivalent_energy_j']:.3e} J"
            f"  ratio={eff['energy_ratio_vs_ann']:.3f}"
        )
    else:
        print(f"  energy (dense MACs): {eff['ann_equivalent_energy_j']:.3e} J")
    if args.per_layer:
        for name, s in eff["per_layer"].items():
            print(f"  {name:45s} AC={s['ac']:.3e} MAC={s['mac']:.3e} dense={s['dense_macs']:.3e}")


if __name__ == "__main__":
    main()
