import torch
import torch.nn as nn

# Standard per-operation energy constants reused across SNN-efficiency literature
# (~45nm process, Horowitz-style figures): accumulate vs 32-bit multiply-accumulate.
ENERGY_PER_AC_J = 0.9e-12
ENERGY_PER_MAC_J = 4.6e-12


class SegMetricsAccumulator:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = (torch.sigmoid(logits) > self.threshold).float()
        target = target.float()
        self.tp += (pred * target).sum().item()
        self.fp += (pred * (1 - target)).sum().item()
        self.fn += ((1 - pred) * target).sum().item()
        self.tn += ((1 - pred) * (1 - target)).sum().item()

    def compute(self) -> dict[str, float]:
        eps = 1e-8
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        oa = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "overall_accuracy": oa,
        }


def summarize_spike_rates(report: dict[str, tuple[float, float]]) -> dict[str, float]:
    out = {name: spikes / max(1.0, neurons) for name, (spikes, neurons) in report.items()}
    total_spikes = sum(s for s, _ in report.values())
    total_neurons = sum(n for _, n in report.values())
    out["overall"] = total_spikes / max(1.0, total_neurons)
    return out


def _dense_macs(m: nn.Module, x: torch.Tensor, out: torch.Tensor) -> int:
    """Dense multiply-accumulates of one conv call, summed over the batch."""
    kh, kw = m.kernel_size
    if isinstance(m, nn.ConvTranspose2d):
        # every input element is scattered to out_ch * kh * kw outputs
        return x.numel() * (m.out_channels // m.groups) * kh * kw
    return out.numel() * (m.in_channels // m.groups) * kh * kw


class OpCounter:
    """Counts synaptic operations by the *input* each conv layer receives.

    Every Conv2d / ConvTranspose2d input element fans out to
    `dense_macs / input_numel` synapses. Per element:
      * 0          -> no operation (event-driven hardware skips it)
      * 1 (spike)  -> one accumulate (AC) per synapse
      * other real -> one multiply-accumulate (MAC) per synapse

    This keeps the estimate honest for layers whose input is not binary:
    the SNN's first layer when fed raw values, the decoder convs after the
    real-valued ConvTranspose upsampling, and every layer of the ANN baseline.
    `dense_macs` is what a conventional dense accelerator would execute.
    """

    def __init__(self, model: nn.Module):
        self.stats: dict[str, dict[str, float]] = {}
        self.handles = [
            m.register_forward_hook(self._make_hook(name))
            for name, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))
        ]

    def _make_hook(self, name: str):
        def hook(m, inp, out):
            x = inp[0]
            dense = _dense_macs(m, x, out)
            fanout = dense / x.numel()
            ones = (x == 1).sum().item()
            nonzero = (x != 0).sum().item()
            s = self.stats.setdefault(name, {"ac": 0.0, "mac": 0.0, "dense_macs": 0.0})
            s["ac"] += ones * fanout
            s["mac"] += (nonzero - ones) * fanout
            s["dense_macs"] += dense

        return hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()

    def summary(self, num_samples: int, time_steps: int = 1) -> dict:
        """Per-sample totals. `time_steps` normalizes `dense_macs_per_pass`
        to one time step, i.e. the MAC count of the equivalent ANN."""
        n = max(1, num_samples)
        per_layer = {name: {k: v / n for k, v in s.items()} for name, s in self.stats.items()}
        ac = sum(s["ac"] for s in per_layer.values())
        mac = sum(s["mac"] for s in per_layer.values())
        dense = sum(s["dense_macs"] for s in per_layer.values())
        dense_per_pass = dense / time_steps
        energy = ac * ENERGY_PER_AC_J + mac * ENERGY_PER_MAC_J
        ann_energy = dense_per_pass * ENERGY_PER_MAC_J
        return {
            "per_layer": per_layer,
            "ac_per_sample": ac,
            "mac_per_sample": mac,
            "dense_macs_per_pass": dense_per_pass,
            "energy_j_per_sample": energy,
            "ann_equivalent_energy_j": ann_energy,
            "energy_ratio_vs_ann": energy / ann_energy if ann_energy else float("nan"),
        }
