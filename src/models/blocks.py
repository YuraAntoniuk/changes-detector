from typing import Literal

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SpikingConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        beta: float = 0.9,
        use_bn: bool = True,
        spike_grad=None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.lif = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad or surrogate.fast_sigmoid(slope=25),
            init_hidden=False,
        )
        self.register_buffer("spike_count", torch.zeros(()))
        self.register_buffer("neuron_count", torch.zeros(()))

    def init_mem(self) -> torch.Tensor:
        return self.lif.init_leaky()

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cur = self.bn(self.conv(x))
        spk, mem = self.lif(cur, mem)
        # eval-only: keeps training fast and avoids double counting when
        # gradient checkpointing recomputes a time step
        if not self.training:
            self.spike_count += spk.sum().detach()
            self.neuron_count += spk.numel()
        return spk, mem

    def reset_spike_stats(self) -> None:
        self.spike_count.zero_()
        self.neuron_count.zero_()


class EncoderStage(nn.Module):
    """One U-Net downsampling level: block followed by 2x2 maxpool.

    Works with either `ConvBlock` (baseline, no `mem`) or `SpikingConvBlock`
    (SNN, explicit `mem` threading) via the `spiking` flag.
    """

    def __init__(self, block: nn.Module, spiking: bool):
        super().__init__()
        self.block = block
        self.spiking = spiking
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, mem=None):
        if self.spiking:
            skip, mem = self.block(x, mem)
            return skip, self.pool(skip), mem
        skip = self.block(x)
        return skip, self.pool(skip), None


class DecoderStage(nn.Module):
    """One U-Net upsampling level: upsample, concat skip, then block.

    upsample="nearest" repeats each value 2x2, so a binary spike map stays
    binary; "transpose" is a learnable ConvTranspose2d with real-valued output.
    """

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        block_factory,
        spiking: bool,
        upsample: Literal["nearest", "transpose"] = "nearest",
    ):
        super().__init__()
        self.spiking = spiking
        if upsample == "nearest":
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
        elif upsample == "transpose":
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        else:
            raise ValueError(f"unknown upsample mode {upsample!r}")
        self.block = block_factory(in_ch + skip_ch, out_ch)

    def forward(self, x, skip, mem=None):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        if self.spiking:
            return self.block(x, mem)
        return self.block(x), None


def build_encoder(in_ch: int, channels: tuple[int, ...], bottleneck: int, block_factory, spiking: bool):
    stages = nn.ModuleList()
    prev = in_ch
    for ch in channels:
        stages.append(EncoderStage(block_factory(prev, ch), spiking))
        prev = ch
    bottleneck_block = block_factory(prev, bottleneck)
    return stages, bottleneck_block


def build_decoder(
    channels: tuple[int, ...],
    skip_channels: tuple[int, ...],
    decoder_in_ch: int,
    block_factory,
    spiking: bool,
    upsample: str = "nearest",
):
    stages = nn.ModuleList()
    prev = decoder_in_ch
    for ch, skip_ch in zip(reversed(channels), reversed(skip_channels)):
        stages.append(DecoderStage(prev, skip_ch, ch, block_factory, spiking, upsample))
        prev = ch
    return stages


class FeatureFusion(nn.Module):
    def __init__(self, mode: Literal["absdiff", "concat", "both"] = "absdiff"):
        super().__init__()
        self.mode = mode

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        if self.mode == "absdiff":
            return (s1 - s2).abs()
        if self.mode == "concat":
            return torch.cat([s1, s2], dim=1)
        return torch.cat([s1, s2, (s1 - s2).abs()], dim=1)

    def out_channels(self, in_channels: int) -> int:
        if self.mode == "absdiff":
            return in_channels
        if self.mode == "concat":
            return 2 * in_channels
        return 3 * in_channels
