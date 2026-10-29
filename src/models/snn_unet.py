import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.models.blocks import FeatureFusion, SpikingConvBlock, build_decoder, build_encoder


def _spike_rate_report(model: nn.Module) -> dict[str, tuple[float, float]]:
    return {
        name: (m.spike_count.item(), m.neuron_count.item())
        for name, m in model.named_modules()
        if isinstance(m, SpikingConvBlock)
    }


def _reset_spike_stats(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, SpikingConvBlock):
            m.reset_spike_stats()


def _run_step(module: nn.Module, step_fn, *args):
    """Per-time-step gradient checkpointing: BPTT activations of a step are
    recomputed during backward instead of stored, so VRAM no longer grows
    with T (at roughly +30% compute). Note: recomputation updates BatchNorm
    running statistics a second time for that step.
    """
    if module.use_checkpointing and module.training and torch.is_grad_enabled():
        return checkpoint(step_fn, *args, use_reentrant=False)
    return step_fn(*args)


class SiameseSpikingUNet(nn.Module):
    """Shared-weight two-branch spiking encoder, feature fusion at every
    skip level + bottleneck, spiking decoder. Non-spiking (real-valued) head
    applied every time step, accumulated and averaged over T -- see
    PROJECT_PLAN.md / plan for the readout-convention rationale.
    """

    def __init__(
        self,
        in_ch: int = 3,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        bottleneck: int = 256,
        beta: float = 0.9,
        use_bn: bool = True,
        fusion_mode: str = "absdiff",
        upsample: str = "nearest",
        use_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        block_factory = lambda i, o: SpikingConvBlock(i, o, beta=beta, use_bn=use_bn)
        self.encoder_stages, self.bottleneck_block = build_encoder(
            in_ch, channels, bottleneck, block_factory, spiking=True
        )
        self.fusion = FeatureFusion(fusion_mode)
        skip_channels = tuple(self.fusion.out_channels(c) for c in channels)
        decoder_in_ch = self.fusion.out_channels(bottleneck)
        self.decoder_stages = build_decoder(
            channels, skip_channels, decoder_in_ch, block_factory, spiking=True, upsample=upsample
        )
        self.head = nn.Conv2d(channels[0], 1, 1)

    def _init_branch_mems(self):
        enc_mems = [stage.block.init_mem() for stage in self.encoder_stages]
        bott_mem = self.bottleneck_block.init_mem()
        return enc_mems, bott_mem

    def _encode(self, x, enc_mems, bott_mem):
        skips = []
        for i, stage in enumerate(self.encoder_stages):
            skip, x, enc_mems[i] = stage(x, enc_mems[i])
            skips.append(skip)
        bott, bott_mem = self.bottleneck_block(x, bott_mem)
        return skips, bott, bott_mem

    def _step(self, x1, x2, enc_mems1, bott_mem1, enc_mems2, bott_mem2, dec_mems):
        # copy the lists: a checkpoint recompute must see the original states
        enc_mems1, enc_mems2, dec_mems = list(enc_mems1), list(enc_mems2), list(dec_mems)
        skips1, bott1, bott_mem1 = self._encode(x1, enc_mems1, bott_mem1)
        skips2, bott2, bott_mem2 = self._encode(x2, enc_mems2, bott_mem2)

        x = self.fusion(bott1, bott2)
        for i, (stage, sk1, sk2) in enumerate(
            zip(self.decoder_stages, reversed(skips1), reversed(skips2))
        ):
            skip = self.fusion(sk1, sk2)
            x, dec_mems[i] = stage(x, skip, dec_mems[i])
        return self.head(x), (enc_mems1, bott_mem1, enc_mems2, bott_mem2, dec_mems)

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        """s1, s2: [T,B,C,H,W] spike trains -> [B,1,H,W] logits."""
        time_steps = s1.shape[0]
        enc_mems1, bott_mem1 = self._init_branch_mems()
        enc_mems2, bott_mem2 = self._init_branch_mems()
        dec_mems = [stage.block.init_mem() for stage in self.decoder_stages]
        state = (enc_mems1, bott_mem1, enc_mems2, bott_mem2, dec_mems)

        acc_logits = 0.0
        for t in range(time_steps):
            logits, state = _run_step(self, self._step, s1[t], s2[t], *state)
            acc_logits = acc_logits + logits
        return acc_logits / time_steps

    def reset_spike_stats(self) -> None:
        _reset_spike_stats(self)

    def spike_rate_report(self) -> dict[str, tuple[float, float]]:
        return _spike_rate_report(self)


class SpikingUNet(nn.Module):
    """Single-stream spiking U-Net for difference/event-coded input: one
    encoder/decoder pass per time step, no fusion (the "comparison" already
    happened in the difference encoding stage).
    """

    def __init__(
        self,
        in_ch: int = 1,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        bottleneck: int = 256,
        beta: float = 0.9,
        use_bn: bool = True,
        upsample: str = "nearest",
        use_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        block_factory = lambda i, o: SpikingConvBlock(i, o, beta=beta, use_bn=use_bn)
        self.encoder_stages, self.bottleneck_block = build_encoder(
            in_ch, channels, bottleneck, block_factory, spiking=True
        )
        self.decoder_stages = build_decoder(
            channels, channels, bottleneck, block_factory, spiking=True, upsample=upsample
        )
        self.head = nn.Conv2d(channels[0], 1, 1)

    def _step(self, x, enc_mems, bott_mem, dec_mems):
        enc_mems, dec_mems = list(enc_mems), list(dec_mems)
        skips = []
        for i, stage in enumerate(self.encoder_stages):
            skip, x, enc_mems[i] = stage(x, enc_mems[i])
            skips.append(skip)
        x, bott_mem = self.bottleneck_block(x, bott_mem)
        for i, (stage, skip) in enumerate(zip(self.decoder_stages, reversed(skips))):
            x, dec_mems[i] = stage(x, skip, dec_mems[i])
        return self.head(x), (enc_mems, bott_mem, dec_mems)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s: [T,B,1,H,W] spike train -> [B,1,H,W] logits."""
        time_steps = s.shape[0]
        enc_mems = [stage.block.init_mem() for stage in self.encoder_stages]
        bott_mem = self.bottleneck_block.init_mem()
        dec_mems = [stage.block.init_mem() for stage in self.decoder_stages]
        state = (enc_mems, bott_mem, dec_mems)

        acc_logits = 0.0
        for t in range(time_steps):
            logits, state = _run_step(self, self._step, s[t], *state)
            acc_logits = acc_logits + logits
        return acc_logits / time_steps

    def reset_spike_stats(self) -> None:
        _reset_spike_stats(self)

    def spike_rate_report(self) -> dict[str, tuple[float, float]]:
        return _spike_rate_report(self)
