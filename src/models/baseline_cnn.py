import torch
import torch.nn as nn

from src.models.blocks import ConvBlock, FeatureFusion, build_decoder, build_encoder


class FCSiamDiff(nn.Module):
    """Daudt et al. 2018 FC-Siam-diff: shared-weight two-branch encoder,
    skip connections fused by absolute difference, U-Net decoder.

    Deviates from the original paper's SegNet-style index-based unpooling:
    uses plain maxpool and the same `upsample` mode as the SNN, so
    down/up-sampling mechanics match the SNN counterpart for a controlled
    comparison.
    """

    def __init__(
        self,
        in_ch: int = 3,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        bottleneck: int = 256,
        use_bn: bool = True,
        fusion_mode: str = "absdiff",
        upsample: str = "nearest",
    ):
        super().__init__()
        block_factory = lambda i, o: ConvBlock(i, o, use_bn=use_bn)
        self.encoder_stages, self.bottleneck_block = build_encoder(
            in_ch, channels, bottleneck, block_factory, spiking=False
        )
        self.fusion = FeatureFusion(fusion_mode)
        skip_channels = tuple(self.fusion.out_channels(c) for c in channels)
        decoder_in_ch = self.fusion.out_channels(bottleneck)
        self.decoder_stages = build_decoder(
            channels, skip_channels, decoder_in_ch, block_factory, spiking=False, upsample=upsample
        )
        self.head = nn.Conv2d(channels[0], 1, 1)

    def _encode(self, x: torch.Tensor):
        skips = []
        for stage in self.encoder_stages:
            skip, x, _ = stage(x)
            skips.append(skip)
        return skips, self.bottleneck_block(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        skips1, bott1 = self._encode(x1)
        skips2, bott2 = self._encode(x2)

        x = self.fusion(bott1, bott2)
        for stage, s1, s2 in zip(self.decoder_stages, reversed(skips1), reversed(skips2)):
            skip = self.fusion(s1, s2)
            x, _ = stage(x, skip)
        return self.head(x)


class FCEF(nn.Module):
    """Early-fusion single-stream baseline: concat(img_t1, img_t2) -> one
    encoder-decoder pass. Non-Siamese counterpart used for the
    difference-encoding ablation, parallel to `SpikingUNet`.
    """

    def __init__(
        self,
        in_ch: int = 3,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        bottleneck: int = 256,
        use_bn: bool = True,
        upsample: str = "nearest",
    ):
        super().__init__()
        block_factory = lambda i, o: ConvBlock(i, o, use_bn=use_bn)
        self.encoder_stages, self.bottleneck_block = build_encoder(
            2 * in_ch, channels, bottleneck, block_factory, spiking=False
        )
        self.decoder_stages = build_decoder(
            channels, channels, bottleneck, block_factory, spiking=False, upsample=upsample
        )
        self.head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x1, x2], dim=1)
        skips = []
        for stage in self.encoder_stages:
            skip, x, _ = stage(x)
            skips.append(skip)
        x = self.bottleneck_block(x)
        for stage, skip in zip(self.decoder_stages, reversed(skips)):
            x, _ = stage(x, skip)
        return self.head(x)
