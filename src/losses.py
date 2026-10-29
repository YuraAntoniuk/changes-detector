import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """Model-type-agnostic: expects a single real-valued logit map, which both
    the baseline CNNs and the SNN models' membrane-accumulated readout produce.
    """

    def __init__(
        self,
        pos_weight: float = 3.0,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (probs * target).sum(dim=dims)
        dice = 1 - (2 * inter + self.smooth) / (
            probs.sum(dim=dims) + target.sum(dim=dims) + self.smooth
        )
        return self.bce_weight * bce + self.dice_weight * dice.mean()
