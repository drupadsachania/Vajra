"""ATT&CK technique multi-label classifier (§6.3).

Output: 716 logits → sigmoid → multi-label probabilities.
Threshold τ=0.35 at inference.
Training: BCE + label smoothing ε=0.1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ATTACKClassifier(nn.Module):
    """Linear(d_model → 716) + sigmoid multi-label classifier."""

    N_TECHNIQUES = 716
    THRESHOLD = 0.35

    def __init__(self, d_model: int = 1024):
        super().__init__()
        self.head = nn.Linear(d_model, self.N_TECHNIQUES)

    def forward(self, fusion_cls: torch.Tensor) -> torch.Tensor:
        """
        fusion_cls : (batch, d_model)
        Returns    : (batch, 716) — sigmoid probabilities
        """
        return torch.sigmoid(self.head(fusion_cls))

    def predict(self, fusion_cls: torch.Tensor) -> torch.Tensor:
        """Returns binary multi-label predictions at threshold τ=0.35."""
        probs = self.forward(fusion_cls)
        return (probs > self.THRESHOLD).float()

    @staticmethod
    def loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
        """BCE with label smoothing ε=0.1."""
        targets_smooth = targets * (1 - eps) + 0.5 * eps
        return F.binary_cross_entropy_with_logits(logits, targets_smooth)
