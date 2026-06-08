"""Decision State Classifier (§6.4).

4 classes: 0=HIGH_CONFIDENCE_ACTION, 1=ESCALATE_FOR_REVIEW, 2=DEFER, 3=INSUFFICIENT_CONTEXT
DCAT override: if D > θ_divergence AND class ∉ {0, 1} → force class=1.
Applied at both F3 and F6 taps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionStateClassifier(nn.Module):
    """Linear(d_model → 4) decision state head.

    Called at F3 and F6 fusion taps. Applies DCAT override when triggered.
    """

    N_CLASSES = 4

    def __init__(self, d_model: int = 1024):
        super().__init__()
        self.head = nn.Linear(d_model, self.N_CLASSES)

    def forward(
        self,
        fusion_cls: torch.Tensor,
        dcat_divergence: torch.Tensor | None = None,
        theta_divergence: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        fusion_cls       : (batch, d_model)
        dcat_divergence  : (batch,) mean divergence or None
        theta_divergence : threshold for DCAT override

        Returns:
          logits         : (batch, 4)
          probs          : (batch, 4)
          decision_class : (batch,) int — class indices after optional DCAT override
        """
        logits = self.head(fusion_cls)                   # (batch, 4)
        probs  = F.softmax(logits, dim=-1)               # (batch, 4)
        decision_class = probs.argmax(dim=-1)            # (batch,)

        if dcat_divergence is not None:
            # DCAT override: high divergence + class ∉ {0,1} → force class=1
            override_mask = (
                (dcat_divergence > theta_divergence) &
                (decision_class != 0) &
                (decision_class != 1)
            )
            decision_class = torch.where(override_mask, torch.ones_like(decision_class), decision_class)

        return logits, probs, decision_class
