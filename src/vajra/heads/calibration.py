"""Calibration: temperature scaling + conformal prediction sets (§6.4).

TemperatureScaler: one learned T per domain; applied post-training.
ConformalPredictor: threshold calibration at 90% and 95% coverage on OOD set.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """Per-domain temperature scaling for calibration.

    T > 1 softens the distribution; T < 1 sharpens it.
    """

    def __init__(self, n_domains: int = 7, n_classes: int = 4):
        super().__init__()
        self.temperatures = nn.ParameterDict({
            f"domain_{i}": nn.Parameter(torch.ones(1)) for i in range(n_domains)
        })
        self.n_classes = n_classes

    def scale(self, logits: torch.Tensor, domain_idx: int) -> torch.Tensor:
        """Scale logits by T for a given domain, return calibrated probabilities."""
        T = self.temperatures[f"domain_{domain_idx}"].clamp(min=0.01)
        return F.softmax(logits / T, dim=-1)

    def forward(self, logits: torch.Tensor, domain_idx: int = 0) -> torch.Tensor:
        return self.scale(logits, domain_idx)


class ConformalPredictor:
    """Conformal prediction set constructor.

    Calibrated on held-out OOD set to achieve target coverage.
    At inference: returns smallest set of classes whose cumulative softmax
    probability exceeds the calibrated threshold.
    """

    def __init__(self, coverage_targets: list[float] | None = None):
        self.coverage_targets = coverage_targets or [0.90, 0.95]
        self.thresholds: dict[float, float] = {c: 0.0 for c in self.coverage_targets}

    def calibrate(self, softmax_scores: torch.Tensor, true_labels: torch.Tensor):
        """Fit thresholds from calibration set using split-conformal quantile.

        softmax_scores : (N, C) probabilities on OOD calibration set
        true_labels    : (N,) long tensor of true class indices
        """
        N = softmax_scores.shape[0]
        true_probs = softmax_scores[torch.arange(N), true_labels]  # (N,) conformity scores
        sorted_scores, _ = true_probs.sort()

        for coverage in self.coverage_targets:
            # Finite-sample correction: ⌈(N+1)(1−coverage)⌉ − 1
            idx = min(N - 1, max(0, math.ceil((N + 1) * (1 - coverage)) - 1))
            self.thresholds[coverage] = sorted_scores[idx].item()

    def predict_set(
        self, probs: torch.Tensor, coverage: float = 0.90
    ) -> list[list[int]]:
        """Return prediction sets for a batch.

        The set is {i : p_i >= threshold} — all classes whose probability
        meets or exceeds the calibrated threshold (split-conformal guarantee).

        probs : (batch, C)
        Returns list of class-index lists, one per example.
        """
        threshold = self.thresholds.get(coverage, 0.0)
        result = []
        for p in probs:
            idx = (p >= threshold).nonzero(as_tuple=True)[0].tolist()
            result.append(idx if idx else [int(p.argmax())])
        return result
