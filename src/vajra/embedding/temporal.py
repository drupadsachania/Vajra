"""ContiFormer continuous-time temporal encoding (§3.3).

128 learnable frequency pairs ω₁…ω₁₂₈, initialized log-uniform over
[10⁻⁷, 10³].  φ(t) ∈ ℝ²⁵⁶ → Linear(256→d_model, GELU) → GRN → d_model.
Null timestamps resolve to a single learned fallback vector.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class GatedResidualNetwork(nn.Module):
    """Lightweight GRN: Linear + ELU gate on top of residual."""

    def __init__(self, d: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.elu(self.fc1(x))
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(x))
        return self.ln(x + g * h)


class ContiFormerEncoding(nn.Module):
    """Continuous-time positional encoding via learnable frequency pairs.

    For a timestamp t (seconds since Unix epoch, or relative offset):
        φ(t) = [cos(ω₁t), sin(ω₁t), …, cos(ω₁₂₈t), sin(ω₁₂₈t)] ∈ ℝ²⁵⁶

    A Linear(256→d_model) + GELU followed by a GRN produces the final
    d_model-dimensional temporal embedding.

    When t is None (null timestamp), a learned null_vector is returned instead.
    """

    FREQ_PAIRS = 128
    PHI_DIM = 256  # 2 * FREQ_PAIRS

    def __init__(self, d_model: int = 1024):
        super().__init__()
        # Log-uniform initialization over [10⁻⁷, 10³]
        log_min, log_max = math.log(1e-7), math.log(1e3)
        freqs = torch.exp(torch.linspace(log_min, log_max, self.FREQ_PAIRS))
        self.log_freqs = nn.Parameter(torch.log(freqs))  # learnable in log-space

        self.proj = nn.Sequential(
            nn.Linear(self.PHI_DIM, d_model),
            nn.GELU(),
        )
        self.grn = GatedResidualNetwork(d_model)
        self.null_vector = nn.Parameter(torch.zeros(d_model))

    @property
    def freqs(self) -> torch.Tensor:
        return torch.exp(self.log_freqs)

    def phi(self, t: torch.Tensor) -> torch.Tensor:
        """t : (batch, seq) → φ(t) : (batch, seq, 256).

        Computation done in float64 to avoid catastrophic precision loss:
        Unix-epoch seconds (~1.7e9) × ω up to 1e3 ≈ 1e12, which exceeds
        float32's ~7 significant digits, making cos/sin pure noise otherwise.
        """
        w = self.freqs.double()  # (128,)
        wt = t.double().unsqueeze(-1) * w  # (batch, seq, 128)
        out = torch.cat([wt.cos(), wt.sin()], dim=-1)  # (batch, seq, 256)
        return out.to(t.dtype if t.is_floating_point() else torch.float32)

    def forward(
        self,
        timestamps: torch.Tensor | None,
        batch: int,
        seq: int,
    ) -> torch.Tensor:
        """
        timestamps : (batch, seq) float or None
        Returns    : (batch, seq, d_model)
        """
        if timestamps is None:
            # Null path: broadcast learned null vector
            return self.null_vector.expand(batch, seq, -1)

        phi = self.phi(timestamps)  # (batch, seq, 256)
        h = self.proj(phi)  # (batch, seq, d_model)
        return self.grn(h)
