"""Dual-Contrastive Attention Transform (DCAT) — §4.3.

Dual stream: observed X and baseline B.
KL divergence per-head, per-position drives the contrastive signal.
Baseline EMA cache: update when decision_class ∈ {0,2}; freeze when ∈ {1,3}.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import RotaryEmbedding, apply_rotary_pos_emb, SwiGLUFFN


class BaselineCache:
    """Per-entity EMA store for the DCAT baseline.

    β=0.9 as specified. Frozen on decision_classes {1, 3} (ESCALATE / INSUFFICIENT).
    Updated on decision_classes {0, 2} (HIGH_CONFIDENCE_ACTION / DEFER).
    """

    BETA = 0.9
    UPDATE_CLASSES = frozenset({0, 2})
    FREEZE_CLASSES = frozenset({1, 3})

    def __init__(self, d_model: int, max_entities: int = 4096):
        self.d_model = d_model
        self._cache: torch.Tensor | None = None

    def update(self, new_state: torch.Tensor, decision_class: int | None = None) -> torch.Tensor:
        """EMA-update baseline, unless frozen by decision_class."""
        if decision_class in self.FREEZE_CLASSES:
            # Frozen: return existing baseline or initialize to new_state
            if self._cache is None:
                self._cache = new_state.detach().clone()
            return self._cache

        if self._cache is None or self._cache.shape != new_state.shape:
            self._cache = new_state.detach().clone()
        else:
            self._cache = (
                self.BETA * self._cache + (1 - self.BETA) * new_state.detach()
            )
        return self._cache

    def get(self, fallback: torch.Tensor) -> torch.Tensor:
        if self._cache is None:
            return fallback
        if self._cache.shape != fallback.shape:
            return fallback
        return self._cache


def _stable_kl_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL(p || q) = sum p * log(p/q), numerically stable.

    p, q: (..., S) attention distributions (post-softmax).
    Returns (..., ) scalar per (batch*head, pos).
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


class DCATBlock(nn.Module):
    """Dual-Contrastive Attention block (§4.3).

    Two independent attention streams share the same Q/K/V projections but
    are applied to observed (x) and baseline (b) inputs separately.

    Output: LayerNorm(x + MHA_obs + alpha * MHA_base)
    D_divergence: KL(A_baseline || A_observed) per head, per position.
    """

    ALPHA = 0.3

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_width: int,
        alpha: float = 0.3,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.alpha = alpha

        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)

        # Shared QKV projections: both streams use the same weights.
        # Contrastive signal comes from different inputs (observed X vs baseline B).
        # This ensures KL(A_base || A_obs) = 0 iff X produces same attn patterns as B.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_obs  = nn.Linear(d_model, d_model, bias=False)
        self.o_base = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim)
        self.ffn  = SwiGLUFFN(d_model, ffn_width)
        self.dropout = nn.Dropout(dropout)
        self.final_ln = nn.LayerNorm(d_model)

        self.baseline_cache = BaselineCache(d_model)

    def _single_stream(
        self,
        x: torch.Tensor,
        o_proj: nn.Linear,
        ln: nn.LayerNorm,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one attention stream using shared Q/K/V projections.

        Returns (output, attn_weights_pre_dropout).
        """
        B, S, D = x.shape
        x_ln = ln(x)
        q = self.q_proj(x_ln).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_ln).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_ln).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(S, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scale = math.sqrt(self.head_dim)
        w = torch.matmul(q, k.transpose(-2, -1)) / scale   # (B, H, S, S)
        w_soft = F.softmax(w, dim=-1)
        w_drop = self.dropout(w_soft)

        out = torch.matmul(w_drop, v).transpose(1, 2).contiguous().view(B, S, D)
        return o_proj(out), w_soft

    def forward(
        self,
        x: torch.Tensor,
        decision_class: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x              : (batch, seq, d_model)  — observed stream
        decision_class : int or None — controls baseline cache freeze/update

        Returns:
          output      : (batch, seq, d_model)
          D_divergence: (batch, n_heads, seq) KL(A_base || A_obs) per position
        """
        baseline = self.baseline_cache.get(fallback=x)

        # Both streams use the same shared QKV weights applied to different inputs.
        attn_obs,  w_obs  = self._single_stream(x,        self.o_obs,  self.ln1)
        attn_base, w_base = self._single_stream(baseline, self.o_base, self.ln1)

        self.baseline_cache.update(x, decision_class)

        # KL(A_base || A_obs): zero iff X produces the same attention patterns as baseline
        D_divergence = _stable_kl_div(w_base, w_obs)   # (B, H, S)

        x_combined = x + self.dropout(attn_obs) + self.alpha * self.dropout(attn_base)
        output = self.final_ln(x_combined + self.dropout(self.ffn(self.ln2(x_combined))))

        return output, D_divergence
