"""Pre-LayerNorm transformer block with SwiGLU FFN and RoPE (§4.1)."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RoPE ─────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """Standard RoPE positional encoding, θ=10000."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cos, sin) each of shape (seq_len, dim//2)."""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)       # (seq, dim/2)
        return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    q, k : (batch, heads, seq, head_dim)
    cos, sin : (seq, head_dim//2) — broadcast over batch/heads
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, d/2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    cos = torch.cat([cos, cos], dim=-1)  # (1, 1, seq, d)
    sin = torch.cat([sin, sin], dim=-1)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


# ── SwiGLU FFN ────────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: gate × SiLU(gate_proj(x)) · up_proj(x) → out_proj.

    Intermediate width = ffn_width × 2/3, rounded up to multiple of 64.
    """

    def __init__(self, d_model: int, ffn_width: int):
        super().__init__()
        inter = int(ffn_width * 2 / 3)
        inter = ((inter + 63) // 64) * 64  # round up to 64
        self.gate_proj = nn.Linear(d_model, inter, bias=False)
        self.up_proj   = nn.Linear(d_model, inter, bias=False)
        self.out_proj  = nn.Linear(inter, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Pre-LN Transformer Block ──────────────────────────────────────────────────

class PreLNTransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block (§4.1).

    Architecture: x + Attn(LN(x)) + FFN(LN(x + Attn(LN(x))))
    Attention: standard multi-head self-attention with RoPE.
    FFN: SwiGLU.
    Dropout: p=0.1 during training, 0 at inference.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_width: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim)
        self.ffn = SwiGLUFFN(d_model, ffn_width)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x         : (batch, seq, d_model)
        attn_mask : (batch, 1, seq, seq) or None — additive mask (0 / -inf)
        Returns   : (batch, seq, d_model)
        """
        B, S, D = x.shape

        # Self-attention (Pre-LN)
        residual = x
        x_ln = self.ln1(x)
        q = self.q_proj(x_ln).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_ln).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_ln).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(S, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scale = math.sqrt(self.head_dim)
        attn_w = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, S, S)
        if attn_mask is not None:
            attn_w = attn_w + attn_mask
        attn_w = F.softmax(attn_w, dim=-1)
        attn_w = self.dropout(attn_w)

        attn_out = torch.matmul(attn_w, v)                         # (B, H, S, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        attn_out = self.o_proj(attn_out)

        x = residual + self.dropout(attn_out)

        # FFN (Pre-LN)
        residual = x
        x = residual + self.dropout(self.ffn(self.ln2(x)))
        return x
