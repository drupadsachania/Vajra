"""Interleaved RoPE/NoPE encoder for Detection and Forensics (§4.1 + §4.3).

Local layers: standard attention with 4096-token sliding window + RoPE.
Global layers (every 4th, 0-indexed): full attention + NoPE (no pos embed).
DCAT replaces layers 7 and 8 (1-indexed).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import RotaryEmbedding, apply_rotary_pos_emb, SwiGLUFFN
from .dcat import DCATBlock


_LOCAL_WINDOW = 4096


class LocalWindowAttention(nn.Module):
    """Multi-head self-attention with RoPE, masked to a local sliding window."""

    def __init__(self, d_model: int, n_heads: int, window_size: int = _LOCAL_WINDOW, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.window   = window_size

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(S, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scale = math.sqrt(self.head_dim)
        w = torch.matmul(q, k.transpose(-2, -1)) / scale   # (B, H, S, S)

        # Local sliding window: a position attends to ±(window/2) neighbours, so
        # the total attended span is ~`window` tokens (§4.1 specifies a 4096-token
        # window). Using ±window would give a ~2×window span (the previous bug).
        half = self.window // 2
        if S > self.window:
            pos = torch.arange(S, device=x.device)
            mask = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs() > half
            w = w.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        w = F.softmax(w, dim=-1)
        w = self.drop(w)
        out = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.o(out)


class GlobalNoPEAttention(nn.Module):
    """Full (non-windowed) multi-head self-attention with NO positional encoding."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # No RoPE — NoPE: query and key used as-is
        scale = math.sqrt(self.head_dim)
        w = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / scale, dim=-1)
        w = self.drop(w)
        out = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.o(out)


class _LocalBlock(nn.Module):
    """Pre-LN wrapper around LocalWindowAttention + SwiGLU."""

    def __init__(self, d_model: int, n_heads: int, ffn_width: int, window: int = _LOCAL_WINDOW, dropout: float = 0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.attn = LocalWindowAttention(d_model, n_heads, window, dropout)
        self.ffn  = SwiGLUFFN(d_model, ffn_width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class _GlobalBlock(nn.Module):
    """Pre-LN wrapper around GlobalNoPEAttention + SwiGLU."""

    def __init__(self, d_model: int, n_heads: int, ffn_width: int, dropout: float = 0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.attn = GlobalNoPEAttention(d_model, n_heads, dropout)
        self.ffn  = SwiGLUFFN(d_model, ffn_width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class InterleavedAttentionEncoder(nn.Module):
    """12-layer interleaved RoPE/NoPE encoder for Detection and Forensics (§4.1).

    The spec says DCAT replaces layers 7 and 8 (1-indexed = 0-indexed 6 and 7),
    and global NoPE layers fall on the stride-4 slots (0-indexed 3, 7, 11). DCAT
    is checked first, so it claims slot 7; the resulting actual schedule is:

      - Layers 0,1,2   → Local + RoPE
      - Layer 3        → Global + NoPE   (stride-4 slot)
      - Layers 4,5     → Local + RoPE
      - Layers 6,7     → DCAT             (replaces local + the slot-7 global)
      - Layers 8,9,10  → Local + RoPE
      - Layer 11       → Global + NoPE   (stride-4 slot)

    Net: two global NoPE layers (3, 11), two DCAT layers (6, 7), eight local.
    """

    GLOBAL_STRIDE = 4   # every 4th layer (0-indexed: 3, 7, 11 for 12 layers)
    DCAT_LAYERS_0IDX = (6, 7)    # 0-indexed positions where DCAT blocks sit

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 12,
        ffn_width: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model  = d_model
        self.n_layers = n_layers
        self.afn_layer = 8  # 1-indexed AFN target for 12-layer encoders

        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            if i in self.DCAT_LAYERS_0IDX:
                self.blocks.append(DCATBlock(d_model, n_heads, ffn_width, dropout=dropout))
            elif i % self.GLOBAL_STRIDE == (self.GLOBAL_STRIDE - 1):
                self.blocks.append(_GlobalBlock(d_model, n_heads, ffn_width, dropout))
            else:
                self.blocks.append(_LocalBlock(d_model, n_heads, ffn_width, dropout=dropout))

        self.final_ln = nn.LayerNorm(d_model)

        # AFN hidden state cache
        self.afn_hidden: torch.Tensor | None = None

        # Last DCAT divergence (for tests / downstream use)
        self.last_dcat_divergence: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        decision_class: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x : (batch, seq, d_model)
        Returns: (hidden_states, cls_token)
        """
        dcat_div = None
        for i, block in enumerate(self.blocks):
            if isinstance(block, DCATBlock):
                x, div = block(x, decision_class)
                dcat_div = div
            else:
                x = block(x)
            if (i + 1) == self.afn_layer:  # 1-indexed
                self.afn_hidden = x.detach()

        self.last_dcat_divergence = dcat_div
        x = self.final_ln(x)
        return x, x[:, 0, :]
