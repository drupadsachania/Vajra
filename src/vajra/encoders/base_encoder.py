"""DomainEncoder base class — stacks N PreLNTransformerBlocks.

Returns (hidden_states, cls_token).  Caches hidden states at the
AFN-target layer for downstream Activation Flow Network scoring.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import PreLNTransformerBlock

_AFN_LAYER_MAP: dict[int, int] = {12: 8, 8: 6, 6: 4, 4: 3}


class DomainEncoder(nn.Module):
    """Generic domain encoder: N Pre-LN transformer blocks.

    CLS token is hidden_states[:, 0, :] after all blocks.
    Hidden states at the AFN-target layer are stored in `self.afn_hidden`
    after each forward pass (used by ActivationFlowNetwork).
    """

    def __init__(
        self,
        layers: int,
        d_model: int,
        n_heads: int,
        ffn_width: int,
        context_window: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.context_window = context_window
        self.afn_layer = _AFN_LAYER_MAP.get(layers, layers - 2)

        self.blocks = nn.ModuleList(
            [
                PreLNTransformerBlock(d_model, n_heads, ffn_width, dropout)
                for _ in range(layers)
            ]
        )
        self.final_ln = nn.LayerNorm(d_model)

        # Populated during forward; used by AFN
        self.afn_hidden: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x : (batch, seq, d_model) — already embedded input
        Returns: (hidden_states, cls_token)
          hidden_states : (batch, seq, d_model)
          cls_token     : (batch, d_model)
        """
        for i, block in enumerate(self.blocks):
            x = block(x, attn_mask)
            if i + 1 == self.afn_layer:  # 1-indexed layer == afn_layer
                self.afn_hidden = x.detach()

        x = self.final_ln(x)
        cls_token = x[:, 0, :]
        return x, cls_token
