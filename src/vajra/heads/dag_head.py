"""Evidence chain DAG head (§6.2).

Architecture: Linear(7×1024 → 2048) → GELU → Linear(2048 → 32²) → reshape (32, 32)
Produces a sparse adjacency matrix over 32 possible evidence nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceDAGHead(nn.Module):
    """Maps concatenated 7-domain CLS tokens to a 32×32 DAG adjacency matrix."""

    MAX_NODES = 32

    def __init__(self, n_domains: int = 7, d_model: int = 1024):
        super().__init__()
        in_dim = n_domains * d_model
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, self.MAX_NODES * self.MAX_NODES),
        )

    def forward(self, domain_cls: torch.Tensor) -> torch.Tensor:
        """
        domain_cls : (batch, 7, d_model)
        Returns    : (batch, 32, 32) — raw logits for adjacency matrix
        """
        B = domain_cls.shape[0]
        flat = domain_cls.reshape(B, -1)             # (batch, 7*d_model)
        adj_flat = self.mlp(flat)                    # (batch, 32*32)
        return adj_flat.reshape(B, self.MAX_NODES, self.MAX_NODES)
