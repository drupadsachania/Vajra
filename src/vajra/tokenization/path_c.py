"""Path C — graph embedding pipeline (TransE + RotatE + GCN for MITRE ATT&CK;
inference-time GCN for consuming-system RAG-provided graphs).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransERotatELayer(nn.Module):
    """Shared entity/relation embeddings for both TransE and RotatE scoring.

    Frozen after Stage 1 pre-training.
    d_kg=256 as specified in §3.
    """

    def __init__(self, num_entities: int, num_relations: int, d_kg: int = 256):
        super().__init__()
        self.d_kg = d_kg
        self.entity_emb = nn.Embedding(num_entities, d_kg)
        self.relation_emb = nn.Embedding(num_relations, d_kg)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(
            self.entity_emb.weight, -6 / math.sqrt(self.d_kg), 6 / math.sqrt(self.d_kg)
        )
        nn.init.uniform_(
            self.relation_emb.weight,
            -6 / math.sqrt(self.d_kg),
            6 / math.sqrt(self.d_kg),
        )
        # RotatE requires each complex element (re_j, im_j) to have unit modulus,
        # not the whole vector to have unit L2 norm. Normalize per (re, im) pair.
        with torch.no_grad():
            w = self.relation_emb.weight.data
            d = self.d_kg // 2
            re, im = w[:, :d], w[:, d:]
            mod = (re**2 + im**2).sqrt().clamp_min(1e-9)
            self.relation_emb.weight.data = torch.cat([re / mod, im / mod], dim=-1)

    def transe_score(
        self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """||h + r - t||₂ (lower = more plausible)."""
        return (h + r - t).norm(dim=-1)

    def rotate_score(
        self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """||h ∘ r - t||₂ in complex space (d_kg must be even)."""
        d = self.d_kg // 2
        h_re, h_im = h[..., :d], h[..., d:]
        r_re, r_im = r[..., :d], r[..., d:]
        t_re, t_im = t[..., :d], t[..., d:]
        prod_re = h_re * r_re - h_im * r_im
        prod_im = h_re * r_im + h_im * r_re
        diff = torch.cat([prod_re - t_re, prod_im - t_im], dim=-1)
        return diff.norm(dim=-1)


class GCNLayer(nn.Module):
    """Single graph convolutional layer: A_hat * X * W."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x   : (N, in_dim)
        adj : (N, N) normalized adjacency (A_hat = D^{-1/2} A D^{-1/2})
        """
        return F.gelu(self.linear(adj @ x))


class MitreKGEmbedding(nn.Module):
    """Frozen MITRE ATT&CK knowledge graph encoder.

    Architecture: TransE + RotatE scoring head + 2-layer GCN → d_model projection.
    Frozen after Stage 1 (set frozen=True to disable gradients).
    """

    def __init__(
        self,
        num_techniques: int = 716,
        num_relations: int = 32,
        d_kg: int = 256,
        d_model: int = 1024,
        frozen: bool = False,
    ):
        super().__init__()
        self.d_kg = d_kg
        self.kge = TransERotatELayer(num_techniques, num_relations, d_kg)
        self.gcn1 = GCNLayer(d_kg, d_kg)
        self.gcn2 = GCNLayer(d_kg, d_kg)
        self.proj = nn.Linear(d_kg, d_model)
        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, entity_ids: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        entity_ids : (N,) long tensor of technique/tactic node IDs
        adj        : (N, N) normalized adjacency
        Returns    : (N, d_model)
        """
        x = self.kge.entity_emb(entity_ids)  # (N, d_kg)
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        return self.proj(x)  # (N, d_model)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad_(True)


class InferenceGCN(nn.Module):
    """Unfrozen GCN for consuming-system RAG-provided graphs at inference time.

    Takes raw node feature vectors (d_graph=512) and produces d_model embeddings.
    Includes Graphormer-style degree features concatenated before projection.
    """

    DEGREE_BINS = 16

    def __init__(self, d_graph: int = 512, d_model: int = 1024):
        super().__init__()
        self.degree_emb = nn.Embedding(self.DEGREE_BINS + 1, 32)
        self.gcn1 = GCNLayer(d_graph + 32, d_graph)
        self.gcn2 = GCNLayer(d_graph, d_model)

    def _degree_features(self, adj: torch.Tensor) -> torch.Tensor:
        # Count non-zero neighbours from the *binary* structure. Using the
        # normalized adjacency (rows summing to ~1) would collapse every degree
        # to 0/1 after .long() truncation, making the Graphormer feature useless.
        degrees = (adj > 0).sum(dim=-1).clamp(max=self.DEGREE_BINS)  # (N,)
        return self.degree_emb(degrees)  # (N, 32)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x   : (N, d_graph) raw node features
        adj : (N, N) normalized adjacency
        Returns: (N, d_model)
        """
        deg_feat = self._degree_features(adj)
        x = torch.cat([x, deg_feat], dim=-1)  # (N, d_graph+32)
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        return x
