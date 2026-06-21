"""EmbeddingLayer — combines all four embedding streams and applies LayerNorm.

Final output: LayerNorm(E_token + E_src_type + E_temporal + E_ontology)
Shape: (batch, seq, d_model=1024)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vajra.config import VajraConfig

from .ontology import OntologyEmbedding
from .source_type import SourceTypeEmbedding
from .temporal import ContiFormerEncoding


class EmbeddingLayer(nn.Module):
    """Combines token, source-type, temporal, and ontology embeddings."""

    def __init__(self, cfg: VajraConfig):
        super().__init__()
        d = cfg.shared_d_model
        vocab = cfg.embedding.vocabulary_size

        self.token_embed = nn.Embedding(vocab, d)
        self.source_type = SourceTypeEmbedding(d)
        self.temporal = ContiFormerEncoding(d)
        self.ontology = OntologyEmbedding(
            d_model=d,
            d_kg=cfg.embedding.ontology_embedding_dim_pre_projection,
            frozen=True,
        )
        self.layer_norm = nn.LayerNorm(d)

    def forward(
        self,
        input_ids: torch.Tensor,
        source_type_ids: torch.Tensor,
        timestamps: torch.Tensor | None = None,
        node_ids: torch.Tensor | None = None,
        adj: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        input_ids       : (batch, seq) long
        source_type_ids : (batch, seq) long, values 0–15
        timestamps      : (batch, seq) float or None
        node_ids        : (N,) long — ATT&CK node IDs for this sequence, or None
        adj             : (N, N) float — adjacency, or None
        Returns         : (batch, seq, d_model)
        """
        batch, seq = input_ids.shape

        e_token = self.token_embed(input_ids)  # (B, S, d)
        e_src = self.source_type(source_type_ids)  # (B, S, d)
        e_temp = self.temporal(timestamps, batch, seq)  # (B, S, d)
        e_ont = self.ontology(node_ids, adj, batch, seq)  # (B, S, d)

        combined = e_token + e_src + e_temp + e_ont
        return self.layer_norm(combined)
