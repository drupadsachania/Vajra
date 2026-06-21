"""MITRE ATT&CK ontology embedding for the embedding layer (§3.4).

Reuses MitreKGEmbedding from path_c but wraps it for the embedding layer
interface: given a sequence of technique/tactic node IDs and an adjacency,
returns (batch, seq, d_model) ontology vectors padded for missing positions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vajra.tokenization.path_c import MitreKGEmbedding


class OntologyEmbedding(nn.Module):
    """Wraps the frozen MITRE KG encoder for use in the embedding layer.

    Tokens that do not correspond to an ontology node receive a learned
    absent_vector placeholder (all-zeros → learned projection).
    """

    def __init__(
        self,
        num_techniques: int = 716,
        num_relations: int = 32,
        d_kg: int = 256,
        d_model: int = 1024,
        frozen: bool = True,
    ):
        super().__init__()
        self.mitre_kg = MitreKGEmbedding(
            num_techniques, num_relations, d_kg, d_model, frozen
        )
        self.absent_vector = nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        node_ids: torch.Tensor | None,
        adj: torch.Tensor | None,
        batch: int,
        seq: int,
    ) -> torch.Tensor:
        """
        node_ids : (N,) long tensor of ATT&CK node IDs, or None
        adj      : (N, N) normalized adjacency, or None
        Returns  : (batch, seq, d_model)
        """
        if node_ids is None or adj is None:
            return self.absent_vector.expand(batch, seq, -1)
        node_embs = self.mitre_kg(node_ids, adj)  # (N, d_model)
        # Pad or truncate to seq length, replicate across batch
        target = seq
        if node_embs.shape[0] >= target:
            out = node_embs[:target]
        else:
            pad = self.absent_vector.unsqueeze(0).expand(
                target - node_embs.shape[0], -1
            )
            out = torch.cat([node_embs, pad], dim=0)
        return out.unsqueeze(0).expand(batch, -1, -1)
