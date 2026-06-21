"""Source-type embeddings — 16 distinct source types, each → d_model."""

import torch
import torch.nn as nn

# The 16 source types referenced in §3
SOURCE_TYPES = [
    "evtx",
    "sysmon",
    "netflow",
    "dns",
    "http",
    "stix",
    "sigma",
    "ldap",
    "cloud_audit",
    "edr",
    "vulnerability",
    "identity",
    "compliance",
    "incident",
    "email",
    "other",
]
assert len(SOURCE_TYPES) == 16


class SourceTypeEmbedding(nn.Module):
    """Lookup table: source_type_id (0–15) → (batch, seq, d_model)."""

    NUM_SOURCE_TYPES = 16

    def __init__(self, d_model: int = 1024):
        super().__init__()
        self.embed = nn.Embedding(self.NUM_SOURCE_TYPES, d_model)

    def forward(self, source_type_ids: torch.Tensor) -> torch.Tensor:
        """
        source_type_ids : (batch, seq) long tensor, values 0–15
        Returns         : (batch, seq, d_model)
        """
        return self.embed(source_type_ids)

    @staticmethod
    def type_to_id(source_type: str) -> int:
        if source_type in SOURCE_TYPES:
            return SOURCE_TYPES.index(source_type)
        return SOURCE_TYPES.index("other")
