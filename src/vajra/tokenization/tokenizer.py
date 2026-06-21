"""Unified VajraTokenizer — routes events to Path A, B, or C."""

from __future__ import annotations

from enum import Enum
from typing import Any

import torch

from vajra.config import VajraConfig

from .path_a import SentinelTokenizer
from .path_b import CvssEncoder, IpEncoder, NetFlowEncoder, NetFlowRecord
from .path_c import InferenceGCN, MitreKGEmbedding


class TokenizationPath(str, Enum):
    A = "sentinel_text"
    B_CVSS = "cvss"
    B_NETFLOW = "netflow"
    B_IP = "ip"
    C_MITRE = "mitre_kg"
    C_GRAPH = "inference_gcn"


class VajraTokenizer:
    """Routes security events to the correct tokenization path.

    Path A: sentinel-bounded text (EVTX, STIX, Sigma, etc.)
    Path B: numerical sub-tokenizer (CVSS strings, NetFlow, IP addresses)
    Path C: graph embeddings (MITRE KG, RAG-provided graphs)
    """

    def __init__(self, cfg: VajraConfig, hf_model_name: str | None = None):
        self.cfg = cfg
        self.path_a = SentinelTokenizer(cfg, hf_model_name)

    @staticmethod
    def detect_path(event: dict[str, Any]) -> TokenizationPath:
        """Heuristic routing based on event 'type' field."""
        etype = event.get("type", "").lower()
        if etype in ("cvss", "cve"):
            return TokenizationPath.B_CVSS
        if etype in ("netflow", "flow"):
            return TokenizationPath.B_NETFLOW
        if etype == "ip":
            return TokenizationPath.B_IP
        if etype in ("mitre", "technique", "tactic"):
            return TokenizationPath.C_MITRE
        if etype in ("graph", "asset_graph"):
            return TokenizationPath.C_GRAPH
        # Default: sentinel-bounded text (EVTX, STIX, SIGMA, LDAP, CLOUD…)
        return TokenizationPath.A

    def encode_text(self, text: str) -> list[int]:
        """Path A: encode text to token IDs."""
        return self.path_a.encode(text)

    @property
    def vocab_size(self) -> int:
        return self.path_a.vocab_size

    def sentinel_id(self, token: str) -> int:
        return self.path_a.sentinel_id(token)
