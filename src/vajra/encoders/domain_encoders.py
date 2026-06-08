"""Instantiated domain encoders for all seven domains (§4.1)."""

from __future__ import annotations

import torch.nn as nn

from vajra.config import VajraConfig
from .base_encoder import DomainEncoder
from .long_context import InterleavedAttentionEncoder


def build_standard_encoders(cfg: VajraConfig) -> dict[str, DomainEncoder]:
    """Build the 5 standard (non-long-context) domain encoders."""
    standard_domains = [
        "cti_stix",
        "vulnerability_risk",
        "identity_access",
        "incident_response",
        "compliance",
    ]
    encoders: dict[str, DomainEncoder] = {}
    for domain in standard_domains:
        enc_cfg = cfg.domain_encoders[domain]
        encoders[domain] = DomainEncoder(
            layers=enc_cfg.layers,
            d_model=enc_cfg.d_model,
            n_heads=enc_cfg.attention_heads,
            ffn_width=enc_cfg.ffn_width,
            context_window=enc_cfg.context_window_tokens,
        )
    return encoders


def build_domain_encoders(cfg: VajraConfig) -> dict[str, nn.Module]:
    """Build all 7 domain encoders."""
    encoders: dict[str, nn.Module] = build_standard_encoders(cfg)

    for domain in ("detection_network", "forensics_provenance"):
        enc_cfg = cfg.domain_encoders[domain]
        encoders[domain] = InterleavedAttentionEncoder(
            d_model=enc_cfg.d_model,
            n_heads=enc_cfg.attention_heads,
            n_layers=enc_cfg.layers,
            ffn_width=enc_cfg.ffn_width,
        )

    return encoders
