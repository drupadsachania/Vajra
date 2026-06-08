"""Shared fixtures for Vajra test suite."""

import pytest
import torch

from vajra.config import VajraConfig


@pytest.fixture(scope="session")
def cfg() -> VajraConfig:
    return VajraConfig()


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)


def tiny_cfg() -> VajraConfig:
    """Minimal config for fast unit tests — smaller d_model and fewer layers."""
    from vajra.config import EncoderConfig, FusionConfig, DecoderConfig
    cfg = VajraConfig()
    cfg.shared_d_model = 64
    cfg.domain_encoders = {
        "detection_network":    EncoderConfig(layers=2, d_model=64, attention_heads=4, ffn_width=256, dcat_layers=[1]),
        "forensics_provenance": EncoderConfig(layers=2, d_model=64, attention_heads=4, ffn_width=256, dcat_layers=[1]),
        "cti_stix":             EncoderConfig(layers=2, d_model=64, attention_heads=4, ffn_width=256),
        "vulnerability_risk":   EncoderConfig(layers=2, d_model=64, attention_heads=4, ffn_width=256),
        "identity_access":      EncoderConfig(layers=2, d_model=64, attention_heads=4, ffn_width=256),
        "incident_response":    EncoderConfig(layers=2, d_model=64, attention_heads=4, ffn_width=256),
        "compliance":           EncoderConfig(layers=2, d_model=64, attention_heads=4, ffn_width=256),
    }
    cfg.fusion = FusionConfig(blocks=2, d_model=64, attention_heads=4, ffn_width=256)
    cfg.decoder = DecoderConfig(layers=2, d_model=64, attention_heads=4, ffn_width=256, max_output_tokens=16)
    return cfg
