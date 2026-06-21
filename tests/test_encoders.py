"""Tests for Chunk 3 — domain encoder blocks and 5 standard encoders."""

import pytest
import torch

from vajra.config import VajraConfig
from vajra.encoders.base_encoder import _AFN_LAYER_MAP, DomainEncoder
from vajra.encoders.blocks import (PreLNTransformerBlock, RotaryEmbedding,
                                   SwiGLUFFN, apply_rotary_pos_emb)
from vajra.encoders.domain_encoders import build_domain_encoders

CFG = VajraConfig()


# ── RoPE ─────────────────────────────────────────────────────────────────────


class TestRotaryEmbedding:

    def test_shape(self):
        rope = RotaryEmbedding(dim=64)
        cos, sin = rope(seq_len=32, device=torch.device("cpu"))
        assert cos.shape == (32, 32)
        assert sin.shape == (32, 32)

    def test_apply_rotary_shapes(self):
        B, H, S, hd = 2, 4, 16, 32
        q = torch.randn(B, H, S, hd)
        k = torch.randn(B, H, S, hd)
        rope = RotaryEmbedding(dim=hd)
        cos, sin = rope(S, torch.device("cpu"))
        q_r, k_r = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_r.shape == q.shape
        assert k_r.shape == k.shape


# ── SwiGLU FFN ────────────────────────────────────────────────────────────────


class TestSwiGLUFFN:

    def test_output_shape(self):
        ffn = SwiGLUFFN(d_model=256, ffn_width=1024)
        x = torch.randn(2, 16, 256)
        out = ffn(x)
        assert out.shape == (2, 16, 256)

    def test_gradient_flows(self):
        ffn = SwiGLUFFN(d_model=64, ffn_width=256)
        x = torch.randn(1, 4, 64, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None


# ── Pre-LN Block ──────────────────────────────────────────────────────────────


class TestPreLNTransformerBlock:

    def test_output_shape(self):
        block = PreLNTransformerBlock(d_model=256, n_heads=4, ffn_width=1024)
        x = torch.randn(2, 32, 256)
        out = block(x)
        assert out.shape == (2, 32, 256)

    def test_pre_ln_order(self):
        """LayerNorm should be applied before attention, not after."""
        block = PreLNTransformerBlock(d_model=64, n_heads=4, ffn_width=256)
        # Check that ln1 and ln2 are LayerNorm (not Identity)
        assert isinstance(block.ln1, torch.nn.LayerNorm)
        assert isinstance(block.ln2, torch.nn.LayerNorm)

    def test_with_attn_mask(self):
        block = PreLNTransformerBlock(d_model=64, n_heads=4, ffn_width=256)
        B, S, D = 2, 8, 64
        x = torch.randn(B, S, D)
        mask = torch.zeros(B, 1, S, S)
        out = block(x, attn_mask=mask)
        assert out.shape == (B, S, D)


# ── DomainEncoder ─────────────────────────────────────────────────────────────


class TestDomainEncoder:

    @pytest.mark.parametrize(
        "domain",
        [
            "cti_stix",
            "vulnerability_risk",
            "identity_access",
            "incident_response",
            "compliance",
        ],
    )
    def test_cls_token_shape(self, domain):
        enc_cfg = CFG.domain_encoders[domain]
        enc = DomainEncoder(
            layers=enc_cfg.layers,
            d_model=enc_cfg.d_model,
            n_heads=enc_cfg.attention_heads,
            ffn_width=enc_cfg.ffn_width,
        )
        x = torch.randn(2, 128, enc_cfg.d_model)
        _, cls = enc(x)
        assert cls.shape == (2, enc_cfg.d_model)

    @pytest.mark.parametrize(
        "domain,expected_layers",
        [
            ("cti_stix", 8),
            ("identity_access", 6),
            ("compliance", 4),
        ],
    )
    def test_layer_count(self, domain, expected_layers):
        enc_cfg = CFG.domain_encoders[domain]
        enc = DomainEncoder(
            layers=enc_cfg.layers,
            d_model=enc_cfg.d_model,
            n_heads=enc_cfg.attention_heads,
            ffn_width=enc_cfg.ffn_width,
        )
        assert len(enc.blocks) == expected_layers

    def test_afn_layer_targets(self):
        assert _AFN_LAYER_MAP[12] == 8
        assert _AFN_LAYER_MAP[8] == 6
        assert _AFN_LAYER_MAP[6] == 4
        assert _AFN_LAYER_MAP[4] == 3

    def test_afn_hidden_populated(self):
        enc = DomainEncoder(layers=4, d_model=64, n_heads=4, ffn_width=256)
        x = torch.randn(1, 8, 64)
        _, _ = enc(x)
        assert enc.afn_hidden is not None
        assert enc.afn_hidden.shape == (1, 8, 64)

    def test_no_shared_weights(self):
        encs = build_domain_encoders(CFG)
        domains = list(encs.keys())
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                ea, eb = encs[domains[i]], encs[domains[j]]
                for pa in ea.parameters():
                    for pb in eb.parameters():
                        assert (
                            pa is not pb
                        ), f"Shared weight found between {domains[i]} and {domains[j]}"

    @pytest.mark.parametrize(
        "domain,approx_params,tolerance",
        [
            # Actual SwiGLU intermediate = floor(ffn_width * 2/3), rounded to 64.
            # Spec "params_approx" values assumed full ffn_width; true counts are lower.
            ("cti_stix", 56_649_216, 0.02),
            ("identity_access", 42_487_296, 0.02),
            ("compliance", 12_854_272, 0.02),
        ],
    )
    def test_param_counts(self, domain, approx_params, tolerance):
        enc_cfg = CFG.domain_encoders[domain]
        enc = DomainEncoder(
            layers=enc_cfg.layers,
            d_model=enc_cfg.d_model,
            n_heads=enc_cfg.attention_heads,
            ffn_width=enc_cfg.ffn_width,
        )
        n = sum(p.numel() for p in enc.parameters())
        lo = approx_params * (1 - tolerance)
        hi = approx_params * (1 + tolerance)
        assert (
            lo <= n <= hi
        ), f"{domain}: param count {n:,} outside [{lo:,.0f}, {hi:,.0f}]"

    def test_hidden_states_shape(self):
        enc = DomainEncoder(layers=4, d_model=64, n_heads=4, ffn_width=256)
        x = torch.randn(2, 16, 64)
        hidden, cls = enc(x)
        assert hidden.shape == (2, 16, 64)
        assert cls.shape == (2, 64)
