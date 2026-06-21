"""Tests for Chunk 4 — DCAT blocks + long-context encoders."""

import pytest
import torch

from vajra.config import VajraConfig
from vajra.encoders.dcat import BaselineCache, DCATBlock, _stable_kl_div
from vajra.encoders.domain_encoders import build_domain_encoders
from vajra.encoders.long_context import (InterleavedAttentionEncoder,
                                         _GlobalBlock, _LocalBlock)

CFG = VajraConfig()


# ── KL divergence helper ──────────────────────────────────────────────────────


class TestStableKLDiv:

    def test_zero_when_equal(self):
        p = torch.softmax(torch.randn(2, 4, 8, 16), dim=-1)
        kl = _stable_kl_div(p, p)
        assert kl.abs().max().item() < 1e-4

    def test_nonnegative(self):
        p = torch.softmax(torch.randn(2, 4, 8, 16), dim=-1)
        q = torch.softmax(torch.randn(2, 4, 8, 16), dim=-1)
        kl = _stable_kl_div(p, q)
        assert (kl >= 0).all(), "KL divergence must be non-negative"

    def test_nonzero_for_different(self):
        p = torch.softmax(torch.randn(2, 4, 8, 16), dim=-1)
        q = torch.softmax(torch.randn(2, 4, 8, 16) + 3.0, dim=-1)
        kl = _stable_kl_div(p, q)
        assert kl.max().item() > 0.01


# ── Baseline Cache ────────────────────────────────────────────────────────────


class TestBaselineCache:

    def test_initializes_on_first_update(self):
        cache = BaselineCache(d_model=64)
        state = torch.randn(1, 8, 64)
        cache.update(state, decision_class=0)
        got = cache.get(fallback=state)
        assert got.shape == state.shape

    def test_update_class_0_modifies_cache(self):
        cache = BaselineCache(d_model=64)
        s1 = torch.zeros(1, 4, 64)
        s2 = torch.ones(1, 4, 64)
        cache.update(s1, decision_class=0)
        cache.update(s2, decision_class=0)
        val = cache.get(fallback=s1)
        # After EMA update: should be between 0 and 1, not still 0
        assert val.mean().item() > 0.05

    def test_freeze_on_class_1(self):
        cache = BaselineCache(d_model=64)
        s1 = torch.zeros(1, 4, 64)
        cache.update(s1, decision_class=0)
        frozen_val = cache.get(fallback=s1).clone()

        s2 = torch.ones(1, 4, 64)
        cache.update(s2, decision_class=1)  # should NOT change cache
        val_after = cache.get(fallback=s1)
        torch.testing.assert_close(frozen_val, val_after)

    def test_freeze_on_class_3(self):
        cache = BaselineCache(d_model=64)
        s1 = torch.full((1, 4, 64), 0.5)
        cache.update(s1, decision_class=0)
        frozen = cache.get(fallback=s1).clone()
        cache.update(torch.ones(1, 4, 64), decision_class=3)
        torch.testing.assert_close(frozen, cache.get(fallback=s1))


# ── DCAT Block ────────────────────────────────────────────────────────────────


class TestDCATBlock:

    def _make_block(self, d=64, h=4):
        return DCATBlock(d_model=d, n_heads=h, ffn_width=256)

    def test_output_shape(self):
        block = self._make_block()
        x = torch.randn(2, 16, 64)
        out, div = block(x)
        assert out.shape == (2, 16, 64)
        assert div.shape == (2, 4, 16)  # (batch, heads, seq)

    def test_divergence_zero_when_x_equals_baseline(self):
        block = self._make_block()
        x = torch.randn(1, 8, 64)
        # Initialize the cache to x so baseline == observed
        block.baseline_cache.update(x, decision_class=0)
        out, div = block(x, decision_class=2)
        assert (
            div.abs().max().item() < 1e-3
        ), f"KL divergence should be ≈0 when X==B, got max={div.abs().max().item()}"

    def test_divergence_nonzero_for_different_streams(self):
        block = self._make_block()
        x = torch.randn(1, 8, 64)
        # Set baseline to a very different state
        baseline = torch.randn(1, 8, 64) * 10
        block.baseline_cache.update(baseline, decision_class=0)
        _, div = block(x, decision_class=2)
        assert (
            div.max().item() > 1e-6
        ), "KL divergence should be non-zero for divergent streams"

    def test_kl_nonnegative(self):
        block = self._make_block()
        x = torch.randn(2, 8, 64)
        _, div = block(x)
        assert (div >= 0).all(), "KL divergence must always be ≥ 0"

    def test_gradient_flow_to_dcat_params(self):
        block = self._make_block()
        x = torch.randn(1, 4, 64)
        out, div = block(x)
        (out.sum() + div.sum()).backward()
        assert block.q_proj.weight.grad is not None
        assert block.k_proj.weight.grad is not None


# ── InterleavedAttentionEncoder ────────────────────────────────────────────────


class TestInterleavedAttentionEncoder:

    def test_output_shape_small(self):
        enc = InterleavedAttentionEncoder(
            d_model=64, n_heads=4, n_layers=4, ffn_width=256
        )
        x = torch.randn(1, 32, 64)
        hidden, cls = enc(x)
        assert hidden.shape == (1, 32, 64)
        assert cls.shape == (1, 64)

    def test_dcat_divergence_populated(self):
        enc = InterleavedAttentionEncoder(
            d_model=64, n_heads=4, n_layers=8, ffn_width=256
        )
        x = torch.randn(1, 16, 64)
        enc(x)
        assert enc.last_dcat_divergence is not None

    def test_global_layers_exist(self):
        enc = InterleavedAttentionEncoder(
            d_model=64, n_heads=4, n_layers=12, ffn_width=256
        )
        global_count = sum(1 for b in enc.blocks if isinstance(b, _GlobalBlock))
        assert global_count > 0, "Should have at least one global NoPE layer"

    def test_dcat_layers_exist(self):
        enc = InterleavedAttentionEncoder(
            d_model=64, n_heads=4, n_layers=12, ffn_width=256
        )
        from vajra.encoders.dcat import DCATBlock

        dcat_count = sum(1 for b in enc.blocks if isinstance(b, DCATBlock))
        assert dcat_count == 2, f"Should have exactly 2 DCAT layers, got {dcat_count}"

    def test_afn_hidden_populated_12l(self):
        enc = InterleavedAttentionEncoder(
            d_model=64, n_heads=4, n_layers=12, ffn_width=256
        )
        x = torch.randn(1, 8, 64)
        enc(x)
        assert enc.afn_hidden is not None
        assert enc.afn_hidden.shape == (1, 8, 64)

    def test_full_config_cls_shape(self):
        """Detection encoder (12L/1024/16) produces correct CLS shape."""
        enc_cfg = CFG.domain_encoders["detection_network"]
        enc = InterleavedAttentionEncoder(
            d_model=enc_cfg.d_model,
            n_heads=enc_cfg.attention_heads,
            n_layers=enc_cfg.layers,
            ffn_width=enc_cfg.ffn_width,
        )
        # Use smaller seq length for speed
        x = torch.randn(1, 16, enc_cfg.d_model)
        _, cls = enc(x)
        assert cls.shape == (1, enc_cfg.d_model)

    def test_dcat_divergence_shape_detection(self):
        """DCAT D_divergence shape correct for Detection encoder."""
        enc = InterleavedAttentionEncoder(
            d_model=64, n_heads=4, n_layers=12, ffn_width=256
        )
        x = torch.randn(1, 16, 64)
        enc(x)
        div = enc.last_dcat_divergence
        assert div is not None
        assert div.shape == (1, 4, 16)  # (batch, heads, seq)


# ── Updated domain encoder tests (include Detection/Forensics) ────────────────


class TestAllDomainEncoders:

    def test_all_7_built(self):
        encs = build_domain_encoders(CFG)
        assert len(encs) == 7

    def test_detection_is_interleaved(self):
        encs = build_domain_encoders(CFG)
        assert isinstance(encs["detection_network"], InterleavedAttentionEncoder)
        assert isinstance(encs["forensics_provenance"], InterleavedAttentionEncoder)

    def test_no_shared_weights_all_7(self):
        encs = build_domain_encoders(CFG)
        domains = list(encs.keys())
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                ea, eb = encs[domains[i]], encs[domains[j]]
                for pa in ea.parameters():
                    for pb in eb.parameters():
                        assert pa is not pb
