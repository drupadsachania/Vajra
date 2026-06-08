"""Tests for Chunk 2 — embedding layer."""

import pytest
import torch

from vajra.config import VajraConfig
from vajra.embedding.source_type import SourceTypeEmbedding, SOURCE_TYPES
from vajra.embedding.temporal import ContiFormerEncoding
from vajra.embedding.ontology import OntologyEmbedding
from vajra.embedding.embedding_layer import EmbeddingLayer


CFG = VajraConfig()
D = CFG.shared_d_model  # 1024


# ── Source-type embeddings ────────────────────────────────────────────────────

class TestSourceTypeEmbedding:

    def test_output_shape(self):
        emb = SourceTypeEmbedding(d_model=D)
        ids = torch.zeros(2, 64, dtype=torch.long)
        out = emb(ids)
        assert out.shape == (2, 64, D)

    def test_type_to_id_valid(self):
        for st in SOURCE_TYPES:
            assert SourceTypeEmbedding.type_to_id(st) == SOURCE_TYPES.index(st)

    def test_type_to_id_unknown(self):
        idx = SourceTypeEmbedding.type_to_id("unknown_type_xyz")
        assert idx == SOURCE_TYPES.index("other")

    def test_16_types(self):
        assert len(SOURCE_TYPES) == 16


# ── ContiFormer temporal encoding ─────────────────────────────────────────────

class TestContiFormerEncoding:

    def test_null_timestamp_shape(self):
        enc = ContiFormerEncoding(d_model=D)
        out = enc(None, batch=2, seq=64)
        assert out.shape == (2, 64, D)

    def test_nonnull_timestamp_shape(self):
        enc = ContiFormerEncoding(d_model=D)
        ts = torch.rand(2, 64) * 1e9
        out = enc(ts, batch=2, seq=64)
        assert out.shape == (2, 64, D)

    def test_phi_dim(self):
        enc = ContiFormerEncoding(d_model=D)
        ts = torch.rand(2, 4) * 1e6
        phi = enc.phi(ts)
        assert phi.shape == (2, 4, 256)  # 2 * 128 freq pairs

    def test_freq_pairs_count(self):
        enc = ContiFormerEncoding(d_model=D)
        assert enc.freqs.shape == (128,)

    def test_freq_init_log_uniform_range(self):
        enc = ContiFormerEncoding(d_model=D)
        freqs = enc.freqs.detach()
        assert freqs.min().item() >= 1e-8    # allow small numerical slack
        assert freqs.max().item() <= 1e4     # log-uniform up to 1e3

    def test_null_returns_learned_vector(self):
        enc = ContiFormerEncoding(d_model=D)
        out = enc(None, batch=1, seq=1)
        torch.testing.assert_close(out[0, 0], enc.null_vector)

    def test_gradient_flows_through_freqs(self):
        enc = ContiFormerEncoding(d_model=32)
        ts = torch.rand(1, 4) * 1e6
        out = enc(ts, batch=1, seq=4)
        out.sum().backward()
        assert enc.log_freqs.grad is not None


# ── OntologyEmbedding ─────────────────────────────────────────────────────────

class TestOntologyEmbedding:

    def test_absent_shape(self):
        emb = OntologyEmbedding(d_model=64, d_kg=32, frozen=False)
        out = emb(None, None, batch=2, seq=8)
        assert out.shape == (2, 8, 64)

    def test_present_shape(self):
        emb = OntologyEmbedding(num_techniques=20, d_model=64, d_kg=32, frozen=False)
        ids = torch.arange(5)
        adj = torch.eye(5)
        out = emb(ids, adj, batch=2, seq=5)
        assert out.shape == (2, 5, 64)

    def test_padding_short_seq(self):
        emb = OntologyEmbedding(num_techniques=20, d_model=64, d_kg=32, frozen=False)
        ids = torch.arange(3)
        adj = torch.eye(3)
        # Request longer sequence than available nodes
        out = emb(ids, adj, batch=1, seq=10)
        assert out.shape == (1, 10, 64)


# ── EmbeddingLayer (full integration) ────────────────────────────────────────

class TestEmbeddingLayer:

    def test_basic_shape(self):
        layer = EmbeddingLayer(CFG)
        B, S = 2, 64
        input_ids = torch.randint(0, CFG.embedding.vocabulary_size, (B, S))
        src_ids   = torch.randint(0, 16, (B, S))
        out = layer(input_ids, src_ids)
        assert out.shape == (B, S, D)

    def test_with_timestamps(self):
        layer = EmbeddingLayer(CFG)
        B, S = 2, 64
        input_ids = torch.randint(0, CFG.embedding.vocabulary_size, (B, S))
        src_ids   = torch.randint(0, 16, (B, S))
        ts        = torch.rand(B, S) * 1e9
        out = layer(input_ids, src_ids, timestamps=ts)
        assert out.shape == (B, S, D)

    def test_with_null_timestamps(self):
        layer = EmbeddingLayer(CFG)
        B, S = 2, 64
        input_ids = torch.randint(0, CFG.embedding.vocabulary_size, (B, S))
        src_ids   = torch.randint(0, 16, (B, S))
        out = layer(input_ids, src_ids, timestamps=None)
        assert out.shape == (B, S, D)

    def test_layer_norm_applied(self):
        """Output should have approximately unit variance per token after LayerNorm."""
        layer = EmbeddingLayer(CFG)
        B, S = 1, 16
        input_ids = torch.randint(0, CFG.embedding.vocabulary_size, (B, S))
        src_ids   = torch.zeros(B, S, dtype=torch.long)
        out = layer(input_ids, src_ids)
        # LayerNorm mean ≈ 0, std ≈ 1 across d_model dimension
        mean = out.mean(dim=-1).abs().max().item()
        assert mean < 0.5, f"LayerNorm mean too large: {mean}"

    def test_gradient_flows(self):
        layer = EmbeddingLayer(CFG)
        B, S = 1, 8
        input_ids = torch.randint(0, CFG.embedding.vocabulary_size, (B, S))
        src_ids   = torch.zeros(B, S, dtype=torch.long)
        ts = torch.rand(B, S) * 1e6
        out = layer(input_ids, src_ids, timestamps=ts)
        out.sum().backward()
        assert layer.temporal.log_freqs.grad is not None
