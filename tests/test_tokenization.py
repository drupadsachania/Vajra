"""Tests for Chunk 2 — tokenization (Paths A, B, C)."""

import pytest
import torch

from vajra.config import VajraConfig
from vajra.tokenization.path_a import SentinelTokenizer
from vajra.tokenization.path_b import CvssEncoder, NetFlowEncoder, IpEncoder, NetFlowRecord
from vajra.tokenization.path_c import MitreKGEmbedding, InferenceGCN, GCNLayer
from vajra.tokenization.tokenizer import VajraTokenizer, TokenizationPath


CFG = VajraConfig()


# ── Path A ────────────────────────────────────────────────────────────────────

class TestSentinelTokenizer:

    def test_sentinel_count(self):
        tok = SentinelTokenizer(CFG)
        assert len(tok.SENTINEL_TOKENS) == 20

    def test_sentinel_ids_unique(self):
        tok = SentinelTokenizer(CFG)
        ids = [tok.sentinel_id(s) for s in tok.SENTINEL_TOKENS]
        assert len(set(ids)) == 20, "sentinel IDs must all be unique"

    def test_sentinel_not_fragmented(self):
        """Each sentinel must map to a single integer ID, not a list."""
        tok = SentinelTokenizer(CFG)
        for s in tok.SENTINEL_TOKENS:
            ids = tok.encode(s)
            assert len(ids) == 1, f"{s!r} was fragmented into {ids}"

    def test_is_sentinel(self):
        tok = SentinelTokenizer(CFG)
        sid = tok.sentinel_id("<|S_NULL|>")
        assert tok.is_sentinel(sid)
        assert not tok.is_sentinel(42)  # unlikely to be a sentinel

    def test_vocab_size(self):
        tok = SentinelTokenizer(CFG)
        assert tok.vocab_size == CFG.embedding.vocabulary_size


# ── Path B — CVSS ─────────────────────────────────────────────────────────────

class TestCvssEncoder:
    CVSS_EXAMPLE = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"

    def test_parse_one_hot_shape(self):
        vec = CvssEncoder.parse_cvss(self.CVSS_EXAMPLE)
        assert vec.shape == (25,)

    def test_one_hot_sum(self):
        vec = CvssEncoder.parse_cvss(self.CVSS_EXAMPLE)
        # 8 metrics each set exactly one bit → sum == 8
        assert int(vec.sum().item()) == 8

    def test_forward_shape(self):
        enc = CvssEncoder(d_model=1024)
        vec = CvssEncoder.parse_cvss(self.CVSS_EXAMPLE).unsqueeze(0)  # (1, 25)
        out = enc(vec)
        assert out.shape == (1, 1024)

    def test_batch_forward(self):
        enc = CvssEncoder(d_model=1024)
        batch = torch.stack([CvssEncoder.parse_cvss(self.CVSS_EXAMPLE)] * 4)  # (4, 25)
        out = enc(batch)
        assert out.shape == (4, 1024)


# ── Path B — NetFlow ──────────────────────────────────────────────────────────

class TestNetFlowEncoder:

    def _make_record(self) -> NetFlowRecord:
        return NetFlowRecord(
            src_ip="192.168.1.10",
            dst_ip="10.0.0.5",
            src_port=54321,
            dst_port=443,
            protocol=6,
            bytes_sent=1500,
            packets=3,
            duration_ms=12.5,
        )

    def test_all_fields_present(self):
        enc = NetFlowEncoder(d_model=1024)
        rec = self._make_record()
        fields = enc.encode_record(rec)
        assert set(fields.keys()) == {"src_ip", "dst_ip", "src_port", "dst_port",
                                      "protocol", "bytes", "packets", "duration"}

    def test_field_shapes(self):
        enc = NetFlowEncoder(d_model=1024)
        rec = self._make_record()
        fields = enc.encode_record(rec)
        for name, t in fields.items():
            assert t.shape == (1, 1024), f"Field {name} has wrong shape: {t.shape}"

    def test_forward_shape(self):
        enc = NetFlowEncoder(d_model=1024)
        out = enc(self._make_record())
        assert out.shape == (1, 8, 1024)


# ── Path B — IP encoder ───────────────────────────────────────────────────────

class TestIpEncoder:

    def test_rfc1918_one_hot(self):
        vec = IpEncoder.encode_ip("192.168.1.10")
        assert vec.shape == (257,)
        assert vec[256].item() == 0.0   # not external
        assert vec.sum().item() == 1.0  # exactly one bit set

    def test_external_ip(self):
        vec = IpEncoder.encode_ip("8.8.8.8")
        assert vec[256].item() == 1.0

    def test_forward_shape(self):
        enc = IpEncoder(d_model=1024)
        ip_vec = IpEncoder.encode_ip("10.0.0.1").unsqueeze(0)
        out = enc(ip_vec)
        assert out.shape == (1, 1024)


# ── Path C — MITRE KG ─────────────────────────────────────────────────────────

class TestMitreKGEmbedding:

    def test_output_shape(self):
        model = MitreKGEmbedding(num_techniques=10, num_relations=4, d_kg=32, d_model=64)
        ids = torch.arange(5)
        adj = torch.eye(5)
        out = model(ids, adj)
        assert out.shape == (5, 64)

    def test_frozen_no_grad(self):
        model = MitreKGEmbedding(num_techniques=10, num_relations=4, d_kg=32, d_model=64, frozen=True)
        for p in model.parameters():
            assert not p.requires_grad

    def test_freeze_unfreeze(self):
        model = MitreKGEmbedding(num_techniques=10, num_relations=4, d_kg=32, d_model=64)
        model.freeze()
        assert not list(model.parameters())[0].requires_grad
        model.unfreeze()
        assert list(model.parameters())[0].requires_grad


# ── Path C — InferenceGCN ─────────────────────────────────────────────────────

class TestInferenceGCN:

    def test_output_shape(self):
        model = InferenceGCN(d_graph=32, d_model=64)
        N = 8
        x = torch.randn(N, 32)
        adj = torch.eye(N)
        out = model(x, adj)
        assert out.shape == (N, 64)


# ── Unified tokenizer routing ─────────────────────────────────────────────────

class TestVajraTokenizer:

    def test_routing_cvss(self):
        path = VajraTokenizer.detect_path({"type": "cvss", "data": "..."})
        assert path == TokenizationPath.B_CVSS

    def test_routing_netflow(self):
        path = VajraTokenizer.detect_path({"type": "netflow"})
        assert path == TokenizationPath.B_NETFLOW

    def test_routing_evtx_defaults_to_a(self):
        path = VajraTokenizer.detect_path({"type": "evtx"})
        assert path == TokenizationPath.A

    def test_routing_stix_defaults_to_a(self):
        path = VajraTokenizer.detect_path({"type": "stix"})
        assert path == TokenizationPath.A

    def test_routing_mitre(self):
        path = VajraTokenizer.detect_path({"type": "technique"})
        assert path == TokenizationPath.C_MITRE

    def test_encode_text_returns_ids(self):
        tok = VajraTokenizer(CFG)
        ids = tok.encode_text("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_sentinel_id_accessible(self):
        tok = VajraTokenizer(CFG)
        sid = tok.sentinel_id("<|S_NULL|>")
        assert isinstance(sid, int)
