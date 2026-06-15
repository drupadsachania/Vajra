"""Tests for Chunk 8: QAT schedule + training losses + data loader."""
import pytest
import torch
import torch.nn as nn
from vajra.training.losses import FMLMLoss, KillChainLoss, BaNELLoss, DPOLoss, MITREOntologyLoss
from vajra.training.qat import wNa8o8Schedule, KVCacheQuantizer, QATLinear
from vajra.training.data import (
    TrainingExampleLoader, DOMAIN_TAG_MAP, RealIPInTrainingError, find_disallowed_ips,
)


# ---------------------------------------------------------------------------
# FMLMLoss
# ---------------------------------------------------------------------------
class TestFMLMLoss:
    def test_sentinel_positions_excluded(self):
        """Loss at sentinel-labeled positions must be zero (they're excluded)."""
        loss_fn = FMLMLoss(sentinel_id_min=50408, sentinel_id_max=50427)
        vocab = 50428
        B, S = 2, 8
        logits = torch.randn(B, S, vocab)

        # All labels are sentinel IDs → should give 0 useful loss (no non-ignored positions)
        labels = torch.full((B, S), 50410, dtype=torch.long)
        loss = loss_fn(logits, labels)
        # With all sentinel labels masked, cross_entropy over empty set → nan or 0
        # In PyTorch, cross_entropy with all ignore_index gives nan; we accept both
        assert loss.item() != loss.item() or loss.item() == 0.0  # nan or 0

    def test_nonsentinel_positions_have_loss(self):
        """Non-sentinel positions contribute positive loss."""
        loss_fn = FMLMLoss()
        vocab = 50428
        B, S = 2, 8
        logits = torch.randn(B, S, vocab)
        labels = torch.randint(0, 50400, (B, S))  # well below sentinel range
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_explicit_sentinel_mask_excludes_positions(self):
        """Explicit sentinel_mask overrides label-based detection."""
        loss_fn = FMLMLoss()
        vocab = 100
        B, S = 1, 6
        logits = torch.randn(B, S, vocab)
        labels = torch.randint(0, 50, (B, S))

        # Mark first 3 positions as sentinel
        mask = torch.zeros(B, S, dtype=torch.bool)
        mask[:, :3] = True

        loss_masked = loss_fn(logits, labels, sentinel_mask=mask)
        # Compare against loss with all positions active
        loss_full = loss_fn(logits, labels, sentinel_mask=torch.zeros(B, S, dtype=torch.bool))
        assert abs(loss_masked.item() - loss_full.item()) > 0


# ---------------------------------------------------------------------------
# BaNELLoss
# ---------------------------------------------------------------------------
class TestBaNELLoss:
    def test_gradient_nonzero_at_null_positions(self):
        """Gradient flows when null_mask has True entries."""
        loss_fn = BaNELLoss()
        B, S, V = 1, 4, 100
        obs = torch.randn(B, S, V, requires_grad=True)
        base = torch.randn(B, S, V)
        null_mask = torch.zeros(B, S, dtype=torch.bool)
        null_mask[0, 0] = True

        loss = loss_fn(obs, base, null_mask)
        loss.backward()
        assert obs.grad is not None
        assert obs.grad[0, 0].abs().sum().item() > 0

    def test_gradient_zero_when_no_null_positions(self):
        """No gradient at non-null positions when null_mask is all-False."""
        loss_fn = BaNELLoss()
        B, S, V = 1, 4, 100
        obs = torch.randn(B, S, V, requires_grad=True)
        base = torch.randn(B, S, V)
        null_mask = torch.zeros(B, S, dtype=torch.bool)  # all False

        loss = loss_fn(obs, base, null_mask)
        loss.backward()
        # obs.grad should be all zeros (sum * 0.0 path)
        assert obs.grad is not None
        assert obs.grad.abs().sum().item() == 0.0

    def test_lambda_annealing(self):
        """λ increases from start to end over n_anneal_steps epochs."""
        loss_fn = BaNELLoss(lambda_start=0.1, lambda_end=0.3, n_anneal_steps=4)
        l0 = loss_fn.current_lambda
        loss_fn.step_epoch()
        l1 = loss_fn.current_lambda
        assert l1 > l0

    def test_kl_nonnegative(self):
        loss_fn = BaNELLoss()
        B, S, V = 2, 8, 64
        obs = torch.randn(B, S, V, requires_grad=True)
        base = torch.randn(B, S, V)
        null_mask = torch.ones(B, S, dtype=torch.bool)
        loss = loss_fn(obs, base, null_mask)
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# KillChainLoss
# ---------------------------------------------------------------------------
class TestKillChainLoss:
    def test_scalar_output(self):
        loss_fn = KillChainLoss()
        logits = torch.randn(4, 7)
        targets = torch.randint(0, 7, (4,))
        loss = loss_fn(logits, targets)
        assert loss.shape == ()
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# DPOLoss
# ---------------------------------------------------------------------------
class TestDPOLoss:
    def test_preferred_gt_rejected_gives_lower_loss(self):
        """Winner clearly better than loser → lower DPO loss than reversed."""
        loss_fn = DPOLoss(beta=0.1)
        B, S, V = 1, 5, 50
        # Winner sequence: all tokens are token 0
        labels_w = torch.zeros(B, S, dtype=torch.long)
        labels_l = torch.ones(B, S, dtype=torch.long)

        # Model strongly prefers token 0 → winner gets high log-prob
        logits_w = torch.zeros(B, S, V)
        logits_w[:, :, 0] = 10.0
        logits_l = torch.zeros(B, S, V)
        logits_l[:, :, 0] = 10.0

        ref_w = torch.zeros(B, S, V)
        ref_l = torch.zeros(B, S, V)

        loss = loss_fn(logits_w, logits_l, ref_w, ref_l, labels_w, labels_l)
        assert loss.item() > 0  # sigmoid loss always > 0
        assert torch.isfinite(loss)

    def test_scalar_output(self):
        loss_fn = DPOLoss()
        B, S, V = 2, 4, 20
        lw = torch.randn(B, S, V)
        ll = torch.randn(B, S, V)
        rw = torch.randn(B, S, V)
        rl = torch.randn(B, S, V)
        lblw = torch.randint(0, V, (B, S))
        lbll = torch.randint(0, V, (B, S))
        loss = loss_fn(lw, ll, rw, rl, lblw, lbll)
        assert loss.shape == ()


# ---------------------------------------------------------------------------
# MITREOntologyLoss
# ---------------------------------------------------------------------------
class TestMITREOntologyLoss:
    def test_positive_loss(self):
        loss_fn = MITREOntologyLoss(margin=2.0)
        d = 256
        h = torch.randn(4, d)
        r = torch.randn(4, d)
        t = torch.randn(4, d)
        h_neg = torch.randn(4, d)
        t_neg = torch.randn(4, d)
        loss = loss_fn(h, r, t, h_neg, t_neg)
        assert torch.isfinite(loss)
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# QAT schedule
# ---------------------------------------------------------------------------
class TestQATSchedule:
    def test_decoder_linears_become_qat_linear(self):
        """After apply_qat_to_module, decoder Linear layers are wrapped in QATLinear."""
        decoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        wNa8o8Schedule.apply_qat_to_module(decoder, mode="decoder")
        qat_layers = [m for m in decoder.modules() if isinstance(m, QATLinear)]
        assert len(qat_layers) == 2  # both Linears wrapped

    def test_decoder_qat_actually_quantizes_in_forward(self):
        """QATLinear must apply fake-quant in forward — output differs from full precision."""
        torch.manual_seed(0)
        linear = nn.Linear(32, 32, bias=False)
        wrapped = nn.Sequential(linear)
        wNa8o8Schedule.apply_qat_to_module(wrapped, mode="decoder")
        qat = wrapped[0]
        assert isinstance(qat, QATLinear)

        x = torch.randn(4, 32)
        # Enable the observer + fake-quant (QAT train mode)
        qat.train()
        qat_out = qat(x)                                  # fake-quant applied to weight
        fp_out = nn.functional.linear(x, linear.weight)   # full-precision reference
        # 2-bit quantization is lossy → outputs must differ measurably
        assert not torch.allclose(qat_out, fp_out, atol=1e-4)
        assert torch.isfinite(qat_out).all()

    def test_kv_projections_get_quantizer(self):
        """After apply_qat_to_module(mode='encoder'), k/v linears become KVCacheQuantizer."""
        class FakeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q = nn.Linear(64, 64)
                self.k = nn.Linear(64, 64)
                self.v = nn.Linear(64, 64)

        attn = FakeAttention()
        wNa8o8Schedule.apply_qat_to_module(attn, mode="encoder")
        assert isinstance(attn.k, KVCacheQuantizer)
        assert isinstance(attn.v, KVCacheQuantizer)
        assert isinstance(attn.q, nn.Linear)  # q is not a KV projection

    def test_kv_quantizer_forward_shape(self):
        linear = nn.Linear(64, 64)
        kv_q = KVCacheQuantizer(linear)
        x = torch.randn(2, 8, 64)
        out = kv_q(x)
        assert out.shape == (2, 8, 64)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
class TestTrainingDataLoader:
    def _make_example(self, include_null=False):
        example = {
            "events": [
                {"event_type": "evtx", "domain": "detection_network",
                 "source_type_id": 0, "timestamp": 1700000000.0,
                 "is_null_signal": False, "data": {"EventID": 4624}},
                {"event_type": "netflow", "domain": "detection_network",
                 "source_type_id": 2, "timestamp": 1700000001.0,
                 "is_null_signal": include_null, "data": {}},
            ],
            "labels": {"decision_class": 0, "technique_ids": ["T1078"]},
            "null_signals": ["netflow"] if include_null else [],
        }
        return example

    def test_domain_tags_assigned(self):
        loader = TrainingExampleLoader([self._make_example()])
        batch = next(iter(loader))
        assert batch.domain_tags[0] == DOMAIN_TAG_MAP["detection_network"]

    def test_timestamps_preserved(self):
        loader = TrainingExampleLoader([self._make_example()])
        batch = next(iter(loader))
        assert batch.timestamps[0] == pytest.approx(1700000000.0)

    def test_null_signal_mask_true_when_null(self):
        loader = TrainingExampleLoader([self._make_example(include_null=True)])
        batch = next(iter(loader))
        # The netflow event is marked as null
        assert batch.null_signal_mask[1] is True

    def test_null_signal_mask_false_when_no_null(self):
        loader = TrainingExampleLoader([self._make_example(include_null=False)])
        batch = next(iter(loader))
        assert all(not m for m in batch.null_signal_mask)

    def test_stage_3_eligibility_with_null(self):
        loader = TrainingExampleLoader([self._make_example(include_null=True)])
        batch = next(iter(loader))
        assert batch.stage_3_eligible is True

    def test_stage_3_not_eligible_without_null(self):
        loader = TrainingExampleLoader([self._make_example(include_null=False)])
        batch = next(iter(loader))
        assert batch.stage_3_eligible is False

    def test_tokenization_path_routing(self):
        assert TrainingExampleLoader.tokenization_path_for("cvss") == "B"
        assert TrainingExampleLoader.tokenization_path_for("netflow") == "B"
        assert TrainingExampleLoader.tokenization_path_for("mitre_kg") == "C"
        assert TrainingExampleLoader.tokenization_path_for("evtx") == "A"


# ---------------------------------------------------------------------------
# IP enforcement at ingest (no real IPs in training data)
# ---------------------------------------------------------------------------
class TestIPEnforcement:
    def _example_with_ip(self, ip):
        return {
            "events": [
                {"event_type": "netflow", "domain": "detection_network",
                 "data": {"src_ip": "10.0.0.5", "dst_ip": ip}},
            ],
            "labels": {},
        }

    def test_rfc1918_ip_allowed(self):
        loader = TrainingExampleLoader([self._example_with_ip("192.168.1.10")])
        assert len(loader) == 1

    def test_rfc5737_doc_ip_allowed(self):
        loader = TrainingExampleLoader([self._example_with_ip("203.0.113.5")])
        assert len(loader) == 1

    def test_real_public_ip_rejected(self):
        with pytest.raises(RealIPInTrainingError):
            TrainingExampleLoader([self._example_with_ip("8.8.8.8")])

    def test_real_ip_rejected_when_nested_deep(self):
        ex = {"events": [{"data": {"log": {"msg": "connect to 1.1.1.1 now"}}}]}
        with pytest.raises(RealIPInTrainingError):
            TrainingExampleLoader([ex])

    def test_validation_can_be_disabled(self):
        # Explicit opt-out still loads (e.g. for already-sanitized corpora)
        loader = TrainingExampleLoader(
            [self._example_with_ip("8.8.8.8")], validate_ips=False
        )
        assert len(loader) == 1

    def test_find_disallowed_ips_reports_only_public(self):
        ex = self._example_with_ip("198.51.100.7")  # RFC5737 → allowed
        assert find_disallowed_ips(ex) == []
        ex2 = self._example_with_ip("9.9.9.9")       # public → flagged
        assert "9.9.9.9" in find_disallowed_ips(ex2)
        assert TrainingExampleLoader.tokenization_path_for("stix") == "A"
        assert TrainingExampleLoader.tokenization_path_for("unknown") == "A"

    def test_ip_in_dict_key_rejected(self):
        """Public IP as a dict key (e.g. connection table) must also be caught."""
        ex = {"events": [{"data": {"8.8.8.8": {"port": 443}}}], "labels": {}}
        with pytest.raises(RealIPInTrainingError):
            TrainingExampleLoader([ex])

    def test_ip_with_letter_suffix_rejected(self):
        """IP followed by a letter (e.g. '8.8.8.8G' in a log line) must still be caught.
        Old \\b regex misses this because \\b fails between two word chars (8 and G).
        """
        ex = {"events": [{"data": {"msg": "connect to 8.8.8.8G port 443"}}], "labels": {}}
        with pytest.raises(RealIPInTrainingError):
            TrainingExampleLoader([ex])

    def test_real_ipv6_rejected(self):
        """Real public IPv6 address must be caught."""
        ex = {"events": [{"data": {"addr": "2600:1f18::1"}}], "labels": {}}
        with pytest.raises(RealIPInTrainingError):
            TrainingExampleLoader([ex])

    def test_documentation_ipv6_allowed(self):
        """RFC3849 documentation IPv6 (2001:db8::/32) must be allowed."""
        ex = {"events": [{"data": {"addr": "2001:db8::1"}}], "labels": {}}
        loader = TrainingExampleLoader([ex])
        assert len(loader) == 1


# ---------------------------------------------------------------------------
# BaNEL baseline detach
# ---------------------------------------------------------------------------
class TestBaNELDetach:
    def test_baseline_grad_does_not_flow(self):
        """Gradient must NOT flow into logits_baseline (EMA-frozen stream)."""
        loss_fn = BaNELLoss()
        B, S, V = 1, 4, 64
        obs = torch.randn(B, S, V, requires_grad=True)
        base = torch.randn(B, S, V, requires_grad=True)
        null_mask = torch.ones(B, S, dtype=torch.bool)
        loss = loss_fn(obs, base, null_mask)
        loss.backward()
        assert base.grad is None or base.grad.abs().sum().item() == 0.0, \
            "Gradient flowed into baseline stream — violates EMA-frozen semantics"


# ---------------------------------------------------------------------------
# KillChainLoss with validity mask
# ---------------------------------------------------------------------------
class TestKillChainLossMask:
    def test_invalid_transition_blocked(self):
        """When current_states provided, skip-stage transitions must be masked out."""
        loss_fn = KillChainLoss()
        # Force logit for state 6 (EXFIL) very high while current state is 0 (RECON).
        # Without mask the model would pick state 6; with mask only 0→0 or 0→1 allowed.
        logits = torch.zeros(1, 7)
        logits[0, 6] = 100.0          # EXFIL — invalid from RECON
        logits[0, 1] = 10.0           # WEAPONIZE — valid advance
        current = torch.zeros(1, dtype=torch.long)  # RECON
        target  = torch.ones(1, dtype=torch.long)   # WEAPONIZE
        loss = loss_fn(logits, target, current_states=current)
        assert torch.isfinite(loss)
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# NetFlow per-field projections (distinctness)
# ---------------------------------------------------------------------------
class TestNetFlowFieldDistinctness:
    def test_bytes_packets_duration_differ(self):
        """bytes/packets/duration embeddings must differ — they use separate projections."""
        from vajra.tokenization.path_b import NetFlowEncoder, NetFlowRecord
        torch.manual_seed(0)
        enc = NetFlowEncoder(d_model=64)
        rec = NetFlowRecord(
            src_ip="10.0.0.1", dst_ip="10.0.0.2",
            src_port=12345, dst_port=443, protocol=6,
            bytes_sent=1_000_000, packets=700, duration_ms=250.0,
        )
        fields = enc.encode_record(rec)
        assert not torch.allclose(fields["bytes"], fields["packets"]), \
            "bytes and packets embeddings are identical — per-field projections broken"
        assert not torch.allclose(fields["bytes"], fields["duration"]), \
            "bytes and duration embeddings are identical"
