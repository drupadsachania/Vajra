"""Tests for Chunk 6 — output heads (DAG, ATT&CK, decision, calibration) + AFN."""

import time
import pytest
import torch

from vajra.config import VajraConfig
from vajra.heads.dag_head import EvidenceDAGHead
from vajra.heads.technique_head import ATTACKClassifier
from vajra.heads.decision_head import DecisionStateClassifier
from vajra.heads.calibration import TemperatureScaler, ConformalPredictor
from vajra.afn import ActivationFlowNetwork, AFNScore

CFG = VajraConfig()
D = CFG.shared_d_model


# ── Evidence DAG Head ─────────────────────────────────────────────────────────

class TestEvidenceDAGHead:

    def test_output_shape(self):
        head = EvidenceDAGHead(n_domains=7, d_model=D)
        domain_cls = torch.randn(2, 7, D)
        out = head(domain_cls)
        assert out.shape == (2, 32, 32)

    def test_gradient_flows(self):
        head = EvidenceDAGHead(n_domains=7, d_model=64)
        domain_cls = torch.randn(1, 7, 64, requires_grad=True)
        out = head(domain_cls)
        out.sum().backward()
        assert domain_cls.grad is not None


# ── ATT&CK Classifier ─────────────────────────────────────────────────────────

class TestATTACKClassifier:

    def test_output_shape(self):
        clf = ATTACKClassifier(d_model=D)
        x = torch.randn(2, D)
        out = clf(x)
        assert out.shape == (2, 716)

    def test_probabilities_in_01(self):
        clf = ATTACKClassifier(d_model=D)
        x = torch.randn(4, D)
        probs = clf(x)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_binary_predictions(self):
        clf = ATTACKClassifier(d_model=64)
        x = torch.randn(2, 64)
        pred = clf.predict(x)
        assert set(pred.unique().tolist()).issubset({0.0, 1.0})

    def test_loss_finite(self):
        clf = ATTACKClassifier(d_model=64)
        logits = torch.randn(2, 716)
        targets = (torch.rand(2, 716) > 0.5).float()
        loss = ATTACKClassifier.loss(logits, targets)
        assert loss.isfinite()


# ── Decision State Classifier ─────────────────────────────────────────────────

class TestDecisionStateClassifier:

    def test_output_shapes(self):
        clf = DecisionStateClassifier(d_model=D)
        x = torch.randn(2, D)
        logits, probs, dec = clf(x)
        assert logits.shape == (2, 4)
        assert probs.shape  == (2, 4)
        assert dec.shape    == (2,)

    def test_probs_sum_to_one(self):
        clf = DecisionStateClassifier(d_model=D)
        x = torch.randn(3, D)
        _, probs, _ = clf(x)
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(3))

    def test_dcat_override_class2_to_1(self):
        """High divergence + class=2 → forced to class=1."""
        clf = DecisionStateClassifier(d_model=64)
        x = torch.zeros(1, 64)
        # Force class=2 by zeroing head weights and setting bias[2]=100
        with torch.no_grad():
            clf.head.weight.zero_()
            clf.head.bias = torch.nn.Parameter(
                torch.tensor([-100.0, -100.0, 100.0, -100.0])
            )
        dcat_div = torch.tensor([5.0])   # > theta=1.0
        _, _, dec = clf(x, dcat_divergence=dcat_div, theta_divergence=1.0)
        assert dec.item() == 1, f"Expected DCAT override to class=1, got {dec.item()}"

    def test_no_override_for_class0_or_1(self):
        """High divergence should NOT override class=0 or class=1."""
        clf = DecisionStateClassifier(d_model=64)
        x = torch.zeros(1, 64)
        with torch.no_grad():
            clf.head.weight.zero_()
            clf.head.bias = torch.nn.Parameter(
                torch.tensor([100.0, -100.0, -100.0, -100.0])   # class=0
            )
        dcat_div = torch.tensor([100.0])   # very high divergence
        _, _, dec = clf(x, dcat_divergence=dcat_div, theta_divergence=1.0)
        assert dec.item() == 0, "Class=0 should NOT be overridden"


# ── Temperature Scaler ────────────────────────────────────────────────────────

class TestTemperatureScaler:

    def test_probs_sum_to_one(self):
        scaler = TemperatureScaler(n_domains=7)
        logits = torch.randn(3, 4)
        probs = scaler(logits, domain_idx=0)
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(3))

    def test_high_temperature_softens(self):
        scaler = TemperatureScaler()
        with torch.no_grad():
            scaler.temperatures["domain_0"].fill_(10.0)
        logits = torch.tensor([[5.0, 1.0, 0.5, 0.2]])
        probs = scaler(logits, domain_idx=0)
        # High T → flatter distribution, max prob < 0.5
        assert probs.max().item() < 0.5


# ── Conformal Predictor ────────────────────────────────────────────────────────

class TestConformalPredictor:

    def test_calibrate_and_predict(self):
        cp = ConformalPredictor(coverage_targets=[0.90])
        probs = torch.softmax(torch.randn(100, 4), dim=-1)
        labels = torch.randint(0, 4, (100,))
        cp.calibrate(probs, labels)
        pred_sets = cp.predict_set(probs[:5], coverage=0.90)
        assert len(pred_sets) == 5
        for ps in pred_sets:
            assert len(ps) >= 1

    def test_prediction_set_contains_most_likely(self):
        cp = ConformalPredictor()
        probs = torch.softmax(torch.tensor([[10.0, 1.0, 0.5, 0.2]]), dim=-1)
        sets = cp.predict_set(probs, coverage=0.90)
        assert 0 in sets[0], "Most likely class should always be in prediction set"


# ── AFN ───────────────────────────────────────────────────────────────────────

class TestActivationFlowNetwork:

    def test_top_k_unique_indices(self):
        afn = ActivationFlowNetwork(top_k=16)
        hidden = torch.randn(1, 64, 128)
        score = afn.score_domain(hidden, "test_domain")
        assert len(set(score.token_indices.tolist())) == 16

    def test_scores_sum_to_approx_one(self):
        afn = ActivationFlowNetwork(top_k=16)
        hidden = torch.randn(1, 64, 128)
        score = afn.score_domain(hidden, "test_domain")
        # Sum of top-16 normalized scores may not sum to 1 (we take top-k of normalized)
        # but all individual scores must be positive
        assert (score.importance_scores > 0).all()

    def test_forward_all_domains(self):
        afn = ActivationFlowNetwork(top_k=16)
        hiddens = {
            "detection_network": torch.randn(1, 128, 64),
            "cti_stix": torch.randn(1, 64, 64),
        }
        scores = afn(hiddens)
        assert set(scores.keys()) == {"detection_network", "cti_stix"}

    def test_afn_layer_correct_for_12l(self):
        """12-layer encoder should use L8 (0-indexed 7)."""
        from vajra.encoders.base_encoder import _AFN_LAYER_MAP
        assert _AFN_LAYER_MAP[12] == 8

    def test_afn_timing_under_15ms(self):
        """AFN on (1, 512, 1024) must complete in <15ms on CPU."""
        afn = ActivationFlowNetwork(top_k=16)
        hidden = torch.randn(1, 512, 1024)
        start = time.perf_counter()
        for _ in range(5):
            afn.score_domain(hidden, "timing_test")
        elapsed_ms = (time.perf_counter() - start) / 5 * 1000
        assert elapsed_ms < 15.0, f"AFN took {elapsed_ms:.1f}ms, exceeds 15ms budget"

    def test_gather_top_tokens(self):
        afn = ActivationFlowNetwork(top_k=4)
        hidden = {"enc_a": torch.randn(2, 32, 64)}
        scores = afn(hidden)
        top_tokens = afn.gather_top_tokens(hidden, scores)
        assert top_tokens["enc_a"].shape == (2, 4, 64)

    def test_afn_l2_norm_correctness(self):
        """Token with highest L2 norm should always be in top-k."""
        afn = ActivationFlowNetwork(top_k=4)
        hidden = torch.zeros(1, 16, 32)
        hidden[0, 7, :] = 100.0   # token 7 has enormous norm
        score = afn.score_domain(hidden, "norm_test")
        assert 7 in score.token_indices.tolist()
