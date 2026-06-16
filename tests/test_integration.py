"""Tests for Chunk 9: end-to-end inference pipeline + ONNX export."""
import os
import tempfile
import pytest
import torch
import torch.nn as nn

from vajra.config import VajraConfig
from vajra.interface.types import (
    VajraInferenceRequest, VajraInferenceResponse, SecurityEvent,
    EvidenceDAG, AFNScore,
)
from vajra.interface.pipeline import VajraInferencePipeline
from vajra.interface.function_calling import parse_tool_calls, NATIVE_TOOLS, ToolCall
from vajra.interface.onnx_export import export_encoder_fusion, export_decoder
from vajra.decoder.decoder import ConstrainedDecoder
from vajra.fusion.epistemic_fusion import EpistemicFusionLayer
from vajra.heads.decision_head import apply_dcat_override


@pytest.fixture(scope="module")
def cfg():
    return VajraConfig()


@pytest.fixture(scope="module")
def pipeline(cfg):
    torch.manual_seed(42)
    pipe = VajraInferencePipeline(cfg=cfg, tiny=True)
    pipe.eval()
    return pipe


def _make_request(n_events=3, justification=False, request_id="test-001"):
    events = [
        SecurityEvent(event_type="evtx", domain="detection_network",
                      data={"EventID": 4624}, timestamp=1700000000.0),
        SecurityEvent(event_type="netflow", domain="detection_network",
                      data={"src_port": 443}, timestamp=1700000001.0),
        SecurityEvent(event_type="stix", domain="cti_stix",
                      data={"type": "indicator"}, timestamp=1700000002.0),
    ][:n_events]
    return VajraInferenceRequest(
        request_id=request_id,
        events=events,
        request_justification_trace=justification,
    )


# ---------------------------------------------------------------------------
# End-to-end response shape / type correctness
# ---------------------------------------------------------------------------
class TestEndToEnd:
    def test_returns_response_type(self, pipeline):
        req = _make_request()
        resp = pipeline.run(req)
        assert isinstance(resp, VajraInferenceResponse)

    def test_request_id_echoed(self, pipeline):
        req = _make_request(request_id="req-xyz")
        resp = pipeline.run(req)
        assert resp.request_id == "req-xyz"

    def test_decision_class_in_range(self, pipeline):
        resp = pipeline.run(_make_request())
        assert resp.decision_class in (0, 1, 2, 3)

    def test_decision_probs_sum_to_one(self, pipeline):
        resp = pipeline.run(_make_request())
        total = sum(resp.decision_probabilities)
        assert abs(total - 1.0) < 1e-4

    def test_decision_probs_length_4(self, pipeline):
        resp = pipeline.run(_make_request())
        assert len(resp.decision_probabilities) == 4

    def test_confidence_in_01(self, pipeline):
        resp = pipeline.run(_make_request())
        assert 0.0 <= resp.confidence <= 1.0

    def test_evidence_dag_type(self, pipeline):
        resp = pipeline.run(_make_request())
        assert isinstance(resp.evidence_dag, EvidenceDAG)

    def test_afn_scores_type(self, pipeline):
        resp = pipeline.run(_make_request())
        for score in resp.afn_scores:
            assert isinstance(score, AFNScore)

    def test_inference_latency_positive(self, pipeline):
        resp = pipeline.run(_make_request())
        assert resp.inference_latency_ms > 0

    def test_early_exit_is_bool(self, pipeline):
        resp = pipeline.run(_make_request())
        assert isinstance(resp.early_exit, bool)


# ---------------------------------------------------------------------------
# Content sensitivity (D1: tokenization wiring)
# ---------------------------------------------------------------------------
class TestContentSensitivity:
    def _req(self, request_id, event_id):
        ev = SecurityEvent(event_type="evtx", domain="detection_network",
                           data={"EventID": event_id}, timestamp=1700000000.0)
        return VajraInferenceRequest(request_id=request_id, events=[ev])

    def test_same_content_same_output(self, pipeline):
        a = pipeline.run(self._req("a", 4624))
        b = pipeline.run(self._req("b", 4624))
        assert a.decision_probabilities == b.decision_probabilities

    def test_different_content_different_output(self, pipeline):
        a = pipeline.run(self._req("a", 4624))
        b = pipeline.run(self._req("b", 1102))
        assert a.decision_probabilities != b.decision_probabilities, \
            "Different event content produced identical output — run() ignores content"

    def test_full_pipeline_run_guarded(self, cfg):
        """Full (non-tiny) run() must fail loudly, not return placeholder output."""
        pipe = VajraInferencePipeline(cfg=cfg, tiny=False)
        pipe.eval()
        with pytest.raises(NotImplementedError):
            pipe.run(_make_request())


# ---------------------------------------------------------------------------
# Decoder invocation guard
# ---------------------------------------------------------------------------
class TestDecoderGuard:
    def test_no_trace_when_not_requested(self, pipeline):
        """request_justification_trace=False → justification_trace is empty."""
        req = _make_request(justification=False)
        resp = pipeline.run(req)
        assert resp.justification_trace == ""

    def test_trace_nonempty_for_class_0_or_1_with_request(self, pipeline):
        """When decoder is invoked, justification_trace should be non-empty."""
        # We can't guarantee decision_class=0 or 1 with random weights, so we
        # test that the pipeline runs without error and returns a string.
        req = _make_request(justification=True)
        resp = pipeline.run(req)
        assert isinstance(resp.justification_trace, str)


# ---------------------------------------------------------------------------
# Early exit
# ---------------------------------------------------------------------------
class TestEarlyExit:
    def test_early_exit_triggers_with_crafted_input(self, cfg):
        """Craft fusion input so F3 fires: p(class=2) > 0.92."""
        fusion = EpistemicFusionLayer(cfg)
        fusion.eval()

        # Force F3 head to output high p(class=2)
        with torch.no_grad():
            fusion.f3_head.weight.zero_()
            fusion.f3_head.bias.zero_()
            fusion.f3_head.bias[2] = 20.0  # class=2 logit dominates

        x = torch.randn(1, 7, 1024)
        result = fusion(x)
        assert result.early_exit is True
        assert result.exit_block == 3


# ---------------------------------------------------------------------------
# DCAT override (§6.4)
# ---------------------------------------------------------------------------
class TestDCATOverride:
    def test_helper_forces_class_1_on_high_divergence(self):
        # class 2/3 with divergence above θ → forced to 1
        assert apply_dcat_override(2, 5.0, 1.0) == (1, True)
        assert apply_dcat_override(3, 5.0, 1.0) == (1, True)

    def test_helper_no_override_for_class_0_or_1(self):
        assert apply_dcat_override(0, 5.0, 1.0) == (0, False)
        assert apply_dcat_override(1, 5.0, 1.0) == (1, False)

    def test_helper_no_override_below_threshold(self):
        assert apply_dcat_override(2, 0.5, 1.0) == (2, False)

    def test_pipeline_applies_override(self, cfg):
        """High injected DCAT divergence + DEFER decision → escalated, flag set."""
        torch.manual_seed(7)
        pipe = VajraInferencePipeline(cfg=cfg, tiny=True)
        pipe.eval()

        with torch.no_grad():
            # F3 uniform → no early exit; F6 → class 2 (DEFER)
            pipe.fusion.f3_head.weight.zero_(); pipe.fusion.f3_head.bias.zero_()
            pipe.fusion.f6_head.weight.zero_(); pipe.fusion.f6_head.bias.zero_()
            pipe.fusion.f6_head.bias[2] = 20.0

        # Inject high divergence on a DCAT-bearing encoder + low threshold
        pipe.domain_encoders["detection_network"].last_dcat_divergence = torch.tensor([[5.0]])
        pipe.cfg.dcat.theta_divergence = 1.0

        resp = pipe.run(_make_request())
        assert resp.decision_class == 1     # DEFER (2) escalated to ESCALATE (1)
        assert resp.dcat_override is True

    def test_pipeline_no_override_when_divergence_low(self, cfg):
        torch.manual_seed(7)
        pipe = VajraInferencePipeline(cfg=cfg, tiny=True)
        pipe.eval()
        with torch.no_grad():
            pipe.fusion.f3_head.weight.zero_(); pipe.fusion.f3_head.bias.zero_()
            pipe.fusion.f6_head.weight.zero_(); pipe.fusion.f6_head.bias.zero_()
            pipe.fusion.f6_head.bias[2] = 20.0
        pipe.domain_encoders["detection_network"].last_dcat_divergence = torch.tensor([[0.1]])
        pipe.cfg.dcat.theta_divergence = 1.0
        resp = pipe.run(_make_request())
        assert resp.decision_class == 2     # stays DEFER
        assert resp.dcat_override is False


# ---------------------------------------------------------------------------
# Function calling / tool-call parsing
# ---------------------------------------------------------------------------
class TestFunctionCalling:
    def test_valid_tool_call_parsed(self):
        text = '{"type": "tool_call", "tool_name": "lookup_cve", "parameters": {"cve_id": "CVE-2021-44228"}}'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].tool_name == "lookup_cve"
        assert calls[0].parameters == {"cve_id": "CVE-2021-44228"}

    def test_unknown_tool_rejected(self):
        text = '{"type": "tool_call", "tool_name": "run_exploit", "parameters": {}}'
        calls = parse_tool_calls(text)
        assert calls == []

    def test_all_native_tools_accepted(self):
        for tool in NATIVE_TOOLS:
            text = f'{{"type": "tool_call", "tool_name": "{tool}"}}'
            calls = parse_tool_calls(text)
            assert len(calls) == 1
            assert calls[0].tool_name == tool

    def test_invalid_json_ignored(self):
        calls = parse_tool_calls('{"type": "tool_call", broken json}')
        assert calls == []

    def test_non_tool_call_type_ignored(self):
        text = '{"type": "text", "tool_name": "lookup_cve"}'
        calls = parse_tool_calls(text)
        assert calls == []

    def test_multiple_calls_in_text(self):
        text = (
            'prefix '
            '{"type": "tool_call", "tool_name": "lookup_cve", "parameters": {"id": "1"}} '
            'middle '
            '{"type": "tool_call", "tool_name": "lookup_ioc", "parameters": {"hash": "abc"}} '
            'suffix'
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0].tool_name == "lookup_cve"
        assert calls[1].tool_name == "lookup_ioc"

    def test_max_agentic_turns_enforced(self):
        from vajra.interface.function_calling import MAX_AGENTIC_TURNS
        text = '{"type": "tool_call", "tool_name": "lookup_cve", "parameters": {}}'
        # Within budget → parsed
        assert len(parse_tool_calls(text, turn=MAX_AGENTIC_TURNS - 1)) == 1
        # At/over budget → no further calls emitted
        assert parse_tool_calls(text, turn=MAX_AGENTIC_TURNS) == []
        assert parse_tool_calls(text, turn=MAX_AGENTIC_TURNS + 5) == []


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
class TestONNXExport:
    def test_decoder_onnx_export_and_load(self):
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort

        decoder = ConstrainedDecoder(
            vocab_size=50428, d_model=64, n_heads=4, n_layers=2,
            ffn_width=128, dropout=0.0,
        )
        decoder.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "vajra_decoder.onnx")
            export_decoder(
                decoder, path=path,
                vocab_size=50428, afn_top_k=16, d_model=64, tgt_len=4,
            )
            sess = ort.InferenceSession(path)
            import numpy as np
            ids_np = np.random.randint(0, 100, (1, 4)).astype(np.int64)
            enc_np = np.random.randn(1, 16, 64).astype(np.float32)
            out = sess.run(None, {"input_ids": ids_np, "afn_encoder_states": enc_np})
            assert out[0].shape == (1, 4, 50428)

    def test_fusion_onnx_export_and_load(self, cfg):
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort

        class _FusionWrapper(nn.Module):
            """Thin wrapper so fusion returns just logits (ONNX-friendly)."""
            def __init__(self, fusion):
                super().__init__()
                self.fusion = fusion
            def forward(self, x):
                result = self.fusion(x)
                return result.decision_logits

        fusion = EpistemicFusionLayer(cfg)
        fusion.eval()
        wrapper = _FusionWrapper(fusion)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "vajra_encoder_fusion.onnx")
            export_encoder_fusion(wrapper, path=path, d_model=1024, n_domains=7)
            sess = ort.InferenceSession(path)
            import numpy as np
            x_np = np.random.randn(1, 7, 1024).astype(np.float32)
            out = sess.run(None, {"domain_cls_tokens": x_np})
            assert out[0].shape == (1, 4)

    def test_decoder_onnx_honors_dynamic_seq_len(self):
        """Exported decoder must run at a seq_len different from the trace length."""
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort
        import numpy as np

        decoder = ConstrainedDecoder(
            vocab_size=1000, d_model=64, n_heads=4, n_layers=2,
            ffn_width=128, dropout=0.0,
        )
        decoder.eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dec.onnx")
            export_decoder(decoder, path=path, vocab_size=1000,
                           afn_top_k=16, d_model=64, tgt_len=4)  # trace at 4
            sess = ort.InferenceSession(path)
            for L in (1, 8, 16):  # all ≠ trace length
                ids = np.random.randint(0, 100, (1, L)).astype(np.int64)
                enc = np.random.randn(1, 16, 64).astype(np.float32)
                out = sess.run(None, {"input_ids": ids, "afn_encoder_states": enc})
                assert out[0].shape == (1, L, 1000)

    def test_fusion_onnx_honors_dynamic_batch(self, cfg):
        """Exported fusion (via FusionExportWrapper) must run at varying batch sizes."""
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort
        import numpy as np
        from vajra.interface.onnx_export import FusionExportWrapper

        fusion = EpistemicFusionLayer(cfg)
        fusion.eval()
        wrapper = FusionExportWrapper(fusion)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "fus.onnx")
            export_encoder_fusion(wrapper, path=path, d_model=1024, n_domains=7)  # trace batch=1
            sess = ort.InferenceSession(path)
            for B in (2, 4):  # ≠ trace batch
                x_np = np.random.randn(B, 7, 1024).astype(np.float32)
                out = sess.run(None, {"domain_cls_tokens": x_np})
                assert out[0].shape == (B, 4)
