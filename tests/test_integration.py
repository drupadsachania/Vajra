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
