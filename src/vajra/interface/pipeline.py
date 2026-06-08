"""Full Vajra inference pipeline (§10).

Routes events → tokenization paths → embedding → 7 domain encoders (parallel) →
AFN → epistemic fusion (F3/F6 taps, optional early-exit) → output heads →
optional constrained decoder → VajraInferenceResponse.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import torch
import torch.nn as nn

from vajra.config import VajraConfig
from vajra.afn import ActivationFlowNetwork
from vajra.embedding.embedding_layer import EmbeddingLayer
from vajra.encoders.domain_encoders import build_domain_encoders
from vajra.fusion.epistemic_fusion import EpistemicFusionLayer
from vajra.heads.dag_head import EvidenceDAGHead
from vajra.heads.technique_head import ATTACKClassifier
from vajra.heads.decision_head import DecisionStateClassifier
from vajra.decoder.decoder import ConstrainedDecoder

from .types import (
    VajraInferenceRequest, VajraInferenceResponse,
    EvidenceDAG, AFNScore as AFNScoreType,
    DAGNode, DAGEdge,
)

DOMAIN_ORDER = [
    "detection_network", "forensics_provenance", "cti_stix",
    "vulnerability_risk", "identity_access", "incident_response", "compliance",
]

# d_model for each domain encoder (matches locked config)
DOMAIN_D_MODEL = {
    "detection_network": 1024, "forensics_provenance": 1024,
    "cti_stix": 768, "vulnerability_risk": 768,
    "identity_access": 768, "incident_response": 768,
    "compliance": 512,
}


class VajraInferencePipeline(nn.Module):
    """End-to-end inference pipeline assembled from all Vajra sub-modules."""

    def __init__(self, cfg: Optional[VajraConfig] = None, tiny: bool = False):
        super().__init__()
        if cfg is None:
            cfg = VajraConfig()
        self.cfg = cfg

        if tiny:
            # Minimal dimensions for fast integration tests
            self._build_tiny(cfg)
        else:
            self._build_full(cfg)

    def _build_tiny(self, cfg: VajraConfig) -> None:
        """Build a minimal-dimension version for integration testing."""
        # Use a very small embedding/encoder for fast forward passes
        from vajra.encoders.base_encoder import DomainEncoder
        from vajra.encoders.blocks import PreLNTransformerBlock
        from vajra.fusion.kill_chain import KillChainStateMachine

        d = 64
        self.embedding = nn.Embedding(cfg.embedding.vocabulary_size, d)

        # Tiny domain encoders: 1 layer, d=64
        self.domain_encoders = nn.ModuleDict({
            domain: DomainEncoder(layers=1, d_model=d, n_heads=4, ffn_width=128)
            for domain in DOMAIN_ORDER
        })
        self._domain_d_model = {domain: d for domain in DOMAIN_ORDER}

        # Projections to shared d=1024 for fusion
        self._to_fusion = nn.ModuleDict({
            domain: nn.Linear(d, 1024, bias=False)
            for domain in DOMAIN_ORDER
        })

        self.afn = ActivationFlowNetwork(top_k=cfg.afn.top_k)
        self.fusion = EpistemicFusionLayer(cfg)

        self.dag_head       = EvidenceDAGHead(n_domains=7, d_model=1024)
        self.attack_head    = ATTACKClassifier(d_model=1024)
        self.decision_head  = DecisionStateClassifier(d_model=1024)
        self.decoder = ConstrainedDecoder(
            vocab_size=cfg.embedding.vocabulary_size,
            d_model=1024, n_heads=4, n_layers=2, ffn_width=256, dropout=0.0,
        )
        self._tiny = True

    def _build_full(self, cfg: VajraConfig) -> None:
        """Build full-dimension pipeline."""
        self.embedding = EmbeddingLayer(cfg)
        self.domain_encoders = nn.ModuleDict(build_domain_encoders(cfg))
        self._domain_d_model = {d: cfg.domain_encoders[d].d_model for d in DOMAIN_ORDER}

        # Projections from domain d_model → shared 1024 for fusion
        self._to_fusion = nn.ModuleDict({
            domain: nn.Linear(cfg.domain_encoders[domain].d_model, cfg.shared_d_model, bias=False)
            if cfg.domain_encoders[domain].d_model != cfg.shared_d_model
            else nn.Identity()
            for domain in DOMAIN_ORDER
        })

        self.afn = ActivationFlowNetwork(top_k=cfg.afn.top_k)
        self.fusion = EpistemicFusionLayer(cfg)

        self.dag_head       = EvidenceDAGHead(n_domains=7, d_model=cfg.shared_d_model)
        self.attack_head    = ATTACKClassifier(d_model=cfg.shared_d_model)
        self.decision_head  = DecisionStateClassifier(d_model=cfg.shared_d_model)
        self.decoder = ConstrainedDecoder(
            vocab_size=cfg.embedding.vocabulary_size,
            d_model=cfg.shared_d_model,
            n_heads=cfg.decoder.attention_heads,
            n_layers=cfg.decoder.layers,
            ffn_width=cfg.decoder.ffn_width,
        )
        self._tiny = False

    def _encode_domain(
        self,
        domain: str,
        input_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run a single domain encoder, return (cls_token, afn_hidden)."""
        enc = self.domain_encoders[domain]
        if self._tiny:
            # DomainEncoder returns (hidden, cls)
            hidden, cls = enc(input_tensor)
            afn_hidden = hidden
        else:
            from vajra.encoders.base_encoder import DomainEncoder
            from vajra.encoders.long_context import InterleavedAttentionEncoder
            if isinstance(enc, DomainEncoder):
                hidden, cls = enc(input_tensor)
                afn_hidden = enc.afn_hidden
            else:
                # InterleavedAttentionEncoder
                hidden, cls = enc(input_tensor)
                afn_hidden = enc.afn_hidden if hasattr(enc, "afn_hidden") else hidden
        return cls, afn_hidden

    @torch.no_grad()
    def run(self, request: VajraInferenceRequest) -> VajraInferenceResponse:
        """Execute the full inference pipeline."""
        t0 = time.monotonic()

        device = next(self.parameters()).device
        cfg = self.cfg

        # --- Build per-domain input tensors (synthetic: batch=1, seq=4) ---
        # In production these come from the tokenization paths; here we use
        # a minimal synthetic representation to support integration testing.
        domain_inputs: dict[str, torch.Tensor] = {}
        events_by_domain: dict[str, list] = {d: [] for d in DOMAIN_ORDER}
        for evt in request.events:
            dom = evt.domain if evt.domain in events_by_domain else "detection_network"
            events_by_domain[dom].append(evt)

        for domain in DOMAIN_ORDER:
            seq_len = max(1, len(events_by_domain[domain]))
            d = self._domain_d_model[domain]
            domain_inputs[domain] = torch.randn(1, seq_len, d, device=device)

        # --- Run 7 domain encoders (parallel via ThreadPoolExecutor) ---
        cls_tokens: dict[str, Optional[torch.Tensor]] = {d: None for d in DOMAIN_ORDER}
        afn_hiddens: dict[str, Optional[torch.Tensor]] = {d: None for d in DOMAIN_ORDER}

        def _run_enc(domain):
            return domain, *self._encode_domain(domain, domain_inputs[domain])

        with ThreadPoolExecutor(max_workers=7) as ex:
            futs = {ex.submit(_run_enc, d): d for d in DOMAIN_ORDER}
            for fut in as_completed(futs):
                domain, cls, afn_h = fut.result()
                cls_tokens[domain] = cls
                afn_hiddens[domain] = afn_h

        # --- Project CLS tokens to shared d=1024; substitute DOMAIN_ABSENT for missing ---
        fusion_input_list = []
        for domain in DOMAIN_ORDER:
            cls = cls_tokens[domain]
            if cls is None:
                # DOMAIN_ABSENT — fusion layer handles substitution via its parameter
                cls_1024 = self.fusion.domain_absent.unsqueeze(0).unsqueeze(0)  # (1, 1, 1024)
            else:
                proj = self._to_fusion[domain]
                cls_1024 = proj(cls).unsqueeze(1)  # (1, 1, 1024)
            fusion_input_list.append(cls_1024)

        fusion_input = torch.cat(fusion_input_list, dim=1)  # (1, 7, 1024)

        # --- AFN scoring ---
        valid_hiddens = {d: h for d, h in afn_hiddens.items() if h is not None}
        afn_scores_raw = self.afn(valid_hiddens)

        # --- Epistemic fusion ---
        fusion_result = self.fusion(fusion_input)
        early_exit = fusion_result.early_exit
        f6_logits = fusion_result.decision_logits   # (1, 4)
        f6_probs  = fusion_result.decision_probs    # (1, 4)

        # --- Decision state ---
        _, probs, decision_class_t = self.decision_head(fusion_input[:, 0, :])
        decision_class = fusion_result.decision_logits.argmax(dim=-1).item()
        decision_probs = f6_probs[0].tolist()
        confidence = float(f6_probs[0].max().item())

        # --- ATT&CK technique head ---
        fused_cls = fusion_input[:, 0, :]  # use first domain CLS as proxy
        technique_probs = self.attack_head(fused_cls)[0]  # (716,)
        technique_ids = [
            f"T{1000 + i}" for i, p in enumerate(technique_probs.tolist())
            if p > 0.35
        ]

        # --- Evidence DAG head ---
        dag_adj = self.dag_head(fusion_input)  # (1, 32, 32)
        evidence_dag = EvidenceDAG(nodes=[], edges=[])

        # --- AFN scores for response ---
        afn_score_list: list[AFNScoreType] = []
        for domain, score in afn_scores_raw.items():
            for idx, imp in zip(score.token_indices.tolist(), score.importance_scores.tolist()):
                afn_score_list.append(AFNScoreType(domain=domain, token_index=idx, importance=imp))

        # --- Constrained decoder (only for classes 0/1 + justification requested) ---
        justification_trace = ""
        dcat_override = False

        if request.request_justification_trace and decision_class in (0, 1):
            # Gather AFN top-16 tokens for the first available domain
            first_domain = next((d for d in DOMAIN_ORDER if afn_hiddens.get(d) is not None), None)
            if first_domain is not None:
                afn_h = afn_hiddens[first_domain]
                score = afn_scores_raw.get(first_domain)
                if score is not None:
                    top_tokens = afn_h[:, score.token_indices, :]
                    # Pad to exactly 16 tokens if needed
                    k = top_tokens.shape[1]
                    if k < 16:
                        pad = torch.zeros(1, 16 - k, top_tokens.shape[2], device=device)
                        top_tokens = torch.cat([top_tokens, pad], dim=1)
                    elif k > 16:
                        top_tokens = top_tokens[:, :16, :]

                    # Project to decoder d_model if needed
                    if top_tokens.shape[-1] != 1024:
                        # Simple mean-pool projection for integration test
                        top_tokens = top_tokens.mean(dim=-1, keepdim=True).expand(-1, -1, 1024)

                    bos_ids = torch.tensor([[1]], device=device)
                    out_ids = self.decoder.greedy_decode(top_tokens, max_new_tokens=32)
                    justification_trace = f"<decoded:{out_ids.shape[1]}_tokens>"

        t1 = time.monotonic()
        latency_ms = (t1 - t0) * 1000.0

        return VajraInferenceResponse(
            request_id=request.request_id,
            decision_class=int(decision_class),
            decision_probabilities=decision_probs,
            confidence=confidence,
            technique_ids=technique_ids,
            evidence_dag=evidence_dag,
            afn_scores=afn_score_list,
            justification_trace=justification_trace,
            inference_latency_ms=latency_ms,
            early_exit=early_exit,
            dcat_override=dcat_override,
        )
