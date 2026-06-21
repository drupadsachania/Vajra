"""Full Vajra inference pipeline (§10).

Routes events → tokenization paths → embedding → 7 domain encoders (parallel) →
AFN → epistemic fusion (F3/F6 taps, optional early-exit) → output heads →
optional constrained decoder → VajraInferenceResponse.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import torch
import torch.nn as nn

from vajra.afn import ActivationFlowNetwork
from vajra.config import VajraConfig
from vajra.decoder.decoder import ConstrainedDecoder
from vajra.embedding.embedding_layer import EmbeddingLayer
from vajra.encoders.domain_encoders import build_domain_encoders
from vajra.fusion.epistemic_fusion import EpistemicFusionLayer
from vajra.heads.dag_head import EvidenceDAGHead
from vajra.heads.decision_head import apply_dcat_override
from vajra.heads.technique_head import ATTACKClassifier

from .types import AFNScore as AFNScoreType
from .types import (DAGEdge, DAGNode, EvidenceDAG, VajraInferenceRequest,
                    VajraInferenceResponse)

DOMAIN_ORDER = [
    "detection_network",
    "forensics_provenance",
    "cti_stix",
    "vulnerability_risk",
    "identity_access",
    "incident_response",
    "compliance",
]


def _event_token_ids(events: list, vocab_size: int, device) -> "torch.Tensor":
    """Deterministically map events to token IDs so inference is content-sensitive.

    This is a content-hash placeholder for the real S-TOON / numerical / graph
    tokenization paths: identical events always yield identical IDs (and thus
    identical model output), while different event content yields different IDs.
    Empty domains get a single constant 'absent' token.
    """
    if not events:
        return torch.zeros(1, 1, dtype=torch.long, device=device)
    ids = []
    for evt in events:
        key = f"{evt.event_type}|{evt.domain}|{repr(sorted(evt.data.items(), key=lambda kv: kv[0]))}"
        ids.append(int(hashlib.md5(key.encode()).hexdigest(), 16) % vocab_size)
    return torch.tensor([ids], dtype=torch.long, device=device)


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
        self.domain_encoders = nn.ModuleDict(
            {
                domain: DomainEncoder(layers=1, d_model=d, n_heads=4, ffn_width=128)
                for domain in DOMAIN_ORDER
            }
        )
        self._domain_d_model = {domain: d for domain in DOMAIN_ORDER}

        # Projections to shared d=1024 for fusion
        self._to_fusion = nn.ModuleDict(
            {domain: nn.Linear(d, 1024, bias=False) for domain in DOMAIN_ORDER}
        )

        self.afn = ActivationFlowNetwork(top_k=cfg.afn.top_k)
        self.fusion = EpistemicFusionLayer(cfg)

        self.dag_head = EvidenceDAGHead(n_domains=7, d_model=1024)
        self.attack_head = ATTACKClassifier(d_model=1024)
        self.decoder = ConstrainedDecoder(
            vocab_size=cfg.embedding.vocabulary_size,
            d_model=1024,
            n_heads=4,
            n_layers=2,
            ffn_width=256,
            dropout=0.0,
        )
        self._tiny = True

    def _build_full(self, cfg: VajraConfig) -> None:
        """Build full-dimension pipeline."""
        self.embedding = EmbeddingLayer(cfg)
        self.domain_encoders = nn.ModuleDict(build_domain_encoders(cfg))
        self._domain_d_model = {d: cfg.domain_encoders[d].d_model for d in DOMAIN_ORDER}

        # Projections from domain d_model → shared 1024 for fusion
        self._to_fusion = nn.ModuleDict(
            {
                domain: (
                    nn.Linear(
                        cfg.domain_encoders[domain].d_model,
                        cfg.shared_d_model,
                        bias=False,
                    )
                    if cfg.domain_encoders[domain].d_model != cfg.shared_d_model
                    else nn.Identity()
                )
                for domain in DOMAIN_ORDER
            }
        )

        self.afn = ActivationFlowNetwork(top_k=cfg.afn.top_k)
        self.fusion = EpistemicFusionLayer(cfg)

        self.dag_head = EvidenceDAGHead(n_domains=7, d_model=cfg.shared_d_model)
        self.attack_head = ATTACKClassifier(d_model=cfg.shared_d_model)
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

        # --- Route events to domains, then build per-domain input embeddings ---
        domain_inputs: dict[str, torch.Tensor] = {}
        events_by_domain: dict[str, list] = {d: [] for d in DOMAIN_ORDER}
        for evt in request.events:
            dom = evt.domain if evt.domain in events_by_domain else "detection_network"
            events_by_domain[dom].append(evt)

        if not self._tiny:
            # The full pipeline needs the real tokenization paths (S-TOON /
            # numerical / graph) wired through per-domain input projections that
            # bring the shared-1024 embedding down to each encoder's d_model
            # (768/512). Until that exists, fail loudly rather than return output
            # derived from placeholder tensors.
            raise NotImplementedError(
                "Full pipeline.run() requires real tokenization wiring. Use "
                "tiny=True for end-to-end testing, or drive the encoder/fusion/"
                "decoder modules directly."
            )

        vocab = self.cfg.embedding.vocabulary_size
        for domain in DOMAIN_ORDER:
            ids = _event_token_ids(events_by_domain[domain], vocab, device)  # (1, seq)
            domain_inputs[domain] = self.embedding(ids)  # (1, seq, d)

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
                cls_1024 = self.fusion.domain_absent.unsqueeze(0).unsqueeze(
                    0
                )  # (1, 1, 1024)
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

        # --- Decision state (F6 full tap, or F3 logits when early-exit fired) ---
        decision_logits = fusion_result.decision_logits  # (1, 4)
        decision_state_probs = fusion_result.decision_probs  # (1, 4)
        decision_class = decision_logits.argmax(dim=-1).item()
        decision_probs = decision_state_probs[0].tolist()
        confidence = float(decision_state_probs[0].max().item())

        # --- DCAT override (§6.4) ---
        # Max DCAT divergence across the DCAT-bearing encoders (Detection,
        # Forensics). A high null-state divergence forces escalation when the
        # decision would otherwise DEFER (2) or be INSUFFICIENT_CONTEXT (3).
        max_dcat_divergence = 0.0
        for domain in ("detection_network", "forensics_provenance"):
            enc = self.domain_encoders[domain]
            div = getattr(enc, "last_dcat_divergence", None)
            if div is not None:
                max_dcat_divergence = max(max_dcat_divergence, float(div.mean().item()))
        decision_class, dcat_override = apply_dcat_override(
            decision_class, max_dcat_divergence, self.cfg.dcat.theta_divergence
        )

        # --- ATT&CK technique head (reads the post-fusion representation, §6.3) ---
        fused_cls = fusion_result.fused_repr  # (1, d_model) post-fusion
        if fused_cls is None:
            fused_cls = fusion_input[:, 0, :]
        technique_logits = self.attack_head(fused_cls)  # (1, 716) — raw logits
        technique_probs = torch.sigmoid(technique_logits)[0]  # (716,)
        technique_ids = [
            f"T{1000 + i}"
            for i, p in enumerate(technique_probs.tolist())
            if p > self.attack_head.THRESHOLD
        ]

        # --- Evidence DAG head ---
        dag_adj = self.dag_head(fusion_input)  # (1, 32, 32)
        evidence_dag = EvidenceDAG(nodes=[], edges=[])

        # --- AFN scores for response ---
        afn_score_list: list[AFNScoreType] = []
        for domain, score in afn_scores_raw.items():
            for idx, imp in zip(
                score.token_indices.tolist(), score.importance_scores.tolist()
            ):
                afn_score_list.append(
                    AFNScoreType(domain=domain, token_index=idx, importance=imp)
                )

        # --- Constrained decoder (only for classes 0/1 + justification requested) ---
        justification_trace = ""

        if request.request_justification_trace and decision_class in (0, 1):
            # Gather AFN top-16 tokens for the first available domain
            first_domain = next(
                (d for d in DOMAIN_ORDER if afn_hiddens.get(d) is not None), None
            )
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

                    # Project domain d_model → decoder d_model (shared 1024) using the
                    # same learned projection used for fusion. This is a no-op (Identity)
                    # for the 1024-dim encoders and a real Linear for the 768/512 ones,
                    # preserving per-token feature structure for decoder cross-attention.
                    top_tokens = self._to_fusion[first_domain](top_tokens)

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
