"""Activation Flow Networks — L2-norm importance scoring (§7.1).

For each domain encoder, AFN scores tokens at the AFN-target layer by their
L2-norm of hidden states. Top-k=16 tokens per domain become the exclusive
cross-attention source for the constrained decoder.

Max latency: 15ms on CPU for a (1, 512, 1024) hidden state tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AFNScore:
    domain: str
    token_indices: torch.Tensor   # (top_k,) long — absolute position indices
    importance_scores: torch.Tensor  # (top_k,) float — normalized L2 norms


class ActivationFlowNetwork(nn.Module):
    """Computes per-domain AFN scores from encoder hidden states.

    Uses hook-based caching: each DomainEncoder stores `afn_hidden` after
    its forward pass. AFN reads those and returns top-k indices + scores.
    """

    TOP_K = 16

    def __init__(self, top_k: int = 16):
        super().__init__()
        self.top_k = top_k

    def score_domain(
        self, hidden: torch.Tensor, domain: str
    ) -> AFNScore:
        """Score all tokens in a single domain's AFN-layer hidden states.

        hidden : (batch, seq, d_model) — hidden states at AFN target layer
        Returns AFNScore for batch index 0 (single-example scoring).
        """
        # L2 norm per token: (batch, seq)
        norms = hidden.norm(dim=-1)   # (batch, seq)
        # Use first example in batch for index selection (per-request operation)
        norms_0 = norms[0]            # (seq,)
        # Normalize to sum ≈ 1
        norm_sum = norms_0.sum().clamp(min=1e-8)
        importance = norms_0 / norm_sum

        k = min(self.top_k, norms_0.shape[0])
        top_scores, top_indices = importance.topk(k)

        return AFNScore(
            domain=domain,
            token_indices=top_indices,
            importance_scores=top_scores,
        )

    def forward(
        self,
        encoder_hiddens: dict[str, torch.Tensor],
    ) -> dict[str, AFNScore]:
        """
        encoder_hiddens : {domain: (batch, seq, d_model)} hidden states at AFN layer

        Returns dict of domain → AFNScore.
        """
        return {
            domain: self.score_domain(hidden, domain)
            for domain, hidden in encoder_hiddens.items()
            if hidden is not None
        }

    def gather_top_tokens(
        self,
        encoder_outputs: dict[str, torch.Tensor],
        afn_scores: dict[str, AFNScore],
    ) -> dict[str, torch.Tensor]:
        """Extract top-k token representations from full encoder hidden states.

        encoder_outputs : {domain: (batch, seq, d_model)} — full hidden states
        afn_scores      : {domain: AFNScore}

        Returns {domain: (batch, top_k, d_model)} — indexed token states.
        """
        result = {}
        for domain, score in afn_scores.items():
            if domain not in encoder_outputs:
                continue
            hidden = encoder_outputs[domain]          # (batch, seq, d_model)
            idx = score.token_indices                 # (top_k,)
            result[domain] = hidden[:, idx, :]        # (batch, top_k, d_model)
        return result
