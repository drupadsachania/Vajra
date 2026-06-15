"""Training losses for Vajra (§8.1–8.4).

FMLMLoss    — masked value-span cross-entropy (sentinel positions excluded)
KillChainLoss — next-state cross-entropy for kill-chain semiautomaton
MITREOntologyLoss — TransE + RotatE margin loss
BaNELLoss   — KL(P_baseline || P_observed) at S_NULL positions; λ annealed 0.1→0.3
DPOLoss     — Direct Preference Optimisation, β=0.1
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FMLMLoss(nn.Module):
    """Masked value-span cross-entropy.

    Sentinel token positions are EXCLUDED from the loss — only value-span
    tokens (non-sentinel positions) contribute.
    """

    def __init__(self, sentinel_id_min: int = 50408, sentinel_id_max: int = 50427,
                 ignore_index: int = -100):
        super().__init__()
        self.sentinel_id_min = sentinel_id_min
        self.sentinel_id_max = sentinel_id_max
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sentinel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        logits       : (batch, seq_len, vocab_size)
        labels       : (batch, seq_len)   — -100 for positions not to predict
        sentinel_mask: (batch, seq_len) bool, True = sentinel position → exclude from loss.
                       If None, derived from labels being in [sentinel_id_min, sentinel_id_max].

        Returns scalar loss.
        """
        B, S, V = logits.shape

        if sentinel_mask is None:
            # Build mask from any label that falls in the sentinel range
            sentinel_mask = (labels >= self.sentinel_id_min) & (labels <= self.sentinel_id_max)

        # Merge: positions that are either ignored or sentinel get -100
        effective_labels = labels.clone()
        effective_labels[sentinel_mask] = self.ignore_index

        # Index only the ~15% valid positions rather than computing full-vocab CE
        # on all B*S positions (most of which are masked anyway).
        flat_logits = logits.reshape(B * S, V)
        flat_labels = effective_labels.reshape(B * S)
        valid = flat_labels != self.ignore_index
        if not valid.any():
            return flat_logits.flatten()[0] * 0.0
        return F.cross_entropy(flat_logits[valid], flat_labels[valid], reduction="mean")


class KillChainLoss(nn.Module):
    """Next-state cross-entropy loss for the 7-state kill-chain semiautomaton.

    When `current_states` is provided the transition mask is applied: only
    s→s and s→s+1 transitions are valid; all other logits are set to −∞.
    """

    N_STATES = 7

    def forward(
        self,
        state_logits: torch.Tensor,
        target_states: torch.Tensor,
        current_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        state_logits   : (batch, 7)
        target_states  : (batch,) — next state index 0..6
        current_states : (batch,) — current state IDs (enables validity mask)

        Returns scalar loss.
        """
        if current_states is not None:
            from vajra.fusion.kill_chain import _build_transition_mask
            mask = _build_transition_mask().to(state_logits.device)  # (7, 7)
            allowed = mask[current_states]   # (batch, 7) — 0.0 allowed, -inf blocked
            state_logits = state_logits + allowed
        return F.cross_entropy(state_logits, target_states, reduction="mean")


class MITREOntologyLoss(nn.Module):
    """TransE + RotatE margin-based ontology embedding loss.

    TransE: ||h + r - t||₂ + margin
    RotatE: ||h ∘ r - t||₂ + margin   (element-wise product in complex space)
    Combined: sum of both with margin γ=2.
    """

    def __init__(self, margin: float = 2.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        h_neg: torch.Tensor,
        t_neg: torch.Tensor,
    ) -> torch.Tensor:
        """
        h, r, t     : (batch, d_emb) — positive triple
        h_neg, t_neg: (batch, d_emb) — negative samples

        Returns scalar loss.
        """
        # TransE
        pos_transe = (h + r - t).norm(dim=-1)
        neg_transe = (h_neg + r - t_neg).norm(dim=-1)
        loss_transe = F.relu(self.margin + pos_transe - neg_transe).mean()

        # RotatE (real-valued approximation via element-wise product)
        pos_rotate = (h * r - t).norm(dim=-1)
        neg_rotate = (h_neg * r - t_neg).norm(dim=-1)
        loss_rotate = F.relu(self.margin + pos_rotate - neg_rotate).mean()

        return loss_transe + loss_rotate


class BaNELLoss(nn.Module):
    """Baseline-Null Entropy Loss (BaNEL).

    KL(P_baseline || P_observed) computed only at S_NULL token positions.
    λ is annealed from λ_start→λ_end over n_anneal_steps.
    No policy gradient: this is a pure KL-divergence supervision loss.

    Gradient is NON-ZERO only when null_mask has True entries.
    """

    NULL_TOKEN_ID = 50419  # <|S_NULL|> sentinel ID (index 11 of 20 sentinels)

    def __init__(
        self,
        lambda_start: float = 0.1,
        lambda_end: float = 0.3,
        n_anneal_steps: int = 5,
    ):
        super().__init__()
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.n_anneal_steps = n_anneal_steps
        self._step = 0

    @property
    def current_lambda(self) -> float:
        t = min(self._step / max(self.n_anneal_steps, 1), 1.0)
        return self.lambda_start + t * (self.lambda_end - self.lambda_start)

    def step_epoch(self):
        self._step += 1

    def forward(
        self,
        logits_observed: torch.Tensor,
        logits_baseline: torch.Tensor,
        null_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits_observed : (batch, seq, vocab)
        logits_baseline : (batch, seq, vocab)
        null_mask       : (batch, seq) bool — True at S_NULL positions

        Returns scalar loss (0 when null_mask is all-False).
        """
        if not null_mask.any():
            # Scalar zero with grad connection; avoids full-tensor reduction.
            return logits_observed.flatten()[0] * 0.0

        # Detach baseline so gradient does not flow into the EMA-frozen stream.
        # Use log_softmax for numerical stability (avoids log(p + eps) in fp16).
        log_b = F.log_softmax(logits_baseline[null_mask].detach(), dim=-1)  # (N, V)
        log_o = F.log_softmax(logits_observed[null_mask], dim=-1)            # (N, V)

        # KL(P_base || P_obs) via F.kl_div (expects log-space input, linear target).
        kl = F.kl_div(log_o, log_b.exp(), reduction="none").sum(dim=-1)  # (N,)
        return self.current_lambda * kl.mean()


class DPOLoss(nn.Module):
    """Direct Preference Optimisation loss (§8.4).

    β=0.1. Scores log-ratio of model probability on winner vs loser sequences,
    normalised by reference model log-ratio.
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        logits_w: torch.Tensor,
        logits_l: torch.Tensor,
        ref_logits_w: torch.Tensor,
        ref_logits_l: torch.Tensor,
        labels_w: torch.Tensor,
        labels_l: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits_w / _l     : (batch, seq, vocab) — model logits for winner / loser
        ref_logits_w / _l : (batch, seq, vocab) — reference model logits
        labels_w / _l     : (batch, seq)         — target token IDs; -100 = ignore

        Returns scalar DPO loss.
        """
        def _seq_log_prob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            # Sum log-probs over non-ignored positions
            log_p = F.log_softmax(logits, dim=-1)
            mask = labels != -100
            token_log_p = log_p.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            token_log_p = token_log_p * mask.float()
            return token_log_p.sum(dim=-1)  # (batch,)

        pi_w  = _seq_log_prob(logits_w, labels_w)
        pi_l  = _seq_log_prob(logits_l, labels_l)
        # Detach reference logits: gradient must not flow into the reference model.
        ref_w = _seq_log_prob(ref_logits_w.detach(), labels_w)
        ref_l = _seq_log_prob(ref_logits_l.detach(), labels_l)

        logits_dpo = self.beta * ((pi_w - ref_w) - (pi_l - ref_l))
        loss = -F.logsigmoid(logits_dpo).mean()
        return loss
