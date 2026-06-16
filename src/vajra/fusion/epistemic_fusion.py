"""Epistemic Fusion Layer — 6 Pre-LN transformer blocks over 7 domain CLS tokens (§4.4).

F3 fast-exit tap: if p(class=2) > 0.92 OR p(class=3) > 0.85, skip F4–F6.
F6 full tap: primary decision state output.
Kill-chain state machine woven into each block.
Missing domains → DOMAIN_ABSENT learned token.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from vajra.config import VajraConfig
from vajra.encoders.blocks import PreLNTransformerBlock
from .kill_chain import KillChainStateMachine, N_STATES


N_DOMAINS = 7                   # seven domain encoders
FAST_EXIT_DEFER_THRESH  = 0.92  # p(class=2) > this → early exit
FAST_EXIT_INSUF_THRESH  = 0.85  # p(class=3) > this → early exit


@dataclass
class EarlyExitResult:
    decision_logits: torch.Tensor        # (batch, 4) — F3 on early-exit, F6 otherwise
    decision_probs:  torch.Tensor        # (batch, 4)
    kill_chain_state: torch.Tensor       # (batch,) current KC state
    early_exit: bool
    exit_block: int                      # 3 for F3 early-exit, 6 for F6
    # F3 tap always populated (hard constraint: markers at F3 and F6 on every call)
    f3_logits: torch.Tensor | None = None  # (batch, 4)
    f3_probs:  torch.Tensor | None = None  # (batch, 4)
    # Per-block kill-chain transition logits (for the EXPERIMENT-2 multi-task
    # objective — these are the trainable signal; the argmax state update is not).
    kc_logits: torch.Tensor | None = None  # (batch, n_blocks_run, 7)
    # Fused representation at the exit block (mean over the 7 domain tokens),
    # the correct input for the ATT&CK technique head (§6.3) rather than a raw
    # pre-fusion domain CLS.
    fused_repr: torch.Tensor | None = None  # (batch, d_model)


class EpistemicFusionLayer(nn.Module):
    """6-block epistemic fusion transformer.

    Input: (batch, 7, d_model) — one CLS token per domain.
    Missing domain slots get replaced by a learned DOMAIN_ABSENT token.
    """

    N_BLOCKS = 6
    N_CLASSES = 4

    def __init__(self, cfg: VajraConfig):
        super().__init__()
        fc = cfg.fusion
        d = fc.d_model

        self.domain_absent = nn.Parameter(torch.zeros(d))

        self.blocks = nn.ModuleList([
            PreLNTransformerBlock(d, fc.attention_heads, fc.ffn_width)
            for _ in range(self.N_BLOCKS)
        ])

        # Kill-chain state machine at each block
        self.kc_heads = nn.ModuleList([
            KillChainStateMachine(d) for _ in range(self.N_BLOCKS)
        ])

        # F3 and F6 decision taps
        self.f3_head = nn.Linear(d, self.N_CLASSES)
        self.f6_head = nn.Linear(d, self.N_CLASSES)

    def _replace_absent(
        self, domain_cls: torch.Tensor, present_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Replace absent domain CLS slots with learned DOMAIN_ABSENT token.

        domain_cls   : (batch, 7, d_model)
        present_mask : (batch, 7) bool, True = domain present; or None = all present
        """
        if present_mask is None:
            return domain_cls
        absent = ~present_mask   # (batch, 7)
        absent_3d = absent.unsqueeze(-1).expand_as(domain_cls)
        absent_tok = self.domain_absent.unsqueeze(0).unsqueeze(0).expand_as(domain_cls)
        return torch.where(absent_3d, absent_tok, domain_cls)

    def forward(
        self,
        domain_cls: torch.Tensor,
        present_mask: torch.Tensor | None = None,
        initial_kc_state: torch.Tensor | None = None,
    ) -> EarlyExitResult:
        """
        domain_cls       : (batch, 7, d_model)
        present_mask     : (batch, 7) bool or None (all present)
        initial_kc_state : (batch,) long or None (all start at RECON=0)

        Returns EarlyExitResult.
        """
        B = domain_cls.shape[0]
        x = self._replace_absent(domain_cls, present_mask)  # (batch, 7, d_model)

        if initial_kc_state is None:
            kc_state = torch.zeros(B, dtype=torch.long, device=x.device)
        else:
            kc_state = initial_kc_state

        f3_logits = None
        f3_probs = None
        kc_logits_per_block: list[torch.Tensor] = []

        for i, (block, kc_head) in enumerate(zip(self.blocks, self.kc_heads)):
            x = block(x)   # (batch, 7, d_model)

            # Kill-chain update using mean CLS representation
            cls_mean = x.mean(dim=1)   # (batch, d_model)
            kc_block_logits, kc_state = kc_head(cls_mean, kc_state)
            kc_logits_per_block.append(kc_block_logits)

            if i == 2:   # F3 tap (0-indexed block 2 = F3)
                f3_logits = self.f3_head(cls_mean)    # (batch, 4)
                f3_probs  = F.softmax(f3_logits, dim=-1)

                defer_flag = f3_probs[:, 2] > FAST_EXIT_DEFER_THRESH
                insuf_flag = f3_probs[:, 3] > FAST_EXIT_INSUF_THRESH
                if (defer_flag | insuf_flag).all():
                    return EarlyExitResult(
                        decision_logits=f3_logits,
                        decision_probs=f3_probs,
                        kill_chain_state=kc_state,
                        early_exit=True,
                        exit_block=3,
                        f3_logits=f3_logits,
                        f3_probs=f3_probs,
                        kc_logits=torch.stack(kc_logits_per_block, dim=1),
                        fused_repr=cls_mean,
                    )

        # F6 tap (all 6 blocks completed)
        cls_mean = x.mean(dim=1)
        f6_logits = self.f6_head(cls_mean)
        f6_probs  = F.softmax(f6_logits, dim=-1)

        return EarlyExitResult(
            decision_logits=f6_logits,
            decision_probs=f6_probs,
            kill_chain_state=kc_state,
            early_exit=False,
            exit_block=6,
            f3_logits=f3_logits,
            f3_probs=f3_probs,
            kc_logits=torch.stack(kc_logits_per_block, dim=1),
            fused_repr=cls_mean,
        )
