"""Kill-chain 7-state semiautomaton (§5.1).

States: RECON=0, WEAPONIZE=1, DELIVER=2, EXPLOIT=3, INSTALL=4, C2=5, EXFIL=6
Valid transitions: s → s  (stay) or s → s+1  (advance one stage only).
Skip-stage transitions are blocked by a transition mask on the logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

KILL_CHAIN_STATES = [
    "RECON",
    "WEAPONIZE",
    "DELIVER",
    "EXPLOIT",
    "INSTALL",
    "C2",
    "EXFIL",
]
N_STATES = len(KILL_CHAIN_STATES)  # 7


# Transition mask: valid_next[s] = {s, s+1}
# Shape (N_STATES, N_STATES): True where transition is ALLOWED
def _build_transition_mask() -> torch.Tensor:
    mask = torch.full((N_STATES, N_STATES), float("-inf"))
    for s in range(N_STATES):
        mask[s, s] = 0.0  # stay
        if s + 1 < N_STATES:
            mask[s, s + 1] = 0.0  # advance one
    return mask


class KillChainStateMachine(nn.Module):
    """Per-block kill-chain transition head.

    Takes fusion hidden state → linear(1024 → 7) + transition mask + argmax.
    Returns (next_state_logits, next_state_id).
    """

    N_STATES = N_STATES

    def __init__(self, d_model: int = 1024):
        super().__init__()
        self.transition_head = nn.Linear(d_model, self.N_STATES)
        self.register_buffer("_transition_mask", _build_transition_mask())

    def forward(
        self,
        fusion_hidden: torch.Tensor,
        current_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        fusion_hidden : (batch, d_model) — CLS token of fusion at this block
        current_state : (batch,) long tensor of current kill-chain state IDs

        Returns:
          masked_logits : (batch, N_STATES)
          next_state    : (batch,) long tensor
        """
        raw_logits = self.transition_head(fusion_hidden)  # (batch, 7)

        # Apply transition mask: add -inf to disallowed transitions
        # _transition_mask[s] has 0.0 for allowed next states, -inf otherwise
        batch = raw_logits.shape[0]
        allowed = self._transition_mask[current_state]  # (batch, 7)
        masked = raw_logits + allowed  # (batch, 7)

        next_state = masked.argmax(dim=-1)  # (batch,)
        return masked, next_state
