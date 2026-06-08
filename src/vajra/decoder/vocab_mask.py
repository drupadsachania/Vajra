"""Vocabulary mask — logit-level blocking at p = −∞ (§6.5).

Blocked tokens are set to −inf BEFORE softmax/sampling, making their
probability exactly 0 regardless of logit magnitude.

The blocklist covers OffSec terms, exploit-synthesis patterns, real-IP
literals, and sentinel tokens that should never appear in decoder output.
In production the blocklist is loaded from a YAML/JSON config file.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Placeholder sentinel tokens that must never appear in decoder output
_DEFAULT_BLOCKED_TOKEN_IDS: set[int] = {
    # Sentinel token IDs 50408-50427 (assigned in SentinelTokenizer stub)
    *range(50408, 50428),
}

# Additional placeholder blocked patterns (token IDs would be computed from vocab)
# In production these come from a curated blocklist config file.
_EXPLOIT_BLOCKED: set[int] = {
    9999,   # placeholder: "shellcode"
    9998,   # placeholder: "exploit_payload"
    9997,   # placeholder: "reverse_shell"
}


class VocabularyMask(nn.Module):
    """Applies −inf logit masking to a fixed set of blocked token IDs.

    The mask is applied at the logit tensor level before softmax or argmax,
    making it impossible for blocked tokens to be sampled regardless of model
    state (satisfies the 'logit_p_neg_infinity' compliance requirement).
    """

    def __init__(
        self,
        vocab_size: int = 50428,
        extra_blocked_ids: set[int] | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        blocked = _DEFAULT_BLOCKED_TOKEN_IDS | _EXPLOIT_BLOCKED
        if extra_blocked_ids:
            blocked |= extra_blocked_ids

        # Keep only IDs within vocab range
        blocked = {i for i in blocked if i < vocab_size}

        # Boolean mask: True = blocked
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        if blocked:
            indices = torch.tensor(sorted(blocked), dtype=torch.long)
            mask[indices] = True
        self.register_buffer("block_mask", mask, persistent=True)
        self._blocked_ids = frozenset(blocked)

    @property
    def blocked_ids(self) -> frozenset[int]:
        return self._blocked_ids

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Set blocked token logits to −inf. Shape: (..., vocab_size) → same."""
        return logits.masked_fill(self.block_mask, float("-inf"))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.apply(logits)
