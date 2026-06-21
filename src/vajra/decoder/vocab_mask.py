"""Vocabulary mask — logit-level blocking at p = −∞ (§6.5).

Blocked tokens are set to −inf BEFORE softmax/sampling, making their
probability exactly 0 regardless of logit magnitude.

The blocklist is loaded from a curated config file (`blocklist.json`) rather
than hardcoded placeholders. It covers:
  - all sentinel tokens (structural markers that must never be emitted), and
  - attack-tooling / exploit-synthesis term tokens, resolved to vocabulary IDs
    via the active tokenizer at construction time.

Logit masking is inherently per-token, so a blocked term is only maskable when
it resolves to a single vocabulary token. A real security-corpus BPE vocab is
expected to carry the curated terms as atomic tokens; supply that tokenizer to
`VocabularyMask` to activate term-level blocking. The structural sentinel block
always applies regardless of tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn

_DEFAULT_BLOCKLIST_PATH = Path(__file__).with_name("blocklist.json")

# Sentinel token IDs occupy the top 20 slots of the default 50,428 vocab
# (50408..50427). Derived from vocab_size at construction so it tracks the
# configured vocabulary.
_N_SENTINELS = 20


class _Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


def _load_blocklist(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _resolve_terms_to_ids(
    blocklist: dict,
    tokenizer: _Tokenizer | None,
    vocab_size: int,
) -> set[int]:
    """Resolve curated blocked terms to single vocabulary token IDs.

    A term is masked only if the tokenizer encodes it to exactly one token
    within range. Without a tokenizer, no term-level IDs are resolved (only the
    structural sentinel block applies).
    """
    ids: set[int] = set()
    if tokenizer is None:
        return ids
    for _category, terms in blocklist.get("blocked_terms", {}).items():
        for term in terms:
            try:
                encoded = tokenizer.encode(term)
            except Exception:
                continue
            if len(encoded) == 1 and 0 <= encoded[0] < vocab_size:
                ids.add(int(encoded[0]))
    return ids


class VocabularyMask(nn.Module):
    """Applies −inf logit masking to a curated set of blocked token IDs.

    The mask is applied at the logit tensor level before softmax or argmax,
    making it impossible for blocked tokens to be sampled regardless of model
    state (satisfies the 'logit_p_neg_infinity' compliance requirement).
    """

    def __init__(
        self,
        vocab_size: int = 50428,
        tokenizer: _Tokenizer | None = None,
        blocklist_path: str | Path | None = None,
        extra_blocked_ids: set[int] | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        path = (
            Path(blocklist_path)
            if blocklist_path is not None
            else _DEFAULT_BLOCKLIST_PATH
        )
        blocklist = _load_blocklist(path)

        # 1) Structural sentinel block — always applied (top _N_SENTINELS slots).
        sentinel_start = max(0, vocab_size - _N_SENTINELS)
        blocked: set[int] = set(range(sentinel_start, vocab_size))

        # 2) Explicit token IDs from the config.
        for tid in blocklist.get("blocked_token_ids", []):
            if 0 <= int(tid) < vocab_size:
                blocked.add(int(tid))

        # 3) Curated attack-tooling terms resolved via the tokenizer.
        blocked |= _resolve_terms_to_ids(blocklist, tokenizer, vocab_size)

        # 4) Caller-supplied extras.
        if extra_blocked_ids:
            blocked |= {i for i in extra_blocked_ids if 0 <= i < vocab_size}

        mask = torch.zeros(vocab_size, dtype=torch.bool)
        if blocked:
            indices = torch.tensor(sorted(blocked), dtype=torch.long)
            mask[indices] = True
        self.register_buffer("block_mask", mask, persistent=True)
        self._blocked_ids = frozenset(blocked)
        self._term_count = len(blocklist.get("blocked_terms", {}))

    @property
    def blocked_ids(self) -> frozenset[int]:
        return self._blocked_ids

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Set blocked token logits to −inf. Shape: (..., vocab_size) → same."""
        return logits.masked_fill(self.block_mask, float("-inf"))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.apply(logits)
