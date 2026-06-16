"""Path A — S-TOON sentinel-bounded text tokenizer.

Wraps a HuggingFace BPE tokenizer, guaranteeing that all 20 sentinel tokens
are treated as single atomic IDs (never split by BPE sub-word rules).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Sequence

from vajra.config import VajraConfig

_SENTINEL_TOKENS: tuple[str, ...] = (
    "<|S_START|>", "<|S_END|>", "<|S_KEY|>", "<|S_VAL|>",
    "<|S_ARR|>", "<|S_NULL|>",
    "<|S_DOMAIN:detection|>", "<|S_DOMAIN:forensics|>", "<|S_DOMAIN:cti|>",
    "<|S_DOMAIN:vuln|>", "<|S_DOMAIN:identity|>", "<|S_DOMAIN:ir|>",
    "<|S_DOMAIN:compliance|>",
    "<|S_EVTX|>", "<|S_STIX|>", "<|S_SIGMA|>", "<|S_CLOUD|>", "<|S_LDAP|>",
    "<|S_BASELINE|>", "<|S_OBSERVED|>",
)


class SentinelTokenizer:
    """BPE tokenizer augmented with guaranteed-atomic sentinel tokens.

    Without a real pre-trained BPE checkpoint the tokenizer falls back to a
    character-level mapping so that tests can exercise routing logic without
    a downloaded model.  In production the `hf_model_name` constructor arg
    loads a real security-corpus BPE model.
    """

    SENTINEL_TOKENS: tuple[str, ...] = _SENTINEL_TOKENS

    def __init__(self, cfg: VajraConfig, hf_model_name: str | None = None):
        self.cfg = cfg
        self._sentinel_ids: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

        if hf_model_name is not None:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_pretrained(hf_model_name)
            # Add sentinels as guaranteed-atomic special tokens
            self._tok.add_special_tokens(list(_SENTINEL_TOKENS))
            for tok in _SENTINEL_TOKENS:
                sid = self._tok.token_to_id(tok)
                self._sentinel_ids[tok] = sid
                self._id_to_token[sid] = tok
        else:
            # Minimal stub: vocab = 0…50407 (chars) + 20 sentinels at 50408…50427
            self._tok = None
            base = 50408
            for i, tok in enumerate(_SENTINEL_TOKENS):
                sid = base + i
                self._sentinel_ids[tok] = sid
                self._id_to_token[sid] = tok

        # The embedding table is sized to cfg.embedding.vocabulary_size; a larger
        # tokenizer vocab would index out of range at embedding time.
        assert self.vocab_size <= cfg.embedding.vocabulary_size, (
            f"Tokenizer vocab_size {self.vocab_size} exceeds configured "
            f"embedding vocabulary_size {cfg.embedding.vocabulary_size}"
        )

    @property
    def vocab_size(self) -> int:
        if self._tok is not None:
            return self._tok.get_vocab_size()
        return self.cfg.embedding.vocabulary_size

    def sentinel_id(self, token: str) -> int:
        if token not in self._sentinel_ids:
            raise KeyError(f"Unknown sentinel: {token!r}")
        return self._sentinel_ids[token]

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs. Sentinels are never fragmented."""
        if self._tok is not None:
            return self._tok.encode(text).ids
        # Stub: map each character to its ASCII value mod 50408, sentinels preserved
        ids: list[int] = []
        i = 0
        while i < len(text):
            matched = False
            for sent in _SENTINEL_TOKENS:
                if text[i:].startswith(sent):
                    ids.append(self._sentinel_ids[sent])
                    i += len(sent)
                    matched = True
                    break
            if not matched:
                ids.append(ord(text[i]) % 50408)
                i += 1
        return ids

    def is_sentinel(self, token_id: int) -> bool:
        return token_id in self._id_to_token
