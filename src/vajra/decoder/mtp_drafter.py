"""MTP (Multi-Token Prediction) Speculative Decoding Drafter (§8.5).

2-layer decoder, d=512, ~15M incremental params over shared embedding weights.
Shares tokenizer + embedding weights with the primary model.
CPU inference alongside GPU primary.
Draft k=4 tokens per step; primary verifies in single forward pass.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vajra.encoders.blocks import SwiGLUFFN


class _MTPDecoderBlock(nn.Module):
    """Lightweight decoder block for the MTP drafter (no cross-attention — uses encoded context)."""

    def __init__(self, d_model: int, n_heads: int, ffn_width: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.ffn = SwiGLUFFN(d_model, ffn_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_ln = self.ln1(x)
        q = self.q(x_ln).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x_ln).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x_ln).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        scale = math.sqrt(self.head_dim)
        w = torch.matmul(q, k.transpose(-2, -1)) / scale
        causal = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        w = w.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
        w = F.softmax(w, dim=-1)
        out = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.o(out)
        x = x + self.ffn(self.ln2(x))
        return x


class MTPDrafter(nn.Module):
    """2-layer speculative decoding drafter.

    Shares token embedding with the primary model via `shared_embed`.
    Projects shared d_model → d_drafter internally.

    Draft k=4 tokens, then primary verifies all simultaneously:
    - Accepted tokens: where drafter top-1 == primary top-1
    - On first mismatch: use primary's token and discard remainder
    """

    DRAFT_STEPS = 4

    def __init__(
        self,
        vocab_size: int = 50428,
        d_model_primary: int = 1024,
        d_model_drafter: int = 512,
        n_heads: int = 8,
        shared_embed: nn.Embedding | None = None,
    ):
        super().__init__()
        self.d_drafter = d_model_drafter
        self.vocab_size = vocab_size

        # Shared embedding (reference — not owned; set by caller)
        if shared_embed is not None:
            self.token_embed = shared_embed
        else:
            self.token_embed = nn.Embedding(vocab_size, d_model_primary)

        # Project from primary d_model → drafter d_model
        self.input_proj = nn.Linear(d_model_primary, d_model_drafter, bias=False)

        self.blocks = nn.ModuleList(
            [
                _MTPDecoderBlock(d_model_drafter, n_heads, d_model_drafter * 4)
                for _ in range(2)
            ]
        )
        self.final_ln = nn.LayerNorm(d_model_drafter)
        self.lm_head = nn.Linear(d_model_drafter, vocab_size, bias=False)

    def draft(
        self,
        context_ids: torch.Tensor,
        k: int | None = None,
    ) -> torch.Tensor:
        """Generate k draft token IDs from the context.

        context_ids : (1, context_len) long
        Returns     : (1, k) draft token IDs
        """
        k = k or self.DRAFT_STEPS
        ids = context_ids.clone()
        for _ in range(k):
            emb = self.token_embed(ids)  # (1, len, d_primary)
            x = self.input_proj(emb)  # (1, len, d_drafter)
            for block in self.blocks:
                x = block(x)
            x = self.final_ln(x)
            logits = self.lm_head(x[:, -1, :])  # (1, vocab)
            next_tok = logits.argmax(dim=-1, keepdim=True)  # (1, 1)
            ids = torch.cat([ids, next_tok], dim=1)
        return ids[:, context_ids.shape[1] :]  # (1, k) only new tokens

    def verify(
        self,
        draft_ids: torch.Tensor,
        primary_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Accept draft tokens where they match primary top-1 predictions.

        draft_ids      : (1, k) token IDs proposed by drafter
        primary_logits : (1, k, vocab_size) logits from primary model over same positions

        Returns accepted token IDs (may be shorter than k).
        """
        primary_preds = primary_logits.argmax(dim=-1)  # (1, k)
        accepted = []
        for i in range(draft_ids.shape[1]):
            if draft_ids[0, i] == primary_preds[0, i]:
                accepted.append(draft_ids[0, i].unsqueeze(0))
            else:
                # First mismatch: accept primary's prediction and stop
                accepted.append(primary_preds[0, i].unsqueeze(0))
                break
        if not accepted:
            return torch.empty(1, 0, dtype=torch.long, device=draft_ids.device)
        # Each element is shape (1,); concat → (accepted_len,), then add batch dim → (1, accepted_len)
        return torch.cat(accepted).unsqueeze(0)  # (1, accepted_len)
