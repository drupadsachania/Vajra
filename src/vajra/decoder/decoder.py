"""Constrained Justification Decoder (§6.5).

8-layer causal transformer decoder.
Cross-attention: ONLY over AFN top-16 tokens (never raw encoder states).
Vocabulary mask: applied at logit computation (before sampling).
Max 256 positional embeddings; position IDs clamped to 255.
Invoked only when decision_class ∈ {0,1} AND request_justification_trace=True.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from vajra.encoders.blocks import RotaryEmbedding, apply_rotary_pos_emb, SwiGLUFFN
from .vocab_mask import VocabularyMask


class _CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(S, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scale = math.sqrt(self.head_dim)
        w = torch.matmul(q, k.transpose(-2, -1)) / scale
        # Causal mask
        causal = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        w = w.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
        w = F.softmax(w, dim=-1)
        w = self.drop(w)
        out = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.o(out)


class _CrossAttention(nn.Module):
    """Cross-attention over AFN top-16 encoder tokens (keys/values from encoder)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_states: torch.Tensor) -> torch.Tensor:
        """
        x             : (batch, tgt_len, d_model)
        encoder_states: (batch, afn_top_k, d_model) — ONLY top-k tokens
        """
        B, T, D = x.shape
        S = encoder_states.shape[1]

        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(encoder_states).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(encoder_states).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        w = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / scale, dim=-1)
        w = self.drop(w)
        out = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, T, D)
        return self.o(out)


class _DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_width: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.self_attn  = _CausalSelfAttention(d_model, n_heads, dropout)
        self.cross_attn = _CrossAttention(d_model, n_heads, dropout)
        self.ffn  = SwiGLUFFN(d_model, ffn_width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_states: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.self_attn(self.ln1(x)))
        x = x + self.drop(self.cross_attn(self.ln2(x), encoder_states))
        x = x + self.drop(self.ffn(self.ln3(x)))
        return x


class ConstrainedDecoder(nn.Module):
    """8-layer causal decoder with AFN-gated cross-attention + vocab masking.

    Cross-attention is restricted to the top-16 AFN-scored tokens per domain.
    Vocabulary mask applied at every logit step (p = −∞ for blocked tokens).
    Max 256 position IDs enforced by clamping.
    """

    MAX_POSITIONS = 256

    def __init__(
        self,
        vocab_size: int = 50428,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 8,
        ffn_width: int = 4096,
        dropout: float = 0.1,
        tokenizer=None,
        extra_blocked_ids=None,
    ):
        super().__init__()
        self.d_model   = d_model
        self.n_layers  = n_layers
        self.vocab_size = vocab_size

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(self.MAX_POSITIONS, d_model)

        self.blocks = nn.ModuleList([
            _DecoderBlock(d_model, n_heads, ffn_width, dropout)
            for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)

        self.vocab_mask = VocabularyMask(
            vocab_size, tokenizer=tokenizer, extra_blocked_ids=extra_blocked_ids
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        afn_encoder_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        input_ids          : (batch, tgt_len) — decoder input token IDs
        afn_encoder_states : (batch, afn_top_k, d_model) — top-k AFN tokens ONLY

        Returns : (batch, tgt_len, vocab_size) — masked logits
        """
        B, T = input_ids.shape
        # Clamp position IDs to MAX_POSITIONS - 1 (hard constraint §6.5)
        pos_ids = torch.arange(T, device=input_ids.device).clamp(max=self.MAX_POSITIONS - 1)

        x = self.token_embed(input_ids) + self.pos_embed(pos_ids).unsqueeze(0)

        for block in self.blocks:
            x = block(x, afn_encoder_states)

        x = self.final_ln(x)
        logits = self.lm_head(x)          # (batch, tgt_len, vocab_size)
        logits = self.vocab_mask.apply(logits)
        return logits

    def greedy_decode(
        self,
        afn_encoder_states: torch.Tensor,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        max_new_tokens: int = 256,
    ) -> torch.Tensor:
        """Simple greedy decode. Returns generated token IDs (batch=1 only)."""
        B = afn_encoder_states.shape[0]
        input_ids = torch.full((B, 1), bos_token_id, dtype=torch.long,
                               device=afn_encoder_states.device)
        for _ in range(min(max_new_tokens, self.MAX_POSITIONS - 1)):
            logits = self.forward(input_ids, afn_encoder_states)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break
        return input_ids
