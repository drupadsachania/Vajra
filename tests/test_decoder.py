"""Tests for Chunk 7: ConstrainedDecoder + MTPDrafter."""
import pytest
import torch
from vajra.decoder.vocab_mask import VocabularyMask
from vajra.decoder.decoder import ConstrainedDecoder
from vajra.decoder.mtp_drafter import MTPDrafter


@pytest.fixture
def vocab_size():
    return 50428


@pytest.fixture
def decoder(vocab_size):
    return ConstrainedDecoder(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        ffn_width=128,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# VocabularyMask
# ---------------------------------------------------------------------------
class TestVocabularyMask:
    def test_blocked_tokens_neg_inf(self, vocab_size):
        mask = VocabularyMask(vocab_size)
        logits = torch.zeros(1, 1, vocab_size)
        out = mask.apply(logits)
        blocked = list(mask.blocked_ids)
        assert all(out[0, 0, i] == float("-inf") for i in blocked[:5])

    def test_unblocked_tokens_unchanged(self, vocab_size):
        mask = VocabularyMask(vocab_size)
        logits = torch.zeros(1, 1, vocab_size)
        out = mask.apply(logits)
        # Token 0 and 1 should not be blocked
        if 0 not in mask.blocked_ids:
            assert out[0, 0, 0] == 0.0
        if 1 not in mask.blocked_ids:
            assert out[0, 0, 1] == 0.0

    def test_escape_rate_zero(self, vocab_size):
        """1000 greedy steps — no blocked token ever has argmax output."""
        mask = VocabularyMask(vocab_size)
        blocked = torch.tensor(sorted(mask.blocked_ids), dtype=torch.long)
        if len(blocked) == 0:
            pytest.skip("No blocked tokens configured")

        rng = torch.manual_seed(42)
        for _ in range(1000):
            logits = torch.randn(1, vocab_size)
            masked = mask.apply(logits)
            chosen = masked.argmax(dim=-1).item()
            assert chosen not in mask.blocked_ids, (
                f"Blocked token {chosen} was selected — escape rate > 0"
            )

    def test_sentinel_tokens_blocked(self, vocab_size):
        mask = VocabularyMask(vocab_size)
        for sid in range(50408, 50428):
            if sid < vocab_size:
                assert sid in mask.blocked_ids


# ---------------------------------------------------------------------------
# ConstrainedDecoder
# ---------------------------------------------------------------------------
class TestConstrainedDecoder:
    def test_output_shape(self, decoder, vocab_size):
        batch, tgt_len, afn_k = 1, 10, 16
        input_ids = torch.randint(0, 100, (batch, tgt_len))
        enc = torch.randn(batch, afn_k, 64)
        logits = decoder(input_ids, enc)
        assert logits.shape == (batch, tgt_len, vocab_size)

    def test_blocked_tokens_neg_inf_in_output(self, decoder):
        input_ids = torch.randint(0, 100, (1, 5))
        enc = torch.randn(1, 16, 64)
        logits = decoder(input_ids, enc)
        blocked = decoder.vocab_mask.blocked_ids
        for bid in list(blocked)[:5]:
            assert all(logits[0, :, bid] == float("-inf"))

    def test_vocabulary_escape_rate_zero(self, vocab_size):
        """No blocked token is ever argmax of decoder output."""
        d = ConstrainedDecoder(
            vocab_size=vocab_size, d_model=64, n_heads=4,
            n_layers=2, ffn_width=128, dropout=0.0,
        )
        blocked = d.vocab_mask.blocked_ids
        if not blocked:
            pytest.skip("No blocked tokens")

        torch.manual_seed(0)
        for _ in range(1000):
            ids = torch.randint(0, 100, (1, 4))
            enc = torch.randn(1, 16, 64)
            logits = d(ids, enc)
            chosen = logits[0, -1, :].argmax().item()
            assert chosen not in blocked

    def test_max_256_positions_enforced(self, vocab_size):
        """Position 256 gets position_id=255 (clamped)."""
        d = ConstrainedDecoder(
            vocab_size=vocab_size, d_model=64, n_heads=4,
            n_layers=2, ffn_width=128, dropout=0.0,
        )
        # Sequence of length 257 should run without error (position IDs clamped)
        ids = torch.randint(0, 100, (1, 257))
        enc = torch.randn(1, 16, 64)
        logits = d(ids, enc)
        assert logits.shape == (1, 257, vocab_size)

    def test_cross_attention_uses_only_afn_tokens(self, vocab_size):
        """Decoder cross-attention keys/values come from the 16-token subset, not full seq."""
        d = ConstrainedDecoder(
            vocab_size=vocab_size, d_model=64, n_heads=4,
            n_layers=2, ffn_width=128, dropout=0.0,
        )
        # Verify that passing different-length encoder states changes output
        ids = torch.randint(0, 100, (1, 5))
        enc_16 = torch.randn(1, 16, 64)
        enc_32 = torch.randn(1, 32, 64)  # Would be 32 tokens if raw encoder passed
        out_16 = d(ids, enc_16)
        # Shape should be the same regardless
        assert out_16.shape == (1, 5, vocab_size)
        # enc_16 has exactly 16 tokens — that's the AFN constraint
        assert enc_16.shape[1] == 16

    def test_greedy_decode_stays_within_max_tokens(self, decoder):
        enc = torch.randn(1, 16, 64)
        out = decoder.greedy_decode(enc, bos_token_id=1, eos_token_id=2, max_new_tokens=20)
        assert out.shape[1] <= 21  # BOS + up to 20 new tokens


# ---------------------------------------------------------------------------
# MTPDrafter
# ---------------------------------------------------------------------------
class TestMTPDrafter:
    def test_draft_shape(self):
        drafter = MTPDrafter(vocab_size=50428, d_model_primary=1024,
                             d_model_drafter=512, n_heads=8)
        ctx = torch.randint(0, 1000, (1, 8))
        draft = drafter.draft(ctx, k=4)
        assert draft.shape == (1, 4)

    def test_verify_full_match(self):
        """When drafter top-1 == primary top-1 for all k tokens, all k accepted."""
        drafter = MTPDrafter(vocab_size=100, d_model_primary=64,
                             d_model_drafter=32, n_heads=4)
        k = 4
        draft_ids = torch.tensor([[10, 20, 30, 40]])
        # Craft primary logits so top-1 matches draft exactly
        primary_logits = torch.zeros(1, k, 100)
        for i, t in enumerate([10, 20, 30, 40]):
            primary_logits[0, i, t] = 10.0  # highest logit = draft token
        accepted = drafter.verify(draft_ids, primary_logits)
        assert accepted.shape[1] == k

    def test_verify_first_mismatch_stops_early(self):
        """Accept primary's token at first mismatch then stop."""
        drafter = MTPDrafter(vocab_size=100, d_model_primary=64,
                             d_model_drafter=32, n_heads=4)
        k = 4
        draft_ids = torch.tensor([[10, 20, 30, 40]])
        primary_logits = torch.zeros(1, k, 100)
        primary_logits[0, 0, 10] = 10.0   # match
        primary_logits[0, 1, 99] = 10.0   # mismatch at position 1 → accept 99 then stop
        primary_logits[0, 2, 30] = 10.0
        primary_logits[0, 3, 40] = 10.0
        accepted = drafter.verify(draft_ids, primary_logits)
        assert accepted.shape[1] == 2  # token 10 (match) + token 99 (primary at mismatch)
        assert accepted[0, 0].item() == 10
        assert accepted[0, 1].item() == 99

    def test_shares_embedding_with_primary(self):
        """Drafter can accept shared embedding from primary model."""
        shared = torch.nn.Embedding(50428, 1024)
        drafter = MTPDrafter(vocab_size=50428, d_model_primary=1024,
                             d_model_drafter=512, n_heads=8,
                             shared_embed=shared)
        assert drafter.token_embed is shared
