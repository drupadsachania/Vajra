"""Tests for Chunk 5 — Epistemic Fusion Layer + Kill-Chain State Machine."""

import pytest
import torch

from vajra.config import VajraConfig, FusionConfig
from vajra.fusion.kill_chain import KillChainStateMachine, N_STATES, KILL_CHAIN_STATES
from vajra.fusion.epistemic_fusion import EpistemicFusionLayer, EarlyExitResult

CFG = VajraConfig()


# ── Kill-chain state machine ───────────────────────────────────────────────────

class TestKillChainStateMachine:

    def _make_kc(self, d=64):
        return KillChainStateMachine(d_model=d)

    def test_stay_transition_allowed(self):
        kc = self._make_kc()
        state = torch.tensor([3])  # EXPLOIT
        h = torch.randn(1, 64)
        logits, next_s = kc(h, state)
        # next_s should be either 3 (stay) or 4 (advance), never 0-2 or 5-6
        assert next_s.item() in (3, 4)

    def test_advance_one_allowed(self):
        """Artificially set logit for s+1 very high — it should be selected."""
        kc = self._make_kc()
        state = torch.tensor([2])  # DELIVER
        h = torch.zeros(1, 64)
        with torch.no_grad():
            kc.transition_head.weight.zero_()
            kc.transition_head.bias = nn.Parameter(torch.zeros(N_STATES))
            kc.transition_head.bias.data[3] = 10.0  # strongly prefer EXPLOIT
        logits, next_s = kc(h, state)
        assert next_s.item() == 3   # EXPLOIT (advance one from DELIVER)

    def test_skip_stage_blocked(self):
        """EXPLOIT (3) → EXFIL (6): gap > 1, should be blocked."""
        kc = self._make_kc()
        state = torch.tensor([3])  # EXPLOIT
        h = torch.zeros(1, 64)
        with torch.no_grad():
            kc.transition_head.weight.zero_()
            kc.transition_head.bias = nn.Parameter(torch.zeros(N_STATES))
            kc.transition_head.bias.data[6] = 100.0   # try to jump to EXFIL
        _, next_s = kc(h, state)
        # EXFIL is blocked; should stay at 3 or advance to 4
        assert next_s.item() in (3, 4)

    def test_seven_states(self):
        assert len(KILL_CHAIN_STATES) == 7

    def test_logits_shape(self):
        kc = self._make_kc()
        state = torch.zeros(2, dtype=torch.long)
        h = torch.randn(2, 64)
        logits, next_s = kc(h, state)
        assert logits.shape == (2, 7)
        assert next_s.shape == (2,)

    def test_last_state_can_only_stay(self):
        """State 6 (EXFIL) — no valid next state, should stay."""
        kc = self._make_kc()
        state = torch.tensor([6])
        h = torch.zeros(1, 64)
        with torch.no_grad():
            kc.transition_head.weight.zero_()
            kc.transition_head.bias = nn.Parameter(torch.zeros(N_STATES))
        _, next_s = kc(h, state)
        assert next_s.item() == 6


import torch.nn as nn   # needed in test above


# ── Epistemic Fusion Layer ─────────────────────────────────────────────────────

def _make_tiny_cfg():
    cfg = VajraConfig()
    cfg.fusion = FusionConfig(blocks=6, d_model=64, attention_heads=4, ffn_width=256)
    return cfg


class TestEpistemicFusionLayer:

    def _make_fusion(self):
        return EpistemicFusionLayer(_make_tiny_cfg())

    def test_f6_output_shapes(self):
        fusion = self._make_fusion()
        x = torch.randn(2, 7, 64)
        result = fusion(x)
        assert result.decision_logits.shape == (2, 4)
        assert result.decision_probs.shape  == (2, 4)
        assert result.kill_chain_state.shape == (2,)
        assert result.exit_block == 6

    def test_probs_sum_to_one(self):
        fusion = self._make_fusion()
        x = torch.randn(2, 7, 64)
        result = fusion(x)
        sums = result.decision_probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(2))

    def test_f3_early_exit_triggered(self):
        """Craft batch where all examples have p(class=2) > 0.92."""
        fusion = self._make_fusion()
        x = torch.randn(2, 7, 64)
        # Patch f3_head to output logits that make class-2 dominate
        with torch.no_grad():
            fusion.f3_head.weight.zero_()
            fusion.f3_head.bias = nn.Parameter(
                torch.tensor([-100.0, -100.0, 100.0, -100.0])
            )
        result = fusion(x)
        assert result.early_exit is True
        assert result.exit_block == 3

    def test_no_early_exit_with_balanced_probs(self):
        """Balanced logits → no class exceeds threshold → F6 runs."""
        fusion = self._make_fusion()
        x = torch.randn(2, 7, 64)
        with torch.no_grad():
            fusion.f3_head.weight.zero_()
            fusion.f3_head.bias = nn.Parameter(torch.zeros(4))
        result = fusion(x)
        # With equal probs (0.25 each) neither threshold is met
        assert result.early_exit is False
        assert result.exit_block == 6

    def test_f3_fail_safe_exit(self):
        """p(class=3) > 0.85 also triggers early exit."""
        fusion = self._make_fusion()
        x = torch.randn(2, 7, 64)
        with torch.no_grad():
            fusion.f3_head.weight.zero_()
            fusion.f3_head.bias = nn.Parameter(
                torch.tensor([-100.0, -100.0, -100.0, 100.0])
            )
        result = fusion(x)
        assert result.early_exit is True
        assert result.exit_block == 3

    def test_domain_absent_substitution(self):
        """Missing domain slots replaced by learned DOMAIN_ABSENT token."""
        fusion = self._make_fusion()
        x = torch.randn(1, 7, 64)
        present_mask = torch.ones(1, 7, dtype=torch.bool)
        present_mask[0, 4] = False   # domain 4 absent

        # Run twice: same input but domain 4 absent vs. present
        with torch.no_grad():
            r_absent  = fusion(x, present_mask=present_mask)
            r_present = fusion(x, present_mask=None)

        # Outputs should differ (absent domain token ≠ original CLS)
        assert not torch.allclose(r_absent.decision_logits, r_present.decision_logits)

    def test_kill_chain_state_range(self):
        fusion = self._make_fusion()
        x = torch.randn(3, 7, 64)
        result = fusion(x)
        assert result.kill_chain_state.shape == (3,)
        assert (result.kill_chain_state >= 0).all()
        assert (result.kill_chain_state < 7).all()

    def test_initial_kc_state_used(self):
        fusion = self._make_fusion()
        x = torch.randn(1, 7, 64)
        init_state = torch.tensor([3])  # start at EXPLOIT
        result = fusion(x, initial_kc_state=init_state)
        # Final state must be ≥ initial (no backward transitions)
        assert result.kill_chain_state.item() >= 3

    def test_param_count_approx_100m(self):
        """Full-size fusion actual ~76M (spec ~100M assumed full FFN; SwiGLU reduces intermediate)."""
        fusion = EpistemicFusionLayer(CFG)
        n = sum(p.numel() for p in fusion.parameters())
        assert 60_000_000 <= n <= 120_000_000, f"Fusion param count {n:,} outside expected range"
