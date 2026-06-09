# Vajra Fix Instructions — Companion to the 2026-06-09 Review

Step-by-step implementation instructions for all 23 findings in
`2026-06-09-code-quality-review.md`. Organized into four phases; each phase is
independently committable and ends with a verification gate. Line numbers refer to
commit `d58bf5a`. Run the full suite (`pytest tests/ -q`) plus
`python validate_requirements.py --impl impl_config.json` after every phase.

---

## Phase A — Hard-constraint compliance (small diffs, do first)

### A1. Activate the attack-tooling blocklist in `ConstrainedDecoder`

**Files:** `src/vajra/decoder/decoder.py`, `src/vajra/interface/pipeline.py`

`ConstrainedDecoder.__init__` (decoder.py:137) builds `VocabularyMask(vocab_size)` with
no tokenizer, so `blocklist.json` terms are never resolved to token IDs — only the 20
sentinel slots are blocked.

1. Add parameters to `ConstrainedDecoder.__init__`:
   ```python
   def __init__(self, ..., tokenizer=None, extra_blocked_ids=None):
       ...
       self.vocab_mask = VocabularyMask(
           vocab_size, tokenizer=tokenizer, extra_blocked_ids=extra_blocked_ids
       )
   ```
2. In `pipeline.py` (decoder construction ~line 116), pass the Path-A tokenizer:
   `tokenizer=self.tokenizer_a` (expose the underlying HF tokenizer's `encode` if needed).
3. **Test:** extend `test_decoder.py::test_curated_terms_resolved_via_tokenizer` to build a
   `ConstrainedDecoder` (not a bare mask) with the fake tokenizer and assert the resolved
   IDs are `-inf` in `forward` output.

### A2. Close the three IP-validation bypasses

**File:** `src/vajra/training/data.py`

1. Replace the IPv4 regex (line 55) — `\b` fails next to word chars:
   ```python
   _IPV4_RE = re.compile(r"(?<![0-9A-Za-z])\d{1,3}(?:\.\d{1,3}){3}(?![0-9])")
   ```
2. Add an IPv6 candidate regex and validate candidates via `ipaddress` (parse failure →
   not an IP → skip):
   ```python
   _IPV6_RE = re.compile(r"(?<![0-9A-Za-z:])(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f:.]+")
   ```
   Allowed IPv6: only documentation range `2001:db8::/32`, loopback `::1`, and unique-local
   `fc00::/7` — add these to `_ALLOWED_IP_NETWORKS`.
3. Scan dict **keys** in `_iter_strings` (lines 70-79):
   ```python
   if isinstance(value, dict):
       for k, v in value.items():
           if isinstance(k, str):
               yield k
           yield from _iter_strings(v)
   ```
4. Document the deliberate fail-open in `_is_allowed_ip` (non-parseable candidate → not an
   IP) and the known false-positive on version strings like `"1.2.3.4"` (fail-closed,
   acceptable).
5. **Tests** (in `tests/test_training.py::TestIPEnforcement`): real public IPv6 rejected;
   `2001:db8::1` allowed; `{"8.8.8.8": {...}}` as a key rejected; `"host8.8.8.8"` rejected.

### A3. Always emit F3 markers from the fusion layer

**File:** `src/vajra/fusion/epistemic_fusion.py`

`forward` computes `f3_logits`/`f3_probs` (lines 113-114) and discards them on the
non-early-exit path. Hard constraint: markers at F3 **and** F6 on every call.

1. Add fields to `EarlyExitResult` (lines 27-34): `f3_logits`, `f3_probs` (always set),
   keep `decision_logits` as the authoritative tap (F3 on early exit, F6 otherwise), and
   make F6 fields `None` on early exit.
2. Populate `response` plumbing in `pipeline.py` if `VajraInferenceResponse` carries
   per-tap probabilities (check `types.py`; if not, no interface change needed — the
   constraint is about the model emitting them, so exposing on the result object suffices).
3. **Test:** `test_fusion.py` — assert `result.f3_logits is not None` on a full (non-exit)
   forward pass and shape `(B, 4)`.

### A4. Fix `ATTACKClassifier` logits/probabilities contract

**File:** `src/vajra/heads/technique_head.py` (+ `pipeline.py:228-233`)

1. `forward` returns raw logits: `return self.head(fusion_cls)`.
2. Add `predict`: `return (torch.sigmoid(self(x)) > self.THRESHOLD)`.
3. `loss` keeps `binary_cross_entropy_with_logits` — now consistent.
4. Update `pipeline.py:228-233` to call `torch.sigmoid` explicitly (or `predict`) and use
   `self.attack_head.THRESHOLD` instead of the re-hardcoded `0.35`.
5. **Test:** `loss(forward(x), targets)` equals manual BCE-with-logits on the same logits;
   add a regression test asserting `forward` output is unbounded (has values outside [0,1]
   for random weights/inputs).

### A5. Fix `ConformalPredictor.predict_set`

**File:** `src/vajra/heads/calibration.py`

1. Replace the descending-iteration break loop (lines 70-81) with the threshold set:
   ```python
   p = probs.flatten()
   idx = (p >= threshold).nonzero(as_tuple=True)[0].tolist()
   return idx or [int(p.argmax())]
   ```
2. In `calibrate` (line 59) use the finite-sample quantile: index
   `max(0, math.ceil((n + 1) * (1 - coverage)) - 1)` into the sorted true-class scores.
3. Fix the annotation: `coverage_targets: list[float] | None = None`.
4. **Tests:** with a calibration set where true-class probs are known, assert empirical
   coverage ≥ target on held-out synthetic data; assert the set is non-empty and not
   always all-classes.

---

## Phase B — Pre-training correctness (fix before any training run)

### B1. Per-field NetFlow projections + magnitude scaling

**File:** `src/vajra/tokenization/path_b.py` (lines 166-197)

1. Replace the shared `scalar_proj` for bytes/packets/duration with three
   `nn.Linear(1, d_model)` projections applied to `torch.log1p(scalars[:, i:i+1])`.
2. While here: build per-record tensors on `self.scalar_proj.weight.device` (lines 45-59),
   hoist the RFC1918 `ip_network` objects to module scope, and populate (or delete) the
   documented-but-empty CVSS severity slots 22-24 (lines 71-85).
3. **Test:** embeddings for records differing only in `bytes` vs only in `packets` must
   differ from each other (currently identical — write the failing test first).

### B2. Rework DCAT `BaselineCache`

**File:** `src/vajra/encoders/dcat.py` (lines 18-54)

1. Fixed-capacity registered buffer keyed by entity:
   ```python
   self.register_buffer("baseline", torch.zeros(max_entities, d_model))
   self.register_buffer("initialized", torch.zeros(max_entities, dtype=torch.bool))
   ```
   `update(entity_id, pooled_state)` does the β=0.9 EMA per entity; `get(entity_id)`
   returns the row or the fallback when uninitialized.
2. Gate updates: skip when `self.training` (shuffled batches must not pollute the EMA) and
   when `decision_class in (1, 3)` (existing freeze rule).
3. Read β from `cfg.dcat.ema_beta_baseline_update` instead of the hardcoded `BETA = 0.9`
   (line 26); delete the dead `ALPHA = 0.3` (line 78) and unused `ln_ffn` (line 109).
4. Document cold-start: with an uninitialized baseline, divergence ≡ 0 and the §6.4
   override cannot fire — intended fail-safe, must be stated.
5. **Tests:** cache survives `.to(device)` and `state_dict` round-trip; two entities EMA
   independently; no mutation in training mode.

### B3. Float64 timestamp phase computation

**File:** `src/vajra/embedding/temporal.py` (lines 67-72)

```python
wt = t.double().unsqueeze(-1) * self.omega.double()
phi = torch.cat([wt.cos(), wt.sin()], dim=-1).to(t.dtype if t.is_floating_point() else torch.float32)
```
**Test:** `phi(t)` and `phi(t + 1.0)` differ measurably for `t = 1.7e9` at high
frequencies (currently indistinguishable in fp32).

### B4. Detach baseline/reference streams in losses; stable KL

**File:** `src/vajra/training/losses.py`

1. `BaNELLoss` (lines 176-182):
   ```python
   log_b = F.log_softmax(logits_baseline[null_mask].detach(), dim=-1)
   log_o = F.log_softmax(logits_observed[null_mask], dim=-1)
   kl = F.kl_div(log_o, log_b, log_target=True, reduction="none").sum(-1).mean()
   ```
   Zero-loss path: `logits_observed.flatten()[0] * 0.0` instead of full-tensor sum.
2. `DPOLoss` (lines 220-223): `.detach()` both reference logits inside the loss.
3. **Tests:** `logits_baseline.grad is None` after backward; KL finite in bf16 with a
   sharp (one-hot-ish) baseline distribution.

### B5. Implement the kill-chain validity mask; guard −inf-in-CE

**Files:** `src/vajra/training/losses.py` (lines 64-85), `src/vajra/decoder/decoder.py`

1. `KillChainLoss.forward(logits, targets, current_states)`: build the additive mask from
   the same `valid_transitions` buffer as `kill_chain.py` (import it — single source),
   apply before CE, and assert all targets are valid transitions (else raise with the
   offending index).
2. Decoder training path: `forward` applies the vocab mask before returning (decoder.py:
   160-161), so CE on a blocked target → `inf`. Add to the (future) LM loss:
   `assert not vocab_mask.blocked_mask[targets].any()` or set those positions to
   `ignore_index`.
3. **Test:** loss is finite for valid labels; raises (or ignores) for a label on a blocked
   transition/token.

### B6. Checkpoint-compatible QAT; quantize K/V outputs

**File:** `src/vajra/training/qat.py`

1. `KVCacheQuantizer.forward` (lines 48-49): `return self.fake_quant(self.linear(x))` —
   the cache stores projection *outputs*; that is what 8-bit cache simulation must see.
2. Key compatibility: implement `_load_from_state_dict`/`state_dict` hooks on `QATLinear`
   and `KVCacheQuantizer` mapping `X.linear.weight ↔ X.weight` so pre-QAT checkpoints load
   into wrapped models and vice versa.
3. Fail loudly: `apply_qat` raises `ImportError` when `_QAT_AVAILABLE` is false (currently
   silent no-op, lines 26-28); add the missing guard to `_wrap_kv_projections` (line 93).
4. Migrate `torch.quantization` → `torch.ao.quantization` imports.
5. **Tests:** `load_state_dict` round-trips pre-QAT → QAT-wrapped; output of wrapped K/V
   projection differs from unwrapped (proves the right tensor is quantized).

### B7. Thread config through hardcoded duplicates

**Files:** many — this is the lock-workflow fix. Single source of truth per value:

| Hardcoded site | Config source |
|---|---|
| `dcat.py:26` BETA, `:78` ALPHA | `cfg.dcat.ema_beta_baseline_update`, `cfg.dcat.dual_stream_alpha` |
| `long_context.py:150` DCAT_LAYERS_0IDX | `cfg.domain_encoders[...].dcat_layers` (pass via `domain_encoders.py:40-45`) |
| `base_encoder.py:15` _AFN_LAYER_MAP | `cfg.afn_layer_for_encoder()` (currently never called) |
| `epistemic_fusion.py:23-24` exit thresholds | `cfg.fusion.ecl_tap_fast_condition_*` |
| `vocab_mask.py:34` _N_SENTINELS | `cfg.sentinel_count` |
| `temporal.py:44`, `source_type.py:18` | `cfg.embedding.temporal_frequency_pairs`, `.source_type_count` |
| `technique_head.py:16-17` N_TECHNIQUES/THRESHOLD | **add** `attack_technique_count`, `attack_threshold` to `VajraConfig` |
| `kill_chain.py:15` state list | `cfg.kill_chain.states` |
| `path_a.py:15-23` sentinel list | `cfg.sentinel_tokens` |
| `dag_head.py:22` hidden 2048 | new `cfg` field or ctor arg |

Also define `DOMAINS: tuple[str, ...]` once in `config.py`; derive `DOMAIN_TAG_MAP`
(`data.py:17-25`) and `DOMAIN_ORDER` (`pipeline.py:32-35`) from it; delete dead
`DOMAIN_D_MODEL` (`pipeline.py:38-43`).
**Test:** construct modules with a mutated config (e.g. `theta_divergence=2.0`,
`source_type_count=17`) and assert the change takes effect.

---

## Phase C — Performance (before any latency/VRAM claims)

### C1. KV-cached decoder generation

**File:** `src/vajra/decoder/decoder.py`

The single largest win (~128× at 256 tokens; the locked 90 ms budget is unreachable
without it).

1. `_CausalSelfAttention.forward(x, kv_cache=None)`: when caching, project only the new
   token's Q/K/V, append K/V to the cache, attend Q (len 1) over the cached keys. RoPE
   position = cache length.
2. Cross-attention: `encoder_states` is constant per generation. Add
   `precompute_cross_kv(encoder_states) -> list[(k, v)]` (one pair per layer) and accept
   it in `forward`; stop calling `self.k/self.v` per layer per step (lines 75-76).
3. `greedy_decode` (164-181): feed one token per step; compute `lm_head` + vocab mask on
   the last position only (`self.lm_head(x[:, -1:])`).
4. Register the causal mask as a sliced non-persistent buffer (line 45) — or better,
   switch to `F.scaled_dot_product_attention(..., is_causal=True)` and drop it entirely.
5. **Tests:** cached and uncached `greedy_decode` produce identical token sequences
   (seeded); wall-clock test asserting cached path is faster at 64+ tokens.
6. This unblocks the wNa8o8 "o8" story: `KVCacheQuantizer` now has a real cache to
   quantize — wire `fake_quant` at cache-append time.

### C2. SDPA everywhere except DCAT

**Files:** `src/vajra/encoders/blocks.py:126-133`, `src/vajra/encoders/long_context.py:51-61,87-90`

1. Extract a shared `MultiHeadAttention(d_model, n_heads, causal=False, rope=True)` in
   `blocks.py` built on `F.scaled_dot_product_attention` (handles dropout via
   `dropout_p=self.p if self.training else 0.0`); reuse it in `PreLNTransformerBlock`,
   `_CausalSelfAttention`, and `_MTPDecoderBlock` (which also needs RoPE added — it
   currently has **no positional encoding**, see C4).
2. Local window attention: pass the window as an `attn_mask` (bool) to SDPA, or use
   flash-attn's native `window_size=(w, 0)` when available. Fix the window-size bug while
   here: ±`window` gives an effective 8193-token window vs the spec'd 4096
   (`long_context.py:54-57`).
3. DCAT (`dcat.py:135-139`) keeps explicit attention weights (needed for the KL) — but
   compute the KL from `log_softmax` of the pre-softmax scores (cheaper + stable) instead
   of clamped post-softmax probs (lines 57-65).
4. Fully-masked-row NaN guard (blocks.py:127-129): SDPA with a bool mask handles this;
   if keeping manual paths anywhere, use a large finite negative instead of `-inf`.
5. **Tests:** SDPA output matches the old manual path within tolerance at seq 128
   (pre-refactor golden values); 8K-seq forward peak memory drops by >10× (use
   `torch.cuda.max_memory_allocated` guard, skip on CPU).

### C3. Cache per-forward allocations

**Files:** `blocks.py:21-25`, `long_context.py:54-57`, `mtp_drafter.py:44`

1. `RotaryEmbedding`: cache cos/sin (including the doubled `torch.cat([cos, cos])` form)
   keyed by `(seq_len, device, dtype)`; recompute only on growth. Share one instance per
   encoder rather than one per block (DCAT currently runs it twice per block per stream).
2. Sliding-window mask: build once per encoder forward at the actual seq length and pass
   down to all local layers; cache by `(S, device)`.
3. **Test:** two consecutive forwards at the same shape trigger zero new mask/RoPE
   allocations (count via a monkeypatched `torch.arange` or memory snapshot).

### C4. MTP drafter: KV cache, RoPE, vectorized verify, bonus token

**File:** `src/vajra/decoder/mtp_drafter.py`

1. Add RoPE to `_MTPDecoderBlock` exactly as `_CausalSelfAttention` does — the drafter is
   currently position-blind, which destroys draft acceptance rate.
2. `draft` (106-114): encode context once with a KV cache; each of the k steps feeds one
   token; apply `final_ln` to the last position only. Drop the no-op
   `context_ids.clone()` (105).
3. `verify` (129-141), vectorized (one sync instead of k):
   ```python
   match = (draft_ids == primary_preds)[0]
   n = int(match.cumprod(0).sum().item())
   if n == draft_ids.shape[1]:           # full acceptance → free bonus token
       return torch.cat([draft_ids, bonus_token], dim=1)
   return torch.cat([draft_ids[:, :n], primary_preds[:, n:n + 1]], dim=1)
   ```
   (`bonus_token` = argmax of the primary's logit at the position after the last draft.)
4. Stop double-serializing the shared embedding (79-82): hold it as a plain attribute via
   `object.__setattr__`, or document the optimizer-dedup requirement.
5. **Tests:** existing verify tests still pass; full-acceptance now returns k+1 tokens;
   drafter `state_dict` no longer contains the 50428×1024 embedding.

### C5. De-noise the pipeline hot path

**File:** `src/vajra/interface/pipeline.py`

1. Persistent executor in `__init__` (or sequential execution for batch=1 — benchmark;
   threads on one CUDA stream serialize anyway), not per-call (line 174).
2. Single GPU→CPU transfer: assemble decision class/probs/confidence/divergence into one
   small tensor, one `.cpu()`, then unpack (replaces ~6 syncs at lines 209-242). For
   techniques: `torch.nonzero(probs > thr)` on-device before transferring.
3. Delete the discarded `dag_head` forward (236-237) or populate `EvidenceDAG` from it.
4. Move the imports out of `_encode_domain` (137-138) to module level.
5. Use `afn.gather_top_tokens` instead of the inline reimplementation (255); fix AFN batch
   handling first (assert `B == 1` in `score_domain`/`gather_top_tokens`, or vectorize
   with `topk(dim=-1)` + `torch.gather`) — `afn.py:39-98` currently applies example-0's
   indices to the whole batch.
6. **Test:** response numerically identical pre/post refactor (seeded); latency smoke test.

### C6. FMLM masked-subset CE

**File:** `src/vajra/training/losses.py:55-60`

```python
valid = effective_labels != self.ignore_index
return F.cross_entropy(logits[valid], effective_labels[valid])
```
(Guard the all-masked case → return zero-connected scalar as in B4.)

---

## Phase D — Real-data wiring and structural consolidation

### D1. Wire tokenization + embedding into `pipeline.run()`

**File:** `src/vajra/interface/pipeline.py:156-168`

Currently `run()` feeds `torch.randn` — output is independent of event content.

1. Route each `SecurityEvent` through `detect_path` → Path A/B/C encoder → `EmbeddingLayer`
   (with source-type IDs and ContiFormer timestamps), per domain.
2. `VajraTokenizer` (tokenization/tokenizer.py) must actually own Path B/C instances —
   today it imports them and routes to paths it cannot encode.
3. Until complete, guard: `if not self._tiny: raise NotImplementedError("tokenization not wired")`.
4. Fix the placeholder justification trace (248-271): detokenize through Path A's
   tokenizer; build decoder cross-attention input from per-domain AFN top-16 (all domains,
   not just the first with hidden states).
5. Replace fabricated `f"T{1000+i}"` IDs (231) with a real index→ATT&CK-ID table (ships
   with the MITRE CTI pull — see `scripts/data/manifest.json` priority 1).
6. **Tests:** two requests with different event content produce different decision
   logits; same request twice (eval mode, seeded) produces identical output.

### D2. ONNX export with real dynamic axes

**File:** `src/vajra/interface/onnx_export.py`

1. Export with `dynamo=True`; rewrite shape-dependent ops tensor-wise where the exporter
   complains (`torch.arange(T)` → `torch.arange(input_ids.shape[1], device=...)` is
   already traced as data-dependent under dynamo).
2. Fusion: export a wrapper that skips the F3 early-exit branch and returns
   `(f3_logits, f6_logits)` tensors only — no dataclasses, no `.all()` control flow.
3. **Test (the important one):** round-trip the exported decoder at a seq_len ≠ the trace
   length (e.g. export at 4, run at 16) and a different batch; assert output shape and
   numerical match vs PyTorch.

### D3. Consolidate decision machinery

**Files:** `heads/decision_head.py`, `fusion/epistemic_fusion.py`, `interface/pipeline.py`

1. Delete the unused `DecisionStateClassifier` instantiation from the pipeline (113-115)
   — dead parameters in checkpoints/ONNX — or make the fusion taps use it. Keep exactly
   one DCAT-override implementation (`apply_dcat_override`); delete the tensorized copy
   in `DecisionStateClassifier` (decision_head.py:65-72).
2. Return per-block kill-chain transition logits from `EpistemicFusionLayer` (110) —
   required for the EXPERIMENT-2 multi-task objective; today they're discarded.
3. Feed the ATT&CK head the **fused** representation: expose final fusion hidden state on
   `EarlyExitResult` and use it in `pipeline.py:228` (currently reads pre-fusion
   detection CLS).

### D4. Long-context encoder hygiene

**File:** `src/vajra/encoders/long_context.py`

1. Unify signatures: `forward(x, attn_mask=None, decision_class=None)` on both
   `DomainEncoder` and `InterleavedAttentionEncoder`; thread the padding mask into
   local/global attention (currently padded batches attend to pad tokens).
2. Reconcile the layer schedule with the docstring (134-146): code yields globals at
   {3, 11} because DCAT consumes layer 7's global slot; either fix the comment or skip
   DCAT positions when striding. Decide and document.
3. Enforce or remove `DomainEncoder.context_window` (base_encoder.py:39) — currently
   stored, never checked.

### D5. Honest naming and small fixes

- `losses.py:119-122` + `path_c.py:30-32`: implement true RotatE (split d into re/im,
  rotate by unit-modulus complex r — normalize per element-pair, not whole-vector L2) or
  rename to what it is.
- `path_c.py:125-127`: compute Graphormer degree features from binary adjacency
  (`(adj > 0).sum(-1)`), not the normalized one (collapses to 0/1 after `.long()`).
- `ontology.py:49-57`: replace positional node→token alignment with an explicit
  `(batch, seq)` token→node index map (−1 = absent).
- `function_calling.py:23`: enforce `MAX_AGENTIC_TURNS` — add
  `parse_tool_calls(text, turn=0)` returning `[]` when `turn >= MAX_AGENTIC_TURNS`, or an
  `AgenticSession` counter object. Remove unused `re` import.
- `path_a.py:60-64`: assert `self.vocab_size <= cfg.embedding.vocabulary_size` in the
  constructor (HF-tokenizer drift → embedding index crash).
- Rename one of the two `AFNScore` classes (`afn.py:20` vs `interface/types.py:52`), e.g.
  `AFNScoreEntry` in types; drop the import alias in pipeline.py:28.
- `epistemic_fusion.py:118`: document that `.all()` early-exit semantics are batch-global
  (every example must pass) — fine for the batch=1 SOC path, surprising otherwise.
- Delete dead code: unused imports in `path_a.py`, `tokenizer.py`, `kill_chain.py`,
  `dag_head.py`, `afn.py` (`time`, `TOP_K`), `vocab_mask.py` (`_term_count`),
  `long_context.py` (`is_global`), `path_b.py` (`_slot_starts`); fix the "Mean DCAT
  divergence" comment at `pipeline.py:216` (it's max — max is correct, fix the comment).

---

## Verification matrix

| Phase | Gate |
|---|---|
| A | `pytest tests/ -q` green incl. new blocklist-through-decoder, IPv6/dict-key, F3-marker, logits-contract, conformal-coverage tests; `validate_requirements.py --impl impl_config.json` exits 0 |
| B | All Phase-A tests + NetFlow field-distinguishability, BaselineCache device/state_dict round-trip, fp64-timestamp, detached-gradient, kill-chain-mask, QAT checkpoint round-trip tests |
| C | Cached == uncached decode (token-exact); SDPA == manual attention (numeric); 8K-seq memory regression guard; verify() single-sync; pipeline output unchanged pre/post |
| D | Content-sensitivity test (different events → different logits); ONNX variable-shape round-trip; EXPERIMENT-2 kill-chain logits exposed; full suite + validator green |

Suggested commit cadence: one commit per lettered item (A1, A2, …) so each fix is
independently revertable and reviewable.
