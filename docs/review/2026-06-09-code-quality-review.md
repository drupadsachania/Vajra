# Vajra Code Quality & Optimization Review — 2026-06-09

Full-codebase review (src/vajra, 25 modules, 195 tests passing). Three parallel deep reviews
covering encoders/tokenization, fusion/decoder/heads, and training/pipeline. Findings ranked
by impact. File:line references verified against the working tree at commit `d58bf5a`.

**Verdict:** the architecture skeleton is faithful to the spec and well-tested at the shape
level, but several hard constraints from REQUIREMENTS.lock are currently *structurally
satisfied yet functionally inert*, and the inference hot path cannot meet the locked latency
targets as written. The items below are ordered: fix Tier 1 before any training run; Tier 2
before performance claims; Tier 3 opportunistically.

---

## Tier 1 — Hard-constraint and correctness blockers (fix before training)

1. **`pipeline.run()` output is independent of event content** — `pipeline.py:156-168` feeds
   `torch.randn` into the encoders; the tokenization paths and `EmbeddingLayer` are constructed
   but never called. Every decision/technique/AFN output is noise. Wire tokenization+embedding
   into `run()`, or guard with `NotImplementedError` outside tiny mode.

2. **IP enforcement has three bypasses** (hard constraint `real_ips_in_training=false`) —
   `data.py:55,70-79`: (a) IPv6 never scanned, (b) `\b` boundary misses word-adjacent IPs like
   `host8.8.8.8`, (c) dict *keys* are never scanned (`{"8.8.8.8": {...}}` passes). Fix regex to
   `(?<![0-9A-Za-z])\d{1,3}(?:\.\d{1,3}){3}(?![0-9])`, add an IPv6 candidate scan, and yield keys
   in `_iter_strings`.

3. **Attack-tooling vocab block is inert in every real instantiation** — `decoder.py:137`
   constructs `VocabularyMask(vocab_size)` with no tokenizer/blocklist passthrough, so only the
   20 sentinel slots are ever blocked; the curated `blocklist.json` never activates. Thread
   `tokenizer`/`extra_blocked_ids` through `ConstrainedDecoder.__init__` (pipeline.py:90,116).

4. **F3 tap output discarded on the full path** (hard constraint: markers at F3 *and* F6 on
   every call) — `epistemic_fusion.py:112-138` computes f3_logits then drops them. Add
   `f3_logits/f3_probs` to `EarlyExitResult`, always populated.

5. **`ATTACKClassifier` double-sigmoid trap** — `technique_head.py:23-39`: `forward` returns
   sigmoid probabilities but `loss` uses `binary_cross_entropy_with_logits`; the natural call
   silently trains with sigmoid applied twice. Make `forward` return logits; move sigmoid to
   `predict`.

6. **`ConformalPredictor.predict_set` is broken** — `calibration.py:70-81`: the loop breaks on
   *individual* prob ≥ threshold while iterating descending, so sets are degenerately `{top1}`
   or all 4 classes; `cum` is dead. Correct split-conformal set: `{i : p_i ≥ threshold}` (fall
   back to argmax if empty). Also apply the finite-sample quantile correction `⌈(N+1)(1−cov)⌉`.

7. **NetFlow field embeddings are identical** — `path_b.py:189-191`: bytes/packets/duration all
   project the same 3-dim scalar vector through the same Linear → three byte-identical tokens;
   the model cannot distinguish the fields. Use per-field projections on scalar slices, and
   `log1p` the raw magnitudes (bytes can be 1e9+).

8. **`BaselineCache` (DCAT) is not per-entity, not device-safe, not persisted** —
   `dcat.py:18-54`: one global cache keyed by nothing (`max_entities` ignored), resets on any
   shape change, EMAs unrelated shuffled training samples, lives outside `state_dict`, and
   stays on CPU after `.to("cuda")`. The §6.4 override signal is unreliable as built. Rework as
   a fixed `(max_entities, d_model)` registered buffer keyed by entity ID; skip updates in
   training mode.

9. **Timestamp encoding loses all precision in fp32** — `temporal.py:67-72`: Unix-epoch seconds
   × ω up to 1e3 ≈ 1e12 exceeds fp32's ~7 significant digits → `cos/sin(wt)` is noise for most
   frequencies. Compute `wt` in float64 then cast, or require relative offsets.

10. **BaNEL loss: undetached baseline + fragile KL** — `losses.py:176-182`: gradient flows into
    the baseline stream (contradicting frozen-EMA semantics) and `log(p+1e-10)` underflows in
    fp16. Use `F.kl_div(log_softmax(obs), log_softmax(base).detach(), log_target=True)`.
    Similarly detach reference logits in `DPOLoss` (`losses.py:220-223`).

11. **`KillChainLoss` validity mask documented but not implemented** — `losses.py:64-85`: plain
    unmasked cross-entropy; the s→s/s→s+1 constraint is unenforced (`N_STATES` unused). Also
    `-inf`-masked logits fed to CE (kill_chain + vocab-masked decoder training path,
    `decoder.py:160`) yield `inf` loss if a label lands on a masked entry — assert or mask
    loss positions.

---

## Tier 2 — Performance (locked latency/VRAM targets unreachable without these)

12. **Decoder has no KV cache** — `decoder.py:164-181`: each greedy step re-forwards the entire
    prefix through 8 layers and computes `lm_head` over all positions (~128× redundant compute
    at 256 tokens; ~11 TFLOPs vs ~90 GFLOPs cached). The 90 ms decoder budget is unreachable.
    Add per-layer KV caching, compute `lm_head` on the last position only, and precompute
    cross-attention K/V once per generation (`decoder.py:75-76` currently recomputes the
    constant AFN-16 K/V every layer, every step). Note: `qat.py`'s 8-bit KV-cache quantization
    presupposes this cache exists.

13. **Full (B,H,S,S) attention matrices at 32K context** — `blocks.py:126-133`,
    `long_context.py:51-61,87-90`: three hand-rolled matmul→softmax→matmul paths materialize
    ~17 GB/layer in fp32 at 32K tokens. Replace all non-DCAT attention with
    `F.scaled_dot_product_attention` (flash/mem-efficient kernels). DCAT keeps explicit weights
    (needs them for KL) — that's fine at its 4K window.

14. **Per-forward allocations in the hot loop** — RoPE cos/sin recomputed every call by every
    block (`blocks.py:21-25`, doubled in DCAT's two streams); the S×S sliding-window mask
    rebuilt per layer per call (`long_context.py:54-57` — 8 GB of int64 at 32K); decoder causal
    mask re-allocated per step (`decoder.py:45`, `mtp_drafter.py:44`). Cache as non-persistent
    buffers keyed by (seq_len, device).

15. **Pipeline hot-path syncs and waste** — `pipeline.py`: per-call `ThreadPoolExecutor`
    construction (174) that serializes on one CUDA stream anyway; ~6 separate `.item()/.tolist()`
    GPU→CPU syncs (209-242); `dag_head` forward computed and discarded (236); module imports
    inside the threaded encode path (137). Persist the executor (or go sequential for batch=1),
    batch the transfers into one `.cpu()`, drop the dead DAG forward.

16. **MTP drafter re-encodes context per draft step and verifies with per-token host syncs** —
    `mtp_drafter.py:106-114,129-141`. KV-cache the context, vectorize verify
    (`cumprod` on the match mask, single sync), and take the standard bonus token on full
    acceptance. Also: the drafter has **no positional encoding at all** (no RoPE, no learned
    pos) — acceptance rate will tank; add RoPE as in `_CausalSelfAttention`.

17. **FMLM full-vocab CE over all positions** — `losses.py:55-60`: index the ~15% labeled
    positions first (`logits[valid]`), as BaNELLoss already does.

---

## Tier 3 — Structural quality

18. **ONNX dynamic axes are baked away** — `onnx_export.py` with `dynamo=False` traces
    `T = input_ids.shape[1]` as a constant: arange/triu/RoPE export at size tgt_len=4 and the
    declared `sequence_length` axis is fictional; fusion export can't carry the early-exit
    branch. Add an export wrapper (no early exit, tensor-only returns) and a round-trip test
    at a *different* seq_len/batch.

19. **QAT wrappers break checkpoints and quantize the wrong tensor** — `qat.py`: wrapping
    renames every key (`X.weight` → `X.linear.weight`) so pre/post-QAT checkpoints can't
    interoperate (add `_load_from_state_dict` remapping); `KVCacheQuantizer` fake-quants the
    *input* to K/V projections instead of the *output* (the actual cached tensor) — swap to
    `self.fake_quant(self.linear(x))`; QAT silently no-ops on ImportError (raise instead);
    migrate `torch.quantization` → `torch.ao.quantization`.

20. **Config exists but is bypassed by hardcoded duplicates** (defeats the lock workflow —
    a REQUIREMENTS.lock change would silently not propagate): DCAT α/β and layer indices
    (`dcat.py:26,78`, `long_context.py:150`), AFN layer map (`base_encoder.py:15` vs the
    never-called `VajraConfig.afn_layer_for_encoder`), fast-exit thresholds
    (`epistemic_fusion.py:23-24`), sentinel count (`vocab_mask.py:34`), temporal freq pairs /
    source-type count (`temporal.py:44`, `source_type.py:18`), technique count + threshold
    (`technique_head.py:16-17`, re-hardcoded at `pipeline.py:231`), kill-chain states
    (`kill_chain.py:15`), sentinel token list (`path_a.py:15-23`). Thread `cfg` through these
    constructors; define `DOMAINS` once in config.py and derive `DOMAIN_TAG_MAP` /
    `DOMAIN_ORDER` from it.

21. **Three hand-rolled attention implementations with copy-paste drift** —
    `encoders/blocks.py`, `decoder.py:_CausalSelfAttention`, `mtp_drafter.py:_MTPDecoderBlock`
    (the drafter copy already lost RoPE). Extract one `MultiHeadAttention(causal=, rope=)` in
    blocks.py on SDPA and reuse.

22. **Dead/duplicated decision machinery** — `DecisionStateClassifier` is instantiated by the
    pipeline but never called (dead params in checkpoints/ONNX) and contains a second copy of
    the DCAT override; keep `apply_dcat_override` only. Kill-chain transition logits are
    discarded per block (`epistemic_fusion.py:110`), blocking the EXPERIMENT-2 multi-task
    objective — return them. ATT&CK head reads pre-fusion input (`pipeline.py:228`) instead of
    the fused representation — expose final fusion hidden state in `EarlyExitResult`.

23. **Assorted**: `MAX_AGENTIC_TURNS` never enforced (`function_calling.py:23`); AFN ignores
    batch dim (scores example 0, applies its indices to whole batch — assert B==1 or
    vectorize, `afn.py:39-98`); "RotatE" in `losses.py:119-122` and `path_c.py:30-32` isn't
    RotatE (no complex rotation / wrong normalization); GCN degree features computed from the
    normalized adjacency collapse to 0/1 (`path_c.py:125-127`); ontology node→token alignment
    is positional and arbitrary (`ontology.py:49-57`); long-context encoder accepts no padding
    mask and its docstring layer schedule contradicts the code (`long_context.py:134-146`);
    sliding window is effectively 8193 tokens vs the spec's 4096 (`long_context.py:54-57`);
    drafter registers the shared 50428×1024 embedding in its own state_dict (double-serialized,
    `mtp_drafter.py:79-82`); fabricated technique IDs `T1000+i` (`pipeline.py:231`); two
    different `AFNScore` classes; dead imports/constants in path_a, tokenizer.py, afn.py,
    kill_chain.py, dag_head.py, vocab_mask.py, pipeline.py (`DOMAIN_D_MODEL`); justification
    trace is a placeholder string, never detokenized (`pipeline.py:248-271`).

---

## Recommended fix order

| Phase | Items | Why |
|---|---|---|
| A (now) | 2, 3, 4, 5, 6 | Hard-constraint compliance + silent-training-bug traps; small diffs |
| B (before training) | 7, 8, 9, 10, 11, 19, 20 | Anything that corrupts gradients/checkpoints is cheapest to fix pre-training |
| C (before latency claims) | 12, 13, 14, 15, 16, 17 | SDPA + KV cache + buffer caching; the locked 90/120 ms and 4 GB targets depend on these |
| D (when wiring real data) | 1, 18, 21, 22, 23 | Pipeline realism, ONNX, structural consolidation |
