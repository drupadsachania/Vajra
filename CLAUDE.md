# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Is

A specification-and-lock repository for **SecureFoundation** — a security-native, ~1.16B parameter encoder-decoder transformer targeting sub-4B inference on a single GPU for real-time SOC operations. There is no application code yet. The three files here are the canonical source of truth that any implementation must conform to.

| File | Purpose |
|------|---------|
| `ARCHITECTURE.md` | Full implementation-ready spec (13 sections, all numeric targets, training pipeline, integration contracts) |
| `REQUIREMENTS.lock` | Machine-readable JSON of every locked decision, numeric value, and hard constraint extracted from the spec. Contains SHA-256 of `ARCHITECTURE.md` to detect drift. |
| `validate_requirements.py` | Stdlib-only Python script that enforces compliance |

## Commands

```bash
# Check that ARCHITECTURE.md has not been edited since lock-in
python validate_requirements.py

# Validate an implementation config JSON against all locked values
python validate_requirements.py --impl path/to/impl_config.json
```

`validate_requirements.py` uses only Python stdlib (no install needed). Exit 0 = compliant; exit 1 = failure with specific mismatch details printed.

## Architecture at a Glance

**Seven parallel domain encoders** (Detection, Forensics, CTI, Vulnerability, Identity/Access, Incident Response, Compliance) run unconditionally and independently on every inference call — no shared weights, no cross-encoder attention. Each produces a single CLS token that feeds into a **6-block epistemic fusion layer** (F1–F6). The fusion layer performs cross-domain cross-attention and taps discrete Kairos ECL states (ACT/ESCALATE/DEFER/FAIL_SAFE) at F3 (fast exit) and F6 (full). A **constrained justification decoder** (8 layers) is invoked only for ACT/ESCALATE states; it cross-attends only over AFN-scored evidence tokens, never raw encoder states, and its vocabulary is hard-masked at logit level.

Input enters via three parallel tokenization paths all projecting to d=1024: S-TOON sentinel-bounded text, a numerical sub-tokenizer (CVSS, NetFlow — never BPE), and a graph-embedding pipeline (TransE/RotatE + GCN for MITRE; inference-time GCN for Argus RAG-provided graphs). Interpretability is via **Activation Flow Networks** (L2-norm of hidden states at the Layer-8 analog), not attention heatmaps.

**Integration context:**
- **Argus XDR** (Go backend): calls SecureFoundation via protobuf; provides RAG subgraphs; receives ECL state + evidence DAG + AFN scores
- **Kairos ECL**: the four-state epistemic control layer that consumes SecureFoundation's classifier output
- **Themis**: agentic orchestration layer; SecureFoundation is a synchronous oracle within Themis dispatch, not an agent itself

## The Lock Workflow

`REQUIREMENTS.lock` has a `_meta.spec_sha256` field. **Any edit to `ARCHITECTURE.md` breaks validation** until `REQUIREMENTS.lock` is regenerated:

```bash
# After editing ARCHITECTURE.md, recompute and update the checksum:
python -c "import hashlib; print(hashlib.sha256(open('ARCHITECTURE.md','rb').read()).hexdigest())"
# Then update _meta.spec_sha256 in REQUIREMENTS.lock with the new hash
```

All four operator-locked decisions (`fusion_architecture`, `output_architecture`, `tokenization`, `knowledge_boundary`) are in `REQUIREMENTS.lock` under `operator_locked_decisions`. Changes require explicit operator sign-off — do not silently change them.

## Three Blocking Experiments

These must be settled empirically before the relevant implementation sections are finalized. They are marked `"EXPERIMENT-1"`, `"EXPERIMENT-2"`, `"EXPERIMENT-3"` in both `ARCHITECTURE.md` (§12) and `REQUIREMENTS.lock` (`experiments_blocking_implementation`).

| ID | What it resolves | Key variable |
|----|-----------------|-------------|
| EXPERIMENT-1 | DCAT baseline window N (§4.3) and latency budget (§10.1) | Sweep N ∈ {32,64,128,256,512} events; target FNR ≤ 0.08, VRAM < 40GB |
| EXPERIMENT-2 | Multi-task training stability of fusion + kill-chain state + ECL (§4.4, §5, §8) | F1 drop across all three objectives must stay ≤ 0.03 after epoch 5 |
| EXPERIMENT-3 | Curriculum ordering for DARPA OpTC pretraining (§8.1) | Isolated-node-first vs. full-graph-from-epoch-1 |

## Hard Constraints (Non-Negotiable)

These are encoded in `REQUIREMENTS.lock` under `hard_constraints` and checked by the validator:

- Total parameters ≤ 4B (current estimate ~1.16B)
- No exploit synthesis in weights; no free-text attack-tooling output
- Decoder vocabulary mask enforced at logit level (`p = −∞`), not sampling level
- No real IP addresses in training data (RFC1918 and RFC5737 documentation ranges only)
- ECL state markers emitted at F3 and F6 tap points on every inference call
- AFN (not raw attention weights) for all interpretability surfaces
- Apache 2.0 license on all weights, code, and training data
