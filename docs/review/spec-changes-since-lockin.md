# Spec Changes Since Architecture Lock-In

Diff of the canonical spec (`ARCHITECTURE.md` + `REQUIREMENTS.lock`) from the
original lock-in to the current locked state. Both files are re-locked and in
sync — `validate_requirements.py` reports `SPEC INTEGRITY OK`
(SHA-256 `cd82b4df…3c1b53e`).

## Provenance (git)

| Commit | Spec change |
|--------|-------------|
| `89fe2f0` / `ec3f252` | **Original lock-in** — SecureFoundation ARCHITECTURE.md + first REQUIREMENTS.lock |
| `46ceafa` | **12 corrections** — decouple from Argus XDR / Kairos ECL / Themis; replace §10 with platform-agnostic interface; add §10.6 function calling |
| `e1c1fa5` | **4 additions** — wNa8o8 QAT from epoch 0, MTP drafter co-training, 256K interleaved-RoPE/NoPE context, EXPERIMENT-1 NoPE sub-hypothesis |

Aggregate diff `89fe2f0 → HEAD`: **ARCHITECTURE.md** +695 / −272 lines;
**REQUIREMENTS.lock** +476 / −0 (the lock was substantially expanded, not rewritten).

---

## 1. Platform decoupling (commit `46ceafa`)

The spec was rewritten to be deployment-agnostic. Vajra emits integer decision
class indices `{0,1,2,3}`; consuming systems map those to their own state names.

| Before (lock-in) | After |
|------------------|-------|
| `# SecureFoundation — Architecture Specification` | `# Vajra — Architecture Specification` |
| `### 6.3 Kairos ECL State Classifier` | `### 6.3 Decision State Classifier` |
| `### 7.2 Surfacing AFN Output to Tier-3 Analysts` | `### 7.2 Surfacing AFN Output to Analysts` |
| `### 8.4 Stage 4 — ECL Alignment via DPO` | `### 8.4 Stage 4 — Decision State Alignment via DPO` |
| Named states `ACT / ESCALATE / DEFER / FAIL_SAFE` | Integer classes `0 / 1 / 2 / 3` |

**§10 was replaced wholesale** — from three vendor-integration subsections to a
platform-agnostic interface contract:

| Before — "Section 10: Integration Specification" | After — "Section 10: Model Interface Specification" |
|--------------------------------------------------|-----------------------------------------------------|
| 10.1 Argus XDR Integration (Go backend) | 10.1 Input Contract (`VajraInferenceRequest`) |
| 10.2 Kairos ECL Confidence-to-State Mapping | 10.2 Output Contract (`VajraInferenceResponse`) |
| 10.3 Themis Integration | 10.3 Latency Budget (A100 80GB, 4096 tok, bs=1) |
| *(protobuf wire format)* | 10.4 ONNX Export |
| | 10.5 Knowledge Boundary |
| | **10.6 Function Calling & Agentic Workflow Interface** (new) — tool-call JSON format, 6-tool native vocabulary, multi-turn format (≤8 turns), structured output mode |

The four **operator-locked decisions** were formalized in `REQUIREMENTS.lock`
(`fusion_architecture`, `output_architecture`, `tokenization`,
`knowledge_boundary`), each recording the chosen option, the rejected
alternatives, and a Decision-Log section reference. Knowledge boundary fixes the
baked-in content to **MITRE ATT&CK Enterprise v16 only** (14 tactics, ~700
techniques, frozen after Stage 1); everything else (CVE, D3FEND, IOC, MISP,
Sigma, post-v16 sub-techniques) arrives via consuming-system RAG.

---

## 2. Capability additions (commit `e1c1fa5`)

New `REQUIREMENTS.lock` blocks and the §-sections that back them:

### 2.1 QAT from epoch zero (§8.1.1, new)
`qat` block: wNa8o8 schema — channel-wise **2-bit** decoder weights + **8-bit**
static KV caches on encoders/fusion, initialized at epoch 0 (QAT, not PTQ).

### 2.2 MTP drafter co-training (§8.5, new)
`mtp_drafter` block: 2-layer / d=512 speculative drafter co-trained with the
primary; locks `decoder_latency_ms_with_drafter: 50` vs
`decoder_latency_ms_without_drafter: 90`.

### 2.3 256K context for Detection & Forensics (§4.1)
Detection and Forensics encoders gain:
```
context_window_default_tokens : 32768
context_window_max_tokens     : 262144     # 256K
attention_pattern             : interleaved_RoPE_NoPE
global_layer_positional_encoding : NoPE
```
The other five encoders are pinned at `context_window_tokens: 4096`. DCAT
baseline cache is a `separate_KV_cache_not_consuming_context_window`.

### 2.4 EXPERIMENT-1 sub-hypothesis (§12)
Added a secondary hypothesis: beyond a threshold baseline window N the
contrastive loss receives **noise-dominated gradients from the NoPE global
layers** (which cannot encode positional distance, so cannot distinguish a
recent absence from a distant one). The sweep gains a 4th metric — *contrastive
loss gradient variance across the NoPE global layers*, flagged if it exceeds 2×
the N=32 baseline. The §10.1→§10.3 latency cross-reference was repointed.

### 2.5 Decision-state formalization (REQUIREMENTS.lock)
- `decision_state_output: integer_class_index_0_1_2_3` with
  `decision_state_class_map` `{0: HIGH_CONFIDENCE_ACTION, 1: ESCALATE_FOR_REVIEW,
  2: DEFER, 3: INSUFFICIENT_CONTEXT}`
- `decision_state_markers_required: true` (F3 and F6 taps, every call)
- DCAT override rule locked:
  `if_dcat_D_gt_theta_and_decision_class_not_0_or_1_force_class_1`
- DCAT baseline `baseline_frozen_on_decision_classes: [1, 3]`
- Four-stage training pipeline named:
  `[Stage_1_pretraining, Stage_2_CoT_distillation, Stage_3_BaNEL_absence,
  Stage_4_DPO_decision_alignment]`
- `swarm_safety_parity_exact_decision_match`: target 0.90 over 200 scenarios vs
  a 3-agent GPT-4o-mini orchestrated baseline

---

## Net effect

No locked numeric value from the original spec was *weakened*; the changes are
(a) a vendor→platform-agnostic rewrite of the interface surface and naming, and
(b) net-new capability locks (QAT, MTP drafter, 256K context) plus one
experiment refinement. The hard-constraint set (≤4B params, logit-level vocab
masking, no real IPs, AFN-only interpretability, Apache-2.0) is unchanged.
