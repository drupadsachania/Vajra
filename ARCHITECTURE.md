# SecureFoundation — Architecture Specification

**Version:** 1.0  
**License:** Apache 2.0  
**Status:** Implementation-ready  
**Operator decisions locked:** 2026-06-07  

---

## Section 0 — Decision Log

Every major architecture decision, the chosen option, the finding that drove it, and the alternatives
rejected. Read this first.

| # | Decision | Chosen | Finding | Alternatives Rejected |
|---|----------|--------|---------|----------------------|
| 0-1 | Fusion architecture | **Parallel domain-specific encoders → epistemic fusion at final layer** | Architecture Decision Input #1: MoE sparse routing produces measurable GPU side-channel footprints and obfuscates which sub-network drove the epistemic conclusion | Dense early-fusion cross-attention; Mixture-of-Experts (MoE) |
| 0-2 | Output architecture | **T5-style encoder-decoder with constrained justification decoder** | Architecture Decision Input #2: encoder-only lacks autoregressive capability for analyst-readable kill-chain traces; latent CoT yields probing inconsistency incompatible with ECL tap requirement | Encoder-only; latent CoT (CoLaR/Coconut) |
| 0-3 | Tokenization | **Modular multi-tokenizer: numerical sub-tokenizer + graph-embedding pipeline + S-TOON sentinel-bounded text tokenizer → shared d=1024** | Architecture Decision Input #3; Gap: Unified Multimodal Tokenization Paradox; Confirmed Borrowing: S-TOON | Universal BPE; single-modality text tokenizer |
| 0-4 | Knowledge boundary | **MITRE tactic/technique embeddings baked in (14 tactics, ~700 techniques); all CVE data, D3FEND, identity graphs, live threat intel via Argus RAG** | Architecture Decision Input #4 (operator chose minimal baked-in boundary); Confirmed Borrowing: Ontological KG Embeddings | Larger static KG including CVE/D3FEND baked in; fully RAG-dependent at inference |
| 0-5 | Interpretability method | **Activation Flow Networks (AFN): L2-norm hidden-state importance at intermediate layers** | Surprise #1: "attention is not explanation" — raw attention maps highlight syntactic delimiters, not causal tokens; AFN provides mathematically provable input→output links | Raw attention heatmaps |
| 0-6 | Absence/null-state reasoning | **Dual-Contrastive Attention (DCAT) + BaNEL contrastive loss** | Confirmed Borrowing: DCAT; Surprise #3: standard RL on null rewards produces zero gradients and collapses to random search | Standard self-attention; policy-gradient RL |
| 0-7 | Temporal encoding | **Continuous-time encoding (ContiFormer/TFT): time as continuous variable, learnable frequencies** | Confirmed Borrowing: Continuous-Time / TFT — discrete sinusoidal indices treat ms-apart and month-apart events as equally adjacent | Discrete sinusoidal positional embedding; learned relative position |
| 0-8 | Ontological knowledge representation | **TransE + RotatE + GCN on MITRE ATT&CK graph; embeddings frozen after Stage 1** | Confirmed Borrowing: Ontological KG Embeddings — TransE/RotatE capture relational translation semantics; GCN captures local structural dependencies | Inference-time RAG for static MITRE facts; hand-coded rule embeddings |
| 0-9 | Reasoning output style | **Discrete ECL markers at depth-tapped fusion layers; no latent CoT** | Gap: Interpretable Latent CoT (probing inconsistency); Surprise #2: small models punish verbose CoT — sentinel-only markers outperform NL deliberation | Verbose NL chain-of-thought; depth-recurrent latent states |
| 0-10 | MoE routing | **Rejected** | Architecture Decision Input #1: sparse routing produces GPU execution-telemetry side-channel observable non-intrusively; routing obfuscates epistemic attribution | N/A |

---

## Section 1 — System Overview

### 1.1 Purpose

SecureFoundation is a security-native encoder-decoder transformer that accepts heterogeneous
cybersecurity telemetry from seven security domains in a single inference call, produces discrete
epistemic state markers mapped to the Kairos ECL (ACT / ESCALATE / DEFER / FAIL_SAFE),
calibrated confidence scores, ATT&CK technique classifications, evidence-chain DAGs, and
human-auditable activation-flow justification traces.

It is designed to operate within the Argus XDR environment: Argus provides dynamic threat
intelligence via RAG; SecureFoundation provides the analytical inference layer. Themis uses
SecureFoundation's ECL state output as a dispatch signal for downstream agentic actions.

### 1.2 The Seven Domains

| # | Domain | Primary Input Types |
|---|--------|-------------------|
| 1 | Detection & Alerting | NetFlow, PCAP features, SIGMA-matched EVTX, EDR telemetry |
| 2 | Forensics & Investigation | Provenance graphs (eCAR JSON, Zeek FLOW), host timeline events |
| 3 | Threat Intelligence | STIX 2.1 objects, MISP galaxies, CTI PDF/HTML (extracted), APT notes |
| 4 | Vulnerability & Risk | CVE JSON, CVSS vectors, ExploitDB entries, source code diffs |
| 5 | Identity & Access | Windows EVTX auth events, LDAP schema objects, CSV auth logs |
| 6 | Incident Response | CISA advisories, SOAR playbook YAML, structured IR reports |
| 7 | Compliance & Posture | CIS Benchmark XML (OVAL), NIST 800-53 JSON, ISO 27001 controls |

### 1.3 What This Is NOT

- Not a generative attack-tooling model. No output head produces exploit code, attack scaffolding, or
  reconnaissance instructions. This constraint is structural, not policy: the decoder vocabulary is
  hard-masked at logit level.
- Not a replacement for a Tier-3 analyst. SecureFoundation produces discrete state and evidence; the
  analyst acts.
- Not a general-purpose LLM. Domain encoders are trained exclusively on security telemetry.
- Not an orchestration engine. Themis orchestrates; SecureFoundation infers.
- Not a replacement for Argus's rule engine. Sigma rule matching remains in Argus; SecureFoundation
  receives matched-event output as input.

### 1.4 Core Thesis: Monolith Rivaling Multi-Agent Swarm Safety

The most advanced deployed security AI systems (CrowdStrike Charlotte AI, Palo Alto XSIAM,
Microsoft Security Copilot) deliberately fracture reasoning into siloed agents to isolate logic,
prevent cross-domain hallucination, and enforce authorization controls [Finding: Surprise #4].

SecureFoundation achieves equivalent safety guarantees within a monolithic architecture via
**rigid parallel stream isolation:**

- Each of the seven domain encoders is a completely independent parameter namespace. No weights
  are shared across domain stacks.
- Domain encoders do not receive each other's intermediate activations at any layer. A domain
  encoder's output is a single CLS-token hidden state vector; it cannot leak its intermediate
  representations to adjacent encoders.
- Cross-domain attention is permitted **only** at the epistemic fusion layer (F1–F6), under a
  read-only contract: the fusion layer cross-attends over domain CLS outputs, never over raw
  domain sequence activations.
- The constrained decoder receives **only** AFN-derived activation vectors (the top-16 scored
  input tokens by L2-norm importance), never raw encoder hidden states.
- Gradient barriers (`stop_grad` operations) are applied at domain encoder outputs during
  domain-specific fine-tuning, preventing cross-domain gradient contamination.

These constraints are implemented in the model code, not in policy. A prompt cannot override them.

### 1.5 Success Criteria

| Metric | Target | Baseline |
|--------|--------|---------|
| ATT&CK stage attribution F1 on DARPA OpTC | ≥ 0.87 | SecureBERT 2.0: ~0.71 |
| Kairos ECL state macro F1 (4-class) | ≥ 0.91 | No published baseline |
| CVSS per-metric classification accuracy | ≥ 0.95 | CVSS-BERT: ~0.89 |
| Mean inference latency (4096-token input, single A100 80GB) | ≤ 120 ms | — |
| Expected Calibration Error (ECE) on OOD distribution | ≤ 0.04 | — |
| AFN trace generation overhead | ≤ 15 ms | — |
| Swarm-safety parity: ECL state exact-match vs. 3-agent baseline | ≥ 90% on 200 DARPA OpTC scenarios | — |
| Decoder trace generation latency (ACT/ESCALATE only) | ≤ 50 ms additional | — |

---

## Section 2 — Input and Tokenization Layer

All input regardless of origin passes through exactly one of three ingestion paths, each outputting
tensors of dimension d=1024. These tensors are concatenated along the sequence dimension, tagged
with source-type embeddings, and routed to the appropriate domain encoder. [Finding: Architecture
Decision Input #3; Gap: Unified Multimodal Tokenization Paradox; Confirmed Borrowing: S-TOON]

### 2.1 Path A — Sentinel-Bounded Text Tokenizer

**Scope:** Windows EVTX JSON, AWS CloudTrail, EDR telemetry JSON, CTI reports, Sigma rules,
STIX 2.1 objects, SOAR playbook YAML, CIS/NIST control definitions.

**Method:** S-TOON adaptive neuro-symbolic protocol — schema-aware sentinel tokens replace
delimiter characters (brackets, colons, quotes) as explicit boundary markers, preventing delimiter
dissolution and type smuggling in small models [Finding: Confirmed Borrowing: S-TOON]. BPE is
applied only to natural-language string values inside sentinel-delimited spans.

**Full sentinel token set** (each is a single guaranteed-atomic vocabulary entry):

| Token | Role |
|-------|------|
| `<\|S_START\|>` | Structure block open |
| `<\|S_END\|>` | Structure block close |
| `<\|S_KEY\|>` | JSON key delimiter |
| `<\|S_VAL\|>` | JSON value delimiter |
| `<\|S_ARR\|>` | Array element boundary |
| `<\|S_NULL\|>` | Explicit null / absent field marker (critical for DCAT absence reasoning) |
| `<\|S_DOMAIN:detection\|>` | Domain routing: Detection |
| `<\|S_DOMAIN:forensics\|>` | Domain routing: Forensics |
| `<\|S_DOMAIN:cti\|>` | Domain routing: CTI |
| `<\|S_DOMAIN:vuln\|>` | Domain routing: Vulnerability/Risk |
| `<\|S_DOMAIN:identity\|>` | Domain routing: Identity/Access |
| `<\|S_DOMAIN:ir\|>` | Domain routing: Incident Response |
| `<\|S_DOMAIN:compliance\|>` | Domain routing: Compliance |
| `<\|S_EVTX\|>` | Schema type: Windows EVTX |
| `<\|S_STIX\|>` | Schema type: STIX 2.1 |
| `<\|S_SIGMA\|>` | Schema type: Sigma rule |
| `<\|S_CLOUD\|>` | Schema type: CloudTrail / cloud event |
| `<\|S_LDAP\|>` | Schema type: LDAP / directory object |
| `<\|S_BASELINE\|>` | DCAT dual-stream: baseline state marker |
| `<\|S_OBSERVED\|>` | DCAT dual-stream: observed state marker |
| `<\|ECL_ACT\|>` | Kairos ECL output (decoder only) |
| `<\|ECL_ESCALATE\|>` | Kairos ECL output (decoder only) |
| `<\|ECL_DEFER\|>` | Kairos ECL output (decoder only) |
| `<\|ECL_FAIL_SAFE\|>` | Kairos ECL output (decoder only) |

The four `<\|ECL_*\|>` tokens appear in decoder output vocabulary only; they are masked from
encoder input paths.

**Projection:** Path A output is already in d=1024 via the shared token embedding matrix.

### 2.2 Path B — Numerical Sub-Tokenizer

**Scope:** NetFlow statistics (bytes, packets, duration, ports, flags), CVSS vector components, EPSS
scores, packet metadata (TTL, window size), host performance metrics, authentication categorical
fields (logon type, auth protocol).

These fields are never passed through BPE. BPE fragmentation destroys the categorical topology of
CVSS and the network-topology semantics of IP addresses and ports [Finding: Gap: Unified
Multimodal Tokenization Paradox].

**CVSS vector processing:**  
Parse the vector string as nine discrete categorical/ordinal components:

| Component | Type | Cardinality |
|-----------|------|------------|
| Attack Vector (AV) | Categorical | 4 (N/A/L/P) |
| Attack Complexity (AC) | Categorical | 2 (L/H) |
| Privileges Required (PR) | Categorical | 3 (N/L/H) |
| User Interaction (UI) | Categorical | 2 (N/R) |
| Scope (S) | Categorical | 2 (U/C) |
| Confidentiality (C) | Ordinal | 3 (N/L/H) |
| Integrity (I) | Ordinal | 3 (N/L/H) |
| Availability (A) | Ordinal | 3 (N/L/H) |
| CVSS Version | Categorical | 3 (2.0/3.0/3.1) |

One-hot encode each component → concatenate (total dim = 25) → MLP(25 → 512 → 1024 with GELU) → single token of d=1024.

**NetFlow processing:**
- `bytes`, `packets`: log₁₀(x + 1) → z-normalize using training-set statistics → Linear(1 → 1024)
- `duration`: continuous-time encoding (see Section 3 temporal embeddings)
- `src_port`, `dst_port`: learnable embedding lookup (65,536 entries × 64-dim) → Linear(64 → 1024)
- `protocol`: one-hot (256) → Linear(256 → 1024)
- `tcp_flags`: 8-bit bitmask → Linear(8 → 1024)
- IP addresses: `/24` subnet one-hot over private address spaces (RFC1918: 10.x.x.x, 172.16.x.x,
  192.168.x.x) + RFC5737 documentation ranges; real public IPs are never in training data; a 1-bit
  external/internal indicator is appended

**Projection:** All Path B outputs are d=1024 tensors via the above per-field linear projections.

### 2.3 Path C — Graph-Embedding Pipeline

**Scope:** MITRE ATT&CK tactic/technique nodes (baked into weights at pretraining), and
inference-time subgraphs provided by Argus XDR (AD topology fragments, asset dependency
graphs, CVE-to-asset linkages).

**Baked-in (MITRE only):**  
TransE + RotatE embeddings trained on MITRE ATT&CK Enterprise v16 graph (14 tactic nodes, ~700
technique nodes, ~400 sub-technique nodes, procedural edges: `prerequisite_for`, `enables`,
`mitigated_by`, `subtechnique_of`). [Finding: Confirmed Borrowing: Ontological KG Embeddings]  
- Entity embedding dimension: d_kg = 256  
- Relation embedding dimension: d_rel = 256  
- TransE objective: minimize ||h + r − t||₂  
- RotatE objective: minimize ||h ∘ r − t||₂ in complex space  
- 2-layer GCN (d=256) run over full MITRE graph post-training; node features = TransE + RotatE  
- Final frozen MITRE embeddings: GCN output → Linear(256 → 1024), frozen after Stage 1 pretraining  
- Any input token whose text matches an ATT&CK technique ID (e.g., `T1003`) has its baked MITRE
  embedding added to its token embedding (see Section 3)  

**RAG-provided (Argus, inference-time):**  
Argus returns `rag_nodes` and `rag_edges` per request (see Section 10 protobuf contract).  
- If subgraph size > 1,024 nodes: apply top-k degree-centrality node sampling, k=256, before GCN  
- 2-layer GCN (d_graph=512, unfrozen) encodes the subgraph; Graphormer structural features
  (centrality encoding, degree encoding, shortest-path bias) concatenated before projection  
- Projection: Linear(512 + graphormer_features → 1024)  
- Each encoded graph node becomes one token of d=1024 in the sequence

**Projection:** All Path C outputs are d=1024 tensors.

### 2.4 Worked Tokenization Examples

**Example 1 — CVSS vector `CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H`:**
```
Path: B (Numerical)
Parse: AV=N, AC=L, PR=N, UI=N, S=U, C=H, I=H, A=H, Version=3.1
One-hot: [1,0,0,0, 1,0, 1,0,0, 1,0, 1,0, 0,0,1, 0,0,1, 0,0,1, 0,1,0]  → dim=25
MLP(25→512→1024) → 1 token, d=1024
Source-type embedding: SRC_CVSS added
Domain tag: <|S_DOMAIN:vuln|> prepended
Routes to: Vulnerability/Risk encoder
```

**Example 2 — EVTX Event 4624 (Interactive Logon):**
```
Path: A (Sentinel text) for structural fields; B (Numerical) for LogonType and timestamps

Input stream:
  <|S_DOMAIN:identity|><|S_EVTX|><|S_START|>
  <|S_KEY|>EventID<|S_VAL|>4624
  <|S_KEY|>SubjectUserName<|S_VAL|>SYSTEM
  <|S_KEY|>TargetUserName<|S_VAL|>jsmith
  <|S_KEY|>LogonType<|S_VAL|>3
  <|S_KEY|>IpAddress<|S_VAL|>192.168.1.45
  <|S_KEY|>WorkstationName<|S_VAL|>WS-FINANCE-04
  <|S_END|>

LogonType=3 → Path B categorical (one-hot 13 logon type values)
IpAddress=192.168.1.45 → Path B (RFC1918 /24 one-hot + internal indicator bit)
User names → Path A BPE inside <|S_VAL|> spans
Routes to: Identity/Access encoder
```

**Example 3 — STIX 2.1 intrusion-set object:**
```
Path: A (Sentinel text) for NL fields; C (Graph) for ATT&CK technique references

Input stream:
  <|S_DOMAIN:cti|><|S_STIX|><|S_START|>
  <|S_KEY|>type<|S_VAL|>intrusion-set
  <|S_KEY|>name<|S_VAL|>APT29
  <|S_KEY|>aliases<|S_ARR|>Cozy Bear<|S_ARR|>YTTRIUM
  <|S_KEY|>sophistication<|S_VAL|>advanced
  <|S_KEY|>uses<|S_ARR|>T1003<|S_ARR|>T1021<|S_ARR|>T1078
  <|S_END|>

T1003, T1021, T1078 → Path C (baked MITRE embeddings; MITRE embedding added to token embedding)
NL values (APT29, advanced) → Path A BPE
Routes to: CTI/STIX encoder
```

**Example 4 — NetFlow record:**
```
Path: B (Numerical) throughout

Fields: src_ip=192.168.5.22, dst_ip=10.0.0.1, sport=49152, dport=443,
        proto=TCP, bytes=14820, pkts=38, duration=0.842s, flags=0x18

Processing:
  bytes   → log10(14821) ≈ 4.17 → z-norm → Linear(1→1024)
  pkts    → log10(39) ≈ 1.59 → z-norm → Linear(1→1024)
  duration→ ContiFormer continuous-time encoding (0.842s)
  sport   → Embedding(65536, 64)[49152] → Linear(64→1024)
  dport   → Embedding(65536, 64)[443] → Linear(64→1024)
  proto   → one-hot(256)[6] → Linear(256→1024)
  flags   → 8-bit vector [0,0,0,1,1,0,0,0] → Linear(8→1024)
  src/dst → RFC1918 /24 + internal indicator

Source-type embedding: SRC_NETFLOW
Routes to: Detection/Network encoder
```

**Example 5 — Authentication event (LANL format):**
```
Path: A for user/host entities; B for categorical auth fields and timestamps

Record: 1234567890,U10@DOM1,U10@DOM1,C1,C2,Kerberos,Network,LogOn,Success

Timestamp → ContiFormer continuous-time encoding
src_user U10@DOM1 → BPE within <|S_VAL|>; if identity graph provided by Argus,
                     also Path C GCN node lookup
AuthType=Kerberos → Path B categorical (6 auth type values)
LogonType=Network → Path B categorical
Success → Path B binary (1-bit → Linear(1→1024))
Domain tag: <|S_DOMAIN:identity|>
Routes to: Identity/Access encoder
```

---

## Section 3 — Embedding Layer

The embedding layer transforms each token's d=1024 path output into a contextualized embedding
before domain encoder input:

```
e_final(i) = LayerNorm( E_token(i) + E_source_type(i) + E_temporal(i) + E_ontology(i) )
```

### 3.1 Token Embeddings

- Vocabulary size: 50,432  
  = 50,000 BPE tokens (trained on security corpora, not internet text)  
  + 24 sentinel tokens (Section 2.1)  
  + 8 ATT&CK technique atomic tokens (`T1003`, `T1021`, `T1078`, `T1059`, `T1055`, `T1047`, `T1071`, `T1083`)  
  + ~400 remaining ATT&CK technique IDs as atomic tokens (guaranteed single-token, prevent BPE fragmentation of technique IDs)  
- Embedding matrix: 50,432 × 1024 → ~52M parameters  
- ATT&CK technique tokens are initialized to their baked TransE/RotatE embeddings (projected from 256 → 1024); not randomly initialized  

### 3.2 Source-Type Embeddings

16 source types × 1024 → learnable lookup, ~16M parameters:

| ID | Type |
|----|------|
| 0 | EVTX |
| 1 | NetFlow |
| 2 | STIX |
| 3 | CVSS |
| 4 | Auth (CSV/LDAP) |
| 5 | PCAP features |
| 6 | CloudTrail |
| 7 | Sigma rule |
| 8 | CVE JSON |
| 9 | LDAP schema |
| 10 | Playbook YAML |
| 11 | CIS/NIST control |
| 12 | Graph node (RAG) |
| 13 | Graph edge (RAG) |
| 14 | Synthetic |
| 15 | Padding |

### 3.3 Continuous-Time Temporal Embeddings

[Finding: Confirmed Borrowing: Continuous-Time / TFT]

Standard sinusoidal positional encodings treat events separated by 2 ms and events separated by 2
weeks as equally adjacent, catastrophically failing to reconstruct APT kill-chain causal ordering.
ContiFormer/TFT continuous-time encoding treats time as a continuous variable:

For each event with timestamp `t` (Unix epoch seconds, float64):

```
φ(t) = [ cos(ω₁t), sin(ω₁t), cos(ω₂t), sin(ω₂t), …, cos(ω₁₂₈t), sin(ω₁₂₈t) ]
```

- 128 learnable frequency pairs → φ(t) ∈ ℝ²⁵⁶  
- Linear(256 → 1024) with GELU → temporal embedding  
- Frequencies ωᵢ are initialized log-uniformly from 10⁻⁷ (≈ annual scale) to 10³ (ms scale),
  then learned during pretraining. This covers the full millisecond-to-month dwell time range
  of APT campaigns without discrete binning.  
- TFT-style Gated Residual Network (GRN) applied on top of temporal features: learns which
  temporal scale (short-term behavioral change vs. long-term dwell) is most relevant per input  
- If `t` is null (event has no timestamp): a learned `<TEMPORAL_UNKNOWN>` embedding vector
  replaces φ(t)

### 3.4 Ontology Embeddings (MITRE only, baked in)

[Finding: Confirmed Borrowing: Ontological KG Embeddings]

MITRE ATT&CK Enterprise v16 graph:  
- 14 tactic nodes, ~700 technique nodes, ~400 sub-technique nodes  
- Edges: `prerequisite_for`, `enables`, `mitigated_by` (cross-reference to D3FEND at training time
  only — D3FEND not baked into inference weights per knowledge boundary decision), `subtechnique_of`  

Training procedure:
1. TransE: minimize ||h + r − t||₂ with margin γ=2, negative sampling ratio 64:1  
2. RotatE: minimize ||h ∘ r − t||₂ in complex space; provides relation-type awareness unavailable
   in TransE  
3. GCN (2 layers, d=256): node features initialized from TransE embeddings; learns structural
   neighborhood context (e.g., T1003 and T1021 become close in GCN space because of their
   `prerequisite_for` edge, encoding the implicit knowledge that credential dumping frequently
   precedes lateral movement)  
4. Final embedding: element-wise sum of GCN output and TransE embedding → Linear(256 → 1024)  

**These embeddings are frozen after Stage 1 pretraining.** They are injected as E_ontology(i) for
any token position i whose vocabulary ID maps to a known ATT&CK tactic or technique. For all
other tokens, E_ontology(i) = 0.

The operator's knowledge boundary decision: only MITRE tactics/techniques are baked; CVE data,
D3FEND mitigations, identity graph topology, and threat actor profiles are retrieved from Argus XDR
at inference time via Path C's inference-time GCN. This maximizes knowledge freshness at the cost
of RAG latency dependency (see Risk 4 in Section 13).

---

## Section 4 — Core Transformer Blocks

### 4.1 Domain Encoder Specifications

Seven parallel, independent encoder stacks. No shared weights. No cross-encoder attention at any
layer within these stacks. [Finding: Architecture Decision Input #1 — parallel encoders fusing at
epistemic layer]

| Domain Encoder | Layers (L) | d_model | Heads (H) | FFN Width | Est. Params | Sizing Rationale |
|----------------|-----------|---------|-----------|-----------|------------|-----------------|
| Detection/Network | 12 | 1024 | 16 | 4096 | ~200M | Richest public data (CICIDS 2017/18, UNSW-NB15, Sigma corpus); high-throughput event sequences demand depth [Finding: Domain Data Inventory — Detection usability 3/5 but volume high] |
| Forensics/Provenance | 12 | 1024 | 16 | 4096 | ~200M | DARPA OpTC 17.4B events; multi-stage APT provenance graphs require maximum encoder depth [Finding: Domain Data Inventory — Forensics usability 4/5] |
| CTI/STIX | 8 | 768 | 12 | 3072 | ~85M | Highly structured, standardized; SecureBERT 2.0 precedent demonstrates 150M encoder is sufficient for CTI NER [Finding: Domain Data Inventory — CTI usability 5/5; Key Reference: SecureBERT 2.0] |
| Vulnerability/Risk | 8 | 768 | 12 | 3072 | ~85M | 200K+ CVEs, CVSS structure reduces required representational depth [Finding: Domain Data Inventory — Vulnerability usability 4/5] |
| Identity/Access | 6 | 768 | 12 | 3072 | ~65M | Data-sparse domain; oversized encoder will overfit to synthetic UEBA data [Finding: Domain Data Inventory — Identity usability 2/5] |
| Incident Response | 6 | 768 | 12 | 3072 | ~65M | Data-sparse; ground-truth ECL labels almost entirely crowdsourced or synthetic [Finding: Domain Data Inventory — IR usability 2/5] |
| Compliance | 4 | 512 | 8 | 2048 | ~28M | Structural control matching; no deep sequential reasoning required [Finding: Domain Data Inventory — Compliance usability 3/5, disconnected from live telemetry] |

**All domain encoders share:**
- Pre-LayerNorm (more stable for fine-tuning than post-LN)
- Rotary Position Embeddings (RoPE) for within-encoder relative position
- Bidirectional (non-causal) self-attention
- GELU activation in FFN
- Dropout p=0.1 during training, p=0 at inference
- SwiGLU FFN variant (FFN width above is the gate width; intermediate = FFN_width × 2/3)

**Total domain encoder parameter budget:** ~728M

### 4.2 Head Group Specialization

Within each domain encoder, attention heads at each layer are assigned to functional groups. These
assignments are **design hypotheses that must be validated by probing classifiers, not assumed
operative from designation alone.** Head groups are soft-regularized via auxiliary probing losses
during Stage 2; they are not hard-wired. If a probing classifier fails its criterion, that group's
designation is dissolved and heads are reassigned to the highest-performing group.
[Finding: Architecture Decision Input #1 — interpretability warning on routing]

| Group Name | Heads per Layer | Hypothesis | Probing Validation Method | Pass Criterion |
|------------|----------------|-----------|--------------------------|---------------|
| Temporal-causal | 4 | Learn cross-event time-ordering dependencies | Linear probe: given two event representations, predict which is temporally prior | AUC ≥ 0.85 on held-out event pairs |
| Entity-relation | 4 | Attend to coreferent security entities (same user/host across events) | Linear probe: predict coreference labels between entity mentions at different sequence positions | F1 ≥ 0.78 on coreference test set |
| Anomaly-contrast | 2 | Weight absent/unexpected token positions over present ones (DCAT stream) | Linear probe: predict null-state binary label from head activations alone | F1 ≥ 0.80 on null-signal validation set |
| Schema-structural | 4 | Maintain JSON hierarchy and sentinel boundary structure | Linear probe: predict JSON nesting depth and schema type from intermediate activations | Accuracy ≥ 0.88 on schema-type test set |
| (Fusion layer only) Domain-bridge | 2 | Cross-attend across domain CLS outputs | Linear probe: predict which domain pair contributed most to final classification | F1 ≥ 0.75 on domain-attribution test set |

Head counts above apply to encoders with H=16 (Detection, Forensics). For H=12 encoders
(CTI, Vuln, Identity, IR): Temporal-causal=3, Entity-relation=3, Anomaly-contrast=2,
Schema-structural=4. For H=8 (Compliance): Temporal-causal=2, Entity-relation=2,
Anomaly-contrast=1, Schema-structural=3.

### 4.3 Dual-Contrastive Attention (DCAT) Blocks

[Finding: Confirmed Borrowing: DCAT; Gap: Absence of Null-State and Negative Evidence Reasoning]

Integrated into Detection/Network encoder (Layers 7–8) and Forensics/Provenance encoder (Layers
7–8). These are augmented attention blocks that replace standard self-attention at these specific
layer positions.

**Architecture:**

```
DCAT_block(X, B):
  Q_obs, K_obs, V_obs = linear projections of X (observed stream)
  Q_base, K_base, V_base = linear projections of B (baseline stream)

  A_observed = softmax(Q_obs @ K_obs.T / √d_k)
  A_baseline = softmax(Q_base @ K_base.T / √d_k)

  D_divergence = KL(A_baseline || A_observed)    # per-head, per-position scalar

  if any(D_divergence > θ_divergence):
      escalation_signal = True
      anomaly_contrast_head_group receives D_divergence as additional input signal

  output = MHA_observed + α * MHA_baseline       # α = 0.3, tunable
  return LayerNorm(X + output), D_divergence
```

**Baseline state mechanism:**  
- `B` is a rolling exponential moving average of the last N inference calls for the same entity
  (host ID or user ID, resolved from the domain routing tag)  
- `B` is stored in a per-entity KV cache, separate from the primary sequence KV cache; it consumes
  no context window tokens from the model's sequence budget  
- `B` is updated with EMA coefficient β=0.9 after each inference call where ECL state ∈
  {DEFER, ACT} (non-anomalous baseline)  
- `B` is frozen (not updated) when ECL state = ESCALATE or FAIL_SAFE, preserving the pre-anomaly
  baseline  
- When an expected field is absent (a `<|S_NULL|>` sentinel appears at a position where a non-null
  token was expected per the schema type): D_divergence is guaranteed to be high by construction,
  because the baseline expects a present token and the observed stream delivers absence  

**EXPERIMENT-1:** The optimal value of N (temporal window size for baseline retention, number of
prior events to retain per entity) before contrastive attention degrades on single-GPU VRAM is
empirically unknown. The memory requirement scales as O(N × d_model × L_DCAT). At d=1024 and
2 DCAT layers, N=512 consumes ~4GB VRAM per entity baseline cache — unviable if tracking many
entities simultaneously. See Section 12 for experiment design. [Finding: Open Question #2]

### 4.4 Epistemic Fusion Layer

6 transformer blocks (F1–F6) that perform cross-domain attention over the seven domain CLS tokens.

**Architecture:**
- Input: 7 CLS tokens, each d=1024, one per domain encoder (domains whose input was empty receive
  a learned `<DOMAIN_ABSENT>` CLS token)
- d_model = 1024, H = 16, FFN = 4096
- Bidirectional cross-attention (each domain CLS can attend to all other domain CLS tokens)
- Parameter budget: ~100M

**ECL tap points:**
- **F3 fast tap:** After fusion block F3, a linear head Linear(1024 → 4) produces a preliminary ECL
  state. If confidence(DEFER) > 0.92 OR confidence(FAIL_SAFE) > 0.85, inference short-circuits
  here (early exit). This eliminates fusion F4–F6 FLOPs for clear low-confidence inputs.
- **F6 full tap:** After fusion block F6, the primary ECL state, ATT&CK classification, and
  evidence-chain DAG are computed.

**Kill-chain state tracking (LEGO-style semiautomaton):**  
[Finding: Adjacent Borrowing: LEGO/algorithmic state tracking]

A 7-bit discrete state vector S ∈ {RECON, WEAPONIZE, DELIVER, EXPLOIT, INSTALL, C2, EXFIL}
is maintained as a register tensor throughout the fusion layer:
- State transitions are predicted by a learned transition head: Linear(1024 → 7) applied to each
  fusion layer's CLS representation
- Training objective: next-state cross-entropy over ATT&CK technique sequence walks derived from
  DARPA OpTC provenance graphs
- The state machine never skips stages: transition from RECON directly to EXFIL requires explicit
  evidence path through intermediate states — this is enforced by a sequence-valid transition mask
  applied to the logits
- The current kill-chain state S is concatenated to the fusion CLS token at each layer as a
  7-dim binary vector, giving the fusion layer explicit state-tracking context

**EXPERIMENT-2:** Whether jointly optimizing (a) continuous cross-domain fusion attention, (b)
discrete kill-chain state tracking, and (c) discrete ECL state output is mathematically stable
and converges without objective conflict is unexplored in the security domain. See Section 12.
[Finding: Open Question #1]

---

## Section 5 — Reasoning Representation

### 5.1 Resolution of Open Question #1

The central tension: continuous latent reasoning (CoLaR/Coconut-style hidden-state recycling) vs.
discrete epistemic markers.

**Research findings:**
- Latent CoT (Huginn-3.5B, CoLaR): exhibits significant probing inconsistencies across recurrent
  blocks; hidden-state interpretability depends on decoding method; incompatible with SOC
  auditability [Finding: Gap: Interpretable Latent CoT]
- Small models (≤4B) actively punish verbose natural language CoT: forcing explicit verbal reasoning
  over structured protocols introduces severe instability, hallucinations, and unacceptable latency
  [Finding: Surprise #2]
- Kairos ECL requires deterministic discrete state markers at specific layer depths, which is
  structurally incompatible with non-deterministic recurrence depth

**Chosen approach: Depth-segmented latent reasoning with discrete ECL tap points.**

The cross-domain causal chain (Detection → Identity → Vulnerability → IR) is processed as a
continuous cross-domain cross-attention operation in the fusion layer (F1–F6). No explicit
reasoning tokens are emitted during this process. The kill-chain state machine (Section 4.4)
provides a discrete representational scaffold that tracks APT stage transitions without
generating natural language.

Discrete ECL markers are extracted at two fusion layer depths (F3 fast, F6 full) via linear
classification heads. The constrained decoder (Section 6) is invoked separately and only on
analyst request.

**No latent CoT is implemented.** Hidden-state recycling is excluded because:
1. Probing inconsistency makes ECL tap points non-deterministic [Finding: Gap: Interpretable Latent CoT]
2. Kairos ECL requires deterministic discrete outputs [Hard Constraint: must emit discrete ECL markers]
3. Small model CoT penalty [Finding: Surprise #2]

### 5.2 Cross-Domain Causal Chain Representation

The fusion layer's cross-attention implicitly encodes the Detection → Identity → Vulnerability → IR
chain. The chain is not represented as text — it is represented as the attention pattern in the
fusion layer, which AFN (Section 7) extracts post-hoc as a causal DAG.

The constrained decoder (Section 6), when invoked, translates the AFN-scored input tokens into a
human-readable narrative of this chain. It does not generate the chain; it translates a
pre-computed structure.

### 5.3 APT State Machine

[Finding: Adjacent Borrowing: LEGO/algorithmic state tracking]

The model frames APT reconstruction as evaluating a semiautomaton over the MITRE ATT&CK kill
chain. Each input event sequence advances the state machine by one step; the model must predict
valid next states (never skipping stages without evidence). This is implemented as the kill-chain
state tracking objective described in Section 4.4, not as a separate module.

**EXPERIMENT-2** (stability of dual-objective optimization): See Section 12.

---

## Section 6 — Output Heads

### 6.1 Evidence Chain DAG

- **Input:** Fusion layer F6 hidden states for all domain CLS tokens (7 vectors × d=1024)
- **Architecture:** 2-layer MLP: Linear(7×1024 → 2048) → GELU → Linear(2048 → max_nodes²)
  where max_nodes = 32
- **Output:** Sparse directed adjacency matrix over 32 evidence nodes; each node is a (domain,
  event_index, technique_id) triple
- **Training objective:** Binary cross-entropy on ground-truth provenance edges from DARPA OpTC;
  node coverage loss penalizes missing evidence nodes

### 6.2 ATT&CK Technique Classifier

- **Input:** Fusion F6 CLS token (d=1024)
- **Architecture:** Linear(1024 → 716) covering MITRE ATT&CK Enterprise v16 techniques and
  sub-techniques
- **Multi-label:** sigmoid activation; threshold τ=0.35 applied at inference (calibrated on
  validation set)
- **Training:** Binary cross-entropy with label smoothing ε=0.1

### 6.3 Kairos ECL State Classifier

- **Architecture:** Linear(1024 → 4) applied at both F3 (fast) and F6 (full) tap points
- **Output:** Softmax over {ACT=0, ESCALATE=1, DEFER=2, FAIL_SAFE=3}
- **Early exit rule:** If F3 confidence(DEFER) > 0.92 or confidence(FAIL_SAFE) > 0.85 → return
  F3 classification without computing F4–F6 or Head 6.4
- **DCAT override:** If DCAT divergence D > θ_divergence in any encoder and no classification was
  already ESCALATE or ACT: force output to ESCALATE
- **Training:** Cross-entropy on DPO-annotated preference pairs (Stage 4)

### 6.4 Calibrated Confidence Scorer

- **Method:** Temperature scaling — single learnable scalar T per domain encoder output,
  applied to logits before softmax [Finding: Architecture Decision Input #2 — research demands
  calibrated confidence, not argmax]
- **Calibration procedure:** Post-training calibration on held-out OOD validation set (Splunk
  Attack Range synthetics not present in training corpus); optimize T by minimizing Negative
  Log-Likelihood (NLL) on OOD set
- **Additionally:** Conformal prediction sets computed offline at p=0.90 and p=0.95 coverage
  levels on the calibration set; these provide deployment-time coverage guarantees — the
  conformal set answers "which ECL states are plausible at 90% confidence" rather than a
  single argmax
- **Output:** Scalar confidence ∈ [0,1] + optional conformal prediction set of valid ECL states

### 6.5 Constrained Justification Decoder

[Finding: Architecture Decision Input #2 — T5-style encoder-decoder with constrained decoder]

- **Architecture:** 8 transformer decoder layers, d=1024, H=16, FFN=4096, ~160M parameters
- **Cross-attention source (critical constraint):** Cross-attention keys and values are derived
  **exclusively** from the top-k=16 AFN-scored input tokens (Section 7), not from raw encoder
  hidden states. The decoder cannot attend to encoder intermediate activations or fusion layer
  states; it can only attend to the pre-scored, AFN-filtered evidence set.
- **Vocabulary constraint:** The output vocabulary is constrained at inference via hard logit
  masking (p = −∞ for blocked tokens) to:
  - **Allowed:** Domain-specific security terminology, ATT&CK technique IDs, Kairos ECL state
    tokens, structured causal connectives ("because", "triggered by", "correlated with",
    "absent", "indicates", "followed by"), anonymized entity references (e.g., HOST_A, USER_B)
  - **Blocked:** All tokens in the OffSec vocabulary block list (exploit tool names, shellcode
    keywords, payload scaffolding terms); real IP addresses outside RFC1918/documentation
    ranges; free-form natural language instructions; any token from the 24 sentinel token set
    except ECL output tokens
- **Max output length:** 256 tokens, enforced by truncating decoder positional embeddings at
  position 256
- **Invocation condition:** The decoder is called **only** when ECL state ∈ {ACT, ESCALATE}
  AND the requesting system (Argus/Themis) sets `request_decoder_trace = true`. It is not
  called on DEFER or FAIL_SAFE. This eliminates decoder latency for the majority of inference
  calls.
- **Freeze policy:** Decoder weights are frozen during domain-specific fine-tuning. Only updated
  during Stage 4 DPO alignment via a separate decoder DPO pass.
- **Safety note:** The vocabulary mask is applied at the logit computation step before sampling —
  it cannot be overridden by temperature, top-p, or repetition penalty hyperparameters.

---

## Section 7 — Interpretability Layer

### 7.1 Activation Flow Networks (AFN)

[Finding: Confirmed Borrowing: AFN; Surprise #1 — "attention is not explanation"]

Raw attention weights are rejected as an interpretability mechanism. Attention maps highlight
syntactically necessary but semantically weak tokens (delimiters, padding, stop words), not causal
elements. Distinct attention configurations can produce identical predictions (lack of uniqueness).
AFN provides mathematically provable input→output causal links via hidden-state L2-norms.

**Computation (≤15ms at batch-size=1 on A100 80GB):**

1. During the forward pass, cache all intermediate hidden states h_l(i) for each token position
   i at each encoder layer l
2. At Layer-8 analog for each domain encoder (Layer 8 of 12-layer encoders; Layer 6 of 8-layer
   encoders; Layer 4 of 6-layer encoders; Layer 3 of 4-layer encoders):
   ```
   importance(i) = ||h_8(i)||₂     (L2-norm of hidden state at position i)
   α(i)          = importance(i) / Σⱼ importance(j)
   ```
3. Normalize α(i) across the full input sequence per domain encoder
4. Select top-k=16 input tokens by α score per domain encoder → the AFN evidence set
5. These 16 tokens are the cross-attention source for the constrained decoder (Section 6.5)
6. AFN scores are persisted as structured metadata in the inference response:
   ```json
   {
     "afn_scores": [
       {"field": "EventID", "value": "4688", "domain": "detection", "score": 0.23},
       {"field": "technique", "value": "T1059.003", "domain": "cti", "score": 0.18},
       ...
     ]
   }
   ```

The L2-norm of the intermediate hidden state provides a causal, adversarially-robust measure of
input contribution. Unlike attention weights, it cannot be trivially manipulated by changing
downstream tokens, and it maintains uniqueness: different inputs produce distinguishably different
L2-norm profiles.

### 7.2 Surfacing AFN Output to Tier-3 Analysts

- Argus XDR renders AFN scores as a ranked field-importance table in the incident UI: field name,
  source domain, importance score, raw value
- The constrained decoder's justification trace is presented alongside, with each sentence
  annotated with which AFN-scored fields drove it (the cross-attention source mapping is logged)
- Both artifacts are appended to the incident audit trail in the Argus event log, timestamped and
  cryptographically signed for compliance chain-of-custody requirements
- The evidence-chain DAG (Section 6.1) is rendered as a provenance graph overlay in the Argus
  timeline view

### 7.3 Probing Classifier Regime

Probing classifiers validate that head groups do what they were designed to do. The regime is:

- Architecture: single Linear(d_model → n_classes) layer trained on **frozen** domain encoder
  representations (encoder weights not updated during probing)
- Training data: held-out validation split not used in any training stage
- Timing: probing suite is run at the end of Stage 2 (CoT distillation) and at the end of Stage 4
  (DPO alignment). Results are reported as part of the evaluation artifact.
- Per-group pass criteria: as specified in Section 4.2 head group table
- Failure action: if a head group fails its probing criterion, the group designation is dissolved
  and heads are reassigned to the best-performing group for the subsequent training stage. The
  specification is updated accordingly.

---

## Section 8 — Training Pipeline

### 8.1 Stage 1 — Pretraining

**Corpus and per-domain sampling weights:**

[Finding: Domain Data Inventory; Data Strategy Synthesis]

| Domain | Primary Sources | Est. Raw Volume | Usability | Sampling Weight | Notes |
|--------|----------------|-----------------|-----------|-----------------|-------|
| Detection/Alerting | CICIDS 2017/18, UNSW-NB15, Simargyl2022, Sigma HQ | Millions of flows; >3,000 Sigma rules | 3/5 | 20% | Penalize CICIDS synthetic homogeneity; upsample Sigma behavioral rules; supplement with Splunk Attack Range simulations |
| Forensics/Provenance | DARPA OpTC, DARPA Transparent Computing, LANL Unified Host | ~17.4B events (OpTC) | 4/5 | 25% | Subsample benign events to ≤10% of raw; inject MATRIX-generated attack overlays; required debiasing of scripted workloads |
| Threat Intelligence | MITRE ATT&CK, MISP galaxies, APTnotes, CTI report corpora | ~25,000 instruction-response pairs (SecureBERT 2.0 derived) + MITRE Enterprise matrix | 5/5 | 20% | Highest quality; full weight |
| Vulnerability/Risk | NVD, CVE datasets, LiveCVEBench, ExploitDB | ~200,000+ CVEs; 190 agentic tasks | 4/5 | 15% | Penalize outdated exploitability context; upweight LiveCVEBench executable tasks |
| Identity/Access | LANL Auth Dataset, CMU CERT Insider Threat | Hundreds of millions of auth events | 2/5 | 10% | Heavy synthetic augmentation via MATRIX; localized behavioral baselines only |
| Incident Response | CISA Advisories, FBI Flash Alerts, Open Source Playbooks | Thousands of reports; limited structured playbooks | 2/5 | 5% | Almost entirely synthetic; ground-truth ECL labels from DPO only |
| Compliance | CIS Benchmarks, NIST 800-53, ISO 27001 | Thousands of control definitions | 3/5 | 5% | Structural only; synthetic EDR→compliance gap mappings generated by MATRIX |

**Pretraining objectives:**
1. **Field-level masked language modeling (FMLM):** Mask entire JSON value fields (not sub-tokens)
   for sentinel-tokenized inputs; the model predicts masked value spans. Masking rate: 15% of
   non-sentinel value spans per example. Sentinel token positions are never masked.
2. **Kill-chain stage prediction:** Auxiliary next-stage cross-entropy over ATT&CK kill-chain
   transitions extracted from DARPA OpTC provenance graph walks (see Section 4.4 state machine)
3. **MITRE ontology embedding training:** TransE + RotatE objectives over the MITRE graph run
   concurrently with FMLM; embedding parameters join the main optimizer with a reduced learning
   rate (1e-5 vs. 1e-4 for encoder parameters)

**Provenance graph representation learning:**  
[Finding: Data Strategy Synthesis]

DARPA OpTC events are loaded as temporal provenance graphs. Host provenance (eCAR JSON) and
network flows (Zeek FLOW) are unified into a single temporal graph schema:
- Nodes: {process, file, socket, user, network_asset}
- Edges: {exec, read, write, connect, fork, spawn, delete} with UTC timestamps
- The ContiFormer continuous-time encoding (Section 3.3) encodes the temporal edge weights
- The Forensics/Provenance encoder receives these as serialized temporal graph sequences with
  Path C GCN pre-encoding for structural features

**EXPERIMENT-3 (curriculum learning trajectory):** Whether isolated node behavior pretraining
before full APT graph sequences produces faster convergence and lower overfitting is unresolved.
See Section 12. [Finding: Open Question #3]

### 8.2 Stage 2 — CoT Distillation from Frontier Teacher

**Teacher model:** GPT-4o or Claude Opus 4.8 prompted with security-specific chain-of-thought
instructions over synthetic scenarios.

**Dataset:** 50,000 synthetic security scenarios generated by MATRIX multi-agent simulator +
manually curated analyst-written examples. All IPs in RFC1918/documentation ranges; all entity
names anonymized.

**Distillation process:**
1. Teacher generates verbose NL reasoning traces over each scenario
2. An automated compression pipeline strips the NL deliberation and extracts:
   - `domain_triggered`: list of involved domains
   - `technique_identified`: list of ATT&CK technique IDs
   - `evidence_fields`: list of input field references
   - `ecl_state`: one of ACT/ESCALATE/DEFER/FAIL_SAFE
   - `confidence_rationale`: single terse sentence
3. Compressed output is re-encoded as a sentinel-only trace:
   ```
   <|S_DOMAIN:detection|><|S_DOMAIN:identity|><|S_START|>
   <|S_KEY|>technique<|S_VAL|>T1003<|S_KEY|>technique<|S_VAL|>T1021
   <|S_KEY|>evidence<|S_VAL|>EventID_4688<|S_KEY|>evidence<|S_VAL|>LogonType_3
   <|ECL_ESCALATE|><|S_END|>
   ```
4. Student (SecureFoundation) is trained via cross-entropy on these compressed traces

**Verbose NL CoT is explicitly excluded** from the training signal [Finding: Surprise #2 — small
models actively punish verbose reasoning; studies on 1.1B class models show sentinel-only markers
outperform natural language deliberation in stability, speed, and accuracy].

Head group probing classifiers (Section 7.3) are evaluated at the end of this stage.

### 8.3 Stage 3 — Absence and Negative Evidence Training (BaNEL)

[Finding: Confirmed Borrowing: BaNEL; Surprise #3 — standard RL collapses on null rewards]

**Training data:** Synthetic null-state events generated by Caldera and Atomic Red Team (ART)
attack simulations where the primary indicator is a missing expected log (e.g., stopped SIEM
agent, absent MFA prompt, dropped audit log).

**BaNEL-style contrastive loss:**

```
L_total = L_classification + λ · L_BaNEL

L_BaNEL = KL(P_baseline(A_i) || P_observed(A_i))
          for each attention position i where expected token is absent (<|S_NULL|>)

λ = 0.3  (tunable; anneal from 0.1 to 0.3 over first 5 epochs)
```

The KL divergence term provides gradient flow when the observed token is null, replacing the
zero-gradient produced by standard null-reward policy gradient. [Finding: Surprise #3]

Policy-gradient RL is explicitly not used for this training stage.

DCAT blocks (Section 4.3) are the primary gradient receivers for L_BaNEL; their baseline and
observed streams are directly trained by this loss.

### 8.4 Stage 4 — ECL Alignment via DPO

[Finding: Confirmed Borrowing: Argilla + evtx-sigma-checker]

**Platforms:** Argilla for preference annotation; evtx-sigma-checker for automated validation.

**DPO data pipeline:**
1. Security analysts annotate preferred vs. rejected ECL state assignments for real incident
   scenarios via Argilla
2. If an example references a Sigma rule, evtx-sigma-checker validates the rule fires on the
   provided Windows events before the example enters the DPO pool
3. PII scan (Section 9) gates all contributions before annotation
4. Preference pairs are assembled: (preferred_output, rejected_output) per scenario

**DPO objective:**
```
L_DPO = -E[ log σ( β · (log π(y_w|x) / π_ref(y_w|x)) − β · (log π(y_l|x) / π_ref(y_l|x)) ) ]
β = 0.1
```

**Decoder DPO pass:** A separate DPO pass is run on the constrained decoder using analyst
preference labels on justification trace quality (clarity, completeness, absence of hallucinated
field references).

**Synthetic data flywheel:**  
[Finding: Data Strategy Synthesis]

MATRIX multi-agent simulator operates in tandem with Caldera and Atomic Red Team throughout
Stages 1–3 to:
- Generate realistic multi-domain attack scenarios where Detection + Identity + Vulnerability
  signals co-occur (cross-domain cross-pollination absent from single-domain public datasets)
- Inject absence signals (simulated logging failures, agent process termination)
- All synthetic IPs: RFC1918 ranges only
- All synthetic usernames, hostnames: anonymized with consistent within-scenario mapping
- Synthetic DPO pairs are generated automatically before human analysts refine them in Argilla

---

## Section 9 — Data Schema and Crowdsource Infrastructure

### 9.1 Training Example JSON Schema

```json
{
  "$schema": "https://securefoundation.io/schema/training-example/v1.0.json",
  "schema_version": "1.0",
  "domain": "<detection|forensics|cti|vulnerability|identity|ir|compliance>",
  "scenario_id": "<uuid4>",
  "timestamp_utc": "<ISO8601>",
  "input_events": [
    {
      "event_type": "<evtx|netflow|stix|cvss|auth|cloudtrail|sigma|cve|ldap|playbook|cis_control>",
      "event_timestamp_utc": "<ISO8601 or null>",
      "raw_content": "<sentinel-tokenized string OR structured JSON object>",
      "source_host": "<anonymized_host_id or null>",
      "domain_tag": "<detection|forensics|cti|vulnerability|identity|ir|compliance>"
    }
  ],
  "expected_techniques": ["<ATT&CK technique ID e.g. T1003>"],
  "expected_ecl_state": "<ACT|ESCALATE|DEFER|FAIL_SAFE>",
  "confidence_label": "<float 0.0–1.0>",
  "reasoning_trace": "<sentinel-only trace string per Section 8.2 compression format>",
  "null_signals": [
    {
      "expected_field": "<field name>",
      "reason_absent": "<brief string>"
    }
  ],
  "provenance_graph": {
    "nodes": [
      {"id": "<string>", "type": "<process|file|socket|user|network_asset>", "label": "<string>"}
    ],
    "edges": [
      {
        "from": "<node id>",
        "to": "<node id>",
        "relation": "<exec|read|write|connect|fork|spawn|delete>",
        "timestamp_utc": "<ISO8601>"
      }
    ]
  },
  "metadata": {
    "contributor_id": "<SHA-256 hash of anonymized contributor identifier>",
    "synthetic": "<bool>",
    "validation_status": "<pending|sigma_validated|dpo_approved|rejected>",
    "license": "Apache-2.0"
  }
}
```

**Required fields** (minimum viable contribution):  
`domain`, `input_events` (≥1 entry), `expected_ecl_state`

**Optional fields** (routing rules):  
- `null_signals` + `provenance_graph` present → eligible for Stage 3 absence training  
- `reasoning_trace` present → eligible for Stage 2 CoT distillation  
- Neither present → eligible for Stage 1 pretraining and Stage 4 DPO only  

### 9.2 Maximum Schema Complexity

This schema is bounded at the complexity level above. [Finding: Domain Data Inventory — Thread 12
on contributor participation] The `provenance_graph` and `null_signals` objects are the most
complex elements; both are optional. Adding further nested objects (e.g., inline binary payloads,
recursive sub-graphs) would degrade contributor participation. Binary data (PCAP) is never
accepted raw; contributors submit pre-extracted flow features only.

### 9.3 Validation Pipeline

1. **JSON Schema linting:** Automated; schema must validate against `v1.0.json` spec
2. **Sigma rule validation:** If `event_type == evtx` AND `reasoning_trace` references a Sigma
   rule ID → `evtx-sigma-checker` validates the rule fires on the provided event; contribution
   rejected if check fails [Finding: Confirmed Borrowing: Argilla + evtx-sigma-checker]
3. **PII scan:** Regex + ML classifier checks for:
   - Real public IP addresses (outside RFC1918: 10.x.x.x, 172.16.x.x, 192.168.x.x and
     RFC5737: 192.0.2.x, 198.51.100.x, 203.0.113.x)
   - Real domain names containing organizational identifiers
   - Person names in free-text fields
   - Contributions failing PII scan are quarantined, not published
4. **Argilla peer annotation:** Validated contributions enter the Argilla queue for ECL state
   preference annotation by ≥2 security analysts; disagreements surfaced for adjudication
5. **DPO approval:** Examples achieving annotation consensus → `validation_status = dpo_approved`
   and enter the Stage 4 DPO pool

### 9.4 HuggingFace Organization and Repository Structure

| Repository | Contents |
|------------|----------|
| `secureaifoundation/sf-pretrain-detection` | Detection/Alerting corpus: CICIDS flows, UNSW-NB15, Sigma rules, Attack Range logs |
| `secureaifoundation/sf-pretrain-forensics` | Forensics corpus: DARPA OpTC (subsampled + debiased), LANL Unified Host |
| `secureaifoundation/sf-pretrain-cti` | CTI corpus: MITRE ATT&CK STIX, MISP galaxies, APTnotes extracted structures |
| `secureaifoundation/sf-pretrain-vuln` | Vulnerability corpus: NVD, CVE JSON, CVSS vectors, LiveCVEBench tasks |
| `secureaifoundation/sf-pretrain-identity` | Identity corpus: LANL Auth, CMU CERT, synthetic UEBA |
| `secureaifoundation/sf-pretrain-ir` | IR corpus: CISA advisories (structured), synthetic playbook SOAR data |
| `secureaifoundation/sf-pretrain-compliance` | Compliance corpus: CIS OVAL, NIST JSON, ISO 27001 controls |
| `secureaifoundation/sf-dpo-pairs` | DPO preference pairs: post-Argilla annotation, ECL state preferences |
| `secureaifoundation/sf-model` | Model weights (PyTorch safetensors), ONNX export, tokenizer, config |

Each dataset repo requires:
- Dataset card: license (Apache 2.0), intended use, known limitations (e.g., DARPA OpTC
  scripted-benign bias), citation, PII policy
- `CONTRIBUTING.md` at repo root: schema download link, validation toolchain setup instructions,
  Argilla annotation instance link, example contributions for each domain

---

## Section 10 — Integration Specification

### 10.1 Argus XDR Integration (Go backend)

**Input contract (Argus XDR → SecureFoundation):**

```protobuf
syntax = "proto3";

message SecurityEvent {
  string event_id = 1;
  string event_type = 2;       // evtx|netflow|stix|cvss|auth|cloudtrail|sigma|cve|ldap|...
  string domain_tag = 3;       // detection|forensics|cti|vulnerability|identity|ir|compliance
  int64 timestamp_unix_ms = 4; // -1 if unknown
  string raw_content = 5;      // sentinel-tokenized string or JSON string
}

message GraphNode {
  string node_id = 1;
  string node_type = 2;        // process|file|socket|user|network_asset
  string label = 3;
  repeated float features = 4; // pre-computed node feature vector (optional)
}

message GraphEdge {
  string from_id = 1;
  string to_id = 2;
  string relation = 3;         // exec|read|write|connect|fork|spawn|delete
  int64 timestamp_unix_ms = 4;
}

message SFInferenceRequest {
  string request_id = 1;
  repeated SecurityEvent events = 2;    // max 512 events per call
  repeated GraphNode rag_nodes = 3;     // Argus-provided subgraph nodes (max 1024, sampled to 256 if exceeded)
  repeated GraphEdge rag_edges = 4;     // Argus-provided subgraph edges
  string signal_taxonomy_level = 5;     // Argus L1-L10 signal taxonomy level
  bool request_decoder_trace = 6;       // true only for ACT/ESCALATE states requiring analyst trace
}
```

**Output contract (SecureFoundation → Argus XDR):**

```protobuf
enum ECLState {
  ACT = 0;
  ESCALATE = 1;
  DEFER = 2;
  FAIL_SAFE = 3;
}

message DAGNode {
  string node_id = 1;
  string domain = 2;
  string technique_id = 3;     // ATT&CK technique ID or empty
  string evidence_field = 4;
}

message DAGEdge {
  string from_id = 1;
  string to_id = 2;
}

message EvidenceDAG {
  repeated DAGNode nodes = 1;
  repeated DAGEdge edges = 2;
}

message AFNScore {
  string field_name = 1;
  string field_value = 2;
  string domain = 3;
  float score = 4;
}

message SFInferenceResponse {
  string request_id = 1;
  ECLState ecl_state = 2;
  float confidence = 3;              // calibrated scalar [0,1]
  repeated string technique_ids = 4; // ATT&CK technique IDs (multi-label)
  EvidenceDAG evidence_chain = 5;
  repeated AFNScore afn_scores = 6;  // top-16 field importances
  string decoder_trace = 7;          // present only if request_decoder_trace=true
  float inference_latency_ms = 8;
  bool early_exit = 9;               // true if F3 fast-tap was used
}
```

**ONNX export:**
- Export via `torch.onnx.export` at opset=17
- Two separate ONNX graphs:
  1. `sf_encoder_fusion.onnx`: domain encoders + DCAT + epistemic fusion + classification heads
  2. `sf_decoder.onnx`: constrained justification decoder (invoked separately on ACT/ESCALATE)
- Dynamic axes on both graphs: `batch_size`, `sequence_length`
- Quantization: INT8 post-training quantization applied to domain encoder weights via ONNX Runtime
  quantization tooling. Decoder weights are left at FP16 to preserve justification trace quality.
- Go runtime: `onnxruntime-go` v1.18+ binding

**RAG boundary — what Argus provides at inference time:**

| Data Type | Update Frequency | Rationale |
|-----------|-----------------|-----------|
| Zero-day IOC lists | Every 15 minutes | Volatile; cannot be baked in |
| MISP live threat actor IP feeds | Every 15 minutes | Volatile |
| CVE records post-training cutoff | Daily | New CVEs published continuously |
| D3FEND mitigation mappings | Weekly | Relatively stable but not baked per operator decision |
| Active Directory subgraphs | On-demand per entity | Enterprise-specific; changes continuously |
| Sigma rule updates (SigmaHQ) | Daily | Community-maintained; evtx-sigma-checker validates |
| ATT&CK sub-technique updates post-v16 | At model update cycle | Only the base technique embeddings are baked |

[Finding: Architecture Decision Input #4; operator knowledge boundary: minimal baked-in
(MITRE tactics/techniques only)]

**Latency budget (single A100 80GB, 4096-token input, batch-size=1):**

| Component | Budget |
|-----------|--------|
| Input tokenization (3 paths) | ≤5 ms |
| 7 domain encoders (parallel execution) | ≤60 ms |
| DCAT blocks (within encoders) | included in above |
| Epistemic fusion F1–F6 | ≤15 ms |
| AFN computation | ≤15 ms |
| Classification heads + calibration | ≤5 ms |
| **Total (no decoder)** | **≤100 ms** |
| Constrained decoder (ACT/ESCALATE only) | ≤50 ms additional |
| **Total (with decoder)** | **≤150 ms** |

Target ≤120 ms for non-decoder path leaves 20 ms margin for inference runtime overhead.

### 10.2 Kairos ECL Confidence-to-State Mapping

| Model Raw Prediction | Confidence Range | Kairos ECL Output | Notes |
|---------------------|-----------------|-------------------|-------|
| ACT | ≥0.85 | ACT | Direct pass-through |
| ESCALATE | ≥0.85 | ESCALATE | Direct pass-through |
| ACT | 0.65–0.84 | ESCALATE | Downgrade; human review required before automated action |
| ESCALATE | 0.65–0.84 | ESCALATE with uncertainty flag | Pass-through with flag |
| Any | 0.40–0.64 | DEFER | Insufficient confidence for autonomous action |
| Any | <0.40 | FAIL_SAFE | Model requests human override |
| Any (DCAT override) | Any, D > θ | ESCALATE | DCAT divergence overrides low-confidence DEFER |
| Any (RAG unavailable) | — | FAIL_SAFE | Hard fallback; do not infer without required RAG context |

### 10.3 Themis Integration

SecureFoundation operates as a **synchronous analytical oracle** within Themis dispatch:

- Themis passes batched event contexts to SecureFoundation via the Argus XDR `SFInferenceRequest`
  protobuf. Themis does not call SecureFoundation directly; all calls flow through Argus XDR's
  inference proxy.
- SecureFoundation returns `SFInferenceResponse`. Themis uses `ecl_state` to dispatch downstream
  agentic actions; it does not inspect intermediate model state.
- SecureFoundation does NOT: call external tools, spawn sub-agents, maintain session state across
  calls, or store any entity state outside the DCAT baseline cache (which is an inference-layer
  concern, not a Themis concern).
- The constrained decoder's `decoder_trace` field is the only text SecureFoundation ever emits
  into Themis's reasoning context. Themis must not inject decoder trace content back into
  subsequent inference calls to SecureFoundation (prevents trace self-amplification).
- The rigid domain-encoder parameter isolation (Section 1.4) provides functional equivalence to
  Themis-style agent isolation without requiring Themis to manage the domain boundaries.
  [Finding: Surprise #4]

---

## Section 11 — Evaluation Framework

### 11.1 Per-Head Metrics

| Head | Primary Metric | Secondary Metric | Target | Baseline |
|------|---------------|-----------------|--------|---------|
| ECL state classifier | Macro F1 (4-class) | Per-class recall matrix | F1 ≥ 0.91 | No published baseline |
| ATT&CK technique (multilabel) | Micro F1 | Top-5 accuracy | Micro F1 ≥ 0.78; Top-5 ≥ 0.90 | Foundation-Sec-8B |
| Calibrated confidence | Expected Calibration Error (ECE) | Reliability diagram flatness | ECE ≤ 0.04 | General LLM zero-shot |
| Evidence chain DAG | Edge F1 vs. ground-truth provenance | Node coverage | Edge F1 ≥ 0.72 | No published baseline |
| DCAT absence detection | False Negative Rate (FNR) on null-signal scenarios | Precision on null signals | FNR ≤ 0.08 | Rule-based SIEM correlation |
| AFN fidelity | Human expert agreement on top-5 ranked fields | Adversarial flip rate | Agreement ≥ 0.80; flip rate < raw-attention flip rate | Raw attention heatmap |
| Decoder justification | Analyst rating (1–5 scale) on trace accuracy and absence of hallucinated fields | Vocabulary escape rate | Mean rating ≥ 4.0; vocabulary escape rate = 0 | N/A |

### 11.2 Calibration Evaluation

- Temperature scaling evaluated on held-out OOD set (Splunk Attack Range synthetic events, not
  present in any training corpus)
- Conformal prediction coverage verified at p=0.90 and p=0.95: the conformal set must contain
  the true ECL label in ≥90% and ≥95% of OOD test examples respectively
- Reliability diagrams plotted for each domain encoder output

### 11.3 Baselines

| Baseline | Type | Parameter Count | Benchmark |
|----------|------|----------------|-----------|
| SecureBERT 2.0 (Cisco, ModernBERT-based) | Domain encoder, encoder-only | ~150M | ATT&CK NER, CVSS classification, TTP extraction |
| Foundation-Sec-8B (Cisco) | Domain LLM, decoder | 8B | TTP extraction, zero-shot ECL approximation, CTI QA |
| GPT-4o zero-shot | General LLM | ~unknown | Cross-domain correlation, evidence DAG construction, ECL assignment |
| Splunk SIEM rule correlation | Rule-based | N/A | Alert triage F1, false positive rate, null-state detection |
| 3-agent orchestrated swarm (GPT-4o-mini, Themis orchestrated) | Multi-agent | 3× ~8B | ECL state exact-match on DARPA OpTC scenarios (swarm-safety parity bar) |

The swarm-safety parity bar requires SecureFoundation to achieve exact ECL state match on ≥90% of
200 held-out DARPA OpTC APT scenarios compared to the 3-agent baseline. "Exact match" means
identical state (ACT/ESCALATE/DEFER/FAIL_SAFE), not just same ordinal direction.

Additionally: the decoder trace may not reference domain information unavailable in the input to
the relevant domain encoder. This constraint is verified by automated cross-referencing of AFN
source fields against decoder trace entity mentions.

### 11.4 Benchmark Datasets

| Dataset | Domains | Task |
|---------|---------|------|
| DARPA OpTC (6-day, 1000 hosts) | Forensics, Detection | ATT&CK stage attribution, provenance graph reconstruction, kill-chain F1 |
| DARPA Transparent Computing | Forensics | Multi-stage lateral movement reconstruction |
| LiveCVEBench (190 agentic tasks) | Vulnerability | CVE contextual reasoning, CVSS scoring accuracy |
| LANL Auth Dataset | Identity | Anomalous authentication session detection |
| UNSW-NB15 | Detection | Network intrusion classification, null-state detection (dropped flows) |
| MITRE ATT&CK Evaluations Round 5 (Carbanak + FIN7) | Multi-domain | Cross-domain kill-chain reconstruction, ECL state assignment |
| CMU CERT Insider Threat v6.2 | Identity, IR | Behavioral anomaly detection, ECL state assignment |
| Sigma HQ detection rules (held-out 20%) | Detection | Sigma rule → ECL mapping accuracy |

---

## Section 12 — Experiment Register

These three experiments must be completed empirically before the corresponding spec sections are
locked for implementation. The spec defines them; it does not resolve them.

| ID | Hypothesis | Method | Success Criterion | Spec Sections Unblocked |
|----|-----------|--------|-------------------|------------------------|
| **EXPERIMENT-1** | There exists an optimal temporal window N (number of prior events retained per entity in the DCAT baseline cache) that maximizes null-signal detection without exhausting single-GPU VRAM | Sweep N ∈ {32, 64, 128, 256, 512} on the Detection/Network encoder running DARPA OpTC data. For each N: measure (a) FNR on null-signal validation set, (b) peak VRAM consumption on A100 80GB, (c) inference latency overhead. Run each N over 500 inference calls with 50 entities tracked simultaneously | FNR ≤ 0.08 AND peak VRAM < 40GB AND latency overhead < 15 ms. If no N satisfies all three: adopt the Pareto-optimal N and accept the trade-off; document in Risk Register | §4.3 DCAT baseline mechanism; §10.1 latency budget; determines whether DCAT is viable on commodity inference hardware |
| **EXPERIMENT-2** | Jointly optimizing (a) continuous cross-domain fusion attention, (b) discrete kill-chain state tracking, and (c) discrete ECL state classification converges stably and without single-objective collapse | Train the full Stage 1–Stage 3 pipeline. At epochs 5, 10, and 15: measure ATT&CK Micro F1, ECL state Macro F1, kill-chain stage prediction accuracy, and loss curve variance. Record whether any objective degrades while others improve (objective conflict signature). Also run head-group probing at each checkpoint to verify head specialization emerges alongside multi-task stability | All three objectives improve together through epoch 15 (no more than 0.03 F1 drop in any objective after epoch 5). Head-group probing pass rates reach their Section 4.2 targets by epoch 10. If multi-task instability is detected: evaluate (a) loss weighting schedules (GradNorm), (b) sequential objective introduction (kill-chain first, then ECL), (c) separate optimization of the ECL head with frozen fusion encoder | §4.4 kill-chain state tracking; §5 reasoning representation; §6.3 ECL classifier; entire training pipeline schedule |
| **EXPERIMENT-3** | Isolated node behavior pretraining before full-graph APT sequences produces faster convergence and lower local-minima overfitting on the DARPA OpTC forensics task | Two-arm experiment on Forensics/Provenance encoder only. Arm A: Stage 1 pretraining begins with single-node behaviors (individual malicious process executions, isolated file write events) for the first 50% of Stage 1 compute, then introduces full multi-stage APT provenance graphs. Arm B: full provenance graphs from epoch 1. Evaluate ATT&CK stage attribution F1 on DARPA OpTC test set at epoch 5, 10, and 15. Compute budget is identical across arms | Arm A achieves ≥0.05 higher ATT&CK stage F1 at epoch 10 with identical total compute; OR both arms reach equivalent final F1 within 0.02 (adopt Arm B for simplicity). If Arm B is strictly better at all checkpoints: adopt Arm B and note isolated-node pretraining as ineffective for provenance data | §8.1 Stage 1 pretraining curriculum; DARPA OpTC subsampling strategy; determines whether a two-phase pretraining schedule is required |

---

## Section 13 — Open Risks

### Risk 1: MoE Side-Channel Rejection

Mixture-of-Experts was explicitly rejected because sparse routing produces measurable
micro-architectural footprints in GPU execution telemetry (active thread counts during the
prefilling phase). These footprints can be observed non-intrusively via hardware performance
counters, enabling side-channel attacks that deduce model routing behavior and, by extension,
infer which security domain was activated for a given input. [Finding: Architecture Decision
Input #1]

The parallel domain-encoder architecture eliminates sparse routing: all seven encoders execute
unconditionally on every inference call, producing constant, predictable GPU execution patterns
regardless of input content. This has a throughput cost (FLOPs proportional to all 7 encoders,
not the 1–2 active experts a MoE would use). Mitigation: INT8 quantization of encoder weights
reduces the throughput penalty.

### Risk 2: Graph Transformer VRAM Ceiling on Single GPU for Enterprise AD Topologies

Integrating Graph Attention Networks or GPS mechanisms for enterprise Active Directory topologies
is computationally heavy. An enterprise AD graph may contain 100,000+ node entities. At inference
time, Argus XDR provides pre-extracted 2-hop neighborhood subgraphs per queried entity; the GCN
encodes a maximum of 256 nodes per call (top-k degree-centrality sampling applied if subgraph
exceeds 1,024 nodes before sampling). [Finding: Open Question #4]

Risk: low-degree nodes that are critical lateral movement pivot points may be dropped by
degree-centrality sampling. Mitigation: Argus XDR can apply domain-aware sampling (e.g., prioritize
nodes with Service Principal Names or AdminCount=1 in AD, which are high-value regardless of degree)
before passing the subgraph to SecureFoundation. This mitigation is implemented in Argus, not in
SecureFoundation.

### Risk 3: DARPA OpTC Scripted-Benign Workload Bias

The Forensics/Provenance encoder's primary pretraining corpus (17.4B DARPA OpTC events) relies on
heavily scripted, repetitive benign workloads. Without debiasing, the model will overfit to scripted
artifact patterns and fail on organic enterprise telemetry. [Finding: Domain Data Inventory —
Forensics assessment note]

Mitigation (applied in Stage 1 pretraining): subsample benign events to ≤10% of raw DARPA volume,
stratified by host; inject MATRIX-generated attack overlays into benign sequences; maintain a
held-out OOD evaluation set from Splunk Attack Range (uses different workload profiles from DARPA).
If the OOD evaluation set shows >0.10 F1 degradation vs. DARPA test set, additional debiasing
(domain-adversarial training) is triggered before Stage 2.

### Risk 4: Minimal RAG Knowledge Boundary Increases Inference Dependency

The operator's decision to bake only MITRE tactic/technique embeddings means that CVE severity,
D3FEND mitigations, identity graph topology, and all threat actor profiles must be retrieved from
Argus XDR on every inference call. This couples model availability to Argus XDR availability.

Mitigation:
- Argus XDR maintains a local replica cache of the most-accessed CVE and D3FEND records
  (target: top 10,000 CVEs by EPSS score, updated daily)
- FAIL_SAFE ECL state is the hard fallback when RAG is unavailable — inference does not proceed
  on vulnerability or identity domain inputs without required graph context
- For air-gapped SOC deployments: a quarterly "snapshot" ONNX variant includes frozen CVE/D3FEND
  embeddings current as of the snapshot date; knowledge staleness is documented in the model card

### Risk 5: Constrained Decoder Vocabulary Escape via Adversarial Input

A maliciously crafted EVTX field or STIX object containing decoder-influencing text could attempt
to steer the constrained decoder's output through prompt injection.

Mitigation layers (all structural, not policy-dependent):
1. Sentinel tokenization strips semantic intent from structural delimiters — the decoder receives
   only AFN-scored evidence tokens, not raw input text
2. The decoder's cross-attention source is restricted to the AFN top-16 evidence set; it cannot
   attend to arbitrary input positions
3. Vocabulary mask is applied at logit computation (p = −∞) before sampling — temperature, top-p,
   and repetition penalties cannot unlock blocked tokens
4. Decoder output is passed through a static blocklist filter before reaching Argus/Themis as a
   final defense-in-depth layer
5. The constrained decoder is only invoked on ACT/ESCALATE states — reducing its attack surface
   to high-priority events where analyst scrutiny is already elevated

---

## Appendix A — Total Parameter Budget Summary

| Component | Estimated Parameters |
|-----------|---------------------|
| Shared embedding stack (token + source-type + temporal + ontology) | ~100M |
| Detection/Network encoder (12L, d=1024) | ~200M |
| Forensics/Provenance encoder (12L, d=1024) | ~200M |
| CTI/STIX encoder (8L, d=768) | ~85M |
| Vulnerability/Risk encoder (8L, d=768) | ~85M |
| Identity/Access encoder (6L, d=768) | ~65M |
| Incident Response encoder (6L, d=768) | ~65M |
| Compliance encoder (4L, d=512) | ~28M |
| DCAT augmentation blocks (2 encoders, 2 layers each) | ~40M |
| Epistemic fusion layer (6L cross-attention, d=1024) | ~100M |
| Constrained justification decoder (8L, d=1024) | ~160M |
| Output heads (DAG + ATT&CK + ECL + calibration) | ~35M |
| **Grand Total** | **~1.16B** |

Target ceiling: 4.0B. Current estimate: ~1.16B. Headroom: ~2.84B.  
The headroom provides margin for architecture ablations identified by EXPERIMENT-2, potential
decoder depth increases for justification quality, and future domain encoder scaling if data
quality allows.

---

## Appendix B — Security Constraints Compliance Checklist

| Constraint | Implementation |
|------------|---------------|
| Sub-4B parameters, single GPU | ~1.16B total; fits on A100 40GB with INT8 quantization |
| Real-time SOC latency | ≤120ms target; ONNX INT8 on A100; early-exit F3 fast path |
| No exploit synthesis in weights | No decoder head produces exploit code; vocabulary mask is structural (logit-level), not policy |
| No generative free-text attack tooling | Constrained decoder: hard vocabulary mask; max 256 tokens; AFN-only cross-attention source |
| No PII / no real IPs | RFC1918/RFC5737 only in training data; PII scan in validation pipeline; contributor guide enforces |
| Apache 2.0 license | All weights, code, and training data published under Apache 2.0 |
| Discrete human-auditable ECL markers | Emitted at F3 (fast) and F6 (full) tap points; persisted in Argus audit trail |
| AFN interpretability (not raw attention) | AFN L2-norm at Layer-8 analog; attention heatmaps not exposed to consumers |
