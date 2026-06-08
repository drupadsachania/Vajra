# Vajra — Architecture Specification
*A security-native foundation model. Built in India.*

**Version:** 1.0  
**License:** Apache 2.0  
**Status:** Implementation-ready — pending Experiment Register resolution (§12)
**Architecture locked:** 2026-06-07
**Repository:** github.com/vajra-foundation/vajra

---

## Section 0 — Decision Log

Every major architecture decision, the chosen option, the finding that drove it, and the alternatives
rejected. Read this first.

| # | Decision | Chosen | Finding | Alternatives Rejected |
|---|----------|--------|---------|----------------------|
| 0-1 | Fusion architecture | **Parallel domain-specific encoders → epistemic fusion at final layer** | Architecture Decision Input #1: MoE sparse routing produces measurable GPU side-channel footprints and obfuscates which sub-network drove the epistemic conclusion | Dense early-fusion cross-attention; Mixture-of-Experts (MoE) |
| 0-2 | Output architecture | **T5-style encoder-decoder with constrained justification decoder** | Architecture Decision Input #2: encoder-only lacks autoregressive capability for analyst-readable kill-chain traces; latent CoT yields probing inconsistency incompatible with depth-tapped decision state requirement | Encoder-only; latent CoT (CoLaR/Coconut) |
| 0-3 | Tokenization | **Modular multi-tokenizer: numerical sub-tokenizer + graph-embedding pipeline + S-TOON sentinel-bounded text tokenizer → shared d=1024** | Architecture Decision Input #3; Gap: Unified Multimodal Tokenization Paradox; Confirmed Borrowing: S-TOON | Universal BPE; single-modality text tokenizer |
| 0-4 | Knowledge boundary | **MITRE tactic/technique embeddings baked in (14 tactics, ~700 techniques); all CVE data, D3FEND, identity graphs, live threat intel via consuming system's RAG** | Architecture Decision Input #4 (operator chose minimal baked-in boundary); Confirmed Borrowing: Ontological KG Embeddings | Larger static KG including CVE/D3FEND baked in; fully RAG-dependent at inference |
| 0-5 | Interpretability method | **Activation Flow Networks (AFN): L2-norm hidden-state importance at intermediate layers** | Surprise #1: "attention is not explanation" — raw attention maps highlight syntactic delimiters, not causal tokens; AFN provides mathematically provable input→output links | Raw attention heatmaps |
| 0-6 | Absence/null-state reasoning | **Dual-Contrastive Attention (DCAT) + BaNEL contrastive loss** | Confirmed Borrowing: DCAT; Surprise #3: standard RL on null rewards produces zero gradients and collapses to random search | Standard self-attention; policy-gradient RL |
| 0-7 | Temporal encoding | **Continuous-time encoding (ContiFormer/TFT): time as continuous variable, learnable frequencies** | Confirmed Borrowing: Continuous-Time / TFT — discrete sinusoidal indices treat ms-apart and month-apart events as equally adjacent | Discrete sinusoidal positional embedding; learned relative position |
| 0-8 | Ontological knowledge representation | **TransE + RotatE + GCN on MITRE ATT&CK graph; embeddings frozen after Stage 1** | Confirmed Borrowing: Ontological KG Embeddings — TransE/RotatE capture relational translation semantics; GCN captures local structural dependencies | Inference-time RAG for static MITRE facts; hand-coded rule embeddings |
| 0-9 | Reasoning output style | **Discrete decision state markers at depth-tapped fusion layers; no latent CoT** | Gap: Interpretable Latent CoT (probing inconsistency); Surprise #2: small models punish verbose CoT — sentinel-only markers outperform NL deliberation | Verbose NL chain-of-thought; depth-recurrent latent states |
| 0-10 | MoE routing | **Rejected** | Architecture Decision Input #1: sparse routing produces GPU execution-telemetry side-channel observable non-intrusively; routing obfuscates epistemic attribution | N/A |

---

## Section 1 — System Overview

### 1.1 Purpose

Vajra is a security-native encoder-decoder transformer that accepts heterogeneous
cybersecurity telemetry from seven security domains in a single inference call, produces discrete
decision state classifications, calibrated confidence scores, ATT&CK technique classifications,
evidence-chain DAGs, and human-auditable activation-flow justification traces.

Vajra is designed to be deployed by any security operations platform that can
provide structured telemetry input and consume structured JSON output. The model
does not depend on any specific XDR, SIEM, or orchestration platform. RAG context
(CVE records, threat intel, asset graphs) is provided at inference time by the
consuming system via the graph input fields. Agentic orchestration layers use
Vajra's `decision_class` output and `justification_trace` as signals for downstream
actions.

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
- Not a replacement for a Tier-3 analyst. Vajra produces discrete state and evidence; the
  analyst acts.
- Not a general-purpose LLM. Domain encoders are trained exclusively on security telemetry.
- Not an orchestration engine. Agentic orchestration layers dispatch actions; Vajra infers.
- Not a replacement for a rule engine. Sigma rule matching and alert triage remain in the
  consuming platform; Vajra receives matched-event output as input.

### 1.4 Core Thesis: Monolith Rivaling Multi-Agent Swarm Safety

The most advanced deployed security AI systems (CrowdStrike Charlotte AI, Palo Alto XSIAM,
Microsoft Security Copilot) deliberately fracture reasoning into siloed agents to isolate logic,
prevent cross-domain hallucination, and enforce authorization controls [Finding: Surprise #4].

Vajra achieves equivalent safety guarantees within a monolithic architecture via
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
| Decision state macro F1 (4-class) | ≥ 0.91 | No published baseline |
| CVSS per-metric classification accuracy | ≥ 0.95 | CVSS-BERT: ~0.89 |
| Mean inference latency (4096-token input, single A100 80GB) | ≤ 120 ms | — |
| Expected Calibration Error (ECE) on OOD distribution | ≤ 0.04 | — |
| AFN trace generation overhead | ≤ 15 ms | — |
| Swarm-safety parity: decision state exact-match vs. 3-agent baseline | ≥ 90% on 200 DARPA OpTC scenarios | — |
| Decoder trace generation latency (classes 0–1 only) | ≤ 50 ms additional | — |

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

The four `<|ECL_*|>` tokens have been removed from the vocabulary. The decision state
classification head (Section 6.3) outputs a 4-class logit vector over integer class indices
{0, 1, 2, 3}. The semantic mapping of these indices to named decision states is the
responsibility of the consuming system. Vajra has no knowledge of how the consuming system
names or acts on these states.

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
inference-time subgraphs provided by the consuming system (AD topology fragments, asset dependency
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

**RAG-provided (inference-time):**  
The consuming system provides `graph_nodes` and `graph_edges` per request (see §10.1 input contract).  
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
src_user U10@DOM1 → BPE within <|S_VAL|>; if identity graph provided by consuming system,
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

- Vocabulary size: 50,428  
  = 50,000 BPE tokens (trained on security corpora, not internet text)  
  + 20 sentinel tokens (Section 2.1)  
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
D3FEND mitigations, identity graph topology, and threat actor profiles are retrieved from the
consuming system's RAG pipeline at inference time via Path C's inference-time GCN. This maximizes
knowledge freshness at the cost of RAG latency dependency (see Risk 4 in Section 13).

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
| Incident Response | 6 | 768 | 12 | 3072 | ~65M | Data-sparse; ground-truth decision state labels almost entirely crowdsourced or synthetic [Finding: Domain Data Inventory — IR usability 2/5] |
| Compliance | 4 | 512 | 8 | 2048 | ~28M | Structural control matching; no deep sequential reasoning required [Finding: Domain Data Inventory — Compliance usability 3/5, disconnected from live telemetry] |

**Context window and attention pattern:**

- **Detection/Network:** Context window: 32K tokens default, expandable to 256K via interleaved
  RoPE/NoPE attention pattern. Local layers use 4096-token sliding window with rotary positional
  embeddings. Global layers (every 4th layer) use full attention with no positional embeddings
  (NoPE). This pattern is borrowed from Gemma 4's interleaved attention design and handles
  continuous NetFlow and PCAP metadata streams that exceed standard context budgets. The
  4096-token input referenced in Section 10.3 latency benchmarks reflects a single-event-batch;
  sustained telemetry ingestion operates at the full context window.
- **Forensics/Provenance:** Context window: 32K tokens default, expandable to 256K via the same
  interleaved RoPE/NoPE pattern as the Detection encoder. Required for multi-day DARPA OpTC-style
  provenance graphs where a complete APT kill chain across 1000 hosts cannot be represented in
  4096 tokens without destructive truncation.
- **CTI/STIX:** Context window: 4096 tokens. Standard context is sufficient for this domain's
  input types.
- **Vulnerability/Risk:** Context window: 4096 tokens. Standard context is sufficient for this
  domain's input types.
- **Identity/Access:** Context window: 4096 tokens. Standard context is sufficient for this
  domain's input types.
- **Incident Response:** Context window: 4096 tokens. Standard context is sufficient for this
  domain's input types.
- **Compliance:** Context window: 4096 tokens. Standard context is sufficient for this domain's
  input types.

**All domain encoders share:**
- Pre-LayerNorm (more stable for fine-tuning than post-LN)
- Rotary Position Embeddings (RoPE) for within-encoder relative position; Detection and Forensics
  encoders use interleaved RoPE/NoPE for long-context support (see context window specifications above)
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
- `B` is updated with EMA coefficient β=0.9 after each inference call where `decision_class` ∈
  {2, 0} (non-anomalous: DEFER or HIGH_CONFIDENCE_ACTION)
- `B` is frozen (not updated) when `decision_class` ∈ {1, 3} (ESCALATE_FOR_REVIEW or
  INSUFFICIENT_CONTEXT), preserving the pre-anomaly baseline  
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

**Decision state tap points:**
- **F3 fast tap:** After fusion block F3, a linear head Linear(1024 → 4) produces a preliminary
  decision state. If confidence(class=2) > 0.92 OR confidence(class=3) > 0.85, inference
  short-circuits here (early exit). This eliminates fusion F4–F6 FLOPs for clear low-confidence
  inputs.
- **F6 full tap:** After fusion block F6, the primary decision state, ATT&CK classification, and
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
discrete kill-chain state tracking, and (c) discrete decision state output is mathematically stable
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
- Deterministic discrete state markers are required at specific layer depths, which is
  structurally incompatible with non-deterministic recurrence depth

**Chosen approach: Depth-segmented latent reasoning with discrete decision state tap points.**

The cross-domain causal chain (Detection → Identity → Vulnerability → IR) is processed as a
continuous cross-domain cross-attention operation in the fusion layer (F1–F6). No explicit
reasoning tokens are emitted during this process. The kill-chain state machine (Section 4.4)
provides a discrete representational scaffold that tracks APT stage transitions without
generating natural language.

Discrete decision state markers are extracted at two fusion layer depths (F3 fast, F6 full) via
linear classification heads. The constrained decoder (Section 6) is invoked separately and only
on analyst request.

**No latent CoT is implemented.** Hidden-state recycling is excluded because:
1. Probing inconsistency makes decision state tap points non-deterministic [Finding: Gap: Interpretable Latent CoT]
2. Deterministic discrete state outputs are a hard architectural requirement [Hard Constraint: must emit discrete decision state markers]
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

### 6.3 Decision State Classifier

- **Architecture:** Linear(1024 → 4) applied at both F3 (fast) and F6 (full) tap points
- **Output:** Softmax over 4 decision classes {0=HIGH_CONFIDENCE_ACTION,
  1=ESCALATE_FOR_REVIEW, 2=DEFER, 3=INSUFFICIENT_CONTEXT}. Class names are
  semantic labels internal to this spec for clarity; the model emits integer class
  indices and calibrated probabilities. Consuming systems map indices to their own
  state names.
- **Early exit rule:** If F3 confidence(class=2) > 0.92 OR confidence(class=3) > 0.85
  → return F3 classification without computing F4–F6 or output head 6.4
- **DCAT override:** If DCAT divergence D > θ_divergence in any encoder and current
  classification is not class 0 or 1: force output to class 1
- **Training:** Cross-entropy on DPO-annotated preference pairs (Stage 4)
- **Output contract:**
  ```json
  {
    "decision_class": 0,
    "decision_probabilities": [0.87, 0.09, 0.03, 0.01],
    "confidence": 0.87,
    "early_exit": false,
    "dcat_override": false
  }
  ```

### 6.4 Calibrated Confidence Scorer

- **Method:** Temperature scaling — single learnable scalar T per domain encoder output,
  applied to logits before softmax [Finding: Architecture Decision Input #2 — research demands
  calibrated confidence, not argmax]
- **Calibration procedure:** Post-training calibration on held-out OOD validation set (Splunk
  Attack Range synthetics not present in training corpus); optimize T by minimizing Negative
  Log-Likelihood (NLL) on OOD set
- **Additionally:** Conformal prediction sets computed offline at p=0.90 and p=0.95 coverage
  levels on the calibration set; these provide deployment-time coverage guarantees — the
  conformal set answers "which decision classes are plausible at 90% confidence" rather than
  a single argmax
- **Output:** Scalar confidence ∈ [0,1] + optional conformal prediction set of valid decision classes

### 6.5 Constrained Justification Decoder

[Finding: Architecture Decision Input #2 — T5-style encoder-decoder with constrained decoder]

- **Architecture:** 8 transformer decoder layers, d=1024, H=16, FFN=4096, ~160M parameters
- **Cross-attention source (critical constraint):** Cross-attention keys and values are derived
  **exclusively** from the top-k=16 AFN-scored input tokens (Section 7), not from raw encoder
  hidden states. The decoder cannot attend to encoder intermediate activations or fusion layer
  states; it can only attend to the pre-scored, AFN-filtered evidence set.
- **Vocabulary constraint:** The output vocabulary is constrained at inference via hard logit
  masking (p = −∞ for blocked tokens) to:
  - **Allowed:** Domain-specific security terminology, ATT&CK technique IDs, native tool call
    JSON tokens (§10.6), structured causal connectives ("because", "triggered by", "correlated
    with", "absent", "indicates", "followed by"), anonymized entity references (e.g., HOST_A, USER_B)
  - **Blocked:** All tokens in the OffSec vocabulary block list (exploit tool names, shellcode
    keywords, payload scaffolding terms); real IP addresses outside RFC1918/documentation
    ranges; free-form natural language instructions; any token from the 20 sentinel token set
- **Max output length:** 256 tokens, enforced by truncating decoder positional embeddings at
  position 256
- **Invocation condition:** The decoder is called **only** when `decision_class` ∈ {0, 1}
  (HIGH_CONFIDENCE_ACTION or ESCALATE_FOR_REVIEW) AND `request_justification_trace=True`.
  It is not called for classes 2 or 3. This eliminates decoder latency for the majority of
  inference calls.
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

### 7.2 Surfacing AFN Output to Analysts

AFN scores are returned in the `afn_scores` field of `VajraInferenceResponse`
as a ranked list of top-16 input fields by L2-norm importance. Consuming systems
are responsible for rendering these for analysts. The recommended presentation:

- A ranked field-importance table: field name, source domain, score, raw value
- The justification trace (if requested) annotated with which AFN-scored fields
  drove each sentence — the cross-attention source mapping is included in the
  decoder's attention metadata
- The evidence-chain DAG rendered as a provenance graph

The `afn_scores` field provides sufficient information for any consuming system
to implement analyst-facing interpretability UI without further model calls.
Chain-of-custody logging and cryptographic signing of inference outputs are the
responsibility of the consuming system.

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
| Incident Response | CISA Advisories, FBI Flash Alerts, Open Source Playbooks | Thousands of reports; limited structured playbooks | 2/5 | 5% | Almost entirely synthetic; ground-truth decision state labels from DPO only |
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

### 8.1.1 Quantization-Aware Training Schedule

Quantization-Aware Training is initialized from epoch zero using the wNa8o8 schema. This is not
post-training quantization — compression tolerance is baked into the weights during the training
loop itself. PTQ applied after training would destroy the precision required for exact security
reasoning on structured fields like CVSS vectors and STIX objects. Three mechanisms are applied
simultaneously from the first training step: channel-wise 2-bit quantization on the constrained
decoder layers, 8-bit static KV caches across all domain encoders and the fusion layer, and
simulated low-precision math during the forward pass so gradients reflect quantized weight
behavior. The target is a model that runs under 4GB VRAM on the encoder-fusion path and under
2GB additional for the decoder, enabling deployment on standard analyst-grade hardware without
a separate quantization pass before shipping.

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
   - `decision_class`: integer 0|1|2|3
   - `confidence_rationale`: single terse sentence
3. Compressed output is re-encoded as a sentinel-only trace:
   ```
   <|S_DOMAIN:detection|><|S_DOMAIN:identity|><|S_START|>
   <|S_KEY|>technique<|S_VAL|>T1003<|S_KEY|>technique<|S_VAL|>T1021
   <|S_KEY|>evidence<|S_VAL|>EventID_4688<|S_KEY|>evidence<|S_VAL|>LogonType_3
   <|S_KEY|>decision_class<|S_VAL|>1<|S_END|>
   ```
4. Student (Vajra) is trained via cross-entropy on these compressed traces

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

### 8.4 Stage 4 — Decision State Alignment via DPO

[Finding: Confirmed Borrowing: Argilla + evtx-sigma-checker]

**Platforms:** Argilla for preference annotation; evtx-sigma-checker for automated validation.

**DPO data pipeline:**
1. Security analysts annotate preferred vs. rejected decision state assignments for real incident
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

### 8.5 MTP Drafter Co-Training

A lightweight Multi-Token Prediction drafter model is co-trained alongside the primary Vajra
model throughout Stages 1 through 4. The drafter uses speculative decoding: it drafts multiple
future tokens in parallel which the primary model then verifies in a single forward pass. This
yields up to 3× latency reduction on the constrained decoder path without quality or reasoning
degradation, because verification is the primary model's judgment — the drafter only proposes.
The drafter architecture is a shallow transformer sharing the primary model's tokenizer and
embedding weights but with 2 decoder layers and d=512, approximately 15M parameters. It is
trained on the same corpus as the primary model with an additional next-token prediction head.
At inference time the drafter runs on CPU while the primary model runs on GPU, eliminating idle
GPU cycles during autoregressive generation. The 50ms decoder latency budget in Section 10.3
assumes MTP speculative decoding is active. Without the drafter the decoder budget rises to
approximately 90ms, which still fits within the 150ms total target but removes the margin for
runtime overhead.

---

## Section 9 — Data Schema and Crowdsource Infrastructure

### 9.1 Training Example JSON Schema

```json
{
  "$schema": "https://raw.githubusercontent.com/vajra-foundation/vajra/main/data/schema/training_example.schema.json",
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
  "decision_class": "<0|1|2|3>",
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
`domain`, `input_events` (≥1 entry), `decision_class`

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
4. **Argilla peer annotation:** Validated contributions enter the Argilla queue for decision state
   preference annotation by ≥2 security analysts; disagreements surfaced for adjudication
5. **DPO approval:** Examples achieving annotation consensus → `validation_status = dpo_approved`
   and enter the Stage 4 DPO pool

### 9.4 HuggingFace Organization and Repository Structure

| Repository | Contents |
|------------|----------|
| `vajra-foundation/vajra-pretrain-detection` | Detection/Alerting corpus: CICIDS flows, UNSW-NB15, Sigma rules, Attack Range logs |
| `vajra-foundation/vajra-pretrain-forensics` | Forensics corpus: DARPA OpTC (subsampled + debiased), LANL Unified Host |
| `vajra-foundation/vajra-pretrain-cti` | CTI corpus: MITRE ATT&CK STIX, MISP galaxies, APTnotes extracted structures |
| `vajra-foundation/vajra-pretrain-vuln` | Vulnerability corpus: NVD, CVE JSON, CVSS vectors, LiveCVEBench tasks |
| `vajra-foundation/vajra-pretrain-identity` | Identity corpus: LANL Auth, CMU CERT, synthetic UEBA |
| `vajra-foundation/vajra-pretrain-ir` | IR corpus: CISA advisories (structured), synthetic playbook SOAR data |
| `vajra-foundation/vajra-pretrain-compliance` | Compliance corpus: CIS OVAL, NIST JSON, ISO 27001 controls |
| `vajra-foundation/vajra-dpo-pairs` | DPO preference pairs: post-Argilla annotation, decision state preferences |
| `vajra-foundation/vajra-model` | Model weights (PyTorch safetensors), ONNX export, tokenizer, config |

Each dataset repo requires:
- Dataset card: license (Apache 2.0), intended use, known limitations (e.g., DARPA OpTC
  scripted-benign bias), citation, PII policy
- `CONTRIBUTING.md` at repo root: schema download link, validation toolchain setup instructions,
  Argilla annotation instance link, example contributions for each domain

---

## Section 10 — Model Interface Specification

Vajra exposes a platform-agnostic inference interface. Consuming systems (XDR
platforms, SIEM integrations, agentic orchestrators, CLI tools) implement their
own adapters against this interface. No deployment-specific wire protocol is
defined here.

### 10.1 Input Contract

All input to Vajra is a structured inference request. The canonical Python
dataclass representation (consuming systems adapt to their own serialization):

```python
@dataclass
class VajraInferenceRequest:
    request_id: str                          # UUID4
    events: list[SecurityEvent]              # max 512 events per call
    graph_nodes: list[GraphNode] = field(default_factory=list)  # optional RAG subgraph
    graph_edges: list[GraphEdge] = field(default_factory=list)
    request_justification_trace: bool = False  # invoke constrained decoder

@dataclass
class SecurityEvent:
    event_type: str       # evtx|netflow|stix|cvss|auth|cloudtrail|sigma|cve|ldap|playbook|cis_control
    domain_tag: str       # detection|forensics|cti|vulnerability|identity|ir|compliance
    timestamp_unix_ms: int  # -1 if unknown
    raw_content: str      # sentinel-tokenized string or JSON string

@dataclass
class GraphNode:
    node_id: str
    node_type: str        # process|file|socket|user|network_asset
    label: str
    features: list[float] = field(default_factory=list)  # optional pre-computed features

@dataclass
class GraphEdge:
    from_id: str
    to_id: str
    relation: str         # exec|read|write|connect|fork|spawn|delete
    timestamp_unix_ms: int
```

**Input constraints:**
- Max 512 events per call
- Max 1,024 graph nodes (sampled to 256 via degree-centrality if exceeded — consuming
  system should apply domain-aware pre-sampling before hitting this limit)
- `request_justification_trace=True` only meaningful when the model's decision_class
  output is 0 or 1; consuming system should check before invoking

### 10.2 Output Contract

```python
@dataclass
class VajraInferenceResponse:
    request_id: str
    decision_class: int                  # 0|1|2|3 — consuming system maps to state names
    decision_probabilities: list[float]  # [p0, p1, p2, p3], sum=1.0
    confidence: float                    # calibrated scalar [0.0, 1.0]
    technique_ids: list[str]             # ATT&CK technique IDs, multi-label
    evidence_dag: EvidenceDAG
    afn_scores: list[AFNScore]           # top-16 field importances
    justification_trace: str             # present only if request_justification_trace=True
                                         # and decision_class in {0, 1}
    inference_latency_ms: float
    early_exit: bool                     # True if F3 fast-tap was used
    dcat_override: bool                  # True if DCAT divergence forced class upgrade

@dataclass
class EvidenceDAG:
    nodes: list[DAGNode]
    edges: list[DAGEdge]

@dataclass
class DAGNode:
    node_id: str
    domain: str
    technique_id: str      # ATT&CK technique ID or empty string
    evidence_field: str

@dataclass
class DAGEdge:
    from_id: str
    to_id: str

@dataclass
class AFNScore:
    field_name: str
    field_value: str
    domain: str
    score: float
```

### 10.3 Latency Budget (single A100 80GB, 4096-token input, batch-size=1)

| Component | Budget |
|-----------|--------|
| Input tokenization (3 paths) | ≤5 ms |
| 7 domain encoders (parallel) | ≤60 ms |
| DCAT blocks (within encoders) | included above |
| Epistemic fusion F1–F6 | ≤15 ms |
| AFN computation | ≤15 ms |
| Classification heads + calibration | ≤5 ms |
| **Total (no justification trace)** | **≤100 ms** |
| Constrained decoder (classes 0–1 only) | ≤50 ms additional¹ |
| **Total (with justification trace)** | **≤150 ms** |

Target ≤120 ms for non-decoder path leaves 20 ms margin for inference runtime overhead.

¹ Assumes MTP drafter co-trained per §8.5 is active. Without drafter: ~90ms. Both scenarios fit
within the 150ms total target.

### 10.4 ONNX Export

Two separate ONNX graphs for deployment flexibility:

1. `vajra_encoder_fusion.onnx` — domain encoders + DCAT + epistemic fusion +
   all classification/output heads
2. `vajra_decoder.onnx` — constrained justification decoder, invoked separately

Export parameters:
- opset=17
- Dynamic axes: `batch_size`, `sequence_length` on both graphs
- Quantization: INT8 ONNX export of QAT-trained encoder weights (wNa8o8 schema, §8.1.1);
  this is a format conversion, not post-training compression. Decoder exported at FP16;
  channel-wise 2-bit quantization on decoder layers is already baked via QAT.
- Consuming systems in compiled languages (Go, Rust, C++) use `onnxruntime` bindings
  against these two graphs; no Python runtime required in production

### 10.5 Knowledge Boundary

The boundary between baked-in and externally-provided knowledge:

| Knowledge Type | Location | Rationale |
|----------------|----------|-----------|
| MITRE ATT&CK tactics (14) + techniques (~700) + sub-techniques (~400) | Baked into weights (frozen after Stage 1) | Stable; innate structural knowledge |
| CVE records | External RAG | Updated continuously |
| D3FEND mitigations | External RAG | Per operator decision |
| Identity / asset graphs | External RAG | Enterprise-specific, changes continuously |
| Live threat actor IOC feeds | External RAG | Volatile (minutes) |
| Sigma rule updates (post-training) | External RAG | Community-maintained |

The consuming system is responsible for providing `graph_nodes` and `graph_edges`
from its RAG pipeline. When graph context is required for a domain (identity,
vulnerability) and no graph nodes are provided: `decision_class` will be 3
(insufficient context). This is a hard safety behavior, not configurable.

[Finding: Architecture Decision Input #4; operator knowledge boundary: minimal baked-in
(MITRE tactics/techniques only)]

### 10.6 Function Calling and Agentic Workflow Interface

Vajra supports structured function calling and multi-turn agentic workflows.
This is a native capability, not a consuming-system overlay. The format is
compatible with the OpenAI tool-use schema and the Anthropic tool-use schema,
allowing Vajra to operate within any agentic framework that supports either.

#### Tool Call Format

When Vajra determines that an external tool call is required to complete a
reasoning step, the justification decoder emits a structured tool call block
within the justification trace (visible only when `request_justification_trace=True`):

```json
{
  "type": "tool_call",
  "tool_name": "string",
  "tool_input": { },
  "reasoning": "string — why this tool is needed, in security-analyst terms"
}
```

The consuming system is responsible for:
1. Detecting `"type": "tool_call"` blocks in the justification trace
2. Executing the tool
3. Constructing a new `VajraInferenceRequest` that includes the tool result
   as an additional event with `event_type = "tool_result"` and
   `domain_tag` matching the domain the tool result is relevant to

Tool calls are only emitted when `decision_class` is 0 or 1 and
`request_justification_trace=True`. They are never emitted as part of the
structured output heads (DAG, technique IDs, confidence — these are always
deterministic model outputs, never tool-dependent).

#### Native Tool Vocabulary

The constrained decoder's vocabulary includes a fixed set of tool names that
Vajra is trained to invoke. These are security-domain tools only:

| Tool Name | Input Schema | Returns |
|-----------|-------------|---------|
| `lookup_cve` | `{"cve_id": "CVE-YYYY-NNNNN"}` | CVE record JSON |
| `lookup_technique` | `{"technique_id": "TNNNN[.NNN]"}` | ATT&CK technique detail |
| `query_asset_graph` | `{"entity_id": "string", "hops": 1\|2}` | Subgraph nodes+edges |
| `lookup_ioc` | `{"indicator": "string", "type": "ip\|hash\|domain"}` | IOC reputation record |
| `get_sigma_rule` | `{"rule_id": "string"}` | Sigma rule YAML |
| `lookup_d3fend` | `{"technique_id": "TNNNN"}` | D3FEND mitigation mappings |

The consuming system implements these tools; Vajra only emits the call.
Tool names outside this vocabulary are masked at the logit level — Vajra
cannot call arbitrary tools.

#### Multi-Turn Agentic Reasoning Format

For multi-step investigations (e.g., pivot from a detection event → query
the asset graph → re-evaluate with enriched context), Vajra follows a
stateless multi-turn protocol:

```
Turn 1: Consumer sends initial events → Vajra returns decision + tool_call
Turn 2: Consumer executes tool, sends initial events + tool_result as new event → Vajra returns updated decision
Turn N: Continue until no tool_call in output OR decision_class = 2|3
```

Vajra maintains no session state between turns. The consuming system is
responsible for accumulating the event list across turns. Maximum 8 turns
per investigation chain; the consuming system enforces this limit.

#### Structured Output Mode

When `request_justification_trace=False` (default), Vajra returns only
structured outputs: `decision_class`, `confidence`, `technique_ids`,
`evidence_dag`, `afn_scores`. This mode is safe for fully automated pipelines.

When `request_justification_trace=True`, Vajra additionally runs the
constrained decoder to produce a human-readable justification and any
required tool calls. This mode is intended for analyst-facing workflows.

The two modes have different latency profiles (see §10.3). Automated
high-throughput pipelines should use structured output mode only.

---

## Section 11 — Evaluation Framework

### 11.1 Per-Head Metrics

| Head | Primary Metric | Secondary Metric | Target | Baseline |
|------|---------------|-----------------|--------|---------|
| Decision state classifier | Macro F1 (4-class) | Per-class recall matrix | F1 ≥ 0.91 | No published baseline |
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
  the true decision class in ≥90% and ≥95% of OOD test examples respectively
- Reliability diagrams plotted for each domain encoder output

### 11.3 Baselines

| Baseline | Type | Parameter Count | Benchmark |
|----------|------|----------------|-----------|
| SecureBERT 2.0 (Cisco, ModernBERT-based) | Domain encoder, encoder-only | ~150M | ATT&CK NER, CVSS classification, TTP extraction |
| Foundation-Sec-8B (Cisco) | Domain LLM, decoder | 8B | TTP extraction, zero-shot decision state approximation, CTI QA |
| GPT-4o zero-shot | General LLM | ~unknown | Cross-domain correlation, evidence DAG construction, decision state assignment |
| Splunk SIEM rule correlation | Rule-based | N/A | Alert triage F1, false positive rate, null-state detection |
| 3-agent orchestrated swarm (GPT-4o-mini) | Multi-agent | 3× ~8B | Decision state exact-match on DARPA OpTC scenarios (swarm-safety parity bar) |

The swarm-safety parity bar requires Vajra to achieve exact decision state match on ≥90% of
200 held-out DARPA OpTC APT scenarios compared to the 3-agent baseline. "Exact match" means
identical class index (0/1/2/3), not just same ordinal direction.

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
| MITRE ATT&CK Evaluations Round 5 (Carbanak + FIN7) | Multi-domain | Cross-domain kill-chain reconstruction, decision state assignment |
| CMU CERT Insider Threat v6.2 | Identity, IR | Behavioral anomaly detection, decision state assignment |
| Sigma HQ detection rules (held-out 20%) | Detection | Sigma rule → decision state mapping accuracy |

---

## Section 12 — Experiment Register

These three experiments must be completed empirically before the corresponding spec sections are
locked for implementation. The spec defines them; it does not resolve them.

| ID | Hypothesis | Method | Success Criterion | Spec Sections Unblocked |
|----|-----------|--------|-------------------|------------------------|
| **EXPERIMENT-1** | There exists an optimal temporal window N (number of prior events retained per entity in the DCAT baseline cache) that maximizes null-signal detection without exhausting single-GPU VRAM. A secondary hypothesis is that the interleaved attention mechanism itself has a degradation threshold independent of VRAM — specifically, that beyond a certain baseline window size N the contrastive loss function begins receiving noise-dominated gradients from the NoPE global layers, which do not encode positional distance and therefore cannot distinguish between a recent absence and a distant one. This threshold may be lower than the VRAM ceiling and would be the binding constraint. The sweep in the method column should therefore measure contrastive loss gradient variance as a fourth metric alongside FNR, VRAM, and latency. | Sweep N ∈ {32, 64, 128, 256, 512} on the Detection/Network encoder running DARPA OpTC data. For each N: measure (a) FNR on null-signal validation set, (b) peak VRAM consumption on A100 80GB, (c) inference latency overhead, (d) contrastive loss gradient variance across the NoPE global attention layers — flag if variance exceeds 2× the variance observed at N=32 baseline. Run each N over 500 inference calls with 50 entities tracked simultaneously | FNR ≤ 0.08 AND peak VRAM < 40GB AND latency overhead < 15 ms. If no N satisfies all three: adopt the Pareto-optimal N and accept the trade-off; document in Risk Register | §4.3 DCAT baseline mechanism; §10.3 latency budget; determines whether DCAT is viable on commodity inference hardware |
| **EXPERIMENT-2** | Jointly optimizing (a) continuous cross-domain fusion attention, (b) discrete kill-chain state tracking, and (c) discrete decision state classification converges stably and without single-objective collapse | Train the full Stage 1–Stage 3 pipeline. At epochs 5, 10, and 15: measure ATT&CK Micro F1, decision state Macro F1, kill-chain stage prediction accuracy, and loss curve variance. Record whether any objective degrades while others improve (objective conflict signature). Also run head-group probing at each checkpoint to verify head specialization emerges alongside multi-task stability | All three objectives improve together through epoch 15 (no more than 0.03 F1 drop in any objective after epoch 5). Head-group probing pass rates reach their Section 4.2 targets by epoch 10. If multi-task instability is detected: evaluate (a) loss weighting schedules (GradNorm), (b) sequential objective introduction (kill-chain first, then decision state), (c) separate optimization of the decision state head with frozen fusion encoder | §4.4 kill-chain state tracking; §5 reasoning representation; §6.3 decision state classifier; entire training pipeline schedule |
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
time, the consuming system provides pre-extracted 2-hop neighborhood subgraphs per queried entity;
the GCN encodes a maximum of 256 nodes per call (top-k degree-centrality sampling applied if
subgraph exceeds 1,024 nodes before sampling). [Finding: Open Question #4]

Risk: low-degree nodes that are critical lateral movement pivot points may be dropped by
degree-centrality sampling. Mitigation: the consuming system can apply domain-aware sampling
(e.g., prioritize nodes with Service Principal Names or AdminCount=1 in AD, which are high-value
regardless of degree) before passing the subgraph to Vajra. This mitigation is implemented in
the consuming system, not in Vajra.

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
D3FEND mitigations, identity graph topology, and all threat actor profiles must be provided by
the consuming system's RAG pipeline on every inference call. This couples model accuracy to
RAG availability.

Mitigation:
- The consuming system should maintain a local replica cache of the most-accessed CVE and
  D3FEND records (recommended: top 10,000 CVEs by EPSS score, updated daily)
- `decision_class=3` (INSUFFICIENT_CONTEXT) is the hard fallback when RAG is unavailable —
  inference does not proceed on vulnerability or identity domain inputs without required graph
  context; this is returned via the standard output contract, not a runtime error
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
4. Decoder output is passed through a static blocklist filter before being returned in the
   `justification_trace` field as a final defense-in-depth layer
5. The constrained decoder is only invoked for classes 0 or 1 — reducing its attack surface
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
| Output heads (DAG + ATT&CK + decision state + calibration) | ~35M |
| MTP drafter (2L, d=512; incremental over shared embedding weights) | ~15M |
| **Grand Total** | **~1.175B** |

Target ceiling: 4.0B. Current estimate: ~1.175B. Headroom: ~2.825B.  
The headroom provides margin for architecture ablations identified by EXPERIMENT-2, potential
decoder depth increases for justification quality, and future domain encoder scaling if data
quality allows.

---

## Appendix B — Security Constraints Compliance Checklist

| Constraint | Implementation |
|------------|---------------|
| Sub-4B parameters, single GPU | ~1.16B total; QAT-trained from epoch zero via wNa8o8 schema; INT8 ONNX export is a format conversion of already-quantization-tolerant weights, not a precision-degrading post-hoc compression. |
| Real-time SOC latency | ≤120ms target; ONNX INT8 on A100; early-exit F3 fast path |
| No exploit synthesis in weights | No decoder head produces exploit code; vocabulary mask is structural (logit-level), not policy |
| No generative free-text attack tooling | Constrained decoder: hard vocabulary mask; max 256 tokens; AFN-only cross-attention source |
| No PII / no real IPs | RFC1918/RFC5737 only in training data; PII scan in validation pipeline; contributor guide enforces |
| Apache 2.0 license | All weights, code, and training data published under Apache 2.0 |
| Discrete human-auditable decision state markers | Emitted at F3 (fast) and F6 (full) tap points; consuming system is responsible for audit trail persistence |
| AFN interpretability (not raw attention) | AFN L2-norm at Layer-8 analog; attention heatmaps not exposed to consumers |
