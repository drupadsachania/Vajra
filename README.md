# Vajra

**India's first security-native foundation model. Built from the ground up.**

Vajra is a sub-4B parameter, full-spectrum cybersecurity SLM trained from scratch — not a fine-tuned general model. It reasons natively across detection, forensics, threat intelligence, vulnerability, identity, incident response, and compliance domains in a single inference pass, with an architecture purpose-built for security telemetry rather than adapted from natural language processing.

Named after the indestructible weapon of Indra — precise, domain-native, adversary-aware.

---

## What Vajra Is Not

Not a fine-tuned GPT variant. Not a BERT model adapted for CTI text. Not a wrapper around a general LLM. Not a multi-agent orchestration system pretending to be a model.

Vajra is a transformer architecture designed from research findings, trained on indigenous security data, with attention heads, tokenization paths, and training objectives that reflect how security actually works — not how language works.

---

## Architecture in Brief

Seven parallel domain encoders (Detection, Forensics, CTI, Vulnerability, Identity, IR, Compliance) with isolated parameter namespaces fuse at an epistemic layer that produces structured outputs: a decision class, calibrated confidence, ATT&CK technique classifications, and an evidence DAG. A constrained decoder produces human-readable justification traces for analyst-facing workflows only. No free-form generation.

Key architectural decisions and their research justifications live in `ARCHITECTURE.md`.

---

## Status

**Active development. Pre-training data collection phase.**

- [x] Architecture specification locked
- [x] Tokenization pipeline designed (S-TOON sentinel + numerical + graph paths)
- [x] Training schema defined
- [ ] Data collection and normalization (current)
- [ ] Pretraining corpus assembly
- [ ] Stage 1 pretraining (forensic MLM)
- [ ] Stage 2 CoT distillation
- [ ] Stage 3 adversarial hardening
- [ ] Stage 4 DPO alignment
- [ ] Evaluation and benchmarking
- [ ] HuggingFace model release

---

## Training Data

Vajra is trained on indigenous, high-signal security data — not scraped internet text. The corpus spans seven security domains sourced from public academic datasets, open-source community repositories, and synthetic attack simulations. All training data is anonymized: RFC1918 IPs only, no real hostnames, no PII.

Data collection scripts are in `scripts/data/`. Each domain has its own collection and normalization pipeline. See `data/schema/training_example.schema.json` for the contribution format.

**Current data targets by domain:**

Detection — UNSW-NB15, CICIDS 2017/2018, Sigma HQ rule corpus, synthetic Splunk Attack Range logs

Forensics — DARPA OpTC (subsampled APT traces), DARPA Transparent Computing, LANL Unified Host and Network

Threat Intelligence — MITRE ATT&CK STIX 2.1, MISP galaxies, APTnotes parsed structures

Vulnerability — NVD/CVE JSON dumps, ExploitDB structured CSV, LiveCVEBench

Identity — LANL Auth dataset, CMU CERT Insider Threat v6.2, synthetic Caldera lateral movement traces

Incident Response — CISA advisories (structured), synthetic IR decision traces via teacher model distillation

Compliance — CIS Benchmarks OVAL, NIST 800-53 OSCAL JSON, NIST CSF controls

---

## Contribute Data

The biggest bottleneck is training data, specifically for identity/access and incident response domains — both rated critically low in our domain inventory. If you work in security and want to contribute labeled scenarios, see `CONTRIBUTING.md`.

Contributions go through automated schema validation, Sigma rule checking where applicable, and PII scanning before entering the annotation pipeline. You contribute the scenario; Argilla handles the DPO preference annotation.

Community data will be hosted at `huggingface.co/vajra-foundation` (org setup in progress).

---

## Research Foundation

Architecture decisions are grounded in a two-pass research study covering 12 research threads across 60+ sources. The full research findings report is in `docs/research/`. Key borrowed techniques: ModernBERT alternating attention, Gemma 4 QAT and MTP drafters, Dual-Contrastive Attention (DCAT) for null-state reasoning, Activation Flow Networks for interpretability, S-TOON sentinel boundary tokenization, TransE/RotatE ontological graph embeddings, BaNEL contrastive loss for absence training, Temporal Fusion Transformer continuous-time encoding.

---

## Why Not Fine-Tune an Existing Model

Fine-tuning puts security reasoning on top of a model whose geometry was built for language. The attention patterns, positional encodings, and embedding spaces reflect natural language co-occurrence — not temporal causality between security events, not entity resolution across log schemas, not absence of expected telemetry as a first-class signal.

Vajra's architecture encodes these inductive biases from layer zero. The difference is not performance at the margin — it is whether the model's world model is security or English.

---

## License

Apache 2.0. See `LICENSE`.

---

## Citation

If you use Vajra's architecture, data schema, or research findings in your work:

```
@misc{vajra2026,
  title  = {Vajra: A Security-Native Foundation Model},
  author = {Sachania, Drupad},
  year   = {2026},
  url    = {https://github.com/drupadsachania/Vajra}
}
```
