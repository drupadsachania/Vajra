# Contributing to Vajra

The highest-value contribution right now is **labeled training data**, especially for the identity/access and incident response domains (both rated critically low in the domain inventory). Architecture contributions are not accepted at this stage — the spec is locked.

---

## What to Contribute

A contribution is a single JSON file conforming to the training example schema at `data/schema/training_example.schema.json`.

**Minimum viable contribution:** `domain`, at least one `input_events` entry, and `decision_class`.

**Higher-value additions:**
- `null_signals` + `provenance_graph` → makes the example eligible for Stage 3 absence training (BaNEL loss)
- `reasoning_trace` → makes the example eligible for Stage 2 CoT distillation
- `expected_techniques` → enables ATT&CK technique head supervision

Domain priority (highest need first):

| Domain | Gap level | What's missing |
|--------|-----------|----------------|
| Identity/Access | Critical | Lateral movement, credential access, auth anomaly scenarios |
| Incident Response | Critical | IR decision traces, triage sequences, containment logic |
| Detection | Moderate | Evasion-aware detection, low-and-slow scenarios |
| Compliance | Moderate | Control-to-telemetry mappings |
| Forensics | Low | Well-served by DARPA OpTC; targeted gaps only |
| CTI | Low | Well-served by ATT&CK STIX; no gap |
| Vulnerability | Low | NVD + ExploitDB cover the core |

---

## Anonymization Requirements

These are hard constraints enforced by the ingest pipeline:

- **No real IP addresses.** Use RFC1918 space (`10.x.x.x`, `172.16–31.x.x`, `192.168.x.x`) or RFC5737 documentation ranges (`192.0.2.x`, `198.51.100.x`, `203.0.113.x`). The loader rejects any example containing a routable public IP.
- **No real hostnames.** Use `host-001`, `ws-finance-03`, or similar opaque identifiers.
- **No PII.** No real usernames, email addresses, or employee identifiers. Use anonymized IDs.
- **No proprietary signatures.** Do not contribute content derived from commercial detection content you do not own.

---

## Validation Pipeline

Every contribution passes through four automated checks before entering annotation:

1. **JSON Schema linting** — validated against `data/schema/training_example.schema.json`
2. **IP/PII scan** — automated scan for real IPs and common PII patterns; fails hard on violation
3. **Sigma rule validation** — if `event_type == evtx` and `reasoning_trace` references a Sigma rule ID, the rule is verified to fire on the provided event
4. **Duplicate detection** — cosine similarity check against the existing corpus; near-duplicates are flagged for human review

---

## Annotation

Contributions that pass automated validation enter the **Argilla DPO annotation queue**. Annotators produce preference pairs `(y_w, y_l)` from your scenario. You do not need to do this yourself — the annotation infrastructure handles it.

If you want to participate in annotation (not just data contribution), reach out via the repository issues.

---

## Submission

1. Fork the repository
2. Add your example as `data/contributions/<your_scenario_id>.json`
3. Run validation locally: `python scripts/validate_contribution.py data/contributions/<your_scenario_id>.json`
4. Open a pull request — automated CI runs the full validation pipeline

The `contributor_id` field in your example's `metadata` block should be the SHA-256 of an anonymized identifier you choose. It is used for attribution tracking only.

---

## Synthetic Data

Synthetic contributions are welcome and treated equally to empirical ones. The `metadata.synthetic` flag must be set to `true`. Preferred generation methods:

- **Caldera** for lateral movement and credential access scenarios (identity domain)
- **Splunk Attack Range** for labeled EVTX generation (detection domain)
- **Teacher model distillation** (Opus 4.8 generating IR decision traces) for incident response — coordinate via issues before large synthetic runs to avoid corpus homogeneity

---

## License

All contributions are accepted under Apache 2.0. By submitting a contribution you confirm you have the right to license it under these terms.
