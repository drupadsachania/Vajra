# Data Acquisition

The acquisition manifest is `manifest.json` — the single source of truth for what goes
into the pretraining corpus, in priority order. Raw pulls land in `data/raw/` which is
**gitignored**: raw data never enters git. Only normalized, IP-sanitized splits
(validated by `vajra.training.data.TrainingExampleLoader`) are published to the
`vajra-foundation` HuggingFace org.

## HuggingFace pulls

```bash
pip install huggingface_hub
python scripts/data/pull_hf_datasets.py            # all manifest entries
python scripts/data/pull_hf_datasets.py --dry-run  # preview
python scripts/data/pull_hf_datasets.py --only s0u9ata/security-kg
```

Each pull writes a `PULL_RECEIPT.json` (commit SHA, file list, license) into the
dataset's staging directory so the corpus snapshot is exactly reproducible.

| Priority | Dataset | Domains | Action |
|---|---|---|---|
| 1 | `s0u9ata/security-kg` | CTI, Vuln, Compliance, KG embeddings | Pull all 28 configs |
| 2 | `witfoo/precinct6-cybersecurity` | Detection, Forensics, IR | Pull `signals` config |
| 5 | `tumeteor/Security-TTP-Mapping` | CTI technique mapping | Pull — **check CC license terms before redistribution** |
| 6 | `clydeiii/cybersecurity` | CTI APT reports | Pull — needs NER pipeline before tokenization |
| 7 | `AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1` | IR, Compliance CoT | Pull |

## External form-request sources (start now — 1-2 week lead time)

| Priority | Dataset | Domain | Access |
|---|---|---|---|
| 3 | DARPA OpTC (6-day APT eCAR traces) | Forensics | Data sharing agreement via DARPA program page |
| 4 | LANL Unified Host and Network / Auth | Identity | Request form at csr.lanl.gov/data |

**Never publish raw DARPA or LANL data** — only derivatives permitted by the
respective agreements, after RFC1918 normalization.

## Synthetic generation (priority 8)

Identity and Detection gaps are closed with MITRE Caldera + Atomic Red Team runs
against an instrumented lab range (RFC1918 by construction). Generation tooling
will live in `scripts/data/synthetic/` once the range is stood up.

## Sandbox note

Claude Code remote environments block `huggingface.co` by default ("Host not in
allowlist"). Either run the pull script locally, or add `huggingface.co` and
`cdn-lfs.huggingface.co` to the environment's network allowlist.
