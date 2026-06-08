# Data Collection Plan

The binding constraint is not storage or bandwidth — it's that every dataset must survive the tokenization pipeline. Numerical fields go through Path B, graph-structured data through Path C, text through Path A with sentinel boundaries. Anything that can't be normalized into one of those three paths without destroying its semantics doesn't go in.

---

## Domain 1 — Detection (usability 3/5, supplement heavily)

Primary sources worth pulling: UNSW-NB15 is the cleanest of the public network intrusion sets — structured CSV flow features, 9 attack categories, reasonable class balance. Pull it. CICIDS 2017 has volume but synthetic homogeneity — pull it but weight it low and plan to debias. The real value here is the Sigma HQ repository — over 3,000 community-maintained detection rules in YAML. These are pure behavioral logic that map directly to ATT&CK techniques. Pull the entire repo. Format is already machine-readable. For raw log generation against Sigma rules, Splunk Attack Range is the tool — it's open source and generates labeled EVTX from simulated attacks. Plan synthetic generation runs here, not pulls from internet archives.

## Domain 2 — Forensics (usability 4/5, richest public source)

DARPA OpTC is 17.4 billion events. At SLM scale you don't use all of it — you subsample aggressively. Target the 6-day APT scenario traces specifically, not the benign workload bulk. The eCAR JSON format is already structured and sentinel-tokenizable. DARPA Transparent Computing datasets are smaller and more tightly labeled — pull those too. LANL Unified Host and Network is useful for the auth sequences specifically. All of these are publicly available via their respective program pages and have been mirrored on academic repositories. The DARPA datasets require a data sharing agreement form — plan for a 1–2 week wait.

## Domain 3 — CTI (usability 5/5, cleanest signal)

MITRE ATT&CK full STIX 2.1 JSON — pull directly from the MITRE CTI GitHub repository, it's always current. This is the baked-in knowledge graph source — TransE/RotatE training runs on this. MISP galaxies are on GitHub, pull them. APTnotes is an archived collection of threat actor reports — PDFs that need NLP extraction, lower priority for the SLM but worth having for Stage 2 CoT distillation. The CTI corpus from SecureBERT 2.0's training data has been partially published — worth checking what Cisco released openly.

## Domain 4 — Vulnerability (usability 4/5)

NVD full CVE dataset is available via their API and as annual JSON dumps — pull the structured JSON, not the HTML. Every CVE record has a structured description, CVSS vector, CWE classification, and affected product list. This is Path B territory for CVSS vectors and Path A for descriptions. ExploitDB is on GitHub as a structured CSV with exploit text — useful for exploitability context. LiveCVEBench is the highest-signal source here — 190 agentic executable reasoning tasks, directly relevant to the model's output heads.

## Domain 5 — Identity (usability 2/5, biggest gap)

LANL Auth dataset is the best public source — hundreds of millions of authentication events, structured CSV. CMU CERT Insider Threat v6.2 is available via request from CMU — behavioral anomaly ground truth. This domain will require the most synthetic augmentation. Plan Caldera runs specifically targeting lateral movement and credential access scenarios to generate labeled auth telemetry. The gap here is real and the synthetic flywheel is the only viable path for SLM-scale training.

## Domain 6 — IR (usability 2/5, almost entirely synthetic)

CISA advisories and FBI Flash Alerts are public PDFs — useful for CoT distillation but not raw pretraining. SOAR playbook YAML from open repositories gives some prescriptive structure. Realistically, this domain's training data is almost entirely generated: Opus 4.8 as teacher producing IR decision traces from synthetic attack scenarios, validated by Argilla DPO. Accept this early.

## Domain 7 — Compliance (usability 3/5)

CIS Benchmarks are available in OVAL XML — machine-readable. NIST 800-53 is available in OSCAL JSON from NIST's GitHub. ISO 27001 controls are not freely available in machine-readable form — use NIST CSF as the equivalent. These need mapping topologies to connect abstract controls to concrete telemetry — this is the hard part and will be largely crowdsourced.

---

## Storage and Staging

HuggingFace datasets repos under `vajra-foundation` org — one repo per domain. Raw pulls go into a private staging repo first, get normalized through the tokenization pipeline, then the normalized splits go public. Never publish raw DARPA data directly — check the data sharing agreement terms. Everything RFC1918-only for IPs before anything goes public.

For the internet archive pulls: use `requests` + `tqdm` for the structured JSON sources (NVD, MITRE, MISP). Git clone for the GitHub-hosted sources (Sigma HQ, ExploitDB, MITRE CTI). DARPA and LANL require form-based access — do those in parallel with everything else since they have lead time.
