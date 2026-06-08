"""Training data loader — §9.1 JSON schema format.

Parses JSON examples and routes events to tokenization path A/B/C.
Produces (input_ids, domain_tags, source_type_ids, timestamps, null_signal_mask) tuples.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


DOMAIN_TAG_MAP = {
    "detection_network": 0,
    "forensics_provenance": 1,
    "cti_stix": 2,
    "vulnerability_risk": 3,
    "identity_access": 4,
    "incident_response": 5,
    "compliance": 6,
}

TOKENIZATION_PATH_MAP = {
    "cvss": "B",
    "netflow": "B",
    "ip": "B",
    "mitre_kg": "C",
    "graph": "C",
    "evtx": "A",
    "stix": "A",
    "text": "A",
    "default": "A",
}


@dataclass
class TrainingExample:
    events: list[dict] = field(default_factory=list)
    graph_nodes: list[dict] = field(default_factory=list)
    graph_edges: list[dict] = field(default_factory=list)
    labels: dict = field(default_factory=dict)
    null_signals: list[str] = field(default_factory=list)
    stage_3_eligible: bool = False


@dataclass
class ProcessedBatch:
    events: list[dict]
    domain_tags: list[int]
    source_type_ids: list[int]
    timestamps: list[float | None]
    null_signal_mask: list[bool]
    labels: dict
    stage_3_eligible: bool


class TrainingExampleLoader:
    """Loads §9.1 JSON training examples from a file or list of dicts."""

    def __init__(self, source: str | Path | list[dict]):
        if isinstance(source, (str, Path)):
            with open(source) as f:
                self._examples: list[dict] = json.load(f)
        else:
            self._examples = source

    def __len__(self) -> int:
        return len(self._examples)

    def __iter__(self) -> Iterator[ProcessedBatch]:
        for raw in self._examples:
            yield self._process(raw)

    def _process(self, raw: dict) -> ProcessedBatch:
        events = raw.get("events", [])
        null_signals = raw.get("null_signals", [])
        labels = raw.get("labels", {})

        domain_tags: list[int] = []
        source_type_ids: list[int] = []
        timestamps: list[float | None] = []
        null_signal_mask: list[bool] = []

        for evt in events:
            event_type = evt.get("event_type", "text").lower()
            domain = evt.get("domain", "detection_network")
            ts = evt.get("timestamp", None)

            domain_tags.append(DOMAIN_TAG_MAP.get(domain, 0))
            source_type_ids.append(evt.get("source_type_id", 0))
            timestamps.append(float(ts) if ts is not None else None)

            is_null = event_type in null_signals or evt.get("is_null_signal", False)
            null_signal_mask.append(is_null)

        # Stage 3 eligibility: example has at least one null_signal event
        stage_3_eligible = any(null_signal_mask)

        return ProcessedBatch(
            events=events,
            domain_tags=domain_tags,
            source_type_ids=source_type_ids,
            timestamps=timestamps,
            null_signal_mask=null_signal_mask,
            labels=labels,
            stage_3_eligible=stage_3_eligible,
        )

    @staticmethod
    def tokenization_path_for(event_type: str) -> str:
        return TOKENIZATION_PATH_MAP.get(event_type.lower(), "A")
