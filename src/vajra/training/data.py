"""Training data loader — §9.1 JSON schema format.

Parses JSON examples and routes events to tokenization path A/B/C.
Produces (input_ids, domain_tags, source_type_ids, timestamps, null_signal_mask) tuples.
"""

from __future__ import annotations

import ipaddress
import json
import re
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


# Only RFC1918 private space and RFC5737 documentation ranges are permitted in
# training data (REQUIREMENTS.lock: real_ips_in_training = false). Loopback and
# unspecified addresses are also allowed as benign non-routable literals.
_ALLOWED_IP_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("192.0.2.0/24"),     # RFC5737 TEST-NET-1
    ipaddress.ip_network("198.51.100.0/24"),  # RFC5737 TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),   # RFC5737 TEST-NET-3
    ipaddress.ip_network("127.0.0.0/8"),       # loopback
    ipaddress.ip_network("0.0.0.0/32"),        # unspecified
]

# Conservative IPv4 dotted-quad matcher; candidates are confirmed via ipaddress.
_IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")


class RealIPInTrainingError(ValueError):
    """Raised when a training example contains a real (public) IP address."""


def _is_allowed_ip(ip_str: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # not actually an IP — not our concern here
    return any(addr in net for net in _ALLOWED_IP_NETWORKS)


def _iter_strings(value) -> Iterator[str]:
    """Yield every string found in a nested dict/list/scalar structure."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _iter_strings(v)


def find_disallowed_ips(example: dict) -> list[str]:
    """Return all real/public IP literals found anywhere in an example."""
    violations: list[str] = []
    for s in _iter_strings(example):
        for candidate in _IPV4_RE.findall(s):
            if not _is_allowed_ip(candidate):
                violations.append(candidate)
    return violations


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
    """Loads §9.1 JSON training examples from a file or list of dicts.

    By default (`validate_ips=True`) every example is scanned at ingest for real
    (public) IP literals; any match raises `RealIPInTrainingError`, enforcing the
    'no real IPs in training data' hard constraint at the data boundary rather
    than relying on a config flag alone.
    """

    def __init__(self, source: str | Path | list[dict], validate_ips: bool = True):
        if isinstance(source, (str, Path)):
            with open(source) as f:
                self._examples: list[dict] = json.load(f)
        else:
            self._examples = source

        if validate_ips:
            self._validate_ips()

    def _validate_ips(self) -> None:
        for idx, example in enumerate(self._examples):
            bad = find_disallowed_ips(example)
            if bad:
                raise RealIPInTrainingError(
                    f"Example {idx} contains disallowed real IP address(es): "
                    f"{sorted(set(bad))}. Only RFC1918 and RFC5737 documentation "
                    f"ranges are permitted in training data."
                )

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
