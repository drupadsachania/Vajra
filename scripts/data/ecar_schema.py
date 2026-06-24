"""
eCAR (Endpoint Compliance and Reporting) schema and Vajra domain mapper.

Defines the canonical eCAR JSON structure used throughout the extraction
pipeline, plus helpers for IP/hostname anonymization and domain routing.

eCAR event types → Vajra domain:
  PROCESS / FILE / REGISTRY  → forensics_provenance
  NETWORK / FLOW             → detection_network
  AUTH / LOGON               → identity_access
  ALERT / SIGNATURE          → cti_stix
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# eCAR action vocabulary (subset used in OpTC + LANL)
# ---------------------------------------------------------------------------

PROCESS_ACTIONS = frozenset({
    "FORK", "EXEC", "EXIT", "CREATE", "INJECT", "OPEN",
})
FILE_ACTIONS = frozenset({
    "READ", "WRITE", "CREATE", "DELETE", "RENAME", "CHMOD", "MMAP",
})
NETWORK_ACTIONS = frozenset({
    "CONNECT", "ACCEPT", "SEND", "RECV", "BIND", "CLOSE",
})
REGISTRY_ACTIONS = frozenset({
    "REG_READ", "REG_WRITE", "REG_CREATE", "REG_DELETE",
})
AUTH_ACTIONS = frozenset({
    "LOGON", "LOGOFF", "AUTH", "KERBEROS", "NTLM",
})

# ---------------------------------------------------------------------------
# Vajra domain routing
# ---------------------------------------------------------------------------

OBJECT_TO_DOMAIN: dict[str, str] = {
    "PROCESS":  "forensics_provenance",
    "FILE":     "forensics_provenance",
    "REGISTRY": "forensics_provenance",
    "NETWORK":  "detection_network",
    "FLOW":     "detection_network",
    "SOCKET":   "detection_network",
    "AUTH":     "identity_access",
    "LOGON":    "identity_access",
    "ALERT":    "cti_stix",
}

_ACTION_TO_DOMAIN: dict[str, str] = {}
for _a in PROCESS_ACTIONS | FILE_ACTIONS | REGISTRY_ACTIONS:
    _ACTION_TO_DOMAIN[_a] = "forensics_provenance"
for _a in NETWORK_ACTIONS:
    _ACTION_TO_DOMAIN[_a] = "detection_network"
for _a in AUTH_ACTIONS:
    _ACTION_TO_DOMAIN[_a] = "identity_access"


def route_domain(obj_type: str, action: str) -> str:
    """Return the Vajra domain for a given eCAR object type + action."""
    if obj_type.upper() in OBJECT_TO_DOMAIN:
        return OBJECT_TO_DOMAIN[obj_type.upper()]
    return _ACTION_TO_DOMAIN.get(action.upper(), "detection_network")


# ---------------------------------------------------------------------------
# Anonymization helpers
# ---------------------------------------------------------------------------

# Deterministic salt — change to re-key the whole corpus.
_ANON_SALT = b"vajra-anon-v1"

_IPV4_RE = re.compile(
    r"(?<![0-9A-Za-z])\d{1,3}(?:\.\d{1,3}){3}(?![0-9])"
)

_RFC1918 = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
]


def _is_rfc1918(ip_str: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_str)
        return any(addr in net for net in _RFC1918)
    except ValueError:
        return False


def _hash4(value: str) -> int:
    """4-byte deterministic hash of a string."""
    return int.from_bytes(
        hashlib.blake2b(value.encode(), key=_ANON_SALT, digest_size=4).digest(),
        "big",
    )


def anonymize_ip(ip_str: str) -> str:
    """Map any routable IPv4 to a deterministic 10.x.x.x address."""
    if _is_rfc1918(ip_str):
        return ip_str
    h = _hash4(ip_str)
    return f"10.{(h >> 16) & 0xFF}.{(h >> 8) & 0xFF}.{h & 0xFF}"


def anonymize_host(hostname: str) -> str:
    """Map a real hostname to an opaque host-NNNN identifier."""
    h = _hash4(hostname)
    return f"host-{h % 9999:04d}"


def anonymize_user(user: str) -> str:
    """Map a real username to user-NNNN."""
    h = _hash4(user)
    return f"user-{h % 9999:04d}"


def scrub_string(s: str) -> str:
    """Replace any routable IPs embedded in a string with anonymized form."""
    def _replace(m: re.Match) -> str:
        return anonymize_ip(m.group(0))
    return _IPV4_RE.sub(_replace, s)


# ---------------------------------------------------------------------------
# eCAR event dataclass
# ---------------------------------------------------------------------------

@dataclass
class EcarEvent:
    """Canonical eCAR event in the Vajra training schema (§9.1 compatible)."""

    domain: str                       # Vajra domain (routed by route_domain)
    event_type: str                   # "ecar"
    event_id: str                     # deterministic ID
    timestamp: float                  # Unix seconds
    source_dataset: str               # "darpa_optc" | "lanl_auth" | etc.

    # eCAR core fields
    action: str                       # EXEC, WRITE, CONNECT, LOGON, …
    object_type: str                  # PROCESS, FILE, NETWORK, AUTH, …
    host_id: str                      # anonymized host identifier

    # Optional enrichment
    pid: int | None = None
    ppid: int | None = None
    subject_exe: str | None = None    # process image path
    predicate_path: str | None = None # file/registry path
    src_ip: str | None = None
    dst_ip: str | None = None
    src_port: int | None = None
    dst_port: int | None = None
    user: str | None = None

    # Attack context (populated when known)
    label: str | None = None          # "APT_SCENARIO" | "RED_TEAM" | "BENIGN"
    attack_phase: str | None = None   # kill-chain phase if labeled
    techniques: list[str] = field(default_factory=list)  # ATT&CK IDs

    # Raw properties passthrough (scrubbed)
    properties: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @classmethod
    def make_id(cls, dataset: str, raw_id: str) -> str:
        h = hashlib.sha1(f"{dataset}:{raw_id}".encode()).hexdigest()[:16]
        return f"{dataset}_{h}"


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------

class EcarWriter:
    """Streaming JSONL writer with per-domain output files."""

    DOMAINS = [
        "detection_network",
        "forensics_provenance",
        "cti_stix",
        "vulnerability_risk",
        "identity_access",
        "incident_response",
        "compliance",
    ]

    def __init__(self, out_dir: str):
        import os
        os.makedirs(out_dir, exist_ok=True)
        self._out_dir = out_dir
        self._handles: dict[str, Any] = {}
        self._counts: dict[str, int] = {d: 0 for d in self.DOMAINS}

    def write(self, event: EcarEvent) -> None:
        domain = event.domain
        if domain not in self._handles:
            path = f"{self._out_dir}/{domain}.jsonl"
            self._handles[domain] = open(path, "a", encoding="utf-8")
        self._handles[domain].write(event.to_json() + "\n")
        self._counts[domain] = self._counts.get(domain, 0) + 1

    def close(self) -> dict[str, int]:
        for fh in self._handles.values():
            fh.close()
        return self._counts

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
