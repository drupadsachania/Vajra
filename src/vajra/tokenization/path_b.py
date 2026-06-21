"""Path B — numerical sub-tokenizer (CVSS, NetFlow, IP).

None of these fields ever go through BPE.  Each gets a dedicated learned
projection to d_model=1024.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class NetFlowRecord:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int  # e.g. 6=TCP, 17=UDP
    bytes_sent: int
    packets: int
    duration_ms: float


_RFC1918_BLOCKS = [
    ("10.0.0.0", 8),
    ("172.16.0.0", 12),
    ("192.168.0.0", 16),
]


def _is_rfc1918(ip_str: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    for block, prefix in _RFC1918_BLOCKS:
        if addr in ipaddress.ip_network(f"{block}/{prefix}", strict=False):
            return True
    return False


def _ip_to_tensor(ip_str: str) -> torch.Tensor:
    """RFC1918 /24-level one-hot (256 dims) + 1 external-indicator bit → 257 dims."""
    vec = torch.zeros(257)
    try:
        addr = ipaddress.ip_address(ip_str)
        octets = list(addr.packed)
        if _is_rfc1918(ip_str) and len(octets) == 4:
            slash24_index = octets[2]  # 3rd octet identifies the /24
            vec[slash24_index] = 1.0
        else:
            vec[256] = 1.0  # external indicator
    except ValueError:
        vec[256] = 1.0
    return vec


class CvssEncoder(nn.Module):
    """CVSS v3.x vector → d_model embedding.

    Input: 25-dim one-hot over 9 metric fields (AV=4, AC=2, PR=3, UI=2,
           S=2, C=3, I=3, A=3, Severity=3 → not exactly 25; we use a
           25-dim binary representation where each CVSS component maps to
           contiguous slots as specified in §2.2).
    """

    CVSS_DIM = 25

    # Metric → number of possible values
    _METRICS = {
        "AV": ["N", "A", "L", "P"],  # 4
        "AC": ["L", "H"],  # 2
        "PR": ["N", "L", "H"],  # 3
        "UI": ["N", "R"],  # 2
        "S": ["U", "C"],  # 2
        "C": ["N", "L", "H"],  # 3
        "I": ["N", "L", "H"],  # 3
        "A": ["N", "L", "H"],  # 3
        # Slot 22-24: severity bucket (None/Low/Med/High/Crit → capped to 3 bits)
    }
    # AV(4)+AC(2)+PR(3)+UI(2)+S(2)+C(3)+I(3)+A(3) = 22; remaining 3 = severity
    _METRIC_ORDER = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]

    @classmethod
    def _build_slot_starts(cls) -> dict[str, int]:
        offset = 0
        starts: dict[str, int] = {}
        for m in cls._METRIC_ORDER:
            starts[m] = offset
            offset += len(cls._METRICS[m])
        return starts

    def __init__(self, d_model: int = 1024):
        super().__init__()
        self._slot_starts = self._build_slot_starts()

        self.proj = nn.Sequential(
            nn.Linear(self.CVSS_DIM, 512),
            nn.GELU(),
            nn.Linear(512, d_model),
        )

    @classmethod
    def parse_cvss(cls, cvss_str: str) -> torch.Tensor:
        """Parse CVSS v3 vector string to 25-dim one-hot tensor."""
        slot_starts = cls._build_slot_starts()
        vec = torch.zeros(cls.CVSS_DIM)
        # Example: "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
        parts = cvss_str.split("/")
        for part in parts:
            if ":" not in part:
                continue
            key, val = part.split(":", 1)
            if key in cls._METRICS and key in slot_starts:
                choices = cls._METRICS[key]
                if val in choices:
                    slot = slot_starts[key] + choices.index(val)
                    vec[slot] = 1.0
        return vec

    def forward(self, cvss_vec: torch.Tensor) -> torch.Tensor:
        """(batch, 25) → (batch, d_model)."""
        return self.proj(cvss_vec)


class IpEncoder(nn.Module):
    """IP address → d_model via learnable projection of 257-dim one-hot."""

    IP_DIM = 257

    def __init__(self, d_model: int = 1024):
        super().__init__()
        self.proj = nn.Linear(self.IP_DIM, d_model)

    def forward(self, ip_vec: torch.Tensor) -> torch.Tensor:
        """(batch, 257) → (batch, d_model)."""
        return self.proj(ip_vec)

    @staticmethod
    def encode_ip(ip_str: str) -> torch.Tensor:
        return _ip_to_tensor(ip_str)


_RFC1918_NETWORKS = [
    ipaddress.ip_network(f"{b}/{p}", strict=False) for b, p in _RFC1918_BLOCKS
]


class NetFlowEncoder(nn.Module):
    """NetFlow record → 8 per-field projections, each → d_model.

    Fields (8): src_ip, dst_ip, src_port, dst_port, protocol,
                bytes_sent, packets, duration_ms.
    Scalar fields (bytes, packets, duration) use separate Linear(1, d_model)
    projections on log1p-scaled values so the model can distinguish them.
    """

    PORT_VOCAB = 65536
    PORT_DIM = 64

    def __init__(self, d_model: int = 1024):
        super().__init__()
        self.ip_enc = IpEncoder(d_model)
        self.port_embed = nn.Embedding(self.PORT_VOCAB, self.PORT_DIM)
        self.port_proj = nn.Linear(self.PORT_DIM, d_model)
        self.proto_proj = nn.Linear(256, d_model)  # protocol 0-255 one-hot
        # Separate projections for each scalar so the model can distinguish them.
        self.bytes_proj = nn.Linear(1, d_model)
        self.packets_proj = nn.Linear(1, d_model)
        self.duration_proj = nn.Linear(1, d_model)

    def encode_record(self, rec: NetFlowRecord) -> dict[str, torch.Tensor]:
        """Return dict of 8 per-field tensors, each shape (1, d_model)."""
        device = self.bytes_proj.weight.device
        src_ip_t = _ip_to_tensor(rec.src_ip).unsqueeze(0).to(device)
        dst_ip_t = _ip_to_tensor(rec.dst_ip).unsqueeze(0).to(device)

        src_port_t = torch.tensor([rec.src_port], dtype=torch.long, device=device)
        dst_port_t = torch.tensor([rec.dst_port], dtype=torch.long, device=device)

        proto_t = torch.zeros(1, 256, device=device)
        proto_t[0, min(rec.protocol, 255)] = 1.0

        # log1p prevents gradient/activation blow-up on large byte counts.
        bytes_t = torch.tensor([[float(rec.bytes_sent)]], device=device).log1p()
        packets_t = torch.tensor([[float(rec.packets)]], device=device).log1p()
        duration_t = torch.tensor([[float(rec.duration_ms)]], device=device).log1p()

        return {
            "src_ip": self.ip_enc(src_ip_t),
            "dst_ip": self.ip_enc(dst_ip_t),
            "src_port": self.port_proj(self.port_embed(src_port_t)),
            "dst_port": self.port_proj(self.port_embed(dst_port_t)),
            "protocol": self.proto_proj(proto_t),
            "bytes": self.bytes_proj(bytes_t),
            "packets": self.packets_proj(packets_t),
            "duration": self.duration_proj(duration_t),
        }

    def forward(self, rec: NetFlowRecord) -> torch.Tensor:
        """Stack all 8 field embeddings → (1, 8, d_model)."""
        fields = self.encode_record(rec)
        return torch.stack(list(fields.values()), dim=1)
