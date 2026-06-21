"""VajraInferenceRequest / VajraInferenceResponse dataclasses (§10.1, §10.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SecurityEvent:
    event_type: str
    domain: str
    data: dict = field(default_factory=dict)
    timestamp: Optional[float] = None
    source_type_id: int = 0


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    attributes: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    src: str
    dst: str
    relation: str


@dataclass
class DAGNode:
    node_id: str
    event_type: str
    confidence: float


@dataclass
class DAGEdge:
    src: str
    dst: str
    relation: str


@dataclass
class EvidenceDAG:
    nodes: list[DAGNode] = field(default_factory=list)
    edges: list[DAGEdge] = field(default_factory=list)


@dataclass
class AFNScore:
    domain: str
    token_index: int
    importance: float


@dataclass
class VajraInferenceRequest:
    request_id: str
    events: list[SecurityEvent]
    graph_nodes: list[GraphNode] = field(default_factory=list)
    graph_edges: list[GraphEdge] = field(default_factory=list)
    request_justification_trace: bool = False


@dataclass
class VajraInferenceResponse:
    request_id: str
    decision_class: int  # 0–3 integer index
    decision_probabilities: list[float]  # softmax over 4 classes
    confidence: float
    technique_ids: list[str]  # ATT&CK technique IDs with prob > τ
    evidence_dag: EvidenceDAG
    afn_scores: list[AFNScore]
    justification_trace: str  # empty when decoder not invoked
    inference_latency_ms: float
    early_exit: bool
    dcat_override: bool
