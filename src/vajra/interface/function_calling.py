"""§10.6 Function calling and agentic workflow interface.

Native tool vocabulary (6 tools). Tool names masked at logit level — any
decoder output containing an unknown tool name is silently discarded.
Max 8 agentic turns per request; stateless protocol.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

NATIVE_TOOLS = frozenset({
    "lookup_cve",
    "lookup_technique",
    "query_asset_graph",
    "lookup_ioc",
    "get_sigma_rule",
    "lookup_d3fend",
})

MAX_AGENTIC_TURNS = 8


@dataclass
class ToolCall:
    tool_name: str
    parameters: dict = field(default_factory=dict)
    call_id: Optional[str] = None


def _extract_json_objects(text: str) -> list[str]:
    """Scan text for top-level {...} JSON objects, handling nested braces."""
    objects = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth = 0
            start = i
            in_string = False
            escape = False
            for j in range(i, len(text)):
                ch = text[j]
                if escape:
                    escape = False
                    continue
                if ch == '\\' and in_string:
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        objects.append(text[start:j + 1])
                        i = j + 1
                        break
            else:
                break
        else:
            i += 1
    return objects


def parse_tool_calls(decoder_output: str, turn: int = 0) -> list[ToolCall]:
    """Extract valid tool-call blocks from decoder text output.

    Searches for JSON objects with `"type": "tool_call"` and `"tool_name"`.
    Unknown tool names are silently rejected (masked).

    `turn` is the 0-indexed agentic turn number; once it reaches
    MAX_AGENTIC_TURNS no further tool calls are emitted (§10.6 caps a request
    at 8 turns), so an agentic loop terminates deterministically.

    Returns list of validated ToolCall objects (may be empty).
    """
    if turn >= MAX_AGENTIC_TURNS:
        return []

    results: list[ToolCall] = []
    for raw in _extract_json_objects(decoder_output):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if obj.get("type") != "tool_call":
            continue

        tool_name = obj.get("tool_name", "")
        if tool_name not in NATIVE_TOOLS:
            # Unknown tool — masked/rejected per §10.6
            continue

        results.append(ToolCall(
            tool_name=tool_name,
            parameters=obj.get("parameters", {}),
            call_id=obj.get("call_id"),
        ))

    return results
