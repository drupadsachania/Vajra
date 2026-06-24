"""
DARPA OpTC extractor — eCAR JSON → Vajra training JSONL.

The raw OpTC dataset is ~350GB of eCAR JSON (newline-delimited, .json.gz).
This script streams the files, keeps only events inside the APT scenario
windows, anonymizes hosts/IPs, routes to Vajra domains, and writes per-domain
JSONL ready for TrainingExampleLoader.

Usage:
    python extract_optc.py \\
        --raw-dir  /path/to/optc/ecar/ \\
        --out-dir  /path/to/vajra-data/extracted/ \\
        [--start   2019-09-23T00:00:00] \\
        [--end     2019-09-28T23:59:59] \\
        [--max-events 2000000]

The APT scenario ran 23–28 September 2019.  The default window covers all 6
days.  Tighten --start/--end to extract a specific sub-phase.

Output: <out-dir>/{detection_network,forensics_provenance,...}.jsonl
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ecar_schema import (
    EcarEvent,
    EcarWriter,
    anonymize_host,
    anonymize_ip,
    anonymize_user,
    route_domain,
    scrub_string,
)

# ---------------------------------------------------------------------------
# APT scenario operation windows (UTC)
# OpTC evaluation period: 23–28 Sep 2019
# Red-team operations annotated in the Friends-of-OpTC README.
# ---------------------------------------------------------------------------

APT_START_DEFAULT = datetime(2019, 9, 23, 0, 0, 0, tzinfo=timezone.utc)
APT_END_DEFAULT   = datetime(2019, 9, 28, 23, 59, 59, tzinfo=timezone.utc)

# Known attack-phase windows from the OpTC evaluation report.
# Format: (start_ts_ms, end_ts_ms, kill_chain_phase, techniques)
LABELED_WINDOWS: list[tuple[int, int, str, list[str]]] = [
    # Day 1 — Reconnaissance & initial foothold
    (1569196800000, 1569240000000, "RECON",    ["T1592", "T1595"]),
    # Day 2 — Delivery & exploitation
    (1569283200000, 1569326400000, "DELIVER",  ["T1566", "T1203"]),
    # Day 3 — Installation & persistence
    (1569369600000, 1569412800000, "INSTALL",  ["T1547", "T1053"]),
    # Day 4 — C2 establishment
    (1569456000000, 1569499200000, "C2",       ["T1071", "T1105"]),
    # Day 5-6 — Lateral movement & exfiltration
    (1569542400000, 1569628800000, "EXFIL",    ["T1021", "T1041"]),
]


def _label_event(ts_ms: int) -> tuple[str, str, list[str]]:
    """Return (label, attack_phase, techniques) for a given timestamp."""
    for start, end, phase, techs in LABELED_WINDOWS:
        if start <= ts_ms <= end:
            return "APT_SCENARIO", phase, techs
    return "BENIGN", "", []


# ---------------------------------------------------------------------------
# eCAR field parsers for OpTC JSON schema variants
# ---------------------------------------------------------------------------

def _parse_optc_event(raw: dict) -> EcarEvent | None:
    """
    Parse one OpTC eCAR JSON object into an EcarEvent.

    OpTC uses two slightly different schemas across dataset versions:
      v1: flat fields  (action, object, subject, timestamp_ms)
      v2: nested datum (datum.com.bbn.tc.schema...)
    Both are handled here.
    """
    # Unwrap CDM-wrapped format
    if "datum" in raw:
        inner = raw["datum"]
        if not isinstance(inner, dict):
            return None
        # CDM schema key looks like "com.bbn.tc.schema.avro.cdm18.Event"
        for key, val in inner.items():
            if "Event" in key and isinstance(val, dict):
                raw = val
                break
        else:
            return None

    # ── Core fields ──────────────────────────────────────────────────────────
    action      = str(raw.get("action", raw.get("type", "UNKNOWN"))).upper()
    object_type = str(raw.get("object", raw.get("objectType", "UNKNOWN"))).upper()

    # Timestamps: OpTC uses milliseconds since epoch
    ts_raw  = raw.get("timestamp", raw.get("timestampNanos", 0))
    ts_ms   = int(ts_raw) // 1_000_000 if int(ts_raw) > 1e12 else int(ts_raw)
    ts_unix = ts_ms / 1000.0

    raw_host = str(raw.get("host", raw.get("hostId", raw.get("hostname", "unknown"))))
    host_id  = anonymize_host(raw_host)

    # ── Subject (process that performed the action) ───────────────────────────
    subject = raw.get("subject", raw.get("actor", {})) or {}
    pid     = subject.get("pid") or raw.get("pid")
    ppid    = subject.get("ppid") or raw.get("ppid")
    exe_raw = subject.get("exe") or subject.get("path") or raw.get("exe", "")
    subject_exe = scrub_string(exe_raw) if exe_raw else None

    # ── Predicate object (file, registry key, etc.) ───────────────────────────
    pred = raw.get("predicateObject", raw.get("actedUpon", {})) or {}
    pred_path_raw = (
        pred.get("path") or pred.get("name") or
        raw.get("file") or raw.get("path") or
        raw.get("predicateObjectPath", {}).get("string", "")
    )
    predicate_path = scrub_string(pred_path_raw) if pred_path_raw else None

    # ── Network fields ────────────────────────────────────────────────────────
    src_ip = dst_ip = None
    src_port = dst_port = None
    if object_type in ("NETWORK", "FLOW", "SOCKET") or action in (
        "CONNECT", "ACCEPT", "SEND", "RECV", "BIND"
    ):
        net = raw.get("network", raw.get("netflow", {})) or {}
        src_raw = net.get("src_ip") or net.get("srcAddr") or raw.get("src_ip", "")
        dst_raw = net.get("dst_ip") or net.get("dstAddr") or raw.get("dst_ip", "")
        src_ip  = anonymize_ip(src_raw) if src_raw else None
        dst_ip  = anonymize_ip(dst_raw) if dst_raw else None
        src_port = net.get("src_port") or net.get("srcPort")
        dst_port = net.get("dst_port") or net.get("dstPort")

    # ── Labeling ──────────────────────────────────────────────────────────────
    label, phase, techniques = _label_event(ts_ms)

    # ── Unique ID ─────────────────────────────────────────────────────────────
    raw_id = raw.get("uuid") or raw.get("id") or f"{raw_host}:{ts_ms}:{action}"
    event_id = EcarEvent.make_id("optc", raw_id)

    return EcarEvent(
        domain         = route_domain(object_type, action),
        event_type     = "ecar",
        event_id       = event_id,
        timestamp      = ts_unix,
        source_dataset = "darpa_optc",
        action         = action,
        object_type    = object_type,
        host_id        = host_id,
        pid            = int(pid) if pid is not None else None,
        ppid           = int(ppid) if ppid is not None else None,
        subject_exe    = subject_exe,
        predicate_path = predicate_path,
        src_ip         = src_ip,
        dst_ip         = dst_ip,
        src_port       = int(src_port) if src_port else None,
        dst_port       = int(dst_port) if dst_port else None,
        label          = label,
        attack_phase   = phase or None,
        techniques     = techniques,
        properties     = {},
    )


# ---------------------------------------------------------------------------
# File iterator — handles .json, .json.gz, .jsonl, .jsonl.gz
# ---------------------------------------------------------------------------

def _iter_events(raw_dir: str, start_ts: float, end_ts: float):
    """Yield raw dicts from all eCAR files under raw_dir, filtered by time."""
    patterns = ["**/*.json.gz", "**/*.jsonl.gz", "**/*.json", "**/*.jsonl"]
    files: list[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(raw_dir, pat), recursive=True))

    files = sorted(set(files))
    if not files:
        print(f"[WARN] No eCAR files found under {raw_dir}", file=sys.stderr)
        return

    print(f"[INFO] Found {len(files)} eCAR files to process.")

    for fpath in files:
        opener = gzip.open if fpath.endswith(".gz") else open
        try:
            with opener(fpath, "rt", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Quick timestamp pre-filter (ms or s)
                    ts_raw = raw.get("timestamp", raw.get("timestampNanos", 0))
                    ts_s = int(ts_raw) / 1000.0 if int(ts_raw) > 1e9 else float(ts_raw)
                    if not (start_ts <= ts_s <= end_ts):
                        continue

                    yield raw
        except Exception as exc:
            print(f"[WARN] Skipping {fpath}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Vajra training events from DARPA OpTC")
    parser.add_argument("--raw-dir",    required=True,
                        help="Root directory containing OpTC eCAR .json.gz files")
    parser.add_argument("--out-dir",    required=True,
                        help="Output directory for per-domain JSONL files")
    parser.add_argument("--start",      default=APT_START_DEFAULT.isoformat(),
                        help="ISO-8601 UTC start of extraction window")
    parser.add_argument("--end",        default=APT_END_DEFAULT.isoformat(),
                        help="ISO-8601 UTC end of extraction window")
    parser.add_argument("--max-events", type=int, default=0,
                        help="Stop after N events per domain (0 = unlimited)")
    parser.add_argument("--labeled-only", action="store_true",
                        help="Discard BENIGN events (keeps only APT_SCENARIO)")
    args = parser.parse_args()

    start_ts = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp()
    end_ts   = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc).timestamp()

    print(f"[INFO] Extracting OpTC events: {args.start} → {args.end}")
    print(f"[INFO] Output: {args.out_dir}")

    total = 0
    skipped = 0
    domain_counts: dict[str, int] = {}

    with EcarWriter(args.out_dir) as writer:
        for raw in _iter_events(args.raw_dir, start_ts, end_ts):
            event = _parse_optc_event(raw)
            if event is None:
                skipped += 1
                continue

            if args.labeled_only and event.label == "BENIGN":
                continue

            if args.max_events:
                count = domain_counts.get(event.domain, 0)
                if count >= args.max_events:
                    continue
                domain_counts[event.domain] = count + 1

            writer.write(event)
            total += 1
            if total % 100_000 == 0:
                print(f"[INFO] {total:,} events written …")

        counts = writer.close()

    print(f"\n[DONE] {total:,} events written, {skipped:,} skipped.")
    print("Per-domain counts:")
    for domain, n in sorted(counts.items()):
        if n:
            print(f"  {domain}: {n:,}")


if __name__ == "__main__":
    main()
