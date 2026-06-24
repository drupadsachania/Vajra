"""
LANL Unified Host and Network dataset extractor → Vajra eCAR JSONL.

The LANL dataset (csr.lanl.gov/data) contains:
  auth.txt.gz     — 1.6 billion authentication events (CSV)
  redteam.txt.gz  — 749 labeled red-team events (the anomalies)
  proc.txt.gz     — process start/stop events
  flows.txt.gz    — network flow data

This script:
  1. Loads the red-team labels (redteam.txt.gz) to know which time windows
     contain attacks.
  2. Streams auth.txt.gz, keeping events in a ±WINDOW_SECONDS radius around
     each red-team event.  This gives attack context + surrounding benign
     baseline — what BaNEL training needs.
  3. Converts each CSV row to an EcarEvent in the identity_access domain.
  4. Optionally streams proc.txt.gz and flows.txt.gz for detection/forensics.

Usage:
    python extract_lanl.py \\
        --raw-dir   /path/to/lanl/          \\
        --out-dir   /path/to/vajra-data/extracted/ \\
        [--window   3600]                   \\   # seconds around each red event
        [--max-events 500000]

Output: <out-dir>/identity_access.jsonl  (and optionally detection_network.jsonl)

CSV formats (LANL documentation):
  auth.txt:     time,user@domain,src_computer,dst_computer,auth_type,
                logon_type,orientation,success/failure
  redteam.txt:  time,user@domain,src_computer,dst_computer
  proc.txt:     time,user@domain,computer,process_name,start/end
  flows.txt:    duration,src_computer,src_port,dst_computer,dst_port,
                protocol,packets,bytes
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ecar_schema import (
    EcarEvent,
    EcarWriter,
    anonymize_host,
    anonymize_user,
    anonymize_ip,
)

# How many seconds either side of a red-team event to keep
DEFAULT_WINDOW_SECONDS = 3600


# ---------------------------------------------------------------------------
# Step 1: Load red-team labels
# ---------------------------------------------------------------------------

def load_redteam_labels(path: str) -> dict[int, list[dict]]:
    """
    Returns a dict: timestamp_seconds → list of red-team event dicts.

    redteam.txt format: time,user@domain,src_computer,dst_computer
    Time is in seconds since an epoch offset (t=1 = first second of dataset).
    """
    labels: dict[int, list[dict]] = defaultdict(list)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                t = int(parts[0])
            except ValueError:
                continue
            labels[t].append({
                "time":     t,
                "user":     parts[1].strip(),
                "src_comp": parts[2].strip(),
                "dst_comp": parts[3].strip(),
            })
    total = sum(len(v) for v in labels.values())
    print(f"[INFO] Loaded {total} red-team events across {len(labels)} timestamps.")
    return labels


def build_window_set(labels: dict[int, list[dict]], window: int) -> set[int]:
    """
    Return the set of all timestamps within ±window seconds of any red-team event.
    This is used as a fast pre-filter when streaming the large auth file.
    """
    windows: set[int] = set()
    for t in labels:
        for offset in range(-window, window + 1):
            windows.add(t + offset)
    return windows


# ---------------------------------------------------------------------------
# Step 2: Stream auth.txt.gz
# ---------------------------------------------------------------------------

AUTH_ACTIONS = {
    "LogOn":   "LOGON",
    "LogOff":  "LOGOFF",
    "TGT":     "KERBEROS_TGT",
    "TGS":     "KERBEROS_TGS",
}

AUTH_LOGON_TYPES = {
    "Network":      "NTLM_NETWORK",
    "Interactive":  "INTERACTIVE",
    "RemoteInteractive": "RDP",
    "Batch":        "BATCH",
    "Service":      "SERVICE",
}


def _parse_auth_row(parts: list[str], t: int,
                    redteam_at_t: list[dict]) -> EcarEvent | None:
    """Convert one auth CSV row to an EcarEvent."""
    if len(parts) < 8:
        return None

    _, user_raw, src_raw, dst_raw, auth_type, logon_type, orientation, success = (
        parts[0], parts[1], parts[2], parts[3],
        parts[4], parts[5], parts[6], parts[7].strip(),
    )

    action     = AUTH_ACTIONS.get(orientation, "AUTH")
    object_type = "AUTH"
    host_id    = anonymize_host(src_raw)
    user_anon  = anonymize_user(user_raw)
    src_anon   = anonymize_host(src_raw)
    dst_anon   = anonymize_host(dst_raw)

    # Is this specific (user, src, dst) a labeled red-team event?
    is_red = any(
        rt["user"] == user_raw and
        rt["src_comp"] == src_raw and
        rt["dst_comp"] == dst_raw
        for rt in redteam_at_t
    )
    label       = "RED_TEAM" if is_red else "BENIGN"
    attack_phase = "C2" if is_red else None
    techniques   = ["T1078", "T1021"] if is_red else []  # Valid Accounts + Remote Services

    return EcarEvent(
        domain         = "identity_access",
        event_type     = "ecar",
        event_id       = EcarEvent.make_id("lanl_auth", f"{t}:{user_raw}:{src_raw}:{dst_raw}"),
        timestamp      = float(t),
        source_dataset = "lanl_auth",
        action         = action,
        object_type    = object_type,
        host_id        = host_id,
        user           = user_anon,
        src_ip         = src_anon,   # hosts stored as src_ip field for pipeline compat
        dst_ip         = dst_anon,
        label          = label,
        attack_phase   = attack_phase,
        techniques     = techniques,
        properties     = {
            "auth_type":  auth_type,
            "logon_type": AUTH_LOGON_TYPES.get(logon_type, logon_type),
            "success":    success == "Success",
        },
    )


def extract_auth(auth_path: str, window_set: set[int],
                 redteam_labels: dict[int, list[dict]],
                 writer: EcarWriter, max_events: int) -> int:
    """Stream auth.txt.gz and write windowed events."""
    total = 0
    opener = gzip.open if auth_path.endswith(".gz") else open
    with opener(auth_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if not parts or not parts[0].isdigit():
                continue
            t = int(parts[0])
            if t not in window_set:
                continue

            event = _parse_auth_row(parts, t, redteam_labels.get(t, []))
            if event is None:
                continue

            writer.write(event)
            total += 1
            if max_events and total >= max_events:
                print(f"[INFO] Reached --max-events limit of {max_events:,}.")
                break
            if total % 100_000 == 0:
                print(f"[INFO] {total:,} auth events written …")
    return total


# ---------------------------------------------------------------------------
# Step 3: Stream proc.txt.gz (optional — forensics_provenance domain)
# ---------------------------------------------------------------------------

def extract_proc(proc_path: str, window_set: set[int],
                 redteam_labels: dict[int, list[dict]],
                 writer: EcarWriter, max_events: int) -> int:
    """
    proc.txt format: time,user@domain,computer,process_name,start/end
    Maps to forensics_provenance domain.
    """
    total = 0
    opener = gzip.open if proc_path.endswith(".gz") else open
    with opener(proc_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 5 or not parts[0].isdigit():
                continue
            t = int(parts[0])
            if t not in window_set:
                continue

            _, user_raw, comp_raw, proc_raw, start_end = (
                parts[0], parts[1], parts[2], parts[3], parts[4].strip()
            )
            action = "EXEC" if start_end == "Start" else "EXIT"
            event = EcarEvent(
                domain         = "forensics_provenance",
                event_type     = "ecar",
                event_id       = EcarEvent.make_id("lanl_proc", f"{t}:{user_raw}:{comp_raw}:{proc_raw}"),
                timestamp      = float(t),
                source_dataset = "lanl_proc",
                action         = action,
                object_type    = "PROCESS",
                host_id        = anonymize_host(comp_raw),
                user           = anonymize_user(user_raw),
                subject_exe    = proc_raw,   # process names are not real IPs
                label          = "RED_TEAM" if t in redteam_labels else "BENIGN",
            )
            writer.write(event)
            total += 1
            if max_events and total >= max_events:
                break
    return total


# ---------------------------------------------------------------------------
# Step 4: Stream flows.txt.gz (optional — detection_network domain)
# ---------------------------------------------------------------------------

def extract_flows(flows_path: str, window_set: set[int],
                  writer: EcarWriter, max_events: int) -> int:
    """
    flows.txt format:
      duration,src_computer,src_port,dst_computer,dst_port,protocol,packets,bytes
    No timestamps — uses sequential row number; map to detection_network.
    NOTE: LANL flow data has no per-row timestamp; we include all flows if
    the file exists (they're already relatively small vs auth).
    """
    total = 0
    opener = gzip.open if flows_path.endswith(".gz") else open
    with opener(flows_path, "rt", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            parts = line.strip().split(",")
            if len(parts) < 8 or not parts[0].isdigit():
                continue

            duration, src_comp, src_port, dst_comp, dst_port, proto, packets, nbytes = (
                parts[0], parts[1], parts[2], parts[3],
                parts[4], parts[5], parts[6], parts[7].strip(),
            )
            event = EcarEvent(
                domain         = "detection_network",
                event_type     = "ecar",
                event_id       = EcarEvent.make_id("lanl_flows", str(i)),
                timestamp      = float(i),   # no wall-clock; use row index
                source_dataset = "lanl_flows",
                action         = "FLOW",
                object_type    = "FLOW",
                host_id        = anonymize_host(src_comp),
                src_ip         = anonymize_host(src_comp),
                dst_ip         = anonymize_host(dst_comp),
                src_port       = int(src_port) if src_port.isdigit() else None,
                dst_port       = int(dst_port) if dst_port.isdigit() else None,
                label          = "BENIGN",
                properties     = {
                    "protocol": proto,
                    "packets":  int(packets) if packets.isdigit() else None,
                    "bytes":    int(nbytes) if nbytes.isdigit() else None,
                    "duration": int(duration) if duration.isdigit() else None,
                },
            )
            writer.write(event)
            total += 1
            if max_events and total >= max_events:
                break
            if total % 100_000 == 0:
                print(f"[INFO] {total:,} flow events written …")
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Vajra identity_access events from LANL Auth dataset"
    )
    parser.add_argument("--raw-dir",    required=True,
                        help="Directory containing auth.txt.gz, redteam.txt.gz, etc.")
    parser.add_argument("--out-dir",    required=True,
                        help="Output directory for per-domain JSONL files")
    parser.add_argument("--window",     type=int, default=DEFAULT_WINDOW_SECONDS,
                        help=f"Seconds around each red-team event to keep (default {DEFAULT_WINDOW_SECONDS})")
    parser.add_argument("--max-events", type=int, default=0,
                        help="Max events per source file (0 = unlimited)")
    parser.add_argument("--include-flows", action="store_true",
                        help="Also extract flows.txt.gz → detection_network.jsonl")
    parser.add_argument("--include-proc",  action="store_true",
                        help="Also extract proc.txt.gz → forensics_provenance.jsonl")
    args = parser.parse_args()

    raw_dir = args.raw_dir
    redteam_path = os.path.join(raw_dir, "redteam.txt.gz")
    auth_path    = os.path.join(raw_dir, "auth.txt.gz")
    proc_path    = os.path.join(raw_dir, "proc.txt.gz")
    flows_path   = os.path.join(raw_dir, "flows.txt.gz")

    for path in [redteam_path, auth_path]:
        if not os.path.exists(path):
            print(f"[ERROR] Required file not found: {path}", file=sys.stderr)
            sys.exit(1)

    print(f"[INFO] Loading red-team labels from {redteam_path}")
    redteam_labels = load_redteam_labels(redteam_path)

    print(f"[INFO] Building ±{args.window}s extraction window …")
    window_set = build_window_set(redteam_labels, args.window)
    print(f"[INFO] Window covers {len(window_set):,} discrete timestamps.")

    total = 0
    with EcarWriter(args.out_dir) as writer:
        print(f"\n[INFO] Extracting auth events …")
        total += extract_auth(auth_path, window_set, redteam_labels, writer, args.max_events)

        if args.include_proc and os.path.exists(proc_path):
            print(f"\n[INFO] Extracting proc events …")
            total += extract_proc(proc_path, window_set, redteam_labels, writer, args.max_events)

        if args.include_flows and os.path.exists(flows_path):
            print(f"\n[INFO] Extracting flow events …")
            total += extract_flows(flows_path, window_set, writer, args.max_events)

        counts = writer.close()

    print(f"\n[DONE] {total:,} total events written.")
    print("Per-domain counts:")
    for domain, n in sorted(counts.items()):
        if n:
            print(f"  {domain}: {n:,}")


if __name__ == "__main__":
    main()
