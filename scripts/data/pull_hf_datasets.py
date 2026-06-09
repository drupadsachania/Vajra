#!/usr/bin/env python3
"""Pull the HuggingFace datasets listed in scripts/data/manifest.json into data/raw/.

Usage:
    python scripts/data/pull_hf_datasets.py                  # pull everything
    python scripts/data/pull_hf_datasets.py --only s0u9ata/security-kg
    python scripts/data/pull_hf_datasets.py --dry-run        # show what would be pulled

Downloads go to data/raw/<org>__<name>/ (gitignored — raw pulls never enter git).
A PULL_RECEIPT.json is written per dataset with the commit SHA, file list, and
sizes so the exact corpus snapshot is reproducible.

Requires: pip install huggingface_hub
Network: needs access to huggingface.co (add to your environment allowlist if sandboxed).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"
RAW_DIR = REPO_ROOT / "data" / "raw"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def staging_dir_for(repo_id: str) -> Path:
    return RAW_DIR / repo_id.replace("/", "__")


def pull_dataset(entry: dict, dry_run: bool = False) -> None:
    from huggingface_hub import HfApi, snapshot_download

    repo_id = entry["repo_id"]
    dest = staging_dir_for(repo_id)
    print(f"[{entry['priority']}] {repo_id} -> {dest}")
    if entry.get("notes"):
        print(f"    note: {entry['notes']}")
    if dry_run:
        return

    api = HfApi()
    info = api.dataset_info(repo_id)

    # configs == "all" pulls the full snapshot; a list pulls matching subtrees only.
    allow_patterns = None
    configs = entry.get("configs", "all")
    if isinstance(configs, list):
        allow_patterns = [f"{c}/**" for c in configs] + [f"{c}*", "*.md", "*.json"]

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=dest,
        allow_patterns=allow_patterns,
    )

    receipt = {
        "repo_id": repo_id,
        "commit_sha": info.sha,
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "configs": configs,
        "license": (info.card_data or {}).get("license") if info.card_data else None,
        "files": sorted(
            str(p.relative_to(dest))
            for p in dest.rglob("*")
            if p.is_file() and p.name != "PULL_RECEIPT.json"
        ),
    }
    (dest / "PULL_RECEIPT.json").write_text(json.dumps(receipt, indent=2))
    print(f"    done: {len(receipt['files'])} files @ {info.sha[:12]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="pull a single repo_id from the manifest")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest()
    entries = manifest["huggingface"]
    if args.only:
        entries = [e for e in entries if e["repo_id"] == args.only]
        if not entries:
            print(f"error: {args.only} not in manifest", file=sys.stderr)
            return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for entry in sorted(entries, key=lambda e: e["priority"]):
        try:
            pull_dataset(entry, dry_run=args.dry_run)
        except Exception as exc:  # keep pulling the rest; report at the end
            failures.append((entry["repo_id"], exc))
            print(f"    FAILED: {exc}", file=sys.stderr)

    if not args.dry_run:
        for ext in manifest.get("external_form_request", []):
            print(f"[{ext['priority']}] {ext['name']}: external — {ext['access']} "
                  f"(lead time {ext['lead_time']})")

    if failures:
        print(f"\n{len(failures)} dataset(s) failed:", file=sys.stderr)
        for repo_id, exc in failures:
            print(f"  {repo_id}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
