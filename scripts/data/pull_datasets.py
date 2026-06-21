import os
import json
from pathlib import Path
from huggingface_hub import snapshot_download

# Project paths
# We resolve the path relative to the script location to ensure it runs anywhere
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = SCRIPT_DIR / "manifest.json"

os.makedirs(RAW_DIR, exist_ok=True)

def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r") as f:
        return json.load(f)

def pull_datasets():
    manifest_data = load_manifest()
    hf_manifest = manifest_data.get("huggingface", [])
    
    # Sort by priority
    hf_manifest = sorted(hf_manifest, key=lambda x: x.get("priority", 99))
    
    print(f"Loaded {len(hf_manifest)} HuggingFace datasets from manifest.")
    
    for entry in hf_manifest:
        repo = entry["repo_id"]
        # Format the staging directory: e.g., s0u9ata__security-kg
        safe_name = repo.replace("/", "__")
        local_dir = RAW_DIR / safe_name
        
        print(f"Pulling Priority {entry.get('priority')}: {repo} into {local_dir}")
        
        # Determine specific allow_patterns if we aren't pulling everything
        allow_patterns = None
        configs = entry.get("configs", "all")
        if configs != "all" and isinstance(configs, list):
            # Example pattern for pulling specific subset folders and root metadata
            allow_patterns = [f"{cfg}/*" for cfg in configs] + ["*.md", "*.json"]
        
        # Execute the download
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=allow_patterns
        )
        print(f"Successfully staged {repo}.\n")

if __name__ == "__main__":
    pull_datasets()
