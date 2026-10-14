"""T1: download the Every Cure drug and disease lists (small Parquet files)."""

import hashlib
import json
from datetime import UTC, datetime

from huggingface_hub import HfApi, hf_hub_download

from stage1.paths import RAW

DATASETS = {
    "drug_list": "everycure/drug-list",
    "disease_list": "everycure/disease-list",
}
FILE = "data/train-00000-of-00001.parquet"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    manifest = {}
    for name, repo in DATASETS.items():
        info = api.dataset_info(repo)
        local = hf_hub_download(repo, FILE, repo_type="dataset", local_dir=RAW / name)
        manifest[name] = {
            "repo": repo,
            "revision": info.sha,
            "last_modified": info.last_modified.isoformat(),
            "path": str(local),
            "sha256": sha256(local),
        }
        print(f"{name}: revision {info.sha[:10]}  {local}")
    manifest["fetched_at"] = datetime.now(UTC).isoformat()
    (RAW / "reference_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
