"""T2: download the MATRIX score shards (about 3.8 GB) and record the exact revision."""

import json
from datetime import UTC, datetime

from huggingface_hub import HfApi, snapshot_download

from stage1.fetch_reference import sha256
from stage1.paths import RAW

REPO = "everycure/matrix-scores"


def main() -> None:
    target = RAW / "matrix_scores"
    info = HfApi().dataset_info(REPO)
    print(f"revision {info.sha}, last modified {info.last_modified}", flush=True)
    snapshot_download(
        REPO,
        repo_type="dataset",
        revision=info.sha,
        allow_patterns=["data/scores/*.parquet", "README.md"],
        local_dir=target,
    )
    shards = sorted((target / "data" / "scores").glob("*.parquet"))
    manifest = {
        "repo": REPO,
        "revision": info.sha,
        "last_modified": info.last_modified.isoformat(),
        "fetched_at": datetime.now(UTC).isoformat(),
        "shards": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in shards},
    }
    (RAW / "matrix_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done: {len(shards)} shards", flush=True)


if __name__ == "__main__":
    main()
