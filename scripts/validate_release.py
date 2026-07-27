#!/usr/bin/env python3
"""Validate the data release embedded in the code-and-data repository."""

from pathlib import Path
import hashlib
import json

import pyarrow.parquet as pq


root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "metadata/release_manifest.json").read_text(encoding="utf-8"))
errors = []
config_rows = {}

for record in manifest["files"]:
    relative_path = "DATASET_CARD.md" if record["path"] == "README.md" else record["path"]
    path = root / relative_path
    if not path.is_file():
        errors.append(f"missing: {relative_path}")
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["sha256"]:
        errors.append(f"sha256: {relative_path}")
    if path.suffix == ".parquet":
        rows = pq.ParquetFile(path).metadata.num_rows
        if rows != record["rows"]:
            errors.append(f"rows: {relative_path} ({rows} != {record['rows']})")
        config = record["path"].split("/", 2)[1]
        config_rows[config] = config_rows.get(config, 0) + rows

for config, expected in manifest["config_rows"].items():
    actual = config_rows.get(config, 0)
    if actual != expected:
        errors.append(f"config rows: {config} ({actual} != {expected})")

if errors:
    raise SystemExit("Release validation failed:\n" + "\n".join(errors))

print(f"Validated {len(manifest['files'])} release files in the combined repository.")
