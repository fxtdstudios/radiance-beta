#!/usr/bin/env python
"""Validate a RUDRA dataset manifest before training."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rudra.dataset import read_manifest, validate_manifest
from rudra.sampler import bucket_from_stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", default="rudra_dataset_report")
    ap.add_argument("--max_items", type=int, default=None)
    args = ap.parse_args()

    samples = read_manifest(args.manifest)
    issues, stats = validate_manifest(samples, args.max_items)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    buckets = {}
    for s in stats:
        buckets[bucket_from_stats(s)] = buckets.get(bucket_from_stats(s), 0) + 1

    report = {
        "num_samples": len(samples),
        "num_checked": args.max_items or len(samples),
        "num_issues": len(issues),
        "num_valid_stats": len(stats),
        "bucket_counts": buckets,
        "issues": [asdict(i) for i in issues],
        "sample_stats": [asdict(s) for s in stats],
    }
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "bad_files.txt").write_text("\n".join(f"{i.severity}: {i.input_path} -> {i.target_path}: {i.message}" for i in issues), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["num_samples", "num_issues", "bucket_counts"]}, indent=2))
    return 1 if any(i.severity == "error" for i in issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())
