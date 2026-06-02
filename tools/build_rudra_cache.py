#!/usr/bin/env python
"""Build scene-linear tensor caches from a RUDRA manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rudra.dataset import build_cache, read_manifest, write_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cached_manifest", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    samples = read_manifest(args.manifest)
    cached = build_cache(samples, args.out_dir, overwrite=args.overwrite)
    cached_manifest = args.cached_manifest or str(Path(args.out_dir) / "manifest_cached.csv")
    write_manifest(cached, cached_manifest)
    print(f"wrote cache for {len(cached)} samples")
    print(f"cached manifest: {cached_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
