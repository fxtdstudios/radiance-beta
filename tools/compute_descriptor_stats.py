#!/usr/bin/env python
"""Compute descriptor normalization stats for RUDRA-Lite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rudra.dataset import RUDRACacheDataset
from rudra.stats import DescriptorStats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="rudra_descriptor_stats.json")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    ds = RUDRACacheDataset(args.manifest)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    chunks = []
    for batch in loader:
        chunks.append(batch["dr_raw"].float())
    stats = DescriptorStats.from_tensor(torch.cat(chunks, dim=0))
    stats.save(args.out)
    print(f"wrote {args.out} from {stats.count} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
