#!/usr/bin/env python3
"""Phase 1 of the Radiance v4 restructure: remove legacy nodes_*.py shims.

The root-level ``nodes_*.py`` files are backward-compatible deprecation shims
that re-export classes which already live under
``radiance.nodes.<domain>.<module>``. This script removes those shims and
rewrites every remaining import that still points at them to the real module.

It deliberately does NOT touch the handful of root modules that still contain
real code (nodes_io, nodes_loader, nodes_realtime_preview, nodes_sampler,
nodes_workspace); those are migrated in a later phase.

Run from the repo root, on a branch:

    python tools/restructure_phase1.py            # dry run - show what would change
    python tools/restructure_phase1.py --apply    # perform the changes

Then run the test suite before committing:

    python -m pytest tests/
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Root modules that still contain real code - never remove in this phase.
KEEP_REAL = {
    "nodes_io", "nodes_loader", "nodes_realtime_preview",
    "nodes_sampler", "nodes_workspace", "__init__",
}


def is_shim(path: Path) -> bool:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    return "deprecat" in txt.lower() and bool(
        re.search(r"from radiance\.nodes\.[\w.]+ import", txt)
    )


def shim_target(path: Path) -> str | None:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"from (radiance\.nodes\.[\w.]+) import", txt)
    return m.group(1) if m else None


def build_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for p in sorted(ROOT.glob("nodes_*.py")):
        if p.stem in KEEP_REAL:
            continue
        if is_shim(p):
            tgt = shim_target(p)
            if tgt:
                mapping[p.stem] = tgt
    return mapping


def source_files(mapping: dict[str, str]) -> list[Path]:
    out = []
    for p in ROOT.rglob("*.py"):
        parts = set(p.parts)
        if ".git" in parts or "__pycache__" in parts:
            continue
        if p.stem in mapping:        # skip the shims themselves (being removed)
            continue
        out.append(p)
    return out


def rewrite_refs(mapping: dict[str, str], apply: bool) -> int:
    changed = 0
    for p in source_files(mapping):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        orig = txt
        for shim, tgt in mapping.items():
            # from radiance.nodes_x import ...   /   from nodes_x import ...
            txt = re.sub(rf"from (?:radiance\.)?{shim}\s+import", f"from {tgt} import", txt)
            # bare:  import radiance.nodes_x  /  import nodes_x  ->  import <tgt> as nodes_x
            txt = re.sub(rf"^(\s*)import (?:radiance\.)?{shim}\b.*$",
                         rf"\1import {tgt} as {shim}", txt, flags=re.M)
        if txt != orig:
            changed += 1
            print(f"  rewrite  {p.relative_to(ROOT)}")
            if apply:
                p.write_text(txt, encoding="utf-8")
    return changed


def remove_shims(mapping: dict[str, str], apply: bool) -> None:
    for shim, tgt in mapping.items():
        path = ROOT / f"{shim}.py"
        print(f"  remove   {path.name:<28} (shim for {tgt})")
        if apply:
            try:
                subprocess.run(["git", "rm", "-q", str(path)], cwd=ROOT, check=True)
            except Exception:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Remove legacy nodes_*.py shims (Radiance v4 phase 1).")
    ap.add_argument("--apply", action="store_true", help="perform changes (default: dry run)")
    args = ap.parse_args()

    mapping = build_map()
    if not mapping:
        print("No shims found - nothing to do.")
        return 0

    print(f"Found {len(mapping)} legacy shim(s).\n")
    print("Rewriting references that still import the shims:")
    n = rewrite_refs(mapping, args.apply) or 0
    if n == 0:
        print("  (none)")
    print("\nRemoving shim files:")
    remove_shims(mapping, args.apply)

    mode = "APPLIED" if args.apply else "DRY RUN - no files changed"
    print(f"\n{mode}: {len(mapping)} shims removed, {n} file(s) rewritten.")
    if not args.apply:
        print("Re-run with --apply on a branch, then: python -m pytest tests/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
