#!/usr/bin/env python3
"""Prepare the Radiance repository for a clean release.

SAFE cleanup only: regenerable caches, confirmed dead/stray files, root
documentation organization, and untracking runtime data. It deliberately does
NOT move or rewrite any code modules, shims, or utilities — those require a
separate, incremental, test-after migration (see docs/dev/RESTRUCTURE_PLAN.md).

Run from the repo root, on a branch:

    python tools/clean_release.py            # dry run - show what would change
    python tools/clean_release.py --apply    # perform the changes

Then verify and commit:

    python -m pytest tests/
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROTECTED = {".git", "workflows", "gizmos", "input", "output", "models", "venv", ".venv", "env"}

JUNK_DIRS = ["__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
             ".hypothesis", "build", "dist", "htmlcov", "artifacts", ".ipynb_checkpoints"]
JUNK_GLOBS = ["**/*.pyc", "**/*.pyo", "**/.DS_Store", "**/Thumbs.db", "**/*.swp",
              "**/*.swo", ".coverage", ".coverage.*", "coverage.xml", "*.egg-info"]

# Confirmed dead / stray files (verified unreferenced).
DEAD_FILES = ["splash_screen.html", "tools/restructure_phase1.py"]

# Root developer docs -> docs/dev/ (no code impact).
DOC_MOVES = ["CLEANUP_REPORT.md", "CODE_STYLE.md", "PRE_RELEASE_REVIEW.md", "NODES.md",
             "RADIANCE_v3.1_RELEASE_NOTES.md", "RELEASE_NOTES_v3.1.0.md",
             "radiance_hdr_vae_decode_report.md"]

# Tracked runtime data to stop tracking (kept on disk).
UNTRACK = ["core/radiance_history.db"]


def _protected(p: Path) -> bool:
    rel = p.relative_to(ROOT)
    return bool(rel.parts) and rel.parts[0] in PROTECTED


def _git(args):
    try:
        subprocess.run(["git", *args], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def clean_junk(apply):
    targets = set()
    for name in JUNK_DIRS:
        targets.update(d for d in ROOT.rglob(name) if d.is_dir() and not _protected(d))
    for pat in JUNK_GLOBS:
        targets.update(p for p in ROOT.glob(pat) if p.exists() and not _protected(p))
    for p in sorted(targets, key=lambda x: len(x.parts), reverse=True):
        print(f"  junk     {p.relative_to(ROOT)}")
        if apply:
            try:
                shutil.rmtree(p) if p.is_dir() else p.unlink()
            except OSError as e:
                print(f"           (skip: {e})")
    return len(targets)


def remove_dead(apply):
    n = 0
    for rel in DEAD_FILES:
        p = ROOT / rel
        if p.exists():
            print(f"  remove   {rel}")
            n += 1
            if apply and not _git(["rm", "-q", "--", rel]):
                try:
                    p.unlink()
                except OSError:
                    pass
    return n


def move_docs(apply):
    dest = ROOT / "docs" / "dev"
    n = 0
    for rel in DOC_MOVES:
        src = ROOT / rel
        if src.exists():
            print(f"  move     {rel}  ->  docs/dev/{src.name}")
            n += 1
            if apply:
                dest.mkdir(parents=True, exist_ok=True)
                if not _git(["mv", "--", rel, f"docs/dev/{src.name}"]):
                    try:
                        shutil.move(str(src), str(dest / src.name))
                    except OSError as e:
                        print(f"           (skip move: {e})")
    return n


def untrack(apply):
    n = 0
    for rel in UNTRACK:
        if (ROOT / rel).exists():
            print(f"  untrack  {rel}  (stays on disk, removed from git)")
            n += 1
            if apply:
                _git(["rm", "-q", "--cached", "--", rel])
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean the Radiance repo for release (safe).")
    ap.add_argument("--apply", action="store_true", help="perform changes (default: dry run)")
    args = ap.parse_args()

    print("1) Regenerable caches / junk:")
    j = clean_junk(args.apply) or print("   (none)")
    print("\n2) Dead / stray files:")
    d = remove_dead(args.apply) or print("   (none)")
    print("\n3) Root dev-docs -> docs/dev/:")
    m = move_docs(args.apply) or print("   (none)")
    print("\n4) Untrack runtime data:")
    u = untrack(args.apply) or print("   (none)")

    mode = "APPLIED" if args.apply else "DRY RUN - no changes written"
    print(f"\n{mode}.")
    if not args.apply:
        print("Re-run with --apply on a branch, then: python -m pytest tests/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
