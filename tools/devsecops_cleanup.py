#!/usr/bin/env python3
"""
Radiance DevSecOps Cleanup Utility
==================================

Purpose:
  Identification and removal of build artifacts, test caches, and OS junk files.
  Features a "Safety First" design with mandatory Dry-Run mode and sensitive file quarantine.

Usage:
  python tools/devsecops_cleanup.py [--force] [--dry-run]

Dependencies:
  Standard Library only (os, sys, shutil, pathlib, logging, argparse)
"""

import os
import sys
import shutil
import logging
import argparse
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Files/Dirs considered "Safe to Delete" (Waste)
TARGETS = {
    "dirs": {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        "build",
        "dist",
        "htmlcov",
        ".ipynb_checkpoints",
        "radiance.egg-info",
    },
    "files": {
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".DS_Store",
        "Thumbs.db",
        ".coverage",
        ".coverage.*",
        "nosetests.xml",
        "coverage.xml",
        "*.log",
        "*.spec",
    },
}

# Sensitive Patterns - NEVER DELETE, BUT WARN IF FOUND
SENSITIVE_PATTERNS = {
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "secrets.json",
    "credentials.json",
    "token.json",
    "id_rsa",
    "*.crt",
}

# Directories to exclude from scanning (e.g. valid data dirs)
EXCLUDE_DIRS = {".git", ".venv", "venv", "env", ".idea", ".vscode"}

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RadianceCleanup")


def should_delete(path: Path) -> bool:
    """Check if a path matches target criteria."""
    if path.is_dir():
        return path.name in TARGETS["dirs"]
    elif path.is_file():
        return any(path.match(p) for p in TARGETS["files"])
    return False


def is_sensitive(path: Path) -> bool:
    """Check if a path matches sensitive patterns."""
    return any(path.match(p) for p in SENSITIVE_PATTERNS)


def scan_and_clean(root: Path, dry_run: bool = True):
    """Recursively scan directory and apply cleanup logic."""
    deleted_count = 0
    sensitive_count = 0
    bytes_freed = 0

    logger.info(f"Starting scan in: {root}")
    if dry_run:
        logger.info("MODE: DRY-RUN (No files will be deleted)")
    else:
        logger.warning("MODE: DESTRUCTIVE (Files WILL be deleted)")

    # Top-down walk to handle directories correctly
    for root_dir, dirs, files in os.walk(root):
        # Skip excluded dirs
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        current_path = Path(root_dir)

        # 1. Process Files
        for file in files:
            file_path = current_path / file

            # Security Check First
            if is_sensitive(file_path):
                logger.warning(
                    f"SENSITIVE FILE DETECTED: {file_path} (Quarantined/Skipped)"
                )
                sensitive_count += 1
                continue

            if should_delete(file_path):
                size = file_path.stat().st_size
                if dry_run:
                    logger.info(
                        f"[DRY-RUN] Would delete file: {file_path} ({size} bytes)"
                    )
                else:
                    try:
                        file_path.unlink()
                        logger.info(f"Deleted file: {file_path}")
                        bytes_freed += size
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete {file_path}: {e}")

        # 2. Process Directories (Check if matches target dirs)
        # Note: os.walk yields dirs. If we see a target dir, we can delete it entirely.
        # But we must modify 'dirs' in-place to stop os.walk from entering it.

        # Identify dirs to delete
        to_remove = []
        for d in dirs:
            dir_path = current_path / d
            if dir_path.name in TARGETS["dirs"]:
                to_remove.append(d)

        for d in to_remove:
            dir_to_delete = current_path / d
            # Calculate size for reporting
            dir_size = sum(
                f.stat().st_size for f in dir_to_delete.rglob("*") if f.is_file()
            )

            if dry_run:
                logger.info(
                    f"[DRY-RUN] Would remove directory: {dir_to_delete} (~{dir_size} bytes)"
                )
            else:
                try:
                    shutil.rmtree(dir_to_delete)
                    logger.info(f"Removed directory: {dir_to_delete}")
                    bytes_freed += dir_size
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove {dir_to_delete}: {e}")

            # Remove from recursion list
            dirs.remove(d)

    logger.info("-" * 40)
    logger.info("CLEANUP COMPLETE")
    mode_str = "DRY-RUN" if dry_run else "DESTRUCTIVE"
    logger.info(f"Mode: {mode_str}")
    logger.info(f"Items processed: {deleted_count}")
    logger.info(f"Space reclaimed: {bytes_freed / 1024:.2f} KB")
    if sensitive_count > 0:
        logger.warning(
            f"SENSITIVE FILES FOUND: {sensitive_count} (Please review manually)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Radiance DevSecOps Cleanup Tool")
    parser.add_argument(
        "--force", action="store_true", help="Execute deletion (Disable Dry-Run)"
    )
    parser.add_argument("--path", type=str, default=".", help="Root directory to scan")

    args = parser.parse_args()

    root_path = Path(args.path).resolve()

    if not root_path.exists():
        logger.error(f"Path does not exist: {root_path}")
        sys.exit(1)

    # By default, dry_run is True. Only False if --force is passed.
    dry_run = not args.force

    scan_and_clean(root_path, dry_run=dry_run)
