"""Lightweight release hygiene checks for Radiance.

This script uses only the Python standard library so it can run before the full
ComfyUI/test environment is installed.
"""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATED_PATTERNS = (
    "core/*.db",
    "workflows/.versions/**",
    "workflows/GENERAL/dashboard_*.rad",
    "workflows/GENERAL/test_*_v*.rad",
    "**/__pycache__",
    "**/*.pyc",
)


def main() -> int:
    errors: list[str] = []
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject.get("project", {})
    comfy = pyproject.get("tool", {}).get("comfy", {})

    _check_project_metadata(project, comfy, errors)
    _check_readme(errors)
    _check_package_paths(pyproject, errors)
    _check_generated_files(errors)

    if errors:
        print("Release checks failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("Release checks passed.")
    return 0


def _check_project_metadata(project: dict, comfy: dict, errors: list[str]) -> None:
    name = str(project.get("name", ""))
    version = str(project.get("version", ""))

    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]{0,98}", name):
        errors.append(f"invalid project.name: {name!r}")
    if any(token in name for token in ("..", "--", "__")):
        errors.append(f"project.name has consecutive special characters: {name!r}")
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.]+)?", version):
        errors.append(f"version is not semantic: {version!r}")

    readme = project.get("readme")
    if not readme or not (ROOT / str(readme)).is_file():
        errors.append("project.readme is missing or does not exist")

    license_file = project.get("license", {}).get("file")
    if not license_file or not (ROOT / str(license_file)).is_file():
        errors.append("project.license.file is missing or does not exist")

    for key in ("PublisherId", "DisplayName"):
        if not comfy.get(key):
            errors.append(f"[tool.comfy].{key} is required")


def _check_readme(errors: list[str]) -> None:
    readme_path = ROOT / "README.md"
    if not readme_path.is_file():
        errors.append("README.md is missing")
        return

    readme = readme_path.read_text(encoding="utf-8")
    banned_claims = ("150 nodes", "121 Professional Nodes", "zero risk")
    for claim in banned_claims:
        if claim.lower() in readme.lower():
            errors.append(f"README contains stale or unsafe claim: {claim!r}")

    required_sections = ("## Install", "## Node Map", "## DCC Handoff", "## Release Status")
    for section in required_sections:
        if section not in readme:
            errors.append(f"README missing section: {section}")


def _check_package_paths(pyproject: dict, errors: list[str]) -> None:
    packages = pyproject.get("tool", {}).get("setuptools", {}).get("packages", [])
    for package_name in packages:
        rel = str(package_name).removeprefix("radiance").replace(".", "/").strip("/")
        package_path = ROOT / rel if rel else ROOT
        if not package_path.exists():
            errors.append(f"package path missing for {package_name}: {package_path}")


def _check_generated_files(errors: list[str]) -> None:
    for pattern in GENERATED_PATTERNS:
        matches = [path for path in ROOT.glob(pattern) if path.exists()]
        if matches:
            preview = ", ".join(str(path.relative_to(ROOT)) for path in matches[:5])
            errors.append(f"generated files present for {pattern}: {preview}")


if __name__ == "__main__":
    raise SystemExit(main())

