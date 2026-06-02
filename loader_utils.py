"""Model loading orchestration — downloading, detection, CLIP assembly, caching.

This module is the public API for loading models. It delegates data to
config/model_map.py (URLs, presets) and model/detect.py (architecture heuristics),
keeping only the loading logic that requires ComfyUI runtime imports.
"""
from __future__ import annotations

import os
import hashlib
import urllib.request
import logging

import tqdm
import folder_paths

from radiance.config.model_map import RADIANCE_MODEL_MAP, CHECKPOINT_PRESETS
from radiance.model.detect import (
    detect_model_type,
    latent_format,
    assemble_clip_paths,
    get_clip_type_enum,
    estimate_vram_usage,
    LATENT_CHANNELS,
    CLIP_SLOT_ORDER,
)
from radiance.model.cache import get_model_cache, _unet_cache, _clip_cache, _vae_cache
from comfy.model_management import get_free_memory as _get_free_vram, get_total_memory as _get_total_vram, get_torch_device

logger = logging.getLogger("radiance.loader")


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download_model(url: str, target_path: str, folder_type: str,
                    expected_sha256: str | None = None) -> bool:
    """Download a model atomically.

    Writes to a temporary ``.part`` file and only renames it into place after a
    successful download (and SHA-256 check, if a digest is pinned). This prevents
    a half-finished download from leaving a truncated checkpoint at the real model
    path, which would otherwise be silently re-used and fail to load later.
    """
    tmp_path = target_path + ".part"
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        logger.info("Radiance: Downloading model from %s...", url)

        def progress_bar(t):
            last_b = [0]
            def update_to(b=1, bsize=1, tsize=None):
                if tsize is not None:
                    t.total = tsize
                t.update((b - last_b[0]) * bsize)
                last_b[0] = b
            return update_to

        with tqdm.tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=os.path.basename(target_path)) as t:
            urllib.request.urlretrieve(url, filename=tmp_path, reporthook=progress_bar(t))

        # Integrity check (only when a digest is pinned in the model map).
        expected = (expected_sha256 or "").strip().lower()
        actual = _sha256_file(tmp_path)
        if expected:
            if actual != expected:
                logger.error(
                    "Radiance: CHECKSUM MISMATCH for %s (expected %s, got %s) — discarding download.",
                    os.path.basename(target_path), expected, actual,
                )
                os.remove(tmp_path)
                return False
            logger.info("Radiance: sha256 verified for %s", os.path.basename(target_path))
        else:
            logger.info(
                "Radiance: %s sha256=%s (pin in model_map['sha256'] for integrity checks).",
                os.path.basename(target_path), actual,
            )

        os.replace(tmp_path, target_path)   # atomic move into final location
        logger.info("Download complete: %s", target_path)
        # Refresh ComfyUI file listing
        try:
            folder_paths.folder_names_and_paths[folder_type]
            folder_paths.get_filename_list(folder_type)
        except Exception:
            folder_paths.get_filename_list(folder_type)
        return True
    except Exception as e:
        logger.error("Download failed for %s: %s", url, e)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return False


def ensure_model_exists(name: str, folder_type: str, auto_download: bool = False) -> str | None:
    if not name or name == "None":
        return None

    path = folder_paths.get_full_path(folder_type, name)
    if path and os.path.exists(path):
        return path

    if auto_download and name in RADIANCE_MODEL_MAP:
        if os.environ.get("RADIANCE_LOADER_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on"):
            logger.error(
                "Radiance: offline mode (RADIANCE_LOADER_OFFLINE=1) — '%s' not found and "
                "auto-download disabled. Place the file manually.", name,
            )
            return None
        res = RADIANCE_MODEL_MAP[name]
        if res["type"] == folder_type:
            base_dir = folder_paths.get_folder_paths(folder_type)[0]
            target_path = os.path.join(base_dir, *name.replace("\\", "/").split("/"))
            if _download_model(res["url"], target_path, folder_type, res.get("sha256")):
                return target_path

    return None


def get_available_vram() -> float:
    """Return free GPU memory in GB."""
    return _get_free_vram(get_torch_device()) / (1024 ** 3)


def get_total_vram() -> float:
    """Return total GPU memory in GB."""
    return _get_total_vram(get_torch_device()) / (1024 ** 3)


def resolve_hint(hints: list[str], available: list[str]) -> str | None:
    for hint in hints:
        h = hint.lower()
        for f in available:
            if h in f.lower():
                return f
    return None


def file_fingerprint(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{st.st_mtime:.0f}:{st.st_size}"
    except OSError:
        return "nostat"
