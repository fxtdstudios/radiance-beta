"""
◎ Radiance VFX Multipass Extractor  v3.0
════════════════════════════════════════════════════════════════════════════════

Extracts industry-standard VFX compositing passes from a decoded float32 image.
Designed to sit immediately after ◎ Radiance VAE Decode and produce named
passes ready for Nuke, DaVinci Resolve, After Effects, or any EXR pipeline.

SIGNAL FLOW:
  IMAGE (float32 linear) ─► VFX Multipass v3.0 ─► beauty
                                                  ├─► diffuse          (guided-filter edge-pres.)
                                                  ├─► specular         (freq-sep high pass)
                                                  ├─► shadow_mask      (partitioned tri-tone)
                                                  ├─► highlight_mask   (partitioned tri-tone)
                                                  ├─► midtone_mask     (partitioned tri-tone)
                                                  ├─► depth            (external or black)
                                                  ├─► ao               (SSAO: depth + normals)
                                                  ├─► edge             (Scharr multi-scale)
                                                  ├─► colorfulness     (RMS saturation energy)
                                                  ├─► reflection_mask  (bright+achromatic)
                                                  ├─► normal_map       (Scharr gradient / DSINE)
                                                  ├─► curvature        (∇·N, mean curvature)
                                                  ├─► world_position   (depth + FOV unproject)
                                                  ├─► albedo           (Retinex IID, no shading)
                                                  ├─► emission         (Z-score local glow)
                                                  ├─► roughness        (multi-scale spec sharpness)
                                                  ├─► transmission     (chroma dispersion + Fresnel)
                                                  ├─► motion_vector    (Lucas-Kanade optical flow)
                                                  ├─► object_id_matte  (k-means crypto-style ID)
                                                  ├─► pass_info        (STRING — human report)
                                                  └─► pass_confidence  (STRING — JSON quality dict)

PASS SUMMARY v3.0:
  ┌───────────────────┬────────────────────────────────────────────────────────────┐
  │ Pass              │ Method                                                     │
  ├───────────────────┼────────────────────────────────────────────────────────────┤
  │ beauty            │ Pass-through (full float32 image)                          │
  │ diffuse           │ Guided filter (edge-preserving low-pass)            v2.0   │
  │ specular          │ beauty − diffuse (high-freq residual)                      │
  │ shadow_mask       │ Smoothstep matte: luma below shadow_threshold              │
  │ highlight_mask    │ Smoothstep matte: luma above highlight_threshold           │
  │ midtone_mask      │ Partitioned remainder (shadow+mid+hi = 1 everywhere)       │
  │ depth             │ External depth_map input or zero if not connected          │
  │ ao                │ Multi-sample SSAO: hemisphere sampling, normal-weighted    │
  │ edge              │ Scharr multi-scale edge magnitude                  v2.0   │
  │ colorfulness      │ RMS saturation energy — NOT YCbCr chroma                  │
  │ reflection_mask   │ Bright + achromatic heuristic (metallic / mirror)          │
  │ normal_map  v2.0  │ Scharr gradient normal (OpenGL/DirectX), DSINE hook opt.  │
  │ curvature   v2.0  │ Mean curvature ∇·N — convex bright, concave dark           │
  │ world_pos   v2.0  │ View-space XYZ from depth + camera FOV                    │
  │ albedo      v2.1  │ Retinex IID — log-space guided filter shading removal      │
  │ emission    v2.1  │ Z-score local brightness excess, colorfulness-weighted     │
  │ roughness   v2.1  │ Multi-scale specular sharpness ratio (inverted)            │
  │ transmission v2.1 │ Chromatic dispersion + Fresnel halo detection              │
  │ motion_vec  v3.0  │ Lucas-Kanade optical flow (HSV, prev_frame optional)       │
  │ object_id   v3.0  │ K-means color+spatial clustering → RGBA ID matte          │
  │ pass_info         │ Human-readable stats report (STRING)                       │
  │ pass_conf   v3.0  │ Per-pass quality confidence dict (STRING/JSON)             │
  └───────────────────┴────────────────────────────────────────────────────────────┘

MOTION VECTOR (v3.0):
  Dense Lucas-Kanade optical flow — no OpenCV dependency, pure PyTorch.
  Connect prev_frame for true inter-frame flow. Without prev_frame the output
  is zero (static placeholder suitable for single-image workflows).
  Visualization: HSV encoding — hue=direction, saturation=magnitude, value=1.
  EXR raw channels: MV.X (horizontal px offset), MV.Y (vertical px offset).

OBJECT ID MATTE (v3.0):
  Cryptomatte-style per-object ID matte via k-means clustering on
  (R, G, B, luma, x_norm, y_norm) feature vectors.
  Each cluster receives a deterministic, visually-distinct RGBA color seeded
  by the golden-ratio hue spiral (maximally distinct hues at any K).
  Computation on max-192×192 downsampled image — keeps memory flat.
  EXR: ID.R/G/B/A for Nuke Cryptomatte or manual matte extraction.

PASS CONFIDENCE (v3.0):
  JSON dictionary mapping pass name → float [0..1] quality estimate.
  Scores: depth (0/1 binary), ao (variance), normal (well-defined ratio),
          albedo (material colour spread), emission (peak outlier),
          roughness (dynamic range), transmission (chroma shift),
          edge (structural density), specular (contrast std-dev),
          motion (mean magnitude relative to frame diagonal).

DSINE AUTO-DISCOVER:
  Set dsine_model_path = "auto" to search ComfyUI model folders:
    models/normal_estimation/  (preferred)
    models/checkpoints/
  Falls back to Scharr gradient normals if no checkpoint is found.

EXR EXPORT v3.0:
  Nuke/Resolve-compatible channel names:
    beauty.RGBA  diffuse.RGB  specular.RGB  N.X/Y/Z  P.X/Y/Z
    Z.R  AO.R  edge.R  albedo.RGB  emission.R  roughness.R  transmission.R
    colorfulness.R  reflection.R  curvature.R  shadow.R  highlight.R  midtone.R
    MV.X  MV.Y  MV_vis.RGB  ID.R  ID.G  ID.B  ID.A

VERSION HISTORY:
  1.0 — Initial (Gaussian diffuse, depth concavity AO, Sobel edge)
  2.0 — Guided filter diffuse, SSAO, Scharr edges, normal map, curvature, world pos
  2.1 — Albedo (Retinex IID), Emission (Z-score glow), Roughness (spec sharpness),
        Transmission (chromatic dispersion + Fresnel)
  3.0 — Motion vector (Lucas-Kanade), Object ID matte (k-means crypto-style),
        Pass confidence scores (JSON), DSINE auto-discover
"""

import os
import json
import math
import logging
import urllib.request
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn.functional as F
import numpy as np

from ....core.system.path_utils import strip_path_quotes

logger = logging.getLogger("radiance.vfx_multipass")


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_LUMA_WEIGHTS: Dict[str, Tuple[float, float, float]] = {
    "Rec.709 / sRGB": (0.2126, 0.7152, 0.0722),
    "ACEScg / AP1":   (0.2722, 0.6741, 0.0537),
    "Rec.2020":        (0.2627, 0.6780, 0.0593),
}

_NORMAL_CONVENTIONS = ["OpenGL (Y-Up)", "DirectX (Y-Down)"]

_AUTO_DEPTH_CHOICES = [
    "disabled",
    "Depth Anything V2 — Small  (99 MB, fast)",
    "Depth Anything V2 — Base   (390 MB, balanced)",
    "Depth Anything V2 — Large  (1.3 GB, highest quality)",
]


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL AUTO-DOWNLOAD REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

# Canonical HuggingFace URLs — weights only, no large dependencies required.
_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # DSINE pretrained normal estimation (Bae et al., 2024)
    # NOTE: baegwangbin/DSINE on HuggingFace is gated (401).  Preferred path is
    # torch.hub.load("hugoycj/DSINE-hub", "DSINE") which needs no auth.
    # Manual download: https://drive.google.com/drive/folders/1t3LMJIIrSnCGwOEf53Cyg0lkSXd3M4Hm
    "dsine": {
        "hf_repo":  "baegwangbin/DSINE",
        "hf_file":  "dsine.pt",
        "url":      "https://huggingface.co/baegwangbin/DSINE/resolve/main/dsine.pt",
        "filename": "dsine.pt",
        "subdir":   "normal_estimation",
        "size_mb":  280,
        "note":     "DSINE — monocular surface normal estimation (HF repo is gated; torch.hub preferred)",
    },
    # Depth Anything V2 (Yang et al., 2024) — three sizes
    "depth_anything_v2_small": {
        "hf_repo":  "depth-anything/Depth-Anything-V2-Small",
        "hf_file":  "depth_anything_v2_vits.pth",
        "url":      "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
        "filename": "depth_anything_v2_vits.pth",
        "subdir":   "depth_estimation",
        "size_mb":  99,
        "note":     "Depth Anything V2 Small (ViT-S) — fast monocular depth",
    },
    "depth_anything_v2_base": {
        "hf_repo":  "depth-anything/Depth-Anything-V2-Base",
        "hf_file":  "depth_anything_v2_vitb.pth",
        "url":      "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
        "filename": "depth_anything_v2_vitb.pth",
        "subdir":   "depth_estimation",
        "size_mb":  390,
        "note":     "Depth Anything V2 Base (ViT-B) — balanced accuracy / speed",
    },
    "depth_anything_v2_large": {
        "hf_repo":  "depth-anything/Depth-Anything-V2-Large",
        "hf_file":  "depth_anything_v2_vitl.pth",
        "url":      "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth",
        "filename": "depth_anything_v2_vitl.pth",
        "subdir":   "depth_estimation",
        "size_mb":  1340,
        "note":     "Depth Anything V2 Large (ViT-L) — highest accuracy",
    },
}

# Map UI choice strings → registry keys
_DA_CHOICE_TO_KEY = {
    "Depth Anything V2 — Small  (99 MB, fast)":           "depth_anything_v2_small",
    "Depth Anything V2 — Base   (390 MB, balanced)":      "depth_anything_v2_base",
    "Depth Anything V2 — Large  (1.3 GB, highest quality)": "depth_anything_v2_large",
}

# Map registry key → HuggingFace transformers model ID (pipeline approach)
_DA_KEY_TO_HF_PIPELINE = {
    "depth_anything_v2_small": "depth-anything/Depth-Anything-V2-Small-hf",
    "depth_anything_v2_base":  "depth-anything/Depth-Anything-V2-Base-hf",
    "depth_anything_v2_large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def _get_comfy_models_dir(subdir: str) -> str:
    """
    Return the ComfyUI models/<subdir>/ directory, creating it if needed.
    Falls back to ~/.cache/radiance/models/<subdir>/ outside ComfyUI.
    """
    try:
        import folder_paths  # type: ignore
        base = getattr(folder_paths, "models_dir", None)
        if base:
            path = os.path.join(base, subdir)
            os.makedirs(path, exist_ok=True)
            return path
    except Exception as exc:
        logger.warning("[nodes_vfx_multipass] _get_comfy_models_dir: %s", exc)
    fallback = os.path.join(os.path.expanduser("~"), ".cache", "radiance", "models", subdir)
    os.makedirs(fallback, exist_ok=True)
    return fallback


def _verify_or_report_sha256(dest: str, info: dict, key: str) -> bool:
    """Verify a downloaded file against a pinned SHA-256, or log the digest.

    Returns True when trusted (hash matches or none pinned). On mismatch the
    file is removed and False is returned.
    """
    import hashlib
    expected = (info.get("sha256") or "").strip().lower()
    try:
        h = hashlib.sha256()
        with open(dest, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        actual = h.hexdigest()
    except OSError:
        return True
    if not expected:
        logger.info(f"[Radiance] {key}: sha256={actual} (pin in registry['sha256'] for integrity checks).")
        return True
    if actual != expected:
        logger.error(f"[Radiance] CHECKSUM MISMATCH for '{key}' (expected {expected}, got {actual}) — removing.")
        try:
            os.remove(dest)
        except OSError:
            pass
        return False
    logger.info(f"[Radiance] ✓ sha256 verified for '{key}'")
    return True


def _download_model(key: str, force: bool = False) -> Optional[str]:
    """
    Download a registered model to the ComfyUI models directory.

    Download order:
      1. huggingface_hub.hf_hub_download  (handles auth, resume, caching)
      2. urllib.request.urlretrieve         (no extra dependencies)

    Returns the local file path on success, None on any failure.
    Model is skipped (path returned immediately) if already present.
    """
    if key not in _MODEL_REGISTRY:
        logger.error(f"[Radiance] Unknown model key: '{key}'")
        return None

    info     = _MODEL_REGISTRY[key]
    save_dir = _get_comfy_models_dir(info["subdir"])
    dest     = os.path.join(save_dir, info["filename"])

    if os.path.isfile(dest) and not force:
        logger.debug(f"[Radiance] Model '{key}' already present: {dest}")
        return dest

    size_mb = info["size_mb"]
    logger.info(f"[Radiance] ── Auto-downloading: {info['note']}")
    logger.info(f"[Radiance]   Size   : ~{size_mb} MB")
    logger.info(f"[Radiance]   Dest   : {dest}")
    logger.info(f"[Radiance]   Source : {info['url']}")

    # ── Method 1: huggingface_hub ──────────────────────────────────────────
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
        local = hf_hub_download(
            repo_id=info["hf_repo"],
            filename=info["hf_file"],
            local_dir=save_dir,
            local_dir_use_symlinks=False,
        )
        # Ensure the file is at the expected dest path
        if os.path.abspath(local) != os.path.abspath(dest):
            import shutil
            shutil.copy2(local, dest)
        if not _verify_or_report_sha256(dest, info, key):
            return None
        logger.info(f"[Radiance] ✓ Downloaded via huggingface_hub → {dest}")
        return dest
    except ImportError:
        logger.debug("[Radiance] huggingface_hub not available — using urllib")
    except Exception as e:
        logger.warning(f"[Radiance] huggingface_hub download failed: {e} — trying urllib")

    # ── Method 2: urllib (no extra deps) ──────────────────────────────────
    tmp = dest + ".part"
    try:
        last_pct = [-1]

        def _progress(count, block, total):
            if total > 0:
                pct = min(100, count * block * 100 // total)
                if pct // 10 != last_pct[0] // 10:
                    logger.info(f"[Radiance]   {key}: {pct}% ({count*block//(1024*1024)} / {total//(1024*1024)} MB)")
                    last_pct[0] = pct

        urllib.request.urlretrieve(info["url"], tmp, reporthook=_progress)
        os.replace(tmp, dest)
        if not _verify_or_report_sha256(dest, info, key):
            return None
        logger.info(f"[Radiance] ✓ Downloaded via urllib → {dest}")
        return dest
    except Exception as e:
        logger.error(f"[Radiance] ✗ Download failed for '{key}': {e}")
        for f in [tmp, dest]:
            try:
                if os.path.isfile(f) and os.path.getsize(f) < 1024:
                    os.remove(f)
            except Exception as exc:
                logger.warning("[nodes_vfx_multipass] _progress: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  BASE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _luminance(img: torch.Tensor, weights: Tuple[float, float, float]) -> torch.Tensor:
    """(B,H,W,C) → (B,H,W) luminance."""
    return weights[0]*img[...,0] + weights[1]*img[...,1] + weights[2]*img[...,2]


def _to_3ch_image(scalar: torch.Tensor) -> torch.Tensor:
    """(B,H,W) → (B,H,W,3) by expanding the scalar across RGB."""
    return scalar.unsqueeze(-1).expand(-1,-1,-1,3).contiguous()


def _smoothstep(t: torch.Tensor) -> torch.Tensor:
    t = torch.clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _threshold_mask(luma, threshold, width, above):
    width = max(width, 1e-4)
    t = (luma - threshold) / width if above else (threshold - luma) / width
    return _smoothstep(t)


# ─────────────────────────────────────────────────────────────────────────────
#  FILTER PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def _box_filter_bhwc(x: torch.Tensor, r: int) -> torch.Tensor:
    """
    Separable mean (box) filter, radius r, on (B,H,W,C).
    Two 1-D avg_pool passes — O(N) regardless of radius.
    """
    if r <= 0:
        return x
    B, H, W, C = x.shape
    ks = 2 * r + 1
    x4 = x.float().permute(0,3,1,2).reshape(B*C, 1, H, W)
    x4 = F.pad(x4, (r,r,0,0), mode="reflect")
    x4 = F.avg_pool2d(x4, kernel_size=(1,ks), stride=1, padding=0)
    x4 = F.pad(x4, (0,0,r,r), mode="reflect")
    x4 = F.avg_pool2d(x4, kernel_size=(ks,1), stride=1, padding=0)
    return x4.reshape(B,C,H,W).permute(0,2,3,1)


def _guided_filter_diffuse(img: torch.Tensor, radius: float, eps: float = 0.01) -> torch.Tensor:
    """
    Edge-preserving low-pass via guided filter (He et al. 2013).
    Self-guided: guide = input. Smooths within uniform regions, preserves
    material boundaries. radius = neighbourhood size; eps = edge sensitivity.
    """
    r  = max(1, int(round(radius)))
    I  = img.float()

    mean_I  = _box_filter_bhwc(I,   r)
    mean_p  = _box_filter_bhwc(I,   r)     # p == I  (self-guided)
    corr_I  = _box_filter_bhwc(I*I, r)
    cov_Ip  = _box_filter_bhwc(I*I, r)     # cov(I,I) = var(I)

    var_I  = corr_I  - mean_I * mean_I
    cov_Ip = cov_Ip  - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box_filter_bhwc(a, r)
    mean_b = _box_filter_bhwc(b, r)

    return (mean_a * I + mean_b).to(img.dtype)


def _build_gaussian_kernel(sigma, device, dtype):
    ks = max(3, int(sigma * 6) | 1)
    x  = torch.arange(ks, device=device, dtype=dtype) - ks // 2
    k  = torch.exp(-(x**2) / (2.0 * sigma**2))
    return k / k.sum(), ks


def _gaussian_blur_bhwc(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on (B,H,W,C)."""
    if sigma <= 0.0:
        return img
    B, H, W, C = img.shape
    k1d, ks = _build_gaussian_kernel(sigma, img.device, torch.float32)
    pad = ks // 2
    x   = img.float().permute(0,3,1,2).reshape(B*C, 1, H, W)
    x   = F.pad(x, (pad,pad,0,0), mode="reflect")
    x   = F.conv2d(x, k1d.view(1,1,1,ks))
    x   = F.pad(x, (0,0,pad,pad), mode="reflect")
    x   = F.conv2d(x, k1d.view(1,1,ks,1))
    return x.reshape(B,C,H,W).permute(0,2,3,1).to(img.dtype)


# ─────────────────────────────────────────────────────────────────────────────
#  SURFACE GEOMETRY (v2.0)
# ─────────────────────────────────────────────────────────────────────────────

def _scharr_gradient(luma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scharr X/Y gradients on (B,H,W). Better rotational isotropy than Sobel."""
    dev = luma.device
    kx  = torch.tensor([[-3.,0.,3.],[-10.,0.,10.],[-3.,0.,3.]],
                       dtype=torch.float32, device=dev).view(1,1,3,3) / 32.0
    ky  = torch.tensor([[-3.,-10.,-3.],[0.,0.,0.],[3.,10.,3.]],
                       dtype=torch.float32, device=dev).view(1,1,3,3) / 32.0
    x   = luma.float().unsqueeze(1)
    xp  = F.pad(x, (1,1,1,1), mode="reflect")
    gx  = F.conv2d(xp, kx, padding=0).squeeze(1)
    gy  = F.conv2d(xp, ky, padding=0).squeeze(1)
    return gx, gy


def _surface_normals_gradient(
    luma: torch.Tensor,
    strength: float = 2.0,
    convention: str = "OpenGL (Y-Up)",
) -> torch.Tensor:
    """
    Surface normals via Scharr gradient on luminance-as-height-field.
    Encoded to [0,1] RGB. Flat surface → (0.5, 0.5, 1.0) in OpenGL convention.
    """
    gx, gy = _scharr_gradient(luma)
    nx = -gx * strength
    ny = -gy * strength
    nz = torch.ones_like(nx)
    ln = torch.sqrt(nx*nx + ny*ny + nz*nz).clamp(min=1e-8)
    nx, ny, nz = nx/ln, ny/ln, nz/ln
    if convention == "DirectX (Y-Down)":
        ny = -ny
    r = nx * 0.5 + 0.5
    g = ny * 0.5 + 0.5
    b = (nz * 0.5 + 0.5).clamp(0.5, 1.0)
    return torch.stack([r, g, b], dim=-1).contiguous()


def _dsine_auto_discover() -> Optional[str]:
    """
    Locate an existing DSINE checkpoint in ComfyUI model folders.
    Search order: models/normal_estimation/ → models/checkpoints/ → models/vae/
    Returns the first matching .pt file path, or None if not found anywhere.
    """
    try:
        import folder_paths  # type: ignore
        search_dirs = []
        models_root = getattr(folder_paths, "models_dir", None)
        if models_root:
            search_dirs.append(os.path.join(models_root, "normal_estimation"))
        for key in ("checkpoints", "vae"):
            if hasattr(folder_paths, "get_folder_paths"):
                search_dirs.extend(folder_paths.get_folder_paths(key))
    except Exception:
        search_dirs = []

    # Also check the registry save_dir in case it was downloaded outside ComfyUI
    search_dirs.append(_get_comfy_models_dir("normal_estimation"))

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if "dsine" in fname.lower() and fname.endswith(".pt"):
                found = os.path.join(d, fname)
                logger.info(f"[Radiance] DSINE found: {found}")
                return found

    return None


def _dsine_ensure_model() -> Optional[str]:
    """
    Ensure DSINE checkpoint is available locally.
    1. Search existing ComfyUI model folders.
    2. If not found, auto-download from HuggingFace (~280 MB).
    Returns local path or None on failure.
    """
    path = _dsine_auto_discover()
    if path:
        return path
    logger.info("[Radiance] DSINE not found locally — auto-downloading (~280 MB) ...")
    return _download_model("dsine")


# Module-level cache for the torch.hub DSINE model.
_DSINE_HUB_CACHE: Dict[str, Any] = {}


def _try_dsine_hub(img_bhwc: torch.Tensor, convention: str) -> "Optional[torch.Tensor]":
    """
    Load DSINE via torch.hub (hugoycj/DSINE-hub on GitHub).
    Weights come from GitHub Releases — no HuggingFace authentication needed.
    Returns encoded normals (B, H, W, 3) in [0, 1], or None on any failure.
    """
    try:
        if "model" not in _DSINE_HUB_CACHE:
            logger.info(
                "[Radiance] Loading DSINE via torch.hub (hugoycj/DSINE-hub) — "
                "first run downloads ~280 MB from GitHub Releases ..."
            )
            model = torch.hub.load(
                "hugoycj/DSINE-hub", "DSINE",
                trust_repo=True, force_reload=False, verbose=False,
            )
            model.eval()
            _DSINE_HUB_CACHE["model"] = model
            logger.info("[Radiance] DSINE (torch.hub) ready.")

        model  = _DSINE_HUB_CACHE["model"]
        device = img_bhwc.device
        normals = []
        for b in range(img_bhwc.shape[0]):
            frame_f   = img_bhwc[b, :, :, :3].float().clamp(0.0, 1.0).cpu().numpy()
            frame_bgr = (frame_f[:, :, ::-1] * 255.0).astype(np.uint8).copy()  # RGB→BGR
            with torch.no_grad():
                pred = model.infer_cv2(frame_bgr)  # (1, 3, H, W) in [-1, 1]
            if isinstance(pred, (list, tuple)):
                pred = pred[-1]
            pred = pred[0]  # (3, H, W)
            nx, ny, nz = pred[0], pred[1], pred[2]
            ln = torch.sqrt(nx*nx + ny*ny + nz*nz).clamp(min=1e-8)
            nx, ny, nz = nx / ln, ny / ln, nz / ln
            if convention == "DirectX (Y-Down)":
                ny = -ny
            enc = torch.stack(
                [nx * 0.5 + 0.5, ny * 0.5 + 0.5, (nz * 0.5 + 0.5).clamp(0.0, 1.0)],
                dim=-1,
            )
            normals.append(enc.to(device))
        return torch.stack(normals, dim=0)  # (B, H, W, 3)
    except Exception as e:
        logger.debug(f"[Radiance] DSINE torch.hub failed: {e}")
        return None


def _normal_from_dsine(img_bhwc, dsine_model_path, convention):
    """
    Optional DSINE model hook. Returns (B,H,W,3) encoded normals or None on failure.

    dsine_model_path == 'auto':
      1. Try torch.hub (hugoycj/DSINE-hub) — no HuggingFace auth, GitHub Releases.
      2. Fall back to ComfyUI model folder auto-discovery.
      3. Fall back to HuggingFace download (requires auth if repo is gated).
    """
    dsine_model_path = strip_path_quotes(dsine_model_path)
    if dsine_model_path == "auto":
        # ── 1. torch.hub (preferred — no auth required) ───────────────────────
        result = _try_dsine_hub(img_bhwc, convention)
        if result is not None:
            return result
        logger.debug("[Radiance] torch.hub DSINE unavailable; trying local file.")

        # ── 2 & 3. Local discover / HuggingFace download ──────────────────────
        resolved = _dsine_ensure_model()
        if not resolved or not os.path.isfile(resolved):
            logger.warning(
                "[Radiance] DSINE not available (torch.hub failed and no local checkpoint). "
                "To download manually: "
                "https://drive.google.com/drive/folders/1t3LMJIIrSnCGwOEf53Cyg0lkSXd3M4Hm "
                "→ save dsine.pt to ComfyUI/models/normal_estimation/"
            )
            return None
    else:
        resolved = dsine_model_path
        if not resolved or not os.path.isfile(resolved):
            return None

    try:
        import sys
        dsine_dir = os.path.dirname(resolved)
        if dsine_dir not in sys.path:
            sys.path.insert(0, dsine_dir)
        from dsine.models.dsine import DSINE  # type: ignore
        device = img_bhwc.device
        if not hasattr(_normal_from_dsine, "_cache"):
            _normal_from_dsine._cache = {}
        if resolved not in _normal_from_dsine._cache:
            m = DSINE()
            st = torch.load(resolved, map_location="cpu", weights_only=True)
            m.load_state_dict(st.get("model", st), strict=False)
            m.eval()
            _normal_from_dsine._cache[resolved] = m
            logger.info(f"[Radiance] DSINE loaded: {resolved}")
        model = _normal_from_dsine._cache[resolved].to(device)
        rgb = img_bhwc[...,:3].float().permute(0,3,1,2).clamp(0,1)
        with torch.no_grad():
            pred = model(rgb)
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        nx, ny, nz = pred[:,0], pred[:,1], pred[:,2]
        ln = torch.sqrt(nx*nx+ny*ny+nz*nz).clamp(min=1e-8)
        nx, ny, nz = nx/ln, ny/ln, nz/ln
        if convention == "DirectX (Y-Down)":
            ny = -ny
        return torch.stack([nx*0.5+0.5, ny*0.5+0.5, (nz*0.5+0.5).clamp(0,1)], dim=-1)
    except Exception as e:
        logger.warning(f"[Radiance] DSINE file-based load failed ({e}) — using gradient normals")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  DEPTH ANYTHING V2 — AUTO-INFER (v3.0)
# ─────────────────────────────────────────────────────────────────────────────

_DA_PIPELINE_CACHE: Dict[str, Any] = {}   # hf_model_id → loaded pipeline


def _depth_anything_v2_infer(
    img_bhwc: torch.Tensor,
    model_key: str = "depth_anything_v2_small",
) -> Optional[torch.Tensor]:
    """
    Run Depth Anything V2 inference on a (B,H,W,C) float32 image.
    Returns (B,H,W) normalised depth in [0,1] (near=bright), or None on failure.

    Two code paths (tried in order):
      1. HuggingFace transformers pipeline  — cleanest, handles pre/post automatically.
      2. Direct checkpoint + architecture   — fallback when transformers unavailable.
         Requires the depth_anything package: pip install depth-anything-v2

    The checkpoint (.pth) is auto-downloaded on first use via _download_model().
    """
    if model_key not in _MODEL_REGISTRY:
        logger.error(f"[Radiance] Depth Anything: unknown key '{model_key}'")
        return None

    B, H, W, C = img_bhwc.shape
    device      = img_bhwc.device
    hf_pipe_id  = _DA_KEY_TO_HF_PIPELINE.get(model_key, "depth-anything/Depth-Anything-V2-Small-hf")

    # ── Path 1: transformers pipeline ──────────────────────────────────────
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore
        import PIL.Image

        if hf_pipe_id not in _DA_PIPELINE_CACHE:
            logger.info(f"[Radiance] Loading Depth Anything V2 pipeline ({model_key}) — first run may download weights …")
            pipe = hf_pipeline(
                task="depth-estimation",
                model=hf_pipe_id,
                device=0 if device.type == "cuda" else -1,
            )
            _DA_PIPELINE_CACHE[hf_pipe_id] = pipe
            logger.info(f"[Radiance] Depth Anything V2 ({model_key}) ready.")

        pipe = _DA_PIPELINE_CACHE[hf_pipe_id]
        depths = []
        for b in range(B):
            arr = (img_bhwc[b, ..., :3].float().clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
            pil = PIL.Image.fromarray(arr)
            out = pipe(pil)
            d   = np.array(out["depth"], dtype=np.float32)
            d_min, d_max = d.min(), d.max()
            d = (d - d_min) / (d_max - d_min + 1e-8)
            if d.shape[0] != H or d.shape[1] != W:
                d = F.interpolate(
                    torch.from_numpy(d).unsqueeze(0).unsqueeze(0),
                    (H, W), mode="bilinear", align_corners=False,
                ).squeeze().numpy()
            depths.append(torch.from_numpy(d))
        return torch.stack(depths, dim=0).to(device)

    except ImportError:
        logger.debug("[Radiance] transformers not installed — trying direct checkpoint path.")
    except Exception as e:
        logger.warning(f"[Radiance] Depth Anything pipeline error: {e} — trying direct checkpoint.")

    # ── Path 2: direct checkpoint + depth_anything package ────────────────
    try:
        from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

        ckpt_path = _download_model(model_key)
        if not ckpt_path:
            return None

        _enc_map = {
            "depth_anything_v2_small": "vits",
            "depth_anything_v2_base":  "vitb",
            "depth_anything_v2_large": "vitl",
        }
        encoder = _enc_map.get(model_key, "vits")

        cache_key = f"da2_{model_key}"
        if cache_key not in _DA_PIPELINE_CACHE:
            model = DepthAnythingV2(encoder=encoder, features=64, out_channels=[48,96,192,384])
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            _DA_PIPELINE_CACHE[cache_key] = model
            logger.info(f"[Radiance] Depth Anything V2 ({encoder}) loaded from {ckpt_path}")

        model = _DA_PIPELINE_CACHE[cache_key].to(device)

        # Normalise to ImageNet stats expected by ViT backbone
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
        rgb  = img_bhwc[...,:3].float().clamp(0,1).permute(0,3,1,2)
        inp  = (rgb - mean) / std

        with torch.no_grad():
            pred = model(inp)   # (B, H, W)

        # Normalise each item in batch to [0,1] (near=bright, inverted from metric)
        depths = []
        for b in range(B):
            d = pred[b]
            d = (d - d.min()) / (d.max() - d.min() + 1e-8)
            depths.append(d)
        return torch.stack(depths, dim=0)

    except ImportError:
        logger.warning(
            "[Radiance] Depth Anything V2: neither 'transformers' nor 'depth_anything_v2' package found.\n"
            "  Install with:  pip install transformers   (recommended)\n"
            "             or  pip install depth-anything-v2"
        )
    except Exception as e:
        logger.error(f"[Radiance] Depth Anything V2 direct inference failed: {e}")

    return None


def _curvature_from_normals(normal_encoded: torch.Tensor) -> torch.Tensor:
    """Mean curvature ∇·N from encoded normal map. 0=concave, 0.5=flat, 1=convex."""
    N          = normal_encoded.float() * 2.0 - 1.0
    dNx_dx, _  = _scharr_gradient(N[...,0])
    _, dNy_dy  = _scharr_gradient(N[...,1])
    kappa      = (dNx_dx + dNy_dy) * 8.0
    return _to_3ch_image((kappa * 0.5 + 0.5).clamp(0.0, 1.0))


def _world_position_from_depth(
    depth_scalar: torch.Tensor,
    fov_degrees: float = 60.0,
    depth_scale: float = 10.0,
    near_is_white: bool = True,
) -> torch.Tensor:
    """View-space XYZ position from depth map + assumed pinhole camera FOV."""
    B, H, W = depth_scalar.shape
    dev     = depth_scalar.device
    d       = depth_scalar.float()
    if near_is_white:
        d = 1.0 - d
    d = d * depth_scale

    ar  = W / max(H, 1)
    u   = torch.linspace(-1.0, 1.0, W, device=dev) * ar
    v   = torch.linspace(-1.0, 1.0, H, device=dev)
    gv, gu = torch.meshgrid(v, u, indexing="ij")
    thf = math.tan(math.radians(fov_degrees / 2.0))

    gu = gu.unsqueeze(0).expand(B,-1,-1)
    gv = gv.unsqueeze(0).expand(B,-1,-1)

    px = gu * thf * d
    py = -gv * thf * d
    pz = d
    pos = torch.stack([px, py, pz], dim=-1)

    for c in range(3):
        ch  = pos[..., c]
        mn  = ch.reshape(B,-1).min(dim=1).values.view(B,1,1)
        mx  = ch.reshape(B,-1).max(dim=1).values.view(B,1,1)
        pos[..., c] = (ch - mn) / (mx - mn).clamp(min=1e-8)

    return pos.contiguous()


# ─────────────────────────────────────────────────────────────────────────────
#  AO — SSAO (v2.0)
# ─────────────────────────────────────────────────────────────────────────────

def _ssao_multisampled(
    depth_scalar: torch.Tensor,
    normal_encoded: Optional[torch.Tensor],
    radius_px: float,
    strength: float,
    n_samples: int,
    near_is_white: bool,
) -> torch.Tensor:
    """
    Multi-sample SSAO with normal-weighted hemisphere sampling.
    Samples n_samples angles at 3 radii (0.4×, 0.7×, 1.0× radius_px).
    """
    B, H, W = depth_scalar.shape
    dev     = depth_scalar.device

    if radius_px < 1.0 or strength <= 0.0:
        return torch.zeros(B, H, W, device=dev, dtype=torch.float32)

    d = depth_scalar.float()
    if near_is_white:
        d = 1.0 - d

    if normal_encoded is not None:
        N  = normal_encoded.float() * 2.0 - 1.0
        Nx, Ny = N[...,0], N[...,1]
    else:
        Nx = Ny = None

    gy   = torch.linspace(-1.0, 1.0, H, device=dev)
    gx   = torch.linspace(-1.0, 1.0, W, device=dev)
    bv, bu = torch.meshgrid(gy, gx, indexing="ij")
    base   = torch.stack([bu, bv], dim=-1).unsqueeze(0).expand(B,-1,-1,-1)

    pu = 2.0 / W;  pv = 2.0 / H
    d_bchw = d.unsqueeze(1)

    angles    = [2.0*math.pi*i/n_samples for i in range(n_samples)]
    r_factors = [0.4, 0.7, 1.0]

    all_grids, all_dirs = [], []
    for rf in r_factors:
        r = radius_px * rf
        for ang in angles:
            ca, sa = math.cos(ang), math.sin(ang)
            sg = base + torch.tensor([ca*r*pu, sa*r*pv], device=dev, dtype=torch.float32)
            all_grids.append(sg)
            all_dirs.append((ca, sa))

    n_tot   = len(all_grids)
    gc      = torch.cat(all_grids, dim=0)
    dr      = d_bchw.repeat(n_tot, 1, 1, 1)
    ds      = F.grid_sample(dr, gc, mode="bilinear",
                            padding_mode="border", align_corners=True)
    ds      = ds.squeeze(1).view(n_tot, B, H, W)
    db      = d.unsqueeze(0).expand(n_tot,-1,-1,-1)
    occ     = (db - ds).clamp(min=0.0)

    if Nx is not None:
        wts = torch.stack(
            [(0.5 + 0.5*(Nx*ca+Ny*sa).clamp(min=0.0)) for ca,sa in all_dirs],
            dim=0
        )
        occ = occ * wts

    ao   = occ.mean(dim=0)
    flat = ao.view(B,-1)
    mx   = flat.max(dim=1).values.view(B,1,1).clamp(min=1e-8)
    return (ao / mx * strength).clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  EDGE — Scharr multi-scale (v2.0)
# ─────────────────────────────────────────────────────────────────────────────

def _scharr_edges(luma: torch.Tensor) -> torch.Tensor:
    """Multi-scale Scharr edge magnitude. Combines σ=0/1/2 at weights 0.50/0.35/0.15."""
    B, H, W = luma.shape
    dev  = luma.device
    kx   = torch.tensor([[-3.,0.,3.],[-10.,0.,10.],[-3.,0.,3.]],
                        dtype=torch.float32, device=dev).view(1,1,3,3) / 32.0
    ky   = torch.tensor([[-3.,-10.,-3.],[0.,0.,0.],[3.,10.,3.]],
                        dtype=torch.float32, device=dev).view(1,1,3,3) / 32.0
    acc  = torch.zeros(B, H, W, device=dev, dtype=torch.float32)

    for sigma, w in [(0.0, 0.50), (1.0, 0.35), (2.0, 0.15)]:
        if sigma > 0.0:
            k1d, ks = _build_gaussian_kernel(sigma, dev, torch.float32)
            pad = ks // 2
            t   = luma.float().unsqueeze(1)
            t   = F.conv2d(F.pad(t,(pad,pad,0,0),mode="reflect"), k1d.view(1,1,1,ks))
            t   = F.conv2d(F.pad(t,(0,0,pad,pad),mode="reflect"), k1d.view(1,1,ks,1))
            src = t.squeeze(1)
        else:
            src = luma.float()

        xp  = F.pad(src.unsqueeze(1),(1,1,1,1), mode="reflect")
        acc = acc + torch.sqrt(
            F.conv2d(xp,kx,padding=0).squeeze(1)**2 +
            F.conv2d(xp,ky,padding=0).squeeze(1)**2
        ) * w

    flat = acc.view(B,-1)
    mx   = flat.max(dim=1).values.view(B,1,1).clamp(min=1e-8)
    return (acc / mx).to(luma.dtype)


# ─────────────────────────────────────────────────────────────────────────────
#  MATERIAL — EXISTING PASSES
# ─────────────────────────────────────────────────────────────────────────────

def _colorfulness(img: torch.Tensor, weights: Tuple[float,float,float]) -> torch.Tensor:
    """RMS saturation energy → (B,H,W) in [0,1]."""
    luma = _luminance(img, weights).unsqueeze(-1)
    diff = img[...,:3] - luma
    sat  = torch.sqrt((diff**2).mean(dim=-1))
    B    = sat.shape[0]
    mx   = sat.view(B,-1).max(dim=1).values.view(B,1,1).clamp(min=1e-8)
    return sat / mx


def _reflection_mask(specular: torch.Tensor, colorfulness: torch.Tensor) -> torch.Tensor:
    """Bright + achromatic heuristic. Returns (B,H,W,3)."""
    refl = specular.mean(dim=-1) * (1.0 - colorfulness)
    B    = refl.shape[0]
    mx   = refl.view(B,-1).max(dim=1).values.view(B,1,1).clamp(min=1e-8)
    return _to_3ch_image((refl / mx).clamp(0.0, 1.0))


def _metallic_mask(img: torch.Tensor, specular: torch.Tensor,
                   colorfulness: torch.Tensor) -> torch.Tensor:
    """Metallic heuristic: strong specular response with low chroma reads as metal
    (metals show bright, near-achromatic highlights; dielectrics keep base colour).
    Returns (B,H,W,3) normalized to 0..1."""
    spec_luma = specular.float().mean(dim=-1)
    metal = spec_luma * (1.0 - colorfulness.clamp(0.0, 1.0))
    B = metal.shape[0]
    mx = metal.view(B, -1).max(dim=1).values.view(B, 1, 1).clamp(min=1e-8)
    return _to_3ch_image((metal / mx).clamp(0.0, 1.0))


def _highpass_filter(img: torch.Tensor, radius: float = 3.0, strength: float = 1.0,
                     contrast: float = 1.0) -> torch.Tensor:
    """High-pass detail pass: image minus a blurred copy, scaled and re-centered to
    mid-grey. Returns (B,H,W,3) in 0..1."""
    low = _gaussian_blur_bhwc(img.float(), max(0.1, float(radius)))
    high = (img.float() - low) * float(strength) * float(contrast)
    return (high + 0.5).clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  MATERIAL — v2.1 PASSES
# ─────────────────────────────────────────────────────────────────────────────

def _albedo_retinex(
    img: torch.Tensor,
    luma: torch.Tensor,
    shading_radius: float = 80.0,
    eps: float = 0.001,
) -> torch.Tensor:
    """
    Albedo via Retinex Intrinsic Image Decomposition.
    I = A × S  →  log(I) = log(A) + log(S).
    Shading S = guided-filter low-pass of log-luma (edge-preserving, large radius).
    Albedo luma = exp(log(I) − log(S)). RGB hue preserved via channel rescaling.
    """
    luma_s      = luma.float().clamp(min=eps)
    log_luma    = torch.log(luma_s)
    log_luma_4d = log_luma.unsqueeze(-1)
    log_shade   = _guided_filter_diffuse(
        log_luma_4d, radius=shading_radius, eps=0.0001
    ).squeeze(-1)
    albedo_luma = torch.exp(log_luma - log_shade)
    scale       = (albedo_luma / luma_s).unsqueeze(-1)
    albedo_rgb  = (img.float() * scale).clamp(min=0.0)
    B    = albedo_rgb.shape[0]
    flat = albedo_rgb[...,:3].reshape(B, -1)
    p995 = torch.quantile(flat, 0.995, dim=1).view(B,1,1,1).clamp(min=1e-8)
    return (albedo_rgb / p995).clamp(0.0, 1.0).to(img.dtype)


def _emission_glow(
    luma: torch.Tensor,
    colorfulness: torch.Tensor,
    radius: float = 30.0,
    boost: float = 1.5,
) -> torch.Tensor:
    """
    Emission mask via local Z-score.
    Pixels > boost σ above local mean are candidates.
    Weighted by colorfulness (coloured outliers → emission, achromatic → specular).
    """
    r          = max(1, int(round(radius)))
    luma_f     = luma.float().unsqueeze(-1)
    local_mean = _box_filter_bhwc(luma_f, r).squeeze(-1)
    local_sqm  = _box_filter_bhwc((luma.float()**2).unsqueeze(-1), r).squeeze(-1)
    local_std  = torch.sqrt((local_sqm - local_mean**2).clamp(min=0.0) + 1e-8)
    z_score    = (luma.float() - local_mean) / local_std
    emit_raw   = (z_score - boost).clamp(min=0.0)
    col_weight = 0.25 + 0.75 * colorfulness.float()
    emit_w     = emit_raw * col_weight
    B    = emit_w.shape[0]
    flat = emit_w.view(B,-1)
    mx   = flat.max(dim=1).values.view(B,1,1).clamp(min=1e-8)
    return _to_3ch_image((emit_w / mx).clamp(0.0, 1.0))


def _roughness_from_specular(
    specular: torch.Tensor,
    fine_radius: float = 2.0,
    coarse_radius: float = 15.0,
) -> torch.Tensor:
    """
    Roughness via fine/coarse specular std-dev ratio.
    High fine/coarse ratio → sharp specular → smooth surface → inverted = low roughness.
    Scale-independent: works regardless of absolute specular brightness.
    """
    spec_luma = specular.float().mean(dim=-1)
    s4   = spec_luma.unsqueeze(-1)
    s2_4 = (spec_luma**2).unsqueeze(-1)
    r_f  = max(1, int(round(fine_radius)))
    r_c  = max(1, int(round(coarse_radius)))

    def _lstd(m4, sq4, r):
        m  = _box_filter_bhwc(m4,  r).squeeze(-1)
        sq = _box_filter_bhwc(sq4, r).squeeze(-1)
        return torch.sqrt((sq - m**2).clamp(min=0.0) + 1e-10)

    std_fine   = _lstd(s4, s2_4, r_f)
    std_coarse = _lstd(s4, s2_4, r_c)
    sharpness  = std_fine / (std_coarse + 1e-8)
    B    = sharpness.shape[0]
    flat = sharpness.view(B,-1)
    mx   = flat.max(dim=1).values.view(B,1,1).clamp(min=1e-8)
    return _to_3ch_image(1.0 - (sharpness / mx).clamp(0.0, 1.0))


def _transmission_mask(
    img: torch.Tensor,
    luma: torch.Tensor,
    edge_map: torch.Tensor,
    highlight_mask: torch.Tensor,
    colorfulness: torch.Tensor,
    sensitivity: float = 2.0,
) -> torch.Tensor:
    """
    Transmission mask (glass, water, ice).
    Heuristic 1: chromatic dispersion — inter-channel variance at edges.
    Heuristic 2: Fresnel halo — achromatic bright edge regions.
    """
    R   = img[...,0].float()
    G   = img[...,1].float()
    Bc  = img[...,2].float()
    lf  = luma.float()

    r_ca = 3
    Rb   = _box_filter_bhwc(R.unsqueeze(-1),   r_ca).squeeze(-1)
    Gb   = _box_filter_bhwc(G.unsqueeze(-1),   r_ca).squeeze(-1)
    Bb   = _box_filter_bhwc(Bc.unsqueeze(-1),  r_ca).squeeze(-1)

    rgb_mean  = (Rb + Gb + Bb) / 3.0
    rgb_var   = ((Rb-rgb_mean)**2 + (Gb-rgb_mean)**2 + (Bb-rgb_mean)**2) / 3.0
    chroma_sh = rgb_var / (lf**2 + 1e-6)
    chroma_ed = chroma_sh * edge_map.float()
    fresnel   = edge_map.float() * highlight_mask.float() * (1.0 - colorfulness.float())
    raw       = chroma_ed * sensitivity + fresnel * 0.5
    Bs        = raw.shape[0]
    flat      = raw.view(Bs,-1)
    mx        = flat.max(dim=1).values.view(Bs,1,1).clamp(min=1e-8)
    return _to_3ch_image((raw / mx).clamp(0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
#  MOTION VECTOR — Lucas-Kanade (v3.0)
# ─────────────────────────────────────────────────────────────────────────────

def _optical_flow_lk(
    frame1: torch.Tensor,
    frame2: Optional[torch.Tensor],
    window_radius: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dense Lucas-Kanade optical flow via windowed least-squares (pure PyTorch).

    Solves the 2×2 per-pixel system over a (2r+1)² integration window:
      [ΣIx²  ΣIxIy] [u]   [-ΣIxIt]
      [ΣIxIy ΣIy² ] [v] = [-ΣIyIt]

    Returns (u, v): (B,H,W) flow in pixel units.
    Returns zeros when frame2 is None (static / single-frame mode).
    """
    B, H, W = frame1.shape
    dev = frame1.device

    if frame2 is None:
        zero = torch.zeros(B, H, W, device=dev, dtype=torch.float32)
        return zero, zero

    r  = max(1, window_radius)
    f1 = frame1.float()
    f2 = frame2.float()

    Ix, Iy = _scharr_gradient(f1)
    It     = f2 - f1

    def _ws(a):
        return _box_filter_bhwc(a.unsqueeze(-1), r).squeeze(-1)

    A11 = _ws(Ix * Ix)
    A12 = _ws(Ix * Iy)
    A22 = _ws(Iy * Iy)
    b1  = _ws(Ix * It)
    b2  = _ws(Iy * It)

    # Cramer's rule — regularise near-singular regions (textureless areas)
    det   = A11 * A22 - A12 * A12
    reg   = (A11 + A22) * 1e-4 + 1e-8
    det_r = det + reg

    u = (A12 * b2 - A22 * b1) / det_r
    v = (A12 * b1 - A11 * b2) / det_r

    max_disp = max(H, W) * 0.5
    u = u.clamp(-max_disp, max_disp)
    v = v.clamp(-max_disp, max_disp)
    return u, v


def _hsv_to_rgb_tensor(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Vectorised HSV → RGB on (B,H,W) tensors ∈ [0,1]. Returns (B,H,W,3)."""
    h6 = h * 6.0
    i  = h6.long() % 6
    f  = h6 - h6.floor()

    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)
    for sec, (rv, gv, bv) in enumerate(
        [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]
    ):
        mask = (i == sec)
        r = torch.where(mask, rv, r)
        g = torch.where(mask, gv, g)
        b = torch.where(mask, bv, b)

    return torch.stack([r, g, b], dim=-1)


def _flow_to_hsv_image(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    HSV-encode optical flow for visualisation.
    Hue = direction (0°=right 90°=down 180°=left 270°=up).
    Saturation = normalised magnitude. Value = 1.
    Returns (B,H,W,3) IMAGE in [0,1].
    """
    B, H, W = u.shape
    angle = torch.atan2(v, u)
    hue   = ((angle / (2.0 * math.pi)) + 0.5) % 1.0

    mag  = torch.sqrt(u**2 + v**2)
    flat = mag.view(B,-1)
    mx   = flat.max(dim=1).values.view(B,1,1).clamp(min=1e-8)
    sat  = (mag / mx).clamp(0.0, 1.0)
    val  = torch.ones_like(hue)

    return _hsv_to_rgb_tensor(hue, sat, val)


# ─────────────────────────────────────────────────────────────────────────────
#  OBJECT ID MATTE — k-means crypto-style (v3.0)
# ─────────────────────────────────────────────────────────────────────────────

def _cluster_id_colors(K: int, device: torch.device) -> torch.Tensor:
    """
    K visually distinct RGBA colors via golden-ratio hue spiral.
    φ-spacing guarantees maximum perceptual distance at any K.
    Returns (K, 4) float32 tensor.
    """
    phi = 1.6180339887
    colors = []
    for k in range(K):
        hue = (k * phi) % 1.0
        sat, val = 0.85, 0.90
        h6  = hue * 6.0
        idx = int(h6) % 6
        f   = h6 - int(h6)
        p_  = val * (1 - sat)
        q_  = val * (1 - f * sat)
        t_  = val * (1 - (1 - f) * sat)
        lut = [(val,t_,p_),(q_,val,p_),(p_,val,t_),(p_,q_,val),(t_,p_,val),(val,p_,q_)]
        r, g, b = lut[idx]
        colors.append([r, g, b, 1.0])
    return torch.tensor(colors, device=device, dtype=torch.float32)


def _object_id_matte(
    img: torch.Tensor,
    luma: torch.Tensor,
    n_segments: int = 16,
    n_iter: int = 10,
    spatial_weight: float = 0.25,
) -> torch.Tensor:
    """
    Cryptomatte-style per-object ID matte via k-means clustering.

    Feature vector per pixel: [R, G, B, luma, x_norm*sw, y_norm*sw]
    Runs on ≤192×192 downsampled image to keep N·K memory-flat.
    Labels bilinearly upsampled back to full resolution.
    Returns (B,H,W,4) RGBA — each cluster has a golden-ratio hue color.
    """
    B, H, W, C = img.shape
    dev = img.device
    K   = min(max(2, n_segments), H * W)

    # Adaptive downsample: max 192 px on longest side
    max_hw = 192
    scale  = max(1, max(H, W) // max_hw)
    H2, W2 = max(1, H // scale), max(1, W // scale)

    if scale > 1:
        img_s  = F.interpolate(
            img[...,:3].float().permute(0,3,1,2),
            (H2, W2), mode="bilinear", align_corners=False,
        ).permute(0,2,3,1)
        luma_s = F.interpolate(
            luma.float().unsqueeze(1),
            (H2, W2), mode="bilinear", align_corners=False,
        ).squeeze(1)
    else:
        img_s, luma_s = img[...,:3].float(), luma.float()

    xg = torch.linspace(0.0, 1.0, W2, device=dev)
    yg = torch.linspace(0.0, 1.0, H2, device=dev)
    gy, gx = torch.meshgrid(yg, xg, indexing="ij")

    N    = H2 * W2
    feat = torch.cat([
        img_s.reshape(B, N, 3),
        luma_s.reshape(B, N, 1),
        gx.unsqueeze(0).expand(B,-1,-1).reshape(B, N, 1) * spatial_weight,
        gy.unsqueeze(0).expand(B,-1,-1).reshape(B, N, 1) * spatial_weight,
    ], dim=-1)   # (B, N, 6)

    # Evenly-spaced centroid initialisation
    step   = max(1, N // K)
    init_i = torch.arange(0, min(K * step, N), step, device=dev)[:K]
    if init_i.shape[0] < K:
        extra  = torch.randint(0, N, (K - init_i.shape[0],), device=dev)
        init_i = torch.cat([init_i, extra])
    centroids = feat[:, init_i, :].clone()   # (B, K, 6)

    # Lloyd's iterations — fully vectorised
    for _ in range(n_iter):
        diffs  = feat.unsqueeze(2) - centroids.unsqueeze(1)   # (B, N, K, 6)
        labels = (diffs*diffs).sum(-1).argmin(-1)              # (B, N)
        one_hot = F.one_hot(labels, K).float()                 # (B, N, K)
        counts  = one_hot.sum(1)                               # (B, K)
        new_c   = torch.bmm(one_hot.permute(0,2,1), feat) / (counts.unsqueeze(-1) + 1e-8)
        empty   = (counts == 0).unsqueeze(-1).expand_as(centroids)
        centroids = torch.where(empty, centroids, new_c)

    diffs  = feat.unsqueeze(2) - centroids.unsqueeze(1)
    labels = (diffs*diffs).sum(-1).argmin(-1).reshape(B, H2, W2)

    if scale > 1:
        labels = F.interpolate(
            labels.float().unsqueeze(1), (H, W), mode="nearest"
        ).squeeze(1).long()

    colors   = _cluster_id_colors(K, dev)    # (K, 4)
    id_matte = colors[labels]                 # (B, H, W, 4)
    return id_matte.to(img.dtype)


# ─────────────────────────────────────────────────────────────────────────────
#  PASS CONFIDENCE (v3.0)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_pass_confidence(
    depth_provided: bool,
    normal_method: str,
    pass_normal: torch.Tensor,
    pass_ao: torch.Tensor,
    pass_albedo: torch.Tensor,
    pass_emission: torch.Tensor,
    pass_roughness: torch.Tensor,
    pass_transmission: torch.Tensor,
    pass_specular: torch.Tensor,
    edge_map: torch.Tensor,
    motion_u: torch.Tensor,
    motion_v: torch.Tensor,
    pass_quality_hint: str = "",
) -> str:
    """
    Per-pass quality confidence scores [0..1], returned as JSON string.

    depth      — 1.0 if depth_map connected, 0.0 otherwise
    ao         — AO map variance × 25 (structure richness)
    normal_map — fraction of pixels with well-defined (near-unit) normal vectors
    albedo     — mean absolute deviation from mean albedo (material colour spread)
    emission   — peak emission value (0=no emitters, 1=strong)
    roughness  — dynamic range of roughness map (spread = reliable)
    transmission — peak chromatic shift value
    edge       — fraction of pixels above 5% edge threshold (structural density)
    specular   — specular contrast std-dev (spread = reliable)
    motion     — mean flow magnitude relative to 10% of frame diagonal
    """
    with torch.no_grad():
        conf: Dict[str, Any] = {}

        conf["depth"] = 1.0 if depth_provided else 0.0

        if depth_provided:
            conf["ao"] = round(min(1.0, float(pass_ao[...,0].float().var()) * 25.0), 3)
        else:
            conf["ao"] = 0.0

        N_dec = pass_normal.float() * 2.0 - 1.0
        N_mag = torch.sqrt((N_dec**2).sum(-1))
        conf["normal_map"]    = round(float((N_mag > 0.85).float().mean()), 3)
        conf["normal_method"] = normal_method

        alb      = pass_albedo[...,:3].float()
        alb_mean = alb.mean(dim=(1,2,3), keepdim=True)
        conf["albedo"] = round(min(1.0, float((alb - alb_mean).abs().mean()) * 6.0), 3)

        conf["emission"]      = round(float(pass_emission[...,0].max()), 3)
        conf["roughness"]     = round(min(1.0, float(pass_roughness[...,0].max())
                                           - float(pass_roughness[...,0].min())), 3)
        conf["transmission"]  = round(float(pass_transmission[...,0].max()), 3)
        conf["edge"]          = round(float((edge_map.float() > 0.05).float().mean()), 3)
        conf["specular"]      = round(min(1.0, float(pass_specular.float().std()) * 6.0), 3)

        diag    = float(math.sqrt(motion_u.shape[1]**2 + motion_u.shape[2]**2))
        mot_mag = float(torch.sqrt(motion_u**2 + motion_v**2).mean())
        conf["motion"] = round(min(1.0, mot_mag / (diag * 0.1 + 1e-8)), 3)

        if pass_quality_hint:
            conf["_hint"] = pass_quality_hint

    return json.dumps(conf, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN NODE
# ─────────────────────────────────────────────────────────────────────────────

