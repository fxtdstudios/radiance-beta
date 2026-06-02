"""Public node naming and menu taxonomy for Radiance.

ComfyUI keeps node type keys as workflow compatibility identifiers. This module
centralizes the user-facing brand and category names without renaming those keys.
"""
from __future__ import annotations

import re
from typing import Any, MutableMapping

DISPLAY_PREFIX = "◎ Radiance"
MENU_ROOT = "FXTD STUDIOS/Radiance"

MENU_STRUCTURE = {
    "Core": "Manager, workspace, resolution, and everyday Radiance utilities.",
    "Load & Save": "Readers, writers, EXR, image, video, and file exchange.",
    "Generate": "Samplers, loaders, prompts, LoRA, regional conditioning, and denoise.",
    "Color": "White balance, CDL, curves, LUTs, OCIO, grades, and color checks.",
    "HDR": "HDR uplift, ACES, tone mapping, analysis, encoding, and delivery.",
    "VFX": "Plate prep, masks, inpaint, depth, optics, motion, multipass, and scene cuts.",
    "Video": "Text-to-video, image-to-video, temporal generation, and video HDR.",
    "Upscale": "AI upscaling, repair, enhancement, and restoration.",
    "Review": "Viewer, scopes, preview servers, contact sheets, and QC.",
    "Pipeline": "DCC bridges, metadata, audio, project handoff, and studio integration.",
    "Developer": "Training, diagnostics, and developer-only utilities.",
}

DEFAULT_MENU_SECTION = "Core"


def category_path(section: str) -> str:
    """Return the full ComfyUI category path for a menu section."""

    return f"{MENU_ROOT}/{section}"


def apply_radiance_branding(
    class_mappings: MutableMapping[str, Any],
    display_name_mappings: MutableMapping[str, Any],
) -> None:
    """Normalize visible node names and categories after discovery.

    Workflow node keys stay unchanged so existing graphs continue to load.
    """

    for node_key, node_class in class_mappings.items():
        display_name = display_name_mappings.get(node_key, node_key)
        display_name_mappings[node_key] = normalize_display_name(node_key, display_name)
        _set_node_category(
            node_class,
            classify_menu_section(node_key, node_class, display_name_mappings[node_key]),
        )


def normalize_display_name(node_key: str, display_name: Any) -> str:
    """Ensure every visible node label starts with the Radiance brand."""

    label = _clean_display_label(str(display_name or ""))
    if not label:
        label = _camel_to_words(node_key)

    label = _strip_known_prefixes(label)
    if not label:
        label = _camel_to_words(node_key)

    return f"{DISPLAY_PREFIX} {label}".strip()


def classify_menu_section(node_key: str, node_class: Any, display_name: str) -> str:
    """Classify a node into the public Radiance menu taxonomy."""

    module_name = getattr(node_class, "__module__", "") or ""
    current_category = str(getattr(node_class, "CATEGORY", "") or "")
    text = _searchable_text(" ".join([node_key, display_name, module_name, current_category]))

    if _has_any(text, "training", "sdr degradation", "turbo train", "developer", "diagnostic"):
        return "Developer"

    if _has_any(text, "manager", "workspace", "utility", "utilities"):
        return "Core"

    if _has_any(text, "upscale", "restore", "restoration", "supir", "esrgan", "swinir", "hat", "face restore"):
        return "Upscale"

    if _has_any(text, "viewer", "scope", "preview", "monitor", "qc", "policy", "contact sheet", "frame stamp"):
        return "Review"

    if _has_any(text, "video", "t2v", "i2v", "dit", "character", "wan", "ltx"):
        return "Video"

    if _has_any(
        text,
        "vfx", "scene cut", "shot", "edit", "router",
        "multipass", "relight", "aov", "cryptomatte", "normal pass",
        "material pass", "lighting pass", "mask", "matte", "roto",
        "alpha", "segmentation", "plate", "stabil", "grain matcher",
        "inpaint", "crop", "stitch", "depth", "geometry", "normal",
        "disparity", "lens", "optic", "camera", "chromatic", "anamorphic",
        "vignette", "film grain", "motion", "flow", "temporal",
    ):
        return "VFX"

    if _has_any(text, "hdr", "aces", "tonemap", "tone map", "luminance", "nit", "exr delivery", "vae"):
        return "HDR"

    if _has_any(text, "color", "colour", "grade", "curve", "ocio", "lut", "cdl", "white balance", "color space", "bit depth"):
        return "Color"

    if _has_any(text, "sampler", "loader", "lora", "prompt", "generate", "conditioning", "regional", "resolution", "denoise", "control net"):
        return "Generate"

    if _has_any(text, "read", "write", "io", "image", "sequence", "file", "exr", "png", "jpeg", "load image", "save image"):
        return "Load & Save"

    if _has_any(text, "project", "dcc", "nuke", "resolve", "mcp", "bridge", "metadata", "audio", "pipeline"):
        return "Pipeline"

    return DEFAULT_MENU_SECTION


def _set_node_category(node_class: Any, section: str) -> None:
    try:
        setattr(node_class, "CATEGORY", category_path(section))
    except Exception:
        # Some third-party node classes may block attribute writes. Their original
        # category remains safer than failing package load.
        return


def _clean_display_label(display_name: str) -> str:
    label = display_name.strip()
    label = re.sub(r"^[◎○●◉▷▶›>\-\s]+", "", label)
    label = re.sub(r"\s+", " ", label)
    return label.strip()


def _strip_known_prefixes(label: str) -> str:
    patterns = (
        r"^radiance\s+",
        r"^fxtd\s+studios?\s+",
        r"^fxtd\s+",
    )
    result = label.strip()
    for pattern in patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE).strip()
    return result


def _camel_to_words(value: str) -> str:
    value = re.sub(r"^Radiance", "", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)
    return value.replace("_", " ").strip()


def _searchable_text(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)
    value = re.sub(r"[^A-Za-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def _has_any(text: str, *tokens: str) -> bool:
    padded = f" {text} "
    return any(f" {_searchable_text(token)} " in padded for token in tokens)


__all__ = [
    "DISPLAY_PREFIX",
    "MENU_ROOT",
    "MENU_STRUCTURE",
    "apply_radiance_branding",
    "category_path",
    "classify_menu_section",
    "normalize_display_name",
]
