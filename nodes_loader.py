import json
import logging
import math
import os
import time

import torch
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management
from comfy.cldm.control_types import UNION_CONTROLNET_TYPES

import urllib.request
import tqdm

logger = logging.getLogger("◎ Radiance.loader")

# ═══════════════════════════════════════════════════════════════════════════════
#                         RADIANCE MODEL RESOURCE MAP
# ═══════════════════════════════════════════════════════════════════════════════

RADIANCE_MODEL_MAP = {
    # Diffusion Models (UNET / DiT)
    "flux1-schnell-fp8.safetensors": {
        "url": "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-schnell-fp8.safetensors",
        "type": "diffusion_models"
    },
    "flux1-dev-fp8.safetensors": {
        "url": "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors",
        "type": "diffusion_models"
    },
    "sd_xl_base_1.0.safetensors": {
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
        "type": "diffusion_models"
    },
    # Text Encoders (CLIP / T5)
    "t5xxl_fp8_e4m3fn.safetensors": {
        "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
        "type": "text_encoders"
    },
    "clip_l.safetensors": {
        "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
        "type": "text_encoders"
    },
    # VAE
    "ae.safetensors": {
        "url": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors",
        "type": "vae"
    }
}


def _download_model(url: str, target_path: str):
    """Download a model with a progress bar in the terminal."""
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        logger.info(f"📥 Radiance: Downloading model from {url}...")
        
        def progress_bar(t):
            last_b = [0]
            def update_to(b=1, bsize=1, tsize=None):
                if tsize is not None: t.total = tsize
                t.update((b - last_b[0]) * bsize)
                last_b[0] = b
            return update_to

        with tqdm.tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=os.path.basename(target_path)) as t:
            urllib.request.urlretrieve(url, filename=target_path, reporthook=progress_bar(t))
        
        logger.info(f"✅ Download complete: {target_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Download failed for {url}: {e}")
        return False


def _ensure_model_exists(name: str, folder_type: str, auto_download: bool = False) -> str | None:
    """Check if model exists, download if missing and auto_download is enabled."""
    if not name or name == "None":
        return None
        
    path = folder_paths.get_full_path(folder_type, name)
    if path and os.path.exists(path):
        return path
        
    if auto_download and name in RADIANCE_MODEL_MAP:
        res = RADIANCE_MODEL_MAP[name]
        if res["type"] == folder_type:
            # Construct target path if it doesn't exist
            # We use the first valid folder path for the type
            base_dir = folder_paths.get_folder_paths(folder_type)[0]
            target_path = os.path.join(base_dir, name)
            
            if _download_model(res["url"], target_path):
                # Re-scan to update ComfyUI's internal list
                folder_paths.get_filename_list(folder_type) 
                return target_path
                
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#                       ARCHITECTURE AUTO-DETECTION  (v3.0)
# ═══════════════════════════════════════════════════════════════════════════════

# Keys sampled from the first N keys of the state dict.
# Order matters — more specific patterns must come first.
_ARCH_HEURISTICS = [
    # Flux: unique double-stream block naming
    (lambda ks: any("double_blocks" in k for k in ks),             "flux"),
    # SD3 / SD3.5: joint transformer blocks
    (lambda ks: any("joint_blocks" in k for k in ks),              "sd3"),
    # HunyuanVideo: image input projection
    (lambda ks: any("img_in" in k and "proj" in k for k in ks),    "hunyuan_video"),
    # LTX: patch embedding
    (lambda ks: any("patch_embedding.weight" in k for k in ks),    "ltx"),
    # Lumina / Z-Image: unique caption projection key
    (lambda ks: any("cap_v_projection.weight" in k for k in ks),   "lumina2"),
    # Wan 2.1: patch_embedding + time_embedding combination unique to Wan
    # FIX 5: Previous heuristic checked for "wan" literal in first 5 key names
    # which fails for all Wan 2.1 checkpoints (keys start with
    # "model.diffusion_model.patch_embedding..." — no "wan" substring).
    (lambda ks: any("patch_embedding.weight" in k for k in ks)
              and any("time_embedding" in k for k in ks)
              and not any("joint_blocks" in k for k in ks),         "wan"),
    # PixArt: adaln single
    (lambda ks: any("adaln_single" in k for k in ks),              "pixart"),
    # Kolors: Chatglm style conditioning
    (lambda ks: any("chatglm" in k.lower() for k in ks),           "kolors"),
    # AuraFlow: unique prefix
    (lambda ks: any("auraflow" in k.lower() for k in ks),          "aura_flow"),
    # SDXL: has both UNet input_blocks AND add_embedding (refiner clue)
    (lambda ks: any("down_blocks.0" in k for k in ks)
              and any("add_embedding" in k for k in ks),            "sdxl"),
    # SD1.5: UNet with input_blocks naming
    (lambda ks: any("input_blocks.0" in k for k in ks),            "sd1.5"),
    # SDXL fallback: down_blocks without add_embedding → still SDXL
    (lambda ks: any("down_blocks.0" in k for k in ks),             "sdxl"),
]

_SAFETENSORS_PEEK = 80   # how many keys to read for heuristics


def _detect_model_type(unet_path: str) -> str | None:
    """
    Detect model architecture from safetensors metadata (key heuristics).
    Returns a model_type string or None if detection failed.
    Only reads the first _SAFETENSORS_PEEK keys for speed.
    """
    try:
        from safetensors import safe_open
        with safe_open(unet_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())[:_SAFETENSORS_PEEK]
        for test_fn, arch in _ARCH_HEURISTICS:
            if test_fn(keys):
                logger.info(f"🔍 Auto-detected architecture: {arch} from {os.path.basename(unet_path)}")
                return arch
    except ImportError:
        logger.debug("safetensors not available — skipping auto-detect")
    except Exception as e:
        logger.debug(f"Auto-detect failed: {e}")
    return None


def _file_fingerprint(path: str) -> str:
    """Return a cache key suffix that changes when the file changes."""
    try:
        st = os.stat(path)
        return f"{st.st_mtime:.0f}:{st.st_size}"
    except OSError:
        return "nostat"


# ═══════════════════════════════════════════════════════════════════════════════
#                         LATENT FORMAT TABLE  (v3.0)
# ═══════════════════════════════════════════════════════════════════════════════

LATENT_CHANNELS = {
    "flux":           16,
    "sd3":            16,
    "sd3.5":          16,
    "ltx":            16,
    "hunyuan_video":  16,
    "wan":            16,
    "lumina2":        16,
    "z_image":        16,
    "sdxl":            4,
    "sd1.5":           4,
    "pixart":          4,
    "aura_flow":       4,
    "kolors":          4,
}


def _latent_format(arch: str) -> str:
    ch = LATENT_CHANNELS.get(arch, 4)
    return f"{ch}ch"


# ═══════════════════════════════════════════════════════════════════════════════
#                       CLIP ASSEMBLY RULES  (v3.0)
# ═══════════════════════════════════════════════════════════════════════════════

# Defines which named CLIP slots to use per architecture, in load order.
# Slots: "clip_l" | "clip_g" | "t5xxl" | "llm_encoder"
CLIP_SLOT_ORDER = {
    "flux":           ["clip_l", "t5xxl"],
    "sd3":            ["clip_l", "clip_g", "t5xxl"],
    "sd3.5":          ["clip_l", "clip_g", "t5xxl"],
    "sdxl":           ["clip_l", "clip_g"],
    "sd1.5":          ["clip_l"],
    "hunyuan_video":  ["llm_encoder", "clip_l"],
    "wan":            ["t5xxl"],
    "ltx":            ["t5xxl"],
    "lumina2":        ["t5xxl"],
    "z_image":        ["t5xxl"],
    "pixart":         ["t5xxl"],
    "aura_flow":      ["clip_l"],
    "kolors":         ["llm_encoder"],
}


def _assemble_clip_paths(arch: str, clip_l, clip_g, t5xxl, llm_encoder) -> list[str]:
    """
    Build ordered list of CLIP paths from named slots for the given architecture.
    Only includes slots that are filled (not None/empty string).
    Falls back to any non-empty slot if arch is unknown.
    """
    slot_map = {
        "clip_l":      clip_l,
        "clip_g":      clip_g,
        "t5xxl":       t5xxl,
        "llm_encoder": llm_encoder,
    }
    order = CLIP_SLOT_ORDER.get(arch, list(slot_map.keys()))
    paths = []
    for slot in order:
        val = slot_map.get(slot)
        if val and val not in ("None", ""):
            p = folder_paths.get_full_path("text_encoders", val)
            if p:
                paths.append(p)
            else:
                logger.warning(f"◎ CLIP slot '{slot}' file not found: {val}")
    return paths


# ═══════════════════════════════════════════════════════════════════════════════
#                         CHECKPOINT PRESETS  (v3.0 — updated)
# ═══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_PRESETS = {
    "None (Manual)": {},
    # ── Flux ──
    "→ Flux Dev": {
        "model_type":    "flux",
        "weight_dtype":  "fp8_e4m3fn",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"clip_l": True, "t5xxl": True},
        "vram_gb":       12,
    },
    "→ Flux Schnell": {
        "model_type":    "flux",
        "weight_dtype":  "fp8_e4m3fn",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"clip_l": True, "t5xxl": True},
        "vram_gb":       10,
    },
    "→ Flux Dev (Low VRAM)": {
        "model_type":    "flux",
        "weight_dtype":  "fp8_e4m3fn",
        "clip_dtype":    "fp8_e4m3fn",
        "offload_mode":  "cpu_offload",
        "clip_slots":    {"clip_l": True, "t5xxl": True},
        "vram_gb":       8,
    },
    # ── SD3.x ──
    "→ SD3.5 Large": {
        "model_type":    "sd3.5",
        "weight_dtype":  "fp16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"clip_l": True, "clip_g": True, "t5xxl": True},
        "vram_gb":       16,
    },
    "→ SD3.5 Medium": {
        "model_type":    "sd3.5",
        "weight_dtype":  "fp16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"clip_l": True, "clip_g": True, "t5xxl": True},
        "vram_gb":       10,
    },
    "→ SD3.5 Turbo": {
        "model_type":    "sd3.5",
        "weight_dtype":  "bf16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"clip_l": True, "clip_g": True, "t5xxl": True},
        "vram_gb":       12,
    },
    # ── SDXL ──
    "→ SDXL Base": {
        "model_type":    "sdxl",
        "weight_dtype":  "fp16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"clip_l": True, "clip_g": True},
        "vram_gb":       8,
    },
    "→ SDXL Turbo": {
        "model_type":    "sdxl",
        "weight_dtype":  "fp16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"clip_l": True, "clip_g": True},
        "vram_gb":       6,
    },
    # ── SD 1.5 ──
    "→ SD 1.5": {
        "model_type":    "sd1.5",
        "weight_dtype":  "fp16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"clip_l": True},
        "vram_gb":       4,
    },
    # ── Video Models ──
    "→ HunyuanVideo": {
        "model_type":    "hunyuan_video",
        "weight_dtype":  "fp8_e4m3fn",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"llm_encoder": True, "clip_l": True},
        "vram_gb":       24,
    },
    "→ Wan 2.1": {
        "model_type":    "wan",
        "weight_dtype":  "fp8_e4m3fn",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"t5xxl": True},
        "vram_gb":       16,
    },
    "→ LTX Video": {
        "model_type":    "ltx",
        "weight_dtype":  "fp16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"t5xxl": True},
        "vram_gb":       12,
    },
    # ── Other Image Models ──
    "→ PixArt Sigma": {
        "model_type":    "pixart",
        "weight_dtype":  "fp16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"t5xxl": True},
        "vram_gb":       8,
    },
    "→ AuraFlow": {
        "model_type":    "aura_flow",
        "weight_dtype":  "fp16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"clip_l": True},
        "vram_gb":       10,
    },
    "→ Kolors": {
        "model_type":    "kolors",
        "weight_dtype":  "fp16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"llm_encoder": True},
        "vram_gb":       10,
    },
    "→ Lumina2": {
        "model_type":    "lumina2",
        "weight_dtype":  "bf16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"t5xxl": True},
        "vram_gb":       14,
    },
    "→ Z-Image": {
        "model_type":    "z_image",
        "weight_dtype":  "bf16",
        "clip_dtype":    "fp16",
        "offload_mode":  "none",
        "clip_slots":    {"t5xxl": True},
        "vram_gb":       16,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#                         VRAM UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_vram_usage(
    model_type: str,
    weight_dtype: str,
    clip_dtype: str = "fp16",
    has_loras: bool = False,
    has_controlnet: bool = False,
) -> float:
    base_vram = {
        "flux": 12.0, "sd3": 10.0, "sd3.5": 12.0,
        "sdxl": 6.5, "sd1.5": 3.5,
        "hunyuan_video": 20.0, "wan": 14.0, "ltx": 10.0,
        "pixart": 6.0, "aura_flow": 8.0, "kolors": 8.0,
    }.get(model_type, 8.0)

    unet_mult = {
        "fp32": 2.0, "fp16": 1.0, "bf16": 1.0,
        "fp8_e4m3fn": 0.6, "fp8_e5m2": 0.6, "default": 1.0,
    }.get(weight_dtype, 1.0)

    clip_vram = {
        "flux": 4.5, "sd3": 3.0, "sd3.5": 3.5,
        "sdxl": 1.5, "sd1.5": 0.8,
        "hunyuan_video": 4.5, "wan": 3.0, "ltx": 2.0,
        "pixart": 2.0, "aura_flow": 2.0, "kolors": 3.0,
    }.get(model_type, 2.0)

    clip_mult = {
        "fp32": 2.0, "fp16": 1.0, "bf16": 1.0,
        "fp8_e4m3fn": 0.55, "fp8_e5m2": 0.55, "default": 1.0,
    }.get(clip_dtype, 1.0)

    vram = (base_vram * unet_mult) + (clip_vram * clip_mult)
    if has_loras:
        vram += 0.5
    if has_controlnet:
        vram += 2.0
    return round(vram, 1)


def get_available_vram() -> float:
    try:
        if torch.cuda.is_available():
            free_mem, _ = torch.cuda.mem_get_info(0)
            return round(free_mem / (1024 ** 3), 1)
    except Exception:  # nosec B110
        pass
    return 0.0


def get_total_vram() -> float:
    try:
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory
            return round(total_mem / (1024 ** 3), 1)
    except Exception:  # nosec B110
        pass
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#                        CLIP TYPE MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

def get_clip_type_enum(model_type: str):
    mapping = {
        "flux":   comfy.sd.CLIPType.FLUX,
        "sd3":    comfy.sd.CLIPType.SD3,
        "sd3.5":  comfy.sd.CLIPType.SD3,
        "sdxl":   comfy.sd.CLIPType.STABLE_DIFFUSION,
        "sd1.5":  comfy.sd.CLIPType.STABLE_DIFFUSION,
    }

    for name in ("hunyuan_video", "wan", "ltx", "pixart", "aura_flow", "kolors", "lumina2", "z_image"):
        enum_name = name.upper().replace(".", "_")
        for variant in (enum_name, name.upper(), name.title().replace("_", "")):
            if hasattr(comfy.sd.CLIPType, variant):
                mapping[name] = getattr(comfy.sd.CLIPType, variant)
                break
        else:
            mapping.setdefault(name, comfy.sd.CLIPType.STABLE_DIFFUSION)

    clip_type = mapping.get(model_type)
    if clip_type is None:
        enum_name = model_type.upper().replace(".", "_")
        if hasattr(comfy.sd.CLIPType, enum_name):
            clip_type = getattr(comfy.sd.CLIPType, enum_name)
        else:
            logger.warning(
                f"◎ No CLIPType mapping for '{model_type}', "
                f"falling back to STABLE_DIFFUSION"
            )
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION

    return clip_type


# ═══════════════════════════════════════════════════════════════════════════════
#                        MODEL CACHE (LRU)
# ═══════════════════════════════════════════════════════════════════════════════

class _LRUCache:
    """
    Least-Recently-Used model cache using OrderedDict for O(1) hit/evict.

    Keys include mtime+size fingerprint so stale entries are automatically
    missed when files change on disk — no manual invalidation needed.

    FIX 3: Previous implementation used list.remove() on every get() and put()
    which is O(n) — scans the entire access list on every cache hit.
    Upgraded to collections.OrderedDict + move_to_end() for O(1) LRU,
    consistent with the SigmaCache upgrade in nodes_sampler.py.
    """

    def __init__(self, max_size: int = 4):
        from collections import OrderedDict
        self._cache: "OrderedDict[str, object]" = OrderedDict()
        self._max_size = max_size

    def get(self, key: str):
        if key in self._cache:
            self._cache.move_to_end(key)   # O(1) — mark as recently used
            return self._cache[key]
        return None

    def put(self, key: str, obj) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                evicted, _ = self._cache.popitem(last=False)  # O(1) LRU evict
                evicted_name = evicted.split(":")[1] if ":" in evicted else evicted
                logger.info(f"Cache evicted: {evicted_name}")
        self._cache[key] = obj

    def has(self, key: str) -> bool:
        return key in self._cache

    def clear(self) -> None:
        self._cache.clear()
        logger.info("Model cache cleared")

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def size(self) -> int:
        return len(self._cache)


# v3.1: Reduce default cache size to 2 to prevent VRAM exhaustion with Flux/SD3.5.
# Users can override via RADIANCE_CACHE_SIZE env var.
# FIX 2: Previous env var "◎ Radiance_CACHE_SIZE" was impossible to set from
# any shell — it contained a Unicode ◎ character and a space. Renamed to the
# standard POSIX-safe name. Set via: export RADIANCE_CACHE_SIZE=4
_DEFAULT_CACHE_SIZE = int(os.environ.get("RADIANCE_CACHE_SIZE", 2))
_cache = _LRUCache(max_size=_DEFAULT_CACHE_SIZE)


# ═══════════════════════════════════════════════════════════════════════════════
#                     RADIANCE LORA STACK NODE  (v3.0)
# ═══════════════════════════════════════════════════════════════════════════════

class RadianceLoraStack:
    """
    Compose up to 5 LoRAs into a LORA_STACK for use with RadianceUnifiedLoader.
    Can accept an upstream LORA_STACK to chain stacks.
    """

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        lora_slot = lambda tooltip: (
            lora_list,
            {"default": "None", "tooltip": tooltip},
        )
        str_slot = lambda tooltip: (
            "FLOAT",
            {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05,
             "tooltip": tooltip},
        )
        return {
            "required": {},
            "optional": {
                "lora_stack":     ("LORA_STACK", {"default": None,
                    "tooltip": "Chain an upstream LORA_STACK before these LoRAs."}),
                "lora_1":         lora_slot("LoRA 1"),
                "lora_1_model":   str_slot("LoRA 1 model strength"),
                "lora_1_clip":    str_slot("LoRA 1 CLIP strength"),
                "lora_2":         lora_slot("LoRA 2"),
                "lora_2_model":   str_slot("LoRA 2 model strength"),
                "lora_2_clip":    str_slot("LoRA 2 CLIP strength"),
                "lora_3":         lora_slot("LoRA 3"),
                "lora_3_model":   str_slot("LoRA 3 model strength"),
                "lora_3_clip":    str_slot("LoRA 3 CLIP strength"),
                "lora_4":         lora_slot("LoRA 4"),
                "lora_4_model":   str_slot("LoRA 4 model strength"),
                "lora_4_clip":    str_slot("LoRA 4 CLIP strength"),
                "lora_5":         lora_slot("LoRA 5"),
                "lora_5_model":   str_slot("LoRA 5 model strength"),
                "lora_5_clip":    str_slot("LoRA 5 CLIP strength"),
            },
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "build_stack"
    CATEGORY = "FXTD Studios/Radiance/Generate"
    DESCRIPTION = (
        "Compose up to 5 LoRAs into an accumulating LORA_STACK. "
        "Chain multiple stacks together. Feed into Radiance Unified Loader."
    )

    def build_stack(
        self,
        lora_stack=None,
        lora_1="None", lora_1_model=1.0, lora_1_clip=1.0,
        lora_2="None", lora_2_model=1.0, lora_2_clip=1.0,
        lora_3="None", lora_3_model=1.0, lora_3_clip=1.0,
        lora_4="None", lora_4_model=1.0, lora_4_clip=1.0,
        lora_5="None", lora_5_model=1.0, lora_5_clip=1.0,
    ) -> tuple:
        stack = list(lora_stack) if lora_stack else []
        for name, ms, cs in [
            (lora_1, lora_1_model, lora_1_clip),
            (lora_2, lora_2_model, lora_2_clip),
            (lora_3, lora_3_model, lora_3_clip),
            (lora_4, lora_4_model, lora_4_clip),
            (lora_5, lora_5_model, lora_5_clip),
        ]:
            if name and name != "None":
                stack.append((name, float(ms), float(cs)))
        return (stack,)


# ═══════════════════════════════════════════════════════════════════════════════
#                     RADIANCE UNIFIED LOADER v2.1.0
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_TYPES = [
    "Auto-Detect",
    "flux", "sd3", "sd3.5",
    "sdxl", "sd1.5",
    "hunyuan_video", "wan", "ltx",
    "lumina2", "z_image",
    "pixart", "aura_flow", "kolors",
]

WEIGHT_DTYPES = ["default", "fp8_e4m3fn", "fp8_e5m2", "fp16", "bf16", "fp32"]
CLIP_DTYPES   = ["default", "fp16", "bf16", "fp8_e4m3fn", "fp32"]
OFFLOAD_MODES = ["none", "cpu_offload", "sequential"]


class RadianceUnifiedLoader:
    """
    Universal diffusion model loader v2.1.0.

    Outputs: MODEL, CLIP, VAE, CONTROL_NET, LORA_STACK,
             load_info (human string), latent_format ("4ch"/"16ch"),
             model_meta (JSON string with full metadata dict).
    """

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        clip_list = ["None"] + folder_paths.get_filename_list("text_encoders")
        cn_list   = ["None"] + folder_paths.get_filename_list("controlnet")

        clip_slot = lambda tip: (clip_list, {"default": "None", "tooltip": tip})

        return {
            "required": {
                # ── Preset ──
                "preset": (
                    list(CHECKPOINT_PRESETS.keys()),
                    {"default": "None (Manual)",
                     "tooltip": "Quick-configure for common architectures. "
                                "Overrides model_type, dtypes, offload_mode, and hints "
                                "which CLIP slots are needed."},
                ),
                # ── UNET ──
                "unet_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {"tooltip": "Main diffusion model (UNET / DiT / Transformer)."},
                ),
                "weight_dtype": (
                    WEIGHT_DTYPES,
                    {"default": "default",
                     "tooltip": "UNET weight precision. fp8_e4m3fn saves ~40% VRAM vs fp16."},
                ),
                # ── Architecture ──
                "model_type": (
                    MODEL_TYPES,
                    {"default": "Auto-Detect",
                     "tooltip": "'Auto-Detect' reads the checkpoint's key names to determine "
                                "architecture. Override manually if detection fails."},
                ),
                # ── VAE ──
                "vae_name": (
                    folder_paths.get_filename_list("vae"),
                    {"tooltip": "VAE for encoding/decoding latents."},
                ),
            },
            "optional": {
                # ── Named CLIP slots ──
                "clip_l":      clip_slot(
                    "CLIP-L (text encoder). Used by: SD1.5, SDXL, Flux, SD3."),
                "clip_g":      clip_slot(
                    "CLIP-G (text encoder). Used by: SDXL, SD3, SD3.5."),
                "t5xxl":       clip_slot(
                    "T5-XXL (text encoder). Used by: Flux, SD3, SD3.5, Wan, LTX, PixArt."),
                "llm_encoder": clip_slot(
                    "LLM encoder (ChatGLM3 etc.). Used by: Kolors, HunyuanVideo."),
                # ── CLIP precision ──
                "clip_dtype": (
                    CLIP_DTYPES,
                    {"default": "default",
                     "tooltip": "CLIP weight precision. Independent from UNET. "
                                "For Flux T5XXL: fp8 saves ~4.7 GB vs fp16."},
                ),
                # ── Offload ──
                "offload_mode": (
                    OFFLOAD_MODES,
                    {"default": "none",
                     "tooltip": "none = GPU only. "
                                "cpu_offload = CLIP loaded to CPU RAM. "
                                "sequential = enable ComfyUI sequential CPU offload (8–12 GB GPUs)."},
                ),
                # ── LoRA (built-in slots) ──
                "lora_stack":      ("LORA_STACK", {"default": None,
                    "tooltip": "Accept a LORA_STACK from RadianceLoraStack node."}),
                "lora_1":          (lora_list, {"default": "None"}),
                "lora_1_model_str":("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05,
                    "tooltip": "LoRA 1 strength on model."}),
                "lora_1_clip_str": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05,
                    "tooltip": "LoRA 1 strength on CLIP."}),
                "lora_2":          (lora_list, {"default": "None"}),
                "lora_2_model_str":("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "lora_2_clip_str": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "lora_3":          (lora_list, {"default": "None"}),
                "lora_3_model_str":("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "lora_3_clip_str": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                # ── ControlNet ──
                "controlnet_name": (cn_list, {"default": "None",
                    "tooltip": "Optional ControlNet model."}),
                "controlnet_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "controlnet_start":    ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "controlnet_end":      ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                # ── Options ──
                "check_vram":  (["On", "Off"], {"default": "On",
                    "tooltip": "Estimate VRAM before load and warn if tight."}),
                "use_cache":   (["On", "Off"], {"default": "On",
                    "tooltip": "Cache loaded models. Skips disk I/O when re-running "
                               "with the same files. Cache auto-invalidates if files change."}),
                "lora_on_error": (["warn", "raise"], {"default": "raise",
                    "tooltip": "'warn' skips failed LoRA and continues. "
                               " 'raise' stops execution."}),
                "auto_download": ("BOOLEAN", {"default": False,
                    "tooltip": "If a selected model is missing, automatically download it from Radiance mirrors."}),
            },
        }

    RETURN_TYPES  = ("MODEL", "CLIP", "VAE", "CONTROL_NET", "LORA_STACK",
                     "STRING", "STRING", "STRING")
    RETURN_NAMES  = ("MODEL", "CLIP", "VAE", "CONTROLNET", "lora_stack",
                     "load_info", "latent_format", "model_meta")
    FUNCTION      = "load_radiance_stack"
    CATEGORY      = "FXTD Studios/Radiance/Generate"
    DESCRIPTION   = (
        "Universal loader v2.1.0 — auto-detects architecture, named CLIP slots, "
        "chainable LORA_STACK, model_meta JSON, latent_format, offload mode. "
        "ZERO stale cache hits (mtime + size fingerprinting)."
    )

    def load_radiance_stack(
        self,
        preset,
        unet_name,
        weight_dtype,
        model_type,
        vae_name,
        clip_l="None",
        clip_g="None",
        t5xxl="None",
        llm_encoder="None",
        clip_dtype="default",
        offload_mode="none",
        lora_stack=None,
        lora_1="None", lora_1_model_str=1.0, lora_1_clip_str=1.0,
        lora_2="None", lora_2_model_str=1.0, lora_2_clip_str=1.0,
        lora_3="None", lora_3_model_str=1.0, lora_3_clip_str=1.0,
        controlnet_name="None",
        controlnet_strength=1.0,
        controlnet_start=0.0,
        controlnet_end=1.0,
        check_vram="On",
        use_cache="On",
        lora_on_error="raise",
        auto_download=False,
    ):
        load_start  = time.time()
        info_lines  = []
        caching     = use_cache == "On"

        # ════════════════════════════════════════════════════════════════
        # 0. APPLY PRESET
        # ════════════════════════════════════════════════════════════════
        if preset != "None (Manual)" and preset in CHECKPOINT_PRESETS:
            cfg = CHECKPOINT_PRESETS[preset]
            overrides = []

            def _apply(field, key, cur):
                new = cfg.get(key, cur)
                if new != cur:
                    overrides.append(f"{field}: {cur}→{new}")
                return new

            model_type   = _apply("model_type",   "model_type",   model_type)
            weight_dtype = _apply("weight_dtype",  "weight_dtype", weight_dtype)
            clip_dtype   = _apply("clip_dtype",    "clip_dtype",   clip_dtype)
            offload_mode = _apply("offload_mode",  "offload_mode", offload_mode)

            msg = (f"✓ Preset '{preset}'" +
                   (f" overrode: {', '.join(overrides)}" if overrides else " (no overrides)"))
            logger.info(msg)
            info_lines.append(msg)

        # ════════════════════════════════════════════════════════════════
        # 1. RESOLVE ARCHITECTURE
        # ════════════════════════════════════════════════════════════════
        unet_path = _ensure_model_exists(unet_name, "diffusion_models", auto_download)
        if not unet_path:
            raise FileNotFoundError(
                f"❌ UNET not found: '{unet_name}'. Enable auto_download or install it manually."
            )

        detected_type = None
        if model_type == "Auto-Detect":
            detected_type = _detect_model_type(unet_path)
            if detected_type:
                resolved_type = detected_type
                info_lines.append(f"◎ Auto-detected: {resolved_type}")
            else:
                resolved_type = "sdxl"   # safe fallback
                logger.warning(
                    "◎ Architecture auto-detect failed. Falling back to 'sdxl'. "
                    "Set model_type manually if this is wrong."
                )
                info_lines.append("◎ Auto-detect failed — fallback: sdxl")
        else:
            resolved_type = model_type

        latent_fmt  = _latent_format(resolved_type)
        lat_msg     = f"◎ Latent format: {latent_fmt} ({resolved_type})"
        logger.info(lat_msg)
        info_lines.append(lat_msg)

        # ════════════════════════════════════════════════════════════════
        # 2. OFFLOAD MODE
        # ════════════════════════════════════════════════════════════════
        if offload_mode == "sequential":
            try:
                comfy.model_management.set_lowvram_mode(True)
                logger.info("◎ Sequential CPU offload enabled")
                info_lines.append("◎ Offload: sequential")
            except Exception as e:
                logger.warning(f"◎ Could not enable sequential offload: {e}")

        # FIX 6: ComfyUI model_options["load_device"] expects torch.device, not str.
        clip_load_device = torch.device("cpu") if offload_mode == "cpu_offload" else None

        # ════════════════════════════════════════════════════════════════
        # 3. VRAM ESTIMATION
        # ════════════════════════════════════════════════════════════════
        has_loras = (
            any(l != "None" for l in [lora_1, lora_2, lora_3])
            or bool(lora_stack)
        )
        has_cn = bool(controlnet_name and controlnet_name != "None")

        if check_vram == "On":
            est   = estimate_vram_usage(resolved_type, weight_dtype, clip_dtype,
                                        has_loras, has_cn)
            avail = get_available_vram()
            total = get_total_vram()
            vram_msg = (f"◎ VRAM: ~{est} GB needed | "
                        f"{avail} GB free / {total} GB total")
            logger.info(vram_msg)
            info_lines.append(vram_msg)
            if avail > 0 and est > avail * 0.9:
                warn = (f"◎ VRAM tight! {est} GB estimated, {avail} GB free. "
                        f"Consider fp8 dtype or cpu_offload.")
                logger.warning(warn)
                info_lines.append(warn)
        else:
            est = estimate_vram_usage(resolved_type, weight_dtype, clip_dtype,
                                      has_loras, has_cn)

        # ════════════════════════════════════════════════════════════════
        # 4. LOAD UNET  (mtime + size cache key)
        # ════════════════════════════════════════════════════════════════
        t0 = time.time()
        unet_fp   = _file_fingerprint(unet_path)
        unet_key  = f"unet:{unet_path}:{weight_dtype}:{unet_fp}"

        # FIX 4: Record cache HIT before loading — after _cache.put() the key
        # is always present, so checking has() post-load always returns True.
        unet_cache_hit = caching and _cache.has(unet_key)
        if unet_cache_hit:
            model = _cache.get(unet_key)
            logger.info(f"◎ UNET from cache: {unet_name}")
            info_lines.append(f"◎ UNET: {unet_name} (cached)")
        else:
            model_options = {}
            is_gguf = unet_name.lower().endswith(".gguf")
            if is_gguf:
                logger.info(f"◎ GGUF detected: {unet_name} (embedded quant)")
            else:
                dtype_map = {
                    "fp8_e4m3fn": torch.float8_e4m3fn,
                    "fp8_e5m2":   torch.float8_e5m2,
                    "fp16":       torch.float16,
                    "bf16":       torch.bfloat16,
                    "fp32":       torch.float32,
                }
                if weight_dtype in dtype_map:
                    model_options["dtype"] = dtype_map[weight_dtype]

            try:
                model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
                elapsed = time.time() - t0
                logger.info(f"◎ UNET: {unet_name} [{weight_dtype}] ({elapsed:.1f}s)")
                info_lines.append(f"◎ UNET: {unet_name} [{weight_dtype}] ({elapsed:.1f}s)")
                if caching:
                    _cache.put(unet_key, model)
            except Exception as e:
                raise RuntimeError(f"❌ Failed to load UNET '{unet_name}': {e}")

        # ════════════════════════════════════════════════════════════════
        # 5. LOAD CLIP  (named slots → ordered paths → mtime cache key)
        # ════════════════════════════════════════════════════════════════
        t0 = time.time()
        
        # Ensure all selected CLIPs exist/downloaded
        for slot, val in [("clip_l", clip_l), ("clip_g", clip_g), 
                          ("t5xxl", t5xxl), ("llm_encoder", llm_encoder)]:
             _ensure_model_exists(val, "text_encoders", auto_download)

        clip_paths = _assemble_clip_paths(resolved_type, clip_l, clip_g, t5xxl, llm_encoder)

        if not clip_paths:
            raise ValueError(
                f"❌ No CLIP encoders provided for architecture '{resolved_type}'. "
                f"Fill the required slot(s): "
                f"{', '.join(CLIP_SLOT_ORDER.get(resolved_type, ['clip_l']))}"
            )

        clip_fps     = ":".join(_file_fingerprint(p) for p in clip_paths)
        clip_key     = f"clip:{':'.join(clip_paths)}:{resolved_type}:{clip_dtype}:{clip_fps}"

        clip_slot_used = []
        for slot, val in [("clip_l", clip_l), ("clip_g", clip_g),
                          ("t5xxl", t5xxl), ("llm_encoder", llm_encoder)]:
            if val and val not in ("None", ""):
                clip_slot_used.append(slot)

        if caching and _cache.has(clip_key):
            clip = _cache.get(clip_key)
            logger.info(f"◎ CLIP from cache: {clip_slot_used}")
            info_lines.append(f"◎ CLIP: {'+'.join(clip_slot_used)} (cached)")
        else:
            clip_type_enum = get_clip_type_enum(resolved_type)
            clip_model_opts = {}
            if clip_load_device:
                clip_model_opts["load_device"] = clip_load_device
            dtype_map = {
                "fp16": torch.float16, "bf16": torch.bfloat16,
                "fp8_e4m3fn": torch.float8_e4m3fn, "fp32": torch.float32,
            }
            if clip_dtype in dtype_map:
                clip_model_opts["dtype"] = dtype_map[clip_dtype]

            try:
                clip = comfy.sd.load_clip(
                    ckpt_paths=clip_paths,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=clip_type_enum,
                    model_options=clip_model_opts if clip_model_opts else {},
                )
                elapsed = time.time() - t0
                logger.info(
                    f"◎ CLIP: {'+'.join(clip_slot_used)} "
                    f"[type={resolved_type}, dtype={clip_dtype}] ({elapsed:.1f}s)"
                )
                info_lines.append(
                    f"◎ CLIP: {'+'.join(clip_slot_used)} [{clip_dtype}] ({elapsed:.1f}s)"
                )
                if caching:
                    _cache.put(clip_key, clip)
            except Exception as e:
                raise RuntimeError(f"❌ Failed to load CLIP: {e}")

        # ════════════════════════════════════════════════════════════════
        # 6. LOAD VAE  (mtime cache key)
        # ════════════════════════════════════════════════════════════════
        t0 = time.time()
        vae_path = _ensure_model_exists(vae_name, "vae", auto_download)
        if not vae_path:
            raise FileNotFoundError(f"❌ VAE not found: '{vae_name}'. Enable auto_download or install it manually.")

        vae_fp  = _file_fingerprint(vae_path)
        vae_key = f"vae:{vae_path}:{vae_fp}"

        if caching and _cache.has(vae_key):
            vae = _cache.get(vae_key)
            logger.info(f"◎ VAE from cache: {vae_name}")
            info_lines.append(f"◎ VAE: {vae_name} (cached)")
        else:
            try:
                sd  = comfy.utils.load_torch_file(vae_path)
                vae = comfy.sd.VAE(sd=sd)
                elapsed = time.time() - t0
                logger.info(f"◎ VAE: {vae_name} ({elapsed:.1f}s)")
                info_lines.append(f"◎ VAE: {vae_name} ({elapsed:.1f}s)")
                if caching:
                    _cache.put(vae_key, vae)
            except Exception as e:
                raise RuntimeError(f"❌ Failed to load VAE '{vae_name}': {e}")

        # ════════════════════════════════════════════════════════════════
        # 7. APPLY LoRA STACK
        #    Priority: upstream lora_stack → built-in lora_1/2/3
        # ════════════════════════════════════════════════════════════════
        combined_loras = list(lora_stack) if lora_stack else []
        for name, ms, cs in [
            (lora_1, lora_1_model_str, lora_1_clip_str),
            (lora_2, lora_2_model_str, lora_2_clip_str),
            (lora_3, lora_3_model_str, lora_3_clip_str),
        ]:
            if name and name != "None":
                combined_loras.append((name, float(ms), float(cs)))

        applied_loras = []
        for i, (lora_name, model_str, clip_str) in enumerate(combined_loras, 1):
            if model_str == 0 and clip_str == 0:
                continue
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if not lora_path:
                msg = f"◎ LoRA not found: '{lora_name}'"
                if lora_on_error == "raise":
                    raise FileNotFoundError(msg)
                logger.warning(f"◎ {msg} — Skipping.")
                info_lines.append(f"◎ LoRA {i}: {lora_name} (not found)")
                continue
            try:
                t0 = time.time()
                lora_data = comfy.utils.load_torch_file(lora_path)
                model, clip = comfy.sd.load_lora_for_models(
                    model, clip, lora_data, model_str, clip_str
                )
                elapsed = time.time() - t0
                logger.info(
                    f"◎ LoRA {i}: {lora_name} (model={model_str}, clip={clip_str}, {elapsed:.1f}s)"
                )
                info_lines.append(
                    f"◎ LoRA {i}: {lora_name} [m={model_str} c={clip_str}] ({elapsed:.1f}s)"
                )
                applied_loras.append({"name": lora_name, "model_str": model_str,
                                      "clip_str": clip_str})
            except Exception as e:
                msg = f"Failed to apply LoRA '{lora_name}': {e}"
                if lora_on_error == "raise":
                    raise RuntimeError(f"❌ {msg}")
                logger.warning(f"◎ {msg} — Skipping.")
                info_lines.append(f"◎ LoRA {i}: {lora_name} (failed)")

        # ════════════════════════════════════════════════════════════════
        # 8. LOAD CONTROLNET
        # ════════════════════════════════════════════════════════════════
        controlnet = None
        if controlnet_name and controlnet_name != "None":
            cn_path = folder_paths.get_full_path("controlnet", controlnet_name)
            if not cn_path:
                logger.warning(f"◎ ControlNet not found: '{controlnet_name}'")
                info_lines.append(f"◎ ControlNet: {controlnet_name} (not found)")
            else:
                try:
                    t0 = time.time()
                    controlnet = comfy.sd.load_controlnet(cn_path)
                    elapsed = time.time() - t0
                    logger.info(
                        f"◎ ControlNet: {controlnet_name} "
                        f"(str={controlnet_strength} range={controlnet_start}-{controlnet_end}, "
                        f"{elapsed:.1f}s)"
                    )
                    info_lines.append(
                        f"◎ ControlNet: {controlnet_name} "
                        f"[str={controlnet_strength} {controlnet_start}-{controlnet_end}] "
                        f"({elapsed:.1f}s)"
                    )
                    try:
                        controlnet.radiance_strength = float(controlnet_strength)
                        controlnet.radiance_start    = float(controlnet_start)
                        controlnet.radiance_end      = float(controlnet_end)
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning(f"◎ ControlNet load failed '{controlnet_name}': {e}")
                    info_lines.append(f"◎ ControlNet: {controlnet_name} (failed)")

        # ════════════════════════════════════════════════════════════════
        # 9. BUILD OUTPUTS
        # ════════════════════════════════════════════════════════════════
        total_ms = round((time.time() - load_start) * 1000)
        summary  = (f"◎ Load complete in {total_ms / 1000:.1f}s"
                    + (f" (cache: {_cache.size})" if caching else ""))
        logger.info(summary)
        info_lines.append(summary)

        load_info = "\n".join(info_lines)

        # model_meta — structured JSON for downstream QC / analytics nodes
        model_meta = {
            "arch":          resolved_type,
            "detected":      detected_type is not None,
            "unet_file":     unet_name,
            "weight_dtype":  weight_dtype,
            "clip_slots":    clip_slot_used,
            "clip_dtype":    clip_dtype,
            "offload_mode":  offload_mode,
            "latent_ch":     LATENT_CHANNELS.get(resolved_type, 4),
            "latent_format": latent_fmt,
            "vram_est_gb":   est,
            "loras":         applied_loras,
            "controlnet":    controlnet_name if controlnet else None,
            "load_ms":       total_ms,
            "cached_unet":   unet_cache_hit,  # FIX 4: True only when loaded from cache
        }

        # Output the accumulated lora list (for chaining downstream loaders
        # or QC nodes that want to know what was applied)
        out_lora_stack = [(e["name"], e["model_str"], e["clip_str"])
                          for e in applied_loras] if applied_loras else None

        return (
            model,
            clip,
            vae,
            controlnet,
            out_lora_stack,
            load_info,
            latent_fmt,
            json.dumps(model_meta, indent=2),
        )


# ═══════════════════════════════════════════════════════════════════════════════
class RadianceControlNetApply:
    """
    Advanced ControlNet application node for the Radiance suite.
    - Gracefully bypasses if CONTROL_NET is None (prevents AttributeError crashes).
    - Supports standard start/end percent and strength controls.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "control_net": ("CONTROL_NET", ),
                "image": ("IMAGE", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Global strength of the control effect."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Percentage of the generation where control starts (0.0 = beginning)."}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Percentage of the generation where control ends (1.0 = end)."}),
                "control_type": (["auto"] + list(UNION_CONTROLNET_TYPES.keys()), {"default": "auto",
                    "tooltip": "For Union ControlNets (like Flux), select the specific control mode (Canny, Depth, etc.)."}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)  # FIX 7: was missing — ComfyUI showed generic labels
    FUNCTION = "apply_controlnet"
    CATEGORY = "FXTD Studios/Radiance/Generate"

    def apply_controlnet(self, conditioning, control_net, image, strength, start_percent, end_percent, control_type="auto"):
        # 1. Graceful Bypass: If no control_net or zero strength, just return the input conditioning.
        if control_net is None:
            logger.info("◎ Radiance Control: No ControlNet connected. Bypassing.")
            return (conditioning, )
            
        if strength == 0:
            return (conditioning, )

        # 2. Set Union ControlNet Type (if applicable)
        control_net = control_net.copy()
        type_number = UNION_CONTROLNET_TYPES.get(control_type, -1)
        if type_number >= 0:
            control_net.set_extra_arg("control_type", [type_number])
        else:
            control_net.set_extra_arg("control_type", [])

        # 3. Advanced Application (Standard ComfyUI Logic with added safety)
        c = []
        try:
            for t in conditioning:
                n = [t[0], t[1].copy()]
                c_net = control_net.copy().set_cond_hint(image, strength, (start_percent, end_percent))
                if 'control' in n[1]:
                    c_net.set_previous_controlnet(n[1]['control'])
                n[1]['control'] = c_net
                n[1]['control_apply_strength'] = strength
                c.append(n)
            return (c, )
        except Exception as e:
            logger.error(f"❌ Radiance Control: Failed to apply ControlNet: {e}")
            return (conditioning, )


# ═══════════════════════════════════════════════════════════════════════════════
#                         NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

# FIX 1: NODE_CLASS_MAPPINGS keys must be plain ASCII identifiers.
# The ◎ prefix belongs only in NODE_DISPLAY_NAME_MAPPINGS (the user-visible label).
# Having it in the type key breaks ComfyUI workflow JSON serialization and node lookup.
NODE_CLASS_MAPPINGS = {
    "RadianceUnifiedLoader": RadianceUnifiedLoader,
    "RadianceLoraStack":     RadianceLoraStack,
    "RadianceControlApply":  RadianceControlNetApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceUnifiedLoader": "◎ Radiance Unified Loader",
    "RadianceLoraStack":     "◎ Radiance LoRA Stack",
    "RadianceControlApply":  "◎ Radiance Control",
}
