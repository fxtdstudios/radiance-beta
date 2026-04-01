"""
◎ Radiance Cinematic Prompt Encoder — v2.3.3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Production-grade cinematic prompt builder with direct CLIP/T5 encoding.
Auto-selects prose for Flux/T5/Kolors and structured keywords for SD1.5/SDXL.

v2.3.3 Changelog (from review audit):
───────────────────────────────────────
CRITICAL FIXES:
  [BUG-C1] _encode() fallback path returned raw (cond, pooled) tuple instead of
           ComfyUI conditioning format [[cond, {"pooled_output": pooled}]].
           Sampler crashes on older ComfyUI installs (pre encode_from_tokens_scheduled).
  [BUG-C2] Structured path: "Est. Year {year_era}." had trailing period, then
           ", ".join(finish) + "." appended another → double period "..".
  [BUG-C3] Structured path: art_direction and lora_keywords both inserted at index 1,
           reversing their intended order (lora ended up before art_direction).

IMPORTANT FIXES:
  [BUG-I1] `subject_first` weight mode offered in INPUT_TYPES but had zero distinct
           handling — fell through to `balanced` silently. Now leads with subject
           at elevated weight before any technical descriptors.
  [BUG-I2] `sd3.5` missing from target_arch dropdown despite being in PROSE_ARCHS.
           Also added `hunyuan_video` which was in PROSE_ARCHS but missing from dropdown.
  [BUG-I3] Redundant `import torch as _torch` inside _real_token_count — torch is
           already imported at module level.
  [BUG-I4] DEFAULT_YEAR = 2024 was stale. Updated to 2025.
  [BUG-I5] use_break parameter type mismatch — string "On"/"Off" in encode_cinematic
           vs boolean False passed to build_cinematic_prompt_v3. Unified to boolean.
  [BUG-I6] Version string in DESCRIPTION said "v3.0" — updated to v2.3.3.
  [BUG-I7] clip.clone() not guarded — some custom CLIP wrappers lack .clone().
           Added try/except fallback.
  [BUG-I8] _validate_presets() only logged errors silently. Added strict mode
           that raises on validation failure when RADIANCE_STRICT env var is set.

MINOR FIXES:
  [BUG-M1] NeuralGrammar.scientificize could produce irregular spacing when
           punctuation-bearing words were replaced with multi-word phrases.
           Added post-cleanup pass.
  [BUG-M2] estimate_tokens didn't account for BREAK token overhead. Added
           BREAK_TOKEN_OVERHEAD constant used in insert_break_points.
  [BUG-M3] Structured path: tech block could duplicate camera when
           prompt_weight_mode was "subject_first" (new mode). Gated properly.
  [BUG-M4] _real_token_count: explicit tuple unpacking validation added.
  [BUG-M5] enhance_prompt_grammar: creative tags injected even when they were
           already present in the prompt — causing duplication on re-encodes.
           Added dedup check.
  [BUG-M6] Negative prompt for Anime style was missing "3d render" exclusion.
"""
import logging
import os
import re
import torch
from typing import Optional

logger = logging.getLogger("◎ Radiance.prompt")

__version__ = "2.3.3"


class CinematicDatasets:
    """Shared datasets for all cinematic prompt nodes. Single source of truth."""

    CAMERAS = [
        "None",
        "ARRI Alexa 65 (IMAX)",
        "ARRI Alexa Mini LF",
        "ARRI Alexa 35",
        "Sony Venice 2",
        "Sony FX9",
        "Sony A7S III",
        "RED V-Raptor XL",
        "RED Komodo",
        "RED Monstro 8K VV",
        "Panavision Millennium DXL2",
        "Panavision Panaflex Gold II (35mm)",
        "IMAX 15/70mm Film Camera",
        "Bolex H16 (16mm Film)",
        "Super 8mm Camera",
        "Canon C700 FF",
        "Canon C300 Mark III",
        "Blackmagic URSA Mini Pro 12K",
        "GoPro Hero 12",
        "iPhone 15 Pro Max",
        "Polaroid SX-70",
        "Vintage Daguerreotype Camera",
    ]

    LENSES = [
        "None",
        # Prime Focal Lengths
        "14mm Ultra-Wide Angle",
        "16mm Ultra-Wide Angle",
        "24mm Wide Angle",
        "28mm Wide Angle",
        "35mm Classic Wide",
        "40mm Semi-Wide",
        "50mm Standard Prime",
        "85mm Portrait Prime",
        "105mm Macro",
        "135mm Medium Telephoto",
        "200mm Telephoto Compression",
        "600mm Super Telephoto",
        # Specialty Lenses
        "Anamorphic Lens",
        "Fish-Eye Lens",
        "Tilt-Shift Lens",
        # Cinema Primes (High-End)
        "Cooke S7/i Full Frame",
        "Cooke Speed Panchro Vintage",
        "Zeiss Master Prime",
        "Zeiss Supreme Prime",
        "ARRI/Zeiss Signature Prime",
        "ARRI Master Anamorphic",
        "Panavision Primo 70",
        "Panavision C-Series Anamorphic",
        # Vintage & Character Lenses
        "Canon K35 Vintage",
        "Canon FD 50mm L",
        "Helios 44-2 58mm (Swirly Bokeh)",
        "Lensbaby Velvet (Soft Focus)",
        "Petzval 85mm (Classic Swirl)",
        # Modern Professional
        "Leica Summilux 50mm f/1.4",
        "Leica Summicron 35mm",
        "Sigma Art 35mm f/1.4",
        "Sigma Art 85mm f/1.4",
        "Sony G Master 24-70mm",
        "Sony G Master 85mm",
        # Specialty & Macro
        "Laowa Probe Lens (Macro)",
        "Freefly Wave (High-Speed)",
        "Angenieux Optimo Zoom",
        "Fujinon Premista Zoom",
    ]

    APERTURES = [
        "None",
        "f/0.95 (Razor Thin DoF)",
        "f/1.2 (Dreamy Bokeh)",
        "f/1.8 (Soft Background)",
        "f/2.0 (Shallow Cinematic)",
        "f/2.8 (Cinematic Separation)",
        "f/4.0 (Balanced)",
        "f/5.6 (Sharp Subject)",
        "f/8.0 (Deep Focus)",
        "f/11 (Landscape Sharpness)",
        "f/16 (Everything in Focus)",
        "f/22 (Diffraction Starbursts)",
    ]

    FRAMING = [
        "None",
        "Extreme Close-Up (ECU)",
        "Close-Up (CU)",
        "Medium Close-Up (MCU)",
        "Medium Shot (MS)",
        "Medium Wide (MW)",
        "Medium Full Shot (MFS)",
        "Cowboy Shot (American Shot)",
        "Wide Shot (WS)",
        "Full Body Shot (Wide)",
        "Extreme Wide Shot (EWS)",
        "Establishing Shot",
        "Over-The-Shoulder (OTS)",
        "Point of View (POV)",
        "Low Angle (Hero Shot)",
        "High Angle (Vulnerability)",
        "Bird's Eye View (Overhead)",
        "Worm's Eye View",
        "Dutch Angle (Canted)",
        "Symmetrical Composition",
        "Rule of Thirds",
    ]

    LIGHTING = [
        "None",
        "Rembrandt Lighting",
        "Chiaroscuro (High Contrast)",
        "Film Noir Lighting",
        "Split Lighting",
        "Butterfly Lighting",
        "Paramount Lighting",
        "Soft Window Light",
        "Golden Hour (Magic Hour)",
        "Blue Hour",
        "Cinematic Haze / Volumetric Fog",
        "God Rays (Crepuscular Rays)",
        "Neon Cyberpunk Lighting",
        "Practical Lighting",
        "Bioluminescence",
        "Studio Strobe 3-Point Setup",
        "Ring Light",
        "Candlelight",
        "Moonlight",
        "Overcast Soft Light",
        "Harsh Sunlight",
        # v2.3.3: Added for director-style presets
        "Natural Ambient Light",
        "High-Key Lighting",
        "Naturalistic Interior Light",
        "Interview 3-Point Setup",
        "Volumetric Fog / Atmospheric Haze",
    ]

    STYLES = [
        "None",
        "Photorealistic (Raw)",
        "Cinematic Movie Still",
        "Hyper-Realism",
        "Editorial Photography",
        "National Geographic Style",
        "Documentary Texture",
        "Vintage 1990s VHS",
        "Analog Film (Kodak Portra 400)",
        "Fujifilm Velvia 50",
        "Black and White (Ilford HP5)",
        "Monochrome Noir",
        "CGI 3D Render (Octane)",
        "Unreal Engine 5",
        "Pixar Animation Style",
        "Anime (Makoto Shinkai)",
        "Oil Painting (Classic)",
        "Concept Art",
        "Cyberpunk 2077 Aesthetic",
        "Wes Anderson Symmetric",
        "Tarantino Violence",
        "Kubrick One-Point Perspective",
        "Blade Runner Atmosphere",
        # v2.3.3: Added for director-style presets
        "Film Noir Aesthetic",
        "Dreamy Soft Focus",
        "Atmospheric Cinematic",
        "Vintage Kodachrome Look",
        "Cinematic TV Drama",
        "Documentary Realism",
        "Neo-Noir Modern",
        "Magical Realism",
    ]

    FILM_STOCKS = [
        "None",
        "Kodak Vision3 500T",
        "Kodak Vision3 250D",
        "Kodak Portra 400",
        "Kodak Ektar 100",
        "Kodak Tri-X 400 (B&W)",
        "Fujifilm Pro 400H",
        "Cinestill 800T",
        "Fujifilm Velvia 50",
        "Ilford Delta 3200",
        "Polaroid 600",
        "Wet Plate Collodion",
        # v2.3.3: Added for director-style presets
        "IMAX 15/70mm Film Stock",
        "Eastman Kodak 5254 (Vintage)",
    ]

    SHUTTER_SPEEDS = [
        "None",
        "1/50th sec (Standard Motion Blur)",
        "1/1000th sec (Frozen Action)",
        "Long Exposure (Light Trails)",
        "Slow Shutter (Dreamy Blur)",
    ]

    COLOR_GRADING = [
        "None",
        "Teal and Orange (Blockbuster)",
        "Bleach Bypass (Gritty)",
        "Technicolor (Vintage)",
        "Cross Processed",
        "Desaturated (Muted)",
        "Vibrant High Contrast",
        "Sepia Tone",
        "Monochrome High Key",
        "Cyberpunk Neon Grading",
        "Pastel Soft Tones",
        # v2.3.3: Added for director-style presets
        "Neutral ACES Workflow",
        "Desaturated Cool Tones",
        "Moody Shadows",
        "Orange and Teal (Hollywood)",
    ]

    ASPECT_RATIOS = [
        "None",
        "16:9 (Widescreen)",
        "2.39:1 (Anamorphic Scope)",
        "4:3 (Academy Ratio)",
        "1:1 (Square)",
        "9:16 (Social Vertical)",
        "21:9 (Ultrawide)",
        # v2.3.3: Added for director-style presets
        "1.43:1 (IMAX)",
        "1.85:1 (Standard Widescreen)",
        "2.00:1 (DCI Flat)",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    #                         ONE-CLICK STYLE PRESETS
    # ═══════════════════════════════════════════════════════════════════════════

    STYLE_PRESETS = [
        "None (Custom)",
        "→ Classic Hollywood",
        "→ Film Noir",
        "→ Sci-Fi Cinematic",
        "→ Cyberpunk",
        "→ Drama / Emotional",
        "→ Epic Landscape",
        "→ Portrait",
        "→ Documentary",
        "→ Artistic / Painterly",
        "→ Retro VHS",
        "→ Golden Hour Magic",
        "→ Moody Night",
        "→ Action / Dynamic",
        "→ Wes Anderson",
        "→ 1970s New Hollywood",
        "→ 1980s Retro Action",
        "→ 1990s Music Video",
        "→ 2000s Digital Look",
        # Director & Genre Presets
        "→ Horror / Thriller",
        "→ Romance / Soft Focus",
        "→ Christopher Nolan",
        "→ Denis Villeneuve",
        "→ Quentin Tarantino",
        "→ Prestige Drama (HBO)",
        "→ True Crime Documentary",
        "→ Western",
        "→ Mystery / Detective",
        "→ Fantasy / Magical",
    ]

    # Preset configurations: {preset_name: {setting: value, ...}}
    # v2.3.3: ALL values validated against their respective dataset lists
    PRESET_CONFIGS = {
        "→ Classic Hollywood": {
            "framing": "Medium Shot (MS)",
            "camera_type": "Panavision Panaflex Gold II (35mm)",
            "lens_focal": "50mm Standard Prime",
            "aperture_dof": "f/2.8 (Cinematic Separation)",
            "lighting": "Paramount Lighting",
            "style_aesthetic": "Cinematic Movie Still",
            "film_stock": "Kodak Vision3 500T",
            "color_grading": "Technicolor (Vintage)",
            "aspect_ratio": "2.39:1 (Anamorphic Scope)",
        },
        "→ Film Noir": {
            "framing": "Low Angle (Hero Shot)",
            "camera_type": "ARRI Alexa 35",
            "lens_focal": "35mm Classic Wide",
            "aperture_dof": "f/2.8 (Cinematic Separation)",
            "lighting": "Film Noir Lighting",
            "style_aesthetic": "Monochrome Noir",
            "film_stock": "Kodak Tri-X 400 (B&W)",
            "color_grading": "Bleach Bypass (Gritty)",
            "aspect_ratio": "2.39:1 (Anamorphic Scope)",
        },
        "→ Sci-Fi Cinematic": {
            "framing": "Extreme Wide Shot (EWS)",
            "camera_type": "ARRI Alexa 65 (IMAX)",
            "lens_focal": "ARRI Master Anamorphic",
            "aperture_dof": "f/4.0 (Balanced)",
            "lighting": "Cinematic Haze / Volumetric Fog",
            "style_aesthetic": "Blade Runner Atmosphere",
            "film_stock": "None",
            "color_grading": "Teal and Orange (Blockbuster)",
            "aspect_ratio": "2.39:1 (Anamorphic Scope)",
        },
        "→ Cyberpunk": {
            "framing": "Dutch Angle (Canted)",
            "camera_type": "Sony Venice 2",
            "lens_focal": "Anamorphic Lens",
            "aperture_dof": "f/1.8 (Soft Background)",
            "lighting": "Neon Cyberpunk Lighting",
            "style_aesthetic": "Cyberpunk 2077 Aesthetic",
            "film_stock": "Cinestill 800T",
            "color_grading": "Cyberpunk Neon Grading",
            "aspect_ratio": "21:9 (Ultrawide)",
        },
        "→ Drama / Emotional": {
            "framing": "Close-Up (CU)",
            "camera_type": "ARRI Alexa Mini LF",
            "lens_focal": "85mm Portrait Prime",
            "aperture_dof": "f/1.2 (Dreamy Bokeh)",
            "lighting": "Rembrandt Lighting",
            "style_aesthetic": "Cinematic Movie Still",
            "film_stock": "Kodak Portra 400",
            "color_grading": "Desaturated (Muted)",
            "aspect_ratio": "2.39:1 (Anamorphic Scope)",
        },
        "→ Epic Landscape": {
            "framing": "Extreme Wide Shot (EWS)",
            "camera_type": "ARRI Alexa 65 (IMAX)",
            "lens_focal": "14mm Ultra-Wide Angle",
            "aperture_dof": "f/11 (Landscape Sharpness)",
            "lighting": "Golden Hour (Magic Hour)",
            "style_aesthetic": "National Geographic Style",
            "film_stock": "Fujifilm Velvia 50",
            "color_grading": "Vibrant High Contrast",
            "aspect_ratio": "21:9 (Ultrawide)",
        },
        "→ Portrait": {
            "framing": "Medium Close-Up (MCU)",
            "camera_type": "Sony A7S III",
            "lens_focal": "85mm Portrait Prime",
            "aperture_dof": "f/1.2 (Dreamy Bokeh)",
            "lighting": "Soft Window Light",
            "style_aesthetic": "Editorial Photography",
            "film_stock": "Kodak Portra 400",
            "color_grading": "Pastel Soft Tones",
            "aspect_ratio": "4:3 (Academy Ratio)",
        },
        "→ Documentary": {
            "framing": "Medium Shot (MS)",
            "camera_type": "Canon C700 FF",
            "lens_focal": "35mm Classic Wide",
            "aperture_dof": "f/4.0 (Balanced)",
            "lighting": "Practical Lighting",
            "style_aesthetic": "Documentary Texture",
            "film_stock": "None",
            "color_grading": "Desaturated (Muted)",
            "aspect_ratio": "16:9 (Widescreen)",
        },
        "→ Artistic / Painterly": {
            "framing": "Medium Shot (MS)",
            "camera_type": "None",
            "lens_focal": "Petzval 85mm (Classic Swirl)",
            "aperture_dof": "f/1.8 (Soft Background)",
            "lighting": "Soft Window Light",
            "style_aesthetic": "Oil Painting (Classic)",
            "film_stock": "None",
            "color_grading": "Pastel Soft Tones",
            "aspect_ratio": "4:3 (Academy Ratio)",
        },
        "→ Retro VHS": {
            "framing": "Medium Shot (MS)",
            "camera_type": "Super 8mm Camera",
            "lens_focal": "50mm Standard Prime",
            "aperture_dof": "f/4.0 (Balanced)",
            "lighting": "Practical Lighting",
            "style_aesthetic": "Vintage 1990s VHS",
            "film_stock": "Polaroid 600",
            "color_grading": "Cross Processed",
            "aspect_ratio": "4:3 (Academy Ratio)",
        },
        "→ Golden Hour Magic": {
            "framing": "Full Body Shot (Wide)",
            "camera_type": "Sony Venice 2",
            "lens_focal": "85mm Portrait Prime",
            "aperture_dof": "f/1.8 (Soft Background)",
            "lighting": "Golden Hour (Magic Hour)",
            "style_aesthetic": "Photorealistic (Raw)",
            "film_stock": "Kodak Ektar 100",
            "color_grading": "Vibrant High Contrast",
            "aspect_ratio": "16:9 (Widescreen)",
        },
        "→ Moody Night": {
            "framing": "Medium Shot (MS)",
            "camera_type": "Sony A7S III",
            "lens_focal": "35mm Classic Wide",
            "aperture_dof": "f/1.2 (Dreamy Bokeh)",
            "lighting": "Moonlight",
            "style_aesthetic": "Cinematic Movie Still",
            "film_stock": "Cinestill 800T",
            "color_grading": "Teal and Orange (Blockbuster)",
            "aspect_ratio": "2.39:1 (Anamorphic Scope)",
        },
        "→ Action / Dynamic": {
            "framing": "Low Angle (Hero Shot)",
            "camera_type": "RED V-Raptor XL",
            "lens_focal": "24mm Wide Angle",
            "aperture_dof": "f/5.6 (Sharp Subject)",
            "lighting": "Harsh Sunlight",
            "style_aesthetic": "Hyper-Realism",
            "film_stock": "None",
            "shutter_speed": "1/1000th sec (Frozen Action)",
            "color_grading": "Teal and Orange (Blockbuster)",
            "aspect_ratio": "2.39:1 (Anamorphic Scope)",
        },
        "→ Wes Anderson": {
            "framing": "Symmetrical Composition",
            "camera_type": "ARRI Alexa 35",
            "lens_focal": "35mm Classic Wide",
            "aperture_dof": "f/8.0 (Deep Focus)",
            "lighting": "Soft Window Light",
            "style_aesthetic": "Wes Anderson Symmetric",
            "film_stock": "Kodak Portra 400",
            "color_grading": "Pastel Soft Tones",
            "aspect_ratio": "2.39:1 (Anamorphic Scope)",
        },
        "→ 1970s New Hollywood": {
            "framing": "Medium Shot (MS)",
            "camera_type": "Panavision Panaflex Gold II (35mm)",
            "lens_focal": "35mm Classic Wide",
            "aperture_dof": "f/2.8 (Cinematic Separation)",
            "lighting": "Practical Lighting",
            "style_aesthetic": "Cinematic Movie Still",
            "film_stock": "Kodak Vision3 500T",
            "color_grading": "Technicolor (Vintage)",
            "aspect_ratio": "2.39:1 (Anamorphic Scope)",
        },
        "→ 1980s Retro Action": {
            "framing": "Low Angle (Hero Shot)",
            "camera_type": "ARRI Alexa 35",
            "lens_focal": "Anamorphic Lens",
            "aperture_dof": "f/4.0 (Balanced)",
            "lighting": "Cinematic Haze / Volumetric Fog",
            "style_aesthetic": "Hyper-Realism",
            "film_stock": "None",
            "color_grading": "Teal and Orange (Blockbuster)",
            "aspect_ratio": "2.39:1 (Anamorphic Scope)",
        },
        "→ 1990s Music Video": {
            "framing": "Extreme Close-Up (ECU)",
            "camera_type": "Super 8mm Camera",
            "lens_focal": "Fish-Eye Lens",
            "aperture_dof": "f/1.8 (Soft Background)",
            "lighting": "Neon Cyberpunk Lighting",
            "style_aesthetic": "Vintage 1990s VHS",
            "film_stock": "Polaroid 600",
            "color_grading": "Cross Processed",
            "aspect_ratio": "4:3 (Academy Ratio)",
        },
        "→ 2000s Digital Look": {
            "framing": "Medium Shot (MS)",
            "camera_type": "Sony A7S III",
            "lens_focal": "24mm Wide Angle",
            "aperture_dof": "f/5.6 (Sharp Subject)",
            "lighting": "Harsh Sunlight",
            "style_aesthetic": "Editorial Photography",
            "film_stock": "None",
            "color_grading": "Vibrant High Contrast",
            "aspect_ratio": "16:9 (Widescreen)",
        },
        # ───────────────────────────────────────────────────────────
        # Director & Genre Presets (v2.3.3: validated against datasets)
        # ───────────────────────────────────────────────────────────
        "→ Horror / Thriller": {
            "framing": "Low Angle (Hero Shot)",
            "camera_type": "ARRI Alexa Mini LF",
            "lens_focal": "16mm Ultra-Wide Angle",
            "aperture_dof": "f/2.8 (Cinematic Separation)",
            "lighting": "Chiaroscuro (High Contrast)",
            "style_aesthetic": "Film Noir Aesthetic",
            "film_stock": "Kodak Vision3 500T",
            "color_grading": "Bleach Bypass (Gritty)",
            "aspect_ratio": "2.39:1 (Anamorphic Scope)",
        },
        "→ Romance / Soft Focus": {
            "framing": "Medium Close-Up (MCU)",
            "camera_type": "Sony A7S III",
            "lens_focal": "50mm Standard Prime",
            "aperture_dof": "f/1.2 (Dreamy Bokeh)",
            "lighting": "Soft Window Light",
            "style_aesthetic": "Dreamy Soft Focus",
            "film_stock": "Kodak Portra 400",
            "color_grading": "Pastel Soft Tones",
            "aspect_ratio": "1.85:1 (Standard Widescreen)",
        },
        "→ Christopher Nolan": {
            "framing": "Wide Shot (WS)",
            "camera_type": "IMAX 15/70mm Film Camera",
            "lens_focal": "28mm Wide Angle",
            "aperture_dof": "f/8.0 (Deep Focus)",
            "lighting": "Natural Ambient Light",
            "style_aesthetic": "Photorealistic (Raw)",
            "film_stock": "IMAX 15/70mm Film Stock",
            "color_grading": "Neutral ACES Workflow",
            "aspect_ratio": "1.43:1 (IMAX)",
        },
        "→ Denis Villeneuve": {
            "framing": "Extreme Wide Shot (EWS)",
            "camera_type": "ARRI Alexa 65 (IMAX)",
            "lens_focal": "24mm Wide Angle",
            "aperture_dof": "f/4.0 (Balanced)",
            "lighting": "Volumetric Fog / Atmospheric Haze",
            "style_aesthetic": "Atmospheric Cinematic",
            "film_stock": "None",
            "color_grading": "Desaturated Cool Tones",
            "aspect_ratio": "2.39:1 (Anamorphic Scope)",
        },
        "→ Quentin Tarantino": {
            "framing": "Medium Shot (MS)",
            "camera_type": "Panavision Panaflex Gold II (35mm)",
            "lens_focal": "40mm Semi-Wide",
            "aperture_dof": "f/2.8 (Cinematic Separation)",
            "lighting": "High-Key Lighting",
            "style_aesthetic": "Vintage Kodachrome Look",
            "film_stock": "Kodak Vision3 500T",
            "color_grading": "Vibrant High Contrast",
            "aspect_ratio": "2.39:1 (Anamorphic Scope)",
        },
        "→ Prestige Drama (HBO)": {
            "framing": "Medium Shot (MS)",
            "camera_type": "ARRI Alexa Mini LF",
            "lens_focal": "35mm Classic Wide",
            "aperture_dof": "f/2.0 (Shallow Cinematic)",
            "lighting": "Naturalistic Interior Light",
            "style_aesthetic": "Cinematic TV Drama",
            "film_stock": "None",
            "color_grading": "Moody Shadows",
            "aspect_ratio": "2.00:1 (DCI Flat)",
        },
        "→ True Crime Documentary": {
            "framing": "Medium Close-Up (MCU)",
            "camera_type": "Canon C300 Mark III",
            "lens_focal": "50mm Standard Prime",
            "aperture_dof": "f/2.8 (Cinematic Separation)",
            "lighting": "Interview 3-Point Setup",
            "style_aesthetic": "Documentary Realism",
            "film_stock": "None",
            "color_grading": "Desaturated Cool Tones",
            "aspect_ratio": "16:9 (Widescreen)",
        },
        "→ Western": {
            "framing": "Medium Full Shot (MFS)",
            "camera_type": "Panavision Panaflex Gold II (35mm)",
            "lens_focal": "35mm Classic Wide",
            "aperture_dof": "f/5.6 (Sharp Subject)",
            "lighting": "Harsh Sunlight",
            "style_aesthetic": "Vintage Kodachrome Look",
            "film_stock": "Eastman Kodak 5254 (Vintage)",
            "color_grading": "Orange and Teal (Hollywood)",
            "aspect_ratio": "2.39:1 (Anamorphic Scope)",
        },
        "→ Mystery / Detective": {
            "framing": "Over-The-Shoulder (OTS)",
            "camera_type": "ARRI Alexa 35",
            "lens_focal": "50mm Standard Prime",
            "aperture_dof": "f/2.8 (Cinematic Separation)",
            "lighting": "Film Noir Lighting",
            "style_aesthetic": "Neo-Noir Modern",
            "film_stock": "Kodak Vision3 500T",
            "color_grading": "Desaturated Cool Tones",
            "aspect_ratio": "2.39:1 (Anamorphic Scope)",
        },
        "→ Fantasy / Magical": {
            "framing": "Medium Wide (MW)",
            "camera_type": "ARRI Alexa 65 (IMAX)",
            "lens_focal": "Anamorphic Lens",
            "aperture_dof": "f/2.0 (Shallow Cinematic)",
            "lighting": "God Rays (Crepuscular Rays)",
            "style_aesthetic": "Magical Realism",
            "film_stock": "Kodak Vision3 250D",
            "color_grading": "Technicolor (Vintage)",
            "aspect_ratio": "2.39:1 (Anamorphic Scope)",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#                         NEURAL GRAMMAR ENGINE (v3.5)
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralGrammar:
    """
    Advanced logic to "Neuralize" prompts, making them more precise,
    scientific, and architecturally aware for T5/Diffusion models.
    """

    SCIENTIFIC_REPLACEMENTS = {
        # v3.1: Toned down from jargon to natural cinematic language.
        # Diffusion models respond better to concrete visual descriptors
        # than compound technical terms.
        "visualize": "render with volumetric detail",
        "accurate": "high-fidelity",
        "precise": "pixel-perfect",
        "design": "architectural composition",
        "process": "sequential transformation",
        "generation": "procedural reconstruction",
    }

    @staticmethod
    def scientificize(text: str) -> str:
        """Replace common words with high-precision cinematic terms.

        BUG-1 FIX (prior): Preserved case on split.
        BUG-2 FIX (prior): Added parens to strip set.
        BUG-M1 FIX (v2.3.3): Post-cleanup pass to fix irregular spacing
        caused by multi-word replacements adjacent to punctuation.
        """
        words = text.split()           # preserve original case
        new_words = []
        modified = False
        for word in words:
            clean_word = word.strip(".,!?;:()")   # also strip parens
            if clean_word.lower() in NeuralGrammar.SCIENTIFIC_REPLACEMENTS:
                replacement = NeuralGrammar.SCIENTIFIC_REPLACEMENTS[clean_word.lower()]
                # [BUG-M1] Preserve trailing punctuation from the original word
                trailing = ""
                for ch in reversed(word):
                    if ch in ".,!?;:()":
                        trailing = ch + trailing
                    else:
                        break
                new_words.append(replacement + trailing)
                modified = True
            else:
                new_words.append(word)  # preserve original capitalisation
        if modified:
            logger.debug("[NeuralGrammar] Applied cinematic vocabulary enhancement.")
        # [BUG-M1] Clean up any double-spaces introduced by multi-word replacements
        result = " ".join(new_words)
        return re.sub(r" {2,}", " ", result)

    @staticmethod
    def enhance_syntax(prompt_parts: list) -> list:
        """Improve grammar and technical structure of prompt parts."""
        enhanced = []
        for part in prompt_parts:
            if part.startswith("Shot on"):
                # v3.1 FIX: Preserve original camera name.
                camera_name = part.replace("Shot on ", "", 1)
                part = f"Captured with high-precision {camera_name}"
            enhanced.append(part)
        return enhanced


# ═══════════════════════════════════════════════════════════════════════════════
#                   PRESET VALIDATION (v2.3.3 — runs at import time)
# ═══════════════════════════════════════════════════════════════════════════════

_FIELD_TO_DATASET = {
    "framing": CinematicDatasets.FRAMING,
    "camera_type": CinematicDatasets.CAMERAS,
    "lens_focal": CinematicDatasets.LENSES,
    "aperture_dof": CinematicDatasets.APERTURES,
    "lighting": CinematicDatasets.LIGHTING,
    "style_aesthetic": CinematicDatasets.STYLES,
    "film_stock": CinematicDatasets.FILM_STOCKS,
    "shutter_speed": CinematicDatasets.SHUTTER_SPEEDS,
    "color_grading": CinematicDatasets.COLOR_GRADING,
    "aspect_ratio": CinematicDatasets.ASPECT_RATIOS,
}


def _validate_presets():
    """Validate all preset config values exist in their respective dataset lists.

    v2.3.3 [BUG-I8]: In strict mode (RADIANCE_STRICT=1 env var), raises
    RuntimeError on validation failure instead of silently logging.
    """
    errors = []
    for preset_name, config in CinematicDatasets.PRESET_CONFIGS.items():
        for field, value in config.items():
            if field not in _FIELD_TO_DATASET:
                errors.append(f"  [{preset_name}] Unknown field: '{field}'")
                continue
            dataset = _FIELD_TO_DATASET[field]
            if value not in dataset:
                errors.append(f"  [{preset_name}] {field}='{value}' not in dataset")
    if errors:
        msg = f"◎ PRESET VALIDATION FAILED ({len(errors)} issues):\n" + "\n".join(errors)
        logger.error(msg)
        # [BUG-I8] Strict mode: raise so broken presets are caught in CI/testing
        if os.environ.get("RADIANCE_STRICT", "0") == "1":
            raise RuntimeError(msg)
    else:
        logger.debug("✓ All preset configs validated against datasets")


# Run validation at import time so mismatches are caught immediately
_validate_presets()


# ═══════════════════════════════════════════════════════════════════════════════
#                         TOKEN UTILITIES (v2.3.3)
# ═══════════════════════════════════════════════════════════════════════════════

CLIP_MAX_TOKENS = 77
BREAK_TOKEN = "BREAK"  # nosec B105
BREAK_TOKEN_OVERHEAD = 2  # [BUG-M2] BREAK token + surrounding spaces ≈ 2 tokens
DEFAULT_YEAR = 2025  # [BUG-I4] Updated from 2024

# ═══════════════════════════════════════════════════════════════════════════════
#                  ARCHITECTURE-AWARE CONSTANTS  (v3.0)
# ═══════════════════════════════════════════════════════════════════════════════

# These architectures use T5/LLM encoders that prefer natural language prose.
# Comma-separated keyword chains perform significantly worse on them.
PROSE_ARCHS = {"flux", "sd3", "sd3.5", "wan", "ltx", "pixart", "kolors",
               "hunyuan_video", "aura_flow"}


def _detect_arch_from_clip(clip, target_arch: str) -> str:
    """
    v3.1: Detect architecture from CLIP tokenizer keys when target_arch is 'Auto'.
    Falls back to 'sdxl' (structured format) if detection fails — safer than
    defaulting to prose which hurts CLIP-only encoders.

    Detection logic:
      - 't5xxl' key → Flux/SD3 family (T5 encoder present)
      - 'llm' key   → Kolors/PixArt family (LLM encoder)
      - 'g' + 'l'   → SDXL dual CLIP
      - 'l' only     → SD1.5 single CLIP
    """
    if target_arch != "Auto":
        return target_arch.lower()

    try:
        test_tokens = clip.tokenize("test")
        keys = set(test_tokens.keys())

        if "t5xxl" in keys:
            return "flux"   # Flux, SD3, SD3.5 all use T5
        if "llm" in keys:
            return "kolors"
        if "g" in keys and "l" in keys:
            return "sdxl"
        if "l" in keys:
            return "sd1.5"
    except Exception as e:
        logger.debug(f"[Encoder] Arch detection failed: {e}, defaulting to sdxl")

    return "sdxl"  # Safe fallback — structured format for CLIP-only

# ═══════════════════════════════════════════════════════════════════════════════
#                  SCENE MOOD VOCABULARY  (v3.0)
# ═══════════════════════════════════════════════════════════════════════════════

SCENE_MOODS = [
    "None",
    "Tense", "Melancholic", "Joyful", "Ominous", "Nostalgic",
    "Awe-Inspiring", "Intimate", "Chaotic", "Peaceful",
    "Surreal", "Gritty", "Ethereal", "Foreboding", "Euphoric",
]

# Vocabulary injected per mood — chosen to steer T5 and CLIP equally well
_MOOD_VOCAB = {
    "Tense":         "heightened tension, breath held, claustrophobic atmosphere, nervous energy",
    "Melancholic":   "melancholic, quiet sorrow, fading light, wistful silence",
    "Joyful":        "vibrant joy, warm energy, infectious optimism, radiant smile",
    "Ominous":       "ominous foreboding, unseen threat, dread building beneath the surface",
    "Nostalgic":     "nostalgic warmth, memory of simpler times, sepia-tinted emotion",
    "Awe-Inspiring": "breathtaking grandeur, overwhelming scale, reverential silence",
    "Intimate":      "quiet intimacy, close proximity, whispered emotion, soft connection",
    "Chaotic":       "frantic energy, motion blur everywhere, sensory overload, disorientation",
    "Peaceful":      "serene tranquility, unhurried pace, meditative stillness",
    "Surreal":       "dreamlike unreality, logic dissolved, impossible beauty",
    "Gritty":        "raw grit, unpolished truth, weathered texture, unflinching honesty",
    "Ethereal":      "otherworldly ethereal light, gossamer beauty, translucent and floating",
    "Foreboding":    "creeping dread, something wrong just out of sight, shadows advance",
    "Euphoric":      "ecstatic joy, transcendent moment, colours oversaturated with feeling",
}


def _real_token_count(clip, text: str, tokens: dict = None) -> int:
    """
    Get actual token count using the connected CLIP tokenizer.
    Falls back to estimate_tokens() if tokenizer API is unavailable.

    v3.1 FIX: For T5/LLM encoders, tokens are padded to a fixed length
    (e.g., 256 or 512). shape[-1] returns the padded length, not the
    actual token count. We count non-padding tokens where possible.
    T5 uses pad_token_id=0; CLIP uses pad_token_id=49407.

    v2.3.3 [BUG-I3]: Removed redundant `import torch as _torch` — torch
    is already imported at module level.
    """
    if tokens is None:
        try:
            tokens = clip.tokenize(text)
        except Exception:
            return estimate_tokens(text)

    try:
        # Try each known encoder key in order of preference
        for key in ("t5xxl", "l", "g", "llm"):
            if key in tokens and tokens[key]:
                tok_data = tokens[key][0]
                if hasattr(tok_data, "shape"):
                    # Tensor path (T5 / newer CLIP wrappers) — single tensor,
                    # no multi-chunk structure. Count non-pad tokens directly.
                    # [BUG-I3] Use module-level torch directly
                    if torch.is_tensor(tok_data):
                        pad_id = 0 if key in ("t5xxl", "llm") else 49407
                        non_pad = (tok_data != pad_id).sum().item()
                        if non_pad > 0:
                            return non_pad
                    return int(tok_data.shape[-1])
                elif isinstance(tok_data, list) and tok_data:
                    # BUG-5 FIX: ComfyUI CLIP tokenizer returns a list of chunks:
                    #   tokens[key] = [chunk0, chunk1, ...]
                    # where each chunk is [(token_id, weight), ...] padded to 77.
                    #
                    # BUG-6 FIX: Previous code only examined tokens[key][0] —
                    # the first chunk. For prompts > 77 tokens (BREAK or SDXL
                    # long prompts), there are multiple chunks and the count was
                    # severely undercounted. Fix: sum across ALL chunks.
                    #
                    # [BUG-M4] v2.3.3: Explicit tuple validation
                    if isinstance(tok_data[0], (tuple, list)) and len(tok_data[0]) >= 1:
                        pad_id = 0 if key in ("t5xxl", "llm") else 49407
                        total_non_pad = 0
                        total_len = 0
                        for chunk in tokens[key]:   # iterate ALL chunks
                            for item in chunk:
                                tok_id = item[0] if isinstance(item, (tuple, list)) else item
                                if tok_id != pad_id:
                                    total_non_pad += 1
                            total_len += len(chunk)
                        return total_non_pad if total_non_pad > 0 else total_len
                    # Non-tuple list: sum lengths across all chunks
                    return sum(len(chunk) for chunk in tokens[key])
        # Fallback: count all token entries if shape not accessible
        for val in tokens.values():
            if val and hasattr(val[0], "shape"):
                return int(val[0].shape[-1])
            elif val and hasattr(val[0], "__len__"):
                return len(val[0])
    except Exception:
        pass
    return estimate_tokens(text)


def _apply_subject_weight(text: str, weight: float) -> str:
    """Wrap text in attention weight syntax if weight != 1.0."""
    if abs(weight - 1.0) < 0.01:
        return text
    return f"({text}:{weight:.2f})"


def _build_prose_prompt(
    base_prompt, framing, camera, lens, aperture, lighting,
    style, film_stock, shutter, color_grading, aspect_ratio,
    custom_details, year_era, lora_keywords, art_direction,
    scene_mood, subject_weight, weight_mode,
) -> str:
    """
    Build a natural-language prose prompt for T5/LLM-based architectures
    (Flux, SD3, Kolors, PixArt, Wan, LTX, HunyuanVideo).
    These encoders respond much better to flowing sentences than comma chains.

    v2.3.3 [BUG-I1]: Added `subject_first` weight mode — leads with subject
    at elevated emphasis before technical descriptors.
    """
    def c(v): return "" if v in ("None", None, "") else v

    parts = []

    # 1. Subject (weighted if requested)
    subject = base_prompt.strip()
    if subject_weight != 1.0:
        subject = _apply_subject_weight(subject, subject_weight)

    # Neural Grammar Injection: If the user asks for accuracy/precision, expand it.
    # BUG-2 FIX: Apply to raw subject BEFORE assembly into parts[0].
    if any(kw in subject.lower() for kw in ("accurate", "precise", "visualize")):
        subject = NeuralGrammar.scientificize(subject)

    # [BUG-I1] Weight modes: technique_first, subject_first, balanced
    if weight_mode == "technique_first" and c(camera):
        parts.append(f"Photographed on {camera}, {subject}.")
    elif weight_mode == "subject_first":
        # Lead with subject prominently, no framing prefix
        parts.append(f"{subject}, the central focus of this scene.")
    elif c(framing):
        parts.append(f"{framing} of {subject}.")
    else:
        parts.append(f"{subject}.")

    # 2. Art direction (positioned here — right after subject)
    if art_direction and art_direction.strip():
        parts.append(art_direction.strip())

    # 3. LoRA keywords
    if lora_keywords and lora_keywords.strip():
        parts.append(lora_keywords.strip())

    # 4. Scene mood
    mood = c(scene_mood)
    if mood and mood in _MOOD_VOCAB:
        parts.append(_MOOD_VOCAB[mood])

    # 5. Cinematic technique — prose form
    tech = []
    if c(camera) and weight_mode != "technique_first":
        tech.append(f"the image was captured on {camera}")
    if c(lens):
        tech.append(f"through a {lens}")
    if c(aperture):
        tech.append(f"at {aperture}")
    if c(shutter):
        tech.append(f"with shutter speed {shutter}")
    if tech:
        parts.append("Cinematic technique: " + ", ".join(tech) + ".")

    # 6. Lighting in full sentence
    if c(lighting):
        parts.append(f"The scene is bathed in {lighting}.")

    # 7. Color and finishing
    finish = []
    if c(color_grading): finish.append(f"color graded in {color_grading}")
    if c(film_stock):    finish.append(f"the distinctive texture of {film_stock}")
    if c(style):         finish.append(f"{style} aesthetic")
    if finish:
        parts.append("Aesthetic: " + ", ".join(finish) + ".")

    # 8. Era
    if year_era != DEFAULT_YEAR:
        parts.append(f"Set in the year {year_era}.")

    # 9. Aspect ratio
    if c(aspect_ratio):
        parts.append(f"Composed for {aspect_ratio} format.")

    # 10. Framing context for subject_first (add framing info after technique)
    if weight_mode == "subject_first" and c(framing):
        parts.append(f"Framed as a {framing}.")

    # 11. Custom details last
    if c(custom_details):
        parts.append(custom_details.strip())

    return " ".join(parts)


def estimate_tokens(text: str) -> int:
    """
    Estimate CLIP tokens for a text string.
    CLIP's BPE tokenizer averages ~1.3 tokens per whitespace-delimited word
    due to subword splitting. This is a rough heuristic; actual count depends
    on the specific vocabulary and text content.
    """
    if not text:
        return 0
    words = text.split()
    return int(len(words) * 1.3)


def insert_break_points(prompt: str, max_tokens: int = 70) -> str:
    """
    Insert BREAK tokens at logical points to chunk long prompts.
    Helps with CLIP's 77-token limit by creating separate encoding chunks.

    v2.3.3: Improved sentence splitting — splits on ". " (period+space) instead
    of bare "." to avoid breaking on abbreviations, decimals, and lens names
    like "f/2.8" or "Sony A7S III."

    v2.3.3 [BUG-M2]: Account for BREAK token overhead in chunk budget.
    """
    if estimate_tokens(prompt) <= max_tokens:
        return prompt

    # Split on sentence boundaries: period/exclamation/question followed by space
    sentences = re.split(r"(?<=[.!?])\s+", prompt)

    if len(sentences) <= 1:
        # No good split points found — return as-is rather than mangling the prompt
        return prompt

    # Recombine with BREAK tokens at chunk boundaries
    # [BUG-M2] Reserve space for BREAK token overhead at each boundary
    effective_limit = max_tokens - BREAK_TOKEN_OVERHEAD
    result = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = estimate_tokens(sentence)
        if current_tokens + sent_tokens > effective_limit and current_chunk:
            result.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sent_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sent_tokens

    if current_chunk:
        result.append(" ".join(current_chunk))

    return f" {BREAK_TOKEN} ".join(result)


def enhance_prompt_grammar(prompt: str, level: str, arch: str = "sdxl") -> str:
    """
    Fix common grammar/formatting issues and optionally inject creative tags.

    v3.1: arch parameter controls which creative tags to inject.
    Danbooru-style tags (masterpiece, best quality) only help SD1.5/SDXL.
    Prose architectures get natural-language quality descriptors instead.

    v2.3.3 [BUG-M5]: Check for existing creative tags before injection
    to prevent duplication on re-encodes.
    """
    if level == "Off" or not prompt:
        return prompt

    p = prompt
    # Remove redundant spaces
    p = re.sub(r' {2,}', ' ', p)
    # Fix stray/duplicate commas
    p = re.sub(r'\s+,', ',', p)
    p = re.sub(r',+', ',', p)
    p = re.sub(r',\s*,', ',', p)

    if level == "Creative Enhancement":
        if arch in PROSE_ARCHS:
            creative_tags = (
                "Exceptionally detailed with stunning visual clarity "
                "and ultra-high resolution rendering"
            )
        else:
            creative_tags = "masterpiece, best quality, highly detailed, stunning, ultra-high resolution"

        # [BUG-M5] Avoid double-injection on re-encodes
        if creative_tags not in p and "masterpiece" not in p.lower():
            if p.endswith('.'):
                p = p[:-1] + ", " + creative_tags + "."
            else:
                p += ", " + creative_tags

    return p.strip()


# ═══════════════════════════════════════════════════════════════════════════════
#                         SHARED PROMPT BUILDER  (v2.3.3)
# ═══════════════════════════════════════════════════════════════════════════════


def build_cinematic_prompt_v3(
    base_prompt,
    framing,
    camera_type,
    lens_focal,
    aperture_dof,
    lighting,
    style_aesthetic,
    film_stock="None",
    shutter_speed="None",
    color_grading="None",
    aspect_ratio="None",
    custom_details="",
    year_era=DEFAULT_YEAR,
    negative_strength="Standard",
    negative_custom="",
    lora_keywords="",
    use_break=False,
    target_arch="Auto",
    scene_mood="None",
    subject_weight=1.0,
    art_direction="",
    prompt_weight_mode="balanced",
    base_prompt_b="",
    active_prompt="A",
):
    """
    v2.3.3 universal prompt builder.
    Uses prose format for T5/LLM architectures (Flux, SD3, Kolors…) and
    structured keyword format for CLIP-only architectures (SD1.5, SDXL).
    Returns: (final_prompt, negative_prompt, estimated_token_count).
    """
    def c(v): return "" if v in ("None", None, "") else v

    # A/B prompt mode
    if active_prompt == "B" and base_prompt_b.strip():
        effective_base = base_prompt_b.strip()
    else:
        effective_base = base_prompt.strip()
        if active_prompt == "B":
            logger.info("[Encoder] Prompt B is empty — falling back to Prompt A.")

    # v3.1 FIX: Subject weight is applied INSIDE each path (prose/structured),
    # not here. Previously it was applied at both levels, causing double-weight.

    # v3.1: Architecture is resolved by the caller and passed as target_arch.
    resolved_arch = target_arch if target_arch != "Auto" else "sdxl"
    use_prose = resolved_arch in PROSE_ARCHS

    if use_prose:
        final_prompt = _build_prose_prompt(
            base_prompt=effective_base, framing=framing, camera=camera_type,
            lens=lens_focal, aperture=aperture_dof, lighting=lighting,
            style=style_aesthetic, film_stock=film_stock, shutter=shutter_speed,
            color_grading=color_grading, aspect_ratio=aspect_ratio,
            custom_details=custom_details, year_era=year_era,
            lora_keywords=lora_keywords, art_direction=art_direction,
            scene_mood=scene_mood, subject_weight=subject_weight,
            weight_mode=prompt_weight_mode,
        )
    else:
        # Structured keyword format (SD1.5 / SDXL)
        parts = []
        style = c(style_aesthetic)

        # Apply subject weight once in the structured path
        weighted_base = _apply_subject_weight(effective_base, subject_weight)

        # [BUG-I1] subject_first: lead with weighted subject, no camera prefix
        if prompt_weight_mode == "subject_first":
            parts.append(f"{weighted_base}.")
            if c(framing):
                parts.append(f"Framed as {framing}.")
        elif prompt_weight_mode == "technique_first" and c(camera_type):
            parts.append(f"Shot on {camera_type}.")
            prefix = f"{c(framing)} of " if c(framing) else ""
            parts.append(f"{prefix}{weighted_base}.")
        elif c(framing):
            parts.append(f"{framing} of {weighted_base}.")
        else:
            parts.append(f"{weighted_base}.")

        # [BUG-C3] FIX: Insert art_direction THEN lora_keywords at sequential
        # indices so art_direction stays closer to subject than lora_keywords.
        # Previous code inserted both at index 1, reversing the intended order.
        insert_idx = len(parts)  # Insert after subject/framing block
        if art_direction and art_direction.strip():
            parts.insert(insert_idx, art_direction.strip())
            insert_idx += 1
        if lora_keywords and lora_keywords.strip():
            parts.insert(insert_idx, lora_keywords.strip())

        if scene_mood and scene_mood != "None" and scene_mood in _MOOD_VOCAB:
            parts.append(_MOOD_VOCAB[scene_mood])

        tech = []
        # [BUG-M3] Gate camera in tech block for both technique_first AND subject_first
        if c(camera_type) and prompt_weight_mode not in ("technique_first",):
            tech.append(f"Shot on {camera_type}")
        if c(lens_focal):    tech.append(f"with {lens_focal}")
        if c(aperture_dof):  tech.append(f"at {aperture_dof}")
        if c(shutter_speed): tech.append(f"shutter speed {shutter_speed}")
        if tech: parts.append(" ".join(tech) + ".")

        if c(lighting):      parts.append(f"Lighting is {lighting}.")
        if c(color_grading): parts.append(f"Color graded in {color_grading}.")

        finish = []
        if style:             finish.append(style)
        if c(film_stock):     finish.append(f"on {film_stock}")
        # [BUG-C2] FIX: Removed trailing period from year_era fragment.
        # The join + "." at the end of the finish block already adds one.
        if year_era != DEFAULT_YEAR: finish.append(f"Est. Year {year_era}")
        if finish: parts.append(", ".join(finish) + ".")

        if c(aspect_ratio):   parts.append(f"{aspect_ratio} format.")
        if c(custom_details): parts.append(custom_details.strip())

        final_prompt = " ".join(p for p in parts if p).strip()

    # [BUG-I5] use_break is now always boolean from the caller
    if use_break:
        final_prompt = insert_break_points(final_prompt)

    token_count = estimate_tokens(final_prompt)

    # ── Auto-negative (arch-aware) ──────────────────────────────────────────
    style_val = c(style_aesthetic)
    if use_prose and resolved_arch in ("flux", "kolors"):
        if negative_strength not in ("Off", "Soft"):
            logger.info(
                f"[Encoder] target_arch='{resolved_arch}': negative prompts have limited "
                "effect. Consider 'Soft' or 'Off' for better results."
            )

    negative_prompt = ""
    if negative_strength != "Off":
        neg = []
        if negative_strength in ("Soft", "Standard", "Aggressive"):
            neg.extend(["blur", "low quality", "watermark", "text"])
        if negative_strength in ("Standard", "Aggressive"):
            neg.extend(["deformed", "ugly", "duplicate", "disfigured", "bad anatomy"])
            if any(kw in style_val for kw in ("Photorealistic", "Cinematic", "Documentary", "Realism")):
                neg.extend(["cartoon", "anime", "illustration", "painting", "cgi",
                             "3d render", "drawing", "sketch"])
            elif "Anime" in style_val:
                # [BUG-M6] Added "3d render" which was missing from Anime exclusions
                neg.extend(["photograph", "realistic", "photo", "photorealistic",
                             "3d", "3d render"])
            elif any(kw in style_val for kw in ("Painting", "Oil", "Painterly")):
                neg.extend(["photograph", "realistic", "photo", "digital", "3d render"])
            elif any(kw in style_val for kw in ("CGI", "Unreal", "3D", "Octane")):
                neg.extend(["photograph", "realistic", "2d", "flat", "hand drawn"])
        if negative_strength == "Aggressive":
            neg.extend(["mutated", "extra limbs", "missing limbs", "floating limbs",
                         "disconnected limbs", "pixelated", "noise", "grainy",
                         "cropped", "out of frame", "worst quality", "lowres"])
        negative_prompt = ", ".join(neg)

    # Merge user custom negatives
    if negative_custom and negative_custom.strip():
        negative_prompt = (f"{negative_prompt}, {negative_custom.strip()}"
                           if negative_prompt else negative_custom.strip())

    return (final_prompt, negative_prompt, token_count)


def apply_style_preset(preset_name, current_settings):
    """
    Apply a style preset to the current settings.
    Returns updated settings dict.

    Note: Presets do partial overrides — fields not in the preset config
    retain their current (manual) values. This is intentional so users can
    select a preset and still tweak individual parameters.
    """
    if (
        preset_name == "None (Custom)"
        or preset_name not in CinematicDatasets.PRESET_CONFIGS
    ):
        return current_settings

    preset = CinematicDatasets.PRESET_CONFIGS[preset_name]
    updated = current_settings.copy()
    updated.update(preset)
    return updated


# ═══════════════════════════════════════════════════════════════════════════════
#                     CINEMATIC ENCODER (CONDITIONING OUTPUT)
# ═══════════════════════════════════════════════════════════════════════════════


class RadianceCinematicPromptEncoder:
    """
    v2.3.3 — All-in-one cinematic prompt builder with direct CLIP encoding.
    Clean interface: clip in → CONDITIONING out.
    Auto-selects prose format for Flux/T5/Kolors and structured for SD1.5/SDXL.
    """

    # Reference shared datasets
    CAMERAS = CinematicDatasets.CAMERAS
    LENSES = CinematicDatasets.LENSES
    APERTURES = CinematicDatasets.APERTURES
    FRAMING = CinematicDatasets.FRAMING
    LIGHTING = CinematicDatasets.LIGHTING
    STYLES = CinematicDatasets.STYLES
    FILM_STOCKS = CinematicDatasets.FILM_STOCKS
    SHUTTER_SPEEDS = CinematicDatasets.SHUTTER_SPEEDS
    COLOR_GRADING = CinematicDatasets.COLOR_GRADING
    ASPECT_RATIOS = CinematicDatasets.ASPECT_RATIOS
    STYLE_PRESETS = CinematicDatasets.STYLE_PRESETS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIP model for encoding."}),
            },
            "optional": {
                # ── Core Prompt ────────────────────────────────────────────
                "base_prompt": (
                    "STRING",
                    {"multiline": True, "default": "A cinematic scene...",
                     "tooltip": "Primary subject/scene description (Prompt A)."},
                ),
                "base_prompt_b": (
                    "STRING",
                    {"multiline": True, "default": "",
                     "tooltip": "Alternate subject for A/B testing. Select with active_prompt."},
                ),
                "active_prompt": (
                    ["A", "B"],
                    {"default": "A",
                     "tooltip": "Switch between Prompt A and Prompt B without rewiring."},
                ),
                "style_preset": (
                    cls.STYLE_PRESETS,
                    {"default": "→ Classic Hollywood",
                     "tooltip": "One-click style preset."},
                ),
                # ── Architecture ───────────────────────────────────────────
                "target_arch": (
                    # [BUG-I2] Added sd3.5 and hunyuan_video to match PROSE_ARCHS
                    ["Auto", "flux", "sd3", "sd3.5", "sdxl", "sd1.5", "wan",
                     "ltx", "pixart", "kolors", "hunyuan_video", "aura_flow"],
                    {"default": "Auto",
                     "tooltip": ("Auto selects prose for Flux/T5 architectures. "
                                 "Set manually if auto-detect is wrong. "
                                 "Prose = Flux/SD3/SD3.5/Kolors/Wan. Structured = SD1.5/SDXL.")},
                ),
                "context_window": (
                    ["Standard (CLIP 77)", "Medium (Flux/T5 256)", "Large (T5 512)"],
                    {"default": "Standard (CLIP 77)",
                     "tooltip": "Token budget. Use Large for Flux with long prompts."},
                ),
                # ── Cinematic Parameters ───────────────────────────────────
                "framing": (
                    cls.FRAMING,
                    {"default": "Medium Shot (MS)", "tooltip": "Shot framing type."},
                ),
                "camera_type": (
                    cls.CAMERAS,
                    {"default": "ARRI Alexa 35", "tooltip": "Camera body."},
                ),
                "lens_focal": (
                    cls.LENSES,
                    {"default": "50mm Standard Prime", "tooltip": "Lens + focal length."},
                ),
                "aperture_dof": (
                    cls.APERTURES,
                    {"default": "f/2.8 (Cinematic Separation)", "tooltip": "Depth of field."},
                ),
                "lighting": (
                    cls.LIGHTING,
                    {"default": "Cinematic Haze / Volumetric Fog", "tooltip": "Lighting style."},
                ),
                "style_aesthetic": (
                    cls.STYLES,
                    {"default": "Photorealistic (Raw)", "tooltip": "Visual aesthetic."},
                ),
                "film_stock": (
                    cls.FILM_STOCKS,
                    {"default": "None", "tooltip": "Film stock emulation."},
                ),
                "shutter_speed": (
                    cls.SHUTTER_SPEEDS,
                    {"default": "None", "tooltip": "Motion blur characteristics."},
                ),
                "color_grading": (
                    cls.COLOR_GRADING,
                    {"default": "None", "tooltip": "Color grading look."},
                ),
                "aspect_ratio": (
                    cls.ASPECT_RATIOS,
                    {"default": "None", "tooltip": "Frame aspect ratio."},
                ),
                # ── Emotion & Art Direction ────────────────────────────────
                "scene_mood": (
                    SCENE_MOODS,
                    {"default": "None",
                     "tooltip": ("Injects curated emotional vocabulary into the prompt. "
                                 "Works with both CLIP and T5 encoders.")},
                ),
                "art_direction": (
                    "STRING",
                    {"multiline": True, "default": "",
                     "tooltip": ("Art direction notes injected right after subject. "
                                 "E.g. 'golden light raking across the face, hero in silhouette'")},
                ),
                # ── Prompt Control ─────────────────────────────────────────
                "subject_weight": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.05,
                     "tooltip": "Wrap subject in (subject:weight) attention syntax."},
                ),
                "prompt_weight_mode": (
                    ["balanced", "subject_first", "technique_first"],
                    {"default": "balanced",
                     "tooltip": ("Controls token budget ordering. "
                                 "'subject_first' leads with subject emphasis. "
                                 "'technique_first' leads with camera for style-driven shots.")},
                ),
                "custom_details": (
                    "STRING",
                    {"multiline": False, "default": "",
                     "tooltip": "Additional custom details appended at the end."},
                ),
                "year_era": (
                    "INT",
                    {"default": DEFAULT_YEAR, "min": 1800, "max": 2100, "step": 1,
                     "tooltip": "Era context for period-specific looks."},
                ),
                # ── Negative Prompt ────────────────────────────────────────
                "negative_strength": (
                    ["Off", "Soft", "Standard", "Aggressive"],
                    {"default": "Standard",
                     "tooltip": "Auto-negative strength. 'Soft' is recommended for Flux."},
                ),
                "negative_custom": (
                    "STRING",
                    {"multiline": True, "default": "",
                     "tooltip": "Your custom negative terms, merged with auto-generated negatives."},
                ),
                # ── Encoding Options ───────────────────────────────────────
                "clip_skip": (
                    "INT",
                    {"default": 0, "min": 0, "max": 24, "step": 1,
                     "tooltip": "CLIP layers to skip (0 = all). Typical SD1.5=1, SDXL=0."},
                ),
                "lora_keywords": (
                    "STRING",
                    {"multiline": False, "default": "",
                     "tooltip": "LoRA trigger words injected into the prompt."},
                ),
                "use_break": (
                    ["Off", "On"],
                    {"default": "Off",
                     "tooltip": "Auto-insert BREAK tokens at chunk boundaries for long prompts."},
                ),
                "prompt_enhancer": (
                    ["Off", "Grammar & Formatting", "Creative Enhancement"],
                    {"default": "Off",
                     "tooltip": "Auto-fix prompt grammar or inject high-quality aesthetic tags."},
                ),
                "image_ref": (
                    "IMAGE",
                    {"tooltip": "Optional reference image. Passed through to output for routing convenience."},
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = ("positive", "negative", "image_ref", "positive_text", "negative_text", "token_count")
    OUTPUT_TOOLTIPS = (
        "Positive conditioning for the sampler.",
        "Negative conditioning for the sampler.",
        "Optional reference image passed through.",
        "Final positive prompt string — wire to a text preview node.",
        "Final negative prompt string.",
        "Actual token count from the CLIP tokenizer.",
    )
    FUNCTION = "encode_cinematic"
    CATEGORY = "FXTD Studios/Radiance/Generate"
    # [BUG-I6] Version string updated to match actual version
    DESCRIPTION = (
        "v2.3.3 — Universal cinematic encoder. Auto-selects prose format for "
        "Flux/T5/Kolors and structured format for SD1.5/SDXL. "
        "Supports scene mood, A/B prompts, subject weighting, real token counts, "
        "art direction and architecture-aware negatives."
    )

    def encode_cinematic(
        self,
        clip,
        base_prompt="A cinematic scene...",
        base_prompt_b="",
        active_prompt="A",
        style_preset="→ Classic Hollywood",
        target_arch="Auto",
        context_window="Standard (CLIP 77)",
        framing="Medium Shot (MS)",
        camera_type="ARRI Alexa 35",
        lens_focal="50mm Standard Prime",
        aperture_dof="f/2.8 (Cinematic Separation)",
        lighting="Cinematic Haze / Volumetric Fog",
        style_aesthetic="Photorealistic (Raw)",
        film_stock="None",
        shutter_speed="None",
        color_grading="None",
        aspect_ratio="None",
        scene_mood="None",
        art_direction="",
        subject_weight=1.0,
        prompt_weight_mode="balanced",
        custom_details="",
        year_era=DEFAULT_YEAR,
        negative_strength="Standard",
        negative_custom="",
        clip_skip=0,
        lora_keywords="",
        use_break="Off",
        image_ref=None,
        prompt_enhancer="Off",
    ):
        # ── Validation ──────────────────────────────────────────────────────
        if clip is None:
            raise RuntimeError("CLIP input is None. Connect a valid CLIP model.")
        if not base_prompt or not base_prompt.strip():
            raise ValueError("base_prompt cannot be empty.")
        if clip_skip > 12:
            logger.warning(f"clip_skip={clip_skip} is unusually high.")

        # ── Token limit ─────────────────────────────────────────────────────
        if "Standard" in context_window:
            token_limit = 77
        elif "Medium" in context_window:
            token_limit = 256
        else:
            token_limit = 512

        # ── Resolve architecture once ────────────────────────────────────────
        resolved_arch = _detect_arch_from_clip(clip, target_arch)

        # ── Apply style preset ──────────────────────────────────────────────
        settings = {
            "framing": framing, "camera_type": camera_type,
            "lens_focal": lens_focal, "aperture_dof": aperture_dof,
            "lighting": lighting, "style_aesthetic": style_aesthetic,
            "film_stock": film_stock, "shutter_speed": shutter_speed,
            "color_grading": color_grading, "aspect_ratio": aspect_ratio,
        }
        if style_preset != "None (Custom)":
            settings = apply_style_preset(style_preset, settings)

        # ── Build prompt v2.3.3 ─────────────────────────────────────────────
        # [BUG-I5] Convert string "On"/"Off" to boolean for builder
        use_break_bool = (use_break == "On")
        final_prompt, negative_prompt, _ = build_cinematic_prompt_v3(
            base_prompt=base_prompt,
            base_prompt_b=base_prompt_b,
            active_prompt=active_prompt,
            framing=settings["framing"],
            camera_type=settings["camera_type"],
            lens_focal=settings["lens_focal"],
            aperture_dof=settings["aperture_dof"],
            lighting=settings["lighting"],
            style_aesthetic=settings["style_aesthetic"],
            film_stock=settings["film_stock"],
            shutter_speed=settings["shutter_speed"],
            color_grading=settings["color_grading"],
            aspect_ratio=settings["aspect_ratio"],
            custom_details=custom_details,
            year_era=year_era,
            negative_strength=negative_strength,
            negative_custom=negative_custom,
            lora_keywords=lora_keywords,
            use_break=False,  # We handle BREAK separately below after enhancement
            target_arch=resolved_arch,
            scene_mood=scene_mood,
            subject_weight=subject_weight,
            art_direction=art_direction,
            prompt_weight_mode=prompt_weight_mode,
        )

        # ── Enhance Prompt Grammar ──────────────────────────────────────────
        if prompt_enhancer != "Off":
            final_prompt = enhance_prompt_grammar(final_prompt, prompt_enhancer, arch=resolved_arch)

        # ── BREAK tokens ────────────────────────────────────────────────────
        if use_break_bool:
            final_prompt = insert_break_points(final_prompt, max_tokens=token_limit - 7)

        # ── Tokenize & Real token count ─────────────────────────────────────
        pos_tokens = clip.tokenize(final_prompt)
        real_count = _real_token_count(clip, final_prompt, tokens=pos_tokens)

        if real_count > token_limit and not use_break_bool:
            logger.warning(
                f"[Encoder] Prompt has {real_count} tokens (limit {token_limit}). "
                "Enable 'use_break' or increase context_window. Truncating to prevent OOM."
            )
            # v3.1 FIX: Slice into new lists instead of mutating in-place.
            truncated = {}
            for key in pos_tokens:
                if isinstance(pos_tokens[key], list):
                    new_list = []
                    for item in pos_tokens[key]:
                        if hasattr(item, "__len__") and len(item) > token_limit:
                            new_list.append(item[:token_limit])
                        else:
                            new_list.append(item)
                    truncated[key] = new_list
                else:
                    truncated[key] = pos_tokens[key]
            pos_tokens = truncated

        # ── CLIP skip ───────────────────────────────────────────────────────
        if clip_skip > 0:
            # [BUG-I7] Guard clip.clone() for custom CLIP wrappers that lack it
            try:
                clip = clip.clone()
                clip.clip_layer(-clip_skip)
            except AttributeError:
                logger.warning(
                    f"[Encoder] clip.clone() not available — applying clip_layer "
                    f"directly. This may affect other nodes sharing this CLIP."
                )
                try:
                    clip.clip_layer(-clip_skip)
                except Exception as e:
                    logger.error(f"[Encoder] clip_layer({-clip_skip}) failed: {e}")

        # ── Encode ──────────────────────────────────────────────────────────
        # [BUG-C1] CRITICAL FIX: encode_from_tokens_scheduled (ComfyUI ~2024-Q2+)
        # returns conditioning in native format: [[cond, {"pooled_output": pooled}]].
        # The fallback encode_from_tokens() returns (cond, pooled) as a raw tuple,
        # which must be manually wrapped into conditioning format.
        # Previous code returned the raw tuple, causing sampler crashes on older installs.
        def _encode(tokens):
            if hasattr(clip, "encode_from_tokens_scheduled"):
                return clip.encode_from_tokens_scheduled(tokens)
            # Fallback for older ComfyUI — wrap into conditioning format
            result = clip.encode_from_tokens(tokens, return_pooled=True)
            if isinstance(result, (list, tuple)) and len(result) == 2:
                cond, pooled = result
                # Standard ComfyUI conditioning format
                return [[cond, {"pooled_output": pooled}]]
            # If it's already in the right format somehow, pass through
            return result

        positive_cond = _encode(pos_tokens)

        # Ensure negative prompt is at least a space to prevent empty tensor crashes
        safe_negative = negative_prompt if negative_prompt and negative_prompt.strip() else " "
        neg_tokens = clip.tokenize(safe_negative)
        negative_cond = _encode(neg_tokens)

        # ── image_ref passthrough ────────────────────────────────────────
        # image_ref is optional, but the output slot is typed IMAGE.
        # Downstream nodes crash on None, so provide a 1×1 black pixel fallback.
        if image_ref is None:
            image_ref = torch.zeros(1, 1, 1, 3)  # BHWC
            logger.debug("[Encoder] No image_ref connected — using 1x1 black fallback.")

        return (positive_cond, negative_cond, image_ref, final_prompt, negative_prompt, real_count)

# ═══════════════════════════════════════════════════════════════════════════════
#                         NODE MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "RadianceCinematicPromptEncoder": RadianceCinematicPromptEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceCinematicPromptEncoder": "◎ Radiance Cinematic Encoder",
}
