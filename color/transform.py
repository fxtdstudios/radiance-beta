import torch
import numpy as np
import os
import re
import logging

# Module logger
logger = logging.getLogger("radiance.color.transform")

# Import shared color utilities
try:
    # Try importing from package root using relative import (most robust)
    from ..color_utils import (
        # Log curves — numpy (CPU path, matrix ops)
        linear_to_logc3,
        logc3_to_linear,
        linear_to_logc4,
        logc4_to_linear,
        linear_to_slog3,
        slog3_to_linear,
        linear_to_vlog,
        vlog_to_linear,
        linear_to_canonlog3,
        canonlog3_to_linear,
        linear_to_acescct,
        acescct_to_linear,
        linear_to_davinci_intermediate,
        davinci_intermediate_to_linear,
        # RED Log3G10
        linear_to_log3g10,
        log3g10_to_linear,
        # GPU tensor log curves — FIX #10: wire these for the GPU path
        tensor_linear_to_logc3,
        tensor_logc3_to_linear,
        tensor_linear_to_logc4,
        tensor_logc4_to_linear,
        tensor_linear_to_slog3,
        tensor_slog3_to_linear,
        tensor_linear_to_vlog,
        tensor_vlog_to_linear,
        tensor_linear_to_log3g10,
        tensor_log3g10_to_linear,
        tensor_linear_to_davinci_intermediate,
        tensor_davinci_intermediate_to_linear,
        tensor_linear_to_acescct,
        tensor_acescct_to_linear,
        tensor_linear_to_canonlog3,
        tensor_canonlog3_to_linear,
        # Tensor helpers
        tensor_to_numpy_float32,
        numpy_to_tensor_float32,
        # Matrices and Transform
        apply_matrix_transform,
        AWG3_TO_ACESCG,
        AWG4_TO_ACESCG,
        SGAMUT3_CINE_TO_ACESCG,
        VGAMUT_TO_ACESCG,
        CINEMA_GAMUT_TO_ACESCG,
        REDWIDEGAMUT_TO_ACESCG,
        DAVINCI_WIDE_TO_ACESCG,
        ACESCG_TO_SRGB,
        ACESCG_TO_P3D65,
        ACESCG_TO_REC2020,
        # ACES 2.0
        aces2_tonemap,
        aces2_gamut_compress,
        linear_to_jmh,
        jmh_to_linear,
        # HDR
        linear_to_srgb,
        linear_to_pq,
        linear_to_hlg,
    )
except ImportError:
    # Fallback to absolute import
    try:
        from radiance.color_utils import (
            linear_to_logc3,
            logc3_to_linear,
            linear_to_logc4,
            logc4_to_linear,
            linear_to_slog3,
            slog3_to_linear,
            linear_to_vlog,
            vlog_to_linear,
            linear_to_canonlog3,
            canonlog3_to_linear,
            linear_to_acescct,
            acescct_to_linear,
            linear_to_davinci_intermediate,
            davinci_intermediate_to_linear,
            linear_to_log3g10,
            log3g10_to_linear,
            tensor_to_numpy_float32,
            numpy_to_tensor_float32,
            apply_matrix_transform,
            AWG3_TO_ACESCG,
            AWG4_TO_ACESCG,
            SGAMUT3_CINE_TO_ACESCG,
            VGAMUT_TO_ACESCG,
            CINEMA_GAMUT_TO_ACESCG,
            REDWIDEGAMUT_TO_ACESCG,
            DAVINCI_WIDE_TO_ACESCG,
            ACESCG_TO_SRGB,
            ACESCG_TO_P3D65,
            ACESCG_TO_REC2020,
            aces2_tonemap,
            aces2_gamut_compress,
            linear_to_srgb,
            linear_to_pq,
            linear_to_hlg,
        )
    except ImportError as e:
        logger.error(f"CRITICAL: Could not import color_utils: {e}")
        raise

# Optional PyOpenColorIO
try:
    import PyOpenColorIO as OCIO

    HAS_OCIO = True
except ImportError:
    OCIO = None
    HAS_OCIO = False


class RadianceGPUColorMatrix:
    """
    Apply custom color matrix transformations (3x3 or 4x4).
    """

    MATRIX_PRESETS = {
        "Custom": None,
        "Identity": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "Sepia": [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]],
        "B&W (Luminance)": [
            [0.2126, 0.7152, 0.0722],
            [0.2126, 0.7152, 0.0722],
            [0.2126, 0.7152, 0.0722],
        ],
        "Invert": [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],  # requires offset [1,1,1]
    }

    # Presets that need a built-in offset to work correctly
    PRESET_OFFSETS = {
        "Invert": [1.0, 1.0, 1.0],
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (list(cls.MATRIX_PRESETS.keys()), {"default": "Custom"}),
                "matrix_type": (["RGB (3x3)", "RGBA (4x4)"], {"default": "RGB (3x3)"}),
                "r_vector": ("STRING", {"default": "1.0, 0.0, 0.0"}),
                "g_vector": ("STRING", {"default": "0.0, 1.0, 0.0"}),
                "b_vector": ("STRING", {"default": "0.0, 0.0, 1.0"}),
            },
            "optional": {
                "a_vector": ("STRING", {"default": "0.0, 0.0, 0.0, 1.0"}),
                "offset": ("STRING", {"default": "0.0, 0.0, 0.0"}),
                "clamp_output": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Clamp to 0-1 range"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_matrix"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Apply a 3×3 or 4×4 color matrix to an image. Includes presets (Identity, Sepia, B&W, Invert) and full custom vector entry with optional offset."

    @staticmethod
    def parse_vector(v_str, width):
        try:
            v_str = re.sub(r"[,\s]+", " ", v_str).strip()
            vals = [float(x) for x in v_str.split()]
            if len(vals) < width:
                vals += [0.0] * (width - len(vals))
            return vals[:width]
        except Exception as e:
            logger.warning(f"Failed to parse vector '{v_str}': {e}")
            return [0.0] * width

    def apply_matrix(
        self,
        image,
        preset,
        matrix_type,
        r_vector,
        g_vector,
        b_vector,
        a_vector=None,
        offset=None,
        clamp_output=False,
    ):
        device = image.device

        if preset != "Custom" and preset in self.MATRIX_PRESETS:
            r_vals, g_vals, b_vals = self.MATRIX_PRESETS[preset]
            # Apply built-in offset for presets that need it (e.g. Invert: out = 1 - in)
            preset_offset = self.PRESET_OFFSETS.get(preset)
        else:
            r_vals = self.parse_vector(r_vector, 3)
            g_vals = self.parse_vector(g_vector, 3)
            b_vals = self.parse_vector(b_vector, 3)
            preset_offset = None

        img_flat = image.reshape(-1, image.shape[-1])

        if matrix_type == "RGB (3x3)":
            matrix = torch.tensor(
                [r_vals, g_vals, b_vals], dtype=torch.float32, device=device
            )
            rgb = img_flat[..., :3]
            res = torch.matmul(rgb, matrix.T)

            # Combine preset offset + user offset
            total_offset = [0.0, 0.0, 0.0]
            if preset_offset:
                total_offset = [a + b for a, b in zip(total_offset, preset_offset)]
            if offset:
                user_off = self.parse_vector(offset, 3)
                total_offset = [a + b for a, b in zip(total_offset, user_off)]
            if any(v != 0.0 for v in total_offset):
                res += torch.tensor(total_offset, device=device)

            if image.shape[-1] > 3:
                res = torch.cat([res, img_flat[..., 3:]], dim=-1)

        else:  # 4x4
            a_vals = self.parse_vector(a_vector, 4) if a_vector else [0, 0, 0, 1]
            matrix = torch.tensor(
                [r_vals + [0], g_vals + [0], b_vals + [0], a_vals],
                dtype=torch.float32,
                device=device,
            )

            if image.shape[-1] == 3:
                alpha = torch.ones((img_flat.shape[0], 1), device=device)
                img_flat = torch.cat([img_flat, alpha], dim=-1)

            res = torch.matmul(img_flat, matrix.T)

            # Combine preset offset + user offset
            total_offset = [0.0, 0.0, 0.0, 0.0]
            if preset_offset:
                for i in range(min(len(preset_offset), 3)):
                    total_offset[i] += preset_offset[i]
            if offset:
                user_off = self.parse_vector(offset, 4)
                total_offset = [a + b for a, b in zip(total_offset, user_off)]
            if any(v != 0.0 for v in total_offset):
                res += torch.tensor(total_offset, device=device)

            if image.shape[-1] == 3:
                res = res[..., :3]

        res = res.reshape(image.shape)
        if clamp_output:
            res = torch.clamp(res, 0.0, 1.0)
        # No min=0 clamp when disabled — scene-linear data has valid negatives

        return (res,)


class RadianceOCIOColorTransform:
    """OpenColorIO Transform node."""

    # FIX #8: module-level caches so config and processor are created once,
    # not on every call. RadianceOCIODisplayView (ocio_view.py) already does
    # this correctly — porting the same pattern here.
    _config_cache: dict = {}  # keyed by config_file path
    _processor_cache: dict = {}  # keyed by (config_file, src_space, dst_space)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "config_file": ("STRING", {"default": "ACES 1.3 Studio"}),
                "input_space": ("STRING", {"default": "ACES - ACEScg"}),
                "output_space": ("STRING", {"default": "Output - sRGB"}),
                "direction": (["Forward", "Inverse"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "transform"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Apply an OpenColorIO color space transform between any two spaces defined in the config."

    def transform(self, image, config_file, input_space, output_space, direction):
        if not HAS_OCIO:
            logger.warning("PyOpenColorIO not found.")
            return (image,)

        try:
            # FIX #8: cache the config — CreateFromFile() can take 200-500 ms.
            # Use the config_file string as cache key (same as ocio_view.py pattern).
            if config_file not in self._config_cache:
                if os.path.exists(config_file):
                    config = OCIO.Config.CreateFromFile(config_file)
                else:
                    try:
                        config = OCIO.Config.CreateFromEnv()
                    except Exception as e:
                        logger.warning(f"OCIO env config failed, using default: {e}")
                        config = OCIO.Config.Create()
                self._config_cache[config_file] = config
            config = self._config_cache[config_file]

            # FIX #8: cache the CPU processor — getDefaultCPUProcessor() is also expensive.
            proc_key = (config_file, input_space, output_space, direction)
            if proc_key not in self._processor_cache:
                if direction == "Forward":
                    tf = OCIO.ColorSpaceTransform(src=input_space, dst=output_space)
                else:
                    tf = OCIO.ColorSpaceTransform(src=output_space, dst=input_space)
                processor = config.getProcessor(tf)
                self._processor_cache[proc_key] = processor.getDefaultCPUProcessor()
            cpu = self._processor_cache[proc_key]

            img_np = image.cpu().numpy().astype(np.float32)
            batch, h, w, c = img_np.shape
            flat = img_np.reshape(-1, c)

            # OCIO processes RGB; handle alpha if present
            if c >= 3:
                # Must copy for contiguous memory — OCIO may not write back to non-contiguous views
                rgb = flat[:, :3].copy()
                cpu.applyRGB(rgb)
                flat[:, :3] = rgb

            res = flat.reshape(batch, h, w, c)
            return (numpy_to_tensor_float32(res),)

        except Exception as e:
            logger.error(f"OCIO Error: {e}")
            return (image,)


class RadianceSceneLinearWorkflow:
    """
    Scene-Linear Workflow preset node. Common presets for professional color pipelines: - ACES: ACEScg working space with ACES output transforms - Nuke Default: sRGB linearized workflow - Film Emulation: LogC/S-Log for film look Automatically configures decode -> working space -> encode chain.
    """

    PRESETS = {
        "ACES 1.3 (ACEScg)": {
            "decode": "None (Linear)",
            "working_space": "ACEScg",
            "encode": "ACES 1.0 SDR (sRGB)",
        },
        "ACES 2.0 (ACEScg)": {
            "decode": "None (Linear)",
            "working_space": "ACEScg",
            "encode": "ACES 2.0 SDR",
        },
        "Film Look (LogC3)": {
            "decode": "ARRI LogC3",
            "working_space": "ACEScg",
            "encode": "ARRI LogC3",
        },
        "Broadcast (Rec.709)": {
            "decode": "Rec.709 (sRGB)",
            "working_space": "Linear sRGB",
            "encode": "Rec.709 (sRGB)",
        },
        "HDR (Rec.2020 PQ)": {
            "decode": "None (Linear)",
            "working_space": "Rec.2020",
            "encode": "Rec.2020 PQ (HDR10)",
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(cls.PRESETS.keys()), {"default": "ACES 1.3 (ACEScg)"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("decode", "working_space", "encode")
    FUNCTION = "get_preset"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Scene-linear workflow presets. Returns decode/working_space/encode settings for color pipeline."

    def get_preset(self, preset):
        settings = self.PRESETS.get(preset, self.PRESETS["ACES 1.3 (ACEScg)"])
        return (settings["decode"], settings["working_space"], settings["encode"])


class RadianceLogCurveDecode:
    """
    Logarithmic to Linear conversion using shared `color_utils`.
    """

    CURVES = {
        "ARRI LogC3": logc3_to_linear,
        "ARRI LogC4": logc4_to_linear,
        "Sony S-Log3": slog3_to_linear,
        "Panasonic V-Log": vlog_to_linear,
        "Canon Log 3": canonlog3_to_linear,
        "RED Log3G10": log3g10_to_linear,
        "ACEScct": acescct_to_linear,
        "DaVinci Intermediate": davinci_intermediate_to_linear,
    }

    GAMUTS = [
        "ACEScg (Linear)",
        "Native / No Transform",
        "Linear sRGB (Rec.709)",
        "Linear P3-D65",
        "Linear Rec.2020",
    ]

    # ARRI LogC3 EI options
    EI_OPTIONS = [
        "EI 160",
        "EI 200",
        "EI 250",
        "EI 320",
        "EI 400",
        "EI 500",
        "EI 640",
        "EI 800",
        "EI 1000",
        "EI 1280",
        "EI 1600",
        "EI 2000",
        "EI 2560",
        "EI 3200",
    ]

    # Map Curve Name to Source Gamut Matrix (to ACEScg)
    CURVE_TO_MATRIX = {
        "ARRI LogC3": AWG3_TO_ACESCG,
        "ARRI LogC4": AWG4_TO_ACESCG,
        "Sony S-Log3": SGAMUT3_CINE_TO_ACESCG,
        "Panasonic V-Log": VGAMUT_TO_ACESCG,
        "Canon Log 3": CINEMA_GAMUT_TO_ACESCG,
        "RED Log3G10": REDWIDEGAMUT_TO_ACESCG,
        "DaVinci Intermediate": DAVINCI_WIDE_TO_ACESCG,
        "ACEScct": None,  # Already AP1
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "curve": (list(cls.CURVES.keys()),),
                "target_gamut": (cls.GAMUTS, {"default": "ACEScg (Linear)"}),
            },
            "optional": {
                "exposure_index": (
                    cls.EI_OPTIONS,
                    {
                        "default": "EI 800",
                        "tooltip": "Exposure Index for ARRI LogC3 (only affects LogC3 curve).",
                    },
                ),
                "use_gpu": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "GPU Accelerated",
                        "label_off": "CPU",
                        "tooltip": "Use GPU for faster processing (falls back if unavailable).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Decode log footage (LogC3/4, S-Log3, Log3G10, etc.) to linear with EI control and gamut transform."

    # FIX #10: GPU tensor curve map — these run entirely on the existing GPU
    # tensor without a host round-trip. Previously use_gpu was silently ignored.
    TENSOR_CURVES = {
        "ARRI LogC3":           (tensor_logc3_to_linear,              tensor_linear_to_logc3),
        "ARRI LogC4":           (tensor_logc4_to_linear,              tensor_linear_to_logc4),
        "Sony S-Log3":          (tensor_slog3_to_linear,              tensor_linear_to_slog3),
        "Panasonic V-Log":      (tensor_vlog_to_linear,               tensor_linear_to_vlog),
        "Canon Log 3":          (tensor_canonlog3_to_linear,          tensor_linear_to_canonlog3),
        "RED Log3G10":          (tensor_log3g10_to_linear,            tensor_linear_to_log3g10),
        "ACEScct":              (tensor_acescct_to_linear,            tensor_linear_to_acescct),
        "DaVinci Intermediate": (tensor_davinci_intermediate_to_linear, tensor_linear_to_davinci_intermediate),
    }

    def decode(
        self,
        image,
        curve,
        target_gamut="ACEScg (Linear)",
        exposure_index="EI 800",
        use_gpu=True,
    ):
        # Parse EI value (used by LogC3 CPU path)
        ei = int(exposure_index.replace("EI ", ""))

        if use_gpu and curve in self.TENSOR_CURVES:
            # ── GPU PATH ──────────────────────────────────────────────────────
            # Runs entirely on the input tensor — no CPU round-trip.
            # EI is only relevant for ARRI LogC3; the tensor variant accepts it
            # as a keyword argument so all other curves are called without it.
            decode_fn, _ = self.TENSOR_CURVES[curve]
            if curve == "ARRI LogC3":
                img_t = decode_fn(image, ei=ei)
            else:
                img_t = decode_fn(image)

            # Gamut transform in tensor space (matrix on CPU, applied via matmul)
            if target_gamut != "Native / No Transform":
                source_matrix = self.CURVE_TO_MATRIX.get(curve)
                if source_matrix is not None:
                    mat = torch.tensor(source_matrix, dtype=torch.float32,
                                       device=image.device)
                    shape = img_t.shape
                    img_t = (img_t[..., :3].reshape(-1, 3) @ mat.T).reshape(shape[:-1] + (3,))

                if target_gamut == "Linear sRGB (Rec.709)":
                    mat2 = torch.tensor(ACESCG_TO_SRGB, dtype=torch.float32, device=image.device)
                    shape = img_t.shape
                    img_t = (img_t[..., :3].reshape(-1, 3) @ mat2.T).reshape(shape[:-1] + (3,))
                elif target_gamut == "Linear P3-D65":
                    mat2 = torch.tensor(ACESCG_TO_P3D65, dtype=torch.float32, device=image.device)
                    shape = img_t.shape
                    img_t = (img_t[..., :3].reshape(-1, 3) @ mat2.T).reshape(shape[:-1] + (3,))
                elif target_gamut == "Linear Rec.2020":
                    mat2 = torch.tensor(ACESCG_TO_REC2020, dtype=torch.float32, device=image.device)
                    shape = img_t.shape
                    img_t = (img_t[..., :3].reshape(-1, 3) @ mat2.T).reshape(shape[:-1] + (3,))

            return (img_t,)

        # ── CPU PATH ──────────────────────────────────────────────────────────
        func = self.CURVES.get(curve)
        if not func:
            return (image,)

        img_np = tensor_to_numpy_float32(image)

        if curve == "ARRI LogC3":
            res_np = func(img_np, ei=ei)
        else:
            res_np = func(img_np)

        if target_gamut != "Native / No Transform":
            source_to_aces_matrix = self.CURVE_TO_MATRIX.get(curve)
            if source_to_aces_matrix is not None:
                res_np = apply_matrix_transform(res_np, source_to_aces_matrix)

            if target_gamut == "Linear sRGB (Rec.709)":
                res_np = apply_matrix_transform(res_np, ACESCG_TO_SRGB)
            elif target_gamut == "Linear P3-D65":
                res_np = apply_matrix_transform(res_np, ACESCG_TO_P3D65)
            elif target_gamut == "Linear Rec.2020":
                res_np = apply_matrix_transform(res_np, ACESCG_TO_REC2020)

        return (numpy_to_tensor_float32(res_np),)


class RadianceLogCurveEncode:
    """
    Linear to Logarithmic conversion using shared `color_utils`.
    """

    CURVES = {
        "ARRI LogC3": linear_to_logc3,
        "ARRI LogC4": linear_to_logc4,
        "Sony S-Log3": linear_to_slog3,
        "Panasonic V-Log": linear_to_vlog,
        "Canon Log 3": linear_to_canonlog3,
        "RED Log3G10": linear_to_log3g10,
        "ACEScct": linear_to_acescct,
        "DaVinci Intermediate": linear_to_davinci_intermediate,
    }

    GAMUTS = [
        "ACEScg (Linear)",
        "Native / No Transform",
        "Linear sRGB (Rec.709)",
        "Linear P3-D65",
        "Linear Rec.2020",
    ]

    # ARRI LogC3 EI options
    EI_OPTIONS = [
        "EI 160",
        "EI 200",
        "EI 250",
        "EI 320",
        "EI 400",
        "EI 500",
        "EI 640",
        "EI 800",
        "EI 1000",
        "EI 1280",
        "EI 1600",
        "EI 2000",
        "EI 2560",
        "EI 3200",
    ]

    # Map Curve Name to Source Gamut Matrix (to ACEScg)
    CURVE_TO_MATRIX = {
        "ARRI LogC3": AWG3_TO_ACESCG,
        "ARRI LogC4": AWG4_TO_ACESCG,
        "Sony S-Log3": SGAMUT3_CINE_TO_ACESCG,
        "Panasonic V-Log": VGAMUT_TO_ACESCG,
        "Canon Log 3": CINEMA_GAMUT_TO_ACESCG,
        "RED Log3G10": REDWIDEGAMUT_TO_ACESCG,
        "DaVinci Intermediate": DAVINCI_WIDE_TO_ACESCG,
        "ACEScct": None,
    }

    # FIX #11: pre-compute the matrix inverses once at class definition time.
    # encode() was calling np.linalg.inv() on these constant matrices on every
    # single invocation — wasted work since they never change.
    _SRGB_TO_ACESCG = np.linalg.inv(np.array(ACESCG_TO_SRGB, dtype=np.float64))
    _P3D65_TO_ACESCG = np.linalg.inv(np.array(ACESCG_TO_P3D65, dtype=np.float64))
    _REC2020_TO_ACESCG = np.linalg.inv(np.array(ACESCG_TO_REC2020, dtype=np.float64))

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "curve": (list(cls.CURVES.keys()),),
                "source_gamut": (cls.GAMUTS, {"default": "ACEScg (Linear)"}),
            },
            "optional": {
                "exposure_index": (
                    cls.EI_OPTIONS,
                    {
                        "default": "EI 800",
                        "tooltip": "Exposure Index for ARRI LogC3 (only affects LogC3 curve).",
                    },
                ),
                "apply_gamma": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Linearise Gamma (÷2.2)",
                        "label_off": "No Gamma",
                    },
                ),
                "gamma": (
                    "FLOAT",
                    {"default": 2.2, "min": 0.1, "max": 10.0, "step": 0.1},
                ),
                "exposure": (
                    "FLOAT",
                    {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1},
                ),
                "clamp_output": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Clamp Output",
                        "label_off": "No Clamp",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "encode"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Encode linear footage to log curves (LogC3/4, S-Log3, Log3G10, etc.) with EI control."

    def encode(
        self,
        image,
        curve,
        source_gamut="ACEScg (Linear)",
        exposure_index="EI 800",
        apply_gamma=True,
        gamma=2.2,
        exposure=0.0,
        clamp_output=True,
    ):
        func = self.CURVES.get(curve)
        if not func:
            return (image,)

        # Parse EI value
        ei = int(exposure_index.replace("EI ", ""))

        # Pre-process: linearise input gamma then apply exposure — stays on GPU
        if apply_gamma:
            sign  = torch.sign(image)
            image = sign * torch.pow(torch.abs(image), 1.0 / gamma)

        if exposure != 0.0:
            image = image * (2.0 ** exposure)

        # ── GPU PATH ──────────────────────────────────────────────────────────
        # FIX #10: use tensor encode variants so there is no CPU round-trip.
        # Gamut matrices are applied as inline matmuls on the GPU tensor.
        _, encode_fn = RadianceLogCurveDecode.TENSOR_CURVES.get(
            curve, (None, None)
        )
        if encode_fn is not None:
            img_t = image

            if source_gamut != "Native / No Transform":
                if source_gamut == "Linear sRGB (Rec.709)":
                    mat = torch.tensor(self._SRGB_TO_ACESCG, dtype=torch.float32,
                                       device=image.device)
                    img_t = (img_t[..., :3].reshape(-1, 3) @ mat.T).reshape(img_t.shape)
                elif source_gamut == "Linear P3-D65":
                    mat = torch.tensor(self._P3D65_TO_ACESCG, dtype=torch.float32,
                                       device=image.device)
                    img_t = (img_t[..., :3].reshape(-1, 3) @ mat.T).reshape(img_t.shape)
                elif source_gamut == "Linear Rec.2020":
                    mat = torch.tensor(self._REC2020_TO_ACESCG, dtype=torch.float32,
                                       device=image.device)
                    img_t = (img_t[..., :3].reshape(-1, 3) @ mat.T).reshape(img_t.shape)

                native_to_aces = self.CURVE_TO_MATRIX.get(curve)
                if native_to_aces is not None:
                    aces_to_native = np.linalg.inv(native_to_aces).astype(np.float32)
                    mat2 = torch.tensor(aces_to_native, dtype=torch.float32,
                                        device=image.device)
                    img_t = (img_t[..., :3].reshape(-1, 3) @ mat2.T).reshape(img_t.shape)

            if curve == "ARRI LogC3":
                result = encode_fn(img_t, ei=ei)
            else:
                result = encode_fn(img_t)

            if clamp_output:
                result = torch.clamp(result, 0.0, 1.0)
            return (result,)

        # ── CPU FALLBACK ───────────────────────────────────────────────────────
        img_np = tensor_to_numpy_float32(image)

        if source_gamut != "Native / No Transform":
            if source_gamut == "Linear sRGB (Rec.709)":
                img_np = apply_matrix_transform(img_np, self._SRGB_TO_ACESCG)
            elif source_gamut == "Linear P3-D65":
                img_np = apply_matrix_transform(img_np, self._P3D65_TO_ACESCG)
            elif source_gamut == "Linear Rec.2020":
                img_np = apply_matrix_transform(img_np, self._REC2020_TO_ACESCG)

            native_to_aces = self.CURVE_TO_MATRIX.get(curve)
            if native_to_aces is not None:
                aces_to_native = np.linalg.inv(native_to_aces)
                img_np = apply_matrix_transform(img_np, aces_to_native)

        if curve == "ARRI LogC3":
            res_np = func(img_np, ei=ei)
        else:
            res_np = func(img_np)

        if clamp_output:
            res_np = np.clip(res_np, 0.0, 1.0)

        return (numpy_to_tensor_float32(res_np),)


# =============================================================================
# NODE MAPPINGS
# FIX #2: NODE_CLASS_MAPPINGS was absent — all 5 nodes were invisible to ComfyUI.
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "RadianceGPUColorMatrix": RadianceGPUColorMatrix,
    "RadianceOCIOColorTransform": RadianceOCIOColorTransform,
    "RadianceSceneLinearWorkflow": RadianceSceneLinearWorkflow,
    "RadianceLogCurveDecode": RadianceLogCurveDecode,
    "RadianceLogCurveEncode": RadianceLogCurveEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceGPUColorMatrix": "◎ GPU Color Matrix",
    "RadianceOCIOColorTransform": "◎ OCIO Color Transform",
    "RadianceSceneLinearWorkflow": "◎ Scene Linear Workflow",
    "RadianceLogCurveDecode": "◎ Log Curve Decode",
    "RadianceLogCurveEncode": "◎ Log Curve Encode",
}
