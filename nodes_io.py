"""
═══════════════════════════════════════════════════════════════════════════════
                    RADIANCE — UNIVERSAL VFX IO v2.1
═══════════════════════════════════════════════════════════════════════════════
Industry-standard readers/writers for Video, Image Sequences, GIF, and WEBP
with built-in Input/Output Transforms (IDT/ODT) for linearizing Log footage.

Advantages over VHS VideoCombine and WAN SaveVideo:
  • IMAGE passthrough on write — chain output to more nodes
  • Image sequence export (PNG, EXR, JPEG numbered frames)
  • GIF/WEBP animated export
  • Audio extraction from video
  • Full color pipeline (10+ Log/gamma spaces)
  • ProRes 4444 XQ 12-bit support
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import glob
import json
import time
import logging
import subprocess  # nosec B404
import tempfile

import torch
import numpy as np
import cv2

from . import color_utils
try:
    from .path_utils import safe_join, get_safe_output_dir, get_safe_input_path
except ImportError:
    from path_utils import safe_join, get_safe_output_dir, get_safe_input_path

import folder_paths


logger = logging.getLogger("radiance.io")

# ═══════════════════════════════════════════════════════════════════════════════
#                         COLOR SPACE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_COLORSPACES = [
    "Linear (sRGB)",
    "sRGB (Standard)",
    "ARRI LogC3",
    "ARRI LogC4",
    "Sony S-Log3",
    "Panasonic V-Log",
    "Canon Log 3",
    "RED Log3G10",
    "ACEScct",
    "DaVinci Intermediate",
]


def apply_input_transform(img_tensor: torch.Tensor, colorspace: str) -> torch.Tensor:
    """Apply Input Device Transform (IDT): Source Gamma/Log → Linear."""
    if colorspace == "Linear (sRGB)":
        return img_tensor
    elif colorspace == "sRGB (Standard)":
        return color_utils.tensor_srgb_to_linear(img_tensor)

    device = img_tensor.device
    img_np = img_tensor.cpu().numpy()

    transform_map = {
        "ARRI LogC3": color_utils.logc3_to_linear,
        "ARRI LogC4": color_utils.logc4_to_linear,
        "Sony S-Log3": color_utils.slog3_to_linear,
        "Panasonic V-Log": color_utils.vlog_to_linear,
        "Canon Log 3": color_utils.canonlog3_to_linear,
        "RED Log3G10": color_utils.log3g10_to_linear,
        "ACEScct": color_utils.acescct_to_linear,
        "DaVinci Intermediate": color_utils.davinci_intermediate_to_linear,
    }

    fn = transform_map.get(colorspace)
    out_np = fn(img_np) if fn else img_np
    return torch.from_numpy(out_np).to(device)


def apply_output_transform(img_tensor: torch.Tensor, colorspace: str) -> torch.Tensor:
    """Apply Output Transform (ODT): Linear → Target Gamma/Log."""
    if colorspace == "Linear (sRGB)":
        return img_tensor
    elif colorspace == "sRGB (Standard)":
        return color_utils.tensor_linear_to_srgb(img_tensor)

    device = img_tensor.device
    img_np = img_tensor.cpu().numpy()

    transform_map = {
        "ARRI LogC3": color_utils.linear_to_logc3,
        "ARRI LogC4": color_utils.linear_to_logc4,
        "Sony S-Log3": color_utils.linear_to_slog3,
        "Panasonic V-Log": color_utils.linear_to_vlog,
        "Canon Log 3": color_utils.linear_to_canonlog3,
        "RED Log3G10": color_utils.linear_to_log3g10,
        "ACEScct": color_utils.linear_to_acescct,
        "DaVinci Intermediate": color_utils.linear_to_davinci_intermediate,
    }

    fn = transform_map.get(colorspace)
    out_np = fn(img_np) if fn else img_np
    return torch.from_numpy(out_np).to(device)


# ═══════════════════════════════════════════════════════════════════════════════
#                         AUDIO EXTRACTION HELPER
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_audio_ffmpeg(video_path: str) -> dict | None:
    """
    Extract audio from video via FFmpeg → returns ComfyUI AUDIO dict
    {waveform: Tensor (1, channels, samples), sample_rate: int} or None.
    """
    try:
        # Probe for audio stream
        probe_cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type,sample_rate,channels",
            "-of",
            "json",
            video_path,
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)  # nosec B603
        if result.returncode != 0:
            return None

        probe_data = json.loads(result.stdout)
        streams = probe_data.get("streams", [])
        if not streams:
            return None  # No audio stream

        sample_rate = int(streams[0].get("sample_rate", 44100))
        channels = int(streams[0].get("channels", 2))

        # Extract raw PCM audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        extract_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            tmp_path,
        ]
        subprocess.run(extract_cmd, capture_output=True, timeout=120)  # nosec B603

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 100:
            return None

        # Read WAV samples
        import struct

        with open(tmp_path, "rb") as f:
            data = f.read()

        os.unlink(tmp_path)

        # Parse WAV: skip header (44 bytes for standard WAV)
        # Find 'data' chunk
        data_offset = data.find(b"data")
        if data_offset == -1:
            return None
        data_offset += 4  # skip 'data'
        data_size = struct.unpack_from("<I", data, data_offset)[0]
        data_offset += 4
        raw_audio = data[data_offset : data_offset + data_size]

        # Parse float32 samples
        n_samples = len(raw_audio) // 4
        samples = np.frombuffer(raw_audio, dtype=np.float32)

        # Reshape to (channels, samples_per_channel)
        samples_per_channel = n_samples // channels
        samples = samples[: samples_per_channel * channels]
        waveform = samples.reshape(
            samples_per_channel, channels
        ).T  # (channels, samples)
        waveform_tensor = torch.from_numpy(waveform.copy()).unsqueeze(
            0
        )  # (1, channels, samples)

        logger.info(
            f"Audio extracted: {sample_rate}Hz, {channels}ch, {samples_per_channel} samples"
        )
        return {"waveform": waveform_tensor, "sample_rate": sample_rate}

    except Exception as e:
        logger.warning(f"Audio extraction failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#                         NODE: READ VIDEO
# ═══════════════════════════════════════════════════════════════════════════════


class RadianceReadVideo:
    """
    Universal VFX Video Reader.
    Reads video files, applies Input Transforms (Log→Linear),
    and extracts audio. Outputs IMAGE batch + metadata for downstream linking.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "C:/Projects/footage.mp4"}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "frame_count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999999,
                        "tooltip": "0 = Load all frames",
                    },
                ),
                "input_colorspace": (INPUT_COLORSPACES, {"default": "sRGB (Standard)"}),
                "frame_stride": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "tooltip": "Skip every Nth frame",
                    },
                ),
                "extract_audio": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vhs_video": ("VHS_VIDEO",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "FLOAT", "AUDIO", "VHS_VIDEO")
    RETURN_NAMES = ("IMAGE", "MASK", "frame_count", "width", "height", "fps", "audio", "video")
    FUNCTION = "read_video"
    CATEGORY = "FXTD Studios/Radiance/IO"
    DESCRIPTION = (
        "Professional video reader with color transforms and audio extraction."
    )

    def read_video(
        self,
        video_path,
        start_frame,
        frame_count,
        input_colorspace,
        frame_stride,
        extract_audio=True,
        vhs_video=None,
    ):
        if vhs_video is not None:
            if isinstance(vhs_video, str):
                video_path = vhs_video
            elif isinstance(vhs_video, dict) and "source" in vhs_video:
                video_path = vhs_video["source"]
            elif isinstance(vhs_video, list) and len(vhs_video) > 0:
                video_path = vhs_video[0]

        video_path = get_safe_input_path(folder_paths.get_input_directory(), video_path)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame >= total_frames:
            cap.release()
            raise ValueError(
                f"Start frame {start_frame} is beyond end of video ({total_frames} frames)"
            )

        # Determine actual frames to read
        if frame_count <= 0:
            frames_to_read = total_frames - start_frame
        else:
            frames_to_read = min(frame_count, total_frames - start_frame)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(
            f"Reading video: {os.path.basename(video_path)} | {width}x{height} @ {fps:.2f}fps | Start: {start_frame} | Count: {frames_to_read} | IDT: {input_colorspace}"
        )

        # Seek to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        frames_read = 0
        current_offset = 0

        while frames_read < frames_to_read:
            ret, frame = cap.read()
            if not ret:
                break

            if current_offset % frame_stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            current_offset += 1
            frames_read += 1

        cap.release()

        if not frames:
            raise RuntimeError("No frames were read from video")

        # Stack and convert to float32 [0,1]
        frames_np = np.stack(frames).astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(frames_np)

        # Apply Input Transform (Linearize)
        frames_linear = apply_input_transform(frames_tensor, input_colorspace)

        # Mask (full white — cv2 doesn't extract alpha from most video codecs)
        b, h, w, c = frames_linear.shape
        mask = torch.ones((b, h, w), dtype=torch.float32)

        # Audio extraction
        audio = None
        if extract_audio:
            audio = _extract_audio_ffmpeg(video_path)

        return (frames_linear, mask, len(frames), width, height, fps, audio, video_path)


# ═══════════════════════════════════════════════════════════════════════════════
#                         NODE: READ IMAGE SEQUENCE
# ═══════════════════════════════════════════════════════════════════════════════


class RadianceReadSequence:
    """
    Professional Image Sequence Reader.
    Reads folder/glob patterns, supports EXR/HDR/PNG/JPG/TIFF.
    Outputs IMAGE batch + metadata for full downstream compatibility.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "C:/Projects/Sequences/Shot01"}),
                "pattern": (
                    "STRING",
                    {
                        "default": "*.exr",
                        "tooltip": "Glob pattern (e.g. *.png, Shot_*.exr)",
                    },
                ),
                "start_frame": (
                    "INT",
                    {
                        "default": 1001,
                        "min": 0,
                        "max": 999999,
                        "tooltip": "Frame number in filename, or 0 for auto-detect first",
                    },
                ),
                "frame_limit": (
                    "INT",
                    {"default": 0, "min": 0, "max": 99999, "tooltip": "0 = Load all"},
                ),
                "input_colorspace": (INPUT_COLORSPACES, {"default": "Linear (sRGB)"}),
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.001,
                        "tooltip": "Assumed FPS for image sequences",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "IMAGE",
        "MASK",
        "frame_count",
        "width",
        "height",
        "fps",
        "filename_list",
    )
    FUNCTION = "read_sequence"
    CATEGORY = "FXTD Studios/Radiance/IO"
    DESCRIPTION = "Professional image sequence reader with EXR/HDR/PNG support and color transforms."

    def read_sequence(
        self, folder_path, pattern, start_frame, frame_limit, input_colorspace, fps=24.0
    ):
        folder_path = get_safe_input_path(folder_paths.get_input_directory(), folder_path)

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        search_path = os.path.join(folder_path, pattern)
        files = sorted(glob.glob(search_path))

        if not files:
            raise FileNotFoundError(
                f"No files found matching '{pattern}' in {folder_path}"
            )

        # Find start frame by number in filename
        startIndex = 0
        if start_frame > 0:
            found = False
            str_frame = str(start_frame)
            for idx, f in enumerate(files):
                if str_frame in os.path.basename(f):
                    startIndex = idx
                    found = True
                    break
            if not found:
                logger.warning(
                    f"Could not find file with frame number {start_frame}, starting from first file."
                )

        files = files[startIndex:]
        if frame_limit > 0:
            files = files[:frame_limit]

        logger.info(f"Reading sequence: {len(files)} frames from {folder_path}")

        images = []
        masks = []
        filenames_str = "\n".join([os.path.basename(f) for f in files])

        img_width = 0
        img_height = 0

        for fpath in files:
            os.path.splitext(fpath)[1].lower()
            img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)

            if img is None:
                logger.warning(f"Failed to load image: {fpath}")
                continue

            # Handle channels
            if img.ndim == 2:
                h, w = img.shape
                img = np.stack([img, img, img], axis=-1)
                mask = np.ones((h, w), dtype=np.float32)
            elif img.ndim == 3:
                h, w, c = img.shape
                if c == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    mask = np.ones((h, w), dtype=np.float32)
                elif c == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                    mask = img[..., 3].astype(np.float32)
                    if img.dtype == np.uint8:
                        mask = mask / 255.0
                    elif img.dtype == np.uint16:
                        mask = mask / 65535.0
                    img = img[..., :3]

            # Normalize to float32
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0
            else:
                img = img.astype(np.float32)

            if img_width == 0:
                img_height, img_width = img.shape[:2]

            images.append(torch.from_numpy(img))
            masks.append(torch.from_numpy(mask))

        if not images:
            raise RuntimeError("No valid images loaded from sequence")

        batch_img = torch.stack(images)
        batch_mask = torch.stack(masks)
        batch_img = apply_input_transform(batch_img, input_colorspace)

        return (
            batch_img,
            batch_mask,
            len(images),
            img_width,
            img_height,
            fps,
            filenames_str,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#                         NODE: UNIVERSAL WRITE
# ═══════════════════════════════════════════════════════════════════════════════

# Output format categories
VIDEO_CODECS = [
    "H.264 High (8-bit)",
    "H.265 Main10 (10-bit)",
    "ProRes 422 HQ (10-bit)",
    "ProRes 4444 (12-bit)",
    "ProRes 4444 XQ (12-bit)",
]

OUTPUT_FORMATS = [
    "Video — MP4 (H.264)",
    "Video — MP4 (H.265 10-bit)",
    "Video — MOV (ProRes 422 HQ)",
    "Video — MOV (ProRes 4444)",
    "Video — MOV (ProRes 4444 XQ)",
    "Image Sequence — PNG (8-bit)",
    "Image Sequence — PNG (16-bit)",
    "Image Sequence — EXR (32-bit)",
    "Image Sequence — JPEG",
    "GIF (Animated)",
    "WEBP (Animated)",
]


class RadianceWrite:
    """
    Universal VFX Writer.
    Exports to video (H.264/H.265/ProRes), image sequences (PNG/EXR/JPEG),
    or animated formats (GIF/WEBP). Applies Output Transforms and passes
    IMAGE through for downstream chaining.
    """

    def __init__(self):
        self.output_dir = "output"
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "Radiance"}),
                "output_format": (OUTPUT_FORMATS, {"default": "Video — MP4 (H.264)"}),
                "fps": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.001},
                ),
                "quality": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 51,
                        "tooltip": "CRF for H.264/5 (lower=better). JPEG quality (0-100) for JPEG sequences. Unused for ProRes/PNG/EXR.",
                    },
                ),
                "output_color_space": (
                    INPUT_COLORSPACES,
                    {"default": "sRGB (Standard)"},
                ),
                "subfolder": ("STRING", {"default": ""}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = (
        "IMAGE",
        "STRING",
        "VHS_VIDEO",
    )
    RETURN_NAMES = (
        "IMAGE",
        "file_path",
        "video",
    )
    FUNCTION = "write"
    OUTPUT_NODE = True
    CATEGORY = "FXTD Studios/Radiance/IO"
    DESCRIPTION = "Universal writer: video, image sequences, GIF, WEBP — with IMAGE passthrough for chaining."

    def write(
        self,
        images,
        filename_prefix,
        output_format,
        fps,
        quality,
        output_color_space,
        subfolder="",
        audio=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        from folder_paths import get_output_directory

        full_output_dir = get_safe_output_dir(get_output_directory(), subfolder)

        timestamp = int(time.time())

        # Apply Output Transform (Linear → Target)
        images_out = apply_output_transform(images, output_color_space)
        images_np = images_out.cpu().numpy()  # (B, H, W, C)

        logger.info(
            f"Writing: {filename_prefix} | {output_format} | {output_color_space} | {len(images_np)} frames"
        )

        # ─── Route to appropriate writer ───────────────────────────────
        if output_format.startswith("Video"):
            filepath = self._write_video(
                images_np,
                filename_prefix,
                output_format,
                fps,
                quality,
                output_color_space,
                full_output_dir,
                timestamp,
                audio,
            )
        elif output_format.startswith("Image Sequence"):
            filepath = self._write_sequence(
                images_np,
                filename_prefix,
                output_format,
                quality,
                full_output_dir,
                timestamp,
            )
        elif "GIF" in output_format:
            filepath = self._write_gif(
                images_np, filename_prefix, fps, full_output_dir, timestamp
            )
        elif "WEBP" in output_format:
            filepath = self._write_webp(
                images_np, filename_prefix, fps, full_output_dir, timestamp
            )
        else:
            raise ValueError(f"Unknown output format: {output_format}")

        logger.info(f"Saved: {filepath}")

        # IMAGE passthrough — return original input images for chaining
        return (images, filepath, filepath)

    # ─── Video Writer (FFmpeg via imageio) ────────────────────────────────

    def _write_video(
        self, images_np, prefix, fmt, fps, quality, color_space, output_dir, ts, audio
    ):
        import imageio.v3 as iio

        # Determine codec and extension
        if "H.264" in fmt:
            ext, codec_name, pixel_format = "mp4", "libx264", "yuv420p"
            video_data = (np.clip(images_np, 0, 1) * 255).astype(np.uint8)
            ffmpeg_params = ["-crf", str(quality), "-preset", "slow"]
        elif "H.265" in fmt:
            ext, codec_name, pixel_format = "mp4", "libx265", "yuv420p10le"
            video_data = (np.clip(images_np, 0, 1) * 65535).astype(np.uint16)
            ffmpeg_params = [
                "-x265-params",
                f"profile=main10:crf={quality}",
                "-tag:v",
                "hvc1",
            ]
        elif "ProRes 422" in fmt:
            ext, codec_name, pixel_format = "mov", "prores_ks", "yuv422p10le"
            video_data = (np.clip(images_np, 0, 1) * 65535).astype(np.uint16)
            ffmpeg_params = ["-profile:v", "3", "-vendor", "apl0"]
        elif "ProRes 4444 XQ" in fmt:
            ext, codec_name, pixel_format = "mov", "prores_ks", "yuv444p12le"
            video_data = (np.clip(images_np, 0, 1) * 65535).astype(np.uint16)
            ffmpeg_params = ["-profile:v", "5", "-vendor", "apl0"]
        elif "ProRes 4444" in fmt:
            ext, codec_name, pixel_format = "mov", "prores_ks", "yuv444p12le"
            video_data = (np.clip(images_np, 0, 1) * 65535).astype(np.uint16)
            ffmpeg_params = ["-profile:v", "4", "-vendor", "apl0"]
        else:
            ext, codec_name, pixel_format = "mp4", "libx264", "yuv420p"
            video_data = (np.clip(images_np, 0, 1) * 255).astype(np.uint8)
            ffmpeg_params = ["-crf", "18", "-preset", "slow"]

        # Color metadata (NCLC)
        color_map = {
            "Linear (sRGB)": ("bt709", "linear", "bt709"),
            "sRGB (Standard)": ("bt709", "iec61966-2-1", "bt709"),
            "ARRI LogC3": ("bt709", "bt709", "bt709"),
            "ARRI LogC4": ("bt2020", "bt2020-10", "bt2020nc"),
            "Sony S-Log3": ("bt2020", "bt2020-10", "bt2020nc"),
            "Panasonic V-Log": ("bt2020", "bt2020-10", "bt2020nc"),
            "Canon Log 3": ("bt2020", "bt2020-10", "bt2020nc"),
            "RED Log3G10": ("bt2020", "bt2020-10", "bt2020nc"),
            "ACEScct": ("bt2020", "linear", "bt2020nc"),
            "DaVinci Intermediate": ("bt2020", "linear", "bt2020nc"),
        }
        if color_space in color_map:
            prim, trc, space = color_map[color_space]
            ffmpeg_params += [
                "-color_primaries",
                prim,
                "-color_trc",
                trc,
                "-colorspace",
                space,
            ]

        filepath = os.path.join(output_dir, f"{prefix}_{ts}.{ext}")

        iio.imwrite(
            filepath,
            video_data,
            fps=fps,
            codec=codec_name,
            pixelformat=pixel_format,
            macro_block_size=1,
            output_params=ffmpeg_params,
        )

        # Mux audio if available
        if audio is not None:
            self._mux_audio(filepath, audio, fps)

        return filepath

    # ─── Image Sequence Writer ───────────────────────────────────────────

    def _write_sequence(self, images_np, prefix, fmt, quality, output_dir, ts):
        """Write numbered image frames to a subfolder."""
        seq_dir = os.path.join(output_dir, f"{prefix}_{ts}")
        os.makedirs(seq_dir, exist_ok=True)

        padding = max(4, len(str(len(images_np))))

        for i, frame in enumerate(images_np):
            frame_num = str(i + 1).zfill(padding)

            if "EXR" in fmt:
                # 32-bit float EXR
                fpath = os.path.join(seq_dir, f"{prefix}.{frame_num}.exr")
                # OpenCV expects BGR for writing
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(fpath, bgr.astype(np.float32))

            elif "16-bit" in fmt:
                # 16-bit PNG
                fpath = os.path.join(seq_dir, f"{prefix}.{frame_num}.png")
                frame_16 = (np.clip(frame, 0, 1) * 65535).astype(np.uint16)
                bgr = cv2.cvtColor(frame_16, cv2.COLOR_RGB2BGR)
                cv2.imwrite(fpath, bgr)

            elif "JPEG" in fmt:
                fpath = os.path.join(seq_dir, f"{prefix}.{frame_num}.jpg")
                frame_8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                bgr = cv2.cvtColor(frame_8, cv2.COLOR_RGB2BGR)
                jpeg_quality = max(1, min(100, quality)) if quality > 0 else 95
                cv2.imwrite(fpath, bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

            else:
                # Default: 8-bit PNG
                fpath = os.path.join(seq_dir, f"{prefix}.{frame_num}.png")
                frame_8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                bgr = cv2.cvtColor(frame_8, cv2.COLOR_RGB2BGR)
                cv2.imwrite(fpath, bgr)

        logger.info(f"Image sequence saved: {len(images_np)} frames to {seq_dir}")
        return seq_dir

    # ─── GIF Writer ──────────────────────────────────────────────────────

    def _write_gif(self, images_np, prefix, fps, output_dir, ts):
        """Write animated GIF."""
        from PIL import Image as PILImage

        filepath = os.path.join(output_dir, f"{prefix}_{ts}.gif")
        duration_ms = int(1000 / fps)

        pil_frames = []
        for frame in images_np:
            frame_8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            pil_frames.append(PILImage.fromarray(frame_8))

        if pil_frames:
            pil_frames[0].save(
                filepath,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=True,
            )

        return filepath

    # ─── WEBP Writer ─────────────────────────────────────────────────────

    def _write_webp(self, images_np, prefix, fps, output_dir, ts):
        """Write animated WEBP."""
        from PIL import Image as PILImage

        filepath = os.path.join(output_dir, f"{prefix}_{ts}.webp")
        duration_ms = int(1000 / fps)

        pil_frames = []
        for frame in images_np:
            frame_8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            pil_frames.append(PILImage.fromarray(frame_8))

        if pil_frames:
            pil_frames[0].save(
                filepath,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0,
                lossless=True,
            )

        return filepath

    # ─── Audio Muxing Helper ─────────────────────────────────────────────

    def _mux_audio(self, video_path, audio, fps):
        """Mux ComfyUI AUDIO dict into the video file."""
        try:
            waveform = audio.get("waveform")
            sample_rate = audio.get("sample_rate", 44100)

            if waveform is None:
                return

            # Write audio to temp WAV
            import struct

            wav_np = waveform.squeeze(0).cpu().numpy()  # (channels, samples)
            channels, n_samples = wav_np.shape

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_wav = tmp.name

            # Write raw WAV
            with open(tmp_wav, "wb") as f:
                # WAV header
                data_size = n_samples * channels * 4
                f.write(b"RIFF")
                f.write(struct.pack("<I", 36 + data_size))
                f.write(b"WAVE")
                f.write(b"fmt ")
                f.write(struct.pack("<I", 16))  # chunk size
                f.write(struct.pack("<H", 3))  # format: IEEE float
                f.write(struct.pack("<H", channels))
                f.write(struct.pack("<I", sample_rate))
                f.write(struct.pack("<I", sample_rate * channels * 4))
                f.write(struct.pack("<H", channels * 4))
                f.write(struct.pack("<H", 32))  # bits per sample
                f.write(b"data")
                f.write(struct.pack("<I", data_size))
                # Interleave and write
                interleaved = wav_np.T.flatten().astype(np.float32)
                f.write(interleaved.tobytes())

            # Mux with FFmpeg
            with tempfile.NamedTemporaryFile(
                suffix=os.path.splitext(video_path)[1], delete=False
            ) as tmp:
                tmp_out = tmp.name

            mux_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-i",
                tmp_wav,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                tmp_out,
            ]
            result = subprocess.run(mux_cmd, capture_output=True, timeout=120)  # nosec B603

            if result.returncode == 0:
                os.replace(tmp_out, video_path)
                logger.info("Audio muxed successfully")
            else:
                logger.warning(f"Audio muxing failed: {result.stderr[:200]}")
                if os.path.exists(tmp_out):
                    os.unlink(tmp_out)

            os.unlink(tmp_wav)

        except Exception as e:
            logger.warning(f"Audio muxing error: {e}")


NODE_CLASS_MAPPINGS = {
    "RadianceReadVideo": RadianceReadVideo,
    "RadianceReadSequence": RadianceReadSequence,
    "RadianceWrite": RadianceWrite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceReadVideo": "◎ Radiance Read (Video)",
    "RadianceReadSequence": "◎ Radiance Read (Sequence)",
    "RadianceWrite": "◎ Radiance Write",
}
