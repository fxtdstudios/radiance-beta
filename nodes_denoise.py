import logging
import torch
import numpy as np
import cv2

logger = logging.getLogger("Radiance")


class RadianceDenoise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "d": ("INT", {"default": 9, "min": 1, "max": 50}),
                "sigmaColor": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 10.0, "step": 0.01,
                     "tooltip": (
                         "Color similarity threshold for the bilateral filter. "
                         "For HDR images (pixel values > 1.0), enable 'hdr_auto_sigma' "
                         "so this is automatically scaled to the image's value range. "
                         "With fixed sigmaColor=0.15, HDR images receive 0% denoising."
                     )},
                ),
                "sigmaSpace": (
                    "FLOAT",
                    {"default": 75.0, "min": 0.0, "max": 500.0, "step": 0.1,
                     "tooltip": "Spatial neighborhood radius. Higher = smoother but slower."},
                ),
                "hdr_auto_sigma": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": (
                         "BUG-C FIX: When enabled, sigmaColor is multiplied by the "
                         "per-frame maximum pixel value so the filter works correctly "
                         "on HDR linear-light images (values > 1.0). "
                         "Leave enabled for 32-bit HDR workflows. "
                         "Disable only if you need a fixed, unscaled sigmaColor."
                     )},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = "FXTD Studios/Radiance/Filter"

    DESCRIPTION = (
        "32-bit float bilateral denoising. "
        "Preserves edges while smoothing noise. "
        "Works on SDR [0,1] and HDR linear-light images equally. "
        "Enable hdr_auto_sigma (default ON) for correct HDR operation."
    )

    def denoise(self, image, d, sigmaColor, sigmaSpace, hdr_auto_sigma=True):
        # BUG-A FIX: cast to float32 before .numpy().
        # cv2.bilateralFilter only supports 8-bit uint and 32-bit float.
        # If the input tensor is fp16 (possible in mixed-precision pipelines),
        # .numpy() gives float16 and cv2 raises:
        #   "Bilateral filtering is only implemented for 8u and 32f images"
        # Casting to float32 is safe — it also ensures the output is full-precision.
        img_np = image.cpu().float().numpy()   # always float32

        output_batch = []
        batch_size, height, width, channels = img_np.shape

        for i in range(batch_size):
            frame = img_np[i]  # (H, W, C), float32

            # BUG-C FIX: Auto-scale sigmaColor for HDR images.
            # cv2.bilateralFilter sigmaColor must be in the same units as pixel values.
            # With linear-light HDR data (e.g., sun at 10.0, highlights at 3.0),
            # a fixed sigmaColor=0.15 is far below the pixel value range — the filter
            # treats every neighbour as "different" and applies 0% smoothing.
            # Scaling by frame_max makes sigmaColor proportional to the image range,
            # restoring the intended denoising behaviour at any exposure level.
            effective_sigma = sigmaColor
            if hdr_auto_sigma:
                frame_max = float(frame.max())
                if frame_max > 1.01:   # only scale for genuine HDR content
                    effective_sigma = sigmaColor * frame_max
                    logger.debug(
                        f"[RadianceDenoise] Frame {i}: HDR max={frame_max:.3f}, "
                        f"sigmaColor scaled {sigmaColor:.3f} → {effective_sigma:.3f}"
                    )

            # BUG-D FIX: Wrap cv2 calls in try/except.
            # bilateralFilter can raise on pathological inputs (e.g., d larger than
            # some internal kernel limits on certain OpenCV builds). Rather than
            # crashing the entire ComfyUI queue, log a warning and return the
            # original frame unchanged.
            try:
                denoised_frame = self._filter_frame(frame, channels, d, effective_sigma, sigmaSpace, i)
            except cv2.error as e:
                logger.warning(
                    f"[RadianceDenoise] cv2.bilateralFilter failed on frame {i} "
                    f"(d={d}, sigmaColor={effective_sigma:.3f}, "
                    f"shape={frame.shape}): {e}. Returning original frame."
                )
                denoised_frame = frame

            output_batch.append(denoised_frame)

        # BUG-E FIX: np.stack guarantees a new contiguous array before torch.from_numpy.
        # np.array(list_of_arrays) can return a non-owning view in edge cases;
        # np.stack always allocates fresh contiguous memory, preventing torch
        # from holding a reference to a numpy array whose lifetime is unclear.
        output_np = np.stack(output_batch, axis=0)  # (B, H, W, C), float32, contiguous
        output_tensor = torch.from_numpy(output_np)

        return (output_tensor,)

    @staticmethod
    def _filter_frame(
        frame: np.ndarray,
        channels: int,
        d: int,
        sigmaColor: float,
        sigmaSpace: float,
        frame_idx: int,
    ) -> np.ndarray:
        """Apply cv2.bilateralFilter to a single (H, W, C) float32 frame."""

        def _ensure_contiguous(arr: np.ndarray) -> np.ndarray:
            return arr if arr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(arr)

        if channels == 4:
            # Separate alpha — bilateral filter only supports 1 or 3 channels
            rgb   = _ensure_contiguous(frame[:, :, :3])
            alpha = frame[:, :, 3:]  # (H, W, 1) — alpha needs no denoising

            denoised_rgb = cv2.bilateralFilter(rgb, d, sigmaColor, sigmaSpace)
            return np.concatenate((denoised_rgb, alpha), axis=2)

        elif channels == 3:
            return cv2.bilateralFilter(_ensure_contiguous(frame), d, sigmaColor, sigmaSpace)

        elif channels == 1:
            # cv2.bilateralFilter expects (H, W) for single-channel, not (H, W, 1)
            frame_2d     = _ensure_contiguous(frame.squeeze(-1))
            denoised_2d  = cv2.bilateralFilter(frame_2d, d, sigmaColor, sigmaSpace)
            return denoised_2d[:, :, np.newaxis]   # restore (H, W, 1)

        else:
            # BUG-D FIX: Log unsupported channel counts instead of silent pass-through.
            # channels==2 (e.g., UV maps) would previously return unfiltered data
            # with no indication that filtering was skipped.
            logger.warning(
                f"[RadianceDenoise] Frame {frame_idx}: unsupported channel count "
                f"{channels} (expected 1, 3, or 4). Returning original frame unfiltered."
            )
            return frame


NODE_CLASS_MAPPINGS = {"◎ RadianceDenoise": RadianceDenoise}

NODE_DISPLAY_NAME_MAPPINGS = {"◎ RadianceDenoise": "◎ Radiance 32-bit Denoise"}
