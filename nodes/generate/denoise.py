import logging
import torch
import numpy as np
import cv2

logger = logging.getLogger("radiance")


def rgb_to_ycrcb(rgb: np.ndarray) -> np.ndarray:
    """
    Convert (H, W, 3) linear float32 RGB to YCrCb space.
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    Cr = (R - Y) / 1.5748
    Cb = (B - Y) / 1.8556
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cr = (r - y) / 1.5748
    cb = (b - y) / 1.8556
    return np.stack([y, cr, cb], axis=-1)


def ycrcb_to_rgb(ycrcb: np.ndarray) -> np.ndarray:
    """
    Convert (H, W, 3) linear float32 YCrCb back to RGB.
    Inverse of the above using analytical coefficients.
    R = Y + 1.5748 * Cr
    B = Y + 1.8556 * Cb
    G = Y - 0.468124273 * Cr - 0.187324273 * Cb
    """
    y, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
    r = y + 1.5748 * cr
    b = y + 1.8556 * cb
    g = y - 0.468124273 * cr - 0.187324273 * cb
    return np.stack([r, g, b], axis=-1)


def guided_filter_2d(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """
    Fast Guided Filter using OpenCV boxFilter.
    I: guidance image (H, W), float32
    p: input image to be filtered (H, W), float32
    r: local window radius
    eps: regularization parameter
    """
    ksize = (2 * r + 1, 2 * r + 1)
    mean_I = cv2.boxFilter(I, -1, ksize, borderType=cv2.BORDER_REFLECT)
    mean_p = cv2.boxFilter(p, -1, ksize, borderType=cv2.BORDER_REFLECT)
    mean_Ip = cv2.boxFilter(I * p, -1, ksize, borderType=cv2.BORDER_REFLECT)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, -1, ksize, borderType=cv2.BORDER_REFLECT)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, ksize, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, -1, ksize, borderType=cv2.BORDER_REFLECT)

    return mean_a * I + mean_b


def motion_compensate_frame(frame: np.ndarray, neighbor: np.ndarray, search_radius: int = 1) -> np.ndarray:
    """
    Highly optimized, vectorized block-matching motion compensation.
    Compares 9 spatial shifts of 'neighbor' against 'frame' (H, W, C)
    and selects the shift per-pixel that minimizes Sum of Absolute Differences (SAD).
    """
    H, W, C = frame.shape
    best_sad = np.full((H, W), np.inf, dtype=np.float32)
    compensated = neighbor.copy()
    
    active_frame = frame[..., :3] if C >= 3 else frame
    active_neighbor = neighbor[..., :3] if C >= 3 else neighbor
    
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            shifted = np.roll(active_neighbor, shift=(dy, dx), axis=(0, 1))
            diff = np.abs(active_frame - shifted)
            sad = diff.sum(axis=-1) if diff.ndim == 3 else diff
            
            mask = sad < best_sad
            best_sad[mask] = sad[mask]
            
            shifted_full = np.roll(neighbor, shift=(dy, dx), axis=(0, 1))
            compensated[mask] = shifted_full[mask]
            
    return compensated


def estimate_noise_profile(image_y: np.ndarray, patch_size: int = 32) -> float:
    """
    Scans the Luma (Y) channel for the flatest region to estimate pure camera noise.
    Splits the image into patch_size x patch_size blocks, calculates the standard
    deviation of each block, and returns the minimum standard deviation found.
    This minimum represents the noise floor of the sensor on a flat, detail-free area.
    """
    H, W = image_y.shape
    num_y = H // patch_size
    num_x = W // patch_size
    
    # Crop to exact multiple of patch_size to avoid index errors
    cropped_y = image_y[:num_y * patch_size, :num_x * patch_size]
    
    # Reshape into blocks: (num_y, patch_size, num_x, patch_size)
    # Then swap axes to get: (num_y, num_x, patch_size, patch_size)
    blocks = cropped_y.reshape(num_y, patch_size, num_x, patch_size).transpose(0, 2, 1, 3)
    
    # Flatten blocks to compute std per block: shape (num_blocks, patch_size * patch_size)
    flattened_blocks = blocks.reshape(-1, patch_size * patch_size)
    
    # Calculate standard deviation along the block pixels
    block_stds = np.std(flattened_blocks, axis=-1)
    
    # Filter out blocks that have zero standard deviation (padded black margins or absolute zeros)
    valid_stds = block_stds[block_stds > 1e-4]
    
    if len(valid_stds) == 0:
        return 0.015  # Robust default noise floor fallback (1.5% noise)
        
    min_std = float(np.min(valid_stds))
    
    # Find coordinates for diagnostic logging
    min_idx = np.argmin(block_stds)
    min_y = int((min_idx // num_x) * patch_size)
    min_x = int((min_idx % num_x) * patch_size)
    logger.debug(f"[RadianceDenoise] Auto-Profile: Measured noise floor sigma={min_std:.5f} at flat patch coordinate ({min_y}, {min_x})")
    
    return min_std


class RadianceDenoise:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "filter_type": (["Bilateral", "Guided"], {"default": "Bilateral",
                    "tooltip": "The core spatial denoising algorithm. Bilateral preserves edges; Guided runs faster, preserves finer detail, and avoids halos."}),
                "d": ("INT", {"default": 9, "min": 1, "max": 50,
                    "tooltip": "Filter diameter. In Bilateral mode, this defines the pixel neighborhood. In Guided mode, this translates to window radius."
                }),
                "sigmaColor": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 10.0, "step": 0.01,
                     "tooltip": (
                         "Color similarity threshold. High = smoother, but can lose edge detail. "
                         "For HDR images, scale is auto-adjusted if hdr_auto_sigma is ON. "
                         "This is bypassed if auto_profiling is enabled."
                     )},
                ),
                "sigmaSpace": (
                    "FLOAT",
                    {"default": 75.0, "min": 0.1, "max": 500.0, "step": 0.5,
                     "tooltip": "Spatial distance threshold. Higher = smoother across wider neighborhoods but slower in Bilateral mode."},
                ),
                "hdr_auto_sigma": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": (
                         "Highly recommended for HDR! Automatically scales the color similarity threshold "
                         "to match the local maximum range of the image, keeping denoise strength uniform."
                     )},
                ),
                "auto_profiling": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Enables fully automatic, hands-free noise profiling. Scans the image for the flatest region (sensor noise floor) and dynamically scales all thresholds."}
                ),
                "profile_multiplier": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                     "tooltip": "Adjusts the strength of the automatically profiled noise signature. Raise if noise remains; lower if details get soft."}
                ),
                "luma_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "Multiplier for spatial denoise strength on brightness (Luma). Lower this (e.g. 0.2) to keep natural fine grain."}
                ),
                "chroma_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "Multiplier for spatial denoise strength on colors (Chroma). Raise this to wash away annoying chromatic noise."}
                ),
                "high_freq_denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "Denoise strength for fine details and high-frequency pixel grain."}
                ),
                "mid_freq_denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "Denoise strength for medium textures and compression artifacts."}
                ),
                "low_freq_denoise": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "Denoise strength for large coarse gradient splotches."}
                ),
                "joint_chroma_guidance": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Enables Joint Guided filtering. Uses the sharp structural boundaries of the Luma channel to guide Chroma smoothing, preventing color bleeding."}
                ),
                "temporal_blend": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.05,
                     "tooltip": "Enables multi-frame temporal de-flickering. 0.0 is off. Higher values blend adjacent frames to stabilize video."}
                ),
                "temporal_radius": (
                    "INT",
                    {"default": 1, "min": 1, "max": 4, "step": 1,
                     "tooltip": "Temporal search window size. 1 searches 1 prev/next frame. 2 searches 2 prev/next frames, etc. Higher values de-flicker better but are slower."}
                ),
                "temporal_threshold": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.01, "max": 0.5, "step": 0.01,
                     "tooltip": "Flicker delta threshold. Lower values prevent ghosting/trailing by only blending static or slow-moving areas."}
                ),
                "motion_compensation": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Enables 9-directional block-matching motion compensation to align adjacent frames, preventing ghosting on moving objects."}
                ),
                "detail_recovery": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "Blends high-frequency detail from the original image back into the denoised image to recover skin pores/grain."}
                ),
                "sharpen_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "Adds a subtle post-sharpening (unsharp mask) to recover perceived edge crispness."}
                ),
                "view_mode": (
                    ["Denoised", "Noise Residual", "Luma (Y)", "Chroma (Cb/Cr)"],
                    {"default": "Denoised",
                     "tooltip": "Diagnostic view options. 'Noise Residual' is extremely helpful to see exactly what details are being removed."}
                )
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "denoise"

    DESCRIPTION = (
        "Ultimate professional-grade 32-bit float spatial-temporal denoising node. "
        "Splits Luma and Chroma, offers automatic hands-free noise profiling, "
        "applies 3-band multiscale frequency decomposition, provides joint guided chroma filtering, "
        "and implements multi-frame block-matching motion-compensated temporal stabilization."
    )

    def denoise(
        self,
        image,
        d,
        sigmaColor,
        sigmaSpace,
        hdr_auto_sigma=True,
        filter_type="Bilateral",
        auto_profiling=False,
        profile_multiplier=1.0,
        luma_strength=1.0,
        chroma_strength=1.0,
        high_freq_denoise=1.0,
        mid_freq_denoise=1.0,
        low_freq_denoise=0.5,
        joint_chroma_guidance=True,
        temporal_blend=0.0,
        temporal_radius=1,
        temporal_threshold=0.05,
        motion_compensation=True,
        detail_recovery=0.0,
        sharpen_strength=0.0,
        view_mode="Denoised"
    ):
        # Support fp16 and float32 by casting to float32
        img_np = image.cpu().float().numpy()

        output_batch = []
        batch_size, height, width, channels = img_np.shape

        # Stage 1: Motion-Compensated Temporal Denoising (Long Range)
        if batch_size > 1 and temporal_blend > 0.0:
            temporal_denoised = []
            for i in range(batch_size):
                frame = img_np[i]
                blended_frame = frame.copy()
                
                # Determine neighbor indices within temporal_radius
                neighbors_with_dist = []
                for dist in range(1, temporal_radius + 1):
                    if i - dist >= 0:
                        neighbors_with_dist.append((img_np[i - dist], dist))
                    if i + dist < batch_size:
                        neighbors_with_dist.append((img_np[i + dist], dist))
                
                if neighbors_with_dist:
                    if channels >= 3:
                        total_weight = np.ones_like(frame[..., :3])
                        for neighbor, dist in neighbors_with_dist:
                            if motion_compensation:
                                neighbor_aligned = motion_compensate_frame(frame, neighbor, search_radius=1)
                            else:
                                neighbor_aligned = neighbor
                                
                            diff = np.abs(frame[..., :3] - neighbor_aligned[..., :3])
                            # Decays based on both absolute difference and temporal distance
                            dist_factor = 1.0 / float(dist)
                            w = np.exp(- (diff / temporal_threshold) ** 2) * (temporal_blend * dist_factor / len(neighbors_with_dist))
                            
                            blended_frame[..., :3] += w * neighbor_aligned[..., :3]
                            total_weight += w
                        blended_frame[..., :3] /= total_weight
                    else:
                        total_weight = np.ones_like(frame)
                        for neighbor, dist in neighbors_with_dist:
                            if motion_compensation:
                                neighbor_aligned = motion_compensate_frame(frame, neighbor, search_radius=1)
                            else:
                                neighbor_aligned = neighbor
                                
                            diff = np.abs(frame - neighbor_aligned)
                            dist_factor = 1.0 / float(dist)
                            w = np.exp(- (diff / temporal_threshold) ** 2) * (temporal_blend * dist_factor / len(neighbors_with_dist))
                            
                            blended_frame += w * neighbor_aligned
                            total_weight += w
                        blended_frame /= total_weight
                        
                temporal_denoised.append(blended_frame)
            img_np = np.stack(temporal_denoised, axis=0)

        # Stage 2: Spatial Denoising via 3-Band Frequency Decomposition & Auto Profiling
        for i in range(batch_size):
            original_frame = image[i].cpu().float().numpy() if batch_size == image.shape[0] else img_np[i]
            frame = img_np[i]
            frame_max = float(frame.max())
            
            try:
                if channels >= 3:
                    rgb = frame[..., :3]
                    ycrcb = rgb_to_ycrcb(rgb)
                    y = ycrcb[..., 0]
                    cr = ycrcb[..., 1]
                    cb = ycrcb[..., 2]
                    
                    # 2.1 Measure Noise Floor via Grid Search if enabled
                    if auto_profiling:
                        measured_sigma = estimate_noise_profile(y, patch_size=32)
                        base_sigma = measured_sigma * profile_multiplier
                    else:
                        base_sigma = sigmaColor
                    
                    # Radii definitions for L/M/H bands
                    r_mid = max(1, d // 2)
                    r_low = max(2, int(d * 1.5))
                    
                    # 2.2 Frequency decomposition on Luma (Y)
                    y_low = cv2.boxFilter(y, -1, (2 * r_low + 1, 2 * r_low + 1), borderType=cv2.BORDER_REFLECT)
                    y_mid_raw = cv2.boxFilter(y, -1, (2 * r_mid + 1, 2 * r_mid + 1), borderType=cv2.BORDER_REFLECT)
                    y_mid = y_mid_raw - y_low
                    y_high = y - y_mid_raw
                    
                    eff_sigma_y = base_sigma * luma_strength
                    if hdr_auto_sigma and frame_max > 1.01:
                        eff_sigma_y *= frame_max
                        
                    # Filter Y bands
                    y_high_denoised = self._denoise_band(y_high, high_freq_denoise, filter_type, max(3, d // 2), eff_sigma_y, sigmaSpace)
                    y_mid_denoised = self._denoise_band(y_mid, mid_freq_denoise, filter_type, d, eff_sigma_y, sigmaSpace)
                    y_low_denoised = self._denoise_band(y_low, low_freq_denoise, filter_type, d * 2, eff_sigma_y, sigmaSpace)
                    
                    denoised_y = y_high_denoised + y_mid_denoised + y_low_denoised
                    
                    # 2.3 Frequency decomposition on Chroma (Cr, Cb)
                    cr_low = cv2.boxFilter(cr, -1, (2 * r_low + 1, 2 * r_low + 1), borderType=cv2.BORDER_REFLECT)
                    cr_mid_raw = cv2.boxFilter(cr, -1, (2 * r_mid + 1, 2 * r_mid + 1), borderType=cv2.BORDER_REFLECT)
                    cr_mid = cr_mid_raw - cr_low
                    cr_high = cr - cr_mid_raw
                    
                    cb_low = cv2.boxFilter(cb, -1, (2 * r_low + 1, 2 * r_low + 1), borderType=cv2.BORDER_REFLECT)
                    cb_mid_raw = cv2.boxFilter(cb, -1, (2 * r_mid + 1, 2 * r_mid + 1), borderType=cv2.BORDER_REFLECT)
                    cb_mid = cb_mid_raw - cb_low
                    cb_high = cb - cb_mid_raw
                    
                    eff_sigma_c = base_sigma * chroma_strength
                    if hdr_auto_sigma and frame_max > 1.01:
                        eff_sigma_c *= frame_max
                        
                    # Joint cross-guidance settings
                    guide_high = y_high_denoised if joint_chroma_guidance else None
                    guide_mid = y_mid_denoised if joint_chroma_guidance else None
                    guide_low = y_low_denoised if joint_chroma_guidance else None
                    
                    cr_high_denoised = self._denoise_band(cr_high, high_freq_denoise, filter_type, max(3, d // 2), eff_sigma_c, sigmaSpace, guide_high)
                    cr_mid_denoised = self._denoise_band(cr_mid, mid_freq_denoise, filter_type, d, eff_sigma_c, sigmaSpace, guide_mid)
                    cr_low_denoised = self._denoise_band(cr_low, low_freq_denoise, filter_type, d * 2, eff_sigma_c, sigmaSpace, guide_low)
                    
                    cb_high_denoised = self._denoise_band(cb_high, high_freq_denoise, filter_type, max(3, d // 2), eff_sigma_c, sigmaSpace, guide_high)
                    cb_mid_denoised = self._denoise_band(cb_mid, mid_freq_denoise, filter_type, d, eff_sigma_c, sigmaSpace, guide_mid)
                    cb_low_denoised = self._denoise_band(cb_low, low_freq_denoise, filter_type, d * 2, eff_sigma_c, sigmaSpace, guide_low)
                    
                    denoised_cr = cr_high_denoised + cr_mid_denoised + cr_low_denoised
                    denoised_cb = cb_high_denoised + cb_mid_denoised + cb_low_denoised
                    
                    # Reconstruct RGB
                    denoised_ycrcb = np.stack([denoised_y, denoised_cr, denoised_cb], axis=-1)
                    denoised_rgb = ycrcb_to_rgb(denoised_ycrcb)
                    
                    if channels == 4:
                        denoised_frame = np.concatenate([denoised_rgb, frame[..., 3:]], axis=-1)
                    else:
                        denoised_frame = denoised_rgb
                else:
                    # Grayscale single band frequency decomposition & auto profiling
                    gray = frame.squeeze(-1)
                    
                    if auto_profiling:
                        measured_sigma = estimate_noise_profile(gray, patch_size=32)
                        base_sigma = measured_sigma * profile_multiplier
                    else:
                        base_sigma = sigmaColor
                        
                    r_mid = max(1, d // 2)
                    r_low = max(2, int(d * 1.5))
                    
                    gray_low = cv2.boxFilter(gray, -1, (2 * r_low + 1, 2 * r_low + 1), borderType=cv2.BORDER_REFLECT)
                    gray_mid_raw = cv2.boxFilter(gray, -1, (2 * r_mid + 1, 2 * r_mid + 1), borderType=cv2.BORDER_REFLECT)
                    gray_mid = gray_mid_raw - gray_low
                    gray_high = gray - gray_mid_raw
                    
                    eff_sigma = base_sigma * luma_strength
                    if hdr_auto_sigma and frame_max > 1.01:
                        eff_sigma *= frame_max
                        
                    gray_high_denoised = self._denoise_band(gray_high, high_freq_denoise, filter_type, max(3, d // 2), eff_sigma, sigmaSpace)
                    gray_mid_denoised = self._denoise_band(gray_mid, mid_freq_denoise, filter_type, d, eff_sigma, sigmaSpace)
                    gray_low_denoised = self._denoise_band(gray_low, low_freq_denoise, filter_type, d * 2, eff_sigma, sigmaSpace)
                    
                    denoised_gray = gray_high_denoised + gray_mid_denoised + gray_low_denoised
                    denoised_frame = denoised_gray[..., np.newaxis]
                        
            except cv2.error as e:
                logger.warning(
                    f"[RadianceDenoise] Filtering failed on frame {i} ({filter_type}, d={d}, "
                    f"sigmaColor={sigmaColor:.3f}): {e}. Returning unfiltered frame."
                )
                denoised_frame = frame

            # Stage 3: Detail Recovery
            if detail_recovery > 0.0:
                if channels >= 3:
                    high_freq = original_frame[..., :3] - denoised_frame[..., :3]
                    denoised_frame[..., :3] += detail_recovery * high_freq
                else:
                    high_freq = original_frame - denoised_frame
                    denoised_frame += detail_recovery * high_freq

            # Stage 4: Post-Sharpening
            if sharpen_strength > 0.0:
                try:
                    if channels >= 3:
                        rgb_part = denoised_frame[..., :3]
                        blurred = cv2.GaussianBlur(rgb_part, (5, 5), 1.0)
                        denoised_frame[..., :3] += sharpen_strength * (rgb_part - blurred)
                    else:
                        blurred = cv2.GaussianBlur(denoised_frame, (5, 5), 1.0)
                        if blurred.ndim == 2 and denoised_frame.ndim == 3:
                            blurred = blurred[..., np.newaxis]
                        denoised_frame += sharpen_strength * (denoised_frame - blurred)
                except Exception as e:
                    logger.warning(f"[RadianceDenoise] Post-sharpening failed: {e}")

            # Stage 5: View Diagnostics
            if view_mode == "Noise Residual":
                if channels >= 3:
                    res_rgb = original_frame[..., :3] - denoised_frame[..., :3] + 0.5
                    if channels == 4:
                        final_frame = np.concatenate([res_rgb, original_frame[..., 3:]], axis=-1)
                    else:
                        final_frame = res_rgb
                else:
                    final_frame = original_frame - denoised_frame + 0.5
            elif view_mode == "Luma (Y)" and channels >= 3:
                y_val = denoised_ycrcb[..., 0] if 'denoised_ycrcb' in locals() else rgb_to_ycrcb(denoised_frame[..., :3])[..., 0]
                y_rgb = np.stack([y_val, y_val, y_val], axis=-1)
                if channels == 4:
                    final_frame = np.concatenate([y_rgb, original_frame[..., 3:]], axis=-1)
                else:
                    final_frame = y_rgb
            elif view_mode == "Chroma (Cb/Cr)" and channels >= 3:
                cr_val = denoised_ycrcb[..., 1] if 'denoised_ycrcb' in locals() else rgb_to_ycrcb(denoised_frame[..., :3])[..., 1]
                cb_val = denoised_ycrcb[..., 2] if 'denoised_ycrcb' in locals() else rgb_to_ycrcb(denoised_frame[..., :3])[..., 2]
                chroma_ycrcb = np.stack([np.full_like(cr_val, 0.5), cr_val, cb_val], axis=-1)
                chroma_rgb = ycrcb_to_rgb(chroma_ycrcb)
                if channels == 4:
                    final_frame = np.concatenate([chroma_rgb, original_frame[..., 3:]], axis=-1)
                else:
                    final_frame = chroma_rgb
            else:
                final_frame = denoised_frame

            output_batch.append(final_frame)

        output_np = np.stack(output_batch, axis=0)
        output_tensor = torch.from_numpy(output_np)

        return (output_tensor,)

    def _denoise_band(
        self,
        band: np.ndarray,
        strength: float,
        filter_type: str,
        d_band: int,
        sigmaColor: float,
        sigmaSpace: float,
        guidance: np.ndarray = None
    ) -> np.ndarray:
        """Denoises a specific frequency band and blends based on user strength."""
        if strength <= 0.0:
            return band
        
        denoised = self._apply_filter(band, filter_type, d_band, sigmaColor, sigmaSpace, guidance)
        return band + strength * (denoised - band)

    def _apply_filter(
        self,
        channel_2d: np.ndarray,
        filter_type: str,
        d: int,
        sigmaColor: float,
        sigmaSpace: float,
        guidance: np.ndarray = None
    ) -> np.ndarray:
        """Applies Bilateral or Joint/Self Guided Filter on a single 2D float32 channel."""
        arr = channel_2d if channel_2d.flags["C_CONTIGUOUS"] else np.ascontiguousarray(channel_2d)
        
        if filter_type == "Guided":
            r = max(1, d // 2)
            eps = max(1e-6, float(sigmaColor) ** 2)
            guide = guidance if guidance is not None else arr
            guide = guide if guide.flags["C_CONTIGUOUS"] else np.ascontiguousarray(guide)
            return guided_filter_2d(guide, arr, r, eps)
        else:
            return cv2.bilateralFilter(arr, d, sigmaColor, sigmaSpace)


NODE_CLASS_MAPPINGS = {"RadianceDenoise": RadianceDenoise}

NODE_DISPLAY_NAME_MAPPINGS = {"RadianceDenoise": "◎ Radiance 32-bit Denoise"}
