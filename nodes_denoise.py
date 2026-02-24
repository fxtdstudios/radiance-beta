import torch
import numpy as np
import cv2


class RadianceDenoise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "d": ("INT", {"default": 9, "min": 1, "max": 50}),
                "sigmaColor": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "sigmaSpace": (
                    "FLOAT",
                    {"default": 75.0, "min": 0.0, "max": 500.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = "FXTD Studios/Radiance/Filter"

    DESCRIPTION = "Removes noise while preserving edges using a 32-bit float compatible Bilateral Filter."

    def denoise(self, image, d, sigmaColor, sigmaSpace):
        # Image is [B, H, W, C]
        # Convert to numpy, assume it's already float32 (comfy standard)
        # Note: We must ensure we don't convert to uint8, so we preserve 32-bit float.
        img_np = image.cpu().numpy()

        output_batch = []

        batch_size, height, width, channels = img_np.shape

        for i in range(batch_size):
            frame = img_np[i]

            # OpenCV Bilateral Filter supports 1 or 3 channels.
            if channels == 4:
                # Separate Alpha
                rgb = frame[:, :, :3]
                alpha = frame[:, :, 3:]  # Keep (H, W, 1) to make concatenation easy

                # Check for contiguous array (OpenCV sometimes requires it)
                if not rgb.flags["C_CONTIGUOUS"]:
                    rgb = np.ascontiguousarray(rgb)

                # Denoise RGB
                denoised_rgb = cv2.bilateralFilter(rgb, d, sigmaColor, sigmaSpace)

                # Recombine
                denoised_frame = np.concatenate((denoised_rgb, alpha), axis=2)

            elif channels == 3:
                if not frame.flags["C_CONTIGUOUS"]:
                    frame = np.ascontiguousarray(frame)
                denoised_frame = cv2.bilateralFilter(frame, d, sigmaColor, sigmaSpace)

            elif channels == 1:
                # Grayscale
                if not frame.flags["C_CONTIGUOUS"]:
                    frame = np.ascontiguousarray(frame)

                # OpenCV usually expects (H,W) for single channel
                frame_2d = frame.squeeze(-1)

                denoised_2d = cv2.bilateralFilter(frame_2d, d, sigmaColor, sigmaSpace)
                # Ensure 3D shape (H, W, 1)
                denoised_frame = denoised_2d[:, :, np.newaxis]
            else:
                # Unsupported channel count, return original
                denoised_frame = frame

            output_batch.append(denoised_frame)

        output_tensor = torch.from_numpy(np.array(output_batch))

        return (output_tensor,)


NODE_CLASS_MAPPINGS = {"RadianceDenoise": RadianceDenoise}

NODE_DISPLAY_NAME_MAPPINGS = {"RadianceDenoise": "Radiance 32-bit Denoise"}
