import os
import cv2
import numpy as np
import argparse
import sys
import logging
import concurrent.futures
from tqdm import tqdm

# Add parent directory to path so we can import internal radiance modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from color_utils import (
        linear_to_logc4, linear_to_logc3, linear_to_slog3, linear_to_vlog,
        apply_matrix_transform, SRGB_TO_ACESCG, tensor_linear_to_srgb
    )
    from hdr.io import write_exr_robust
except ImportError:
    # Fallback for standalone use outside of the radiance folder.
    # Constants match the official ARRI LogC4 specification (2023, ALEXA 35).
    # BUG FIX: previous fallback used A=4296.65, D=11.593 (no B/C offset)
    # which placed 18% grey at ~0.32 instead of the correct ~0.277.
    def linear_to_logc4(img):
        a = 2231.826309067637
        b = 0.9071358691330627
        c = 0.0928641308669373
        t = -0.0180569961199123
        s = 0.1135773173772412
        out = np.empty_like(img, dtype=np.float32)
        mask = img >= t
        out[mask]  = ((np.log2(a * img[mask] + 64.0) - 6.0) / 14.0) * b + c
        out[~mask] = (img[~mask] - t) / s
        return out
    
    SRGB_TO_ACESCG = np.array([
        [0.613097, 0.339523, 0.047379],
        [0.070194, 0.916354, 0.013452],
        [0.020616, 0.109570, 0.869815]
    ], dtype=np.float32)

    def apply_matrix_transform(img, matrix):
        return np.dot(img, matrix.T)

    def write_exr_robust(filepath, image, **kwargs):
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return cv2.imwrite(filepath, img_bgr)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("radiance.batch")

CURVE_MAP = {
    "logc4": linear_to_logc4,
    "linear": lambda x: x
}

def process_file(in_p, out_p, args):
    """Core processing function for a single file."""
    try:
        # 1. Load Source
        img = cv2.imread(in_p, cv2.IMREAD_UNCHANGED)
        if img is None: return False
        
        # Ensure RGB and float32
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        else:
            img = img.astype(np.float32)

        # 2. Exposure Shift
        if args.exposure != 0:
            img *= (2.0 ** args.exposure)

        # 3. Gamut Transform
        if args.acescg:
            img = apply_matrix_transform(img, SRGB_TO_ACESCG)

        # 4. Log Encoding
        curve_fn = CURVE_MAP.get(args.curve.lower(), linear_to_logc4)
        img = curve_fn(img)

        # 5. Output Save
        ext = os.path.splitext(out_p)[1].lower()
        if ext == ".png":
            img = np.clip(img, 0.0, 1.0)
            img_16 = (img * 65535).astype(np.uint16)
            cv2.imwrite(out_p, cv2.cvtColor(img_16, cv2.COLOR_RGB2BGR))
        elif ext == ".exr":
            write_exr_robust(out_p, img)
        
        return True
    except Exception as e:
        logger.error(f"Error processing {in_p}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Radiance HDR Batch Converter")
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--curve", default="logc4", help="Target log curve")
    parser.add_argument("--format", default="png", choices=["png", "exr"], help="Output file format")
    parser.add_argument("--acescg", action="store_true", help="Convert to ACEScg gamut")
    parser.add_argument("--exposure", type=float, default=0.0, help="Exposure shift in stops")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    files = [f for f in os.listdir(args.input) if f.lower().endswith(('.exr', '.hdr'))]
    logger.info(f"Radiance Batch: Processing {len(files)} files...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for f in files:
            in_p = os.path.join(args.input, f)
            out_p = os.path.join(args.output, f.rsplit('.', 1)[0] + f".{args.format}")
            futures.append(executor.submit(process_file, in_p, out_p, args))
        
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

if __name__ == "__main__":
    main()
