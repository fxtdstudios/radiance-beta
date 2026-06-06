"""Gaussian Splatting training nodes — COLMAP import and gsplat optimization."""
from __future__ import annotations

_CATEGORY = "FXTD STUDIOS/Radiance/Gaussian Splatting"


class RadianceColmapLoad:
    """Load a COLMAP sparse model into cameras + an initial SPLAT (CPU, no GPU)."""

    CATEGORY = _CATEGORY
    DESCRIPTION = "Read a COLMAP sparse reconstruction -> camera rig + point-cloud-initialized SPLAT."
    FUNCTION = "load"
    RETURN_TYPES = ("RAD_CAMERAS", "IMAGE", "SPLAT", "STRING")
    RETURN_NAMES = ("cameras", "images", "init_splat", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_dir": ("STRING", {
                    "default": "",
                    "tooltip": "COLMAP sparse model folder (cameras/images/points3D .bin or .txt).",
                }),
                "sh_degree": ("INT", {"default": 3, "min": 0, "max": 4}),
            },
            "optional": {
                "images_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Folder with the source images. When set, they are loaded in camera "
                               "order (matched by filename) ready for Splat Train.",
                }),
            },
        }

    def load(self, model_dir, sh_degree, images_dir=""):
        import torch
        from radiance.splatting.colmap import load_colmap, load_images
        from radiance.splatting.init import init_from_points

        cams, pts, cols, names = load_colmap(model_dir.strip().strip('"'))
        splat = init_from_points(pts, cols, sh_degree=int(sh_degree))

        images_dir = images_dir.strip().strip('"')
        if images_dir:
            arr = load_images(images_dir, names, cams.width, cams.height)
            images = torch.from_numpy(arr)
            img_note = f"{arr.shape[0]} images @ {cams.width}x{cams.height} (camera-ordered)"
        else:
            images = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            img_note = "none (set images_dir to load training images)"

        info = (f"COLMAP model loaded\n  cameras : {len(cams)}\n"
                f"  points  : {pts.shape[0]:,}\n  init    : {splat.count:,} gaussians (SH {sh_degree})\n"
                f"  images  : {img_note}")
        return (cams, images, splat, info)


class RadianceSplatTrain:
    """Fit a SPLAT to posed images via gsplat (needs gsplat + CUDA)."""

    CATEGORY = _CATEGORY
    DESCRIPTION = "Optimize a SPLAT against posed images (gsplat/CUDA). Image order must match cameras."
    FUNCTION = "train"
    RETURN_TYPES = ("SPLAT", "STRING")
    RETURN_NAMES = ("splat", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "cameras": ("RAD_CAMERAS",),
                "init_splat": ("SPLAT",),
                "steps": ("INT", {"default": 7000, "min": 1, "max": 100000}),
                "sh_degree": ("INT", {"default": 3, "min": 0, "max": 4}),
                "densify": ("BOOLEAN", {"default": True, "tooltip": "Adaptive density control (clone/split/prune). Off = plain optimization."}),
            }
        }

    def train(self, images, cameras, init_splat, steps, sh_degree, densify=True):
        from radiance.splatting.train import train as _train, TrainConfig

        pbar = None
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(int(steps))
        except Exception:
            pbar = None

        def _progress(step, total, loss):
            if pbar is not None:
                pbar.update_absolute(step)

        def _interrupt():
            try:
                import comfy.model_management as mm
                mm.throw_exception_if_processing_interrupted()
            except ImportError:
                pass

        cfg = TrainConfig(steps=int(steps), sh_degree=int(sh_degree), densify=bool(densify))
        splat = _train(images, cameras, init_splat, cfg,
                       progress=_progress, interrupt=_interrupt)
        return (splat, splat.info())


NODE_CLASS_MAPPINGS = {
    "RadianceColmapLoad": RadianceColmapLoad,
    "RadianceSplatTrain": RadianceSplatTrain,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceColmapLoad": "COLMAP Load",
    "RadianceSplatTrain": "Splat Train",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
