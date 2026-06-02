import torch
import torch.nn.functional as F
import json
import logging
import numpy as np

logger = logging.getLogger("radiance.3d")

class RadianceCameraSync:
    """
    ◎ Radiance Camera Sync
    
    Imports 3D camera metadata (JSON/ABC) to drive optical nodes.
    Supports: Focal Length, F-Stop, Camera Transform, and Shutter info.
    
    Compatible with the Radiance Relight Engine and Motion Blur nodes.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_data": ("STRING", {"multiline": True, "default": "{}", "tooltip": "JSON camera data. Schema: { 'focal_length': 35.0, 'transform': [...], 'shutter': 180.0 }"}),
                "frame_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1, "tooltip": "Offset into the camera animation data. Shifts which frame of the camera track is used."}),
            },
            "optional": {
                "camera_file": ("STRING", {"default": "", "tooltip": "Path to a .json or .abc camera file."}),
            }
        }

    RETURN_TYPES = ("RADIANCE_CAMERA", "FLOAT", "FLOAT", "INT")
    RETURN_NAMES = ("camera", "focal_length", "f_stop", "shutter_angle")
    FUNCTION = "sync"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
    DESCRIPTION = "Synchronise camera transforms between 3D and compositing pipelines."

    def sync(self, camera_data, frame_offset, camera_file=""):
        # 1. Load Data
        data = {}
        if camera_file and camera_file.endswith(".json"):
            try:
                import os
                if os.path.exists(camera_file):
                    with open(camera_file, 'r') as f:
                        data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load camera file: {e}")
        else:
            try:
                data = json.loads(camera_data)
            except Exception:
                data = {}

        # 2. Extract Properties
        focal_length = data.get("focal_length", 35.0)
        f_stop = data.get("f_stop", 2.8)
        shutter_angle = data.get("shutter_angle", 180.0)
        transform = data.get("transform", np.eye(4).tolist())
        
        camera_obj = {
            "focal_length": focal_length,
            "f_stop": f_stop,
            "shutter_angle": shutter_angle,
            "transform": transform,
            "frame_offset": frame_offset
        }
        
        logger.info(f"[Camera Sync] Imported camera (Focal: {focal_length}mm, F-Stop: {f_stop})")
        
        return (camera_obj, float(focal_length), float(f_stop), int(shutter_angle))





NODE_CLASS_MAPPINGS = {
    "RadianceCameraSync": RadianceCameraSync,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceCameraSync": "◎ Radiance Camera Sync",
}
