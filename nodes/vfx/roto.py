import torch
import torch.nn.functional as F
import numpy as np
import logging
import json
import re

logger = logging.getLogger("radiance.vfx.roto")

class RadianceVectorMaskDraw:
    """
    ◎ Radiance Vector Mask Draw
    
    Renders a closed polygon or a smooth closed spline through the given
    points, with winding-number fill and distance-based edge anti-aliasing.

    Bezier_Spline passes a closed Catmull-Rom curve through every point (each
    span is the equivalent cubic Bezier with handles derived from the
    neighbours). It used to be ignored: both options drew the polygon.

    Points: a JSON list of [x, y], or pasted text with "x y" number pairs
    (Nuke shapes paste this way; their tangent handles are read as points).
    """
    
    DESCRIPTION = "Draw a single-frame filled mask from a closed polygon or smooth closed spline defined by pixel coordinates, with an anti-aliased edge. Accepts JSON points or pasted Nuke shape data."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "Output mask width in pixels. Match your plate."}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8, "tooltip": "Output mask height in pixels. Match your plate."}),
                "shape_type": (["Polygon", "Bezier_Spline"], {"default": "Polygon", "tooltip": "Polygon: straight edges between the points. Bezier_Spline: a smooth closed Catmull-Rom curve through every point."}),
                "points_data": ("STRING", {
                    "default": "[[128, 128], [384, 128], [384, 384], [128, 384]]",
                    "multiline": True,
                    "tooltip": "Paste a JSON list of [x, y] pixel coordinates (origin top-left, y down) or a Nuke control points block. At least 3 points; fewer gives an empty mask."
                }),
                "anti_alias_width": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 8.0, "step": 0.1, "tooltip": "Edge softness in pixels on each side of the outline. 0.05 or less gives a hard, aliased edge."}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("vector_mask",)
    FUNCTION = "draw"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX/Masking"

    def _parse_points(self, data: str) -> list:
        # 1. Try standard JSON parser
        try:
            cleaned = data.strip()
            if cleaned.startswith("[") or cleaned.startswith("{"):
                return json.loads(cleaned)
        except Exception as _exc:
            logger.debug(
                "[Radiance] _parse_points(): ignoring %s from `cleaned = data.strip()`: %s",
                type(_exc).__name__, _exc,
            )
            
        # 2. Try parsing raw float pairs (handles Nuke copy-paste shapes)
        # Find all blocks of float coordinates e.g. "{ 100.5 200.2 }" or just numbers
        coords = re.findall(r"([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)", data)
        if coords:
            return [[float(c[0]), float(c[1])] for c in coords]
            
        return []

    @staticmethod
    def _catmull_rom_closed(p: torch.Tensor, samples: int = 12) -> torch.Tensor:
        """Densify a closed point loop into a smooth centripetal-free (uniform)
        Catmull-Rom curve, `samples` vertices per span."""
        p0 = torch.roll(p, 1, 0)
        p1 = p
        p2 = torch.roll(p, -1, 0)
        p3 = torch.roll(p, -2, 0)
        t = torch.linspace(0, 1, samples + 1, device=p.device)[:-1].view(1, -1, 1)
        t2, t3 = t * t, t * t * t
        a, b, c, d = (x.unsqueeze(1) for x in (p0, p1, p2, p3))
        out = 0.5 * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2
                     + (-a + 3 * b - 3 * c + d) * t3)
        return out.reshape(-1, 2)

    def draw(self, width: int, height: int, shape_type: str, points_data: str, anti_alias_width: float):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pts = self._parse_points(points_data)
        
        if not pts or len(pts) < 3:
            # Empty fallback mask
            logger.warning("[Vector Roto] Not enough valid coordinates parsed. Returning blank mask.")
            return (torch.zeros((1, height, width), dtype=torch.float32, device=device),)
            
        # Convert points to Tensor [N, 2]
        pts_tensor = torch.tensor(pts, dtype=torch.float32, device=device)
        if shape_type == "Bezier_Spline":
            pts_tensor = self._catmull_rom_closed(pts_tensor, samples=12)
        N = pts_tensor.shape[0]
        
        # Grid coordinates
        y, x = torch.meshgrid(
            torch.linspace(0, height - 1, height, device=device),
            torch.linspace(0, width - 1, width, device=device),
            indexing="ij"
        )
        
        # Ray casting / Winding Number algorithm in vectorized Torch space
        # Winding number is computed by summing angular displacements around each pixel
        # shape [H, W, 1, 2] for coordinates
        px = x.unsqueeze(-1)
        py = y.unsqueeze(-1)
        
        # Segment start and end points
        # pts_tensor shape [N, 2]
        pt1 = pts_tensor
        pt2 = torch.roll(pts_tensor, -1, dims=0)
        
        # Vector from pixel to segment vertices
        # shape [H, W, N]
        dx1 = pt1[:, 0].view(1, 1, N) - px
        dy1 = pt1[:, 1].view(1, 1, N) - py
        dx2 = pt2[:, 0].view(1, 1, N) - px
        dy2 = pt2[:, 1].view(1, 1, N) - py
        
        # Compute angles to vertices
        theta1 = torch.atan2(dy1, dx1)
        theta2 = torch.atan2(dy2, dx2)
        
        # Compute angular difference
        dtheta = theta2 - theta1
        # Wrap to [-pi, pi]
        dtheta = torch.remainder(dtheta + np.pi, 2 * np.pi) - np.pi
        
        # Sum angles
        winding_number = torch.sum(dtheta, dim=-1) / (2 * np.pi)
        inside_mask = (torch.abs(winding_number) > 0.5).float()
        
        # Subpixel anti-aliasing via nearest boundary distance mapping (SDF)
        # Compute distance to each line segment
        # Segment vector: v = pt2 - pt1
        # Pixel vector: w = p - pt1
        if anti_alias_width > 0.05:
            dist_to_edges = []
            for i in range(N):
                p1 = pt1[i]
                p2 = pt2[i]
                v = p2 - p1
                l2 = torch.sum(v**2)
                
                # Projection factor t = dot(w, v) / |v|^2
                w_x = x - p1[0]
                w_y = y - p1[1]
                t = (w_x * v[0] + w_y * v[1]) / (l2 + 1e-12)
                t = torch.clamp(t, 0.0, 1.0)
                
                # Nearest point on segment
                proj_x = p1[0] + t * v[0]
                proj_y = p1[1] + t * v[1]
                
                # Distance
                dist = torch.sqrt((x - proj_x)**2 + (y - proj_y)**2)
                dist_to_edges.append(dist)
                
            min_dist = torch.stack(dist_to_edges, dim=-1).min(dim=-1)[0]
            
            # Smoothstep edge blend
            # Inside: distance represents distance from border inwards (not computed here for simplicity)
            # Edge transition:
            edge_blend = torch.clamp(0.5 - (min_dist / anti_alias_width) * (0.5 - inside_mask), 0.0, 1.0)
            final_mask = torch.where(min_dist <= anti_alias_width, edge_blend, inside_mask)
        else:
            final_mask = inside_mask
            
        logger.info(f"[Vector Roto] {shape_type}: {len(pts)} points, {N} edges drawn.")
        return (final_mask.unsqueeze(0),)


class RadianceVideoMaskPropagator:
    """
    ◎ Radiance Video Mask Propagator
    
    A GPU-native mask propagation system. Uses Dense Optical Flow vectors
    to warp and propagate roto masks dynamically across the video sequence timeline.
    """
    
    DESCRIPTION = "Fill empty frames of a mask sequence by warping neighbouring keyframe masks along optical flow. Frames that already contain a mask are kept as keyframes."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK", {"tooltip": "Mask sequence, one per frame. Frames with any mask content are keyframes; empty frames are filled by propagation."}),
                "flow_vectors": ("IMAGE", {"tooltip": "32-bit flow vectors from Radiance Optical Flow."}),
                "propagation_mode": (["Forward", "Backward", "Bidirectional"], {"default": "Bidirectional", "tooltip": "Forward carries masks from earlier frames, Backward from later frames. Bidirectional runs both and keeps the union (maximum) on filled frames."}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("propagated_masks",)
    FUNCTION = "propagate"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX/Masking"

    def propagate(self, masks: torch.Tensor, flow_vectors: torch.Tensor, propagation_mode: str):
        # masks shape [B, H, W]
        # flow_vectors shape [B, H, W, 3] where R=U (dx), G=V (dy)
        B, H, W = masks.shape
        device = masks.device
        
        if B <= 1:
            return (masks,)
            
        prop_masks = masks.clone()
        
        # Grid coordinates mapping [-1, 1] range for grid_sample
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij"
        )
        
        # Radiance Optical Flow convention (measured on a moving plate):
        # flow_vectors[i] lives on frame i and points to where each pixel was
        # in frame i-1, i.e. frame_i(p) ~ frame_{i-1}(p + flow_i(p));
        # flow_vectors[0] is zero. The old code read flow_vectors[i-1] with
        # the opposite sign and a half-pixel-off normalisation, so a mask
        # drifted the wrong way (IoU 0.6 after one frame, 0.0 after three).
        sx = 2.0 / max(W - 1, 1)
        sy = 2.0 / max(H - 1, 1)

        def _warp(src, gx, gy):
            g = torch.stack([gx, gy], dim=-1).unsqueeze(0)
            return F.grid_sample(src.unsqueeze(0).unsqueeze(0), g, mode="bilinear",
                                 padding_mode="zeros", align_corners=True).squeeze(0).squeeze(0)

        # Forward: frame i from frame i-1, sampled at p + flow_i(p).
        if propagation_mode in ("Forward", "Bidirectional"):
            for i in range(1, B):
                if torch.sum(masks[i]) < 1.0:
                    flow = flow_vectors[i].to(device)
                    prop_masks[i] = _warp(prop_masks[i - 1],
                                          grid_x + flow[..., 0] * sx,
                                          grid_y + flow[..., 1] * sy)

        # Backward: frame i from frame i+1, sampled at p - flow_{i+1}(p)
        # (the forward field at i+1 stands in for the inverse field).
        if propagation_mode in ("Backward", "Bidirectional"):
            for i in reversed(range(B - 1)):
                if torch.sum(masks[i]) < 1.0:
                    flow = flow_vectors[i + 1].to(device)
                    warped = _warp(prop_masks[i + 1],
                                   grid_x - flow[..., 0] * sx,
                                   grid_y - flow[..., 1] * sy)
                    prop_masks[i] = torch.max(prop_masks[i], warped)

        logger.info(f"[Video Mask Propagator] Propagated sequence ({propagation_mode} mode) along timeline.")
        return (prop_masks,)
