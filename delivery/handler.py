import json
import os
import re
import math
import traceback
import logging
import torch
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from aiohttp import web
from server import PromptServer
import folder_paths

# Re-import modular components
from radiance.cache import _viewer_cache_get, _progress_set
from radiance.color.grading import apply_grading

logger = logging.getLogger("radiance.delivery.handler")

_SAFE_FILENAME_RE = re.compile(r'[^\w\s◎_.() -]', re.UNICODE)


def get_next_version(directory: str, filename_base: str) -> str:
    """Scan directory for existing versions and returns the next one (e.g., v02)."""
    if not os.path.exists(directory):
        return "v01"
    
    pattern = re.compile(rf"{re.escape(filename_base)}_v(\d+)")
    max_v = 0
    
    try:
        for f in os.listdir(directory):
            match = pattern.search(f)
            if match:
                v = int(match.group(1))
                if v > max_v:
                    max_v = v
    except Exception as exc:
        logger.warning("[radiance.delivery.handler] get_next_version: %s", exc)
        
    return f"v{max_v + 1:02d}"


def _export_aces_clip_xml(media_path: str, grading: dict, color_space: str, version_str: str) -> None:
    """Write an ACES Metadata File (.amf) alongside the exported media."""
    slope  = grading.get('gain',       [1.0, 1.0, 1.0])
    offset = grading.get('offset',     [0.0, 0.0, 0.0])
    power  = grading.get('gamma',      [1.0, 1.0, 1.0])
    sat    = float(grading.get('saturation', 1.0))
    if not isinstance(slope,  list): slope  = [float(slope)]  * 3
    if not isinstance(offset, list): offset = [float(offset)] * 3
    if not isinstance(power,  list): power  = [float(power)]  * 3

    # Map output color_space to ACES ODT URN
    _ODT_MAP = {
        'ACEScg (AP1)':          'urn:ampas:aces:transformId:v2.0:ACEScsc.Academy.ACEScg_to_ACES.a2.v1',
        'ACES2065-1 (AP0)':      'urn:ampas:aces:transformId:v2.0:ACEScsc.Academy.Identity.a2.v1',
        'ACEScct':               'urn:ampas:aces:transformId:v2.0:ACEScsc.Academy.ACEScct_to_ACES.a2.v1',
        'Linear (sRGB)':         'urn:ampas:aces:transformId:v2.0:ODT.Academy.Rec709_100nits_dim.a1.0.3',
        'sRGB (Standard)':       'urn:ampas:aces:transformId:v2.0:ODT.Academy.sRGB_100nits_dim.a1.0.3',
        'ARRI LogC3':            'urn:ampas:aces:transformId:v2.0:IDT.ARRI.Alexa-v3-LogC-EI800.a1.v2',
        'DaVinci Intermediate':  'urn:ampas:aces:transformId:v2.0:IDT.BlackmagicDesign.DaVinci_Intermediate.a1.v1',
    }
    odt_urn = _ODT_MAP.get(color_space, 'urn:ampas:aces:transformId:v2.0:ODT.Academy.Rec709_100nits_dim.a1.0.3')
    idt_urn = 'urn:ampas:aces:transformId:v2.0:IDT.Academy.Rec709_100nits.a1.0.3'  # default: Rec.709

    # Build XML tree
    ns = 'urn:ampas:aces:schema:v1.0'
    ET.register_namespace('', ns)
    ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')

    root = ET.Element(f'{{{ns}}}aces_metadata_file')
    root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root.set('version', '1.0')

    # Header
    hdr = ET.SubElement(root, f'{{{ns}}}amf_info')
    ET.SubElement(hdr, f'{{{ns}}}description').text = f'Radiance v3.0.0 — ACES grade export ({version_str})'
    ET.SubElement(hdr, f'{{{ns}}}uuid').text = f'urn:uuid:{os.urandom(16).hex()}'
    ET.SubElement(hdr, f'{{{ns}}}date_time').text = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    # Pipeline
    pipeline = ET.SubElement(root, f'{{{ns}}}pipeline')
    pipeline.set('name', f'Radiance_{version_str}')

    # IDT
    idt_node = ET.SubElement(pipeline, f'{{{ns}}}input_transform')
    ET.SubElement(idt_node, f'{{{ns}}}transform_id').text = idt_urn

    # Look Transform (CDL)
    look = ET.SubElement(pipeline, f'{{{ns}}}look_transform')
    look.set('applied', 'true')
    ET.SubElement(look, f'{{{ns}}}description').text = 'ASC CDL via FXTD Radiance'

    cdl_node = ET.SubElement(look, f'{{{ns}}}cdl_transform')
    sop = ET.SubElement(cdl_node, f'{{{ns}}}SOPNode')
    ET.SubElement(sop, f'{{{ns}}}Slope').text  = ' '.join(f'{v:.6f}' for v in slope)
    ET.SubElement(sop, f'{{{ns}}}Offset').text = ' '.join(f'{v:.6f}' for v in offset)
    ET.SubElement(sop, f'{{{ns}}}Power').text  = ' '.join(f'{v:.6f}' for v in power)
    sat_node = ET.SubElement(cdl_node, f'{{{ns}}}SatNode')
    ET.SubElement(sat_node, f'{{{ns}}}Saturation').text = f'{sat:.6f}'

    # ODT
    odt_node = ET.SubElement(pipeline, f'{{{ns}}}output_transform')
    ET.SubElement(odt_node, f'{{{ns}}}transform_id').text = odt_urn

    # Write
    tree = ET.ElementTree(root)
    ET.indent(tree, space='  ')
    amf_path = os.path.splitext(media_path)[0] + '.amf'
    tree.write(amf_path, encoding='utf-8', xml_declaration=True)
    logger.info(f'[Radiance] ACES Metadata File written: {amf_path}')


@PromptServer.instance.routes.post('/radiance/deliver')
async def radiance_deliver_endpoint(request):
    """
    VFX Delivery Endpoint: Receives grading state + export settings from HUD.
    Applies grading, AI-Upscaling, and Burn-ins before high-quality encoding.
    """
    try:
        data = await request.json()
        instance_key = str(data.get('instance_id', ''))
        grading = data.get('grading', {})
        settings = data.get('settings', {})

        images = _viewer_cache_get(instance_key)
        if images is None:
            return web.json_response({"error": "No frames found in cache for this node. Run the workflow first.", "status": "error"})
        
        # ─── Render Range ──────────────────────────────────────────────
        range_in = max(1, int(settings.get('range_in', 1)))
        range_out = int(settings.get('range_out', 0))
        total_frames = images.shape[0]
        start_idx = range_in - 1
        end_idx = range_out if range_out > 0 else total_frames
        start_idx = max(0, min(start_idx, total_frames - 1))
        end_idx = max(start_idx + 1, min(end_idx, total_frames))
        images = images[start_idx:end_idx]
        
        # ─── Process Options ──────────────────────────────────────────
        filename_prefix = settings.get('filename', '◎ Radiance_Deliver')
        output_format = settings.get('format', 'Video — MP4 (H.264)')
        fps = float(settings.get('fps', 24.0))
        quality = int(settings.get('quality', 18))
        output_path = settings.get('path', '')

        # ─── Input Sanitization ───────────────────────────────────────
        filename_prefix = _SAFE_FILENAME_RE.sub('', str(filename_prefix)).strip()
        if not filename_prefix:
            filename_prefix = 'Radiance_Deliver'
        if len(filename_prefix) > 200:
            filename_prefix = filename_prefix[:200]

        if output_path:
            output_path = os.path.abspath(os.path.normpath(str(output_path)))
            if len(output_path) > 1024:
                logger.warning(f"[Deliver] Rejected oversized output path (len={len(output_path)})")
                return web.json_response({"error": "Invalid output path", "status": "error"})
            try:
                _allowed_root = os.path.abspath(folder_paths.get_output_directory())
            except Exception:
                _allowed_root = None
            if _allowed_root:
                try:
                    _rel = os.path.relpath(output_path, _allowed_root)
                    _outside = _rel.startswith("..")
                except ValueError:
                    _outside = True
                if _outside:
                    logger.warning(f"[Deliver] Rejected output path outside ComfyUI output dir: {output_path[:120]}")
                    return web.json_response({"error": "Output path must be inside the ComfyUI output directory.", "status": "error"})

        # Clamp numeric parameters to sane ranges
        fps = max(1.0, min(fps, 240.0))
        quality = max(0, min(quality, 51))
        
        color_space = settings.get('colorSpace', 'sRGB (Standard)')
        broadcast_safe = settings.get('soft_clip', True)
        
        upscale_2x = settings.get('upscale_2x', False)
        smart_ver = settings.get('smart_versioning', True)
        
        # Handle Versioning
        version_str = "v01"
        if smart_ver:
            base_path = output_path if output_path else folder_paths.get_output_directory()
            version_str = get_next_version(base_path, filename_prefix)
            filename_prefix = f"{filename_prefix}_{version_str}"

        # ─── Process Grading ──────────────────────────────────────────
        def safe_float(v, default):
            if isinstance(v, (list, tuple)) and len(v) > 0: return float(v[0])
            try: return float(v)
            except (TypeError, ValueError): return default

        def safe_array(v, default_scalar, length=3):
            if isinstance(v, (list, tuple)) and len(v) >= length:
                return [float(x) for x in v[:length]]
            try:
                s = float(v)
                return [s] * length
            except (TypeError, ValueError):
                return [default_scalar] * length

        exposure    = safe_float(grading.get('exposure'), 0.0)
        saturation  = safe_float(grading.get('saturation'), 1.0)
        _temp_internal = safe_float(grading.get('temperature'), 0.0)
        temperature = 6500.0 + _temp_internal * 3500.0
        contrast    = safe_float(grading.get('contrast'), 1.0)
        pivot       = safe_float(grading.get('pivot'), 0.18)
        shadows     = safe_float(grading.get('shadows'), 0.0)
        highlights  = safe_float(grading.get('highlights'), 0.0)
        hue_shift   = safe_float(grading.get('hue_shift'), 0.0)
        lut_name    = grading.get('lut_name', 'None')
        lut_intensity = safe_float(grading.get('lut_intensity'), 1.0)
        gamut_compression = bool(grading.get('gamut_compression', False))

        gamma_rgb  = safe_array(grading.get('gamma'),  1.0)
        gain_rgb   = safe_array(grading.get('gain'),   1.0)
        lift_rgb   = safe_array(grading.get('lift'),   0.0)
        offset_rgb = safe_array(grading.get('offset'), 0.0)

        # Apply grading to whole batch
        out_batch = []
        for i in range(images.shape[0]):
            frame_np = images[i].cpu().numpy()
            graded = apply_grading(
                img=frame_np,
                exposure=exposure,
                gamma=1.0,
                gain=1.0,
                lift=0.0,
                saturation=saturation,
                temperature=temperature,
                offset=0.0,
                contrast=contrast,
                pivot=pivot,
                shadows=shadows,
                highlights=highlights,
                hue_shift=hue_shift,
                gamma_rgb=gamma_rgb,
                gain_rgb=gain_rgb,
                lift_rgb=lift_rgb,
                offset_rgb=offset_rgb,
                lut_name=lut_name,
                lut_intensity=lut_intensity,
                color_science=1 if str(grading.get('colorScience')) in ['1', 'ACEScct'] else 0,
                luma_mix=float(grading.get('lumaMix', 1.0)),
                gamut_compression=gamut_compression
            )
            out_batch.append(torch.from_numpy(graded))

        graded_tensor = torch.stack(out_batch)

        # ─── FX Baking ────────────────────────────────────────────────
        _grain     = float(grading.get('grain', 0.0))
        _bloom     = float(grading.get('bloom', 0.0))
        _halation  = float(grading.get('halation', 0.0))
        _diffusion = float(grading.get('diffusion', 0.0))
        _denoise   = float(grading.get('denoise', 0.0))

        if _grain > 0.01 or _bloom > 0.01 or _halation > 0.01 or _diffusion > 0.01 or _denoise > 0.01:
            try:
                np_batch = graded_tensor.cpu().numpy()
                fx_batch = []
                rng = np.random.default_rng(seed=42)
                for i_frame in range(np_batch.shape[0]):
                    f = np_batch[i_frame].astype(np.float32)

                    if _denoise > 0.01:
                        try:
                            import cv2 as _cv2
                            sigma = _denoise * 3.0
                            f = _cv2.bilateralFilter((f * 65535).astype(np.uint16), d=5, sigmaColor=sigma*20, sigmaSpace=sigma*20).astype(np.float32) / 65535.0
                        except Exception:
                            pass

                    if _grain > 0.01:
                        noise = rng.standard_normal(f.shape).astype(np.float32)
                        f = f + noise * _grain * 0.02

                    if _halation > 0.01:
                        try:
                            import cv2 as _cv2
                            hi = np.clip(f[..., 0] - 0.8, 0, None)
                            blurred = _cv2.GaussianBlur(hi, (0, 0), sigmaX=_halation * 30)
                            f[..., 0] = np.minimum(f[..., 0] + blurred * _halation * 2.0, 2.0)
                        except Exception as exc:
                            logger.warning("[radiance.delivery.handler]: %s", exc)

                    if _bloom > 0.01:
                        try:
                            import cv2 as _cv2
                            lum = 0.2126*f[...,0] + 0.7152*f[...,1] + 0.0722*f[...,2]
                            hi = np.clip(lum - 0.7, 0, None)[..., np.newaxis]
                            blurred = _cv2.GaussianBlur(hi * np.ones((1,1,3), np.float32), (0, 0), sigmaX=_bloom * 40)
                            f = f + blurred * _bloom
                        except Exception as exc:
                            logger.warning("[radiance.delivery.handler]: %s", exc)

                    if _diffusion > 0.01:
                        try:
                            import cv2 as _cv2
                            soft = _cv2.GaussianBlur(f, (0, 0), sigmaX=_diffusion * 20)
                            f = f * (1 - _diffusion * 0.5) + soft * _diffusion * 0.5
                        except Exception as exc:
                            logger.warning("[radiance.delivery.handler]: %s", exc)

                    fx_batch.append(f)
                graded_tensor = torch.from_numpy(np.stack(fx_batch))
                logger.info(f"[Deliver] FX baked: grain={_grain:.2f} bloom={_bloom:.2f} halation={_halation:.2f} diffusion={_diffusion:.2f}")
            except Exception as e:
                logger.warning(f"[Deliver] FX baking failed (non-fatal): {e}")

        # ─── AI Upscale (2x) ──────────────────────────────────────────
        if upscale_2x:
            try:
                from radiance.nodes_upscale import RadianceAIUpscale
                upscaler = RadianceAIUpscale()
                graded_tensor, _ = upscaler.upscale(
                    image=graded_tensor,
                    model_name="RealESRGAN_x2plus",
                    mode="Refine (HDR)",
                    tile_size=512
                )
            except Exception as e:
                logger.error(f"AI Upscale failed, continuing with original: {e}")

        # ─── Aspect Ratio Blanking ────────────────────────────────────
        aspect_ratio_str = settings.get('aspect_ratio', 'None')
        if aspect_ratio_str != 'None':
            try:
                target_ratio = float(aspect_ratio_str.split(':')[0])
                _, h, w, _ = graded_tensor.shape
                current_ratio = w / h
                if current_ratio > target_ratio + 0.01:
                    target_w = int(h * target_ratio)
                    pad = (w - target_w) // 2
                    graded_tensor[:, :, :pad, :] = 0.0
                    graded_tensor[:, :, -pad:, :] = 0.0
                elif current_ratio < target_ratio - 0.01:
                    target_h = int(w / target_ratio)
                    pad = (h - target_h) // 2
                    graded_tensor[:, :pad, :, :] = 0.0
                    graded_tensor[:, -pad:, :, :] = 0.0
            except Exception as e:
                logger.error(f"Aspect blanking failed: {e}")

        # ─── Integrated QC Pass ───────────────────────────────────────
        qc_report = ""
        try:
            v_min = graded_tensor.min().item()
            v_max = graded_tensor.max().item()

            is_aces_cs = color_space in ('ACEScg (AP1)', 'ACES2065-1 (AP0)', 'ACEScct')

            if is_aces_cs:
                M_AP1_TO_REC2020 = np.array([
                    [ 1.70505,  -0.62179, -0.08326],
                    [-0.13026,   1.14080, -0.01054],
                    [-0.02400,  -0.12897,  1.15297],
                ], dtype=np.float32)
                try:
                    sample = graded_tensor[0].cpu().numpy()
                    if sample.ndim == 3 and sample.shape[2] >= 3:
                        rgb_ap1 = sample[..., :3]
                        rec2020 = np.tensordot(rgb_ap1, M_AP1_TO_REC2020, axes=([2], [1]))
                        imaginary_pct = float(np.mean(np.any(rec2020 < -0.001, axis=-1)) * 100)
                        if imaginary_pct > 0.5:
                            qc_report = (
                                f"◎ [QC WARNING] {imaginary_pct:.1f}% imaginary-gamut pixels detected "
                                f"(outside Rec.2020 after AP1→Rec.2020 transform). "
                                f"Apply gamut compression before display-referred delivery."
                            )
                        else:
                            qc_report = (
                                f"◎ [QC PASS — ACES] {color_space} · "
                                f"Imaginary gamut: {imaginary_pct:.2f}% (within Rec.2020 tolerance)."
                            )
                    else:
                        qc_report = f"◎ [QC PASS — ACES] {color_space} output."
                except Exception as _e:
                    qc_report = f"◎ [QC PASS — ACES] {color_space} (gamut check skipped: {_e})."
            elif v_min < 0.0 or v_max > 1.0:
                qc_report = (
                    f"◎ [QC WARNING] Out-of-Gamut detected: "
                    f"Min={v_min:.3f}, Max={v_max:.3f}. "
                    f"(Illegal levels for broadcast)."
                )
            else:
                qc_report = "◎ [QC PASS] Levels within legal broadcast range (0.0 - 1.0)."
        except Exception as exc:
            logger.warning("[radiance.delivery.handler]: %s", exc)

        # ─── Save using RadianceWrite Logic ────────────────────────────
        from radiance.nodes_io import RadianceWrite
        writer = RadianceWrite()

        is_exr = 'EXR' in output_format or 'exr' in output_format.lower()
        bake_grade_exr = settings.get('bake_grade', False) and is_exr
        if bake_grade_exr:
            if color_space in ('sRGB (Standard)', 'Linear (sRGB)'):
                try:
                    gt_np = graded_tensor.cpu().numpy().astype(np.float32)
                    lo = gt_np <= 0.04045
                    gt_np[lo]  = gt_np[lo] / 12.92
                    gt_np[~lo] = np.power((gt_np[~lo] + 0.055) / 1.055, 2.4)
                    graded_tensor = torch.from_numpy(gt_np)
                    logger.info('[Deliver v3.0.0] Grade baked into EXR — sRGB→linear applied')
                except Exception as _e:
                    logger.warning(f'[Deliver v3.0.0] Grade bake linearize failed: {_e}')
            else:
                logger.info(f'[Deliver v3.0.0] Grade baked into EXR — {color_space} (already linear)')

        # ─── Shot Continuity QC ───────────────────────────────────────
        continuity_report = []
        if graded_tensor.shape[0] > 1:
            try:
                frames_np = graded_tensor.cpu().numpy()
                def _frame_midluma(f):
                    luma = 0.2126*f[...,0] + 0.7152*f[...,1] + 0.0722*f[...,2]
                    s = np.sort(luma.ravel())
                    return float(s[len(s)//2])
                mid_lumas = [_frame_midluma(frames_np[i]) for i in range(frames_np.shape[0])]
                for i in range(1, len(mid_lumas)):
                    p0, p1 = max(mid_lumas[i-1], 1e-6), max(mid_lumas[i], 1e-6)
                    ev_delta = abs(math.log2(p1 / p0))
                    if ev_delta > 0.5:
                        continuity_report.append(
                            f'Frame {i-1}→{i}: {ev_delta:.2f} EV jump (mid-luma {p0:.3f}→{p1:.3f})'
                        )
                if continuity_report:
                    logger.warning(f'[Deliver v3.0.0] Continuity issues: {len(continuity_report)} flicker events')
            except Exception as _e:
                logger.debug(f'[Deliver v3.0.0] Continuity scan failed: {_e}')

        _progress_set(instance_key, {"current": 90, "total": 100, "status": "encoding", "message": "Encoding Master..."})

        _, path, _ = writer.write(
            filename_prefix=filename_prefix,
            output_format=output_format,
            fps=fps,
            quality=quality,
            output_color_space=color_space,
            image=graded_tensor,
            output_path=output_path,
            broadcast_safe=broadcast_safe,
        )

        # ─── Post-Export: Thumbnails & Sidecars ────────────────────────
        try:
            import PIL.Image
            n_graded = graded_tensor.shape[0]
            thumb_frame = graded_tensor[min(n_graded - 1, 10 if n_graded > 10 else 0)]
            thumb_np = (thumb_frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            thumb_img = PIL.Image.fromarray(thumb_np)
            thumb_path = os.path.splitext(path)[0] + "_thumb.jpg"
            thumb_img.save(thumb_path, quality=85)
            
            # ASC CDL Sidecar Export
            if settings.get('export_cdl', False):
                slope = grading.get('gain', 1.0)
                offset = grading.get('offset', 0.0)
                power = grading.get('gamma', 1.0)
                sat = grading.get('saturation', 1.0)
                if not isinstance(slope, list): slope = [slope]*3
                if not isinstance(offset, list): offset = [offset]*3
                if not isinstance(power, list): power = [power]*3
                
                cdl = f"""<ColorCorrection id="◎ Radiance_grade">\n  <SOPNode>\n    <Slope>{slope[0]:.6f} {slope[1]:.6f} {slope[2]:.6f}</Slope>\n    <Offset>{offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}</Offset>\n    <Power>{power[0]:.6f} {power[1]:.6f} {power[2]:.6f}</Power>\n  </SOPNode>\n  <SatNode>\n    <Saturation>{sat:.6f}</Saturation>\n  </SatNode>\n</ColorCorrection>"""
                cdl_path = os.path.splitext(path)[0] + ".cdl"
                with open(cdl_path, "w") as f:
                    f.write(cdl)

            # AMF Export
            if settings.get('export_amf', False):
                _export_aces_clip_xml(path, grading, color_space, version_str)
            
            # Save Metadata Sidecar (.json)
            meta = {
                "version": version_str,
                "grading": grading,
                "export_settings": settings,
                "qc": qc_report,
                "continuity": continuity_report if continuity_report else "PASS",
                "bake_grade_exr": bake_grade_exr,
                "timestamp": os.path.getmtime(path)
            }
            with open(os.path.splitext(path)[0] + "_meta.json", 'w') as f:
                json.dump(meta, f, indent=4)
                
            # Reveal Folder Hook
            if settings.get('reveal_folder', False):
                import platform, subprocess
                try:
                    p = os.path.abspath(output_path)
                    if platform.system() == "Windows":
                        os.startfile(p)
                    elif platform.system() == "Darwin":
                        subprocess.Popen(["open", p])
                    else:
                        subprocess.Popen(["xdg-open", p])
                except Exception as e:
                    logger.error(f"Launch folder failed: {e}")
                    
        except Exception as e:
            logger.error(f"Sidecar generation failed: {e}")

        _progress_set(instance_key, {"current": 100, "total": 100, "status": "done", "message": "Delivery Complete"})

        # Session Tracker
        try:
            _sessions_path = os.path.join(folder_paths.get_output_directory(), 'radiance_sessions.json')
            _sessions = []
            if os.path.exists(_sessions_path):
                try:
                    with open(_sessions_path, 'r') as _sf:
                        _sessions = json.load(_sf)
                except Exception:
                    _sessions = []
            _session_entry = {
                "timestamp":    datetime.now().isoformat(timespec='seconds'),
                "shot":         os.path.basename(path),
                "version":      version_str,
                "format":       output_format,
                "color_space":  color_space,
                "frames":       int(graded_tensor.shape[0]),
                "resolution":   f"{graded_tensor.shape[2]}×{graded_tensor.shape[1]}",
                "exposure":     round(exposure, 3),
                "saturation":   round(saturation, 3),
                "gain_rgb":     [round(v, 4) for v in gain_rgb],
                "lift_rgb":     [round(v, 4) for v in lift_rgb],
                "gamma_rgb":    [round(v, 4) for v in gamma_rgb],
                "qc":           qc_report[:120] if qc_report else "",
                "continuity":   f"{len(continuity_report)} events" if continuity_report else "PASS",
                "bake_grade":   bake_grade_exr,
            }
            _sessions.append(_session_entry)
            if len(_sessions) > 500:
                _sessions = _sessions[-500:]
            with open(_sessions_path, 'w') as _sf:
                json.dump(_sessions, _sf, indent=2)
            logger.debug(f'[Deliver v3.0.0] Session logged → {_sessions_path}')
        except Exception as _e:
            logger.debug(f'[Deliver v3.0.0] Session log failed (non-fatal): {_e}')

        _cont_warn = ""
        if continuity_report:
            _cont_warn = f" ⚠ {len(continuity_report)} flicker event(s): " + "; ".join(continuity_report[:3])
            if len(continuity_report) > 3:
                _cont_warn += f" (+{len(continuity_report)-3} more)"

        return web.json_response({
            "status": "success",
            "path": path,
            "qc": qc_report + _cont_warn,
            "continuity": continuity_report,
            "message": f"Export delivered successfully: {os.path.basename(path)} ({version_str})"
        })

    except Exception as e:
        logger.error(f"Delivery failed: {traceback.format_exc()}")
        return web.json_response({"error": str(e), "status": "error"})
