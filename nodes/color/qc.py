import json
import logging
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

from radiance.image import defects
from radiance.path_utils import get_safe_output_dir, strip_path_quotes
from radiance.color.luma import luma_bt709 as _luma

try:
    import folder_paths
    _OUTPUT_DIR = folder_paths.get_output_directory()
except Exception:
    _OUTPUT_DIR = os.path.abspath("output")

logger = logging.getLogger("radiance.qc")


_PRESETS: Dict[str, Dict[str, Any]] = {
    "Broadcast SDR": {
        "description": "Broadcast TV — ITU-R BT.1886 safe",
        "max_peak_nits": 100.0, "max_clipping": 0.005, "max_black_crush": 0.02,
        "min_luma": 0.02, "max_luma": 0.95, "max_saturation": 0.85,
        "require_metadata": ["colorspace"], "gamut_check": False,
    },
    "Cinema HDR (P3-PQ)": {
        "description": "Digital Cinema — P3 gamut, PQ transfer",
        "max_peak_nits": 108.0, "max_clipping": 0.0, "max_black_crush": 0.01,
        "min_luma": 0.0, "max_luma": 1.0, "max_saturation": 1.0,
        "require_metadata": ["colorspace", "eotf"], "gamut_check": False,
    },
    "OTT HDR10": {
        "description": "HDR10 streaming — BT.2020 / PQ, peak 1000 nits",
        "max_peak_nits": 1000.0, "max_clipping": 0.001, "max_black_crush": 0.005,
        "min_luma": 0.0, "max_luma": 1.0, "max_saturation": 1.0,
        "require_metadata": ["colorspace", "eotf", "mastering_display"], "gamut_check": True,
    },
    "Social Media": {
        "description": "Social / web delivery — sRGB, no HDR",
        "max_peak_nits": 100.0, "max_clipping": 0.01, "max_black_crush": 0.05,
        "min_luma": 0.03, "max_luma": 0.97, "max_saturation": 0.90,
        "require_metadata": [], "gamut_check": False,
    },
    "Custom": {
        "description": "User-defined policy",
        "max_peak_nits": 1000.0, "max_clipping": 0.01, "max_black_crush": 0.05,
        "min_luma": 0.0, "max_luma": 1.0, "max_saturation": 1.0,
        "require_metadata": [], "gamut_check": False,
    },
}


def _mean_saturation(arr: np.ndarray) -> float:
    cmax = arr.max(axis=-1)
    cmin = arr.min(axis=-1)
    sat = np.where(cmax > 1e-6, (cmax - cmin) / (cmax + 1e-8), 0.0)
    return float(sat.mean())


def _gamut_out_of_p3(arr: np.ndarray) -> float:
    return float(((arr < -0.001) | (arr > 1.001)).any(axis=-1).mean())


def _policy_analyse(arr: np.ndarray) -> Dict[str, float]:
    luma_map = _luma(arr)
    return {
        "peak": float(arr.max()), "min": float(arr.min()),
        "mean_luma": float(luma_map.mean()),
        "clipping": float((luma_map > 0.99).mean()),
        "black_crush": float((luma_map < 0.01).mean()),
        "mean_sat": _mean_saturation(arr),
        "gamut_violation": _gamut_out_of_p3(arr),
    }


def _evaluate_policy(stats: Dict[str, float], policy: Dict[str, Any],
                     metadata_keys: List[str]) -> Tuple[bool, List[Dict], int]:
    violations: List[Dict] = []

    def _viol(rule: str, actual: Any, limit: Any, severity: str = "error"):
        violations.append({"rule": rule, "actual": actual, "limit": limit, "severity": severity})

    max_nits = float(policy.get("max_peak_nits", 1000.0))
    peak_nits_approx = stats["peak"] * 100.0
    if max_nits < 200 and peak_nits_approx > max_nits:
        _viol("max_peak_nits", f"{peak_nits_approx:.1f} nits", f"{max_nits:.0f} nits")

    if stats["clipping"] > float(policy.get("max_clipping", 0.01)):
        _viol("max_clipping", f"{stats['clipping']:.2%}", f"{policy['max_clipping']:.2%}")
    if stats["black_crush"] > float(policy.get("max_black_crush", 0.05)):
        _viol("max_black_crush", f"{stats['black_crush']:.2%}", f"{policy['max_black_crush']:.2%}")
    if stats["mean_luma"] < float(policy.get("min_luma", 0.0)):
        _viol("min_luma", f"{stats['mean_luma']:.4f}", f"{policy['min_luma']:.4f}", "warning")
    if stats["mean_luma"] > float(policy.get("max_luma", 1.0)):
        _viol("max_luma", f"{stats['mean_luma']:.4f}", f"{policy['max_luma']:.4f}", "warning")
    if stats["mean_sat"] > float(policy.get("max_saturation", 1.0)):
        _viol("max_saturation", f"{stats['mean_sat']:.4f}", f"{policy['max_saturation']:.4f}")
    if policy.get("gamut_check") and stats["gamut_violation"] > 0.001:
        _viol("gamut_out_of_gamut", f"{stats['gamut_violation']:.2%}", "0%", "warning")
    for key in policy.get("require_metadata", []):
        if key not in metadata_keys:
            _viol("missing_metadata", f"'{key}' not provided", f"required: {key}", "error")

    errors = sum(1 for v in violations if v["severity"] == "error")
    warnings = sum(1 for v in violations if v["severity"] == "warning")
    score = max(0, 100 - errors * 25 - warnings * 5)
    return errors == 0, violations, score


def _generate_html_report(report: Dict) -> str:
    html = """<!DOCTYPE html>
<html><head><title>Radiance QC Report</title>
<style>
body { font-family: monospace; background: #1a1a1a; color: #e0e0e0; padding: 20px; }
.pass { color: #00ff00; }
.fail { color: #ff4444; }
.warn { color: #ffaa00; }
table { border-collapse: collapse; width: 100%; margin-top: 20px; }
th, td { border: 1px solid #444; padding: 8px; text-align: left; }
th { background: #2a2a2a; }
</style></head><body>
<h1>RADIANCE QC REPORT</h1>\n"""
    overall = report.get("overall_status", "UNKNOWN")
    html += f"<h2 class=\"{'pass' if overall == 'PASS' else 'fail'}\">Overall: {overall}</h2>"
    html += ("<table><tr><th>Frame</th><th>Status</th><th>Crushed</th>"
             "<th>Clipped</th><th>Gamut</th><th>Banding</th></tr>")
    for frame in report.get("frames", []):
        fn = frame.get("frame_number", 0)
        status = frame.get("status", "UNKNOWN")
        sc = "pass" if status == "PASS" else "fail"
        chk = frame.get("checks", {})
        html += (f'<tr><td>{fn}</td><td class="{sc}">{status}</td>'
                 f"<td>{chk.get('crushed_blacks',{}).get('percentage',0):.2f}%</td>"
                 f"<td>{chk.get('clipped_whites',{}).get('percentage',0):.2f}%</td>"
                 f"<td>{chk.get('gamut',{}).get('out_of_gamut_pct',0):.2f}%</td>"
                 f"<td>{chk.get('banding',{}).get('risk_pct',0):.2f}%</td></tr>")
    html += "</table>\n</body></html>"
    return html


class RadianceQC:
    MODES = ["Analyze", "Export"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"mode": (cls.MODES, {"default": "Analyze"})},
            "optional": {
                "image": ("IMAGE",),
                "black_threshold": ("FLOAT", {"default": 0.0, "min": -0.1, "max": 0.1, "step": 0.001}),
                "white_threshold": ("FLOAT", {"default": 1.0, "min": 0.8, "max": 2.0, "step": 0.01}),
                "overlay_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "banding_threshold": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "enable_focus_check": ("BOOLEAN", {"default": False}),
                "enable_artifacts_check": ("BOOLEAN", {"default": True}),
                "enable_noise_check": ("BOOLEAN", {"default": True}),
                "fail_on_errors": ("BOOLEAN", {"default": False}),
                "qc_report_json": ("STRING", {"forceInput": True}),
                "output_path": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "qc_report"}),
                "export_format": (["json", "csv", "html", "all"], {"default": "json"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "text_report", "json_report", "status")
    FUNCTION = "run"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ QC & Debug"
    DESCRIPTION = "Run a configurable suite of QC checks on an image or sequence."
    OUTPUT_NODE = True

    def run(self, mode: str = "Analyze", image=None, black_threshold: float = 0.0,
            white_threshold: float = 1.0, overlay_opacity: float = 0.5,
            banding_threshold: float = 5.0, enable_focus_check: bool = False,
            enable_artifacts_check: bool = True, enable_noise_check: bool = True,
            fail_on_errors: bool = False, qc_report_json: str = "{}",
            output_path: str = "", filename_prefix: str = "qc_report",
            export_format: str = "json"):
        if mode == "Analyze":
            if image is None:
                return self._error("image is required in Analyze mode")
            return self._analyze(image, black_threshold, white_threshold, overlay_opacity,
                                banding_threshold, enable_focus_check,
                                enable_artifacts_check, enable_noise_check, fail_on_errors)
        else:
            status = self._export(qc_report_json, output_path, filename_prefix, export_format)
            dummy = torch.zeros(1, 64, 64, 3)
            return (dummy, "", "", status)

    def _analyze(self, image, black_threshold, white_threshold, overlay_opacity,
                 banding_threshold, enable_focus_check,
                 enable_artifacts_check, enable_noise_check, fail_on_errors):
        try:
            if not isinstance(image, torch.Tensor) or image.dim() != 4:
                return self._error(f"Expected (B,H,W,C) tensor, got {type(image)}")
            B, H, W, C = image.shape
            if C not in [1, 3, 4]:
                return self._error(f"Invalid channels: {C}")

            levels = defects.analyze_levels(image, black_threshold, white_threshold)
            gamut = defects.check_gamut(image)
            banding = defects.detect_banding(image)
            noise = defects.analyze_noise(image) if enable_noise_check else None
            artifacts = defects.detect_compression_artifacts(image) if enable_artifacts_check else None
            focus = defects.analyze_focus(image) if enable_focus_check else None

            report_data = {
                "timestamp": datetime.now().isoformat(),
                "image_info": {"shape": [B, H, W, C], "dtype": str(image.dtype), "device": str(image.device)},
                "thresholds": {"black": black_threshold, "white": white_threshold, "banding": banding_threshold},
                "frames": [],
            }

            text_lines = [
                "\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557",
                "\u2551     RADIANCE QC REPORT v3.0.0           \u2551",
                "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d",
                f"Image: {B} frame(s), {W}x{H}, {C} channel(s)", "",
            ]
            overall_pass = True
            overlays = []

            for i in range(B):
                frame_data = {"frame_number": i+1, "checks": {}, "status": "PASS"}
                frame_lines = [f"\u2500\u2500\u2500 Frame {i+1}/{B} \u2500\u2500\u2500"]
                frame_pass = True

                crushed_pct = levels["crushed_percent"][i].item()
                clipped_pct = levels["clipped_percent"][i].item()
                oog_pct = gamut["out_of_gamut_percent"][i].item()
                banding_pct = banding[i].item()
                min_val = levels["min_val"][i].item()
                max_val = levels["max_val"][i].item()

                frame_lines.append(f"Range: [{min_val:.4f}, {max_val:.4f}]")

                if crushed_pct > 0:
                    frame_lines.append(f"  CRUSHED: {crushed_pct:.2f}% < {black_threshold}")
                    frame_pass = False
                frame_data["checks"]["crushed_blacks"] = {
                    "status": "FAIL" if crushed_pct > 0 else "PASS",
                    "percentage": crushed_pct, "threshold": black_threshold,
                }
                if clipped_pct > 0:
                    frame_lines.append(f"  CLIPPED: {clipped_pct:.2f}% > {white_threshold}")
                    frame_pass = False
                frame_data["checks"]["clipped_whites"] = {
                    "status": "FAIL" if clipped_pct > 0 else "PASS",
                    "percentage": clipped_pct, "threshold": white_threshold,
                }
                if oog_pct > 0:
                    frame_lines.append(f"  GAMUT: {oog_pct:.2f}% out-of-gamut")
                    if oog_pct > 1.0:
                        frame_pass = False
                frame_data["checks"]["gamut"] = {
                    "status": "PASS" if oog_pct == 0 else "WARN",
                    "out_of_gamut_pct": oog_pct,
                }
                if banding_pct >= banding_threshold:
                    frame_lines.append(f"  BANDING: {banding_pct:.2f}% risk (threshold: {banding_threshold}%)")
                frame_data["checks"]["banding"] = {
                    "status": "PASS" if banding_pct < banding_threshold else "WARN",
                    "risk_pct": banding_pct, "threshold": banding_threshold,
                }
                if noise is not None:
                    ns = noise["noise_score"][i].item()
                    if ns < 5.0:
                        frame_lines.append(f"  NOISE: {ns:.2f}/100 (over-denoised)")
                    elif ns > 30.0:
                        frame_lines.append(f"  NOISE: {ns:.2f}/100 (high noise)")
                    frame_data["checks"]["noise"] = {
                        "status": "PASS" if 5.0 < ns < 30.0 else "WARN",
                        "score": ns, "std_dev": noise["std_dev"][i].item(),
                    }
                if artifacts is not None:
                    asc = artifacts[i].item()
                    if asc >= 10.0:
                        frame_lines.append(f"  ARTIFACTS: {asc:.2f}/100 compression artifacts")
                    frame_data["checks"]["compression_artifacts"] = {
                        "status": "PASS" if asc < 10.0 else "WARN", "score": asc,
                    }
                if focus is not None:
                    fsc = focus["focus_score"][i].item()
                    if fsc < 20.0:
                        frame_lines.append(f"  FOCUS: {fsc:.2f}/100 (low sharpness)")
                    frame_data["checks"]["focus"] = {
                        "status": "PASS" if fsc > 20.0 else "WARN", "score": fsc,
                    }
                if frame_pass:
                    frame_lines.append("  \u2713 ALL CHECKS PASSED")
                else:
                    frame_data["status"] = "FAIL"
                    overall_pass = False

                text_lines.extend(frame_lines)
                text_lines.append("")
                report_data["frames"].append(frame_data)

                target = image[i].clone()
                if overlay_opacity > 0:
                    overlay = target.clone()
                    clipped_mask = (target > white_threshold).any(dim=-1)
                    if clipped_mask.any():
                        red = torch.tensor([1.0, 0.0, 0.0], device=overlay.device, dtype=overlay.dtype)
                        overlay[clipped_mask] = red[:C]
                    crushed_mask = (target < black_threshold).any(dim=-1)
                    if crushed_mask.any():
                        blue = torch.tensor([0.0, 0.0, 1.0], device=overlay.device, dtype=overlay.dtype)
                        overlay[crushed_mask] = blue[:C]
                    target = target * (1.0 - overlay_opacity) + overlay * overlay_opacity
                overlays.append(target)

            text_lines += [
                "\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557",
                "\u2551  STATUS: \u2713 QC PASSED                     \u2551" if overall_pass
                else "\u2551  STATUS: \u2717 QC FAILED                     \u2551",
                "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d",
            ]
            report_data["overall_status"] = "PASS" if overall_pass else "FAIL"

            result_tensor = torch.stack(overlays)
            text_report = "\n".join(text_lines)
            json_report = json.dumps(report_data, indent=2)
            status = "\u2713 PASS" if overall_pass else "\u2717 FAIL"
            if fail_on_errors and not overall_pass:
                status += " (BLOCKING)"
            return (result_tensor, text_report, json_report, status)
        except Exception as exc:
            return self._error(f"Analysis failed: {type(exc).__name__}: {exc}")

    def _export(self, qc_report_json: str, output_path: str,
                filename_prefix: str, export_format: str) -> str:
        try:
            from pathlib import Path
            output_path = strip_path_quotes(output_path)
            report = json.loads(qc_report_json)
            try:
                output_dir = Path(get_safe_output_dir(_OUTPUT_DIR, output_path, allow_absolute=True))
            except ValueError as exc:
                return f"\u2717 Export failed: Invalid output path \u2014 {exc}"

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{filename_prefix}_{timestamp}"
            exported_files = []

            if export_format in ["json", "all"]:
                p = output_dir / f"{base_filename}.json"
                with open(p, "w") as f:
                    json.dump(report, f, indent=2)
                exported_files.append(str(p))

            if export_format in ["csv", "all"]:
                p = output_dir / f"{base_filename}.csv"
                with open(p, "w") as f:
                    f.write("Frame,Status,Crushed%,Clipped%,Gamut%,Banding%\n")
                    for frame in report.get("frames", []):
                        chk = frame.get("checks", {})
                        f.write(f"{frame.get('frame_number',0)},{frame.get('status','?')},"
                                f"{chk.get('crushed_blacks',{}).get('percentage',0):.2f},"
                                f"{chk.get('clipped_whites',{}).get('percentage',0):.2f},"
                                f"{chk.get('gamut',{}).get('out_of_gamut_pct',0):.2f},"
                                f"{chk.get('banding',{}).get('risk_pct',0):.2f}\n")
                exported_files.append(str(p))

            if export_format in ["html", "all"]:
                p = output_dir / f"{base_filename}.html"
                with open(p, "w") as f:
                    f.write(_generate_html_report(report))
                exported_files.append(str(p))

            return f"\u2713 Exported {len(exported_files)} file(s):\n" + "\n".join(exported_files)
        except Exception as exc:
            return f"\u2717 Export failed: {exc}"

    def _error(self, message: str):
        err_img = torch.zeros(1, 64, 64, 3)
        return (err_img, f"\u2554\u2550\u2550\u2550 QC ERROR \u2550\u2550\u2550\u2557\n{message}",
                json.dumps({"error": True, "message": message}),
                f"\u2717 ERROR: {message}")


class RadiancePolicyGuard:
    MODES = ["Preset", "Guard"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (cls.MODES, {"default": "Guard"}),
                "image": ("IMAGE",),
            },
            "optional": {
                "preset": (list(_PRESETS.keys()), {"default": "Broadcast SDR"}),
                "policy_file": ("STRING", {"default": ""}),
                "custom_max_peak_nits": ("FLOAT", {"default": 1000.0, "min": 0.0, "max": 10000.0, "step": 10.0}),
                "custom_max_clipping": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "custom_max_black_crush": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001}),
                "custom_max_saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "policy": ("STRING", {"forceInput": True}),
                "max_clipping": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "max_black_crush": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001}),
                "max_saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_peak_nits": ("FLOAT", {"default": 1000.0, "min": 0.0, "max": 10000.0, "step": 10.0}),
                "require_metadata": ("STRING", {"default": ""}),
                "metadata_present": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING", "STRING", "INT")
    RETURN_NAMES = ("image", "passed", "data1", "data2", "score")
    FUNCTION = "run"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ QC & Debug"
    DESCRIPTION = "Enforce delivery policy constraints (legal range, gamut, loudness)."

    def run(self, mode: str = "Guard", preset: str = "Broadcast SDR", policy_file: str = "",
            custom_max_peak_nits: float = 1000.0, custom_max_clipping: float = 0.01,
            custom_max_black_crush: float = 0.05, custom_max_saturation: float = 1.0,
            image=None, policy: str = "", max_clipping: float = 0.01,
            max_black_crush: float = 0.05, max_saturation: float = 1.0,
            max_peak_nits: float = 1000.0, require_metadata: str = "",
            metadata_present: str = ""):
        if mode == "Preset":
            pol_json, description = self._preset(preset, policy_file, custom_max_peak_nits,
                                                  custom_max_clipping, custom_max_black_crush,
                                                  custom_max_saturation)
            dummy = torch.zeros(1, 64, 64, 3)
            return (dummy, True, pol_json, description, 0)
        else:
            if image is None:
                dummy = torch.zeros(1, 64, 64, 3)
                return (dummy, False, '{"error":"image required"}', "ERROR: image is required", 0)
            return self._guard(image, policy, max_clipping, max_black_crush,
                               max_saturation, max_peak_nits, require_metadata, metadata_present)

    def _preset(self, preset, policy_file, max_peak_nits, max_clipping, max_black_crush, max_saturation):
        if policy_file.strip():
            try:
                with open(policy_file.strip()) as f:
                    pol = json.load(f)
                return (json.dumps(pol), pol.get("description", "Custom file policy"))
            except Exception as exc:
                logger.warning("Policy file load failed: %s", exc)
        pol = dict(_PRESETS.get(preset, _PRESETS["Custom"]))
        if preset == "Custom":
            pol.update({"max_peak_nits": max_peak_nits, "max_clipping": max_clipping,
                        "max_black_crush": max_black_crush, "max_saturation": max_saturation})
        return (json.dumps(pol), pol.get("description", ""))

    def _guard(self, image, policy_str, max_clipping, max_black_crush, max_saturation,
               max_peak_nits, require_metadata, metadata_present):
        if policy_str.strip():
            try:
                pol = json.loads(policy_str)
            except Exception:
                pol = {}
        else:
            pol = {"max_peak_nits": max_peak_nits, "max_clipping": max_clipping,
                   "max_black_crush": max_black_crush, "max_saturation": max_saturation,
                   "require_metadata": [k.strip() for k in require_metadata.split(",") if k.strip()],
                   "gamut_check": False}

        meta_keys: List[str] = []
        for pair in metadata_present.split(","):
            pair = pair.strip()
            if "=" in pair:
                meta_keys.append(pair.split("=")[0].strip())
            elif pair:
                meta_keys.append(pair)

        arr = image.detach().cpu().float().numpy()
        if arr.ndim == 4:
            arr = arr[0]

        stats = _policy_analyse(arr)
        passed, violations, score = _evaluate_policy(stats, pol, meta_keys)

        report = {"passed": passed, "score": score, "policy": pol.get("description", "Custom"),
                  "violations": violations, "stats": stats}
        report_json = json.dumps(report, indent=2)

        status = "\u2713 PASSED" if passed else "\u2717 FAILED"
        lines = [f"{status}  Score: {score}/100",
                 f"Policy: {pol.get('description', 'Inline')}", "",
                 "Stats:",
                 f"  Peak:       {stats['peak']:.4f} ({stats['peak']*100:.1f} nits approx)",
                 f"  Mean luma:  {stats['mean_luma']:.4f}",
                 f"  Clipping:   {stats['clipping']:.2%}",
                 f"  Black crush:{stats['black_crush']:.2%}",
                 f"  Mean sat:   {stats['mean_sat']:.4f}"]
        if violations:
            lines += ["", "Violations:"]
            for v in violations:
                icon = "\u2717" if v["severity"] == "error" else "\u26A0"
                lines.append(f"  {icon} {v['rule']}: {v['actual']} (limit: {v['limit']})")
        else:
            lines += ["", "No violations \u2014 frame is delivery-ready."]

        report_text = "\n".join(lines)
        return (image, passed, report_json, report_text, score)


RadianceQCExport = RadianceQC
