import torch
import json
import os
import logging
from typing import Dict, Tuple
from .image import defects

try:
    from .path_utils import get_safe_output_dir
except ImportError:
    from path_utils import get_safe_output_dir

try:
    import folder_paths
    _OUTPUT_DIR = folder_paths.get_output_directory()
except Exception:
    _OUTPUT_DIR = os.path.abspath("output")

logger = logging.getLogger("◎ Radiance.qc")


class RadianceQC:
    """
    Production-grade QC tool with comprehensive analysis:

    Technical Checks:
    - Crushed Blacks / Clipped Whites
    - Gamut Violations (out of [0,1] for SDR)
    - Banding / Posterization
    - Noise Levels
    - Compression Artifacts
    - Focus/Sharpness Analysis

    Outputs:
    - Visual overlay with color-coded issue highlighting
    - Structured JSON report for pipeline integration
    - Per-frame statistics
    - Pass/Fail status per check
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "black_threshold": (
                    "FLOAT",
                    {"default": 0.0, "min": -0.1, "max": 0.1, "step": 0.001},
                ),
                "white_threshold": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.8, "max": 2.0, "step": 0.01},
                ),
                "overlay_opacity": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "banding_threshold": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.5},
                ),
                "enable_focus_check": ("BOOLEAN", {"default": False}),
                "enable_artifacts_check": ("BOOLEAN", {"default": True}),
                "enable_noise_check": ("BOOLEAN", {"default": True}),
                "fail_on_errors": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("overlay_image", "qc_report_text", "qc_report_json", "status")
    FUNCTION = "analyze"
    CATEGORY = "FXTD Studios/Radiance/Analyze"
    DESCRIPTION = "Production QC with comprehensive checks and structured reporting"

    def analyze(
        self,
        image: torch.Tensor,
        black_threshold: float = 0.0,
        white_threshold: float = 1.0,
        overlay_opacity: float = 0.5,
        banding_threshold: float = 5.0,
        enable_focus_check: bool = False,
        enable_artifacts_check: bool = True,
        enable_noise_check: bool = True,
        fail_on_errors: bool = False,
    ) -> Tuple:

        try:
            # ═══════════════════════════════════════════════════════
            # STEP 1: VALIDATE INPUT
            # ═══════════════════════════════════════════════════════

            if not isinstance(image, torch.Tensor):
                return self._create_error("Invalid input: expected torch.Tensor")

            if image.dim() != 4:
                return self._create_error(
                    f"Invalid shape: expected (B,H,W,C), got {image.shape}"
                )

            B, H, W, C = image.shape

            if C not in [1, 3, 4]:
                return self._create_error(f"Invalid channels: expected 1/3/4, got {C}")

            # ═══════════════════════════════════════════════════════
            # STEP 2: RUN ANALYSIS (VECTORIZED)
            # ═══════════════════════════════════════════════════════

            # Core checks (always enabled)
            levels = defects.analyze_levels(image, black_threshold, white_threshold)
            gamut = defects.check_gamut(image)
            banding = defects.detect_banding(image)

            # Optional checks
            noise = defects.analyze_noise(image) if enable_noise_check else None
            artifacts = (
                defects.detect_compression_artifacts(image)
                if enable_artifacts_check
                else None
            )
            focus = defects.analyze_focus(image) if enable_focus_check else None

            # ═══════════════════════════════════════════════════════
            # STEP 3: BUILD STRUCTURED REPORTS
            # ═══════════════════════════════════════════════════════

            report_data = {
                "timestamp": (
                    torch.cuda.Event().record() if torch.cuda.is_available() else None
                ),
                "image_info": {
                    "shape": [B, H, W, C],
                    "dtype": str(image.dtype),
                    "device": str(image.device),
                },
                "thresholds": {
                    "black": black_threshold,
                    "white": white_threshold,
                    "banding": banding_threshold,
                },
                "frames": [],
            }

            text_report_lines = ["╔═══════════════════════════════════════════╗"]
            text_report_lines.append("║     RADIANCE QC REPORT v2.3.2            ║")
            text_report_lines.append("╚═══════════════════════════════════════════╝")
            text_report_lines.append(f"Image: {B} frame(s), {W}x{H}, {C} channel(s)")
            text_report_lines.append("")

            overall_pass = True

            # ═══════════════════════════════════════════════════════
            # STEP 4: PER-FRAME ANALYSIS
            # ═══════════════════════════════════════════════════════

            overlays = []

            for i in range(B):
                frame_data = {"frame_number": i + 1, "checks": {}, "status": "PASS"}

                frame_report = [f"─── Frame {i+1}/{B} ───"]
                frame_pass = True

                # Extract per-frame values
                crushed_pct = levels["crushed_percent"][i].item()
                clipped_pct = levels["clipped_percent"][i].item()
                oog_pct = gamut["out_of_gamut_percent"][i].item()
                banding_pct = banding[i].item()
                min_val = levels["min_val"][i].item()
                max_val = levels["max_val"][i].item()

                frame_report.append(f"Range: [{min_val:.4f}, {max_val:.4f}]")

                # Check: Crushed Blacks
                crushed_status = "PASS" if crushed_pct == 0 else "FAIL"
                frame_data["checks"]["crushed_blacks"] = {
                    "status": crushed_status,
                    "percentage": crushed_pct,
                    "threshold": black_threshold,
                }
                if crushed_pct > 0:
                    frame_report.append(
                        f"  ◎ CRUSHED: {crushed_pct:.2f}% pixels < {black_threshold}"
                    )
                    frame_pass = False

                # Check: Clipped Whites
                clipped_status = "PASS" if clipped_pct == 0 else "FAIL"
                frame_data["checks"]["clipped_whites"] = {
                    "status": clipped_status,
                    "percentage": clipped_pct,
                    "threshold": white_threshold,
                }
                if clipped_pct > 0:
                    frame_report.append(
                        f"  ◎ CLIPPED: {clipped_pct:.2f}% pixels > {white_threshold}"
                    )
                    frame_pass = False

                # Check: Gamut
                gamut_status = "PASS" if oog_pct == 0 else "WARN"
                frame_data["checks"]["gamut"] = {
                    "status": gamut_status,
                    "out_of_gamut_pct": oog_pct,
                }
                if oog_pct > 0:
                    frame_report.append(f"  ◎ GAMUT: {oog_pct:.2f}% out-of-gamut")
                    if oog_pct > 1.0:  # Only fail if significant
                        frame_pass = False

                # Check: Banding
                banding_status = "PASS" if banding_pct < banding_threshold else "WARN"
                frame_data["checks"]["banding"] = {
                    "status": banding_status,
                    "risk_pct": banding_pct,
                    "threshold": banding_threshold,
                }
                if banding_pct >= banding_threshold:
                    frame_report.append(
                        f"  ◎ BANDING: {banding_pct:.2f}% risk (threshold: {banding_threshold}%)"
                    )

                # Optional: Noise
                if noise is not None:
                    noise_score = noise["noise_score"][i].item()
                    noise_status = "PASS" if 5.0 < noise_score < 30.0 else "WARN"
                    frame_data["checks"]["noise"] = {
                        "status": noise_status,
                        "score": noise_score,
                        "std_dev": noise["std_dev"][i].item(),
                    }
                    if noise_score < 5.0:
                        frame_report.append(
                            f"  ◎ NOISE: {noise_score:.2f}/100 (unusually clean, possible over-denoising)"
                        )
                    elif noise_score > 30.0:
                        frame_report.append(
                            f"  ◎ NOISE: {noise_score:.2f}/100 (high noise level)"
                        )

                # Optional: Artifacts
                if artifacts is not None:
                    artifact_score = artifacts[i].item()
                    artifact_status = "PASS" if artifact_score < 10.0 else "WARN"
                    frame_data["checks"]["compression_artifacts"] = {
                        "status": artifact_status,
                        "score": artifact_score,
                    }
                    if artifact_score >= 10.0:
                        frame_report.append(
                            f"  ◎ ARTIFACTS: {artifact_score:.2f}/100 compression artifacts detected"
                        )

                # Optional: Focus
                if focus is not None:
                    focus_score = focus["focus_score"][i].item()
                    focus_status = "PASS" if focus_score > 20.0 else "WARN"
                    frame_data["checks"]["focus"] = {
                        "status": focus_status,
                        "score": focus_score,
                    }
                    if focus_score < 20.0:
                        frame_report.append(
                            f"  ◎ FOCUS: {focus_score:.2f}/100 (low sharpness)"
                        )

                # Frame status
                if frame_pass:
                    frame_report.append("  ✓ ALL CHECKS PASSED")
                    frame_data["status"] = "PASS"
                else:
                    frame_data["status"] = "FAIL"
                    overall_pass = False

                text_report_lines.extend(frame_report)
                text_report_lines.append("")

                report_data["frames"].append(frame_data)

                # ═══════════════════════════════════════════════════
                # STEP 5: GENERATE OVERLAY
                # ═══════════════════════════════════════════════════

                target = image[i].clone()  # (H, W, C)

                if overlay_opacity > 0:
                    overlay = target.clone()

                    # Clipped pixels -> Red
                    clipped_mask = (target > white_threshold).any(dim=-1)  # (H, W)
                    if clipped_mask.any():
                        red = torch.tensor(
                            [1.0, 0.0, 0.0], device=overlay.device, dtype=overlay.dtype
                        )
                        overlay[clipped_mask] = red.view(1, 1, 3)[:, :, :C]

                    # Crushed pixels -> Blue
                    crushed_mask = (target < black_threshold).any(dim=-1)  # (H, W)
                    if crushed_mask.any():
                        blue = torch.tensor(
                            [0.0, 0.0, 1.0], device=overlay.device, dtype=overlay.dtype
                        )
                        overlay[crushed_mask] = blue.view(1, 1, 3)[:, :, :C]

                    # Gamma-correct blend (more accurate for perceptual blending)
                    # Linear blend: target * (1 - α) + overlay * α
                    blended = (
                        target * (1.0 - overlay_opacity) + overlay * overlay_opacity
                    )
                    overlays.append(blended)
                else:
                    overlays.append(target)

            # ═══════════════════════════════════════════════════════
            # STEP 6: FINALIZE OUTPUTS
            # ═══════════════════════════════════════════════════════

            result_tensor = torch.stack(overlays)  # (B, H, W, C)

            # Summary
            text_report_lines.append("╔═══════════════════════════════════════════╗")
            if overall_pass:
                text_report_lines.append("║  STATUS: ✓ QC PASSED                     ║")
                report_data["overall_status"] = "PASS"
            else:
                text_report_lines.append("║  STATUS: ✗ QC FAILED                     ║")
                report_data["overall_status"] = "FAIL"
            text_report_lines.append("╔═══════════════════════════════════════════╗")

            text_report = "\n".join(text_report_lines)
            json_report = json.dumps(report_data, indent=2)

            # Status string for node UI
            status = "✓ PASS" if overall_pass else "✗ FAIL"
            if fail_on_errors and not overall_pass:
                status += " (BLOCKING)"

            return (result_tensor, text_report, json_report, status)

        except Exception as e:
            return self._create_error(f"Analysis failed: {type(e).__name__}: {str(e)}")

    def _create_error(self, message: str) -> Tuple:
        """Generate error response."""
        error_image = torch.zeros((1, 64, 64, 3))  # Black placeholder
        error_report = f"╔═══ QC ERROR ═══╗\n{message}"
        error_json = json.dumps({"error": True, "message": message})
        return (error_image, error_report, error_json, f"✗ ERROR: {message}")


# ═══════════════════════════════════════════════════════════════
# QC EXPORT NODE
# ═══════════════════════════════════════════════════════════════


class RadianceQCExport:
    """
    Export QC reports to files for pipeline integration.
    Supports JSON, CSV, and HTML formats.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qc_report_json": ("STRING", {"forceInput": True}),
                "output_path": ("STRING", {"default": "C:/Projects/qc_reports"}),
                "filename_prefix": ("STRING", {"default": "qc_report"}),
                "export_format": (["json", "csv", "html", "all"], {"default": "json"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("export_status",)
    FUNCTION = "export_report"
    CATEGORY = "FXTD Studios/Radiance/Analyze"
    OUTPUT_NODE = True

    def export_report(
        self,
        qc_report_json: str,
        output_path: str,
        filename_prefix: str,
        export_format: str,
    ) -> Tuple[str]:
        try:
            from pathlib import Path
            from datetime import datetime

            # Parse JSON
            report = json.loads(qc_report_json)

            # Security: validate output path via safe_join
            try:
                output_dir = Path(get_safe_output_dir(_OUTPUT_DIR, output_path, allow_absolute=True))
            except ValueError as e:
                return (f"✗ Export failed: Invalid output path — {e}",)

            # Timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{filename_prefix}_{timestamp}"

            exported_files = []

            # Export JSON
            if export_format in ["json", "all"]:
                json_path = output_dir / f"{base_filename}.json"
                with open(json_path, "w") as f:
                    json.dump(report, f, indent=2)
                exported_files.append(str(json_path))

            # Export CSV
            if export_format in ["csv", "all"]:
                csv_path = output_dir / f"{base_filename}.csv"
                with open(csv_path, "w") as f:
                    # Header
                    f.write("Frame,Status,Crushed%,Clipped%,Gamut%,Banding%\n")
                    # Data
                    for frame in report.get("frames", []):
                        frame_num = frame.get("frame_number", 0)
                        status = frame.get("status", "UNKNOWN")
                        checks = frame.get("checks", {})
                        crushed = checks.get("crushed_blacks", {}).get("percentage", 0)
                        clipped = checks.get("clipped_whites", {}).get("percentage", 0)
                        gamut = checks.get("gamut", {}).get("out_of_gamut_pct", 0)
                        banding = checks.get("banding", {}).get("risk_pct", 0)
                        f.write(
                            f"{frame_num},{status},{crushed:.2f},{clipped:.2f},{gamut:.2f},{banding:.2f}\n"
                        )
                exported_files.append(str(csv_path))

            # Export HTML (basic)
            if export_format in ["html", "all"]:
                html_path = output_dir / f"{base_filename}.html"
                with open(html_path, "w") as f:
                    f.write(self._generate_html_report(report))
                exported_files.append(str(html_path))

            status = f"✓ Exported {len(exported_files)} file(s):\n" + "\n".join(
                exported_files
            )
            return (status,)

        except Exception as e:
            return (f"✗ Export failed: {str(e)}",)

    def _generate_html_report(self, report: Dict) -> str:
        """Generate basic HTML report."""
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
<h1>RADIANCE QC REPORT</h1>
"""

        overall = report.get("overall_status", "UNKNOWN")
        html += f"<h2 class=\"{'pass' if overall == 'PASS' else 'fail'}\">Overall: {overall}</h2>"

        html += "<table><tr><th>Frame</th><th>Status</th><th>Crushed</th><th>Clipped</th><th>Gamut</th><th>Banding</th></tr>"

        for frame in report.get("frames", []):
            frame_num = frame.get("frame_number", 0)
            status = frame.get("status", "UNKNOWN")
            status_class = "pass" if status == "PASS" else "fail"
            checks = frame.get("checks", {})

            html += f'<tr><td>{frame_num}</td><td class="{status_class}">{status}</td>'
            html += (
                f"<td>{checks.get('crushed_blacks', {}).get('percentage', 0):.2f}%</td>"
            )
            html += (
                f"<td>{checks.get('clipped_whites', {}).get('percentage', 0):.2f}%</td>"
            )
            html += (
                f"<td>{checks.get('gamut', {}).get('out_of_gamut_pct', 0):.2f}%</td>"
            )
            html += f"<td>{checks.get('banding', {}).get('risk_pct', 0):.2f}%</td></tr>"

        html += "</table></body></html>"
        return html


# ═══════════════════════════════════════════════════════════════
# NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "◎ RadianceQC": RadianceQC,
    "◎ RadianceQCExport": RadianceQCExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "◎ RadianceQC": "◎ Radiance QC Pro",
    "◎ RadianceQCExport": "◎ Radiance Export QC Report",
}
