import os
import io
import base64
import json
import time
import shutil
import traceback
import logging
from pathlib import Path
from aiohttp import web
from server import PromptServer

import folder_paths

logger = logging.getLogger("Radiance.Workspace")

# ═══════════════════════════════════════════════════════════════════════════════
#                           CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MAX_WORKFLOW_SIZE_MB = 50
MAX_WORKFLOW_SIZE_BYTES = MAX_WORKFLOW_SIZE_MB * 1024 * 1024
MAX_FILENAME_LENGTH = 200
ALLOWED_EXTENSIONS = {".rad"}
MAX_VERSIONS = 10  # Auto-versioning backup depth


class RadianceWorkspace:
    """
    A unified node that sits on the ComfyUI canvas acting as a hub for
    saving and loading entire workflows securely as .rad files.

    The actual file generation and graph rebuilding is handled natively
    by the companion JavaScript extension: js/radiance_workspace.js
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "noop"
    CATEGORY = "FXTD Studios/Radiance/Layout"
    OUTPUT_NODE = False

    def noop(self, **kwargs):
        """
        No execution needed. This node exists purely to provide the
        UI buttons and anchor point on the canvas.
        """
        return {}


NODE_CLASS_MAPPINGS = {
    "RadianceWorkspace": RadianceWorkspace
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceWorkspace": "◎ Radiance .rad Workspace"
}


# ═══════════════════════════════════════════════════════════════════════════════
#                         PATH SECURITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

WORKFLOW_DIR = os.path.join(os.path.dirname(__file__), "workflows")
os.makedirs(WORKFLOW_DIR, exist_ok=True)

# Pre-resolve once at startup for consistent comparison
_WORKFLOW_DIR_RESOLVED = Path(WORKFLOW_DIR).resolve()


def _resolve_safe_path(filename: str) -> Path | None:
    """
    Resolve a user-supplied filename to a safe absolute path within WORKFLOW_DIR.
    Returns None if the resolved path escapes the allowed directory.
    Uses pathlib.is_relative_to() which is immune to the os.sep prefix trick.
    """
    if not filename or len(filename) > MAX_FILENAME_LENGTH:
        return None

    # Reject null bytes and other control characters
    if any(ord(c) < 32 for c in filename):
        return None

    candidate = (_WORKFLOW_DIR_RESOLVED / filename).resolve()

    try:
        candidate.relative_to(_WORKFLOW_DIR_RESOLVED)
    except ValueError:
        return None

    return candidate


def _validate_extension(filepath: Path) -> bool:
    """Ensure the file has an allowed extension."""
    return filepath.suffix.lower() in ALLOWED_EXTENSIONS


# ═══════════════════════════════════════════════════════════════════════════════
#                         AUTO-VERSIONING
# ═══════════════════════════════════════════════════════════════════════════════

def _create_version_backup(filepath: Path) -> None:
    """
    Rotate existing file into a .v1, .v2, ... backup before overwrite.
    Keeps at most MAX_VERSIONS backups, deleting the oldest.
    """
    if not filepath.exists():
        return

    versions_dir = filepath.parent / ".versions"
    versions_dir.mkdir(exist_ok=True)

    stem = filepath.stem
    suffix = filepath.suffix

    # Find next version number
    existing = sorted(versions_dir.glob(f"{stem}.v*{suffix}"))
    next_num = len(existing) + 1

    # Evict oldest if at capacity
    if len(existing) >= MAX_VERSIONS:
        for old in existing[: len(existing) - MAX_VERSIONS + 1]:
            old.unlink(missing_ok=True)

    backup_name = f"{stem}.v{next_num}{suffix}"
    shutil.copy2(str(filepath), str(versions_dir / backup_name))
    logger.info(f"[Radiance] Backed up {filepath.name} → .versions/{backup_name}")


# ═══════════════════════════════════════════════════════════════════════════════
#                       ASSETS LIBRARY API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


@PromptServer.instance.routes.post("/radiance/workflows/save")
async def save_workflow(request):
    try:
        # Guard against oversized payloads early
        content_length = request.content_length
        if content_length is not None and content_length > MAX_WORKFLOW_SIZE_BYTES:
            return web.json_response(
                {"error": f"Payload exceeds {MAX_WORKFLOW_SIZE_MB}MB limit"},
                status=413,
            )

        data = await request.json()
        filename = data.get("filename", "").strip()
        description = data.get("description", "").strip()

        if not filename:
            return web.json_response({"error": "Missing filename"}, status=400)

        if not filename.endswith(".rad"):
            filename += ".rad"

        filepath = _resolve_safe_path(filename)
        if filepath is None:
            return web.json_response({"error": "Invalid or unsafe path"}, status=403)

        if not _validate_extension(filepath):
            return web.json_response({"error": "Invalid file extension"}, status=400)

        content = data.get("content", "")
        if not content:
            return web.json_response({"error": "Missing content"}, status=400)

        if len(content.encode("utf-8")) > MAX_WORKFLOW_SIZE_BYTES:
            return web.json_response(
                {"error": f"Workflow content exceeds {MAX_WORKFLOW_SIZE_MB}MB limit"},
                status=413,
            )

        # Auto-version existing file before overwrite
        filepath.parent.mkdir(parents=True, exist_ok=True)
        _create_version_backup(filepath)

        filepath.write_text(content, encoding="utf-8")

        # Save metadata sidecar
        meta_path = filepath.with_suffix(filepath.suffix + ".json")
        meta = {
            "description": description,
            "saved_at": time.time(),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        logger.info(f"[Radiance] Saved workflow: {filename}")
        return web.json_response({"success": True, "filename": filename})

    except json.JSONDecodeError:
        return web.json_response({"error": "Malformed JSON body"}, status=400)
    except Exception as e:
        logger.exception("[Radiance] save_workflow failed")
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.get("/radiance/workflows/list")
async def list_workflows(request):
    try:
        workflows = []

        for rad_file in _WORKFLOW_DIR_RESOLVED.rglob("*.rad"):
            # Skip anything inside .versions directories
            if ".versions" in rad_file.parts:
                continue

            try:
                rel_path = rad_file.relative_to(_WORKFLOW_DIR_RESOLVED)
            except ValueError:
                continue

            stat = rad_file.stat()

            # Load metadata sidecar
            meta = {}
            meta_path = rad_file.with_suffix(rad_file.suffix + ".json")
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError, PermissionError) as meta_err:
                    logger.warning(f"[Radiance] Failed to read metadata for {rel_path}: {meta_err}")

            workflows.append({
                "filename": str(rel_path).replace("\\", "/"),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "metadata": meta,
            })

        workflows.sort(key=lambda x: x["mtime"], reverse=True)
        return web.json_response({"workflows": workflows})

    except Exception as e:
        logger.exception("[Radiance] list_workflows failed")
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.get("/radiance/workflows/get")
async def get_workflow(request):
    try:
        filename = request.rel_url.query.get("filename", "").strip()
        if not filename:
            return web.json_response({"error": "Missing filename"}, status=400)

        filepath = _resolve_safe_path(filename)
        if filepath is None:
            return web.json_response({"error": "Invalid or unsafe path"}, status=403)

        if not filepath.exists():
            return web.json_response({"error": "File not found"}, status=404)

        content = filepath.read_text(encoding="utf-8")
        return web.json_response({"success": True, "content": content})

    except Exception as e:
        logger.exception("[Radiance] get_workflow failed")
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/radiance/workflows/delete")
async def delete_workflow(request):
    try:
        data = await request.json()
        filename = data.get("filename", "").strip()
        if not filename:
            return web.json_response({"error": "Missing filename"}, status=400)

        filepath = _resolve_safe_path(filename)
        if filepath is None:
            return web.json_response({"error": "Invalid or unsafe path"}, status=403)

        if not filepath.exists():
            return web.json_response({"error": "File not found"}, status=404)

        filepath.unlink()
        logger.info(f"[Radiance] Deleted workflow: {filename}")

        # Also remove sidecar if exists
        meta_path = filepath.with_suffix(filepath.suffix + ".json")
        meta_path.unlink(missing_ok=True)

        # Clean up empty parent directories (but never WORKFLOW_DIR itself)
        try:
            parent = filepath.parent
            while parent != _WORKFLOW_DIR_RESOLVED:
                if not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
                else:
                    break
        except OSError:
            pass  # Directory not empty or permission issue, safe to ignore

        return web.json_response({"success": True})

    except json.JSONDecodeError:
        return web.json_response({"error": "Malformed JSON body"}, status=400)
    except Exception as e:
        logger.exception("[Radiance] delete_workflow failed")
        return web.json_response({"error": str(e)}, status=500)
