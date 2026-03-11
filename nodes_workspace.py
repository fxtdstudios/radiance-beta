import os
import io
import base64
import json
import traceback
from aiohttp import web
from server import PromptServer

import folder_paths

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

# Register the node
NODE_CLASS_MAPPINGS = {
    "RadianceWorkspace": RadianceWorkspace
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceWorkspace": "◎ Radiance .rad Workspace"
}

# ═══════════════════════════════════════════════════════════════════════════════
#                       ASSETS LIBRARY API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

WORKFLOW_DIR = os.path.join(os.path.dirname(__file__), "workflows")
os.makedirs(WORKFLOW_DIR, exist_ok=True)

@PromptServer.instance.routes.post("/radiance/workflows/save")
async def save_workflow(request):
    try:
        data = await request.json()
        filename = data.get("filename")
        description = data.get("description", "")
        
        if not filename:
             return web.json_response({"error": "Missing filename"}, status=400)
        
        if not filename.endswith(".rad"):
             filename += ".rad"
             
        # Allow subfolder paths in filename
        # Ensure we stay within WORKFLOW_DIR
        filepath = os.path.normpath(os.path.join(WORKFLOW_DIR, filename))
        if not filepath.startswith(os.path.abspath(WORKFLOW_DIR)):
             return web.json_response({"error": "Invalid path"}, status=403)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        content = data.get("content")
        if not content:
             return web.json_response({"error": "Missing content"}, status=400)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
            
        # Save metadata sidecar
        if description:
            meta_path = filepath + ".json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"description": description}, f)
            
        return web.json_response({"success": True, "filename": filename})
    except Exception as e:
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.get("/radiance/workflows/list")
async def list_workflows(request):
    try:
        workflows = []
        if os.path.exists(WORKFLOW_DIR):
            for root, dirs, files in os.walk(WORKFLOW_DIR):
                for filename in files:
                    if filename.endswith(".rad"):
                        filepath = os.path.join(root, filename)
                        rel_path = os.path.relpath(filepath, WORKFLOW_DIR).replace("\\", "/")
                        
                        stat = os.stat(filepath)
                        
                        # Load metadata if exists
                        meta = {}
                        meta_path = filepath + ".json"
                        if os.path.exists(meta_path):
                            try:
                                with open(meta_path, "r", encoding="utf-8") as f:
                                    meta = json.load(f)
                            except: pass
                            
                        workflows.append({
                             "filename": rel_path,
                             "size": stat.st_size,
                             "mtime": stat.st_mtime,
                             "metadata": meta
                        })
        
        # Sort by modification time descending
        workflows.sort(key=lambda x: x["mtime"], reverse=True)
        return web.json_response({"workflows": workflows})
    except Exception as e:
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.get("/radiance/workflows/get")
async def get_workflow(request):
    try:
        filename = request.rel_url.query.get("filename", "")
        if not filename:
             return web.json_response({"error": "Missing filename"}, status=400)
             
        filepath = os.path.normpath(os.path.join(WORKFLOW_DIR, filename))
        if not filepath.startswith(os.path.abspath(WORKFLOW_DIR)):
             return web.json_response({"error": "Invalid path"}, status=403)
             
        if not os.path.exists(filepath):
             return web.json_response({"error": "File not found"}, status=404)
             
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return web.json_response({"success": True, "content": content})
    except Exception as e:
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.post("/radiance/workflows/delete")
async def delete_workflow(request):
    try:
        data = await request.json()
        filename = data.get("filename")
        if not filename:
             return web.json_response({"error": "Missing filename"}, status=400)
             
        filepath = os.path.normpath(os.path.join(WORKFLOW_DIR, filename))
        if not filepath.startswith(os.path.abspath(WORKFLOW_DIR)):
             return web.json_response({"error": "Invalid path"}, status=403)
             
        if os.path.exists(filepath):
             os.remove(filepath)
             # Also remove sidecar if exists
             meta_path = filepath + ".json"
             if os.path.exists(meta_path):
                 os.remove(meta_path)
             return web.json_response({"success": True})
        else:
             return web.json_response({"error": "File not found"}, status=404)
    except Exception as e:
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)

