#!/usr/bin/env python3
"""
Radiance DaVinci Resolve Bridge
-------------------------------
A pipeline script to execute Radiance (ComfyUI) workflows directly from DaVinci Resolve.
Requires DaVinci Resolve Studio and its Python 3.6+ environment configured.
"""

import sys
import json
import os
import urllib.request
import urllib.error

COMFY_URL = os.environ.get("RADIANCE_COMFY_URL", "http://127.0.0.1:8188").rstrip("/")

def trigger_comfy_workflow(prompt_data):
    """Sends a job payload to the ComfyUI API."""
    data = {"prompt": prompt_data, "client_id": "radiance_resolve_bridge"}
    json_data = json.dumps(data).encode("utf-8")
    
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=json_data, 
                                 headers={"Content-Type": "application/json"})
    try:
        response = urllib.request.urlopen(req)
        return json.loads(response.read())
    except urllib.error.URLError as e:
        print(f"Failed to connect to Radiance Server: {e}")
        return None

def main():
    try:
        # DaVinci API is exposed automatically when running within the Studio script menu
        import DaVinciResolveScript as dvr_script
    except ImportError:
        print("This script must be run inside DaVinci Resolve Studio.")
        return

    resolve = dvr_script.scriptapp("Resolve")
    if not resolve:
        print("ERROR: Could not get Resolve app.")
        return

    pmspa = resolve.GetProjectManager()
    proj = pmspa.GetCurrentProject()
    timeline = proj.GetCurrentTimeline()
    
    if not timeline:
        print("ERROR: No timeline is currently active.")
        return
        
    clip = timeline.GetCurrentVideoItem()
    clip_name = clip.GetName() if clip else "Unknown"
    
    print(f"Radiance Bridge: Found active clip '{clip_name}'")
    
    # Normally we would use the Deliver page API to write a temporary EXR.
    # For now, we mock the workflow payload using our `◎ Radiance` nodes.
    workflow = {
        "3": {
            "class_type": "RadianceLoadImage",
            "inputs": {
                "image": f"timeline_temp_{clip_name}.exr"
            }
        },
        "4": {
            "class_type": "RadianceRelightEngine",
            "inputs": {
                "image": ["3", 0],
                "normal_map": ["3", 0], # Placeholder
                "diffuse_intensity": 1.5
            }
        },
        "5": {
            "class_type": "RadianceWrite",
            "inputs": {
                "image": ["4", 0],
                "filename_prefix": "radiance_resolve_out"
            }
        }
    }
    
    print("Dispatching job to Radiance Engine...")
    result = trigger_comfy_workflow(workflow)
    if result:
        print(f"Job queued successfully. ID: {result.get('prompt_id')}")
        # Next steps: poll the API for completion, then import 'radiance_resolve_out.exr'
        # into the Media Pool using proj.GetMediaPool().ImportMedia()

if __name__ == "__main__":
    main()
