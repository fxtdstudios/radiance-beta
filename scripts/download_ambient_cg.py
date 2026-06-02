import os
import requests
import argparse
from pathlib import Path
from tqdm import tqdm
import time

def download_ambient_cg(output_dir, limit=500):
    """
    Downloads 2K EXR HDRIs from AmbientCG.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Fetch metadata for all HDRI assets
    print("Fetching metadata from AmbientCG v3 API...")
    api_url = "https://ambientCG.com/api/v3/assets"
    params = {
        "type": "hdri", # Try lowercase
        "limit": limit
    }
    headers = {
        "User-Agent": "RadianceHDREncoder/1.0 (Training Pipeline)"
    }
    
    try:
        response = requests.get(api_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching metadata: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")
        return

    assets = data.get("foundAssets", [])
    print(f"Found {len(assets)} HDRI assets.")

    # 2. Iterate and download 2K-HDR EXR files
    downloaded = 0
    for asset in assets:
        asset_id = asset.get("assetId")
        download_data = asset.get("downloadData", {})
        
        # We look for "2K-HDR" which is typically the scene-linear EXR 
        # (It can be under .exr or .hdr, but we prefer .exr)
        target_token = "2K-HDR"
        chosen_download = None
        
        # Find the best 2K EXR link
        for key, info in download_data.items():
            if target_token in key and info.get("fileExtension") == ".exr":
                chosen_download = info
                break
        
        if not chosen_download:
            # Fallback to .hdr if .exr missing
            for key, info in download_data.items():
                if target_token in key and info.get("fileExtension") == ".hdr":
                    chosen_download = info
                    break
        
        if not chosen_download:
            print(f"Skipping {asset_id}: No 2K-HDR EXR/HDR found.")
            continue

        file_url = chosen_download.get("downloadLink")
        file_ext = chosen_download.get("fileExtension")
        target_path = os.path.join(output_dir, f"{asset_id}{file_ext}")

        if os.path.exists(target_path):
            print(f"Skipping {asset_id} (already exists)")
            continue

        print(f"Downloading {asset_id} ({chosen_download.get('sizeBytes', 0) / 1e6:.1f} MB)...")
        try:
            # AmbientCG requires following redirects for downloadLink
            with requests.get(file_url, stream=True, allow_redirects=True) as r:
                r.raise_for_status()
                with open(target_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            downloaded += 1
            # Rate limiting / Respect the server
            time.sleep(0.5)
        except Exception as e:
            print(f"Failed to download {asset_id}: {e}")

    print(f"\nDone! Successfully downloaded {downloaded} new HDRIs to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="custom_nodes/radiance/hdris_ambient_cg")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    
    download_ambient_cg(args.output, args.limit)
