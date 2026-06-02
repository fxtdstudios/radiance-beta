"""
download_hdris.py — Radiance Bulk HDRI Downloader
════════════════════════════════════════════════════════════════════════════════

Downloads scene-linear HDR images from Poly Haven for training Radiance VAEs.
Uses the Poly Haven Public API (api.polyhaven.com).

USAGE:
    python scripts/download_hdris.py --count 1000 --output_dir ./raw_hdri --res 2k
"""

import os
import sys
import json
import argparse
import requests
from tqdm import tqdm
from pathlib import Path

def download_file(url, dest):
    """Download a file with a progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(dest, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=dest.name, leave=False) as pbar:
            for data in response.iter_content(block_size):
                f.write(data)
                pbar.update(len(data))

def main():
    parser = argparse.ArgumentParser(description="Bulk download HDRIs from Poly Haven")
    parser.add_argument("--count", type=int, default=100, help="Number of HDRIs to download")
    parser.add_argument("--output_dir", type=str, default="raw_hdri", help="Output directory")
    parser.add_argument("--res", type=str, default="2k", choices=["1k", "2k", "4k", "8k"], help="Resolution")
    parser.add_argument("--format", type=str, default="exr", choices=["exr", "hdr"], help="File format")
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching asset list from Poly Haven...")
    try:
        # User-Agent is required by Poly Haven API
        headers = {"User-Agent": "Radiance-HDR-Training-Utility/1.0"}
        r = requests.get("https://api.polyhaven.com/assets?type=hdris", headers=headers)
        r.raise_for_status()
        assets = r.json()
    except Exception as e:
        print(f"Error fetching asset list: {e}")
        sys.exit(1)
        
    all_keys = list(assets.keys())
    count = min(args.count, len(all_keys))
    selected_keys = all_keys[:count]
    
    print(f"Downloading {count} HDRIs (Resolution: {args.res}, Format: {args.format})...")
    
    n_success = 0
    toplevel_pbar = tqdm(selected_keys, desc="Total Progress")
    
    for key in toplevel_pbar:
        try:
            # Get file info for this asset
            file_r = requests.get(f"https://api.polyhaven.com/files/{key}", headers=headers)
            file_r.raise_for_status()
            file_info = file_r.json()
            
            # Find the download URL for the requested resolution and format
            hdri_files = file_info.get("hdri", {})
            res_info = hdri_files.get(args.res, {})
            format_info = res_info.get(args.format, {})
            
            if not format_info:
                # Fallback to hdr if exr not available or vice versa
                alt_format = "hdr" if args.format == "exr" else "exr"
                format_info = res_info.get(alt_format, {})
                if not format_info:
                    print(f"\nSkipping {key}: Resolution {args.res} not available.")
                    continue
                print(f"\nNote: {key} format {args.format} not found, using {alt_format} instead.")
            
            download_url = format_info.get("url")
            if not download_url:
                print(f"\nSkipping {key}: No download URL found.")
                continue
                
            file_name = f"{key}_{args.res}.{format_info.get('extension', args.format)}"
            dest_path = output_path / file_name
            
            if dest_path.exists():
                n_success += 1
                continue
                
            download_file(download_url, dest_path)
            n_success += 1
            
        except Exception as e:
            print(f"\nError downloading {key}: {e}")
            
    print(f"\nFinished! Downloaded {n_success} images to {args.output_dir}")

if __name__ == "__main__":
    main()
