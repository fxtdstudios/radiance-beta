#!/usr/bin/env python
"""Create a simple validation contact sheet from generated preview PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview_dir", required=True)
    ap.add_argument("--out", default="validation_contact_sheet.jpg")
    args = ap.parse_args()
    paths = sorted(Path(args.preview_dir).glob("step_*/*.png"))
    if not paths:
        raise SystemExit("No preview images found")
    thumbs = []
    for p in paths[:40]:
        im = Image.open(p).convert("RGB"); im.thumbnail((220, 140))
        canvas = Image.new("RGB", (240, 175), "white")
        canvas.paste(im, ((240 - im.width)//2, 5))
        d = ImageDraw.Draw(canvas); d.text((8, 148), p.parent.name + "/" + p.name[:18], fill=(0,0,0))
        thumbs.append(canvas)
    cols = 5; rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 240, rows * 175), "white")
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * 240, (i // cols) * 175))
    sheet.save(args.out, quality=92)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
