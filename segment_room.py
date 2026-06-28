#!/usr/bin/env python3
"""
segment_room.py -- identify surfaces (wall, floor, ceiling, window, door) in a
room interior photo using semantic segmentation.

Uses SegFormer (ADE20K) via HuggingFace Transformers. Outputs a JSON with
per-surface average colors and area percentages, plus an optional visualization.

Usage:
    python segment_room.py photo.jpg
    python segment_room.py photo.jpg --name living_room --visualize

Deps:  pip install torch transformers Pillow numpy
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image


# ADE20K class index -> name (0-indexed, 150 classes)
# Only the indoor-relevant subset is listed; full list at
# https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8
ADE20K_SURFACE_MAP = {
    "wall": [0],
    "floor": [3],
    "ceiling": [5],
    "window": [8],
    "door": [14],
    "curtain": [18],
    "painting": [22],
    "mirror": [27],
    "rug": [28],
    "column": [42],
    "light": [82],
    "chandelier": [85],
    "sconce": [134],
}

SURFACE_COLORS = {
    "wall": (180, 120, 80),
    "floor": (140, 100, 60),
    "ceiling": (200, 200, 220),
    "window": (100, 180, 240),
    "door": (160, 100, 50),
    "curtain": (120, 80, 140),
    "painting": (200, 50, 50),
    "mirror": (150, 200, 200),
    "rug": (100, 140, 80),
    "column": (180, 180, 160),
    "light": (255, 240, 100),
    "chandelier": (255, 220, 80),
    "sconce": (255, 200, 100),
    "object": (128, 128, 128),
}


_segmenter = None


def load_segmenter():
    """Load SegFormer ADE20K model, cached after first call."""
    global _segmenter
    if _segmenter is not None:
        return _segmenter

    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    import torch

    model_id = "nvidia/segformer-b5-finetuned-ade-640-640"
    print(f"Loading {model_id}...")
    processor = SegformerImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id)
    model = model.to("cuda").eval()
    _segmenter = (processor, model)
    return _segmenter


def segment_image(image_path: str) -> tuple[np.ndarray, dict]:
    """Run segmentation on an image.

    Returns (label_map HxW int32, id2label dict).
    """
    import torch
    import torch.nn.functional as F

    processor, model = load_segmenter()
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    inputs = processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    upsampled = F.interpolate(logits, size=(h, w), mode="bilinear",
                              align_corners=False)
    label_map = upsampled.argmax(dim=1).squeeze().cpu().numpy().astype(np.int32)

    return label_map, model.config.id2label


def group_surfaces(
    label_map: np.ndarray, id2label: dict
) -> dict[str, np.ndarray]:
    """Group ADE20K pixel labels into surface categories.

    Returns {surface_name: boolean mask HxW}.
    """
    reverse_map = {}
    for surface, ade_ids in ADE20K_SURFACE_MAP.items():
        for aid in ade_ids:
            reverse_map[aid] = surface

    masks = {}
    for aid in np.unique(label_map):
        surface = reverse_map.get(int(aid), "object")
        if surface not in masks:
            masks[surface] = np.zeros_like(label_map, dtype=bool)
        masks[surface] |= (label_map == aid)

    return masks


def sample_surface_colors(
    image: np.ndarray, masks: dict[str, np.ndarray]
) -> dict:
    """Compute per-surface statistics: average color, area, bounding box."""
    h, w = image.shape[:2]
    total_px = h * w
    surfaces = {}

    for name, mask in masks.items():
        count = int(mask.sum())
        if count == 0:
            continue

        pixels = image[mask]
        avg_color = pixels.mean(axis=0).round().astype(int).tolist()

        ys, xs = np.where(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        surfaces[name] = {
            "color": avg_color,
            "area_pct": round(100.0 * count / total_px, 1),
            "pixel_count": count,
            "bbox": bbox,
        }

    surfaces = dict(sorted(surfaces.items(),
                           key=lambda x: x[1]["area_pct"], reverse=True))
    return surfaces


def save_visualization(
    image: np.ndarray, masks: dict[str, np.ndarray], out_path: str
) -> None:
    """Save a color-coded segmentation overlay."""
    overlay = image.copy().astype(np.float32)
    alpha = 0.45

    for name, mask in masks.items():
        if not mask.any():
            continue
        color = np.array(SURFACE_COLORS.get(name, (128, 128, 128)),
                         dtype=np.float32)
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

    out = Image.fromarray(overlay.astype(np.uint8))
    out.save(out_path)
    print(f"  visualization: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Segment room surfaces in an interior photo")
    ap.add_argument("image", help="room interior photo")
    ap.add_argument("--name", default="room",
                    help="base name for output files")
    ap.add_argument("--out-dir",
                    default=os.path.join(os.path.dirname(__file__),
                                         "godot_viewer"),
                    help="output directory")
    ap.add_argument("--visualize", action="store_true",
                    help="save a color-coded overlay image")
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"Image not found: {args.image}")

    print(f"[1/3] segmenting {args.image}...")
    label_map, id2label = segment_image(args.image)
    print(f"  {label_map.shape[1]}x{label_map.shape[0]} px, "
          f"{len(np.unique(label_map))} classes detected")

    print("[2/3] grouping surfaces...")
    masks = group_surfaces(label_map, id2label)
    image = np.array(Image.open(args.image).convert("RGB"))
    surfaces = sample_surface_colors(image, masks)

    for name, info in surfaces.items():
        r, g, b = info["color"]
        print(f"  {name:12s}  RGB({r:3d},{g:3d},{b:3d})  "
              f"{info['area_pct']:5.1f}% of image")

    print("[3/3] saving outputs...")
    os.makedirs(args.out_dir, exist_ok=True)

    json_path = os.path.join(args.out_dir, f"{args.name}_surfaces.json")
    with open(json_path, "w") as f:
        json.dump(surfaces, f, indent=2)
    print(f"  surfaces: {json_path}")

    if args.visualize:
        vis_path = os.path.join(args.out_dir, f"{args.name}_segments.png")
        save_visualization(image, masks, vis_path)

    print(f"\nDone. {len(surfaces)} surfaces identified.")


if __name__ == "__main__":
    main()
