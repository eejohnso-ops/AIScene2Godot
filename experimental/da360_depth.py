"""
da360_depth.py -- run Depth Anywhere (default UniFuse) on ONE equirect panorama
and save the depth as a float .npy.

Why this exists: Depth-Anywhere's own inference.py normalises depth to 0-1 and
writes an 8-bit PNG, which quantises geometry to 256 levels and throws away
metric scale. build_room.py's seam-free path wants the raw float depth, so this
script saves out["pred_depth"] directly.

RUN IT FROM INSIDE THE Depth-Anywhere REPO (so the baseline_models/ imports
resolve), in that repo's conda env:

    conda activate depth-anywhere
    python da360_depth.py \
        --pano  "C:/Users/eejoh/projects/World Builder/out/pano_2.png" \
        --weight checkpoints/UniFuse/UniFuse_SpatialAudioGen.pth \
        --out   "C:/Users/eejoh/projects/World Builder/out/pano_2_depth.npy"

Then back in your build_room env:

    python build_room.py --pano out/pano_2.png --depth out/pano_2_depth.npy --name room_v2
"""
import argparse
import numpy as np
import torch

# Reuse Depth-Anywhere's own loaders (UniFuse model + the E2C preprocessing).
from inference import load_model, load_rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pano", required=True, help="equirect panorama PNG/JPG")
    ap.add_argument("--out", required=True, help="output .npy float depth path")
    ap.add_argument("--weight",
                    default="checkpoints/UniFuse/UniFuse_SpatialAudioGen.pth",
                    help="path to the 360 model checkpoint (.pth)")
    ap.add_argument("--model", default="UniFuse",
                    help="UniFuse | BiFuseV2 | HoHoNet | EGformer")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  model: {args.model}")
    model = load_model(args.weight, device, args.model)
    model.eval()

    rgb, cube = load_rgb(args.pano, device=device)
    with torch.no_grad():
        out = model(rgb, cube) if args.model.upper() == "UNIFUSE" else model(rgb)

    depth = out["pred_depth"].squeeze().cpu().numpy().astype(np.float32)
    np.save(args.out, depth)
    print(f"saved float depth {depth.shape}  range "
          f"{depth.min():.3f}..{depth.max():.3f} m  -> {args.out}")


if __name__ == "__main__":
    main()
