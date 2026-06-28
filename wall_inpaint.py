#!/usr/bin/env python3
"""wall_inpaint.py -- rectify each wall to a head-on texture, inpaint the occluded
parts, and preview the result.

This is the "complete the wall" half of the flat-wall-mask work (see wall_mask.py).
For each wall:
  1. Take the wall's 3D rectangle (from extract_room_params) and project its corners
     into the photo -> a homography image->ortho-rectangle.
  2. Warp the photo to a head-on wall texture (cv2.warpPerspective). This removes the
     foreshortening that plain projection leaves on side walls.
  3. Holes = pixels that are NOT flat wall (furniture/hangings, from the flat-wall
     mask) OR fell outside the photo (wall the camera never saw).
  4. Inpaint the holes from the surrounding real wall (cv2.inpaint; classical Telea/
     NS to gauge -- diffusion would be the quality upgrade).

The rectified+inpainted image is a clean, head-on, tileable wall texture: applied
with ordinary [0,1] UVs it needs no per-vertex projection. This tool just previews;
wiring into build_room_scene is the next step once the quality looks right.

Run (GPU; needs DepthAnything + SegFormer):
    python wall_inpaint.py godot_viewer/demo_depthalign/source.png
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

from room_from_image import (classify_planes, depth_to_pointcloud, detect_planes,
                             estimate_depth, extract_room_params)
from wall_mask import DEFAULT_CKPT, flat_wall_masks, sample_wall_color

PPM_TEX = 96        # texture pixels per metre
MAX_TEX = 1024      # clamp the larger texture dimension
INPAINT_RADIUS = 4


def _wall_corners(params):
    """3D corners (TL, TR, BR, BL) per wall in the camera frame, top=ceiling.
    Texture x runs left->right (back) or front->back (sides)."""
    lx, rx = params["left_x"], params["right_x"]
    fl, ce = params["floor_y"], params["ceiling_y"]      # floor below, ceiling above
    fz, bz = params["front_z"], params["back_z"]
    return {
        "wall_back":  [(lx, ce, bz), (rx, ce, bz), (rx, fl, bz), (lx, fl, bz)],
        "wall_left":  [(lx, ce, fz), (lx, ce, bz), (lx, fl, bz), (lx, fl, fz)],
        "wall_right": [(rx, ce, fz), (rx, ce, bz), (rx, fl, bz), (rx, fl, fz)],
    }


def _tex_size(corners):
    """Texture WxH in pixels from the wall's metric extents (corner spacing)."""
    p = np.array(corners, np.float32)
    w_m = np.linalg.norm(p[1] - p[0])      # TL->TR
    h_m = np.linalg.norm(p[3] - p[0])      # TL->BL
    w = int(np.clip(w_m * PPM_TEX, 16, MAX_TEX))
    h = int(np.clip(h_m * PPM_TEX, 16, MAX_TEX))
    return w, h


def _project(pts3d, fx, fy, cxp, cyp):
    p = np.asarray(pts3d, np.float32)
    z = np.clip(p[:, 2], 1e-3, None)
    u = fx * p[:, 0] / z + cxp
    v = fy * p[:, 1] / z + cyp
    return np.stack([u, v], axis=-1).astype(np.float32)


def rectify_wall(rgb, flat, corners, fx, fy, cxp, cyp):
    """Warp the wall to a head-on rectangle. Returns (warp RGB, holes uint8 0/1).
    holes = pixels that aren't visible flat wall (furniture/hangings) or fell
    outside the photo (wall the camera never saw) -- the regions to fill."""
    Wt, Ht = _tex_size(corners)
    src = _project(corners, fx, fy, cxp, cyp)
    dst = np.float32([[0, 0], [Wt, 0], [Wt, Ht], [0, Ht]])
    Hmat = cv2.getPerspectiveTransform(src, dst)

    warp = cv2.warpPerspective(rgb, Hmat, (Wt, Ht), flags=cv2.INTER_LINEAR)
    flat_w = cv2.warpPerspective((flat.astype(np.uint8) * 255), Hmat, (Wt, Ht),
                                 flags=cv2.INTER_NEAREST)
    inside = cv2.warpPerspective(np.ones(rgb.shape[:2], np.uint8), Hmat, (Wt, Ht),
                                 flags=cv2.INTER_NEAREST)
    holes = ((flat_w < 128) | (inside == 0)).astype(np.uint8)
    # Clean up speckle so an inpainter has solid wall to borrow from.
    holes = cv2.morphologyEx(holes, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return warp, holes


def classical_fill(warp, holes, method):
    flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA
    return cv2.inpaint(warp, holes, INPAINT_RADIUS, flag)


def bake_wall_texture(warp, holes, base_color, grain_amp=2.5):
    """A clean, head-on wall texture: the sampled flat-wall base colour modulated by
    the photo's real LOW-FREQUENCY lighting, plus faint grain. The holes are smeared
    (classical) then heavily blurred, so only the smooth lighting gradient survives
    -- no furniture, no diamond smear artifacts (those are high-frequency). Returns a
    PIL image flipped vertically to match the viewer's V convention (same reason
    _project_uvs uses 1-v)."""
    h, w = holes.shape
    gray = (warp.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32))
    filled = cv2.inpaint(gray.astype(np.uint8), holes, 8,
                         cv2.INPAINT_TELEA).astype(np.float32)
    light = cv2.GaussianBlur(filled, (0, 0), sigmaX=max(w, h) / 12.0)
    flat = holes == 0
    ref = float(np.median(light[flat])) if flat.any() else float(np.median(light))
    light = np.clip(light / max(ref, 1e-3), 0.82, 1.22)        # gentle ~1 multiplier
    tex = np.array(base_color, np.float32)[None, None, :] * light[:, :, None]
    tex += np.random.RandomState(0).normal(0, grain_amp, tex.shape)
    return Image.fromarray(np.flipud(np.clip(tex, 0, 255).astype(np.uint8)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--method", choices=["telea", "ns"], default="telea")
    ap.add_argument("--out", default=None)
    ap.add_argument("--emit-dir", default=None,
                    help="also write each wall's rectified RGB + hole mask here "
                         "(<label>_wall.png / <label>_holes.png) as inputs for the "
                         "decoupled ComfyUI diffusion pass (comfy_inpaint.py)")
    args = ap.parse_args()
    if not os.path.isfile(args.image):
        sys.exit(f"Image not found: {args.image}")

    print("[1/3] depth + planes...")
    depth, rgb = estimate_depth(args.image, args.checkpoint)
    points, _, fx, fy, cxp, cyp = depth_to_pointcloud(depth, rgb, args.hfov)
    planes = detect_planes(points)
    classify_planes(planes, points)
    params = extract_room_params(planes, points)

    print("[2/3] flat-wall mask...")
    m = flat_wall_masks(depth, planes, fx, fy, cxp, cyp,
                        image_path=args.image, semantics=True)

    print("[3/3] rectify + inpaint each wall...")
    corners = _wall_corners(params)
    if args.emit_dir:
        os.makedirs(args.emit_dir, exist_ok=True)
    rows = []
    for label in ("wall_back", "wall_left", "wall_right"):
        warp, holes = rectify_wall(rgb, m["flat"], corners[label],
                                   fx, fy, cxp, cyp)
        tex = classical_fill(warp, holes, args.method)
        print(f"  {label}: {warp.shape[1]}x{warp.shape[0]}  holes "
              f"{100 * holes.mean():.0f}%")
        if args.emit_dir:
            Image.fromarray(warp).save(os.path.join(args.emit_dir, f"{label}_wall.png"))
            Image.fromarray(holes * 255).save(
                os.path.join(args.emit_dir, f"{label}_holes.png"))
        marked = warp.copy()
        marked[holes.astype(bool)] = (255, 0, 255)
        rows.append(np.concatenate([marked, tex], axis=1))

    Wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 8, 8, 0, Wmax - r.shape[1], cv2.BORDER_CONSTANT,
                               value=(20, 20, 20)) for r in rows]
    grid = np.concatenate(rows, axis=0)
    out = args.out or os.path.splitext(args.image)[0] + "_inpaint.png"
    Image.fromarray(grid).save(out)
    print(f"\nwrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    print("rows: wall_back / wall_left / wall_right;  cols: [holes magenta | inpainted]")


if __name__ == "__main__":
    main()
