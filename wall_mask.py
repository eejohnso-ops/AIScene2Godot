#!/usr/bin/env python3
"""wall_mask.py -- infer which pixels are FLAT WALL vs an OBJECT AGAINST the wall.

Two independent signals, both already produced by the reconstruction pipeline, are
fused into a debug image so we can judge how cleanly a single photo separates real
wall from the stuff in front of it (furniture, plants, sofa backs, wall hangings):

  1. GEOMETRY (the strong one). detect_planes() already RANSAC-fits the wall planes.
     Back-project every pixel with the depth map and compare its depth to where the
     nearest wall plane sits along that ray:
        residual = actual_depth - wall_plane_depth
        |residual| ~ 0           -> flat wall (on the plane)
        residual  <  -thresh     -> IN FRONT of the wall = object against it
        residual  >  +thresh     -> behind (opening/window/door)
     "Sticks out toward the camera" is almost the literal definition of
     object-against-wall, and it doesn't care what the object is.

  2. SEMANTICS. SegFormer/ADE20K labels wall(0) vs painting/mirror/window/curtain/
     furniture. Used to confirm the geometry and to flag wall *hangings* (which are
     near-planar but not paintable flat wall).

Output: <stem>_wallmask.png, a 2x2 panel:
  [ original ]                 [ geometry: green=flat, red=object, blue=behind ]
  [ semantics: wall/hangings ] [ FUSED flat-wall mask (green) + objects (red)  ]

Run (GPU; needs the DepthAnything checkpoint + SegFormer):
    python wall_mask.py godot_viewer/demo_depthalign/source.png
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

from room_from_image import (classify_planes, depth_to_pointcloud, detect_planes,
                             estimate_depth)

THRESH = 0.06       # metres: |residual| under this is "on the wall plane"
NEAR_WALL = 0.75    # metres: only classify pixels within this of a wall plane
DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints",
                            "depth_anything_v2_metric_hypersim_vitl.pth")


def _plane_residual(depth, plane, dirx, diry, dirz):
    """Signed depth residual to one plane along each pixel ray (depth - plane_depth);
    inf where the plane is edge-on or behind the camera."""
    n = plane.normal.astype(np.float32)
    ndotd = dirx * n[0] + diry * n[1] + dirz * n[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = plane.offset / ndotd
    res = depth - t
    valid = (np.abs(ndotd) > 1e-3) & (t > 0.1) & (t < 25.0)
    return np.where(valid, res, np.inf)


def nearest_wall_residual(depth, planes, fx, fy, cx, cy):
    """Per-pixel signed residual to the nearest WALL plane, plus a `wall_region`
    mask (the pixel's nearest plane overall is a wall, not floor/ceiling -- this
    keeps the ceiling/floor from being mistaken for objects in front of a wall).
    residual = depth - plane_depth; <0 means in front of the wall (object)."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    dirx, diry, dirz = (u - cx) / fx, (v - cy) / fy, np.ones((H, W), np.float32)

    best_res = np.full((H, W), np.inf, dtype=np.float32)
    best_abs = np.full((H, W), np.inf, dtype=np.float32)
    other_abs = np.full((H, W), np.inf, dtype=np.float32)   # floor/ceiling
    walls = [p for p in planes if p.label.startswith("wall")]
    for p in planes:
        res = _plane_residual(depth, p, dirx, diry, dirz)
        a = np.abs(res)
        if p.label.startswith("wall"):
            take = a < best_abs
            best_abs = np.where(take, a, best_abs)
            best_res = np.where(take, res, best_res)
        else:
            other_abs = np.minimum(other_abs, a)
    wall_region = best_abs < other_abs        # nearest plane is a wall
    return best_res, best_abs, wall_region, len(walls)


def flat_wall_masks(depth, planes, fx, fy, cx, cy, *, image_path=None,
                    thresh=THRESH, semantics=True):
    """Fuse geometry + semantics into the flat-wall / object-against-wall masks.

    Returns a dict of boolean HxW masks: flat (sample-able wall), objects
    (against-wall stuff to exclude), plus the intermediate geom/semantic layers.
    semantics=True runs SegFormer (needs the image path) to also exclude wall
    hangings (paintings/windows/curtains) that are near-coplanar with the wall.
    """
    res, absres, wall_region, n_walls = nearest_wall_residual(
        depth, planes, fx, fy, cx, cy)
    near = wall_region & (absres < NEAR_WALL)
    flat_geom = near & (absres < thresh)
    object_front = near & (res < -thresh)          # protrudes toward camera
    behind = near & (res > thresh)                 # opening / window

    wall_sem = np.ones(depth.shape, bool)
    hanging = np.zeros(depth.shape, bool)
    if semantics and image_path:
        from segment_room import group_surfaces, segment_image
        label_map, id2label = segment_image(image_path)
        masks = group_surfaces(label_map, id2label)
        wall_sem = masks.get("wall", wall_sem)
        for k in ("painting", "mirror", "window", "curtain", "door"):
            if k in masks:
                hanging |= masks[k]

    return dict(flat=flat_geom & wall_sem,
                objects=object_front | (near & hanging),
                flat_geom=flat_geom, object_front=object_front, behind=behind,
                wall_sem=wall_sem, hanging=hanging, n_walls=n_walls)


def sample_wall_color(rgb, flat, min_px=200):
    """Robust median wall colour from flat-wall pixels only, dropping the darkest/
    brightest 10% by luminance (kills shadow behind furniture and window/spec
    leak). Returns (r,g,b) ints, or None if too few flat pixels."""
    px = rgb[flat].astype(np.float32)
    if len(px) < min_px:
        return None
    lum = px @ np.array([0.299, 0.587, 0.114], np.float32)
    lo, hi = np.percentile(lum, [10, 90])
    keep = (lum >= lo) & (lum <= hi)
    if keep.sum() < min_px // 2:
        keep = np.ones(len(px), bool)
    return tuple(int(round(c)) for c in np.median(px[keep], axis=0))


def overlay(rgb, dim=0.35):
    return (rgb.astype(np.float32) * dim).astype(np.uint8)


def paint(base, mask, color):
    out = base.copy()
    out[mask] = color
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="room interior photo")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--thresh", type=float, default=THRESH,
                    help="metres: on-plane tolerance (default 0.06)")
    ap.add_argument("--no-semantics", action="store_true",
                    help="geometry only (skip SegFormer)")
    ap.add_argument("--out", default=None, help="output PNG (default <stem>_wallmask.png)")
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"Image not found: {args.image}")

    print("[1/3] depth + planes...")
    depth, rgb = estimate_depth(args.image, args.checkpoint)
    points, _, fx, fy, cx, cy = depth_to_pointcloud(depth, rgb, args.hfov)
    planes = detect_planes(points)
    classify_planes(planes, points)
    print("  planes: " + ", ".join(p.label for p in planes))

    print("[2/3] geometry residual + semantics...")
    m = flat_wall_masks(depth, planes, fx, fy, cx, cy, image_path=args.image,
                        thresh=args.thresh, semantics=not args.no_semantics)
    if m["n_walls"] == 0:
        sys.exit("No wall planes detected; nothing to classify.")
    flat, objects = m["flat"], m["objects"]
    flat_geom, behind = m["flat_geom"], m["behind"]
    object_front, wall_sem, hanging = m["object_front"], m["wall_sem"], m["hanging"]

    color = sample_wall_color(rgb, flat)
    print(f"[3/3] flat wall: {100*flat.mean():.1f}%  "
          f"object-against-wall: {100*objects.mean():.1f}%  "
          f"(geom-only flat: {100*flat_geom.mean():.1f}%, "
          f"semantic wall: {100*wall_sem.mean():.1f}%)")
    print(f"  robust flat-wall colour: RGB{color}")

    # --- panels ---
    base = overlay(rgb)
    pA = rgb
    pB = base.copy()
    pB[behind] = (40, 90, 200)
    pB[object_front] = (210, 40, 40)
    pB[flat_geom] = (40, 200, 70)
    pC = base.copy()
    pC[wall_sem] = (40, 160, 70)
    pC[hanging] = (210, 60, 200)
    pD = base.copy()
    pD[objects] = (210, 40, 40)
    pD[flat] = (40, 200, 70)

    top = np.concatenate([pA, pB], axis=1)
    bot = np.concatenate([pC, pD], axis=1)
    grid = np.concatenate([top, bot], axis=0)
    out = args.out or os.path.splitext(args.image)[0] + "_wallmask.png"
    Image.fromarray(grid).save(out)
    mb = os.path.getsize(out) / 1e6
    print(f"\nwrote {out}  ({mb:.1f} MB)")
    print("panels: [orig | geometry]  /  [semantics | fused flat-wall mask]")


if __name__ == "__main__":
    main()
