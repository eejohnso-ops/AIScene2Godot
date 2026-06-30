"""Phase A of ComfyUI-based novel-view generation (the ViewCrafter pivot).

Render the base photo's RGB-D point cloud from a planned set of NEW camera poses,
emitting for each a PARTIAL colour image + a HOLE mask (the pixels the base view
never saw). Phase B (novel_views_comfy.py) then fills the holes via ComfyUI.

This is torch-only (DepthAnything for depth) and EXITS before Phase B runs, so no
CUDA context is held next to ComfyUI -- the same two-phase split as
wall_inpaint.py / comfy_inpaint.py (see [[comfyui-cuda-crash]]).

HONEST SCOPE: a single photo sees ~one wall, so LARGE rotations render mostly
holes and Phase B would have to hallucinate (unreliable + unregistrable). The
useful regime is MODERATE baselines -- nudging the camera to de-occlude
foreground furniture and reveal the wall behind it. Trajectories default to that.

    python novel_views.py photo.png --emit-dir views          # Phase A (GPU)
    python novel_views_comfy.py views                          # Phase B (ComfyUI)
    python build_room_multiview.py --images views/frames ...   # fuse with VGGT
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

from room_from_image import estimate_depth, depth_to_pointcloud


def _yaw(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _pitch(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def plan_trajectory(mode: str, n: int, *, pivot: float, max_yaw: float,
                    max_shift: float):
    """List of novel camera poses as (R, C): R = base->novel rotation, C = novel
    camera centre in the base-camera frame. The base view (R=I, C=0) is index 0.

    'orbit'    : yaw around a pivot point `pivot` m down +z (look stays on the
                 scene, small parallax -- de-occludes foreground). DEFAULT.
    'pan'      : rotate yaw in place (wider coverage, more holes).
    'parallax' : pure lateral/vertical translation, look direction unchanged
                 (cleanest de-occlusion, smallest holes).
    """
    poses = [(np.eye(3), np.zeros(3))]
    if n <= 1:
        return poses
    angs = np.linspace(-1, 1, n)
    for a in angs:
        if abs(a) < 1e-6:
            continue
        if mode == "pan":
            poses.append((_yaw(a * max_yaw), np.zeros(3)))
        elif mode == "parallax":
            poses.append((np.eye(3), np.array([a * max_shift, 0.0, 0.0])))
        else:  # orbit: rotate the camera about a pivot in front of it
            R = _yaw(a * max_yaw)
            piv = np.array([0.0, 0.0, pivot])
            C = piv - R.T @ piv           # keep the pivot fixed in both frames
            poses.append((R, C))
    return poses


def render_view(points: np.ndarray, colors: np.ndarray, K: np.ndarray,
                R: np.ndarray, C: np.ndarray, W: int, H: int, splat: int = 1):
    """Z-buffer splat of the base cloud from a novel pose. Returns (rgb uint8,
    holes bool) where holes marks pixels no point reached."""
    p = (points - C) @ R.T                       # base frame -> novel camera frame
    z = p[:, 2]
    front = z > 1e-3
    zc = np.clip(z, 1e-3, None)
    u = (K[0, 0] * p[:, 0] / zc + K[0, 2])
    v = (K[1, 1] * p[:, 1] / zc + K[1, 2])
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    img = np.zeros((H, W, 3), np.uint8)
    filled = np.zeros((H, W), bool)
    order = np.argsort(z)[::-1]                   # far first; near overwrites
    uo, vo, co, fo = ui[order], vi[order], colors[order], front[order]
    r = splat
    for du in range(-r, r + 1):
        for dv in range(-r, r + 1):
            uu, vv = uo + du, vo + dv
            m = fo & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            img[vv[m], uu[m]] = (co[m] * 255).astype(np.uint8)
            filled[vv[m], uu[m]] = True
    return img, ~filled


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="base room photo")
    ap.add_argument("--emit-dir", required=True, help="output dir for partials")
    ap.add_argument("--checkpoint",
                    default=os.path.join(os.path.dirname(__file__), "checkpoints",
                                         "depth_anything_v2_metric_hypersim_vitl.pth"))
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--mode", default="orbit", choices=["orbit", "pan", "parallax"])
    ap.add_argument("--views", type=int, default=5, help="total views incl. base")
    ap.add_argument("--max-yaw", type=float, default=18.0, help="deg (orbit/pan)")
    ap.add_argument("--max-shift", type=float, default=0.35, help="m (parallax)")
    ap.add_argument("--pivot", type=float, default=2.5, help="orbit pivot depth m")
    ap.add_argument("--splat", type=int, default=1, help="splat radius px")
    args = ap.parse_args()

    os.makedirs(args.emit_dir, exist_ok=True)
    frames_dir = os.path.join(args.emit_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print("[1/3] depth + base cloud...")
    depth, rgb = estimate_depth(args.image, args.checkpoint)
    points, colors, fx, fy, cx, cy = depth_to_pointcloud(depth, rgb, args.hfov)
    H, W = rgb.shape[:2]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    print(f"    {len(points)} pts, {W}x{H}, hfov {args.hfov}")

    print(f"[2/3] planning '{args.mode}' trajectory ({args.views} views)...")
    poses = plan_trajectory(args.mode, args.views, pivot=args.pivot,
                            max_yaw=args.max_yaw, max_shift=args.max_shift)

    print("[3/3] rendering novel views...")
    meta = {"image": os.path.abspath(args.image), "mode": args.mode,
            "intrinsics": [float(fx), float(fy), float(cx), float(cy)],
            "width": W, "height": H, "views": []}
    # base view = the real photo (no holes); novel views = rendered partials
    Image.fromarray(rgb).save(os.path.join(frames_dir, "view_0.png"))
    meta["views"].append({"index": 0, "base": True, "hole_frac": 0.0,
                          "R": np.eye(3).tolist(), "C": [0, 0, 0]})
    for i, (R, C) in enumerate(poses[1:], start=1):
        img, holes = render_view(points, colors, K, R, C, W, H, args.splat)
        hf = float(holes.mean())
        Image.fromarray(img).save(os.path.join(args.emit_dir, f"view_{i}_partial.png"))
        Image.fromarray((holes * 255).astype(np.uint8)).save(
            os.path.join(args.emit_dir, f"view_{i}_holes.png"))
        meta["views"].append({"index": i, "base": False, "hole_frac": hf,
                              "R": R.tolist(), "C": C.tolist()})
        print(f"    view {i}: {hf*100:5.1f}% holes")
    with open(os.path.join(args.emit_dir, "poses.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {len(poses)} views to {args.emit_dir} "
          f"(base photo + {len(poses)-1} partials).\n"
          f"Next: python novel_views_comfy.py {args.emit_dir}")


if __name__ == "__main__":
    main()
