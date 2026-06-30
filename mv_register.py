"""Multi-view registration via VGGT.

Feed N views of one room to VGGT (Visual Geometry Grounded Transformer) and get
back, in ONE shared world frame:
  - per-view camera extrinsics (cam-from-world, OpenCV convention) + intrinsics,
  - per-view metric-ish depth maps + confidence,
  - a fused point cloud (all views unprojected into the shared frame).

This is the pose/registration layer the single-view pipeline never had (see
room_from_image.py, which is single-camera-at-origin with a blind 60deg hfov).
The fused cloud + per-view cameras feed build_room_multiview.py, which runs the
existing plane/room-box/mesh geometry on the fused cloud and textures each wall
from whichever view saw it best.

VGGT is pure PyTorch (no custom CUDA kernels) and runs in the project's main
.venv on Blackwell -- so this imports `vggt` directly rather than shelling out to
a separate interpreter the way MIDI does. Register the package once with:
    uv pip install --python .venv/Scripts/python.exe --no-deps -e external/VGGT
(--no-deps so its pinned torch==2.3.1, which predates Blackwell, is NOT pulled;
the main venv's torch 2.8.0+cu128 is used instead).

CLI:
    python mv_register.py --images path/to/views_dir --out bundle.npz
    python mv_register.py --images a.png b.png c.png --out bundle.npz
"""
from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class ViewCamera:
    """One view's camera in the shared VGGT world frame (OpenCV convention:
    +x right, +y down, +z forward; extrinsic maps world -> camera)."""
    name: str
    K: np.ndarray          # (3,3) intrinsics in pixels of (width, height)
    R: np.ndarray          # (3,3) rotation, world -> camera
    t: np.ndarray          # (3,)  translation, world -> camera
    width: int
    height: int

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coords: C = -R^T t."""
        return -self.R.T @ self.t

    @property
    def forward(self) -> np.ndarray:
        """Viewing direction in world coords (camera +z mapped to world)."""
        return self.R.T @ np.array([0.0, 0.0, 1.0])

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.width / (2 * self.K[0, 0])))


def _select_images(patterns: list[str]) -> list[str]:
    """Expand a directory / globs / explicit files into a sorted image list."""
    out: list[str] = []
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    for p in patterns:
        if os.path.isdir(p):
            for e in exts:
                out += glob.glob(os.path.join(p, f"*{e}"))
                out += glob.glob(os.path.join(p, f"*{e.upper()}"))
        elif any(ch in p for ch in "*?[]"):
            out += glob.glob(p)
        else:
            out.append(p)
    # de-dup, keep deterministic order
    seen, uniq = set(), []
    for f in sorted(out):
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def run_vggt(image_paths: list[str], *, device: str | None = None,
             mode: str = "crop", weights: str = "facebook/VGGT-1B") -> dict:
    """Run VGGT on `image_paths`. Returns a dict with numpy arrays:
        names      : list[str]            (length N)
        extrinsics : (N,3,4) world->cam
        intrinsics : (N,3,3)
        depths     : (N,H,W)              per-view depth (metric-ish, up to scale)
        depth_conf : (N,H,W)              confidence
        world_pts  : (N,H,W,3)            depth unprojected into shared world frame
        images     : (N,H,W,3) uint8      VGGT-preprocessed RGB (aligns with maps)
    H, W are VGGT's internal processing resolution (depends on `mode`).
    """
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    if not image_paths:
        raise ValueError("no images given to run_vggt")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16
             if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
             else torch.float16)

    print(f"  [vggt] loading {weights} on {device} ({dtype})...")
    model = VGGT.from_pretrained(weights).to(device).eval()

    print(f"  [vggt] preprocessing {len(image_paths)} views (mode={mode})...")
    images = load_and_preprocess_images(image_paths, mode=mode).to(device)
    H, W = int(images.shape[-2]), int(images.shape[-1])

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype) if device == "cuda" \
                else _nullctx():
            batch = images[None]                         # (1,N,3,H,W)
            tokens, ps_idx = model.aggregator(batch)
        pose_enc = model.camera_head(tokens)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc, batch.shape[-2:])
        depth_map, depth_conf = model.depth_head(tokens, batch, ps_idx)

    extr = extrinsic.squeeze(0).float().cpu().numpy()    # (N,3,4)
    intr = intrinsic.squeeze(0).float().cpu().numpy()    # (N,3,3)
    depth = depth_map.squeeze(0).float().cpu().numpy()   # (N,H,W,1)
    conf = depth_conf.squeeze(0).float().cpu().numpy()   # (N,H,W)
    world_pts = unproject_depth_map_to_point_map(depth, extr, intr)  # (N,H,W,3)

    rgb = images.permute(0, 2, 3, 1).cpu().numpy()       # (N,H,W,3) 0..1
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    print(f"  [vggt] done: {len(image_paths)} views at {W}x{H}, "
          f"hfov {_hfov(intr[0,0,0], W):.1f}-{_hfov(intr[-1,0,0], W):.1f}deg")

    return {
        "names": [os.path.basename(p) for p in image_paths],
        "paths": list(image_paths),
        "extrinsics": extr,
        "intrinsics": intr,
        "depths": depth[..., 0],          # (N,H,W)
        "depth_conf": conf,
        "world_pts": world_pts,
        "images": rgb,
        "width": W,
        "height": H,
    }


def cameras(result: dict) -> list[ViewCamera]:
    """Per-view ViewCamera objects from a run_vggt / load_bundle result."""
    W, H = int(result["width"]), int(result["height"])
    out = []
    for i, name in enumerate(result["names"]):
        e = result["extrinsics"][i]
        out.append(ViewCamera(name=name, K=result["intrinsics"][i].copy(),
                              R=e[:3, :3].copy(), t=e[:3, 3].copy(),
                              width=W, height=H))
    return out


def fuse_world_points(result: dict, conf_percentile: float = 40.0,
                      max_points: int = 1_500_000):
    """Fuse all views' unprojected points into one (P,3) cloud + (P,3) colours.

    Keeps points whose VGGT depth-confidence is above the `conf_percentile`-th
    percentile (per-view), then optionally subsamples to `max_points`.
    """
    pts_all, col_all = [], []
    for i in range(len(result["names"])):
        wp = result["world_pts"][i].reshape(-1, 3)
        cf = result["depth_conf"][i].reshape(-1)
        rgb = result["images"][i].reshape(-1, 3).astype(np.float32) / 255.0
        thr = np.percentile(cf, conf_percentile)
        keep = (cf >= thr) & np.isfinite(wp).all(axis=1)
        pts_all.append(wp[keep])
        col_all.append(rgb[keep])
    pts = np.concatenate(pts_all, axis=0)
    cols = np.concatenate(col_all, axis=0)
    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    print(f"  [fuse] {len(pts)} fused points "
          f"(conf>={conf_percentile:.0f}th pct, {len(result['names'])} views)")
    return pts, cols


def save_bundle(result: dict, path: str) -> None:
    """Persist a run_vggt result to a .npz (object dtype for the name list)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        names=np.array(result["names"], dtype=object),
        paths=np.array(result["paths"], dtype=object),
        extrinsics=result["extrinsics"], intrinsics=result["intrinsics"],
        depths=result["depths"], depth_conf=result["depth_conf"],
        world_pts=result["world_pts"], images=result["images"],
        width=result["width"], height=result["height"])
    print(f"  [bundle] wrote {path} ({os.path.getsize(path)/1e6:.1f} MB)")


def load_bundle(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    return {
        "names": list(z["names"]), "paths": list(z["paths"]),
        "extrinsics": z["extrinsics"], "intrinsics": z["intrinsics"],
        "depths": z["depths"], "depth_conf": z["depth_conf"],
        "world_pts": z["world_pts"], "images": z["images"],
        "width": int(z["width"]), "height": int(z["height"]),
    }


def _hfov(fx: float, w: int) -> float:
    return math.degrees(2 * math.atan(w / (2 * fx)))


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main():
    ap = argparse.ArgumentParser(description="VGGT multi-view registration")
    ap.add_argument("--images", nargs="+", required=True,
                    help="a directory, glob(s), or explicit image files")
    ap.add_argument("--out", required=True, help="output bundle .npz")
    ap.add_argument("--mode", default="crop", choices=["crop", "pad"],
                    help="VGGT preprocessing (crop=default, pad=keep full FOV)")
    ap.add_argument("--weights", default="facebook/VGGT-1B")
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    args = ap.parse_args()

    paths = _select_images(args.images)
    if not paths:
        raise SystemExit(f"no images matched: {args.images}")
    print(f"registering {len(paths)} views:")
    for p in paths:
        print("  ", p)

    result = run_vggt(paths, mode=args.mode, weights=args.weights)
    save_bundle(result, args.out)

    cams = cameras(result)
    print("\nper-view cameras (world frame):")
    for c in cams:
        C = c.center
        print(f"  {c.name:24s} hfov={c.hfov_deg:5.1f}deg "
              f"C=({C[0]:+.2f},{C[1]:+.2f},{C[2]:+.2f})")
    fuse_world_points(result, args.conf_percentile)


if __name__ == "__main__":
    main()
