"""Multi-view TEXTURE de-occlusion on the PROVEN single-view geometry (Phase 3).

The single-photo room box is reliable; what's missing is wall/floor texture HIDDEN
behind furniture. This keeps the single-view geometry and completes each wall's
texture by compositing several sources in priority order, per texel:

  1. base photo, where it shows genuine FLAT WALL (real, unoccluded)   -> 'base'
  2. a novel view's REAL reprojected pixel (not in that view's hole)   -> 'real'
  3. a novel view's INPAINTED pixel (Phase B filled the disocclusion)  -> 'fill'
  4. nothing saw it                                                    -> 'hole'

Novel views come from novel_views.py (+ novel_views_comfy.py) with KNOWN poses, so
we project with those poses directly -- no VGGT re-triangulation (which is
degenerate for single-image-derived views; see [[multiview-reconstruction]]).

    python wall_complete.py base.png --views out/nv_demo --out out/walls
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

from room_from_image import (reconstruct_room_scene, extract_room_params,
                             build_room_quads)
from wall_mask import flat_wall_masks


def _project(P, K, R, t):
    """Project base-frame points P (N,3) into a camera (K,R,t). Returns u,v px +
    in-front mask."""
    p = P @ R.T + t
    z = p[:, 2]
    infront = z > 1e-3
    zc = np.clip(z, 1e-3, None)
    u = K[0, 0] * p[:, 0] / zc + K[0, 2]
    v = K[1, 1] * p[:, 1] / zc + K[1, 2]
    return u, v, infront


def _gather(img, mask_imgs, u, v, infront):
    """Nearest-pixel sample of img (+ optional bool masks) at float (u,v). Returns
    rgb (M,3) and in-bounds mask; mask_imgs sampled alongside as a dict."""
    H, W = img.shape[:2]
    ui = np.round(u).astype(np.int64); vi = np.round(v).astype(np.int64)
    ok = infront & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    uic = np.clip(ui, 0, W - 1); vic = np.clip(vi, 0, H - 1)
    rgb = img[vic, uic]
    sampled = {k: m[vic, uic] for k, m in mask_imgs.items()}
    return rgb, ok, sampled


def complete_wall(label, corners, K, base_rgb, base_flat, novels, ppm=80):
    """Composite a completed texture for one wall. `corners` = the 4 quad verts
    (v0,v1,v2,v3) in base frame from build_room_quads (v0 origin, v0->v2 = width,
    v0->v1 = height). `novels` = list of dicts {rgb, holes, R, t}. Returns
    (rgb uint8 HtxWs x3, source-id map, stats)."""
    v0, v1, v2, v3 = corners
    Ls = np.linalg.norm(v2 - v0); Lt = np.linalg.norm(v1 - v0)
    ws = max(8, int(Ls * ppm)); ht = max(8, int(Lt * ppm))
    s = (np.arange(ws) + 0.5) / ws
    t = (np.arange(ht) + 0.5) / ht
    ss, tt = np.meshgrid(s, t)                       # (ht, ws)
    P = (v0[None, None] + ss[..., None] * (v2 - v0)[None, None]
         + tt[..., None] * (v1 - v0)[None, None]).reshape(-1, 3)

    out = np.zeros((ht * ws, 3), np.uint8)
    src = np.zeros(ht * ws, np.uint8)                # 0 hole,1 base,2 real,3 fill
    need = np.ones(ht * ws, bool)

    # 1) base flat-wall pixels
    u, v, inf = _project(P, K, np.eye(3), np.zeros(3))
    rgb, ok, sm = _gather(base_rgb, {"flat": base_flat}, u, v, inf)
    take = need & ok & sm["flat"]
    out[take] = rgb[take]; src[take] = 1; need[take] = False

    # 2) novel REAL reprojected pixels, then 3) novel inpainted fills
    for pref, want_real in ((2, True), (3, False)):
        for nv in novels:
            if not need.any():
                break
            u, v, inf = _project(P, K, nv["R"], nv["t"])
            rgb, ok, sm = _gather(nv["rgb"], {"hole": nv["holes"]}, u, v, inf)
            real = ~sm["hole"]
            take = need & ok & (real if want_real else ~real)
            out[take] = rgb[take]; src[take] = pref; need[take] = False

    out = out.reshape(ht, ws, 3); src = src.reshape(ht, ws)
    n = ht * ws
    stats = {"base": int((src == 1).sum()), "real": int((src == 2).sum()),
             "fill": int((src == 3).sum()), "hole": int((src == 0).sum()), "n": n}
    return out, src, stats


def _load_novels(views_dir, base_w, base_h):
    """Load novel frames (frames/view_i.png) + hole masks + poses. view_0 is the
    base photo (skipped here -- it's the base_rgb input)."""
    pj = os.path.join(views_dir, "poses.json")
    with open(pj) as f:
        meta = json.load(f)
    novels = []
    for vmeta in meta["views"]:
        i = vmeta["index"]
        if vmeta.get("base"):
            continue
        fp = os.path.join(views_dir, "frames", f"view_{i}.png")
        hp = os.path.join(views_dir, f"view_{i}_holes.png")
        if not (os.path.isfile(fp) and os.path.isfile(hp)):
            continue
        img = Image.open(fp).convert("RGB")
        if img.size != (base_w, base_h):
            img = img.resize((base_w, base_h), Image.BILINEAR)
        holes = Image.open(hp).convert("L")
        if holes.size != (base_w, base_h):
            holes = holes.resize((base_w, base_h), Image.NEAREST)
        R = np.array(vmeta["R"], float); C = np.array(vmeta["C"], float)
        novels.append({"rgb": np.array(img),
                       "holes": np.array(holes) > 128,
                       "R": R, "t": -R @ C})        # world->cam: p_nov = R(P - C)
    return novels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="base room photo")
    ap.add_argument("--views", required=True, help="novel_views emit-dir")
    ap.add_argument("--out", default="out/walls", help="output dir for textures")
    ap.add_argument("--checkpoint",
                    default=os.path.join(os.path.dirname(__file__), "checkpoints",
                                         "depth_anything_v2_metric_hypersim_vitl.pth"))
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--ppm", type=int, default=80, help="texture px per metre")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("[1/3] single-view geometry + flat-wall mask...")
    scene, ex = reconstruct_room_scene(args.image, args.checkpoint, hfov=args.hfov,
                                       subdivisions=1, return_extras=True)
    fx, fy, cx, cy = ex["intrinsics"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    rgb = ex["rgb"]; H, W = rgb.shape[:2]
    masks = flat_wall_masks(ex["depth"], ex["planes"], fx, fy, cx, cy,
                            image_path=args.image, semantics=True)
    base_flat = masks["flat"]

    print("[2/3] loading novel views...")
    novels = _load_novels(args.views, W, H)
    print(f"    {len(novels)} novel views")

    print("[3/3] completing wall textures...")
    params = extract_room_params(ex["planes"], ex["points"])
    quads = build_room_quads(params, subdivisions=1)
    grand = {"base": 0, "real": 0, "fill": 0, "hole": 0, "n": 0}
    for label, verts, faces, uvs in quads:
        if not label.startswith("wall"):
            continue
        tex, src, st = complete_wall(label, verts, K, rgb, base_flat, novels,
                                     args.ppm)
        Image.fromarray(tex).save(os.path.join(args.out, f"{label}_complete.png"))
        # base-only baseline (no novels) for comparison
        _, _, st0 = complete_wall(label, verts, K, rgb, base_flat, [], args.ppm)
        for k in grand:
            grand[k] += st[k]
        n = st["n"]
        print(f"  {label}: base {100*st['base']/n:4.1f}%  "
              f"+real {100*st['real']/n:4.1f}%  +fill {100*st['fill']/n:4.1f}%  "
              f"hole {100*st['hole']/n:4.1f}%   "
              f"(base-only would leave {100*(n-st0['base'])/n:4.1f}% blank)")
    n = grand["n"] or 1
    print(f"\nALL WALLS: base {100*grand['base']/n:.1f}%  real {100*grand['real']/n:.1f}%"
          f"  fill {100*grand['fill']/n:.1f}%  hole {100*grand['hole']/n:.1f}%")
    print(f"wrote completed wall textures to {args.out}/")


if __name__ == "__main__":
    main()
