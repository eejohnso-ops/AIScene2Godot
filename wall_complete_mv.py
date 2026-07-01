"""HYBRID room reconstruction: DepthAnything box + Wan-orbit texture completion.

The lesson from the multi-view experiments ([[multiview-reconstruction]]):
  - DepthAnything monocular-metric depth gives the correct room BOX (incl. height,
    which it extrapolates from indoor priors) -- VGGT can't (a horizontal orbit has
    no vertical parallax, so it flattens height).
  - The Wan-Fun-Camera ORBIT frames DO carry real revealed wall content (a door,
    a second window, side walls) and VGGT recovers clean poses for them.

So: take the room BOX from DepthAnything (single view), then complete each wall's
texture by projecting the base photo + the posed orbit frames onto it, in priority
  base flat-wall  >  orbit frame (occlusion-checked)  >  procedural
with a per-view occlusion gate (VGGT per-view depth) so foreground furniture in an
orbit frame does NOT bleed onto the wall (the bug the first wall_complete had).

Frames are the base photo (view 0) + the clean orbit frames. VGGT is run on all of
them together; view 0 = the source photo anchors VGGT-world to the DepthAnything
camera frame, and VGGT's arbitrary scale is solved by matching view-0 depth to the
metric DepthAnything depth.

    python wall_complete_mv.py source.png --orbit out/wan_orbit2/frames --name hybrid
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from PIL import Image

import mv_register as mvr
from room_from_image import (reconstruct_room_scene, extract_room_params,
                             build_room_quads, generate_herringbone,
                             generate_wall_texture, _center_floor_matrix, PPM)
from wall_mask import flat_wall_masks

_RNG = np.random.default_rng(0)


def _sample_quad(verts, faces, per_tri=260):
    out = []
    for f in faces:
        a, b, c = verts[f[0]], verts[f[1]], verts[f[2]]
        u = _RNG.uniform(0, 1, (per_tri, 1)); v = _RNG.uniform(0, 1, (per_tri, 1))
        m = (u + v) > 1; u = np.where(m, 1 - u, u); v = np.where(m, 1 - v, v)
        out.append(a + u * (b - a) + v * (c - a))
    return np.concatenate(out, 0)


def _face_normal(verts, face):
    n = np.cross(verts[face[1]] - verts[face[0]], verts[face[2]] - verts[face[0]])
    return n / (np.linalg.norm(n) + 1e-12)


def _proj(P, K):
    """Project camera-frame points P (N,3) with intrinsics K -> u,v,z,infront."""
    z = P[:, 2]; infront = z > 1e-4; zc = np.clip(z, 1e-4, None)
    u = K[0, 0] * P[:, 0] / zc + K[0, 2]
    v = K[1, 1] * P[:, 1] / zc + K[1, 2]
    return u, v, z, infront


def _sample_rgb(img, u, v):
    H, W = img.shape[:2]
    ui = np.clip(np.round(u).astype(int), 0, W - 1)
    vi = np.clip(np.round(v).astype(int), 0, H - 1)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return img[vi, ui], inb, ui, vi


def complete_wall(verts, faces, room_center, K_da, base_rgb, base_flat,
                  orbit, fill_rgb=(180, 170, 160), occ_tol=0.08, ppm=70):
    """Composite one wall's texture. `orbit` = list of dicts with keys
    R,t,K,img,depth (VGGT frame, view i>0) already mapped so a DA-frame point ->
    view i via P_view = R @ P_da_vggt + t, where P_da_vggt is the DA point in
    VGGT scale/orientation. Priority base-flat > orbit(occlusion-ok) > procedural."""
    v0, v1, v2 = verts[0], verts[1], verts[2]
    Ls = np.linalg.norm(v2 - v0); Lt = np.linalg.norm(v1 - v0)
    ws = max(8, int(Ls * ppm)); ht = max(8, int(Lt * ppm))
    s = (np.arange(ws) + 0.5) / ws; t = (np.arange(ht) + 0.5) / ht
    ss, tt = np.meshgrid(s, t)
    P = (v0[None, None] + ss[..., None] * (v2 - v0)[None, None]
         + tt[..., None] * (v1 - v0)[None, None]).reshape(-1, 3)   # DA metric frame

    out = np.zeros((ht * ws, 3), np.uint8)
    src = np.zeros(ht * ws, np.uint8)     # 0 hole,1 base,2 orbit
    need = np.ones(ht * ws, bool)

    # 1) base photo where it shows genuine flat wall (source cam = DA frame, K_da)
    u, v, z, inf = _proj(P, K_da)
    rgb, inb, ui, vi = _sample_rgb(base_rgb, u, v)
    flat = base_flat[np.clip(vi, 0, base_flat.shape[0] - 1),
                     np.clip(ui, 0, base_flat.shape[1] - 1)]
    take = need & inf & inb & flat
    out[take] = rgb[take]; src[take] = 1; need[take] = False

    # 2) orbit frames, most head-on first, with VGGT-depth occlusion gate
    n = _face_normal(verts, faces[0])
    if np.dot(n, verts.mean(0) - room_center) < 0:
        n = -n
    order = sorted(range(len(orbit)),
                   key=lambda i: -float(np.dot(orbit[i]["fwd"], n)))
    for i in order:
        if not need.any():
            break
        o = orbit[i]
        if np.dot(o["fwd"], n) <= 0.05:      # not facing the wall
            continue
        Pv = P / o["scale"]                  # DA metric -> VGGT scale
        Pw = o["R0T"] @ (Pv - o["t0"]).T     # view0-cam -> VGGT world
        cam = (o["R"] @ Pw).T + o["t"]       # -> view i camera
        u, v, z, inf = _proj(cam, o["K"])
        rgb, inb, ui, vi = _sample_rgb(o["img"], u, v)
        dv = o["depth"][np.clip(vi, 0, o["depth"].shape[0] - 1),
                        np.clip(ui, 0, o["depth"].shape[1] - 1)]
        visible = z <= dv * (1.0 + occ_tol)  # nothing closer occludes this texel
        take = need & inf & inb & visible
        out[take] = rgb[take]; src[take] = 2; need[take] = False

    N = ht * ws
    stats = dict(base=int((src == 1).sum()), orbit=int((src == 2).sum()),
                 hole=int((need).sum()), n=N)
    # Fill genuinely-unseen texels with the sampled flat-wall colour (+ faint
    # grain) so the wall reads as a complete painted wall with the real revealed
    # features (pictures/windows/door) composited on top.
    grain = _RNG.normal(0, 4, (need.sum(), 3))
    out[need] = np.clip(np.array(fill_rgb) + grain, 0, 255).astype(np.uint8)
    out = out.reshape(ht, ws, 3)
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="base room photo (also VGGT view 0)")
    ap.add_argument("--orbit", required=True, help="dir of clean orbit frames")
    ap.add_argument("--name", default="hybrid")
    ap.add_argument("--viewer-dir",
                    default=os.path.join(os.path.dirname(__file__), "godot_viewer"))
    ap.add_argument("--checkpoint",
                    default=os.path.join(os.path.dirname(__file__), "checkpoints",
                                         "depth_anything_v2_metric_hypersim_vitl.pth"))
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--ppm", type=int, default=70)
    ap.add_argument("--occ-tol", type=float, default=0.08)
    args = ap.parse_args()
    import trimesh

    print("[1/4] DepthAnything room box + flat-wall mask...")
    scene, ex = reconstruct_room_scene(args.image, args.checkpoint, hfov=args.hfov,
                                       subdivisions=1, return_extras=True)
    fx, fy, cx, cy = ex["intrinsics"]
    K_da = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    rgb = ex["rgb"]
    base_flat = flat_wall_masks(ex["depth"], ex["planes"], fx, fy, cx, cy,
                                image_path=args.image, semantics=True)["flat"]
    from wall_mask import sample_wall_color
    wall_rgb = sample_wall_color(rgb, base_flat) or (180, 170, 160)
    print(f"    sampled flat-wall fill colour RGB{tuple(wall_rgb)}")
    da_depth_med = float(np.median(ex["depth"]))

    print("[2/4] VGGT on [source + orbit] for orbit poses...")
    orbit_files = sorted(glob.glob(os.path.join(args.orbit, "*.png")))
    res = mvr.run_vggt([args.image] + orbit_files)   # view 0 = source
    R0 = res["extrinsics"][0][:3, :3]; t0 = res["extrinsics"][0][:3, 3]
    vggt_v0_depth_med = float(np.median(res["depths"][0]))
    scale = da_depth_med / max(1e-6, vggt_v0_depth_med)   # VGGT->metric
    print(f"    scale VGGT->metric = {scale:.3f} "
          f"(DA depth {da_depth_med:.2f} / VGGT v0 {vggt_v0_depth_med:.2f})")

    orbit = []
    for i in range(1, len(res["names"])):
        e = res["extrinsics"][i]
        orbit.append(dict(R=e[:3, :3], t=e[:3, 3], K=res["intrinsics"][i],
                          img=res["images"][i], depth=res["depths"][i],
                          fwd=(e[:3, :3].T @ np.array([0.0, 0, 1])),
                          R0T=R0.T, t0=t0, scale=scale))

    print("[3/4] completing wall textures (base + occlusion-gated orbit)...")
    params = extract_room_params(ex["planes"], ex["points"])
    quads = build_room_quads(params, subdivisions=1)
    rc = np.array([(params["left_x"] + params["right_x"]) / 2,
                   (params["floor_y"] + params["ceiling_y"]) / 2,
                   (params["front_z"] + params["back_z"]) / 2])
    W = params["right_x"] - params["left_x"]; H = params["floor_y"] - params["ceiling_y"]
    D = params["back_z"] - params["front_z"]
    proc = {"floor": generate_herringbone(int(W * PPM), int(D * PPM), (120, 90, 60)),
            "ceiling": generate_wall_texture(int(W * PPM), int(D * PPM), (200, 195, 190))}

    out_scene = trimesh.Scene()
    grand = dict(base=0, orbit=0, hole=0, n=0)
    for label, verts, faces, uvs in quads:
        if label.startswith("wall"):
            tex, st = complete_wall(verts, faces, rc, K_da, rgb, base_flat, orbit,
                                    fill_rgb=wall_rgb, occ_tol=args.occ_tol,
                                    ppm=args.ppm)
            for k in grand:
                grand[k] += st[k]
            n = st["n"]
            print(f"  {label}: base {100*st['base']/n:4.1f}%  "
                  f"orbit {100*st['orbit']/n:4.1f}%  hole {100*st['hole']/n:4.1f}%")
            teximg = Image.fromarray(tex)
        else:
            teximg = proc[label]
        vis = trimesh.visual.TextureVisuals(uv=uvs, image=teximg)
        out_scene.add_geometry(trimesh.Trimesh(vertices=verts, faces=faces,
                               visual=vis, process=False), geom_name=label)

    n = grand["n"] or 1
    print(f"  ALL WALLS: base {100*grand['base']/n:.1f}%  "
          f"orbit {100*grand['orbit']/n:.1f}%  hole {100*grand['hole']/n:.1f}%")

    print("[4/4] export...")
    out_scene.apply_transform(np.diag([1.0, -1.0, -1.0, 1.0]))   # OpenCV->Godot
    out_scene.apply_transform(_center_floor_matrix(out_scene))
    proj = os.path.join(args.viewer_dir, args.name); os.makedirs(proj, exist_ok=True)
    glb = os.path.join(proj, f"{args.name}_room.glb")
    out_scene.export(glb)
    tscn = ("[gd_scene load_steps=2 format=3]\n\n"
            f'[ext_resource type="PackedScene" path="res://{args.name}/{args.name}_room.glb" id="1_glb"]\n\n'
            f'[node name="{args.name}" type="Node3D"]\n\n'
            f'[node name="{args.name}" parent="." instance=ExtResource("1_glb")]\n')
    open(os.path.join(args.viewer_dir, f"{args.name}.tscn"), "w", newline="\n").write(tscn)
    print(f"wrote {glb} ({os.path.getsize(glb)/1e6:.1f} MB). F5 godot_viewer/.")


if __name__ == "__main__":
    main()
