"""Multi-view room reconstruction -> walkable Godot scene.

Takes N views of one room, registers them with VGGT (mv_register.py), fuses them
into one point cloud, then runs the EXISTING single-view geometry
(detect_planes / extract_room_params / build_room_quads from room_from_image.py)
on the fused cloud and textures each wall from whichever view saw it best. The
result is exported exactly like the single-view path (camera->Godot flip, centred
floor at Y=0, per-project folder + editor .tscn).

Why this breaks the single-photo ceiling: one photo sees ~20% of a wall as clean
surface and foreshortens the side walls; fusing several views fills more of the
room and lets each wall be textured head-on from its own best view.

The fused cloud lands in VGGT's world frame (= view 0's camera), which is NOT
gravity-aligned in general, so we first canonicalise it: rotate so the floor
normal is +y (down, OpenCV) and the dominant wall faces an axis, which is what
classify_planes / extract_room_params assume.

Run (VGGT is in the main venv -- see mv_register.py):
    python build_room_multiview.py --images path/to/views_dir --name myroom
    python build_room_multiview.py --bundle bundle.npz --name myroom
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

import mv_register as mvr
from room_from_image import (
    detect_planes, classify_planes, extract_room_params, build_room_quads,
    generate_herringbone, generate_wall_texture, _project_uvs_view,
    _center_floor_matrix, PPM,
)

TARGET_HEIGHT = 2.6   # metres, floor->ceiling; normalises VGGT's arbitrary scale


# ---------------------------------------------------------------------------
# Canonicalisation: rotate the fused cloud so +y = down and walls are axis-aligned
# ---------------------------------------------------------------------------
def _rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit-ish vector a onto b (Rodrigues)."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        # antiparallel: 180deg about any axis perpendicular to a
        perp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        axis = np.cross(a, perp); axis /= np.linalg.norm(axis)
        K = _skew(axis)
        return np.eye(3) + 2 * K @ K  # Rodrigues at theta=pi
    K = _skew(v)
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def canonical_rotation(planes, *, seed_up=(0.0, -1.0, 0.0)) -> np.ndarray:
    """Rc (world->canonical) so that points_canon = points @ Rc.T has +y = down
    (OpenCV) and the dominant wall normal aligned to the z axis.

    Seeded from `seed_up` = OpenCV up (cameras are held roughly level, and we'll
    control novel-view cameras), then refined to the largest near-horizontal plane
    normal. Falls back to identity for the missing pieces.
    """
    up0 = np.array(seed_up, dtype=float)
    normals = [(p.normal / (np.linalg.norm(p.normal) + 1e-12), len(p.inlier_indices))
               for p in planes]

    # 1) Level: up = largest plane whose normal is ~vertical (|n.up0| high).
    horiz = [(n, c) for n, c in normals if abs(np.dot(n, up0)) > 0.7]
    if horiz:
        n, _ = max(horiz, key=lambda nc: nc[1])
        up = n if np.dot(n, up0) > 0 else -n      # orient toward seed up
    else:
        up = up0
    R1 = _rot_between(up, np.array([0.0, -1.0, 0.0]))  # up -> OpenCV up (-y)

    # 2) Yaw: dominant near-vertical (wall) normal, levelled, aligned to z.
    vert = [(R1 @ n, c) for n, c in normals if abs(np.dot(n, up0)) < 0.3]
    if vert:
        nw, _ = max(vert, key=lambda nc: nc[1])
        ang = np.arctan2(nw[0], nw[2])            # angle of (x,z) from +z
        ca, sa = np.cos(-ang), np.sin(-ang)
        R2 = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    else:
        R2 = np.eye(3)
    return R2 @ R1


def _rotate_camera(cam: "mvr.ViewCamera", Rc: np.ndarray) -> "mvr.ViewCamera":
    """Express a camera in the canonical frame (points_canon = points @ Rc.T)."""
    return mvr.ViewCamera(name=cam.name, K=cam.K, R=cam.R @ Rc.T, t=cam.t,
                          width=cam.width, height=cam.height)


# ---------------------------------------------------------------------------
# Best-view-per-wall texturing
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(0)


def _face_normal(verts: np.ndarray, face: np.ndarray) -> np.ndarray:
    a, b, c = verts[face[0]], verts[face[1]], verts[face[2]]
    n = np.cross(b - a, c - a)
    return n / (np.linalg.norm(n) + 1e-12)


def _sample_quad(verts: np.ndarray, faces: np.ndarray, per_tri: int = 200):
    """Uniform points across a wall's triangles (not just its corners -- a head-on
    camera's FOV is narrower than a near wall, so the corners fall out of frame
    while the wall is still the best-captured one)."""
    out = []
    for f in faces:
        a, b, c = verts[f[0]], verts[f[1]], verts[f[2]]
        u = _RNG.uniform(0, 1, (per_tri, 1)); v = _RNG.uniform(0, 1, (per_tri, 1))
        flip = (u + v) > 1
        u = np.where(flip, 1 - u, u); v = np.where(flip, 1 - v, v)
        out.append(a + u * (b - a) + v * (c - a))
    return np.concatenate(out, axis=0)


def _coverage(samp: np.ndarray, cam: "mvr.ViewCamera") -> float:
    """Fraction of sample points that land in front of the camera AND inside its
    image. Uses the UNCLAMPED projection (the texturing projector clamps to the
    border, which would hide out-of-frame points)."""
    cp = samp @ cam.R.T + cam.t
    z = cp[:, 2]
    infront = z > 1e-3
    zc = np.clip(z, 1e-3, None)
    u = cam.K[0, 0] * cp[:, 0] / zc + cam.K[0, 2]
    v = cam.K[1, 1] * cp[:, 1] / zc + cam.K[1, 2]
    inside = infront & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    return float(inside.mean())


def _best_view(verts, faces, room_center, cams):
    """Pick the view that sees this wall best. Returns (cam_index, uvs, outward_n)
    or (None, None, outward_n). Score = in-frame coverage * frontality."""
    n = _face_normal(verts, faces[0])
    if np.dot(n, verts.mean(0) - room_center) < 0:   # orient outward
        n = -n
    samp = _sample_quad(verts, faces)
    best_i, best_uv, best_score = None, None, 0.0
    for i, cam in enumerate(cams):
        front = float(np.dot(cam.forward, n))        # cam looks toward the wall
        if front <= 0.1:
            continue
        score = _coverage(samp, cam) * front
        if score > best_score:
            best_i, best_score = i, score
    if best_i is not None:
        cam = cams[best_i]
        best_uv = _project_uvs_view(verts, cam.K, cam.R, cam.t,
                                    cam.width, cam.height)
    return best_i, best_uv, n


# ---------------------------------------------------------------------------
# Build + export
# ---------------------------------------------------------------------------
def build_multiview_scene(result: dict, *, conf_percentile: float = 40.0,
                          ransac_threshold: float = 0.02,
                          min_plane_points: int = 3000, max_planes: int = 8):
    """Fuse + canonicalise + reuse geometry + best-view texture. Returns a
    trimesh.Scene in the canonical camera frame (caller flips to Godot)."""
    import trimesh

    pts, cols = mvr.fuse_world_points(result, conf_percentile)
    cams = mvr.cameras(result)

    planes0 = detect_planes(pts, min_plane_points, ransac_threshold, max_planes)
    Rc = canonical_rotation(planes0) if planes0 else np.eye(3)
    pts_c = pts @ Rc.T
    cams_c = [_rotate_camera(c, Rc) for c in cams]

    planes = detect_planes(pts_c, min_plane_points, ransac_threshold, max_planes)
    if len(planes) < 2:
        from room_from_image import box_fallback
        print(f"    only {len(planes)} plane(s); using box fallback")
        planes = box_fallback(pts_c)
    classify_planes(planes, pts_c)
    print(f"    {len(planes)} planes: "
          + ", ".join(f"{p.label}({len(p.inlier_indices)})" for p in planes))

    params = extract_room_params(planes, pts_c)
    quads = build_room_quads(params, subdivisions=1)
    room_center = np.array([(params["left_x"] + params["right_x"]) / 2,
                            (params["floor_y"] + params["ceiling_y"]) / 2,
                            (params["front_z"] + params["back_z"]) / 2])

    W = params["right_x"] - params["left_x"]
    H = params["floor_y"] - params["ceiling_y"]
    D = params["back_z"] - params["front_z"]
    wall_rgb = tuple(int(v) for v in (cols.mean(0) * 255)) if len(cols) else (140, 130, 120)
    proc = {
        "floor": generate_herringbone(int(W * PPM), int(D * PPM), (120, 90, 60)),
        "ceiling": generate_wall_texture(int(W * PPM), int(D * PPM), (200, 195, 190)),
    }
    view_imgs = [Image.fromarray(result["images"][i]) for i in range(len(cams_c))]

    scene = trimesh.Scene()
    for label, verts, faces, uvs in quads:
        if label.startswith("wall"):
            ci, vuv, _ = _best_view(verts, faces, room_center, cams_c)
            if ci is not None:
                uvs, tex = vuv, view_imgs[ci]
                print(f"    {label}: textured from view {ci} "
                      f"({result['names'][ci]})")
            else:
                tex = generate_wall_texture(int(D * PPM), int(H * PPM), wall_rgb)
                print(f"    {label}: no head-on view; procedural fallback")
        else:
            tex = proc[label]
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False,
                               visual=trimesh.visual.TextureVisuals(uv=uvs, image=tex))
        scene.add_geometry(mesh, geom_name=label)

    scene.metadata["room_height"] = float(H)
    return scene


def _write_editor_tscn(viewer_dir: str, name: str, glb_file: str) -> None:
    tscn = (
        "[gd_scene load_steps=2 format=3]\n\n"
        f'[ext_resource type="PackedScene" path="res://{name}/{glb_file}" id="1_glb"]\n\n'
        f'[node name="{name}" type="Node3D"]\n\n'
        f'[node name="{name}" parent="." instance=ExtResource("1_glb")]\n')
    with open(os.path.join(viewer_dir, f"{name}.tscn"), "w", newline="\n") as f:
        f.write(tscn)


def export_multiview(scene, name: str, viewer_dir: str,
                     target_height: float = TARGET_HEIGHT,
                     scale: float | None = None) -> str:
    """Normalise scale, flip camera->Godot, centre floor at Y=0, write
    godot_viewer/<name>/<name>_room.glb + an editor <name>.tscn."""
    import trimesh
    if scale is None:
        h = scene.metadata.get("room_height", 0) or 1.0
        scale = target_height / h
    scene.apply_transform(trimesh.transformations.scale_matrix(scale))
    scene.apply_transform(np.diag([1.0, -1.0, -1.0, 1.0]))  # OpenCV -> Godot
    scene.apply_transform(_center_floor_matrix(scene))      # floor at Y=0

    project_dir = os.path.join(viewer_dir, name)
    os.makedirs(project_dir, exist_ok=True)
    glb_file = f"{name}_room.glb"     # "room" so the viewer treats it as a shell
    out_path = os.path.join(project_dir, glb_file)
    scene.export(out_path)
    _write_editor_tscn(viewer_dir, name, glb_file)
    print(f"\nwrote {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB, "
          f"scale x{scale:.2f})")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Multi-view room reconstruction -> walkable Godot scene")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", nargs="+",
                     help="a directory, glob(s), or explicit view image files")
    src.add_argument("--bundle", help="a pre-computed mv_register .npz bundle")
    ap.add_argument("--name", default="mvroom", help="output project name")
    ap.add_argument("--viewer-dir",
                    default=os.path.join(os.path.dirname(__file__), "godot_viewer"))
    ap.add_argument("--mode", default="crop", choices=["crop", "pad"],
                    help="VGGT preprocessing (pad keeps full FOV)")
    ap.add_argument("--save-bundle", default=None,
                    help="also write the VGGT bundle here (.npz) for reuse")
    ap.add_argument("--conf-percentile", type=float, default=40.0)
    ap.add_argument("--ransac-threshold", type=float, default=0.02)
    ap.add_argument("--min-plane-points", type=int, default=3000)
    ap.add_argument("--target-height", type=float, default=TARGET_HEIGHT)
    ap.add_argument("--scale", type=float, default=None,
                    help="override scale (default: normalise floor->ceiling to "
                         "--target-height)")
    args = ap.parse_args()

    if args.bundle:
        print(f"loading bundle {args.bundle}")
        result = mvr.load_bundle(args.bundle)
    else:
        paths = mvr._select_images(args.images)
        if not paths:
            raise SystemExit(f"no images matched: {args.images}")
        print(f"registering {len(paths)} views with VGGT...")
        result = mvr.run_vggt(paths, mode=args.mode)
        if args.save_bundle:
            mvr.save_bundle(result, args.save_bundle)

    scene = build_multiview_scene(
        result, conf_percentile=args.conf_percentile,
        ransac_threshold=args.ransac_threshold,
        min_plane_points=args.min_plane_points)
    export_multiview(scene, args.name, args.viewer_dir,
                     target_height=args.target_height, scale=args.scale)
    print("Open godot_viewer/ in Godot 4 and press F5 (loads the newest scene).")


if __name__ == "__main__":
    main()
