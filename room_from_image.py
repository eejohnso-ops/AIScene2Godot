#!/usr/bin/env python3
"""
room_from_image.py -- estimate room layout (walls, floor, ceiling) from a
perspective interior photo and export as a textured GLB for the Godot viewer.

Uses DepthAnything V2 (metric indoor) for depth estimation, then RANSAC plane
fitting to detect architectural surfaces. Works alongside to_godot.py: that
script handles furniture/objects from MIDI; this one handles the room shell.

Usage:
    python room_from_image.py photo.jpg --name room_shell
    python room_from_image.py render.png --name room --scale 3 --hfov 50

Deps:  pip install torch numpy open3d trimesh scipy Pillow
       # DepthAnything V2: see https://github.com/DepthAnything/Depth-Anything-V2
       #   pip install depth-anything-v2   (or clone + pip install -e .)
       # Download metric indoor checkpoint:
       #   depth_anything_v2_metric_indoor_vitl.pth (~330MB)
       #   from the repo's HuggingFace model page
"""
import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
from PIL import Image


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DetectedPlane:
    normal: np.ndarray
    offset: float
    inlier_indices: np.ndarray
    label: str = ""


# ============================================================================
# STAGE 1: DEPTH ESTIMATION
# ============================================================================

DEPTH_ANYTHING_REPO = os.environ.get(
    "DEPTH_ANYTHING_V2",
    os.path.join(os.path.dirname(__file__), "..", "..", "Depth-Anything-V2"),
)


def estimate_depth(image_path: str, checkpoint: str) -> tuple[np.ndarray, np.ndarray]:
    """Run DepthAnything V2 metric indoor model.

    Returns (depth HxW float32 in metres, rgb HxWx3 uint8).
    """
    import torch

    metric_path = os.path.normpath(os.path.join(DEPTH_ANYTHING_REPO, "metric_depth"))
    if metric_path not in sys.path:
        sys.path.insert(0, metric_path)

    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError:
        sys.exit(
            "DepthAnything V2 not found. Clone the repo and set DEPTH_ANYTHING_V2:\n"
            "  git clone https://github.com/DepthAnything/Depth-Anything-V2\n"
            "  set DEPTH_ANYTHING_V2=C:\\path\\to\\Depth-Anything-V2\n"
            "Then download the metric indoor ViT-L checkpoint."
        )

    rgb = np.array(Image.open(image_path).convert("RGB"))

    model = DepthAnythingV2(
        encoder="vitl", features=256,
        out_channels=[256, 512, 1024, 1024],
        max_depth=20,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to("cuda").eval()

    with torch.no_grad():
        depth = model.infer_image(rgb)

    return depth.astype(np.float32), rgb


# ============================================================================
# STAGE 2: DEPTH -> POINT CLOUD
# ============================================================================

def depth_to_pointcloud(
    depth: np.ndarray, rgb: np.ndarray, hfov: float = 60.0
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Unproject depth to 3D using a pinhole camera model.

    Camera frame (OpenCV): +x right, +y down, +z forward.
    Returns (points Nx3, colors Nx3 float 0-1, fx, fy, cx, cy).
    """
    H, W = depth.shape
    fx = fy = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx, cy = W / 2.0, H / 2.0

    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0

    valid = (depth.ravel() > 0.1) & (depth.ravel() < 19.0)
    return points[valid], colors[valid], fx, fy, cx, cy


# ============================================================================
# STAGE 3: RANSAC PLANE DETECTION
# ============================================================================

def detect_planes(
    points: np.ndarray,
    min_points: int = 5000,
    threshold: float = 0.02,
    max_planes: int = 8,
) -> list[DetectedPlane]:
    """Sequential RANSAC: fit plane, remove inliers, repeat."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    planes = []
    remaining_idx = np.arange(len(points))

    for _ in range(max_planes):
        if len(remaining_idx) < min_points:
            break

        sub_pcd = pcd.select_by_index(remaining_idx.tolist())
        plane_model, inliers = sub_pcd.segment_plane(
            distance_threshold=threshold, ransac_n=3, num_iterations=1000
        )

        if len(inliers) < min_points:
            break

        a, b, c, d = plane_model
        normal = np.array([a, b, c], dtype=np.float64)
        norm_len = np.linalg.norm(normal)
        normal /= norm_len
        offset = -d / norm_len

        global_inliers = remaining_idx[np.array(inliers)]
        planes.append(DetectedPlane(
            normal=normal, offset=offset, inlier_indices=global_inliers
        ))
        remaining_idx = np.delete(remaining_idx, inliers)

    return planes


def classify_planes(planes: list[DetectedPlane], points: np.ndarray) -> list[DetectedPlane]:
    """Assign labels: floor, ceiling, wall_N based on normal direction and position."""
    all_y = points[:, 1]
    median_y = np.median(all_y)

    floor_candidates = []
    ceiling_candidates = []
    walls = []

    for plane in planes:
        ny = abs(plane.normal[1])
        if ny > 0.7:
            mean_y = points[plane.inlier_indices, 1].mean()
            if mean_y > median_y:
                floor_candidates.append(plane)
            else:
                ceiling_candidates.append(plane)
        else:
            walls.append(plane)

    if floor_candidates:
        best = max(floor_candidates, key=lambda p: len(p.inlier_indices))
        best.label = "floor"
        for p in floor_candidates:
            if p is not best:
                p.label = "floor_extra"
    if ceiling_candidates:
        best = max(ceiling_candidates, key=lambda p: len(p.inlier_indices))
        best.label = "ceiling"
        for p in ceiling_candidates:
            if p is not best:
                p.label = "ceiling_extra"

    for i, w in enumerate(walls):
        w.label = f"wall_{i}"

    return planes


def box_fallback(points: np.ndarray) -> list[DetectedPlane]:
    """Create a simple box room from the point cloud's AABB."""
    mn = np.percentile(points, 5, axis=0)
    mx = np.percentile(points, 95, axis=0)

    planes = []

    # Floor (normal pointing -y in camera frame = "up")
    floor_mask = points[:, 1] > (mx[1] - (mx[1] - mn[1]) * 0.15)
    if floor_mask.sum() > 100:
        planes.append(DetectedPlane(
            normal=np.array([0, -1, 0], dtype=np.float64),
            offset=-mx[1],
            inlier_indices=np.where(floor_mask)[0],
            label="floor"
        ))

    # Ceiling (normal pointing +y)
    ceil_mask = points[:, 1] < (mn[1] + (mx[1] - mn[1]) * 0.15)
    if ceil_mask.sum() > 100:
        planes.append(DetectedPlane(
            normal=np.array([0, 1, 0], dtype=np.float64),
            offset=mn[1],
            inlier_indices=np.where(ceil_mask)[0],
            label="ceiling"
        ))

    # Back wall (+z direction)
    back_mask = points[:, 2] > (mx[2] - (mx[2] - mn[2]) * 0.15)
    if back_mask.sum() > 100:
        planes.append(DetectedPlane(
            normal=np.array([0, 0, -1], dtype=np.float64),
            offset=-mx[2],
            inlier_indices=np.where(back_mask)[0],
            label="wall_0"
        ))

    # Left wall (-x direction)
    left_mask = points[:, 0] < (mn[0] + (mx[0] - mn[0]) * 0.15)
    if left_mask.sum() > 100:
        planes.append(DetectedPlane(
            normal=np.array([1, 0, 0], dtype=np.float64),
            offset=mn[0],
            inlier_indices=np.where(left_mask)[0],
            label="wall_1"
        ))

    # Right wall (+x direction)
    right_mask = points[:, 0] > (mx[0] - (mx[0] - mn[0]) * 0.15)
    if right_mask.sum() > 100:
        planes.append(DetectedPlane(
            normal=np.array([-1, 0, 0], dtype=np.float64),
            offset=-mx[0],
            inlier_indices=np.where(right_mask)[0],
            label="wall_2"
        ))

    # Front wall (behind camera, -z direction) -- synthetic, no inliers
    planes.append(DetectedPlane(
        normal=np.array([0, 0, 1], dtype=np.float64),
        offset=0.3,
        inlier_indices=np.array([], dtype=int),
        label="wall_3"
    ))

    return planes


# ============================================================================
# STAGE 4: MESH CONSTRUCTION
# ============================================================================

def _make_quad(corner, edge1, edge2):
    """Create a quad (2 triangles) from a corner and two edge vectors, facing the origin."""
    verts = np.array([corner,
                      corner + edge1,
                      corner + edge1 + edge2,
                      corner + edge2], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    e1 = verts[1] - verts[0]
    e2 = verts[2] - verts[0]
    if np.dot(np.cross(e1, e2), -verts.mean(axis=0)) < 0:
        faces = faces[:, ::-1]
    return verts, faces


def extract_room_params(planes: list[DetectedPlane], points: np.ndarray) -> dict:
    """Extract room dimensions.

    Floor/ceiling Y come from RANSAC planes (reliable — large horizontal
    surfaces).  Wall positions come from the point cloud's percentile bounds
    (more reliable than RANSAC wall detection, which is thrown off by
    furniture occlusion).
    """
    mn = np.percentile(points, 2, axis=0)
    mx = np.percentile(points, 98, axis=0)

    floor_y = None
    ceiling_y = None

    for p in planes:
        pts = points[p.inlier_indices]
        if p.label == "floor":
            floor_y = float(pts[:, 1].mean())
        elif p.label == "ceiling":
            ceiling_y = float(pts[:, 1].mean())

    if floor_y is None:
        floor_y = float(mx[1])
    if ceiling_y is None:
        ceiling_y = float(mn[1])

    left_x = float(mn[0])
    right_x = float(mx[0])
    front_z = max(float(mn[2]), 0.3)
    back_z = float(mx[2])

    params = dict(floor_y=floor_y, ceiling_y=ceiling_y,
                  left_x=left_x, right_x=right_x,
                  front_z=front_z, back_z=back_z)
    print(f"  room box: width={right_x - left_x:.2f}m  "
          f"height={floor_y - ceiling_y:.2f}m  "
          f"depth={back_z - front_z:.2f}m")
    print(f"  floor_y={floor_y:.2f}  ceiling_y={ceiling_y:.2f}  "
          f"X=[{left_x:.2f}, {right_x:.2f}]  Z=[{front_z:.2f}, {back_z:.2f}]")
    return params


def build_room_quads(params: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Build 5 quads (floor, ceiling, 3 walls) from room measurements.

    The front wall (behind the camera) is omitted — it's never visible in
    the source image and would just be a solid-color blocker.
    """
    fy = params["floor_y"]
    cy = params["ceiling_y"]
    lx = params["left_x"]
    rx = params["right_x"]
    fz = params["front_z"]
    bz = params["back_z"]

    quads = []
    quads.append(("floor", *_make_quad(
        np.array([lx, fy, fz]),
        np.array([rx - lx, 0, 0]),
        np.array([0, 0, bz - fz]))))
    quads.append(("ceiling", *_make_quad(
        np.array([lx, cy, fz]),
        np.array([rx - lx, 0, 0]),
        np.array([0, 0, bz - fz]))))
    quads.append(("wall_back", *_make_quad(
        np.array([lx, cy, bz]),
        np.array([rx - lx, 0, 0]),
        np.array([0, fy - cy, 0]))))
    quads.append(("wall_left", *_make_quad(
        np.array([lx, cy, fz]),
        np.array([0, 0, bz - fz]),
        np.array([0, fy - cy, 0]))))
    quads.append(("wall_right", *_make_quad(
        np.array([rx, cy, fz]),
        np.array([0, 0, bz - fz]),
        np.array([0, fy - cy, 0]))))
    return quads


# ============================================================================
# STAGE 5: PROCEDURAL TEXTURE GENERATION
# ============================================================================

def generate_wall_texture(w_px: int, h_px: int, color: tuple[int, ...]) -> Image.Image:
    """Matte paint wall texture with subtle surface variation."""
    rng = np.random.RandomState(hash(color) & 0x7FFFFFFF)
    img = np.full((h_px, w_px, 3), color[:3], dtype=np.float32)
    yy, xx = np.mgrid[0:h_px, 0:w_px].astype(np.float32)
    for _ in range(5):
        fx = rng.uniform(0.002, 0.015)
        fy = rng.uniform(0.002, 0.015)
        amp = rng.uniform(1.5, 3.5)
        img += (np.sin(xx * fx + rng.uniform(0, 6.28))
                * np.sin(yy * fy + rng.uniform(0, 6.28))
                * amp)[:, :, np.newaxis]
    img += rng.normal(0, 1.2, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def generate_herringbone(w_px: int, h_px: int,
                         color: tuple[int, ...],
                         pw: int = 48, pl: int = 240) -> Image.Image:
    """Wood herringbone floor texture with grain detail."""
    from PIL import ImageFilter
    rng = np.random.RandomState(42)
    col_a = np.array(color[:3], dtype=np.float32)
    col_b = col_a * 0.88

    yy, xx = np.mgrid[0:h_px, 0:w_px].astype(np.float32)

    band = np.floor(yy / pw).astype(int)
    local_y = yy - band * pw
    is_even = (band % 2 == 0)

    d_fwd = xx - yy
    d_bck = xx + yy
    plank_fwd = np.floor(d_fwd / pl).astype(int)
    plank_bck = np.floor(d_bck / pl).astype(int)
    local_fwd = d_fwd - plank_fwd * pl
    local_bck = d_bck - plank_bck * pl

    plank_id = np.where(is_even, plank_fwd, plank_bck)
    local_along = np.where(is_even, local_fwd, local_bck)

    gap = 0.8
    gap_mask = ((local_y < gap) | (local_y > pw - gap)
                | (local_along < gap) | (local_along > pl - gap))

    seed = ((band * 37 + plank_id * 73) % 10000).astype(np.float32)
    img = np.zeros((h_px, w_px, 3), dtype=np.float32)
    for ch in range(3):
        shift = np.sin(seed * 0.7 + ch * 2.1) * 8 + np.sin(seed * 1.3 + ch) * 5
        img[:, :, ch] = col_a[ch] + shift

    grain_coord = np.where(is_even, d_fwd, d_bck)
    grain_cross = local_y / pw
    grain = (np.sin(grain_coord * 0.25 + seed * 0.5) * 3
             + np.sin(grain_coord * 0.7 + grain_cross * 12 + seed * 0.3) * 2.5
             + np.sin(grain_coord * 1.8 + seed * 1.1) * 1.5)
    for ch in range(3):
        img[:, :, ch] += grain * [0.9, 0.7, 0.5][ch]

    img += rng.normal(0, 1.0, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    gap_dark = np.clip(img.astype(np.float32) * 0.82, 0, 255).astype(np.uint8)
    img[gap_mask] = gap_dark[gap_mask]

    pil = Image.fromarray(img)
    return pil.filter(ImageFilter.GaussianBlur(0.4))


# ============================================================================
# STAGE 6: BUILD SCENE + EXPORT
# ============================================================================

PPM = 128  # pixels per metre for generated textures


def build_room_scene(
    planes: list[DetectedPlane],
    points: np.ndarray,
    surface_colors: dict | None = None,
) -> "trimesh.Scene":
    """Build a room box with procedural textures.

    surface_colors: {surface_name: {"color": [r,g,b], ...}} from segment_room.py.
    Falls back to neutral tones if not provided.
    """
    import trimesh

    params = extract_room_params(planes, points)
    quads = build_room_quads(params)

    W = params["right_x"] - params["left_x"]
    H = params["floor_y"] - params["ceiling_y"]
    D = params["back_z"] - params["front_z"]

    def get_color(name):
        if surface_colors and name in surface_colors:
            return tuple(surface_colors[name]["color"])
        defaults = {"wall": (140, 130, 120), "floor": (120, 90, 60),
                    "ceiling": (200, 195, 190)}
        return defaults.get(name, (160, 155, 150))

    wall_color = get_color("wall")
    floor_color = get_color("floor")
    ceil_color = get_color("ceiling")

    textures = {
        "floor": generate_herringbone(int(W * PPM), int(D * PPM), floor_color),
        "ceiling": None,
        "wall_back": generate_wall_texture(int(W * PPM), int(H * PPM), wall_color),
        "wall_left": generate_wall_texture(int(D * PPM), int(H * PPM), wall_color),
        "wall_right": generate_wall_texture(int(D * PPM), int(H * PPM), wall_color),
    }

    rect_uv = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    scene = trimesh.Scene()

    for label, verts, faces in quads:
        tex = textures.get(label)
        if tex is not None:
            visual = trimesh.visual.TextureVisuals(uv=rect_uv, image=tex)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                                   visual=visual, process=False)
        else:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            mesh.visual.face_colors = list(ceil_color) + [255]
        scene.add_geometry(mesh, geom_name=label)

    return scene


def export_room(scene, name: str, scale: float, viewer_dir: str) -> str:
    """Scale the scene, center at origin, export GLB."""
    import trimesh

    scene.apply_transform(trimesh.transformations.scale_matrix(scale))

    # Camera frame -> Godot frame: flip Y and Z
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    scene.apply_transform(flip)

    # Center horizontally and depth-wise so MIDI objects (at origin) land
    # inside the room. Keep floor at Y=0.
    bounds = scene.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    shift = np.eye(4)
    shift[0, 3] = -center[0]
    shift[1, 3] = -bounds[0][1]  # floor at Y=0
    shift[2, 3] = -center[2]
    scene.apply_transform(shift)

    os.makedirs(viewer_dir, exist_ok=True)
    out_path = os.path.join(viewer_dir, f"{name}.glb")
    scene.export(out_path)

    mb = os.path.getsize(out_path) / 1e6
    print(f"\nwrote {out_path}  ({mb:.1f} MB)")
    return out_path


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Room interior image -> room shell GLB for Godot")
    ap.add_argument("image", help="perspective photo/render of a room interior")
    ap.add_argument("--name", default="room_shell", help="output GLB base name")
    ap.add_argument("--scale", type=float, default=3.0,
                    help="uniform scale (match to_godot.py for same scene). Default 3.")
    ap.add_argument("--viewer-dir",
                    default=os.path.join(os.path.dirname(__file__), "godot_viewer"),
                    help="Godot project dir to drop the result into")
    ap.add_argument("--hfov", type=float, default=60.0,
                    help="horizontal field of view in degrees (default 60)")
    ap.add_argument("--checkpoint",
                    default=os.path.join(os.path.dirname(__file__),
                                         "checkpoints",
                                         "depth_anything_v2_metric_hypersim_vitl.pth"),
                    help="path to DepthAnything V2 metric indoor checkpoint")
    ap.add_argument("--min-plane-points", type=int, default=5000,
                    help="RANSAC: minimum inliers to accept a plane")
    ap.add_argument("--ransac-threshold", type=float, default=0.02,
                    help="RANSAC: inlier distance threshold in metres")
    ap.add_argument("--max-planes", type=int, default=8,
                    help="max planes to detect")
    ap.add_argument("--surfaces", default=None,
                    help="surfaces JSON from segment_room.py (auto-colors)")
    ap.add_argument("--no-texture", action="store_true",
                    help="skip texture projection (solid colors only)")

    args = ap.parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"Image not found: {args.image}")

    # --- Stage 1: depth ---
    print("[1/4] estimating depth...")
    if not os.path.isfile(args.checkpoint):
        sys.exit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Download the DepthAnything V2 metric indoor ViT-L checkpoint:\n"
            "  https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large"
        )
    depth, rgb = estimate_depth(args.image, args.checkpoint)
    print(f"  depth range: {depth.min():.2f} - {depth.max():.2f} m  "
          f"({rgb.shape[1]}x{rgb.shape[0]} px)")

    # --- Stage 2: point cloud ---
    print("[2/4] building point cloud...")
    points, colors, fx, fy, cx, cy = depth_to_pointcloud(depth, rgb, args.hfov)
    print(f"  {len(points)} valid points")

    # --- Stage 3: planes ---
    print("[3/4] detecting planes...")
    planes = detect_planes(points, args.min_plane_points,
                           args.ransac_threshold, args.max_planes)

    if len(planes) < 2:
        print(f"  only {len(planes)} plane(s) found, using box fallback")
        planes = box_fallback(points)
    classify_planes(planes, points)
    print(f"  {len(planes)} planes: "
          + ", ".join(f"{p.label} ({len(p.inlier_indices)} pts)" for p in planes))

    # --- Stage 4-6: mesh + texture + export ---
    surface_colors = None
    if args.surfaces:
        import json
        if not os.path.isfile(args.surfaces):
            sys.exit(f"Surfaces JSON not found: {args.surfaces}")
        with open(args.surfaces) as f:
            surface_colors = json.load(f)
        print(f"[4/4] building room box from segmentation colors...")
        for name in ("wall", "floor", "ceiling"):
            if name in surface_colors:
                c = surface_colors[name]["color"]
                print(f"  {name}: RGB({c[0]},{c[1]},{c[2]})")
    else:
        print("[4/4] building room box (no segmentation, using defaults)...")

    scene = build_room_scene(planes, points, surface_colors=surface_colors)
    out = export_room(scene, args.name, args.scale, args.viewer_dir)
    print("Open the Godot project in godot_viewer/ and press F5 -- it loads "
          "all .glb files automatically.")


if __name__ == "__main__":
    main()
