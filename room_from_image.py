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

def _find_depth_anything() -> str:
    """Locate the Depth-Anything-V2 repo, preferring the consolidated external/."""
    env = os.environ.get("DEPTH_ANYTHING_V2")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "external", "Depth-Anything-V2"),  # consolidated
        os.path.join(here, "..", "..", "Depth-Anything-V2"),        # legacy sibling
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return os.path.normpath(cand)
    return os.path.normpath(candidates[0])


DEPTH_ANYTHING_REPO = _find_depth_anything()


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


def _make_subdivided_quad(corner, edge1, edge2, subdivisions):
    """Create a subdivided quad mesh with UVs, facing the origin."""
    n = subdivisions + 1
    verts = np.zeros((n * n, 3), dtype=np.float32)
    uvs = np.zeros((n * n, 2), dtype=np.float32)
    for j in range(n):
        v = j / subdivisions
        for i in range(n):
            u = i / subdivisions
            verts[j * n + i] = corner + edge1 * u + edge2 * v
            uvs[j * n + i] = [u, v]
    faces = []
    for j in range(subdivisions):
        for i in range(subdivisions):
            idx = j * n + i
            faces.append([idx, idx + 1, idx + n + 1])
            faces.append([idx, idx + n + 1, idx + n])
    faces = np.array(faces, dtype=np.int32)
    e1 = verts[1] - verts[0]
    e2 = verts[n] - verts[0]
    if np.dot(np.cross(e1, e2), -verts.mean(axis=0)) < 0:
        faces = faces[:, ::-1]
    return verts, faces, uvs


def _displace_from_depth(verts, depth_map, fx, fy, cx, cy,
                          max_displacement=0.15):
    """Displace subdivided vertices toward depth-map-implied 3D positions.

    For each vertex, projects to image space, samples the depth map with
    bilinear interpolation, reconstructs the implied 3D point, and moves
    the vertex toward it (clamped to max_displacement metres).

    Returns (displaced_verts, count, mean_displacement, max_displacement_actual).
    """
    H, W = depth_map.shape
    displaced = verts.copy()
    z = verts[:, 2]

    safe_z = np.where(z > 0.01, z, 1.0)
    px = fx * verts[:, 0] / safe_z + cx
    py = fy * verts[:, 1] / safe_z + cy

    ok = (z > 0.01) & (px >= 0) & (px < W - 1) & (py >= 0) & (py < H - 1)
    ii = np.where(ok)[0]
    if len(ii) == 0:
        return displaced, 0, 0.0, 0.0

    pxv, pyv = px[ii], py[ii]
    x0 = np.floor(pxv).astype(int)
    y0 = np.floor(pyv).astype(int)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    xf, yf = pxv - x0, pyv - y0
    sz = (depth_map[y0, x0] * (1 - xf) * (1 - yf) +
          depth_map[y0, x1] * xf * (1 - yf) +
          depth_map[y1, x0] * (1 - xf) * yf +
          depth_map[y1, x1] * xf * yf)

    good = (sz > 0) & np.isfinite(sz)
    ii, sz, pxv, pyv = ii[good], sz[good], pxv[good], pyv[good]
    if len(ii) == 0:
        return displaced, 0, 0.0, 0.0

    actual = np.column_stack([(pxv - cx) / fx * sz,
                              (pyv - cy) / fy * sz,
                              sz]).astype(np.float32)
    delta = actual - verts[ii]
    dist = np.linalg.norm(delta, axis=1)
    scale = np.minimum(max_displacement / np.maximum(dist, 1e-9), 1.0)
    delta *= scale[:, np.newaxis]
    displaced[ii] = verts[ii] + delta

    final_dist = np.linalg.norm(delta, axis=1)
    return displaced, len(ii), float(final_dist.mean()), float(final_dist.max())


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


def build_room_quads(params: dict, subdivisions: int = 1) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Build 5 quads (floor, ceiling, 3 walls) from room measurements.

    The front wall (behind the camera) is omitted — it's never visible in
    the source image and would just be a solid-color blocker.
    Returns [(label, verts, faces, uvs), ...].
    """
    fy = params["floor_y"]
    cy = params["ceiling_y"]
    lx = params["left_x"]
    rx = params["right_x"]
    fz = params["front_z"]
    bz = params["back_z"]

    def _quad(corner, edge1, edge2):
        if subdivisions > 1:
            return _make_subdivided_quad(corner, edge1, edge2, subdivisions)
        v, f = _make_quad(corner, edge1, edge2)
        return v, f, np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])

    quads = []
    quads.append(("floor", *_quad(
        np.array([lx, fy, fz]),
        np.array([rx - lx, 0, 0]),
        np.array([0, 0, bz - fz]))))
    quads.append(("ceiling", *_quad(
        np.array([lx, cy, fz]),
        np.array([rx - lx, 0, 0]),
        np.array([0, 0, bz - fz]))))
    quads.append(("wall_back", *_quad(
        np.array([lx, cy, bz]),
        np.array([rx - lx, 0, 0]),
        np.array([0, fy - cy, 0]))))
    quads.append(("wall_left", *_quad(
        np.array([lx, cy, fz]),
        np.array([0, 0, bz - fz]),
        np.array([0, fy - cy, 0]))))
    quads.append(("wall_right", *_quad(
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
    depth_map: np.ndarray | None = None,
    cam_intrinsics: tuple | None = None,
    subdivisions: int = 1,
    max_displacement: float = 0.15,
) -> "trimesh.Scene":
    """Build a room box with procedural textures and optional depth displacement.

    surface_colors: {surface_name: {"color": [r,g,b], ...}} from segment_room.py.
    depth_map + cam_intrinsics: when provided with subdivisions > 1, vertices are
        displaced toward their depth-map-implied positions for architectural relief.
    cam_intrinsics: (fx, fy, cx, cy) from depth_to_pointcloud.
    """
    import trimesh

    params = extract_room_params(planes, points)
    quads = build_room_quads(params, subdivisions=subdivisions)

    if depth_map is not None and cam_intrinsics is not None and subdivisions > 1:
        fx, fy, cx, cy = cam_intrinsics
        displaced = []
        for label, verts, faces, uvs in quads:
            new_verts, count, mean_d, max_d = _displace_from_depth(
                verts, depth_map, fx, fy, cx, cy, max_displacement)
            if count:
                print(f"  {label}: displaced {count} verts "
                      f"(mean {mean_d * 100:.1f}cm, max {max_d * 100:.1f}cm)")
            displaced.append((label, new_verts, faces, uvs))
        quads = displaced

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
        "ceiling": generate_wall_texture(int(W * PPM), int(D * PPM), ceil_color),
        "wall_back": generate_wall_texture(int(W * PPM), int(H * PPM), wall_color),
        "wall_left": generate_wall_texture(int(D * PPM), int(H * PPM), wall_color),
        "wall_right": generate_wall_texture(int(D * PPM), int(H * PPM), wall_color),
    }

    scene = trimesh.Scene()
    for label, verts, faces, uvs in quads:
        tex = textures.get(label)
        visual = trimesh.visual.TextureVisuals(uv=uvs, image=tex)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                               visual=visual, process=False)
        scene.add_geometry(mesh, geom_name=label)

    return scene


@dataclass
class Placement:
    """Where a room sits in the shared dwelling frame (Godot coords, floor Y=0).

    pos:           world (x, z) of the room centre, in metres.
    yaw:           rotation about the Y axis, in degrees.
    target_height: if set (and target_size is None), uniformly rescale the room so
                   its floor->ceiling height equals this value. Floor-to-ceiling is
                   the most reliably reconstructed dimension, so matching it to a
                   shared constant keeps rooms at a consistent scale even when
                   absolute metric depth drifts between photos.
    target_size:   if set as (width, depth), *non-uniformly* rescale the room's
                   local X/Z to exactly these extents (and Y to target_height).
                   This "conforms" a depth-reconstructed room to its spec footprint
                   so it tiles with its neighbours -- trading a little relief
                   distortion (displacement is clamped, so it stays subtle) for a
                   room that fits the floor plan. Applied before yaw, so the spec's
                   (width, depth) map to local X/Z regardless of orientation.
    """
    pos: tuple[float, float] = (0.0, 0.0)
    yaw: float = 0.0
    target_height: float | None = None
    target_size: tuple[float, float] | None = None


def placement_matrix(pos: tuple[float, float] = (0.0, 0.0),
                     yaw: float = 0.0) -> np.ndarray:
    """4x4 that yaws about Y then translates to world (x, z). Floor stays at Y=0."""
    import trimesh
    R = trimesh.transformations.rotation_matrix(np.radians(yaw), [0, 1, 0])
    T = np.eye(4)
    T[0, 3] = pos[0]
    T[2, 3] = pos[1]
    return T @ R


def _center_floor_matrix(scene) -> np.ndarray:
    """Centre the scene on the XZ origin with its floor at Y=0."""
    bounds = scene.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    shift = np.eye(4)
    shift[0, 3] = -center[0]
    shift[1, 3] = -bounds[0][1]  # floor at Y=0
    shift[2, 3] = -center[2]
    return shift


def place_room(scene, placement: Placement) -> None:
    """Position an already-Godot-framed room scene per `placement`, in place.

    Order: normalise height -> centre on XZ origin with floor at Y=0 -> yaw +
    translate to the room's slot. Shared by the single-room exporter and the
    multi-room dwelling builder so both compose into the same frame.
    """
    import trimesh
    if placement.target_size is not None:
        ext = scene.bounds[1] - scene.bounds[0]
        sx = placement.target_size[0] / ext[0] if ext[0] > 1e-6 else 1.0
        sz = placement.target_size[1] / ext[2] if ext[2] > 1e-6 else 1.0
        if placement.target_height and ext[1] > 1e-6:
            sy = placement.target_height / ext[1]
        else:
            sy = (sx + sz) / 2.0
        scene.apply_transform(np.diag([sx, sy, sz, 1.0]))
    elif placement.target_height:
        h = scene.bounds[1][1] - scene.bounds[0][1]
        if h > 1e-6:
            scene.apply_transform(
                trimesh.transformations.scale_matrix(placement.target_height / h))
    scene.apply_transform(_center_floor_matrix(scene))
    scene.apply_transform(placement_matrix(placement.pos, placement.yaw))


def export_room(scene, name: str, scale: float, viewer_dir: str,
                placement: "Placement | None" = None) -> str:
    """Scale the scene into Godot's frame and export a GLB.

    With no `placement`, the room is centred at the origin (floor at Y=0) so
    MIDI objects at the origin land inside it — the original single-room
    behaviour. With a `placement`, the room is positioned into its slot in a
    shared dwelling frame instead (see `build_dwelling.py`).
    """
    import trimesh

    scene.apply_transform(trimesh.transformations.scale_matrix(scale))

    # Camera frame -> Godot frame: flip Y and Z
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    scene.apply_transform(flip)

    if placement is None:
        scene.apply_transform(_center_floor_matrix(scene))
    else:
        place_room(scene, placement)

    os.makedirs(viewer_dir, exist_ok=True)
    out_path = os.path.join(viewer_dir, f"{name}.glb")
    scene.export(out_path)

    mb = os.path.getsize(out_path) / 1e6
    print(f"\nwrote {out_path}  ({mb:.1f} MB)")
    return out_path


# ============================================================================
# RECONSTRUCTION (reusable: photo -> textured, depth-displaced shell)
# ============================================================================

def reconstruct_room_scene(image: str, checkpoint: str, *,
                           hfov: float = 60.0,
                           surface_colors: dict | None = None,
                           subdivisions: int = 24,
                           max_displacement: float = 0.15,
                           min_plane_points: int = 5000,
                           ransac_threshold: float = 0.02,
                           max_planes: int = 8):
    """Run the full photo -> room-shell reconstruction and return a textured,
    depth-displaced trimesh.Scene in the **camera frame** (before scale/flip/
    placement). Stages mirror the CLI; callers apply scale+flip+placement.

    Used both by this script's CLI (single room) and by build_dwelling.py, which
    drops the result into a floor-plan slot via `place_room()`.
    """
    print("  [1/4] estimating depth...")
    depth, rgb = estimate_depth(image, checkpoint)
    print(f"    depth range: {depth.min():.2f} - {depth.max():.2f} m  "
          f"({rgb.shape[1]}x{rgb.shape[0]} px)")

    print("  [2/4] building point cloud...")
    points, colors, fx, fy, cx, cy = depth_to_pointcloud(depth, rgb, hfov)
    print(f"    {len(points)} valid points")

    print("  [3/4] detecting planes...")
    planes = detect_planes(points, min_plane_points, ransac_threshold, max_planes)
    if len(planes) < 2:
        print(f"    only {len(planes)} plane(s) found, using box fallback")
        planes = box_fallback(points)
    classify_planes(planes, points)
    print(f"    {len(planes)} planes: "
          + ", ".join(f"{p.label} ({len(p.inlier_indices)} pts)" for p in planes))

    print("  [4/4] building textured, depth-displaced shell...")
    return build_room_scene(planes, points, surface_colors=surface_colors,
                            depth_map=depth, cam_intrinsics=(fx, fy, cx, cy),
                            subdivisions=subdivisions,
                            max_displacement=max_displacement)


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
    ap.add_argument("--subdivisions", type=int, default=24,
                    help="wall subdivision grid size for depth displacement (1=flat)")
    ap.add_argument("--max-displacement", type=float, default=0.15,
                    help="max depth displacement in metres (default 0.15)")

    args = ap.parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"Image not found: {args.image}")
    if not os.path.isfile(args.checkpoint):
        sys.exit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Download the DepthAnything V2 metric indoor ViT-L checkpoint:\n"
            "  https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large"
        )

    surface_colors = None
    if args.surfaces:
        import json
        if not os.path.isfile(args.surfaces):
            sys.exit(f"Surfaces JSON not found: {args.surfaces}")
        with open(args.surfaces) as f:
            surface_colors = json.load(f)
        for name in ("wall", "floor", "ceiling"):
            if name in surface_colors:
                c = surface_colors[name]["color"]
                print(f"  {name}: RGB({c[0]},{c[1]},{c[2]})")

    scene = reconstruct_room_scene(
        args.image, args.checkpoint, hfov=args.hfov,
        surface_colors=surface_colors, subdivisions=args.subdivisions,
        max_displacement=args.max_displacement,
        min_plane_points=args.min_plane_points,
        ransac_threshold=args.ransac_threshold, max_planes=args.max_planes)
    out = export_room(scene, args.name, args.scale, args.viewer_dir)
    print("Open the Godot project in godot_viewer/ and press F5 -- it loads "
          "all .glb files automatically.")


if __name__ == "__main__":
    main()
