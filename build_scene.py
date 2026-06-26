#!/usr/bin/env python3
"""
build_scene.py -- single-command pipeline from a room photo to a walkable
Godot scene.

    python build_scene.py photo.jpg
    python build_scene.py photo.jpg --name living_room --scale 3
    python build_scene.py photo.jpg --midi-output path/to/output.glb

Chains:
    1. environment check
    2. surface segmentation (SegFormer, for wall/floor/ceiling colors)
    3. depth estimation (DepthAnything V2)
    4. room shell with depth-displaced walls
    5. 3D object generation (MIDI, if installed + seg map provided)
    6. object processing (decimation / scaling)
    7. combine room + objects -> final GLB

Every step prints elapsed time. Long GPU steps show a status line so you
know it hasn't hung.

Deps (minimal -- room shell only):
    pip install torch numpy open3d trimesh scipy Pillow

Optional (for segmentation):
    pip install transformers

Optional (for MIDI objects):
    See docs/midi-windows-blackwell-setup.md
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# PROGRESS TRACKER
# ---------------------------------------------------------------------------

class Progress:
    """Step-by-step progress with elapsed time."""

    def __init__(self, steps: list[str]):
        self._steps = steps
        self._total = len(steps)
        self._current = 0
        self._t0 = time.time()
        self._step_t0 = 0.0

    @staticmethod
    def _fmt(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"

    def begin(self, hint: str = ""):
        self._current += 1
        self._step_t0 = time.time()
        label = self._steps[self._current - 1]
        total_elapsed = self._fmt(time.time() - self._t0)
        bar = f"[{self._current}/{self._total}]"
        print(f"\n{'=' * 70}")
        print(f"{bar} {label}  (total: {total_elapsed})")
        if hint:
            print(f"    {hint}")
        print(f"{'=' * 70}")

    def detail(self, msg: str):
        elapsed = self._fmt(time.time() - self._step_t0)
        print(f"  {msg}  ({elapsed})")

    def done(self, msg: str = ""):
        elapsed = self._fmt(time.time() - self._step_t0)
        suffix = f"  {msg}" if msg else ""
        print(f"  Done.{suffix}  ({elapsed})")

    def finish(self, msg: str = ""):
        total = self._fmt(time.time() - self._t0)
        print(f"\n{'=' * 70}")
        print(f"Pipeline complete!  Total: {total}")
        if msg:
            print(msg)
        print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# ENVIRONMENT DETECTION
# ---------------------------------------------------------------------------

def _try_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _check_env(args) -> dict:
    """Detect which tools are available."""
    env = {}

    env["torch"] = _try_import("torch")
    if env["torch"]:
        import torch
        env["cuda"] = torch.cuda.is_available()
        env["device"] = "cuda" if env["cuda"] else "cpu"
    else:
        env["cuda"] = False
        env["device"] = "cpu"

    env["segformer"] = _try_import("transformers")
    env["open3d"] = _try_import("open3d")
    env["trimesh"] = _try_import("trimesh")

    checkpoint = Path(args.checkpoint)
    env["depth_checkpoint"] = checkpoint.is_file()
    env["checkpoint_path"] = str(checkpoint)

    midi_dir = Path(args.midi_dir) if args.midi_dir else None
    if midi_dir is None:
        for candidate in [
            Path(__file__).resolve().parent.parent / "external" / "MIDI-3D",
            Path(__file__).resolve().parent.parent.parent / "MIDI-3D",
            Path.home() / "projects" / "MIDI-3D",
        ]:
            if candidate.is_dir() and (candidate / "scripts").is_dir():
                midi_dir = candidate
                break
    env["midi"] = (midi_dir is not None
                   and midi_dir.is_dir()
                   and (midi_dir / "scripts").is_dir())
    env["midi_dir"] = str(midi_dir) if midi_dir else None

    env["midi_output"] = (args.midi_output
                          and Path(args.midi_output).is_file())

    return env


# ---------------------------------------------------------------------------
# DEPTH-BASED OBJECT PLACEMENT
# ---------------------------------------------------------------------------

def compute_object_placements(
    depth: "np.ndarray",
    seg_path: str,
    label_list: list[int],
    fx: float, fy: float, cx: float, cy: float,
    scale: float,
) -> dict[int, dict]:
    """Compute per-object centroid and extent from depth + segmentation.

    For each object, unprojects its segmentation pixels through the depth map
    to get a real-world 3D bounding box.

    Returns {object_index: {"centroid": array(3), "extent": array(3)}}
    in Godot frame (after flip + scale), but BEFORE room centering shift.
    """
    import numpy as np
    from PIL import Image

    seg = np.array(Image.open(seg_path).convert("L"))
    H, W = depth.shape
    if seg.shape != (H, W):
        from PIL import Image as PILImage
        seg = np.array(PILImage.open(seg_path).convert("L").resize(
            (W, H), PILImage.NEAREST))

    placements = {}
    for i, label in enumerate(label_list):
        mask = (seg == label)
        if mask.sum() < 50:
            continue

        vs, us = np.where(mask)
        zs = depth[vs, us]
        valid = (zs > 0.1) & (zs < 19.0) & np.isfinite(zs)
        if valid.sum() < 50:
            continue

        u, v, z = us[valid].astype(np.float32), vs[valid].astype(np.float32), zs[valid]
        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy

        # Camera frame → Godot frame: flip Y and Z, then scale
        x_god = x_cam * scale
        y_god = -y_cam * scale
        z_god = -z * scale

        pts = np.stack([x_god, y_god, z_god], axis=-1)

        # Median centroid: robust to outlier background pixels.
        centroid = np.median(pts, axis=0)
        # IQR (25th–75th percentile) for extent: background pixels leaking
        # through the segmentation mask inflate q10/q90 badly (a sofa at 5m
        # with 10% background at 8m makes q10 look like a 10m-deep object).
        # IQR excludes those tails and gives the object's real apparent size.
        q25 = np.percentile(pts, 25, axis=0)
        q75 = np.percentile(pts, 75, axis=0)

        placements[i] = {
            "centroid": centroid,
            "extent": q75 - q25,
        }

    return placements


# ---------------------------------------------------------------------------
# PIPELINE STEPS
# ---------------------------------------------------------------------------

def step_segment(image_path: str, progress: Progress) -> dict | None:
    """Run SegFormer surface segmentation."""
    progress.begin("Loads SegFormer model on first run (~30s download)")

    from segment_room import segment_image, group_surfaces, sample_surface_colors
    import numpy as np
    from PIL import Image

    progress.detail("Running segmentation...")
    label_map, id2label = segment_image(image_path)
    progress.detail(f"{label_map.shape[1]}x{label_map.shape[0]} px, "
                    f"{len(np.unique(label_map))} classes")

    masks = group_surfaces(label_map, id2label)
    image = np.array(Image.open(image_path).convert("RGB"))
    surfaces = sample_surface_colors(image, masks)

    summary = []
    for name, info in list(surfaces.items())[:4]:
        r, g, b = info["color"]
        summary.append(f"{name}: RGB({r},{g},{b}) {info['area_pct']}%")
    progress.done(" | ".join(summary))
    return surfaces


def step_depth(image_path: str, checkpoint: str, hfov: float,
               progress: Progress) -> tuple:
    """Estimate depth with DepthAnything V2."""
    progress.begin("First run downloads the model (~1 min); inference ~30-120s")

    from room_from_image import estimate_depth, depth_to_pointcloud

    progress.detail("Loading model + running inference...")
    depth, rgb = estimate_depth(image_path, checkpoint)
    progress.detail(f"Depth range: {depth.min():.2f} - {depth.max():.2f} m  "
                    f"({rgb.shape[1]}x{rgb.shape[0]} px)")

    progress.detail("Building point cloud...")
    points, colors, fx, fy, cx, cy = depth_to_pointcloud(depth, rgb, hfov)
    progress.done(f"{len(points):,} valid points")

    return depth, rgb, points, colors, fx, fy, cx, cy


def step_room_shell(depth, points, fx, fy, cx, cy, surface_colors,
                    args, progress: Progress) -> "trimesh.Scene":
    """Build the room shell with depth displacement."""
    progress.begin()

    from room_from_image import (detect_planes, classify_planes,
                                 build_room_scene)

    progress.detail("Detecting planes (RANSAC)...")
    planes = detect_planes(points, args.min_plane_points,
                           args.ransac_threshold, args.max_planes)
    if len(planes) < 2:
        progress.detail(f"Only {len(planes)} plane(s), using box fallback")
        from room_from_image import box_fallback
        planes = box_fallback(points)
    classify_planes(planes, points)
    labels = ", ".join(f"{p.label} ({len(p.inlier_indices):,})"
                       for p in planes)
    progress.detail(f"{len(planes)} planes: {labels}")

    if args.subdivisions > 1:
        progress.detail(f"Subdividing walls ({args.subdivisions}x grid, "
                        f"{args.max_displacement * 100:.0f}cm max displacement)...")
    scene = build_room_scene(
        planes, points, surface_colors=surface_colors,
        depth_map=depth, cam_intrinsics=(fx, fy, cx, cy),
        subdivisions=args.subdivisions,
        max_displacement=args.max_displacement)
    progress.done()
    return scene


def step_segment_objects(image_path: str, midi_dir: str, output_dir: str,
                         progress: Progress) -> str | None:
    """Run Grounding-DINO + SAM to create instance segmentation for MIDI."""
    progress.begin("Downloads models on first run (~1 min)")

    os.makedirs(output_dir, exist_ok=True)
    seg_path = os.path.join(output_dir, "segmentation.png")

    labels = ["sofa", "chair", "table", "bookshelf", "shelf", "plant",
              "rug", "lamp", "desk", "bed", "cabinet", "dresser"]

    env = os.environ.copy()
    env["PYTHONPATH"] = midi_dir

    cmd = [
        sys.executable, os.path.join(midi_dir, "scripts", "grounding_sam.py"),
        "--image", str(image_path),
        "--labels", *labels,
        "--output", str(output_dir),
        "--threshold", "0.25",
    ]

    progress.detail("Detecting and segmenting objects...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        progress.detail(f"Segmentation failed: {result.stderr.strip()[:300]}")
        progress.done("Failed")
        return None

    if not Path(seg_path).is_file():
        progress.detail("No segmentation output produced")
        progress.done("Failed — no objects detected?")
        return None

    progress.done(seg_path)
    return seg_path


def step_midi(image_path: str, seg_path: str, midi_dir: str,
              output_dir: str, batch_size: int,
              progress: Progress) -> tuple[str | None, list[int] | None]:
    """Run MIDI 3D object generation in VRAM-safe batches.

    Returns (glb_path, label_list) where label_list[i] is the original
    segmentation label for object_i in the merged GLB.
    """
    progress.begin("Each batch of ~4 objects takes 5-10 min")
    import json
    import trimesh

    from batch_midi import split_seg_mask, create_batch_mask, run_midi_batch, merge_glbs

    labels, seg = split_seg_mask(seg_path)
    batches = [labels[i:i + batch_size]
               for i in range(0, len(labels), batch_size)]
    progress.detail(f"{len(labels)} objects -> {len(batches)} batch(es) "
                    f"of up to {batch_size}")

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    glb_paths = []
    successful_labels = []

    for i, batch_labels in enumerate(batches):
        progress.detail(f"Batch {i + 1}/{len(batches)}: "
                        f"{len(batch_labels)} objects (labels {batch_labels})...")
        batch_seg = os.path.join(output_dir, f"seg_batch_{i}.png")
        create_batch_mask(seg, batch_labels, batch_seg)
        glb = run_midi_batch(image_path, batch_seg, output_dir, midi_dir, i)
        if glb:
            mb = os.path.getsize(glb) / 1e6
            scene = trimesh.load(glb, process=False)
            n_meshes = (len(scene.geometry)
                        if isinstance(scene, trimesh.Scene) else 1)
            successful_labels.extend(batch_labels[:n_meshes])
            progress.detail(f"  Batch {i + 1} done: {n_meshes} objects, "
                            f"{mb:.1f} MB")
            glb_paths.append(glb)
        else:
            progress.detail(f"  Batch {i + 1} failed")

    if not glb_paths:
        progress.done("All batches failed")
        return None, None

    merged = os.path.join(output_dir, "output.glb")
    n = merge_glbs(glb_paths, merged)
    mb = os.path.getsize(merged) / 1e6

    # Save label mapping for reuse with --midi-output
    map_path = os.path.join(output_dir, "object_label_map.json")
    with open(map_path, "w") as f:
        json.dump({"labels": [int(l) for l in successful_labels]}, f)

    progress.done(f"{n} objects, {mb:.1f} MB, "
                  f"took {Progress._fmt(time.time() - t0)}")
    return merged, successful_labels


def step_process_objects(glb_path: str, target_faces: int,
                         progress: Progress) -> "trimesh.Scene":
    """Decimate MIDI objects. CRITICAL: original geometry names are preserved
    so label_list[i] correctly maps to object_i in step_combine.

    scene.dump() was intentionally replaced with scene.geometry.items():
    dump() uses scene-graph traversal order (lexicographic for 'object_10' <
    'object_2'), which scrambles the label mapping for any run with ≥ 10 objects.
    """
    progress.begin()
    import numpy as np
    import trimesh

    scene = trimesh.load(glb_path, process=False)
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(scene)

    from to_godot import _is_textured

    try:
        import fast_simplification as fs
        can_decimate = True
    except ImportError:
        progress.detail("fast-simplification not installed, skipping decimation")
        can_decimate = False

    # Sort by numeric index to guarantee consistent order regardless of how
    # trimesh loaded the scene graph (avoids lexicographic scrambling).
    def _sort_key(kv):
        tail = kv[0].rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    out_scene = trimesh.Scene()
    before = after = n = 0
    for gname, geom in sorted(scene.geometry.items(), key=_sort_key):
        g = geom.copy()
        tf = scene.graph.get(gname)
        if tf is not None and isinstance(tf, tuple):
            g.apply_transform(tf[0])

        V = np.asarray(g.vertices, np.float32)
        F = np.asarray(g.faces, np.int32)
        before += len(F)
        if can_decimate and not _is_textured(g) and len(F) > target_faces:
            reduction = 1.0 - target_faces / len(F)
            V, F = fs.simplify(V, F, target_reduction=reduction)
        after += len(F)
        n += 1
        out_scene.add_geometry(
            trimesh.Trimesh(vertices=V, faces=F, process=False),
            geom_name=gname)  # preserve name so label_list mapping holds

    progress.done(f"{n} objects, {before:,} -> {after:,} faces")
    return out_scene


def _resolve_overlaps(
    placed: list[tuple[str, "trimesh.Trimesh"]],
    room_bounds: "np.ndarray",
    floor_y: float,
    margin: float = 0.3,
    max_iter: int = 30,
) -> list[tuple[str, "trimesh.Trimesh"]]:
    """Push objects apart in XZ until no bounding-box overlaps remain."""
    import numpy as np

    if len(placed) < 2:
        return placed

    rx_min, rx_max = room_bounds[0][0], room_bounds[1][0]
    rz_min, rz_max = room_bounds[0][2], room_bounds[1][2]

    def _xz(g):
        v = g.vertices
        return (v[:, 0].min(), v[:, 0].max(),
                v[:, 2].min(), v[:, 2].max())

    for _ in range(max_iter):
        moved = False
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                gi, gj = placed[i][1], placed[j][1]
                xi0, xi1, zi0, zi1 = _xz(gi)
                xj0, xj1, zj0, zj1 = _xz(gj)

                ox = min(xi1, xj1) - max(xi0, xj0)  # overlap in X
                oz = min(zi1, zj1) - max(zi0, zj0)  # overlap in Z
                if ox <= 0 or oz <= 0:
                    continue

                # Push along the axis with smaller overlap + margin
                push_x = (ox + margin) / 2
                push_z = (oz + margin) / 2
                cxi = (xi0 + xi1) / 2
                cxj = (xj0 + xj1) / 2
                czi = (zi0 + zi1) / 2
                czj = (zj0 + zj1) / 2

                if push_x <= push_z:
                    sign = 1 if cxi <= cxj else -1
                    shift_i = np.array([-sign * push_x, 0, 0])
                    shift_j = np.array([+sign * push_x, 0, 0])
                else:
                    sign = 1 if czi <= czj else -1
                    shift_i = np.array([0, 0, -sign * push_z])
                    shift_j = np.array([0, 0, +sign * push_z])

                gi.vertices = gi.vertices + shift_i
                gj.vertices = gj.vertices + shift_j
                moved = True

        # Clamp to room walls (leave a small margin from each wall)
        wall_m = 0.2
        for _, g in placed:
            v = g.vertices
            xl, xr = v[:, 0].min(), v[:, 0].max()
            zb, zf = v[:, 2].min(), v[:, 2].max()
            dx = dz = 0.0
            if xl < rx_min + wall_m:
                dx = rx_min + wall_m - xl
            elif xr > rx_max - wall_m:
                dx = rx_max - wall_m - xr
            if zb < rz_min + wall_m:
                dz = rz_min + wall_m - zb
            elif zf > rx_max - wall_m:
                dz = rz_max - wall_m - zf
            if dx or dz:
                g.vertices = g.vertices + np.array([dx, 0, dz])
                moved = True

        if not moved:
            break

    # Re-seat all objects on the floor after shifts
    for _, g in placed:
        y_min = g.vertices[:, 1].min()
        g.vertices[:, 1] += floor_y - y_min

    return placed


def step_combine(room_scene, object_scene, name: str, scale: float,
                 project_dir: str, image_path: str,
                 surface_colors: dict | None,
                 progress: Progress,
                 *,
                 depth: "np.ndarray | None" = None,
                 cam_intrinsics: tuple | None = None,
                 seg_path: str | None = None,
                 label_list: list[int] | None = None,
                 ) -> str:
    """Combine room + objects, export final GLB into the project folder.

    When depth/cam_intrinsics/seg_path/label_list are provided, each MIDI
    object is individually scaled and positioned using its depth-estimated
    real-world bounding box. Otherwise falls back to global placement.
    """
    progress.begin()
    import json
    import shutil
    import numpy as np
    import trimesh

    combined = trimesh.Scene()

    # Camera frame (OpenCV: +x right, +y down, +z forward) to
    # Godot frame (+x right, +y up, +z backward): negate Y and Z.
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    room_xform = flip @ trimesh.transformations.scale_matrix(scale)

    # Bake transforms into vertices (scene.apply_transform only sets graph
    # transforms, which get lost when we copy geometries to 'combined').
    room_meshes = []
    for gname, geom in room_scene.geometry.items():
        g = geom.copy()
        g.apply_transform(room_xform)
        room_meshes.append((gname, g))

    # Center the room: X and Z centered, floor at Y=0.
    all_verts = np.vstack([g.vertices for _, g in room_meshes])
    room_min = all_verts.min(axis=0)
    room_max = all_verts.max(axis=0)
    room_center = (room_min + room_max) / 2.0
    room_shift = np.array([-room_center[0], -room_min[1], -room_center[2]])

    shift_mat = np.eye(4)
    shift_mat[0, 3], shift_mat[1, 3], shift_mat[2, 3] = room_shift

    room_count = 0
    for gname, g in room_meshes:
        g.apply_transform(shift_mat)
        combined.add_geometry(g, geom_name=f"room_{gname}")
        room_count += 1

    room_bounds = combined.bounds
    room_ext = room_bounds[1] - room_bounds[0]
    room_height = room_ext[1]

    # The floor quad is depth-displaced, so its median Y is the actual visible
    # surface (~17 cm above absolute Y=0). Place objects here, not at Y=0.
    floor_y = room_bounds[0][1]  # fallback
    for gname, geom in combined.geometry.items():
        if "room_floor" in gname:
            floor_y = float(np.median(geom.vertices[:, 1]))
            break

    progress.detail(f"Room: {room_ext[0]:.1f} x {room_ext[1]:.1f} x "
                    f"{room_ext[2]:.1f}m  floor surface at Y={floor_y:.2f}")

    # --- Place objects ---
    obj_count = 0
    use_depth = (object_scene is not None
                 and depth is not None and cam_intrinsics is not None
                 and seg_path is not None and label_list is not None)

    if object_scene is not None and use_depth:
        fx, fy, cx, cy = cam_intrinsics
        placements = compute_object_placements(
            depth, seg_path, label_list, fx, fy, cx, cy, scale)
        progress.detail(f"Depth centroids for {len(placements)}/{len(label_list)} objects")

        # One global scale keeps objects proportional to each other.
        # MIDI normalises each object independently so per-object depth
        # scaling produces inconsistent sizes (a plant's leaves hit the
        # wall behind them, inflating its depth extent). The global scale
        # is derived from the whole MIDI scene vs room floor area.
        all_obj_meshes = []
        for gname, geom in object_scene.geometry.items():
            g = geom.copy()
            tf = object_scene.graph.get(gname)
            if tf is not None and isinstance(tf, tuple):
                g.apply_transform(tf[0])
            all_obj_meshes.append((gname, g))

        all_verts = np.vstack([g.vertices for _, g in all_obj_meshes])
        scene_ext = all_verts.max(axis=0) - all_verts.min(axis=0)
        margin = 0.75
        sx = room_ext[0] / max(scene_ext[0], 1e-6)
        sz = room_ext[2] / max(scene_ext[2], 1e-6)
        global_scale = min(sx, sz) * margin
        # Also clamp so the tallest object doesn't exceed room height
        scene_tall = scene_ext[1]
        if scene_tall * global_scale > room_height * 0.90:
            global_scale = min(global_scale,
                               room_height * 0.90 / max(scene_tall, 1e-6))
        progress.detail(f"Global scale: {global_scale:.2f}x "
                        f"(MIDI scene {scene_ext[0]:.1f}x{scene_ext[2]:.1f}m "
                        f"-> room {room_ext[0]:.1f}x{room_ext[2]:.1f}m)")

        # Apply global scale, then clamp each object's individual extents using
        # the depth-estimated size so MIDI's depth-axis ambiguity doesn't make
        # objects unrealistically large (e.g., sofa 6.5m deep in a 15m room).
        scale_mat = trimesh.transformations.scale_matrix(global_scale)
        for _, g in all_obj_meshes:
            g.apply_transform(scale_mat)

        # Depth-based per-object size clamp (Z and X only — Y height stays global).
        # For each object, the depth-map percentile extent gives its real apparent
        # size in the image. Clamp the MIDI extent to 1.5× that; this prevents
        # depth-axis ambiguity from making objects occupy half the room.
        max_z = room_ext[2] * 0.12  # hard cap: no object > 12% of room depth (~0.6m real)
        max_x = room_ext[0] * 0.22  # hard cap: no object > 22% of room width (~1.3m real)
        for gname, g in all_obj_meshes:
            idx = int(gname.split("_")[-1])
            verts = g.vertices
            obj_ext = verts.max(axis=0) - verts.min(axis=0)
            obj_ctr = (verts.min(axis=0) + verts.max(axis=0)) / 2.0

            target_z = max_z
            target_x = max_x
            if idx in placements:
                # Prefer depth-map estimate × 1.5 when it's tighter than hard cap
                depth_ext = placements[idx]["extent"]
                target_z = min(max_z, depth_ext[2] * 1.5)
                target_x = min(max_x, depth_ext[0] * 1.5)

            new_verts = verts.copy()
            if obj_ext[2] > target_z:
                sz = target_z / obj_ext[2]
                new_verts[:, 2] = (new_verts[:, 2] - obj_ctr[2]) * sz + obj_ctr[2]
            if obj_ext[0] > target_x:
                sx = target_x / obj_ext[0]
                new_verts[:, 0] = (new_verts[:, 0] - obj_ctr[0]) * sx + obj_ctr[0]
            g.vertices = new_verts

        # Position each object at its depth-estimated centroid.
        placed = []
        for gname, g in all_obj_meshes:
            idx = int(gname.split("_")[-1])
            midi_min = g.vertices.min(axis=0)
            midi_ctr = (midi_min + g.vertices.max(axis=0)) / 2.0

            if idx in placements:
                target_ctr = placements[idx]["centroid"] + room_shift
                place = np.eye(4)
                place[0, 3] = target_ctr[0] - midi_ctr[0]
                place[1, 3] = floor_y - midi_min[1]
                place[2, 3] = target_ctr[2] - midi_ctr[2]
            else:
                place = np.eye(4)
                place[1, 3] = floor_y - midi_min[1]

            g.apply_transform(place)
            placed.append((gname, g))

        # Push overlapping objects apart in XZ so they don't share the same space.
        placed = _resolve_overlaps(placed, room_bounds, floor_y)

        for gname, g in placed:
            combined.add_geometry(g, geom_name=gname)
            obj_count += 1

    elif object_scene is not None:
        # Global fallback when depth data unavailable
        obj_meshes = []
        for gname, geom in object_scene.geometry.items():
            g = geom.copy()
            tf = object_scene.graph.get(gname)
            if tf is not None and isinstance(tf, tuple):
                g.apply_transform(tf[0])
            obj_meshes.append((gname, g))

        obj_verts = np.vstack([g.vertices for _, g in obj_meshes])
        obj_ext = obj_verts.max(axis=0) - obj_verts.min(axis=0)
        margin = 0.80
        sx = room_ext[0] / max(obj_ext[0], 1e-6)
        sz = room_ext[2] / max(obj_ext[2], 1e-6)
        obj_scale = min(sx, sz) * margin
        progress.detail(f"Global scale fallback: {obj_scale:.1f}x")

        scale_mat = trimesh.transformations.scale_matrix(obj_scale)
        for _, g in obj_meshes:
            g.apply_transform(scale_mat)

        obj_verts = np.vstack([g.vertices for _, g in obj_meshes])
        obj_min = obj_verts.min(axis=0)
        obj_center = (obj_min + obj_verts.max(axis=0)) / 2.0
        place = np.eye(4)
        place[0, 3] = -obj_center[0]
        place[1, 3] = floor_y - obj_min[1]
        place[2, 3] = -obj_center[2]
        for gname, g in obj_meshes:
            g.apply_transform(place)
            combined.add_geometry(g, geom_name=gname)
            obj_count += 1

    os.makedirs(project_dir, exist_ok=True)
    out_path = os.path.join(project_dir, f"{name}.glb")
    combined.export(out_path)
    mb = os.path.getsize(out_path) / 1e6

    # Keep the source image and surface colors alongside the GLB for reference
    src_copy = os.path.join(project_dir, "source" + Path(image_path).suffix)
    shutil.copy2(image_path, src_copy)
    if surface_colors:
        with open(os.path.join(project_dir, "surfaces.json"), "w") as f:
            json.dump(surface_colors, f, indent=2)

    progress.done(f"{room_count} room surfaces + {obj_count} objects -> "
                  f"{name}.glb ({mb:.1f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser(
        description="Room photo -> walkable Godot scene (one command)")
    ap.add_argument("image", help="perspective photo/render of a room interior")
    ap.add_argument("--name", default=None,
                    help="output name (default: derived from filename)")
    ap.add_argument("--scale", type=float, default=3.0,
                    help="uniform scale for the scene (default 3)")
    ap.add_argument("--hfov", type=float, default=60.0,
                    help="camera horizontal FOV in degrees (default 60)")

    ap.add_argument("--subdivisions", type=int, default=24,
                    help="wall grid size for depth displacement (1=flat, default 24)")
    ap.add_argument("--max-displacement", type=float, default=0.15,
                    help="max depth displacement in metres (default 0.15)")

    ap.add_argument("--skip-segmentation", action="store_true",
                    help="skip SegFormer (use default surface colors)")
    ap.add_argument("--skip-midi", action="store_true",
                    help="skip MIDI even if available")

    ap.add_argument("--seg", default=None,
                    help="instance segmentation image for MIDI (optional)")
    ap.add_argument("--midi-dir", default=None,
                    help="path to MIDI-3D repo (auto-detected if on PATH)")
    ap.add_argument("--midi-output", default=None,
                    help="use an existing MIDI .glb instead of running MIDI")
    ap.add_argument("--target-faces", type=int, default=12000,
                    help="per-object face budget for untextured MIDI meshes")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="objects per MIDI batch to stay within VRAM (default 4)")

    ap.add_argument("--viewer-dir",
                    default=os.path.join(script_dir, "godot_viewer"),
                    help="Godot project dir (default: godot_viewer/)")
    ap.add_argument("--checkpoint",
                    default=os.path.join(script_dir, "checkpoints",
                                         "depth_anything_v2_metric_hypersim_vitl.pth"),
                    help="DepthAnything V2 checkpoint path")
    ap.add_argument("--min-plane-points", type=int, default=5000)
    ap.add_argument("--ransac-threshold", type=float, default=0.02)
    ap.add_argument("--max-planes", type=int, default=8)

    args = ap.parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"Image not found: {args.image}")

    # Prompt for project name if not given on the command line
    if args.name is None:
        default = Path(args.image).stem
        try:
            answer = input(f"Project name [{default}]: ").strip()
        except EOFError:
            answer = ""
        args.name = answer if answer else default

    # Make sure this script's directory is importable
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Project subfolder keeps each run isolated
    project_dir = os.path.join(args.viewer_dir, args.name)
    os.makedirs(project_dir, exist_ok=True)

    # --- Plan steps ---
    print(f"\n  Image:   {args.image}")
    print(f"  Project: {args.name} -> {project_dir}")
    print(f"  Scale:   {args.scale}")

    env = _check_env(args)

    steps = ["Check environment"]
    do_segment = env["segformer"] and not args.skip_segmentation
    if do_segment:
        steps.append("Segment surfaces")
    steps.append("Estimate depth")
    steps.append("Build room shell")

    do_midi_run = env["midi"] and not args.skip_midi and not args.midi_output
    do_midi_load = bool(args.midi_output and env["midi_output"])
    if do_midi_run:
        steps.append("Segment objects (Grounding-DINO + SAM)")
        steps.append("Generate 3D objects (MIDI)")
        steps.append("Process objects for Godot")
    elif do_midi_load:
        steps.append("Process existing MIDI output")
    steps.append("Combine & export")

    progress = Progress(steps)

    # --- Step 1: Environment ---
    progress.begin()
    status = {
        "PyTorch":     "available" + (f" (CUDA: {env['device']})" if env["torch"] else ""),
        "DepthAnything": "checkpoint found" if env["depth_checkpoint"] else f"NOT FOUND: {env['checkpoint_path']}",
        "SegFormer":   "available" if env["segformer"] else "not installed (pip install transformers)",
        "Open3D":      "available" if env["open3d"] else "NOT FOUND (pip install open3d)",
        "trimesh":     "available" if env["trimesh"] else "NOT FOUND (pip install trimesh)",
        "MIDI":        f"found at {env['midi_dir']}" if env["midi"] else "not configured",
    }
    for k, v in status.items():
        print(f"  {k:16s} {v}")

    if not env["torch"]:
        sys.exit("\nFATAL: PyTorch is required. Install it first.")
    if not env["depth_checkpoint"]:
        sys.exit(f"\nFATAL: DepthAnything checkpoint not found at "
                 f"{env['checkpoint_path']}\n"
                 "Download from: https://huggingface.co/depth-anything/"
                 "Depth-Anything-V2-Metric-Hypersim-Large")
    if not env["open3d"] or not env["trimesh"]:
        sys.exit("\nFATAL: open3d and trimesh are required. "
                 "pip install open3d trimesh")

    progress.done()

    # --- Step 2: Segmentation ---
    surface_colors = None
    if do_segment:
        try:
            surface_colors = step_segment(args.image, progress)
        except Exception as e:
            progress.detail(f"Segmentation failed: {e}")
            progress.done("Continuing with default colors")

    # --- Step 3: Depth ---
    depth, rgb, points, colors, fx, fy, cx, cy = step_depth(
        args.image, args.checkpoint, args.hfov, progress)

    # --- Step 4: Room shell ---
    room_scene = step_room_shell(
        depth, points, fx, fy, cx, cy, surface_colors, args, progress)

    # --- Step 5-7: MIDI objects ---
    object_scene = None
    label_list = None
    midi_seg_path = None

    if do_midi_run:
        midi_out_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.image)),
            f"midi_{args.name}")
        midi_seg_path = args.seg
        if not midi_seg_path:
            midi_seg_path = step_segment_objects(
                args.image, env["midi_dir"], midi_out_dir, progress)
        if midi_seg_path:
            glb_path, label_list = step_midi(
                args.image, midi_seg_path, env["midi_dir"],
                midi_out_dir, args.batch_size, progress)
            if glb_path:
                object_scene = step_process_objects(
                    glb_path, args.target_faces, progress)
    elif do_midi_load:
        object_scene = step_process_objects(
            args.midi_output, args.target_faces, progress)
        # Try to recover label mapping and seg path from sidecar / --seg
        midi_dir = os.path.dirname(os.path.abspath(args.midi_output))
        map_file = os.path.join(midi_dir, "object_label_map.json")
        if os.path.isfile(map_file):
            import json as _json
            label_list = _json.load(open(map_file))["labels"]
        seg_candidate = os.path.join(midi_dir, "segmentation.png")
        midi_seg_path = args.seg or (seg_candidate
                                      if os.path.isfile(seg_candidate) else None)

    # --- Final: Combine ---
    out_path = step_combine(
        room_scene, object_scene, args.name, args.scale,
        project_dir, args.image, surface_colors, progress,
        depth=depth,
        cam_intrinsics=(fx, fy, cx, cy),
        seg_path=midi_seg_path,
        label_list=label_list,
    )

    progress.finish(
        f"Open {args.viewer_dir}/ in Godot and press F5.\n"
        f"Project folder: {project_dir}\n"
        f"Output: {out_path}")


if __name__ == "__main__":
    main()
