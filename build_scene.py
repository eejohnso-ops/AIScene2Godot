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
        # Wider band approximates the object's full apparent size (used to
        # anchor per-object scale). Still trimmed at 10/90 to drop mask leakage.
        q10 = np.percentile(pts, 10, axis=0)
        q90 = np.percentile(pts, 90, axis=0)

        # Yaw from the floor-projected footprint (XZ plane, Y is up in Godot
        # frame). PCA's dominant axis is the object's horizontal long axis; we
        # align the MIDI mesh's long axis to it. `aniso` (long/short eigenvalue
        # ratio) gauges how trustworthy that axis is — a near-circular footprint
        # has no meaningful direction, so callers skip rotation when it's ~1.
        xz = pts[:, [0, 2]]
        xz_c = xz - xz.mean(axis=0)
        try:
            evals, evecs = np.linalg.eigh(np.cov(xz_c.T))
            order = np.argsort(evals)[::-1]
            evals = evals[order]
            principal = evecs[:, order[0]]
            yaw = float(np.arctan2(principal[1], principal[0]))  # atan2(z, x)
            aniso = float(evals[0] / max(evals[1], 1e-9))
        except np.linalg.LinAlgError:
            yaw, aniso = 0.0, 1.0

        placements[i] = {
            "centroid": centroid,
            "extent": q75 - q25,
            "extent_full": q90 - q10,
            "yaw": yaw,
            "aniso": aniso,
        }

    return placements


def _footprint_yaw(verts: "np.ndarray") -> tuple[float, float]:
    """Dominant horizontal (XZ) axis of a vertex set, as (yaw, aniso).

    Mirrors the depth-footprint PCA so a MIDI mesh's long axis can be aligned
    to the depth-derived one. `aniso` is the long/short eigenvalue ratio.
    """
    import numpy as np
    xz = verts[:, [0, 2]]
    xz_c = xz - xz.mean(axis=0)
    try:
        evals, evecs = np.linalg.eigh(np.cov(xz_c.T))
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        principal = evecs[:, order[0]]
        return float(np.arctan2(principal[1], principal[0])), float(evals[0] / max(evals[1], 1e-9))
    except np.linalg.LinAlgError:
        return 0.0, 1.0


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


def _midi_venv_python(midi_dir: str) -> str:
    """MIDI's own venv interpreter (its texturing deps — MV-Adapter, nvdiffrast,
    triton-windows — live there, not in the main pipeline env). Mirrors
    batch_midi._midi_python; falls back to the current interpreter."""
    for rel in (os.path.join(".venv", "Scripts", "python.exe"),
                os.path.join(".venv", "bin", "python")):
        cand = os.path.join(midi_dir, rel)
        if os.path.isfile(cand):
            return cand
    return sys.executable


def step_texture(image_path: str, seg_path: str, untextured_glb: str,
                 label_map_path: str | None, midi_dir: str, output_dir: str,
                 n_objects: int, vcvars: str, cuda_bin: str, seed: int,
                 force: bool, progress: Progress) -> str | None:
    """Texture the merged MIDI scene with MV-Adapter, returning the output dir.

    Texturing must run in the MIDI venv AND inside the VS Build Tools env (triton
    Poisson-blend + nvdiffrast compile CUDA kernels at first use, needing cl.exe +
    nvcc + ninja on PATH). build_scene runs in the main venv, so we shell out to a
    generated wrapper .bat that calls vcvars64, puts the CUDA toolkit + the MIDI
    venv's ninja on PATH, then runs texture_midi.py with cwd/PYTHONPATH = MIDI dir.
    """
    progress.begin("MV-Adapter + UV projection, ~2-3 min/object "
                   "(first object also compiles CUDA kernels)")
    os.makedirs(output_dir, exist_ok=True)

    # Reuse existing per-object results unless forced — texturing is the slowest
    # step, and a reprocess (placement-only) shouldn't pay for it again.
    shaded = [os.path.join(output_dir, f"mesh_{i}_shaded.glb")
              for i in range(n_objects)]
    if not force and all(os.path.isfile(p) for p in shaded):
        progress.done(f"Reusing {n_objects} cached textured meshes in {output_dir}")
        return output_dir

    if not os.path.isfile(vcvars):
        progress.detail(f"vcvars64 not found at {vcvars} — pass --vcvars")
        progress.done("Skipped texturing")
        return None
    if not os.path.isdir(cuda_bin):
        progress.detail(f"CUDA bin not found at {cuda_bin} — pass --cuda-bin")
        progress.done("Skipped texturing")
        return None

    midi_py = _midi_venv_python(midi_dir)
    venv_scripts = os.path.dirname(midi_py)
    cuda_home = os.path.dirname(os.path.abspath(cuda_bin))
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "texture_midi.py")

    cmd = (f'"{midi_py}" "{script}" '
           f'--rgb "{os.path.abspath(image_path)}" '
           f'--seg "{os.path.abspath(seg_path)}" '
           f'--scene "{os.path.abspath(untextured_glb)}" '
           f'--output "{os.path.abspath(output_dir)}" '
           f'--seed {seed}')
    if label_map_path and os.path.isfile(label_map_path):
        cmd += f' --label-map "{os.path.abspath(label_map_path)}"'

    # Generated wrapper: VS build env + CUDA toolkit + MIDI venv ninja on PATH,
    # cwd = MIDI dir so relative checkpoints/ (big-lama, RealESRGAN) resolve.
    bat = os.path.join(output_dir, "_run_texture.bat")
    with open(bat, "w") as f:
        f.write(
            "@echo off\r\n"
            f'call "{vcvars}" >nul\r\n'
            "set DISTUTILS_USE_SDK=1\r\n"
            f'set "CUDA_HOME={cuda_home}"\r\n'
            f'set "PATH={venv_scripts};{cuda_bin};%PATH%"\r\n'
            f'set "PYTHONPATH={midi_dir}"\r\n'
            f'cd /d "{midi_dir}"\r\n'
            f"{cmd}\r\n"
            "exit /b %ERRORLEVEL%\r\n")

    progress.detail(f"Running texture_midi.py in MIDI venv ({os.path.basename(midi_py)}) "
                    f"via VS build env...")
    result = subprocess.run(["cmd", "/c", bat])
    if result.returncode != 0:
        progress.detail(f"Texturing failed (exit {result.returncode})")
        progress.done("Falling back to untextured objects")
        return None

    produced = sum(os.path.isfile(p) for p in shaded)
    if produced == 0:
        progress.done("No textured meshes produced — using untextured objects")
        return None
    progress.done(f"{produced}/{n_objects} objects textured -> {output_dir}")
    return output_dir


def _load_textured_scene(tex_dir: str, n_objects: int,
                         progress: Progress) -> "trimesh.Scene | None":
    """Rebuild an object scene from per-object mesh_<i>_shaded.glb files, naming
    each geometry object_<i> so the depth-align label mapping still holds.

    Falls back to the original object index for any object whose textured file is
    missing (so a single failed object doesn't drop the whole scene)."""
    import trimesh

    scene = trimesh.Scene()
    loaded = 0
    for i in range(n_objects):
        p = os.path.join(tex_dir, f"mesh_{i}_shaded.glb")
        if not os.path.isfile(p):
            continue
        # force='mesh' flattens the single textured mesh and bakes its scene-graph
        # transform into the vertices (the shaded export's graph node name differs
        # from the geometry key, so scene.graph.get(geom_name) can't be used).
        mesh = trimesh.load(p, process=False, force="mesh")
        scene.add_geometry(mesh, geom_name=f"object_{i}")
        loaded += 1

    if loaded == 0:
        return None
    progress.detail(f"Loaded {loaded}/{n_objects} textured meshes")
    return scene


def _resolve_overlaps(
    placed: list[tuple[str, "trimesh.Trimesh"]],
    room_bounds: "np.ndarray",
    floor_y: float,
    margin: float = 0.3,
    max_iter: int = 30,
    reseat_floor: bool = True,
) -> list[tuple[str, "trimesh.Trimesh"]]:
    """Push objects apart until no *3D* bounding-box overlaps remain.

    Only objects that overlap in X, Z **and** Y are separated, so a lamp resting
    on a table (above it, no Y overlap) is left in place while a table embedded
    in a sofa is pushed apart. When ``reseat_floor`` is False, objects keep their
    Y (used by midi-fit, which preserves MIDI's relative heights).
    """
    import numpy as np

    if len(placed) < 2:
        return placed

    rx_min, rx_max = room_bounds[0][0], room_bounds[1][0]
    rz_min, rz_max = room_bounds[0][2], room_bounds[1][2]

    def _xz(g):
        v = g.vertices
        return (v[:, 0].min(), v[:, 0].max(),
                v[:, 2].min(), v[:, 2].max())

    def _y(g):
        v = g.vertices
        return v[:, 1].min(), v[:, 1].max()

    for _ in range(max_iter):
        moved = False
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                gi, gj = placed[i][1], placed[j][1]
                xi0, xi1, zi0, zi1 = _xz(gi)
                xj0, xj1, zj0, zj1 = _xz(gj)
                yi0, yi1 = _y(gi)
                yj0, yj1 = _y(gj)

                ox = min(xi1, xj1) - max(xi0, xj0)  # overlap in X
                oz = min(zi1, zj1) - max(zi0, zj0)  # overlap in Z
                oy = min(yi1, yj1) - max(yi0, yj0)  # overlap in Y
                if ox <= 0 or oz <= 0 or oy <= 0:
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

    # Re-seat all objects on the floor after shifts (skipped for midi-fit, which
    # preserves MIDI's relative heights — e.g. a lamp standing on a table).
    if reseat_floor:
        for _, g in placed:
            y_min = g.vertices[:, 1].min()
            g.vertices[:, 1] += floor_y - y_min

    return placed


def _snap_to_walls(
    placed: list[tuple[str, "trimesh.Trimesh"]],
    room_bounds: "np.ndarray",
    cam_side_zmax: bool = True,
    snap_frac: float = 0.18,
    wall_gap: float = 0.05,
) -> list[tuple[str, "trimesh.Trimesh"]]:
    """Push near-wall furniture flush against its nearest wall.

    The depth centroid sits on an object's visible front face, so wall furniture
    lands ~0.5 m off the wall. For each object we find the nearest of the three
    non-camera walls (left/right/back) and, if it's within `snap_frac` of the room
    size, translate the object so its edge meets the wall (minus a small gap).
    The camera-side wall (Z-max by convention) is skipped so nothing is pushed
    toward the viewer; genuinely central objects (large gaps to all walls) are
    left alone.
    """
    import numpy as np

    rx0, rx1 = room_bounds[0][0], room_bounds[1][0]
    rz0, rz1 = room_bounds[0][2], room_bounds[1][2]
    thr_x = snap_frac * (rx1 - rx0)
    thr_z = snap_frac * (rz1 - rz0)

    for _, g in placed:
        v = g.vertices
        x0, x1 = v[:, 0].min(), v[:, 0].max()
        z0, z1 = v[:, 2].min(), v[:, 2].max()
        # Candidate walls: left (X-min), right (X-max), back (Z-min). The
        # camera-side wall (Z-max) is excluded when cam_side_zmax is True.
        cands = [("L", x0 - rx0, thr_x), ("R", rx1 - x1, thr_x),
                 ("B", z0 - rz0, thr_z)]
        if not cam_side_zmax:
            cands.append(("F", rz1 - z1, thr_z))
        wall, gap, thr = min(cands, key=lambda c: abs(c[1]))
        if abs(gap) > thr:
            continue  # too far from any wall -> a central object, leave it
        dx = dz = 0.0
        if wall == "L":
            dx = rx0 + wall_gap - x0
        elif wall == "R":
            dx = rx1 - wall_gap - x1
        elif wall == "B":
            dz = rz0 + wall_gap - z0
        else:  # "F"
            dz = rz1 - wall_gap - z1
        if dx or dz:
            g.vertices = v + np.array([dx, 0.0, dz])

    return placed


def _place_midi_fit(
    object_scene: "trimesh.Scene",
    interior_bounds: "np.ndarray",
    floor_y: float,
    room_height: float,
    yaw: float = 0.0,
    margin: float = 0.92,
) -> list[tuple[str, "trimesh.Trimesh"]]:
    """Fit the WHOLE single-batch MIDI scene into the room with one similarity
    transform, preserving MIDI's native arrangement.

    Single-batch MIDI lays out all objects in one coherent scene — relative
    positions, facings, and relative sizes are already correct. Rather than
    reconstruct each object's pose from depth (fragile under a single oblique
    view), we keep that arrangement intact and only: (optionally) yaw the whole
    scene to the room axes, uniformly scale it to the room footprint, centre it,
    and drop it on the floor.
    """
    import numpy as np

    meshes = []
    for gname, geom in object_scene.geometry.items():
        g = geom.copy()
        tf = object_scene.graph.get(gname)
        if tf is not None and isinstance(tf, tuple):
            g.apply_transform(tf[0])
        meshes.append((gname, g))

    def _all_v():
        return np.vstack([g.vertices for _, g in meshes])

    # Global yaw about the scene centre (XZ), if requested.
    c = (_all_v().min(0) + _all_v().max(0)) / 2.0
    if abs(yaw) > 1e-6:
        cy, sy = np.cos(yaw), np.sin(yaw)
        R = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        for _, g in meshes:
            g.vertices = (g.vertices - c) @ R.T + c

    # Uniform scale to fit the footprint inside the interior walls, capped by
    # room height. One scale for the whole scene keeps MIDI's relative sizes.
    ext = _all_v().max(0) - _all_v().min(0)
    int_ext = interior_bounds[1] - interior_bounds[0]
    S = min(int_ext[0] / max(ext[0], 1e-6), int_ext[2] / max(ext[2], 1e-6)) * margin
    if ext[1] * S > room_height * 0.95:
        S = min(S, room_height * 0.95 / max(ext[1], 1e-6))
    c = (_all_v().min(0) + _all_v().max(0)) / 2.0
    for _, g in meshes:
        g.vertices = (g.vertices - c) * S + c

    # Centre the footprint in the room and seat the scene on the floor.
    mn = _all_v().min(0)
    mx = _all_v().max(0)
    ctr = (mn + mx) / 2.0
    room_ctr = (interior_bounds[0] + interior_bounds[1]) / 2.0
    T = np.array([room_ctr[0] - ctr[0], floor_y - mn[1], room_ctr[2] - ctr[2]])
    placed = []
    for gname, g in meshes:
        g.vertices = g.vertices + T
        placed.append((gname, g))
    return placed


def _ground_objects(
    placed: list[tuple[str, "trimesh.Trimesh"]],
    floor_y: float,
    contain_thresh: float = 0.55,
    size_ratio: float = 1.5,
) -> list[tuple[str, "trimesh.Trimesh"]]:
    """Set each object's height from floor contact + support, ignoring MIDI's
    (ungrounded) Y. Furniture rests on the floor; a small object whose XZ
    footprint sits mostly inside a larger object's rests on top of it
    (lamp -> table). Object heights themselves come from the meshes, which MIDI
    gets right — only the vertical *placement* is recomputed.
    """
    import numpy as np

    n = len(placed)
    fp = []
    for _, g in placed:
        v = g.vertices
        fp.append({"x0": v[:, 0].min(), "x1": v[:, 0].max(),
                   "z0": v[:, 2].min(), "z1": v[:, 2].max(),
                   "y0": v[:, 1].min(), "y1": v[:, 1].max()})
    for f in fp:
        f["area"] = max((f["x1"] - f["x0"]) * (f["z1"] - f["z0"]), 1e-9)
        f["h"] = f["y1"] - f["y0"]

    # supporter = the highest-topped larger object that contains this footprint
    support = [None] * n
    for i in range(n):
        best, best_top = None, -1e18
        for j in range(n):
            if i == j:
                continue
            ox = max(0.0, min(fp[i]["x1"], fp[j]["x1"]) - max(fp[i]["x0"], fp[j]["x0"]))
            oz = max(0.0, min(fp[i]["z1"], fp[j]["z1"]) - max(fp[i]["z0"], fp[j]["z0"]))
            contain = (ox * oz) / fp[i]["area"]
            # i must (a) sit mostly within j's footprint, (b) be smaller, and
            # (c) actually be ABOVE j in MIDI's relative layout — not beside it.
            # MIDI's absolute heights are unusable but its relative ordering
            # ("lamp higher than table") is the signal that separates on-top
            # from next-to.
            above = fp[i]["y0"] >= fp[j]["y0"] + 0.25 * fp[j]["h"]
            if (contain >= contain_thresh and fp[j]["area"] >= size_ratio * fp[i]["area"]
                    and above):
                if fp[j]["y1"] > best_top:
                    best, best_top = j, fp[j]["y1"]
        support[i] = best

    # resolve target base heights in dependency order (floor first, then stacks)
    base = [None] * n
    for _ in range(n + 1):
        changed = False
        for i in range(n):
            if base[i] is not None:
                continue
            j = support[i]
            if j is None:
                base[i] = floor_y
                changed = True
            elif base[j] is not None:
                base[i] = base[j] + fp[j]["h"]   # rest on supporter's top
                changed = True
        if not changed:
            break
    for i in range(n):
        if base[i] is None:        # support cycle -> floor
            base[i] = floor_y

    for i, (_, g) in enumerate(placed):
        g.vertices[:, 1] += base[i] - g.vertices[:, 1].min()
    return placed


def _unproject_mask(depth, seg, label, fx, fy, cx, cy, scale):
    """Unproject one segmentation label's pixels through metric depth + camera
    intrinsics into the Godot frame (Y-up, scaled). Returns Nx3 points or None."""
    import numpy as np
    mask = (seg == label)
    vs, us = np.where(mask)
    if len(vs) < 50:
        return None
    zs = depth[vs, us]
    valid = (zs > 0.1) & (zs < 19.0) & np.isfinite(zs)
    if valid.sum() < 50:
        return None
    u = us[valid].astype(np.float32)
    v = vs[valid].astype(np.float32)
    z = zs[valid]
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    # camera (OpenCV, y-down, z-forward) -> Godot (y-up), then scale
    return np.stack([x_cam * scale, -y_cam * scale, -z * scale], axis=-1)


def _place_depth_align(object_scene, depth, seg_path, label_list, intr, scale,
                       room_shift, floor_y, interior_bounds):
    """Gen3DSR-style per-object alignment: anchor each MIDI mesh to its own
    metric-depth point cloud in the camera frame — no floor-snap, no PCA, no
    support rules. Orientation is MIDI's native (already ~camera-facing); scale
    comes from the object's apparent in-plane size; position from matching the
    mesh's visible (front) surface to where the depth observes it.
    """
    import numpy as np
    from PIL import Image

    fx, fy, cx, cy = intr
    seg = np.array(Image.open(seg_path).convert("L"))
    H, W = depth.shape
    if seg.shape != (H, W):
        seg = np.array(Image.open(seg_path).convert("L").resize((W, H), Image.NEAREST))

    def _rng(a):                      # robust extent (10-90 percentile)
        return float(np.percentile(a, 90) - np.percentile(a, 10))

    placed = []
    for gname, geom in object_scene.geometry.items():
        g = geom.copy()
        tf = object_scene.graph.get(gname)
        if tf is not None and isinstance(tf, tuple):
            g.apply_transform(tf[0])

        idx = int(gname.split("_")[-1])
        P = (_unproject_mask(depth, seg, label_list[idx], fx, fy, cx, cy, scale)
             if idx < len(label_list) else None)
        if P is None:
            placed.append((gname, g))      # no depth evidence -> leave as-is
            continue
        P = P + room_shift                 # into the combined/room frame

        V = g.vertices
        # --- scale: match apparent in-plane size (X width, Y height: reliable) ---
        pX, pY = _rng(P[:, 0]), _rng(P[:, 1])
        vX, vY = _rng(V[:, 0]), _rng(V[:, 1])
        ratios = [r for r in (pX / vX if vX > 1e-6 else 0,
                              pY / vY if vY > 1e-6 else 0) if r > 1e-6]
        s = float(np.clip(np.median(ratios), 0.05, 50.0)) if ratios else 1.0
        c = (V.min(0) + V.max(0)) / 2.0
        V = (V - c) * s + c

        # --- position: in-plane centroid (X,Y); depth via front-face match (Z) ---
        bb = (V.min(0) + V.max(0)) / 2.0
        dx = float(np.median(P[:, 0])) - bb[0]
        dy = float(np.median(P[:, 1])) - bb[1]
        # camera looks down -Z, so the near (visible) surface is high Z
        dz = float(np.percentile(P[:, 2], 75)) - float(np.percentile(V[:, 2], 90))
        V = V + np.array([dx, dy, dz])

        # safety: nothing sinks through the floor (depth noise)
        base = V[:, 1].min()
        if base < floor_y:
            V[:, 1] += floor_y - base
        g.vertices = V
        placed.append((gname, g))

    return placed


def _write_editor_tscn(project_dir: str, name: str) -> str:
    """Write <name>.tscn next to <name>.glb: a Godot scene that statically
    instances the exported GLB so it's visible/editable in the editor (the
    runtime viewer loads GLBs by script, so they never appear in the editor tree).

    The GLB is referenced by res:// path only — no uid — because Godot assigns the
    import uid the first time it scans the file (which happens after this run). The
    scene header uid is likewise omitted; Godot fills both in on first open. The
    res:// path is the project subfolder name, matching godot_viewer/<name>/<name>.glb.
    """
    tscn = (
        "[gd_scene load_steps=2 format=3]\n\n"
        f'[ext_resource type="PackedScene" path="res://{name}/{name}.glb" id="1_glb"]\n\n'
        # Wrapper Node3D root so the GLB is a child you can right-click ->
        # "Editable Children" to hand-arrange individual objects, then save.
        f'[node name="{name}" type="Node3D"]\n\n'
        f'[node name="{name}" parent="." instance=ExtResource("1_glb")]\n')
    path = os.path.join(project_dir, f"{name}.tscn")
    with open(path, "w", newline="\n") as f:
        f.write(tscn)
    return path


def step_combine(room_scene, object_scene, name: str, scale: float,
                 project_dir: str, image_path: str,
                 surface_colors: dict | None,
                 progress: Progress,
                 *,
                 depth: "np.ndarray | None" = None,
                 cam_intrinsics: tuple | None = None,
                 seg_path: str | None = None,
                 label_list: list[int] | None = None,
                 placement: str = "depth",
                 global_yaw: float = 0.0,
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

    # Interior bounds = the INNER faces of the wall slabs. Snapping/clamping to
    # the raw AABB (outer faces) embeds objects in the ~0.4m-thick walls, so they
    # poke "outside" the visible room. Derive inner faces from the wall meshes.
    interior_min = room_bounds[0].copy()
    interior_max = room_bounds[1].copy()
    rcx, rcz = (room_bounds[0][0] + room_bounds[1][0]) / 2, (room_bounds[0][2] + room_bounds[1][2]) / 2
    for gname, geom in combined.geometry.items():
        if "wall" not in gname:
            continue
        b = geom.bounds
        if (b[1][0] - b[0][0]) < (b[1][2] - b[0][2]):   # thin in X -> side wall
            if (b[0][0] + b[1][0]) / 2 < rcx:
                interior_min[0] = max(interior_min[0], b[1][0])   # left: inner = max X
            else:
                interior_max[0] = min(interior_max[0], b[0][0])   # right: inner = min X
        else:                                            # thin in Z -> back/front wall
            if (b[0][2] + b[1][2]) / 2 < rcz:
                interior_min[2] = max(interior_min[2], b[1][2])   # back: inner = max Z
            else:
                interior_max[2] = min(interior_max[2], b[0][2])
    interior_bounds = np.array([interior_min, interior_max])

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

    if object_scene is not None and placement == "midi-fit":
        # Trust MIDI's native arrangement; fit the whole scene into the room.
        progress.detail(f"MIDI-fit placement (global yaw {global_yaw:.0f}°)")
        placed = _place_midi_fit(object_scene, interior_bounds, floor_y,
                                 room_height, yaw=np.radians(global_yaw))
        # Ground via floor-contact + support (using MIDI's relative Y as the
        # on-top signal), then separate true 3D overlaps. Grounding runs first so
        # the Y-aware overlap pass won't shove a lamp off its table.
        placed = _ground_objects(placed, floor_y)
        placed = _resolve_overlaps(placed, interior_bounds, floor_y,
                                   reseat_floor=False)
        for gname, g in placed:
            combined.add_geometry(g, geom_name=gname)
            obj_count += 1
    elif object_scene is not None and placement == "depth-align" and use_depth:
        # Gen3DSR-style: anchor each MIDI mesh to its own metric-depth points.
        progress.detail("Depth-align placement (per-object depth anchoring)")
        placed = _place_depth_align(object_scene, depth, seg_path, label_list,
                                    cam_intrinsics, scale, room_shift, floor_y,
                                    interior_bounds)
        placed = _resolve_overlaps(placed, interior_bounds, floor_y,
                                   reseat_floor=False)
        for gname, g in placed:
            combined.add_geometry(g, geom_name=gname)
            obj_count += 1
    elif object_scene is not None and use_depth:
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

        # Per-object placement using the depth map as the room-frame "map":
        #   * yaw      — align the MIDI mesh's footprint long-axis to the depth
        #                footprint's, when both are clearly elongated.
        #   * scale    — match each object's apparent in-plane size (the reliable
        #                width/height directions), clamped to a band around the
        #                global fit so a bad depth estimate can't blow it up.
        #   * position — depth centroid in XZ, base snapped to the floor.
        # global_scale stays as the anchor and as the fallback for objects with
        # no depth placement.
        ANISO_MIN = 1.6            # below this a footprint has no reliable axis
        SCALE_LO, SCALE_HI = 0.5, 2.0  # per-object scale stays within this × global

        def _yaw_matrix(theta, pivot):
            # Rotate about the vertical (Y) line through pivot (X,Z).
            c, s = np.cos(theta), np.sin(theta)
            m = np.eye(4)
            m[0, 0], m[0, 2] = c, s
            m[2, 0], m[2, 2] = -s, c
            m[0, 3] = pivot[0] - (c * pivot[0] + s * pivot[2])
            m[2, 3] = pivot[2] - (-s * pivot[0] + c * pivot[2])
            return m

        placed = []
        for gname, g in all_obj_meshes:
            idx = int(gname.split("_")[-1])
            p = placements.get(idx)

            # --- yaw: align long axis to the nearest room wall ---
            # The room is axis-aligned in this frame (walls along X / Z). A
            # per-object depth footprint is unreliable for orientation under an
            # oblique camera (its visible faces form a ~45° diagonal), but the
            # object's POSITION relative to the walls is robust: furniture near a
            # wall runs parallel to it. So pick the room axis by which wall the
            # object sits closest to, and rotate the MIDI long-axis onto it.
            midi_yaw, midi_aniso = _footprint_yaw(g.vertices)
            if p is not None and midi_aniso >= ANISO_MIN:
                ctr = p["centroid"] + room_shift
                rc = (room_bounds[0] + room_bounds[1]) / 2.0  # combined-frame center
                fx_ = abs(ctr[0] - rc[0]) / max(room_ext[0] / 2, 1e-6)
                fz_ = abs(ctr[2] - rc[2]) / max(room_ext[2] / 2, 1e-6)
                # nearer a side wall (X-displaced) -> long axis along Z (yaw 90°);
                # nearer a back/front wall -> long axis along X (yaw 0°).
                target = (np.pi / 2) if fx_ >= fz_ else 0.0
                d = target - midi_yaw
                d = (d + np.pi / 2) % np.pi - np.pi / 2  # 180°-ambiguous: smaller turn
                pivot = (g.vertices.min(axis=0) + g.vertices.max(axis=0)) / 2.0
                g.apply_transform(_yaw_matrix(d, pivot))

            # --- scale: anchor on apparent in-plane size (width X + height Y) ---
            ext = g.vertices.max(axis=0) - g.vertices.min(axis=0)
            per_scale = global_scale
            if p is not None:
                midi_inplane = float(np.hypot(ext[0], ext[1]))
                depth_inplane = float(np.hypot(p["extent_full"][0], p["extent_full"][1]))
                if midi_inplane > 1e-6 and depth_inplane > 1e-6:
                    per_scale = depth_inplane / midi_inplane
            per_scale = float(np.clip(per_scale,
                                      SCALE_LO * global_scale, SCALE_HI * global_scale))
            pivot = (g.vertices.min(axis=0) + g.vertices.max(axis=0)) / 2.0
            g.apply_transform(trimesh.transformations.scale_matrix(per_scale, pivot))

            # --- position: depth centroid in XZ, base on the floor ---
            midi_min = g.vertices.min(axis=0)
            midi_ctr = (midi_min + g.vertices.max(axis=0)) / 2.0
            place = np.eye(4)
            if p is not None:
                target_ctr = p["centroid"] + room_shift
                place[0, 3] = target_ctr[0] - midi_ctr[0]
                place[2, 3] = target_ctr[2] - midi_ctr[2]
            place[1, 3] = floor_y - midi_min[1]
            g.apply_transform(place)
            placed.append((gname, g))

        # Push overlapping objects apart, snap wall furniture flush, then resolve
        # again (snapping several objects to one wall can re-overlap them) and
        # re-snap. The second pass separates same-wall neighbours *along* the
        # wall while the final snap keeps them flush.
        placed = _resolve_overlaps(placed, interior_bounds, floor_y)
        placed = _snap_to_walls(placed, interior_bounds)
        placed = _resolve_overlaps(placed, interior_bounds, floor_y)
        placed = _snap_to_walls(placed, interior_bounds)

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

    # Editor-friendly scene: the runtime viewer (main.gd) loads the .glb via
    # script at F5, so nothing shows in the editor scene tree. This .tscn instances
    # the .glb statically so opening it in Godot shows the assembled room+objects,
    # and (with Editable Children) lets you hand-arrange objects and save.
    _write_editor_tscn(project_dir, name)

    # Keep the source image and surface colors alongside the GLB for reference
    src_copy = os.path.join(project_dir, "source" + Path(image_path).suffix)
    shutil.copy2(image_path, src_copy)
    if surface_colors:
        with open(os.path.join(project_dir, "surfaces.json"), "w") as f:
            json.dump(surface_colors, f, indent=2)

    progress.done(f"{room_count} room surfaces + {obj_count} objects -> "
                  f"{name}.glb + {name}.tscn ({mb:.1f} MB)")
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
    ap.add_argument("--placement", choices=["depth", "midi-fit", "depth-align"],
                    default="depth",
                    help="object placement: 'depth' reconstructs each object's "
                         "pose from the depth map; 'midi-fit' trusts MIDI's "
                         "single-batch arrangement and fits the whole scene to "
                         "the room (use with --batch-size >= object count)")
    ap.add_argument("--global-yaw", type=float, default=0.0,
                    help="midi-fit: rotate the whole MIDI scene by this many "
                         "degrees about vertical to align it to the room")

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

    ap.add_argument("--texture", action="store_true",
                    help="texture each MIDI object with MV-Adapter before "
                         "placement (runs in the MIDI venv; ~2-3 min/object). "
                         "Best paired with --placement depth-align.")
    ap.add_argument("--retexture", action="store_true",
                    help="re-run texturing even if cached mesh_<i>_shaded.glb exist")
    ap.add_argument("--texture-seed", type=int, default=42,
                    help="seed for MV-Adapter texturing (default 42)")
    ap.add_argument("--vcvars",
                    default=r"C:\Program Files (x86)\Microsoft Visual Studio\2022"
                            r"\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
                    help="VS Build Tools vcvars64.bat (for CUDA-kernel compile)")
    ap.add_argument("--cuda-bin",
                    default=r"C:\Program Files\NVIDIA GPU Computing Toolkit"
                            r"\CUDA\v12.8\bin",
                    help="CUDA toolkit bin dir with nvcc (cu128 for Blackwell)")

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
    do_texture = bool(args.texture and env["midi"] and (do_midi_run or do_midi_load))
    if do_texture:
        steps.append("Texture objects (MV-Adapter)")
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
    untextured_glb = None   # the merged, pre-decimation MIDI GLB (texturing input)
    label_map_path = None
    midi_out_dir = None

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
                untextured_glb = glb_path
                label_map_path = os.path.join(midi_out_dir, "object_label_map.json")
                object_scene = step_process_objects(
                    glb_path, args.target_faces, progress)
    elif do_midi_load:
        object_scene = step_process_objects(
            args.midi_output, args.target_faces, progress)
        untextured_glb = args.midi_output
        # Try to recover label mapping and seg path from sidecar / --seg
        midi_out_dir = os.path.dirname(os.path.abspath(args.midi_output))
        map_file = os.path.join(midi_out_dir, "object_label_map.json")
        if os.path.isfile(map_file):
            import json as _json
            label_list = _json.load(open(map_file))["labels"]
            label_map_path = map_file
        seg_candidate = os.path.join(midi_out_dir, "segmentation.png")
        midi_seg_path = args.seg or (seg_candidate
                                      if os.path.isfile(seg_candidate) else None)

    # --- Texture objects (optional) ---
    if do_texture and object_scene is not None:
        if untextured_glb and label_list and midi_seg_path:
            if args.placement != "depth-align":
                print("  NOTE: --texture re-centers/normalizes each mesh; "
                      "--placement depth-align is recommended (re-anchors per "
                      "object from depth). Current placement: "
                      f"{args.placement}.")
            tex_dir = step_texture(
                args.image, midi_seg_path, untextured_glb, label_map_path,
                env["midi_dir"], os.path.join(midi_out_dir, "tex"),
                len(label_list), args.vcvars, args.cuda_bin,
                args.texture_seed, args.retexture, progress)
            if tex_dir:
                textured = _load_textured_scene(tex_dir, len(label_list), progress)
                if textured is not None:
                    object_scene = textured
        else:
            progress.begin("Texture objects (MV-Adapter)")
            progress.detail("Need merged MIDI GLB + label map + seg mask to "
                            "texture; one is missing")
            progress.done("Skipped — using untextured objects")
    elif args.texture and not env["midi"]:
        print("  NOTE: --texture needs the MIDI-3D repo/venv; not found. Skipping.")

    # --- Final: Combine ---
    out_path = step_combine(
        room_scene, object_scene, args.name, args.scale,
        project_dir, args.image, surface_colors, progress,
        depth=depth,
        cam_intrinsics=(fx, fy, cx, cy),
        seg_path=midi_seg_path,
        label_list=label_list,
        placement=args.placement,
        global_yaw=args.global_yaw,
    )

    progress.finish(
        f"Open {args.viewer_dir}/ in Godot and press F5.\n"
        f"Project folder: {project_dir}\n"
        f"Output: {out_path}")


if __name__ == "__main__":
    main()
