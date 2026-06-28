#!/usr/bin/env python3
"""
batch_midi.py -- run MIDI in batches to avoid VRAM exhaustion.

Splits a multi-object segmentation mask into batches of N objects, runs MIDI
on each batch separately, then merges the output GLBs into one scene.

    python batch_midi.py --rgb photo.png --seg segmentation.png --batch-size 4

The segmentation mask is a palette/grayscale PNG where each unique nonzero
pixel value is a separate object instance.
"""
import argparse
import os
import sys
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image


def split_seg_mask(seg_path: str, min_pixels: int = 500) -> tuple[list[int], np.ndarray]:
    """Return label IDs with at least min_pixels coverage, and the full mask."""
    seg = np.array(Image.open(seg_path).convert("L"))
    all_labels = sorted(set(np.unique(seg)) - {0})
    labels = [l for l in all_labels if int((seg == l).sum()) >= min_pixels]
    skipped = len(all_labels) - len(labels)
    if skipped:
        print(f"  Filtered out {skipped} tiny objects (< {min_pixels} px)")
    return labels, seg


def create_batch_mask(seg: np.ndarray, labels: list[int], out_path: str):
    """Write a new seg mask containing only the given labels, renumbered 1..N."""
    batch_seg = np.zeros_like(seg, dtype=np.uint8)
    for new_id, old_id in enumerate(labels, start=1):
        batch_seg[seg == old_id] = new_id
    img = Image.fromarray(batch_seg, mode="L")
    img.save(out_path)
    return out_path


def _midi_python(midi_dir: str) -> str:
    """Prefer MIDI's own venv interpreter — its deps (gradio 4.x, transformers
    4.49) conflict with the main pipeline env, so MIDI gets a dedicated venv.
    Falls back to the current interpreter if that venv isn't present."""
    for rel in (os.path.join(".venv", "Scripts", "python.exe"),   # Windows
                os.path.join(".venv", "bin", "python")):          # POSIX
        cand = os.path.join(midi_dir, rel)
        if os.path.isfile(cand):
            return cand
    return sys.executable


def run_midi_batch(rgb_path: str, seg_path: str, output_dir: str,
                   midi_dir: str, batch_idx: int, seed: int = 42) -> str | None:
    """Run MIDI inference on one batch. Returns the output GLB path or None."""
    # inference_midi.py loads weights from the *relative* path
    # "pretrained_weights/MIDI-3D", so it must run with cwd=midi_dir. Because cwd
    # changes, pass all I/O paths as absolute.
    rgb_path = os.path.abspath(rgb_path)
    seg_path = os.path.abspath(seg_path)
    output_dir = os.path.abspath(output_dir)
    out_glb = os.path.join(output_dir, f"batch_{batch_idx}.glb")

    env = os.environ.copy()
    env["PYTHONPATH"] = midi_dir

    py = _midi_python(midi_dir)
    cmd = [
        py, os.path.join(midi_dir, "scripts", "inference_midi.py"),
        "--rgb", rgb_path,
        "--seg", seg_path,
        "--output-dir", output_dir,
        "--seed", str(seed),
    ]

    result = subprocess.run(cmd, env=env, cwd=midi_dir)

    # MIDI always writes "output.glb"; rename to batch-specific name
    raw = os.path.join(output_dir, "output.glb")
    if os.path.isfile(raw):
        os.replace(raw, out_glb)
        return out_glb
    return None


def merge_glbs(glb_paths: list[str], out_path: str, scale: float = 1.0):
    """Merge multiple MIDI GLBs into one scene."""
    import trimesh

    combined = trimesh.Scene()
    obj_idx = 0
    for glb in glb_paths:
        scene = trimesh.load(glb, process=False)
        if not isinstance(scene, trimesh.Scene):
            scene = trimesh.Scene(scene)
        for name, geom in scene.geometry.items():
            combined.add_geometry(geom, geom_name=f"object_{obj_idx}")
            obj_idx += 1

    if scale != 1.0:
        combined.apply_transform(np.diag([scale, scale, scale, 1.0]))

    combined.export(out_path)
    return obj_idx


def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.1f}m"


def main():
    ap = argparse.ArgumentParser(description="Run MIDI in VRAM-safe batches")
    ap.add_argument("--rgb", required=True, help="source room image")
    ap.add_argument("--seg", required=True, help="instance segmentation mask")
    ap.add_argument("--output-dir", default=".", help="output directory")
    ap.add_argument("--midi-dir", default=None,
                    help="MIDI-3D repo path (auto-detected)")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="objects per MIDI batch (default 4)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="output",
                    help="output GLB base name")
    args = ap.parse_args()

    if args.midi_dir is None:
        for candidate in [
            Path(__file__).resolve().parent.parent / "external" / "MIDI-3D",
            Path.home() / "projects" / "MIDI-3D",
            Path(__file__).resolve().parent.parent.parent / "MIDI-3D",
        ]:
            if candidate.is_dir() and (candidate / "scripts").is_dir():
                args.midi_dir = str(candidate)
                break
    if not args.midi_dir:
        sys.exit("MIDI-3D repo not found. Pass --midi-dir.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Split segmentation into batches
    labels, seg = split_seg_mask(args.seg)
    n_objects = len(labels)
    batches = [labels[i:i + args.batch_size]
               for i in range(0, len(labels), args.batch_size)]

    print(f"  {n_objects} objects -> {len(batches)} batch(es) "
          f"of up to {args.batch_size}")
    print(f"  Labels: {labels}")

    t0 = time.time()
    glb_paths = []

    for i, batch_labels in enumerate(batches):
        batch_t0 = time.time()
        print(f"\n{'=' * 60}")
        print(f"  Batch {i + 1}/{len(batches)}: "
              f"{len(batch_labels)} objects (labels {batch_labels})")
        print(f"{'=' * 60}")

        # Create batch-specific mask
        batch_seg_path = os.path.join(args.output_dir, f"seg_batch_{i}.png")
        create_batch_mask(seg, batch_labels, batch_seg_path)

        glb = run_midi_batch(
            args.rgb, batch_seg_path, args.output_dir,
            args.midi_dir, i, args.seed)

        if glb:
            mb = os.path.getsize(glb) / 1e6
            print(f"  Batch {i + 1} done: {mb:.1f} MB "
                  f"({fmt_time(time.time() - batch_t0)})")
            glb_paths.append(glb)
        else:
            print(f"  Batch {i + 1} FAILED "
                  f"({fmt_time(time.time() - batch_t0)})")

    if not glb_paths:
        sys.exit("All batches failed.")

    # Merge
    print(f"\nMerging {len(glb_paths)} batch(es)...")
    out_path = os.path.join(args.output_dir, f"{args.name}.glb")
    n_objs = merge_glbs(glb_paths, out_path)
    mb = os.path.getsize(out_path) / 1e6
    print(f"Done: {n_objs} objects -> {out_path} ({mb:.1f} MB)")
    print(f"Total: {fmt_time(time.time() - t0)}")


if __name__ == "__main__":
    main()
