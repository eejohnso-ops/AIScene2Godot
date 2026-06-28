#!/usr/bin/env python3
"""
to_godot.py -- turn an AI-generated 3D scene GLB into a game-ready scene the
Godot viewer can walk through.

Works with the output of MIDI (and any image->3D tool that emits a multi-mesh
GLB scene). It:
  * keeps each object a SEPARATE mesh (so you get per-object collision in Godot),
  * decimates heavy untextured geometry to a game budget (raw diffusion meshes
    can be millions of faces) -- textured scenes are left alone so UVs survive,
  * scales the (usually unit-normalized) scene up to a walkable size,
  * writes the result into godot_viewer/ where the viewer auto-loads the newest .glb.

Usage:
    python to_godot.py path/to/output.glb
    python to_godot.py output_tex/textured_scene.glb --scale 3 --name living_room

Deps:  pip install trimesh numpy
       pip install fast-simplification   # only needed to decimate untextured meshes
"""
import argparse
import os
import sys

import numpy as np
import trimesh


def _is_textured(geom) -> bool:
    mat = getattr(geom.visual, "material", None)
    return mat is not None and getattr(mat, "baseColorTexture", None) is not None


def main() -> None:
    ap = argparse.ArgumentParser(description="AI scene GLB -> walkable Godot scene")
    ap.add_argument("glb", help="input scene GLB (e.g. MIDI output.glb)")
    ap.add_argument("--scale", type=float, default=3.0,
                    help="uniform scale up (MIDI output is ~unit-cube). Default 3.")
    ap.add_argument("--scale-xyz", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="non-uniform scale (overrides --scale)")
    ap.add_argument("--target-faces", type=int, default=12000,
                    help="per-object face budget when decimating UNtextured meshes")
    ap.add_argument("--viewer-dir", default=os.path.join(os.path.dirname(__file__),
                                                         "godot_viewer"),
                    help="Godot project dir to drop the result into")
    ap.add_argument("--name", default=None,
                    help="project name (creates a subfolder; prompts if omitted)")
    args = ap.parse_args()

    if args.name is None:
        default = os.path.splitext(os.path.basename(args.glb))[0]
        try:
            answer = input(f"Project name [{default}]: ").strip()
        except EOFError:
            answer = ""
        args.name = answer if answer else default

    scene = trimesh.load(args.glb, process=False)
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(scene)
    meshes = scene.dump(concatenate=False)
    any_textured = any(_is_textured(m) for m in meshes)

    scale_vec = np.array(args.scale_xyz if args.scale_xyz else [args.scale] * 3,
                         dtype=np.float32)

    if any_textured:
        print(f"{len(meshes)} objects, textured -> scaling {scale_vec}, no decimation")
        scene.apply_transform(np.diag([*scale_vec, 1.0]))
        out_scene = scene
    else:
        try:
            import fast_simplification as fs
        except ImportError:
            sys.exit("Untextured meshes need decimation: pip install fast-simplification")
        out_scene = trimesh.Scene()
        before = after = 0
        for i, g in enumerate(meshes):
            V = np.asarray(g.vertices, np.float32)
            F = np.asarray(g.faces, np.int32)
            before += len(F)
            if len(F) > args.target_faces:
                reduction = 1.0 - args.target_faces / len(F)
                V, F = fs.simplify(V, F, target_reduction=reduction)
            after += len(F)
            out_scene.add_geometry(
                trimesh.Trimesh(vertices=V * scale_vec, faces=F, process=False),
                geom_name=f"object_{i}")
        print(f"{len(meshes)} objects, untextured -> decimated {before} -> {after} "
              f"faces, scaled {scale_vec}")

    # Shift objects so their bottom sits on Y=0 (the floor)
    bounds = out_scene.bounds
    if bounds is not None:
        shift = np.eye(4)
        shift[1, 3] = -bounds[0][1]
        out_scene.apply_transform(shift)

    project_dir = os.path.join(args.viewer_dir, args.name)
    os.makedirs(project_dir, exist_ok=True)
    out_path = os.path.join(project_dir, f"{args.name}.glb")
    out_scene.export(out_path)
    mb = os.path.getsize(out_path) / 1e6
    print(f"\nwrote {out_path}  ({mb:.1f} MB)")
    print(f"Project folder: {project_dir}")
    print("Open the Godot project in godot_viewer/ and press F5 -- it loads the "
          "newest project automatically.")


if __name__ == "__main__":
    main()
