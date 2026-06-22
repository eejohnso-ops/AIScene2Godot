# Experiments: single-panorama → 3D scene (and why it tops out)

Before landing on object generation (MIDI), this project spent a long time on the
intuitive idea: **generate a 360° panorama, estimate depth, unproject to a mesh.**
It doesn't produce a usable scene. This is the record of *why* — the lesson is the
deliverable. The code lives in `experimental/`.

## The pipeline we built

`experimental/build_room.py` does: prompt → ComfyUI 360 panorama → depth →
unproject → Poisson mesh → textured GLB. Two depth routes were tried.

### Route A — per-face cubemap + MoGe-2

Slice the equirect into 6 cube faces, run [MoGe-2](https://github.com/microsoft/MoGe)
per face, fuse. Findings:

- **Geometry is sharp but tears at the seams.** Each face gets an *independent*
  depth estimate, so adjacent faces don't agree at their shared edges — you get
  gaps and floating/doubled surfaces (especially the ceiling vs. wall faces).
- **Per-face focal mismatch shears the room.** MoGe estimates its own focal length
  per face; disagreement between faces fused into a 47 m smear from a ~10 m room.
  Re-unprojecting with the *known* 90° cubemap intrinsics fixed the shear but not
  the seams.
- Open doorways/windows read as near-infinite depth → blowouts.

### Route B — native-360 depth (UniFuse via Depth Anywhere)

One depth map for the whole equirect → one coherent point cloud, no face seams.

- **Seams gone**, but **the room collapses to a sphere.** UniFuse's depth on a
  generated panorama was nearly constant (~3 m in every direction), so every
  direction unprojects to roughly the same radius — a ball, not a room. It
  captured the scene's *scale* but almost none of its *shape*.

## The actual reason it can't work

Both routes are **single-viewpoint**. A panorama — flat image or full 360° —
only contains the surfaces visible *from one point*. The geometry behind objects,
through doorways, the far side of anything: that data was never captured, so no
depth model can recover it. You can only ever get a 2.5D **shell**.

The "interpolate the missing parts" step — generating the occluded geometry — is
a *generative* problem, not a depth problem. That's exactly what the methods we
moved toward do:

- **Object generators** (MIDI, TRELLIS, Hunyuan3D) generate each object *whole*,
  hallucinating its unseen sides. → clean, separate meshes. **This is what works**,
  and what `to_godot.py` + the viewer are built around.
- **Holistic scene generators** with an inpainting loop (Text2Room, SceneScape)
  or panorama→splat methods (DreamScene360, WonderWorld) generate the occluded
  regions too — but mostly output Gaussian splats, not game meshes.

## Still useful from this branch

- The **DiT360 panorama generator** (`experimental/workflows/panorama_api.json`,
  via [ComfyUI-DiT360Plus](https://github.com/thomashollier/ComfyUI-DiT360Plus))
  makes genuinely good seamless equirects — great for **skyboxes/backdrops**, even
  though they're a poor source for geometry.
- `experimental/da360_depth.py` — pulls *float* depth out of Depth Anywhere's
  UniFuse (its own script saves lossy 8-bit), if you want to play with 360 depth.

Run `build_room.py --help` if you want to reproduce the shells. Just know going in
that the output is a look-around shell, not a walkable scene.
