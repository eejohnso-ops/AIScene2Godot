# AIScene2Godot

Turn AI-generated 3D scenes into **walkable Godot scenes** — with per-object
collision, in one command.

Tools like [MIDI](https://github.com/VAST-AI-Research/MIDI-3D), TRELLIS, and
Hunyuan3D can generate 3D from a single image, but their output is a raw `.glb`
sitting in a folder. AIScene2Godot is the missing last step: it makes that output
**game-ready** (decimated, scaled, per-object collision) and drops you *inside*
it to fly or walk around in seconds.

It also documents — in painful, hard-won detail — how to actually get these
models running on **Windows + an NVIDIA Blackwell (RTX 50-series) GPU**, which is
its own gauntlet (see [the setup guide](docs/midi-windows-blackwell-setup.md)).

---

## What's here

| Piece | What it is |
| --- | --- |
| **`build_scene.py`** | **The main entry point.** One command: photo → segmentation → depth → room shell → objects → walkable Godot scene. Shows step-by-step progress with elapsed time. |
| **`setup_env.py`** | Automated environment setup. Handles the entire dependency gauntlet (torch, MIDI, nvdiffrast, patches, checkpoints). Run with `--full` for everything. |
| **`godot_viewer/`** | A minimal Godot 4 project that auto-loads the newest `.glb` in its folder, builds per-object collision, and drops you in with fly + first-person walk modes. The hero. |
| **`to_godot.py`** | Standalone: takes an AI scene `.glb`, decimates/scales it, and writes it into the viewer. Keeps objects separate so each gets its own collision. |
| **`room_from_image.py`** | Standalone: room photo → depth-displaced room shell with procedural textures. Now with subdivided walls that capture architectural relief from the depth map. |
| **`build_dwelling.py`** | Compose **several rooms into one walkable dwelling** from a floor-plan spec: thick deduped walls + sized doorways (box mode), or full depth-reconstructed room shells dropped into slots (`--reconstruct`). See [Multi-room dwellings](#multi-room-dwellings). |
| **`segment_room.py`** | Standalone: SegFormer surface segmentation for wall/floor/ceiling color extraction. |
| **`docs/midi-windows-blackwell-setup.md`** | The install guide for getting MIDI (+ MV-Adapter texturing) working on Windows/Blackwell. Every wall, every fix. |
| **`experimental/`** | An honest dead-end: a single-panorama → depth → mesh pipeline. It produces a *shell*, not a usable scene — and the writeup explains exactly why. Kept because the lesson is the point. |

---

## Quickstart

### One-command pipeline (recommended)

```bash
python setup_env.py              # install core deps (once)
python build_scene.py photo.jpg  # room photo -> walkable Godot scene
```

`build_scene.py` chains segmentation → depth → room shell → export, with
step-by-step progress and elapsed time. Open `godot_viewer/` in Godot 4 and
press **F5**.

For MIDI 3D objects too:
```bash
python setup_env.py --full                                    # install everything (once)
python build_scene.py photo.jpg --midi-output path/to/output.glb  # room + objects
```

For **textured** objects, add `--texture` (best with `--placement depth-align`):
```bash
python build_scene.py photo.jpg --midi-output path/to/output.glb \
    --placement depth-align --texture
```
`--texture` runs MV-Adapter per object (~2-3 min each) in the MIDI venv, inside
the VS Build Tools env so the triton/nvdiffrast CUDA kernels can compile — paths
default to the standard install but are overridable with `--vcvars`/`--cuda-bin`.
Per-object results (`mesh_<i>_shaded.glb`) are cached under `<midi-out>/tex/` and
reused on reruns; pass `--retexture` to force a rebuild.

### Manual workflow (individual scripts)

1. **Generate a scene** with [MIDI](https://github.com/VAST-AI-Research/MIDI-3D)
   from a single image — you get an `output.glb` of separate object meshes.

2. **Make it game-ready and load it:**
   ```bash
   python to_godot.py path/to/output.glb --name my_scene
   ```

3. **Walk it.** Open `godot_viewer/` in Godot 4 and press **F5**. Controls:
   mouse to look, **WASD** to move, **Space/Shift** up/down (fly), **F** to
   toggle fly/walk, **Esc** to free the cursor.

That's it — image to walkable, textured, per-object scene.

---

## Multi-room dwellings

`build_dwelling.py` composes **several rooms into one walkable scene** from a
floor-plan spec (rooms with position/yaw/size + a door graph). It's the path from
single rooms toward a complete dwelling. Full design + roadmap:
[`docs/multi-room-dwelling.md`](docs/multi-room-dwelling.md).

```bash
python build_dwelling.py examples/dwelling_two_room.json          # box-shell dwelling
python build_dwelling.py examples/dwelling_reconstruct.json --reconstruct   # real reconstructed rooms
```
Then open `godot_viewer/` and press **F5** — it loads the newest dwelling folder.

Two composition modes, one spec:

- **Box mode (default).** Each room is a clean box with **thick, deduped walls**
  (shared walls built once, no z-fighting) and **sized doorways** cut where the
  spec's door graph says. Rooms **tile and connect cleanly**. No GPU needed —
  geometry is the spec; the photo only *dresses* surfaces.
- **Reconstruct mode (`--reconstruct`).** Each room with an `image` is built as a
  **full depth-reconstructed shell** (textured + depth relief) placed into its
  slot, normalized to a shared ceiling height. You get real per-room geometry, but
  each room keeps its **depth-derived footprint**, so rooms **don't tile/connect
  cleanly** — you hand-arrange the join in Godot. (Needs the DepthAnything
  checkpoint + GPU; missing/failed reconstruction falls back to a box shell.)

**Dressing.** Per-room wall/floor/ceiling colours resolve from (priority) explicit
`colors` → a precomputed `surfaces` JSON (from `segment_room.py`) → live
segmentation of the room's `image` (`--segment`) → defaults. The shell stays
spec-driven; the photo tints it.

See `examples/dwelling_two_room.json` and `examples/dwelling_reconstruct.json`.

---

## The Godot viewer

`godot_viewer/` is intentionally tiny and reusable. On launch it:

- **Auto-loads the newest `.glb`** in the project folder — no scene editing, just
  drop a file in and press F5.
- **Builds per-object trimesh collision** so each mesh is solid and individually
  collidable.
- **Forces materials unshaded + double-sided** so baked textures show correctly
  and you can see surfaces from inside.
- Adds a **ground plane**, a sun, and a soft sky so objects aren't floating in
  void (handy because object-generators like MIDI produce furniture, not rooms).
- Gives you **fly (noclip)** and **first-person walk** modes.

Point `to_godot.py --viewer-dir` at any Godot project to reuse the loader
elsewhere.

### Editing in the Godot editor (hand-arrange)

The viewer loads GLBs **at runtime by script**, so they don't appear in the
editor's scene tree (only when you press F5). For that reason each
`build_scene.py` run also writes `godot_viewer/<name>/<name>.tscn` — a scene that
statically instances the GLB. **Double-click that `.tscn`** in Godot's FileSystem
dock to see the assembled room + objects in the editor. To hand-arrange: select
the instanced child, right-click → **Editable Children**, move objects with the
gizmo, and save the `.tscn`. (Opening the raw `.glb` instead shows a read-only
preview.)

---

## What works, and what doesn't

This project is the residue of a long exploration. Being honest about the
boundary is the most useful thing it can offer:

- ✅ **Object-generation → Godot (MIDI):** clean, complete, *separate* textured
  meshes. This is the path that works, because each object is generated *whole* —
  the occluded back of a chair is invented, not reconstructed.
- ❌ **Single-panorama → depth → mesh (`experimental/`):** produces a 2.5D shell
  with tears (per-face cubemap depth) or a sphere (native-360 depth). One
  viewpoint never contains the occluded geometry, so depth estimation can only
  ever recover a shell. [Full writeup + evidence.](docs/panorama-experiments.md)

The takeaway: **object generators give you props; they don't give you rooms.** A
full game scene is "generated objects + a shell you build or generate separately."

---

## Requirements

- **Godot 4.x** (developed on 4.6) for the viewer.
- **Python 3.10+** with `trimesh`, `numpy`, and `fast-simplification` for
  `to_godot.py`.
- The actual generative models (MIDI, etc.) are **not** included — see
  [`THIRD_PARTY.md`](THIRD_PARTY.md). Getting them running on Windows/Blackwell is
  covered in [the setup guide](docs/midi-windows-blackwell-setup.md).

## License

The code in this repo (viewer, `to_godot.py`, scripts) is **MIT** — see
[`LICENSE`](LICENSE). The models it orchestrates have their own, often
**non-commercial**, licenses; this repo bundles none of them. See
[`THIRD_PARTY.md`](THIRD_PARTY.md) before using any output commercially.

## Credits

Built on the work of others — MIDI (VAST-AI), MV-Adapter, nvdiffrast (NVIDIA),
DiT360, Depth Anywhere, MoGe, and the Godot engine. This repo is glue and notes.
