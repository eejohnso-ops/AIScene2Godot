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
| **`godot_viewer/`** | A minimal Godot 4 project that auto-loads the newest `.glb` in its folder, builds per-object collision, and drops you in with fly + first-person walk modes. The hero. |
| **`to_godot.py`** | One command: takes an AI scene `.glb`, decimates/scales it, and writes it into the viewer. Keeps objects separate so each gets its own collision. |
| **`docs/midi-windows-blackwell-setup.md`** | The install guide for getting MIDI (+ MV-Adapter texturing) working on Windows/Blackwell. Every wall, every fix. |
| **`experimental/`** | An honest dead-end: a single-panorama → depth → mesh pipeline. It produces a *shell*, not a usable scene — and the writeup explains exactly why. Kept because the lesson is the point. |

---

## Quickstart (the MIDI workflow)

1. **Generate a scene** with [MIDI](https://github.com/VAST-AI-Research/MIDI-3D)
   from a single image — you get an `output.glb` of separate object meshes.
   (Optionally run MIDI's textured pipeline for `textured_scene.glb`.)

2. **Make it game-ready and load it:**
   ```bash
   pip install trimesh numpy fast-simplification
   python to_godot.py path/to/output.glb --name my_scene
   ```
   Untextured meshes get decimated to a game budget; textured scenes are scaled
   only (so UVs survive). The result lands in `godot_viewer/`.

3. **Walk it.** Open `godot_viewer/` in Godot 4 and press **F5**. It auto-loads
   the newest `.glb`. Controls: mouse to look, **WASD** to move, **Space/Shift**
   up/down (fly), **F** to toggle fly/walk, **Esc** to free the cursor.

That's it — image to walkable, textured, per-object scene.

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
