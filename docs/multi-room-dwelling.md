# Multi-room dwellings — design doc

**Status:** Phases 1–2 prototyped; Phase 3 proposed.
**Goal:** go from single-room scenes to a complete, walkable **dwelling** —
several rooms connected by doorways in one Godot scene.

This doc captures the architecture, a phased build plan, and an honest read on
the more speculative idea of **generating the dwelling's blueprint with
diffusion** and building rooms from it.

---

## 1. The core insight

The jump from "one room" to "a dwelling" is **not** a reconstruction problem —
it's a **composition** problem layered on the per-room pipeline that already
works. Two facts from the current code drive the whole design:

- `room_from_image.py` builds a room as a **box** (floor, ceiling, 3 walls; the
  camera-side wall is omitted) from the metric-indoor depth point cloud, then
  `export_room()` **centers it at origin, floor at Y=0**. Every room lives in its
  own coordinate frame, all stacked on the same spot.
- `godot_viewer/` **auto-loads every `.glb`** in its folder and builds per-object
  collision. Walking already works; it just doesn't know where rooms *go*.

So a dwelling needs three things the pipeline lacks today:

1. A **layout authority** — something that says where each room sits on a shared
   ground plane (position + yaw), instead of every room at origin.
2. **Scale consistency** across rooms (see §4 — this is the silent killer).
3. **Doorways** — openings cut in shared walls so you can walk between rooms.

### The decision that drives everything: separate *shell* from *dressing*

A single photo sees ~one corner of a room. You **cannot** reliably recover a
multi-room footprint from photos alone — that is the same "no external layout
signal" limit already documented for MIDI object placement, now at room scale.
So don't fight it:

> **Make a floor-plan spec the authority. Demote photos to interior dressing —
> textures, architectural relief, and furniture — not global geometry.**

The spec dictates each room's rectangle and where it sits; the photo fills it in.
This is how human level designers work, and it sidesteps the unsolved problem
instead of fighting it.

---

## 2. The floor-plan spec

A small declarative file is the single source of truth for layout. Rooms carry a
position and yaw on a shared plane; doors are edges in a room-adjacency graph.

```jsonc
{
  "ceiling_height": 2.6,            // global; used to normalize per-room scale (§4)
  "rooms": [
    { "id": "living",  "image": "living.jpg",  "pos": [0, 0], "yaw": 0, "size": [5, 4] },
    { "id": "kitchen", "image": "kitchen.jpg", "pos": [5, 0], "yaw": 0, "size": [3, 4] },
    { "id": "hall",    "image": "hall.jpg",    "pos": [0, 4], "yaw": 90, "size": [8, 1.5] }
  ],
  "doors": [
    { "between": ["living", "kitchen"], "width": 0.9 },
    { "between": ["living", "hall"],    "width": 0.9 }
  ]
}
```

- `pos` / `yaw` / `size` are in **metres** on the shared XZ ground plane, floor at
  Y=0. `size` is `[width, depth]`; height comes from `ceiling_height`.
- `image` is the per-room interior photo (user-supplied **or** generated — §5).
- `doors` are an adjacency graph; the compositor cuts the openings.
- `size` is authoritative; the depth-reconstructed box is only a *hint*.

Everything below is about (a) producing this spec and (b) turning it into a
walkable scene.

---

## 3. Build plan (phased, tied to existing code)

### Phase 1 — Composition (no new ML; this is the real unlock) ✅ prototyped

Done (`room_from_image.py` + `build_dwelling.py`):
- Refactored the hard-coded "center at origin" in `export_room()` into a shared
  `place_room()` / `Placement` helper, so a room can be positioned at its
  `pos`/`yaw` (with optional ceiling-height normalization) and the floor stays at
  Y=0. The single-room path is unchanged when no `placement` is passed.
- New `build_dwelling.py` reads a floor-plan spec and builds a textured **box
  shell** per room — geometry from the spec, **no GPU/checkpoints/photos needed**,
  so the compositor is testable on its own. It bakes each room's transform into
  the vertices, opens the shared wall for every door, and writes one combined GLB
  into `godot_viewer/<name>/` plus a composite `.tscn` for editor hand-tuning.
- The viewer's existing auto-loader places it with **zero viewer changes**; the
  GLB filename contains "room" so the catch-floor is skipped.

Try it:
```bash
python build_dwelling.py examples/dwelling_two_room.json
# -> godot_viewer/twobr/twobr_rooms.glb ; open godot_viewer/ in Godot 4, press F5
```

**Outcome:** you can already walk through several adjacent rooms. Per-object
trimesh collision means walking Just Works.

### Phase 2 — Doorways and walls ✅ prototyped

Done (`build_dwelling.py`):
- **Wall thickness.** Walls are now solid prisms (`wall_thickness`, default
  0.12 m) centred on the boundary line. Shared edges are **deduped**
  (`wall_segments()`): two rooms sharing an identical edge collapse to one wall,
  so adjacent rooms no longer z-fight. Floors/ceilings stay full-footprint quads.
- **Sized doorways.** `attach_doorways()` resolves each door to a rectangular
  opening (`width` 0.9, `height` 2.1, `offset` 0) clamped to leave jambs, and
  `build_wall_meshes()` splits the wall into side panels + a lintel above the
  opening — no boolean-subtract dependency. Verified on the example: a 0.9×2.1 m
  opening through the 0.12 m shared wall, jambs + lintel, clear and walkable.
- Walls are textured prisms (procedural wall texture on every face), so the
  Phase-1 interior look is preserved.

**Known Phase 2 limits (handed to Phase 3):** clean dedup only where rooms share
a *full* edge of equal length (partial/unequal-depth overlaps may double-wall);
rooms still axis-aligned (yaw a multiple of 90); no door frame/threshold trim; no
furniture/relief yet (shell only).

- Optional next: bake a Godot **navmesh** across the whole dwelling so AI/teleport
  and "walk" mode respect the full footprint. (This is also the point where a
  Godot run-and-capture-errors loop / MCP starts to earn its keep — see the MCP
  note in project history.)

### Dressing tier 1 — per-surface colours ✅ prototyped

Done (`build_dwelling.py` `dress_rooms()`): the shell stays spec-driven; the photo
only **tints** it. Each room resolves wall/floor/ceiling colours, highest priority
first, from: explicit `colors` → a precomputed `surfaces` JSON (from
`segment_room.py`) → live segmentation of the room's `image` (`--segment`, results
cached per room as `<project>/<room_id>_surfaces.json`) → defaults. Segmentation is
opt-in and wrapped so the no-model path stays CPU-only and never breaks the build.
Verified: a room dressed from a surfaces JSON exports walls whose texture mean
matches the JSON colour, distinct from a sibling room's.

This is the faithful "shell vs dressing" wire-in: it keeps Phase 2's tiling +
doorways intact (structure from spec) while pulling real surface colour from the
photo. **MIDI furniture** per room slots in via the same world transform (place an
existing object GLB at the room centre, floor Y=0) — cheap to add next.

### Reconstruction into slots (Level-A) ✅ wired

The alternative to spec geometry: drop a **full depth-reconstructed room shell**
(textured + depth-displaced relief) into each slot, instead of a flat box. Done:
- `room_from_image.py` factors the photo→shell pipeline into a reusable
  `reconstruct_room_scene()` (camera frame); `main()` now calls it too.
- `build_dwelling.py --reconstruct`: any room with an `image` is reconstructed and
  placed via `place_room(Placement(pos, yaw, target_height=ceiling_height))`.
  `_place_and_merge()` scales → flips camera→Godot → places → **bakes graph
  transforms into vertices** (so meshes survive the merge and stay separate for
  per-mesh collision). Needs the DepthAnything checkpoint + GPU; missing/failed
  reconstruction warns and falls back to a box shell per room.
- Verified non-GPU: the place+merge math lands a synthetic room with floor at Y=0,
  height normalized to the shared ceiling, centred on its `pos`, aspect preserved.
- **GPU-validated end-to-end** (`examples/dwelling_reconstruct.json --reconstruct`):
  two rooms reconstructed with depth relief (walls displaced ~13–14 cm), no
  fallback, both normalized to a 2.60 m ceiling with floors welded to Y=0 and
  placed at their slots (room_b at X=6).

**The accepted trade:** reconstructed rooms keep their own depth-derived footprint,
so they **don't tile or connect cleanly** — `pos`/`yaw` position them, but walls
won't line up and doorways aren't cut (a door touching a reconstructed room joins
by overlap, not a sized opening; the script notes this). Use the spec-box path
(Phases 1–2) when you need clean tiling/doors; use `--reconstruct` when you want
real per-room geometry and will hand-arrange the join in Godot.

**Scale consistency** is handled by `target_height` = `ceiling_height`: each room
is uniformly rescaled so its ceiling matches, which also cancels the pre-placement
`--scale` (leave it at 1.0). This is the §4 fix, applied automatically.

**Richer still (future):** per-wall photo *projection* onto spec walls (so you get
real geometry *and* clean tiling) needs camera-pose↔wall correspondence — a Level-B
CV task left for later.

### Phase 3 — Reduce manual authoring

- Auto-suggest each room's `size` from the metric box already computed in
  `extract_room_params()` — author the layout, let depth fill dimensions.
- Generate the per-room **photos** automatically (ComfyUI; §5).
- Generate the **spec itself** (§5) — the speculative part.

---

## 4. Scale consistency — the silent killer

`--scale` defaults to **3.0**, an arbitrary multiplier on the metric depth. Fine
for one room; **fatal** for a dwelling, because every room would end up a
different arbitrary size and they won't tile.

Two fixes, in order of robustness:

1. **Normalize each room to a fixed ceiling height** (the `ceiling_height` in the
   spec, e.g. 2.6 m). Floor-to-ceiling is the **most reliably detected**
   dimension — it comes from the large horizontal RANSAC planes, not from
   furniture-occluded walls. Scaling each room so its detected height matches a
   shared constant guarantees cross-room consistency *without* trusting absolute
   metric depth. **Recommended.**
2. **Pin `--scale 1.0`** and trust the metric-indoor model directly. Simpler, but
   inherits any per-image metric drift.

Either way: the dwelling builder must **ignore the default 3.0**. Footprints
(`size`) come from the spec; height comes from `ceiling_height`.

---

## 5. Generating the blueprint (the speculative idea)

The open question: can we **generate** the floor-plan spec instead of authoring
it? "Diffusion blueprint → rooms" is appealing, but the naive version is a trap.
Three approaches, worst to best fit:

### (a) Raster diffusion + parse — tempting, fragile ❌
Prompt SDXL/Flux for "architectural floor plan," then vectorize walls/rooms/doors
(CubiCasa5K-style) into the spec. **Problem:** raster floor-plan images are
*structurally invalid* — rooms that don't close, doors to nowhere, gibberish
text, inconsistent scale. The parse step is the fragile link, and it fails
silently. High effort, low reliability. Not recommended as the primary path.

### (b) Structured floor-plan diffusion — good fit, heavier lift
Purpose-built models (e.g. **HouseDiffusion**, House-GAN++, Graph2Plan) output
**vector** layouts directly — room polygons, types, and an adjacency graph — which
is almost exactly our spec. No raster-parse gamble. Cost: integrating another
research model and its weights (and these emit *abstract* layouts, not photos —
you still generate per-room imagery separately).

### (c) LLM emits the spec directly — cheapest, most controllable ✅
Have an LLM produce the spec JSON under hard constraints: a connected door graph,
non-overlapping rectangles, plausible dimensions. You get a **guaranteed-valid,
walkable** layout, full control via prompt ("2-bed apartment, open-plan kitchen"),
and no parsing. Less organic than a trained floor-plan model, but by far the best
effort-to-reliability ratio for a first generative pass. **Recommended first
step toward generation.** A geometric validator (rectangles tile, graph is
connected, doors lie on shared edges) gates the output regardless of source.

### The synergy worth noting: fully-generative dwellings
The blueprint does **not** replace photos — the pipeline is still
photo → room. A generated spec produces the *layout*; you still need **one
interior photo per room**. You already have that lever: `comfy_gen.py` is a
minimal ComfyUI txt2img client (currently hardcoded to a living-room prompt). Per
room:

```
spec (room type, size) ──▶ comfy_gen prompt ("interior of a {type}, {size}…")
                          ──▶ interior photo ──▶ existing per-room pipeline
```

That closes the loop to a **fully generative, walkable dwelling**: generate
layout → generate each room's interior → reconstruct → compose.

### The honest risk: cross-step coherence
Chaining generators compounds error, and there's a real **coherence gap**: the
room-photo generator doesn't know where *this* room's doors and windows are
supposed to be (the spec does). So a generated "kitchen" photo may put a window
where the spec wants a door to the hall. Mitigations, none free:
- Treat photos as **dressing only** — doors/walls come from the spec geometry, so
  a misplaced window is cosmetic, not structural. (This is why §1's shell/dressing
  split matters here too.)
- Condition the photo generator on door/window positions (ControlNet on a crude
  per-room layout sketch) — more plumbing, better coherence.
- Accept that the generated dwelling is *plausible*, not *consistent*, and lean on
  hand-tuning via the composite `.tscn`.

**Recommendation:** prove the deterministic spine first (Phases 1–2 with an
authored spec). Add generation as **(c) LLM-direct spec** before reaching for
diffusion. Revisit **(b) structured floor-plan diffusion** only once the
deterministic compositor is solid and the bottleneck is genuinely *layout
variety*, not plumbing.

---

## 6. Open questions / risks

- **Wall thickness & shared walls** — do adjacent rooms share one wall, or does
  each own its four walls (doubled at seams)? Thin solid walls + dedup at shared
  edges is cleaner but more bookkeeping.
- **Furniture grounding** — still the existing un-solved per-room issue (MIDI has
  no common floor plane). Doesn't get worse in a dwelling, doesn't get solved by
  it either.
- **Texture seams** — rooms textured independently; adjacent rooms won't match.
  Fine (they're different rooms), but doorway thresholds will show the seam.
- **Non-rectangular rooms / L-shapes** — the spec assumes axis-aligned rectangles.
  Polygonal footprints are a later generalization.
- **Multi-floor** — out of scope for v1; `pos` is 2D and floor is Y=0 by
  assumption.

---

## 7. Recommended next step

Prototype **Phase 1**: the `export_room()` transform refactor plus a minimal
`build_dwelling.py` that places two authored rooms side by side so you can walk
between them. Everything else (doorways, scale normalization, generation) builds
on that spine.
