#!/usr/bin/env python3
"""build_dwelling.py -- compose several rooms into one walkable Godot scene.

Phases 1-2 of the multi-room dwelling plan (see docs/multi-room-dwelling.md):
takes a floor-plan spec (rooms with position/yaw/size + a door graph) and builds
a textured room *shell* per room on a shared ground plane (floor at Y=0).

  Phase 1: box shells placed on a shared frame.
  Phase 2: walls have **thickness** (solid prisms, deduped at shared edges so
           adjacent rooms don't z-fight) and doors are **sized openings** cut
           through the shared wall (side panels + a lintel), not whole-wall gaps.

This is the "shell" half of the shell-vs-dressing split: geometry comes from the
spec, not from photos. Reconstructed/textured interiors (depth relief, MIDI
furniture) layer on top later -- this script deliberately needs no GPU, no
checkpoints, and no per-room photo, so the compositor is testable on its own.

Usage:
    python build_dwelling.py examples/dwelling_two_room.json
    python build_dwelling.py spec.json --name myhouse --viewer-dir godot_viewer

Spec format (metres, on the XZ plane, floor at Y=0):
    {
      "name": "twobr",
      "ceiling_height": 2.6,
      "wall_thickness": 0.12,
      "rooms": [
        {"id": "living",  "pos": [0, 0], "yaw": 0, "size": [5, 4],
         "image": "examples/living.jpg"},
        {"id": "kitchen", "pos": [4, 0], "yaw": 0, "size": [3, 4],
         "surfaces": "kitchen_surfaces.json"}
      ],
      "doors": [ {"between": ["living", "kitchen"], "width": 0.9, "height": 2.1, "offset": 0.0} ]
    }

`pos` is the room centre; `size` is [width, depth].
Door fields are optional: width 0.9, height 2.1, offset 0 (along the shared edge
from its centre). Rooms connected by a door must share a wall edge (abut exactly).

Per-room *dressing* (wall/floor/ceiling colours) is resolved, highest priority
first, from: explicit `colors` (per key) -> a precomputed `surfaces` JSON (from
segment_room.py) -> live segmentation of the room's `image` (with --segment) ->
built-in defaults. This is the shell-vs-dressing split: geometry comes from the
spec; the photo only tints the surfaces. (Richer dressing -- per-wall photo
projection, depth relief, MIDI furniture -- is the next tier; see the design doc.)
"""
import argparse
import json
import os
import sys

import numpy as np

from room_from_image import PPM, generate_herringbone, generate_wall_texture

TOL = 1e-3              # metres: edge-coincidence tolerance for adjacency
MARGIN = 0.05           # metres: keep at least this much jamb beside a doorway
MAX_TEX_PX = 1024       # clamp generated texture dimensions

DEFAULT_COLORS = {"wall": (150, 142, 132), "floor": (120, 90, 60),
                  "ceiling": (210, 205, 200)}
DEFAULT_THICKNESS = 0.12
DEFAULT_DOOR = {"width": 0.9, "height": 2.1, "offset": 0.0}


# ----------------------------------------------------------------------------
# Geometry primitives (world frame: Y up, floor at Y=0)
# ----------------------------------------------------------------------------

def _tex_px(metres: float) -> int:
    return max(8, min(MAX_TEX_PX, int(metres * PPM)))


def _color(colors: dict, key: str) -> tuple:
    return tuple(colors.get(key, DEFAULT_COLORS[key]))


def _quad_mesh(corners, tex):
    """A textured quad (2 tris). Winding is irrelevant -- the viewer forces
    materials double-sided, so every surface shows from inside."""
    import trimesh
    verts = np.array(corners, dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    visual = trimesh.visual.TextureVisuals(uv=uvs, image=tex)
    return trimesh.Trimesh(vertices=verts, faces=faces, visual=visual, process=False)


def _box_mesh(x0, x1, y0, y1, z0, z1, tex):
    """An axis-aligned box as 6 textured quads (each face UV [0,1], matching the
    Phase-1 per-quad mapping). One shared texture image keeps it a single mesh."""
    import trimesh
    f = [
        [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)],  # -x
        [(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)],  # +x
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],  # -y
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],  # +y
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],  # -z
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],  # +z
    ]
    verts, faces, uvs = [], [], []
    for quad in f:
        b = len(verts)
        verts.extend(quad)
        uvs.extend([[0, 0], [1, 0], [1, 1], [0, 1]])
        faces.append([b, b + 1, b + 2])
        faces.append([b, b + 2, b + 3])
    visual = trimesh.visual.TextureVisuals(uv=np.array(uvs, dtype=np.float32),
                                           image=tex)
    return trimesh.Trimesh(vertices=np.array(verts, dtype=np.float32),
                           faces=np.array(faces, dtype=np.int32),
                           visual=visual, process=False)


# ----------------------------------------------------------------------------
# Layout: world footprints + shared edges
# ----------------------------------------------------------------------------

def world_rect(room: dict) -> tuple[float, float, float, float]:
    """Axis-aligned world footprint (x0, x1, z0, z1). Assumes yaw is a multiple
    of 90 degrees (Phase 1-2); a yaw of 90/270 swaps width and depth."""
    px, pz = room["pos"]
    w, d = room["size"]
    if round(room.get("yaw", 0)) % 180 == 90:
        w, d = d, w
    return (px - w / 2.0, px + w / 2.0, pz - d / 2.0, pz + d / 2.0)


def shared_edge(ra, rb):
    """The wall two footprints share as (axis, coord, lo, hi), or None. `axis` is
    the axis the wall is perpendicular to; [lo, hi] is the overlap along the wall."""
    ax0, ax1, az0, az1 = ra
    bx0, bx1, bz0, bz1 = rb
    zlo, zhi = max(az0, bz0), min(az1, bz1)
    if zhi - zlo > TOL:                       # vertical shared edge (left/right)
        if abs(ax1 - bx0) < TOL:
            return ("x", ax1, zlo, zhi)
        if abs(ax0 - bx1) < TOL:
            return ("x", ax0, zlo, zhi)
    xlo, xhi = max(ax0, bx0), min(ax1, bx1)
    if xhi - xlo > TOL:                       # horizontal shared edge (front/back)
        if abs(az1 - bz0) < TOL:
            return ("z", az1, xlo, xhi)
        if abs(az0 - bz1) < TOL:
            return ("z", az0, xlo, xhi)
    return None


def wall_segments(rooms: list[dict], ceiling_h: float) -> dict:
    """One wall segment per unique room edge. Two rooms sharing an identical edge
    collapse to a single segment (dedup) so the shared wall is built once."""
    segs: dict = {}
    for r in rooms:
        x0, x1, z0, z1 = world_rect(r)
        wall_c = _color(r.get("colors", {}), "wall")
        sides = [("x", x0, z0, z1), ("x", x1, z0, z1),
                 ("z", z0, x0, x1), ("z", z1, x0, x1)]
        for axis, coord, lo, hi in sides:
            key = (axis, round(coord, 3), round(lo, 3), round(hi, 3))
            if key not in segs:
                segs[key] = {"axis": axis, "coord": coord, "lo": lo, "hi": hi,
                             "height": ceiling_h, "color": wall_c, "doors": []}
    return segs


def _door_opening(edge, door) -> tuple[float, float, bool]:
    """Resolve a door spec to a sized (lo, hi, full_bay) opening along a wall edge,
    clamped to leave jambs. `full_bay` is True when the door is too wide for the
    wall (minus margins) and the whole bay is opened. Geometry only -- the door
    height is clamped to the wall height by the caller."""
    _, _, slo, shi = edge
    width = float(door.get("width", DEFAULT_DOOR["width"]))
    offset = float(door.get("offset", DEFAULT_DOOR["offset"]))
    span = shi - slo
    if width >= span - 2 * MARGIN:            # opening wider than wall -> full bay
        return slo, shi, True
    half = width / 2.0
    center = (slo + shi) / 2.0 + offset
    center = min(max(center, slo + MARGIN + half), shi - MARGIN - half)
    return center - half, center + half, False


def attach_doorways(segs: dict, rooms: list[dict], doors: list[dict]) -> None:
    """Resolve each door to a sized opening on the wall segment(s) it passes
    through, clamped to leave jambs."""
    rects = {r["id"]: world_rect(r) for r in rooms}
    for door in doors:
        a, b = door["between"]
        if a not in rects or b not in rects:
            sys.exit(f"Door references unknown room: {door['between']}")
        edge = shared_edge(rects[a], rects[b])
        if edge is None:
            print(f"  WARNING: rooms {a!r} and {b!r} do not share a wall edge; "
                  f"skipping door (check pos/size so they abut exactly).")
            continue
        axis, coord, slo, shi = edge
        dlo, dhi, full = _door_opening(edge, door)
        height = float(door.get("height", DEFAULT_DOOR["height"]))
        if full:
            print(f"  note: door {a}<->{b} width {dhi - dlo:g}m >= wall "
                  f"{shi - slo:g}m; opening full bay.")

        matched = False
        for seg in segs.values():
            if (seg["axis"] == axis and abs(seg["coord"] - coord) < TOL
                    and seg["lo"] - TOL <= slo and seg["hi"] + TOL >= shi):
                seg["doors"].append({"lo": dlo, "hi": dhi,
                                     "height": min(height, seg["height"])})
                matched = True
        if matched:
            print(f"  door: {a!r}<->{b!r} opening {dhi - dlo:g}x{height:g}m "
                  f"on {axis}={coord:g}, span [{dlo:.2f},{dhi:.2f}]")
        else:
            print(f"  WARNING: no wall segment found for door {a!r}<->{b!r}.")


# ----------------------------------------------------------------------------
# Dressing: per-surface colours from the room photo (shell stays spec-driven)
# ----------------------------------------------------------------------------

def _colors_from_surfaces(surfaces: dict) -> dict:
    """Pull {wall, floor, ceiling} colours out of a segment_room.py surfaces dict
    ({name: {"color": [r,g,b], ...}}). Floor falls back to a rug if present."""
    out = {}
    if isinstance(surfaces.get("wall"), dict):
        out["wall"] = tuple(surfaces["wall"]["color"])
    floor_src = surfaces.get("floor") or surfaces.get("rug")
    if isinstance(floor_src, dict):
        out["floor"] = tuple(floor_src["color"])
    if isinstance(surfaces.get("ceiling"), dict):
        out["ceiling"] = tuple(surfaces["ceiling"]["color"])
    return out


def _segment_colors(image_path: str) -> dict:
    """Run SegFormer on a room photo and return its full surfaces dict. Imports
    torch/transformers lazily so the no-segmentation path stays CPU-only."""
    import numpy as np
    from PIL import Image
    from segment_room import (group_surfaces, sample_surface_colors,
                              segment_image)
    label_map, id2label = segment_image(image_path)
    masks = group_surfaces(label_map, id2label)
    rgb = np.array(Image.open(image_path).convert("RGB"))
    return sample_surface_colors(rgb, masks)


def dress_rooms(spec: dict, spec_dir: str, project_dir: str,
                do_segment: bool, resegment: bool) -> None:
    """Resolve each room's wall/floor/ceiling colours in place. Segmentation
    results are cached as <project_dir>/<room_id>_surfaces.json and reused."""
    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(spec_dir, p)

    for r in spec["rooms"]:
        resolved: dict = {}
        if r.get("image"):
            r["_image_abs"] = _abs(r["image"])  # used by --reconstruct

        # Precomputed surfaces JSON (deterministic, no GPU).
        surf_path = r.get("surfaces")
        if surf_path and os.path.isfile(_abs(surf_path)):
            with open(_abs(surf_path)) as f:
                resolved.update(_colors_from_surfaces(json.load(f)))

        # Otherwise segment the room photo (opt-in, cached, graceful on failure).
        elif r.get("image"):
            cache = os.path.join(project_dir, f"{r['id']}_surfaces.json")
            if os.path.isfile(cache) and not resegment:
                with open(cache) as f:
                    resolved.update(_colors_from_surfaces(json.load(f)))
                print(f"  {r['id']}: dressing from cache {os.path.basename(cache)}")
            elif do_segment:
                img = _abs(r["image"])
                if not os.path.isfile(img):
                    print(f"  WARNING: {r['id']} image not found: {img}")
                else:
                    try:
                        surfaces = _segment_colors(img)
                        os.makedirs(project_dir, exist_ok=True)
                        with open(cache, "w") as f:
                            json.dump(surfaces, f, indent=2)
                        resolved.update(_colors_from_surfaces(surfaces))
                        print(f"  {r['id']}: segmented {os.path.basename(img)} "
                              f"-> {', '.join(resolved)} (cached)")
                    except Exception as e:  # torch/model/CUDA not available
                        print(f"  WARNING: segmentation failed for {r['id']} "
                              f"({type(e).__name__}: {e}); using defaults.")
            else:
                print(f"  note: {r['id']} has an image but --segment is off; "
                      f"using explicit/default colours.")

        # Explicit per-key colours always win.
        resolved.update({k: tuple(v) for k, v in r.get("colors", {}).items()})
        if resolved:
            r["colors"] = resolved


# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------

def build_wall_meshes(seg: dict, thickness: float) -> list:
    """Turn a wall segment into solid prisms. A doorway splits the wall into side
    panels plus a lintel above the opening; the gap below the lintel is the door."""
    axis, coord, color = seg["axis"], seg["coord"], seg["color"]
    H = seg["height"]
    t = thickness
    # Extend the run by half a thickness each end so corners fill cleanly.
    lo, hi = seg["lo"] - t / 2.0, seg["hi"] + t / 2.0
    doors = sorted(seg["doors"], key=lambda d: d["lo"])

    # Panels as (free_lo, free_hi, y_lo, y_hi) along the wall's free axis.
    panels = []
    if not doors:
        panels.append((lo, hi, 0.0, H))
    else:
        cursor = lo
        for d in doors:
            if d["lo"] - cursor > TOL:
                panels.append((cursor, d["lo"], 0.0, H))      # jamb / side panel
            if H - d["height"] > TOL:
                panels.append((d["lo"], d["hi"], d["height"], H))  # lintel
            cursor = d["hi"]
        if hi - cursor > TOL:
            panels.append((cursor, hi, 0.0, H))

    meshes = []
    for flo, fhi, ylo, yhi in panels:
        tex = generate_wall_texture(_tex_px(fhi - flo), _tex_px(yhi - ylo), color)
        if axis == "x":
            meshes.append(_box_mesh(coord - t / 2, coord + t / 2,
                                    ylo, yhi, flo, fhi, tex))
        else:  # "z"
            meshes.append(_box_mesh(flo, fhi, ylo, yhi,
                                    coord - t / 2, coord + t / 2, tex))
    return meshes


def _floor_ceiling_meshes(room: dict, ceiling_h: float):
    """Clean flat floor + ceiling quads spanning the room's exact spec footprint.
    Used for box rooms and for conformed reconstructed rooms (whose depth-displaced
    floors don't reach the footprint edge, which would leave gaps at the seams)."""
    x0, x1, z0, z1 = world_rect(room)
    w, d = x1 - x0, z1 - z0
    colors = room.get("colors", {})
    floor = _quad_mesh([(x0, 0, z0), (x1, 0, z0), (x1, 0, z1), (x0, 0, z1)],
                       generate_herringbone(_tex_px(w), _tex_px(d),
                                            _color(colors, "floor")))
    ceil = _quad_mesh([(x0, ceiling_h, z0), (x1, ceiling_h, z0),
                       (x1, ceiling_h, z1), (x0, ceiling_h, z1)],
                      generate_wall_texture(_tex_px(w), _tex_px(d),
                                            _color(colors, "ceiling")))
    return floor, ceil


def _surfaces_from_colors(colors: dict) -> dict | None:
    """Adapt a {wall/floor/ceiling: (r,g,b)} dict into the surfaces shape that
    room_from_image.build_room_scene expects ({name: {"color": [...]}})."""
    if not colors:
        return None
    return {k: {"color": list(v)} for k, v in colors.items()}


def _place_room_meshes(room_scene, placement, scale: float) -> dict:
    """Scale a camera-frame room scene into Godot's frame, position it into its
    slot via `place_room`, then bake graph transforms into the vertices. Returns
    {geometry_name: world-space Trimesh}. Baking (not graph transforms) is required
    so the meshes survive the merge; returning them separately lets the caller drop
    a shared wall before merging, and preserves the viewer's per-mesh collision."""
    import trimesh
    from room_from_image import place_room

    room_scene.apply_transform(trimesh.transformations.scale_matrix(scale))
    room_scene.apply_transform(np.diag([1.0, -1.0, -1.0, 1.0]))  # camera -> Godot
    place_room(room_scene, placement)

    out = {}
    for name in list(room_scene.geometry.keys()):
        mesh = room_scene.geometry[name].copy()
        tf = room_scene.graph.get(name)
        if tf is not None and tf[0] is not None:
            mesh.apply_transform(tf[0])
        out[name] = mesh
    return out


def _reconstruct_room_meshes(room: dict, ceiling_h: float, scale: float,
                             checkpoint: str, recon: dict, conform: bool) -> dict:
    """Reconstruct one room from its photo into its floor-plan slot (pos/yaw),
    normalized to the shared ceiling height. With conform=True the room is also
    stretched to its spec `size` footprint so it tiles with its neighbours.
    Returns {geometry_name: world-space Trimesh}."""
    from room_from_image import Placement, reconstruct_room_scene

    cam_scene = reconstruct_room_scene(
        _abs_image(room), checkpoint,
        hfov=recon["hfov"],
        surface_colors=_surfaces_from_colors(room.get("colors")),
        subdivisions=recon["subdivisions"],
        max_displacement=recon["max_displacement"],
        project_walls=recon.get("project_walls", False),
        sample_walls=recon.get("sample_walls", False))
    placement = Placement(
        pos=tuple(room["pos"]), yaw=float(room.get("yaw", 0)),
        target_height=ceiling_h,
        target_size=tuple(room["size"]) if conform else None)
    return _place_room_meshes(cam_scene, placement, scale)


def _room_edges(rect) -> list:
    """The four footprint edges as (axis, coord): the planes a room's walls sit on."""
    x0, x1, z0, z1 = rect
    return [("x", round(x0, 3)), ("x", round(x1, 3)),
            ("z", round(z0, 3)), ("z", round(z1, 3))]


def _nearest_edge(cx: float, cz: float, rect) -> tuple:
    """Which footprint edge a wall (at world centroid cx,cz) sits on."""
    x0, x1, z0, z1 = rect
    cands = [(("x", round(x0, 3)), abs(cx - x0)), (("x", round(x1, 3)), abs(cx - x1)),
             (("z", round(z0, 3)), abs(cz - z0)), (("z", round(z1, 3)), abs(cz - z1))]
    return min(cands, key=lambda c: c[1])[0]


def _cap_wall_mesh(rect, edge, ceiling_h: float, color: tuple):
    """A clean flat wall quad spanning a footprint edge, floor to ceiling -- used to
    close a reconstructed room's open (camera-side) front so it's enclosed."""
    x0, x1, z0, z1 = rect
    axis, coord = edge
    H = ceiling_h
    if axis == "x":
        corners = [(coord, 0, z0), (coord, 0, z1), (coord, H, z1), (coord, H, z0)]
        tex = generate_wall_texture(_tex_px(z1 - z0), _tex_px(H), color)
    else:
        corners = [(x0, 0, coord), (x1, 0, coord), (x1, H, coord), (x0, H, coord)]
        tex = generate_wall_texture(_tex_px(x1 - x0), _tex_px(H), color)
    return _quad_mesh(corners, tex)


def _nearest_wall_name(meshes: dict, midpoint) -> str | None:
    """Name of the wall_* geometry whose world centroid (in XZ) is closest to
    `midpoint` -- the wall facing a neighbour, to be opened for a doorway."""
    mx, mz = midpoint
    best, best_d = None, float("inf")
    for name, mesh in meshes.items():
        if not name.startswith("wall"):
            continue
        c = mesh.vertices.mean(axis=0)
        d = (c[0] - mx) ** 2 + (c[2] - mz) ** 2
        if d < best_d:
            best, best_d = name, d
    return best


def _abs_image(room: dict) -> str:
    return room["_image_abs"]


def build_dwelling(spec: dict, *, reconstruct: bool = False,
                   checkpoint: str | None = None, scale: float = 1.0,
                   recon: dict | None = None, conform: bool = False,
                   cap: bool = True):
    """Compose all rooms into a single trimesh.Scene in the shared frame.

    With reconstruct=True, any room that has an `image` (and a usable checkpoint)
    is built as a full depth-reconstructed shell placed into its slot. By default
    such rooms keep their own depth-derived footprint, so they may not tile or
    connect. With conform=True they are stretched to their spec `size` footprint,
    so reconstructed rooms tile like box rooms and a door between two of them opens
    the shared wall (full bay) for a walk-through. Rooms without an image fall back
    to spec box shells with deduped thick walls + sized doorways.
    """
    import trimesh

    ceiling_h = float(spec.get("ceiling_height", 2.6))
    thickness = float(spec.get("wall_thickness", DEFAULT_THICKNESS))
    recon = recon or {}
    rooms = spec["rooms"]
    by_id = {r["id"]: r for r in rooms}
    if len(by_id) != len(rooms):
        sys.exit("Duplicate room id in spec.")

    scene = trimesh.Scene()
    box_rooms = []                 # built as spec box shells (walls + doorways)
    recon_meshes: dict[str, dict] = {}  # room_id -> {geom_name: world Trimesh}

    for r in rooms:
        ckpt_ok = bool(checkpoint and os.path.isfile(checkpoint))
        if reconstruct and r.get("image") and not ckpt_ok:
            print(f"  WARNING: --reconstruct set but checkpoint missing "
                  f"({checkpoint}); building {r['id']!r} as a box shell.")
        if reconstruct and r.get("image") and ckpt_ok:
            try:
                print(f"  room {r['id']!r}: reconstructing from "
                      f"{os.path.basename(_abs_image(r))} ...")
                meshes = _reconstruct_room_meshes(r, ceiling_h, scale,
                                                  checkpoint, recon, conform)
                recon_meshes[r["id"]] = meshes
                tag = ("conformed to spec footprint" if conform
                       else "footprint from depth")
                print(f"    placed {len(meshes)} mesh(es) @ pos {r['pos']} "
                      f"yaw {r.get('yaw', 0)} ({tag})")
                continue
            except Exception as e:
                print(f"  WARNING: reconstruction failed for {r['id']!r} "
                      f"({type(e).__name__}: {e}); falling back to box shell.")

        # Box-shell room: floor + ceiling now, walls after dedup.
        floor, ceil = _floor_ceiling_meshes(r, ceiling_h)
        scene.add_geometry(floor, geom_name=f"{r['id']}_floor")
        scene.add_geometry(ceil, geom_name=f"{r['id']}_ceiling")
        box_rooms.append(r)
        x0, x1, z0, z1 = world_rect(r)
        print(f"  room {r['id']!r}: box {x1 - x0:g}x{z1 - z0:g}m @ pos {r['pos']} "
              f"yaw {r.get('yaw', 0)}")

    # Sort doors by which rooms they connect.
    box_ids = {r["id"] for r in box_rooms}
    recon_ids = set(recon_meshes)
    box_doors, recon_doors, mixed_doors = [], [], []
    for dr in spec.get("doors", []):
        pair = set(dr["between"])
        if pair <= box_ids:
            box_doors.append(dr)
        elif conform and pair <= recon_ids:
            recon_doors.append(dr)
        else:
            mixed_doors.append(dr)
    for dr in mixed_doors:
        why = ("a reconstructed room is not conformed" if not conform
               else "it joins a reconstructed and a box room")
        print(f"  note: door {dr['between']} not cut ({why}); rooms join by overlap.")

    rects = {rid: world_rect(by_id[rid]) for rid in recon_ids}

    # Which footprint edge each room's walls cover (computed before any door drop,
    # while all 3 reconstructed walls are present). The uncovered edge is the open
    # camera-side front.
    covered = {rid: {_nearest_edge(m.vertices.mean(0)[0], m.vertices.mean(0)[2],
                                    rects[rid])
                     for n, m in meshes.items() if n.startswith("wall")}
               for rid, meshes in recon_meshes.items()}
    door_edges = {rid: set() for rid in recon_ids}

    # Conformed reconstructed rooms tile, so a door cuts the shared wall. Both
    # rooms' depth-displaced walls on the seam are dropped and replaced with ONE
    # clean deduped wall segment carrying a sized opening (jambs + lintel), so the
    # join is a real doorway instead of a full-bay hole. The clean wall on the seam
    # also closes the corners against the perpendicular reconstructed walls.
    door_walls = 0
    for dr in recon_doors:
        a, b = dr["between"]
        edge = shared_edge(rects[a], rects[b])
        if edge is None:
            print(f"  WARNING: reconstructed rooms {a!r},{b!r} share no edge; "
                  f"door skipped.")
            continue
        axis, coord, lo, hi = edge
        mid = (coord, (lo + hi) / 2) if axis == "x" else ((lo + hi) / 2, coord)
        for rid in (a, b):
            wname = _nearest_wall_name(recon_meshes[rid], mid)
            if wname:
                del recon_meshes[rid][wname]
            door_edges[rid].add((axis, round(coord, 3)))

        dlo, dhi, full = _door_opening(edge, dr)
        height = min(float(dr.get("height", DEFAULT_DOOR["height"])), ceiling_h)
        # build_wall_meshes extends each end by thickness/2 to fill box-mode corners;
        # in the reconstructed case the perpendicular walls already sit on the
        # footprint edge, so pre-shrink the span by that much to land flush instead
        # of poking a nub outside the shell.
        seg = {"axis": axis, "coord": coord,
               "lo": lo + thickness / 2.0, "hi": hi - thickness / 2.0,
               "height": ceiling_h, "color": _color(by_id[a].get("colors", {}), "wall"),
               "doors": [{"lo": dlo, "hi": dhi, "height": height}]}
        for mesh in build_wall_meshes(seg, thickness):
            scene.add_geometry(mesh, geom_name=f"door_{a}_{b}_{door_walls}")
            door_walls += 1
        kind = "full bay" if full else f"{dhi - dlo:g}x{height:g}m"
        print(f"  door: {a!r}<->{b!r} sized opening ({kind}) on "
              f"{axis}={coord:g} (reconstructed)")

    # Merge reconstructed room meshes. In conform mode, swap the depth-displaced
    # floor/ceiling for clean spec-footprint quads so rooms tile without seam gaps;
    # keep the reconstructed walls (their relief is the point). Optionally cap each
    # room's open front (the footprint edge no wall covers and no door opens).
    for rid, meshes in recon_meshes.items():
        if conform:
            meshes.pop("floor", None)
            meshes.pop("ceiling", None)
            floor, ceil = _floor_ceiling_meshes(by_id[rid], ceiling_h)
            scene.add_geometry(floor, geom_name=f"{rid}_floor")
            scene.add_geometry(ceil, geom_name=f"{rid}_ceiling")
            if cap:
                wall_c = _color(by_id[rid].get("colors", {}), "wall")
                open_edges = (set(_room_edges(rects[rid]))
                              - covered[rid] - door_edges[rid])
                for i, edge in enumerate(sorted(open_edges)):
                    scene.add_geometry(
                        _cap_wall_mesh(rects[rid], edge, ceiling_h, wall_c),
                        geom_name=f"{rid}_cap{i}")
                if open_edges:
                    print(f"  capped {len(open_edges)} open side(s) of {rid!r}")
        for name, mesh in meshes.items():
            scene.add_geometry(mesh, geom_name=f"{rid}_recon_{name}")

    # Box-room walls (deduped) + sized doorways.
    segs = wall_segments(box_rooms, ceiling_h)
    attach_doorways(segs, box_rooms, box_doors)
    n = 0
    for seg in segs.values():
        for mesh in build_wall_meshes(seg, thickness):
            scene.add_geometry(mesh, geom_name=f"wall{n}")
            n += 1
    if box_rooms:
        print(f"  {len(segs)} wall segment(s) -> {n} panel mesh(es), "
              f"thickness {thickness:g}m")
    return scene


def _write_editor_tscn(project_dir: str, name: str, glb_file: str) -> str:
    """Write <name>.tscn that statically instances the combined GLB, matching
    build_scene.py's convention so the dwelling is editable in the Godot editor."""
    tscn = (
        "[gd_scene load_steps=2 format=3]\n\n"
        f'[ext_resource type="PackedScene" path="res://{name}/{glb_file}" id="1_glb"]\n\n'
        f'[node name="{name}" type="Node3D"]\n\n'
        f'[node name="{name}" parent="." instance=ExtResource("1_glb")]\n')
    path = os.path.join(project_dir, f"{name}.tscn")
    with open(path, "w", newline="\n") as f:
        f.write(tscn)
    return path


def export_dwelling(scene, name: str, viewer_dir: str) -> str:
    """Write the combined GLB into godot_viewer/<name>/ plus an editor .tscn.

    The GLB filename contains "room" so the viewer treats it as a real shell and
    skips its fall-through catch floor (see godot_viewer/main.gd)."""
    project_dir = os.path.join(viewer_dir, name)
    os.makedirs(project_dir, exist_ok=True)
    glb_file = f"{name}_rooms.glb"
    out_path = os.path.join(project_dir, glb_file)
    scene.export(out_path)
    _write_editor_tscn(viewer_dir, name, glb_file)
    mb = os.path.getsize(out_path) / 1e6
    print(f"\nwrote {out_path}  ({mb:.1f} MB)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Floor-plan spec -> walkable multi-room Godot scene.")
    ap.add_argument("spec", help="floor-plan spec JSON")
    ap.add_argument("--name", default=None,
                    help="output project name (default: spec 'name' or file stem)")
    ap.add_argument("--viewer-dir",
                    default=os.path.join(os.path.dirname(__file__), "godot_viewer"),
                    help="Godot project dir to drop the result into")
    ap.add_argument("--segment", action="store_true",
                    help="segment each room's `image` (SegFormer, needs torch) to "
                         "derive real wall/floor/ceiling colours; cached per room")
    ap.add_argument("--resegment", action="store_true",
                    help="ignore cached segmentation and re-run it")
    ap.add_argument("--reconstruct", action="store_true",
                    help="build rooms that have an `image` as full depth-"
                         "reconstructed shells placed into their slots (needs the "
                         "DepthAnything checkpoint + GPU). Footprints come from "
                         "depth, so reconstructed rooms may not tile/connect.")
    ap.add_argument("--conform", action="store_true",
                    help="with --reconstruct, stretch each reconstructed room to "
                         "its spec `size` footprint so rooms tile and doors between "
                         "two reconstructed rooms open the shared wall (walk-through)")
    ap.add_argument("--no-cap", action="store_true",
                    help="with --conform, leave each reconstructed room's open "
                         "camera-side front open instead of capping it with a wall")
    ap.add_argument("--project-walls", action="store_true",
                    help="with --reconstruct, texture each room's walls by "
                         "projecting its source photo onto them (real wall imagery) "
                         "instead of a flat procedural tint")
    ap.add_argument("--sample-walls", action="store_true",
                    help="with --reconstruct, tint each room's walls with the "
                         "robust flat-wall colour sampled from genuine wall pixels "
                         "only (depth+semantics mask; no furniture/shadow bleed)")
    ap.add_argument("--checkpoint",
                    default=os.path.join(os.path.dirname(__file__), "checkpoints",
                                         "depth_anything_v2_metric_hypersim_vitl.pth"),
                    help="DepthAnything V2 metric-indoor checkpoint (for --reconstruct)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="pre-placement scale for reconstructed rooms; cancels out "
                         "under ceiling-height normalization, so leave at 1.0")
    ap.add_argument("--hfov", type=float, default=60.0,
                    help="horizontal FOV (deg) for reconstruction")
    ap.add_argument("--subdivisions", type=int, default=24,
                    help="wall subdivision for depth relief (1=flat)")
    ap.add_argument("--max-displacement", type=float, default=0.15,
                    help="max depth displacement in metres")
    args = ap.parse_args()

    if not os.path.isfile(args.spec):
        sys.exit(f"Spec not found: {args.spec}")
    with open(args.spec) as f:
        spec = json.load(f)
    if not spec.get("rooms"):
        sys.exit("Spec has no rooms.")

    name = args.name or spec.get("name") \
        or os.path.splitext(os.path.basename(args.spec))[0]
    spec_dir = os.path.dirname(os.path.abspath(args.spec))
    project_dir = os.path.join(args.viewer_dir, name)

    print(f"[1/3] resolving dressing for {len(spec['rooms'])} room(s)...")
    dress_rooms(spec, spec_dir, project_dir, args.segment, args.resegment)
    print(f"[2/3] building rooms...")
    recon = {"hfov": args.hfov, "subdivisions": args.subdivisions,
             "max_displacement": args.max_displacement,
             "project_walls": args.project_walls,
             "sample_walls": args.sample_walls}
    scene = build_dwelling(spec, reconstruct=args.reconstruct,
                           checkpoint=args.checkpoint, scale=args.scale,
                           recon=recon, conform=args.conform, cap=not args.no_cap)
    print("[3/3] exporting...")
    export_dwelling(scene, name, args.viewer_dir)
    print("Open godot_viewer/ in Godot 4 and press F5 -- it loads the newest "
          "project folder automatically.")


if __name__ == "__main__":
    main()
