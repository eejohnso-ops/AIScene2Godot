#!/usr/bin/env python3
"""
build_room.py  --  Phase 1: AI panorama -> walkable-shell GLB for Godot.

Pipeline:
    prompt
      -> ComfyUI (equirectangular 360 panorama, via HTTP API)
      -> [batch + review gate]
      -> split panorama into 6 cubemap faces
      -> MoGe-2 per face  -> metric XYZ point map per face
      -> rotate faces into world space + fuse into one point cloud
      -> Poisson surface reconstruction -> shell mesh
      -> spherical UVs into the panorama (the panorama IS the texture)
      -> export GLB (+ optional copy into Godot project)

WHERE TO RUN THIS
    Recommended: on the Windows machine that runs ComfyUI, with MOGE_BACKEND="local".
    That keeps the GPU, float depth, and meshing in one place (no precision loss
    pushing depth over HTTP). The Mac only needs to see the output folder.

    Alternative: run on the Mac with MOGE_BACKEND="comfy" -- but you must add an
    EXR/.npy depth save node to your MoGe ComfyUI workflow so depth survives as
    float. See get_depth_comfy() for where to plug that in.

DEPENDENCIES (Windows / CUDA box for "local"):
    pip install requests numpy pillow py360convert open3d trimesh
    pip install git+https://github.com/microsoft/MoGe.git     # provides `moge`
    (torch with CUDA must already be installed for MoGe-2)

STATUS: scaffold. The two spots that depend on YOUR ComfyUI setup are marked
    `# >>> CONFIGURE`. Everything else (cubemap math, fusion, meshing, UVs,
    export) is complete and deterministic.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import uuid
import shutil
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import requests
from PIL import Image

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

COMFY_URL = "http://127.0.0.1:7821"      # localhost if running on the Comfy box;
                                          # else e.g. "http://192.168.1.50:8188"

# >>> CONFIGURE: export your panorama workflow from ComfyUI via
#     "Save (API Format)" and point to it here. The script injects the prompt
#     and seed by locating nodes by class_type (see submit_panorama()).
PANO_WORKFLOW_JSON = "workflows/panorama_api.json"

PANO_W, PANO_H = 2048, 1024               # equirect, 2:1
FACE_SIZE = 1024                          # cubemap face resolution fed to MoGe-2
BATCH = 2                                 # panoramas per run (FLUX/DiT360 is slow;
                                          # was 4 for fast SDXL). Lower = quicker.

MOGE_BACKEND = "local"                    # "local" (CUDA, recommended) | "comfy"
MOGE_MODEL = "Ruicheng/moge-2-vitl-normal"  # HF id used by the moge package

# Re-unproject MoGe's depth with the TRUE 90-deg cubemap intrinsics instead of
# trusting MoGe's own per-image focal estimate. MoGe guesses a focal length per
# face; when those guesses disagree the 6 faces shear against each other on
# fusion (the room comes out as a long stretched smear). Each face IS a 90-deg
# pinhole, so imposing that makes the faces mutually consistent. Off = MoGe as-is.
REPROJECT_INTRINSICS = True
# Drop points farther than this many metres (per face). Dark doorways read as
# near-infinite depth and the floor smears toward the horizon at grazing angles;
# clipping turns those into holes instead of a giant skirt. 0 = keep everything.
DEPTH_CLIP_M = 12.0

OUTPUT_DIR = Path("out")
GODOT_PROJECT_DIR: Path | None = Path(__file__).parent / "godot_viewer"
                                          # GLBs are auto-copied here so the Godot
                                          # viewer picks up the newest room. Set to
                                          # None to leave the GLB only in OUTPUT_DIR.

POISSON_DEPTH = 9                         # higher = finer mesh, slower
DENSITY_TRIM_QUANTILE = 0.04              # drop lowest-density Poisson verts
                                          # (the "skirt" that closes openings).
                                          # 0 = keep all; raise if blobs persist.
TARGET_TRIS = 150_000                     # decimate to this for game use
# --- mesh cleanup (de-blob) ---
OUTLIER_NB = 20                           # statistical-outlier removal: neighbours
OUTLIER_STD = 2.0                         # ...and std ratio (lower=more aggressive).
                                          # Removes noisy points -> fewer spikes. 0=off
MIN_COMPONENT_FRAC = 0.02                 # drop mesh islands smaller than this frac of
                                          # the largest component (floating blobs). 0=off
SMOOTH_ITERS = 8                          # Taubin smoothing passes (de-blob). 0=off
NADIR_PATCH = True                        # paste clean floor over the bottom pole
GEN_TIMEOUT_S = 1800                      # max wait for one ComfyUI generation.
                                          # FLUX.1-dev at 2048 + offload is MINUTES
                                          # per image (plus first-run model load),
                                          # so 30 min. SDXL only needed ~30s.

# Prepended to every prompt before injection. DiT360 wants this exact phrase;
# for an SDXL-360 graph use "equirectangular 360 view, " instead, or "" for none.
PROMPT_PREFIX = "This is a panorama image. "
# Equirect sanity gate: a real equirect MUST wrap left<->right seamlessly. This
# is the max acceptable mean edge mismatch (0-255). Bad SDXL panos scored 45-63;
# a DiT360 + edge-blended pano scores near 0. Raise/disable via --force.
SEAM_MAX = 25.0

CLIENT_ID = uuid.uuid4().hex


# ----------------------------------------------------------------------------
# COMFYUI CLIENT
# ----------------------------------------------------------------------------

def _find_nodes(wf: dict, class_type: str) -> list[str]:
    return [nid for nid, n in wf.items() if n.get("class_type") == class_type]


def submit_panorama(prompt: str, seed: int) -> bytes:
    """Queue one panorama generation and return the PNG bytes."""
    wf = json.loads(Path(PANO_WORKFLOW_JSON).read_text())

    full_prompt = PROMPT_PREFIX + prompt

    # Inject prompt + seed. Two graph shapes are supported automatically:
    #   1) DiT360 (recommended): one DiT360TextToPanorama node that takes the
    #      prompt and seed directly as inputs.
    #   2) Standard SDXL/FLUX: a positive CLIPTextEncode + a KSampler.
    dit = _find_nodes(wf, "DiT360TextToPanorama")
    if dit:
        for nid in dit:
            wf[nid]["inputs"]["prompt"] = full_prompt
            wf[nid]["inputs"]["seed"] = seed
    else:
        # >>> CONFIGURE (SDXL/FLUX only): a graph usually has TWO CLIPTextEncode
        # nodes (positive + negative). Picking "the first" can hit the negative
        # one. Prefer a node whose _meta.title looks positive; else pick one not
        # labelled negative, and warn. Hard-code the node id if it guesses wrong.
        encoders = _find_nodes(wf, "CLIPTextEncode")
        def _title(nid):
            return wf[nid].get("_meta", {}).get("title", "").lower()
        target = next((n for n in encoders if "pos" in _title(n)), None)
        if target is None:
            target = next((n for n in encoders
                           if "neg" not in _title(n)), encoders[0] if encoders else None)
        if target is not None and "text" in wf[target]["inputs"]:
            wf[target]["inputs"]["text"] = full_prompt
            if len(encoders) > 1 and "pos" not in _title(target):
                print(f"  WARN: guessed positive prompt node = {target}; "
                      "hard-code it in submit_panorama() if the result looks wrong.")
        else:
            raise RuntimeError("No DiT360TextToPanorama or CLIPTextEncode node found")
        for nid in _find_nodes(wf, "KSampler"):
            wf[nid]["inputs"]["seed"] = seed

    r = requests.post(f"{COMFY_URL}/prompt",
                      json={"prompt": wf, "client_id": CLIENT_ID}, timeout=30)
    r.raise_for_status()
    pid = r.json()["prompt_id"]

    # poll history until this prompt completes -- bounded, so a failed or stuck
    # generation surfaces as an error instead of hanging the script forever.
    deadline = time.time() + GEN_TIMEOUT_S
    outputs = None
    while time.time() < deadline:
        time.sleep(1.5)
        h = requests.get(f"{COMFY_URL}/history/{pid}", timeout=30).json()
        if pid in h:
            if h[pid].get("status", {}).get("status_str") == "error":
                raise RuntimeError(f"ComfyUI reported an error for prompt {pid}")
            outputs = h[pid]["outputs"]
            break
    if outputs is None:
        raise TimeoutError(
            f"ComfyUI did not finish prompt {pid} within {GEN_TIMEOUT_S}s")

    # grab the first SaveImage/PreviewImage output
    for node_out in outputs.values():
        for img in node_out.get("images", []):
            params = {"filename": img["filename"],
                      "subfolder": img.get("subfolder", ""),
                      "type": img.get("type", "output")}
            data = requests.get(f"{COMFY_URL}/view", params=params, timeout=60).content
            return data
    raise RuntimeError("No image found in ComfyUI outputs for prompt " + pid)


# ----------------------------------------------------------------------------
# BATCH + REVIEW GATE
# ----------------------------------------------------------------------------

def seam_score(equirect: np.ndarray) -> float:
    """Cheap left<->right wrap mismatch metric. Lower = better. Auto-flag only."""
    left, right = equirect[:, :2].astype(float), equirect[:, -2:].astype(float)
    return float(np.mean(np.abs(left.mean(1) - right.mean(1))))


def check_equirect(equirect: np.ndarray) -> tuple[bool, str]:
    """Validate the image is plausibly a true equirect before meshing.

    Hard signal: left/right edges must wrap (seam mismatch <= SEAM_MAX), else the
    cubemap split is geometrically wrong and the room will be distorted.
    Soft signal: in a real equirect the top/bottom rows are near-single points
    (low horizontal std vs the middle) -- reported as a warning, not a failure.
    """
    g = equirect.mean(2) if equirect.ndim == 3 else equirect.astype(float)
    H, W = g.shape
    seam = float(np.mean(np.abs(g[:, :3].mean(1) - g[:, -3:].mean(1))))
    nrow = max(1, H // 50)
    top = float(np.mean([g[r].std() for r in range(nrow)]))
    bot = float(np.mean([g[r].std() for r in range(H - nrow, H)]))
    mid = float(np.mean([g[r].std() for r in range(H // 2 - 3, H // 2 + 3)]))
    pole_ratio = (top + bot) / (2 * mid + 1e-9)
    ok = seam <= SEAM_MAX
    msg = (f"seam={seam:.1f} (max {SEAM_MAX:.0f})  "
           f"pole/mid std ratio={pole_ratio:.2f} (lower is more equirect)")
    if pole_ratio > 0.8:
        msg += "  [WARN: poles look flat -- may not be a true equirect]"
    return ok, msg


def generate_batch(prompt: str, base_seed: int) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cands = []
    for i in range(BATCH):
        seed = base_seed + i
        png = submit_panorama(prompt, seed)
        p = OUTPUT_DIR / f"pano_{seed}.png"
        p.write_bytes(png)
        arr = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
        s = seam_score(arr)
        cands.append((s, p))
        print(f"  [{i+1}/{BATCH}] seed={seed}  seam={s:6.2f}  -> {p.name}")
    cands.sort(key=lambda t: t[0])  # best seam first as a hint
    print("\nReview the PNGs in", OUTPUT_DIR, "(sorted-by-seam hint above).")
    print("Lower seam score = cleaner wrap, but EYEBALL the poles/geometry too.")
    choice = input(f"Pick file to build [{cands[0][1].name}]: ").strip()
    return OUTPUT_DIR / choice if choice else cands[0][1]


# ----------------------------------------------------------------------------
# CUBEMAP + MoGe-2
# ----------------------------------------------------------------------------

# Per-face matrix mapping MoGe-2 camera coords (OpenCV: +x right, +y DOWN,
# +z forward) into py360convert world coords (+x right, +y UP, +z front).
# py360convert e2c list order is [F, R, B, L, U, D].
# Verified against py360convert utils.xyzcube() by a texture-coherence test:
# with these matrices a fused face's spherical UV samples the exact panorama
# pixel it came from (err ~0.3/255). The previous matrices assumed MoGe was
# y-up and produced a vertically flipped room whose texture didn't line up.
# If your MoGe build ever returns y-up points, negate the middle row of each.
_FACE_ROT = {
    "F": np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], float),
    "R": np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]], float),
    "B": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], float),
    "L": np.array([[0, 0, -1], [0, -1, 0], [1, 0, 0]], float),
    "U": np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], float),
    "D": np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], float),
}
_FACE_ORDER = ["F", "R", "B", "L", "U", "D"]


def to_cubemap(equirect: np.ndarray) -> dict[str, np.ndarray]:
    import py360convert
    faces = py360convert.e2c(equirect, face_w=FACE_SIZE, mode="bilinear",
                             cube_format="list")
    return {name: faces[i] for i, name in enumerate(_FACE_ORDER)}


# --- depth backends ---------------------------------------------------------

_moge_model = None

def _load_moge():
    global _moge_model
    if _moge_model is None:
        import torch
        from moge.model.v2 import MoGeModel
        _moge_model = MoGeModel.from_pretrained(MOGE_MODEL).to("cuda").eval()
    return _moge_model


def get_depth_local(face_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (points HxWx3 camera-frame metric, mask HxW bool).

    Re-unprojects MoGe's predicted z-depth with the known 90-deg cubemap
    intrinsics (REPROJECT_INTRINSICS) so faces are mutually consistent, and clips
    depth blow-outs (DEPTH_CLIP_M). MoGe frame is OpenCV: +x right, +y down,
    +z forward -- matching what _FACE_ROT expects.
    """
    import torch
    model = _load_moge()
    t = torch.from_numpy(face_rgb).permute(2, 0, 1).float().div(255).to("cuda")
    with torch.no_grad():
        out = model.infer(t)
    pts = out["points"].cpu().numpy().astype(np.float32)   # HxWx3, MoGe metric
    mask = out["mask"].cpu().numpy().astype(bool)

    if REPROJECT_INTRINSICS:
        H, W = mask.shape
        z = pts[..., 2]                                    # forward depth (metric)
        f = W * 0.5                                        # 90-deg FOV: f = (W/2)/tan(45)
        uu, vv = np.meshgrid(np.arange(W, dtype=np.float32),
                             np.arange(H, dtype=np.float32))
        x = (uu - W * 0.5) / f
        y = (vv - H * 0.5) / f
        pts = np.stack([z * x, z * y, z], axis=-1).astype(np.float32)

    if DEPTH_CLIP_M > 0:
        mask = mask & (pts[..., 2] > 0) & (pts[..., 2] <= DEPTH_CLIP_M)
    return pts, mask


def get_depth_comfy(face_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    >>> CONFIGURE: comfy-API depth path. Upload face via /upload/image, run a
    MoGe workflow that SAVES A FLOAT point map (EXR or .npy) -- an 8-bit PNG
    will quantize the geometry. Fetch it back via /view and load to HxWx3.
    Left as NotImplemented because it requires a save-EXR node in your graph.
    """
    raise NotImplementedError(
        "Add an EXR/.npy point-map save node to your MoGe workflow, then load it "
        "here. Recommended: just run this script on the Windows box with "
        'MOGE_BACKEND="local".')


def get_depth(face_rgb):
    return get_depth_local(face_rgb) if MOGE_BACKEND == "local" else get_depth_comfy(face_rgb)


# ----------------------------------------------------------------------------
# FUSE -> MESH -> UV -> GLB
# ----------------------------------------------------------------------------

def fuse_faces(faces: dict[str, np.ndarray]):
    """Unproject every face into world space and concatenate points + colors."""
    all_pts, all_col = [], []
    for name, rgb in faces.items():
        pts, mask = get_depth(rgb)
        R = _FACE_ROT[name]
        p = pts[mask]                          # Nx3 camera frame
        world = p @ R.T                        # rotate into shared world frame
        all_pts.append(world)
        all_col.append(rgb[mask].astype(float) / 255.0)
        med = float(np.median(p[:, 2])) if len(p) else 0.0
        print(f"  face {name}: {mask.sum():>8} pts  median depth={med:5.2f}m")
    pts = np.concatenate(all_pts)
    col = np.concatenate(all_col)
    r = np.linalg.norm(pts, axis=1)
    # If per-face median depths above are wildly different, MoGe's metric scale is
    # drifting between faces and we'd add per-face normalisation next.
    print(f"  fused {len(pts)} pts  radius median={np.median(r):.2f} "
          f"95th={np.percentile(r, 95):.2f} max={r.max():.2f} m")
    return pts, col


def unproject_equirect(depth: np.ndarray, rgb: np.ndarray):
    """Single equirect depth map -> ONE coherent point cloud. No cubemap, so no
    per-face seams (the source of the 'tearing'/floating surfaces).

    depth: HxW range (radial distance from centre per equirect pixel).
    rgb:   HxWx3, resized to match depth. Convention matches spherical_uv().
    """
    H, W = depth.shape
    uu, ww = np.meshgrid((np.arange(W) + 0.5) / W, (np.arange(H) + 0.5) / H)
    lon = (uu - 0.5) * 2 * np.pi          # = atan2(x, z)
    lat = (0.5 - ww) * np.pi              # = asin(y); top row -> +pi/2
    cl = np.cos(lat)
    dirs = np.stack([np.sin(lon) * cl, np.sin(lat), np.cos(lon) * cl], axis=-1)
    pts = (dirs * depth[..., None]).astype(np.float32)
    mask = np.isfinite(depth) & (depth > 0)
    if DEPTH_CLIP_M > 0:
        mask &= depth <= DEPTH_CLIP_M
    p = pts[mask]
    c = rgb[mask].astype(np.float32) / 255.0
    r = np.linalg.norm(p, axis=1)
    print(f"  equirect unproject: {len(p)} pts  radius median={np.median(r):.2f} "
          f"95th={np.percentile(r, 95):.2f} max={r.max():.2f}")
    return p, c


def load_equirect_depth(depth_path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    """Load a float depth map (.npy preferred; 16-bit PNG tolerated) and resize to
    the panorama's (H, W). .npy keeps full precision -- an 8-bit PNG would quantise
    the geometry, so produce float depth from your 360 model (see da360_depth.py)."""
    import cv2
    if depth_path.suffix == ".npy":
        d = np.load(depth_path).astype(np.float32)
    else:
        d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    H, W = target_hw
    if d.shape != (H, W):
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
    return d


def reconstruct(points: np.ndarray, colors: np.ndarray):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    n0 = len(pcd.points)
    # Drop noisy isolated points so Poisson doesn't grow spikes/stalactites.
    if OUTLIER_NB > 0:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=OUTLIER_NB,
                                                std_ratio=OUTLIER_STD)
        print(f"  outlier removal: {n0} -> {len(pcd.points)} pts")
    pcd.estimate_normals()
    pcd.orient_normals_towards_camera_location(np.zeros(3))  # normals face inward->flip
    pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH)
    # Trim the low-density Poisson "skirt" -- the thin ballooned surface Poisson
    # invents over openings and far from samples.
    if DENSITY_TRIM_QUANTILE > 0:
        densities = np.asarray(densities)
        cutoff = np.quantile(densities, DENSITY_TRIM_QUANTILE)
        mesh.remove_vertices_by_mask(densities < cutoff)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    # Drop small floating islands (blobs disconnected from the main room).
    if MIN_COMPONENT_FRAC > 0:
        clusters, counts, _ = mesh.cluster_connected_triangles()
        clusters = np.asarray(clusters)
        counts = np.asarray(counts)
        if counts.size:
            keep_min = counts.max() * MIN_COMPONENT_FRAC
            mesh.remove_triangles_by_mask(counts[clusters] < keep_min)
            mesh.remove_unreferenced_vertices()
            print(f"  islands: kept {int((counts >= keep_min).sum())}/{counts.size}")
    # Smooth the blobby surface (Taubin barely shrinks, unlike Laplacian).
    if SMOOTH_ITERS > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=SMOOTH_ITERS)
    if len(mesh.triangles) > TARGET_TRIS:
        mesh = mesh.simplify_quadric_decimation(TARGET_TRIS)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()
    print(f"  final mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")
    return mesh


def spherical_uv(vertices: np.ndarray) -> np.ndarray:
    """Direction from center -> equirect UV. The panorama is the texture."""
    v = vertices / (np.linalg.norm(vertices, axis=1, keepdims=True) + 1e-9)
    u = 0.5 + np.arctan2(v[:, 0], v[:, 2]) / (2 * np.pi)
    w = 0.5 - np.arcsin(np.clip(v[:, 1], -1, 1)) / np.pi
    return np.stack([u, w], axis=1)


def export_glb(mesh, equirect_png_path: Path, out_path: Path):
    import trimesh
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    uv = spherical_uv(V)
    img = Image.open(equirect_png_path).convert("RGB")
    tm = trimesh.Trimesh(
        vertices=V, faces=F,
        visual=trimesh.visual.TextureVisuals(
            uv=uv, image=img),
        process=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tm.export(out_path)
    print("  wrote", out_path)
    return out_path


# ----------------------------------------------------------------------------
# DRIVER
# ----------------------------------------------------------------------------

def build(prompt: str, seed: int, name: str, auto: bool, force: bool = False,
          pano: str | None = None, depth: str | None = None):
    if pano:
        pano_path = Path(pano)
        print(f"\n[1/5] using existing panorama: {pano_path}")
        if not pano_path.exists():
            raise SystemExit(f"--pano file not found: {pano_path}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    else:
        print(f"\n[1/5] generating {BATCH} panorama candidate(s)...")
        if auto:
            png = submit_panorama(prompt, seed)
            pano_path = OUTPUT_DIR / f"pano_{seed}.png"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            pano_path.write_bytes(png)
        else:
            pano_path = generate_batch(prompt, seed)

    print("[2/5] loading panorama...")
    equirect = np.array(Image.open(pano_path).convert("RGB"))
    ok, msg = check_equirect(equirect)
    print(f"  equirect check: {msg}")
    if not ok and not force:
        raise SystemExit(
            "\nABORT: panorama does not wrap seamlessly, so it isn't a valid "
            "equirect -- the room would come out distorted. Use a true-equirect "
            "generator (DiT360). Re-run with --force to build anyway.")

    if depth:
        # SEAM-FREE path: one native-360 depth map -> one coherent point cloud.
        print("[3/5] native-360 depth -> unproject (no cubemap seams)...")
        import cv2
        d = load_equirect_depth(Path(depth), equirect.shape[:2])
        rgb = cv2.resize(equirect, (d.shape[1], d.shape[0]),
                         interpolation=cv2.INTER_AREA) if equirect.shape[:2] != d.shape else equirect
        pts, col = unproject_equirect(d, rgb)
    else:
        # Cubemap path: 6 per-face MoGe estimates (can seam/tear at face edges).
        print("[3/5] cubemap split + MoGe-2 per face + fusing...")
        faces = to_cubemap(equirect)
        pts, col = fuse_faces(faces)

    print("[4/5] Poisson reconstruction...")
    mesh = reconstruct(pts, col)

    print("[5/5] exporting GLB...")
    glb = export_glb(mesh, pano_path, OUTPUT_DIR / f"{name}.glb")

    if GODOT_PROJECT_DIR:
        GODOT_PROJECT_DIR.mkdir(parents=True, exist_ok=True)
        dst = GODOT_PROJECT_DIR / glb.name
        shutil.copy2(glb, dst)
        print("  copied to Godot:", dst)
    print("\nDone:", glb)


def main():
    ap = argparse.ArgumentParser(description="AI panorama -> Godot room shell")
    ap.add_argument("prompt", nargs="?", default="",
                    help="room description (omit when using --pano)")
    ap.add_argument("--name", default="room", help="output GLB base name")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pano", default=None,
                    help="mesh an EXISTING panorama PNG instead of generating one "
                         "(skips ComfyUI entirely)")
    ap.add_argument("--depth", default=None,
                    help="float equirect depth map (.npy) from a native-360 model; "
                         "uses the seam-free unproject path instead of the cubemap")
    ap.add_argument("--auto", action="store_true",
                    help="skip the review gate (generate 1, build it)")
    ap.add_argument("--force", action="store_true",
                    help="build even if the panorama fails the equirect seam check")
    args = ap.parse_args()
    if not args.pano and not args.prompt:
        ap.error("provide a prompt, or --pano PATH to mesh an existing panorama")
    build(args.prompt, args.seed, args.name, args.auto, args.force,
          args.pano, args.depth)


if __name__ == "__main__":
    main()
