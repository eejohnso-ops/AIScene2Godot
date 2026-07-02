# Experiments: multi-view from a single photo (and where it tops out)

One photo of a room sees only ~20% of any wall cleanly ([flat-wall mask
findings](../wall_mask.py)); everything behind the sofa, the side walls, the wall
behind the camera — occluded or absent. This track asked: **can AI-generated
extra views break that ceiling?** The answer is a precise *mostly no*, and the
boundary is worth recording. As with the [panorama experiments](panorama-experiments.md),
the lesson is the deliverable. The track is **parked**; the code stays in the repo
root as documented experiments.

## What was tried, in order

### 1. Reprojection novel views → VGGT fusion — fails on first principles

`novel_views.py` renders the base photo's depth-unprojected point cloud from new
camera poses (clean parallax de-occlusion, 12–28% holes exactly where they should
be), `novel_views_comfy.py` refines the holes with low-denoise ComfyUI inpainting,
and `mv_register.py` fuses the views with [VGGT](https://github.com/facebookresearch/vggt)
(pose-free multi-view reconstruction — pure PyTorch, runs on Blackwell in the main
venv, no compile).

The fused "room" came out **0.96 × 0.58 × 0.71 m** — a shoebox. Root cause:
every synthetic view is a *reprojection of the same single depth map*. There is
no real baseline, and triangulation with near-zero baseline is ill-conditioned,
so the scene collapses. Inpainting adds pixels but **no independent geometry**.

The texture-only variant (`wall_complete.py`: keep the proven single-view room
box, composite wall textures from the novel frames) hits the *same wall from the
texture side*: a wall texel occluded by the sofa in the base photo has **no real
pixel in any reprojected view either** — it's still sofa-covered or a hole. What
looked like recovered coverage was projector bleed (furniture smeared onto the
wall plane).

**Single-image novel-view synthesis adds no real multi-view information. It can
only hallucinate.**

### 2. Camera-controlled video generation — the generation half works

[Wan2.1-Fun-Camera-Control 14B](https://huggingface.co/alibaba-pai/Wan2.1-Fun-Camera-Control-14B)
(image + camera trajectory → video), driven through ComfyUI via
`comfy_workflows/room_orbit_api.json` + `video_to_frames.py`, is a different
regime: a video model *invents coherent new content* as the camera moves. An
orbit around the demo living room revealed a real door, a second window, and the
side walls — consistently across frames — and VGGT recovered a clean 36° camera
arc with healthy baseline from them.

But VGGT is still the wrong tool for the room *shell*: a horizontal orbit has
**no vertical parallax**, so VGGT cannot constrain height and the box flattens
(0.09–0.14 m tall). Better frames don't fix it — it's intrinsic to the motion.
Meanwhile DepthAnything's monocular-metric depth nails the 2.8 m ceiling from
the single base photo, because it extrapolates from indoor priors.

### 3. The hybrid (`wall_complete_mv.py`) — right architecture, alignment too crude

So: DepthAnything single view → room box (geometry); VGGT-posed orbit frames →
wall-texture completion, with a per-view VGGT-depth **occlusion gate** so
furniture in an orbit frame can't bleed onto the wall (fixing experiment 1's
bleed). Coverage genuinely improved (left wall: 20% real pixels → 53%, including
a window the orbit revealed) with no bleed.

**Why it was parked:** the projected features land *skewed, at odd angles*. The
DA box and the VGGT reconstruction are two independent estimates of the same
room, reconciled here by a single scalar (median-depth ratio). A wall texel sits
on the *DA* box plane, not on *VGGT's* surface for that wall — and projecting a
point that's slightly off the true surface into an oblique camera shifts the
sampled pixel laterally. Straight window frames shear; each orbit frame errs
differently, so stitched texels ghost. The math is internally correct; the
**alignment model is too crude for texture projection**, where even 2–3° of
pose/plane error is visually loud on a flat wall.

## If this is ever revived

- **Don't extend the pose-chain projection.** Each wall is a plane, so the
  correct wall-texture → frame mapping is a **homography**; estimate it directly
  by feature-matching the base photo's flat-wall region against each orbit frame
  instead of trusting scalar-scaled poses. (Caveat: video-gen frames are only
  *approximately* geometrically consistent, so matching may be shaky.)
- **Real photos change everything.** With 3–8 genuinely independent photos of a
  room, VGGT geometry works and `build_room_multiview.py --images` already
  consumes them. The machinery here (`mv_register.py`, the occlusion gate,
  view-aware projection in `room_from_image.py`) was built correctly and is
  valuable in that regime.

## Still useful from this branch

- **`comfy_workflows/room_orbit_api.json`** — working Wan-Fun-Camera orbit
  generation via the ComfyUI API (832×576, bf16, 30 steps; the SwarmUI ComfyUI
  port is dynamic). The "AI path to multiple consistent images of the same room"
  question is answered: **yes, this works.**
- **`mv_register.py`** — VGGT wrapper (poses + intrinsics + depth + fused cloud,
  npz bundles) that runs on Windows/Blackwell in the main venv.
- **`novel_views.py`** — depth-cloud reprojection with hole masks; correct
  parallax de-occlusion, useful wherever known-pose partial views are needed.

## The product path, unchanged

For the single-photo pipeline the honest best remains: **single-view geometry
(DepthAnything box) + walls from the photo's genuine flat-wall pixels, with
sampled flat colour (plain walls) or rectified inpainting (patterned walls)**.
That projection goes through the *same* camera that took the photo, so it is
exact — no skew.
