"""Phase B of ComfyUI novel-view generation: fill the disocclusion holes in the
rendered partial views (from novel_views.py) into complete frames, via ComfyUI.

stdlib + cv2 only -- imports NO torch, so it never holds a CUDA context next to
ComfyUI (same rule as comfy_inpaint.py; see [[comfyui-cuda-crash]]). Reuses that
module's HTTP client (upload / post_prompt / wait / fetch_image).

Default mode = 'refine': classically pre-fill each hole (cv2 Telea -> a smooth,
plausible continuation of the surrounding pixels) then run a LOW-denoise img2img
ONLY inside the holes (SetLatentNoiseMask). This adds realistic grain/lighting
to the revealed region WITHOUT re-inventing the whole room -- appropriate for the
moderate-baseline / parallax regime novel_views.py targets. 'full' (high-denoise
VAEEncodeForInpaint) is kept for comparison but hallucinates on big holes.

ComfyUI must be running on :7821 (start it yourself; don't run a torch GPU job at
the same time).

    python novel_views.py photo.png --emit-dir views     # Phase A (GPU, exits)
    python novel_views_comfy.py views                     # Phase B (this)
"""
import argparse
import glob
import json
import os
import uuid

import cv2
import numpy as np
from PIL import Image

# Reuse the proven ComfyUI HTTP client (no torch in it).
from comfy_inpaint import BASE, upload, post_prompt, wait, fetch_image
import urllib.request

POS = ("interior room photo, consistent room, walls floor ceiling and furniture, "
       "photoreal, even ambient lighting, sharp, coherent perspective")
NEG = ("extra room, duplicated furniture, warped geometry, distorted walls, "
       "people, text, watermark, seam, blur, lowres, jpeg artifacts")


def refine_workflow(base_name, mask_name, ckpt, steps, cfg, denoise, seed):
    """Low-denoise img2img confined to the (pre-filled) holes."""
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": POS, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["4", 1]}},
        "10": {"class_type": "LoadImage", "inputs": {"image": base_name}},
        "11": {"class_type": "LoadImageMask",
               "inputs": {"image": mask_name, "channel": "red"}},
        "12": {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["4", 2]}},
        "13": {"class_type": "SetLatentNoiseMask",
               "inputs": {"samples": ["12", 0], "mask": ["11", 0]}},
        "3": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "dpmpp_2m", "scheduler": "karras",
                         "denoise": denoise, "model": ["4", 0], "positive": ["6", 0],
                         "negative": ["7", 0], "latent_image": ["13", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "wb_novelview", "images": ["8", 0]}},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="emit-dir from novel_views.py (view_*_partial/_holes)")
    ap.add_argument("--ckpt", default="epicrealismXL_vxviiCrystalclear.safetensors")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--denoise", type=float, default=0.4,
                    help="refine denoise; higher fills bigger holes but drifts")
    ap.add_argument("--prefill", type=int, default=6, help="cv2 Telea radius px")
    ap.add_argument("--grow", type=int, default=6, help="dilate hole mask px")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    try:
        urllib.request.urlopen(f"{BASE}/system_stats", timeout=5)
    except Exception:
        raise SystemExit(f"ComfyUI not reachable at {BASE}. Start it first "
                         "(and don't run a torch GPU job at the same time).")

    frames_dir = os.path.join(args.dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    partials = sorted(glob.glob(os.path.join(args.dir, "view_*_partial.png")))
    if not partials:
        raise SystemExit(f"no view_*_partial.png in {args.dir} (run novel_views.py).")

    for pth in partials:
        stem = os.path.basename(pth)[:-len("_partial.png")]   # view_<i>
        mask_pth = os.path.join(args.dir, f"{stem}_holes.png")
        if not os.path.isfile(mask_pth):
            print(f"  {stem}: no hole mask, skipping"); continue

        partial = np.array(Image.open(pth).convert("RGB"))
        holes = (np.array(Image.open(mask_pth).convert("L")) > 128).astype(np.uint8)
        if args.grow:
            holes = cv2.dilate(holes, np.ones((args.grow, args.grow), np.uint8))
        # Classical pre-fill so img2img has a smooth base, not black holes.
        base = cv2.inpaint(partial, holes, args.prefill, cv2.INPAINT_TELEA)
        base_pth = os.path.join(args.dir, f"{stem}_prefill.png")
        Image.fromarray(base).save(base_pth)
        mask_img = os.path.join(args.dir, f"{stem}_mask.png")
        Image.fromarray((holes * 255).astype(np.uint8)).save(mask_img)

        print(f"  {stem}: {100*holes.mean():.1f}% holes, uploading...")
        wf = refine_workflow(upload(base_pth), upload(mask_img), args.ckpt,
                             args.steps, args.cfg, args.denoise, args.seed)
        info = wait(post_prompt(wf, str(uuid.uuid4())))
        out = os.path.join(frames_dir, f"{stem}.png")
        fetch_image(info, out)
        print(f"  {stem}: -> {out}")

    # poses.json (from Phase A) already lists the base photo as view_0; the filled
    # frames now sit beside it in frames/ ready for build_room_multiview --images.
    pj = os.path.join(args.dir, "poses.json")
    if os.path.isfile(pj):
        with open(pj) as f:
            meta = json.load(f)
        print(f"\ndone. {len(meta['views'])} views in {frames_dir}/.")
    print(f"Next: python build_room_multiview.py --images {frames_dir} --name myroom")


if __name__ == "__main__":
    main()
