#!/usr/bin/env python3
"""comfy_inpaint.py -- fill rectified wall holes with SDXL inpaint via ComfyUI.

Phase B of the wall-completion pipeline (Phase A = wall_inpaint.py --emit-dir,
which writes <label>_wall.png + <label>_holes.png). This script is **stdlib only,
imports no torch**, so it never holds a CUDA context next to ComfyUI -- it just
uploads the rectified wall + hole mask, runs a regular-checkpoint inpaint graph
(VAEEncodeForInpaint + KSampler), and saves <label>_filled.png.

ComfyUI must be running on :7821 (start it yourself; this only talks HTTP).

    python wall_inpaint.py photo.png --emit-dir walls   # Phase A (GPU, then exits)
    python comfy_inpaint.py walls                        # Phase B (this, no torch)
"""
import argparse
import glob
import json
import mimetypes
import os
import time
import urllib.parse
import urllib.request
import uuid

import cv2          # opencv only, no torch -> safe to run beside ComfyUI
import numpy as np
from PIL import Image

BASE = "http://127.0.0.1:7821"

POS = ("plain interior wall, smooth painted drywall, flat matte paint, even soft "
       "ambient lighting, subtle wall texture, photoreal, consistent colour")
NEG = ("furniture, sofa, chair, table, bookshelf, lamp, window, door, curtain, "
       "picture frame, painting, poster, plant, clutter, object, shadow, people, "
       "text, watermark, seam, blurry, lowres, jpeg artifacts")


def _multipart(image_path, overwrite=True):
    boundary = "----wbformboundary" + uuid.uuid4().hex
    name = os.path.basename(image_path)
    ctype = mimetypes.guess_type(name)[0] or "image/png"
    with open(image_path, "rb") as f:
        data = f.read()
    parts = []
    def field(headers, body):
        parts.append(("--" + boundary + "\r\n" + headers + "\r\n\r\n").encode()
                     + body + b"\r\n")
    field(f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'
          f"Content-Type: {ctype}", data)
    field('Content-Disposition: form-data; name="overwrite"',
          b"true" if overwrite else b"false")
    parts.append(("--" + boundary + "--\r\n").encode())
    return b"".join(parts), boundary


def upload(image_path):
    """Upload an image to ComfyUI's input dir; return the server-side filename."""
    body, boundary = _multipart(image_path)
    req = urllib.request.Request(
        f"{BASE}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    info = json.load(urllib.request.urlopen(req, timeout=60))
    sub = info.get("subfolder", "")
    return (sub + "/" + info["name"]) if sub else info["name"]


def post_prompt(workflow, client_id):
    data = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(f"{BASE}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=30))
    if "prompt_id" not in resp:
        raise RuntimeError(f"ComfyUI rejected the workflow: {resp}")
    return resp["prompt_id"]


def wait(prompt_id, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with urllib.request.urlopen(f"{BASE}/history/{prompt_id}", timeout=30) as r:
            hist = json.load(r)
        if prompt_id in hist:
            return hist[prompt_id]
        time.sleep(1.5)
    raise TimeoutError("inpaint timed out")


def fetch_image(info, out_path):
    for node_out in info["outputs"].values():
        for img in node_out.get("images", []):
            q = urllib.parse.urlencode({"filename": img["filename"],
                                        "subfolder": img.get("subfolder", ""),
                                        "type": img.get("type", "output")})
            with urllib.request.urlopen(f"{BASE}/view?{q}", timeout=60) as r:
                data = r.read()
            with open(out_path, "wb") as f:
                f.write(data)
            return img["filename"]
    raise RuntimeError("no image in history outputs")


def _common(ckpt):
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": POS, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "wb_wallfill", "images": ["8", 0]}},
    }


def full_inpaint_workflow(wall_name, mask_name, ckpt, steps, cfg, seed, grow):
    """Full denoise in the masked region (VAEEncodeForInpaint). A general checkpoint
    HALLUCINATES furniture here over large masks -- kept for comparison."""
    wf = _common(ckpt)
    wf.update({
        "10": {"class_type": "LoadImage", "inputs": {"image": wall_name}},
        "11": {"class_type": "LoadImageMask",
               "inputs": {"image": mask_name, "channel": "red"}},
        "12": {"class_type": "VAEEncodeForInpaint",
               "inputs": {"pixels": ["10", 0], "vae": ["4", 2],
                          "mask": ["11", 0], "grow_mask_by": grow}},
        "3": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "dpmpp_2m", "scheduler": "karras",
                         "denoise": 1.0, "model": ["4", 0], "positive": ["6", 0],
                         "negative": ["7", 0], "latent_image": ["12", 0]}},
    })
    return wf


def refine_workflow(base_name, mask_name, ckpt, steps, cfg, denoise, seed):
    """LOW-denoise refine: the holes are already classically pre-filled to a smooth
    wall, so img2img at low denoise (only inside the mask, via SetLatentNoiseMask)
    adds realistic wall grain/lighting WITHOUT re-inventing furniture."""
    wf = _common(ckpt)
    wf.update({
        "10": {"class_type": "LoadImage", "inputs": {"image": base_name}},
        "11": {"class_type": "LoadImageMask",
               "inputs": {"image": mask_name, "channel": "red"}},
        "12": {"class_type": "VAEEncode",
               "inputs": {"pixels": ["10", 0], "vae": ["4", 2]}},
        "13": {"class_type": "SetLatentNoiseMask",
               "inputs": {"samples": ["12", 0], "mask": ["11", 0]}},
        "3": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "dpmpp_2m", "scheduler": "karras",
                         "denoise": denoise, "model": ["4", 0], "positive": ["6", 0],
                         "negative": ["7", 0], "latent_image": ["13", 0]}},
    })
    return wf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="dir with <label>_wall.png + <label>_holes.png")
    ap.add_argument("--ckpt", default="epicrealismXL_vxviiCrystalclear.safetensors")
    ap.add_argument("--mode", choices=["refine", "full"], default="refine",
                    help="refine (classical pre-fill + low-denoise; clean walls) or "
                         "full (VAEEncodeForInpaint; hallucinates furniture)")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--denoise", type=float, default=0.35,
                    help="refine-mode denoise (0.25-0.45 sweet spot)")
    ap.add_argument("--grow", type=int, default=8, help="grow mask by N px")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    try:
        urllib.request.urlopen(f"{BASE}/system_stats", timeout=5)
    except Exception:
        raise SystemExit(f"ComfyUI not reachable at {BASE}. Start it first "
                         "(and don't run a torch GPU job at the same time).")

    walls = sorted(glob.glob(os.path.join(args.dir, "*_wall.png")))
    if not walls:
        raise SystemExit(f"No *_wall.png in {args.dir} (run wall_inpaint.py --emit-dir).")

    for wall_path in walls:
        label = os.path.basename(wall_path)[:-len("_wall.png")]
        mask_path = os.path.join(args.dir, f"{label}_holes.png")
        if not os.path.isfile(mask_path):
            print(f"  {label}: no hole mask, skipping"); continue

        if args.mode == "refine":
            # Classical pre-fill (cv2, no torch) -> smooth wall base for img2img.
            warp = np.array(Image.open(wall_path).convert("RGB"))
            holes = (np.array(Image.open(mask_path).convert("L")) > 128).astype(np.uint8)
            base = cv2.inpaint(warp, holes, 6, cv2.INPAINT_TELEA)
            base_path = os.path.join(args.dir, f"{label}_base.png")
            Image.fromarray(base).save(base_path)
            print(f"  {label}: pre-filled, uploading...")
            wf = refine_workflow(upload(base_path), upload(mask_path), args.ckpt,
                                 args.steps, args.cfg, args.denoise, args.seed)
        else:
            print(f"  {label}: uploading...")
            wf = full_inpaint_workflow(upload(wall_path), upload(mask_path),
                                       args.ckpt, args.steps, args.cfg, args.seed,
                                       args.grow)
        pid = post_prompt(wf, str(uuid.uuid4()))
        info = wait(pid)
        out = os.path.join(args.dir, f"{label}_filled.png")
        fetch_image(info, out)
        print(f"  {label}: -> {out}")

    print(f"done ({args.mode}). Compare <label>_wall.png vs <label>_filled.png.")


if __name__ == "__main__":
    main()
