#!/usr/bin/env python3
"""Minimal ComfyUI txt2img client (stdlib only). Generates a perspective
living-room interior with epicrealismXL and saves the PNG to --out."""
import argparse
import json
import time
import urllib.request
import urllib.parse
import uuid
import random

BASE = "http://127.0.0.1:7821"

POS = ("interior photograph of a furnished living room, eye-level perspective, "
       "wide angle, a fabric sofa, a wooden coffee table, an armchair, a tall "
       "bookshelf, a floor lamp, a patterned area rug, framed art on the wall, "
       "large window with soft daylight, hardwood floor, realistic, photoreal, "
       "sharp focus, well lit, professional interior design photography")
NEG = ("blurry, low quality, distorted, fisheye, warped, watermark, text, "
       "people, person, hands, cluttered, deformed, cartoon, illustration, "
       "lowres, jpeg artifacts")


def post_prompt(workflow, client_id):
    data = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(f"{BASE}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))["prompt_id"]


def wait(prompt_id, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with urllib.request.urlopen(f"{BASE}/history/{prompt_id}", timeout=30) as r:
            hist = json.load(r)
        if prompt_id in hist:
            return hist[prompt_id]
        time.sleep(2)
    raise TimeoutError("generation timed out")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="epicrealismXL_vxviiCrystalclear.safetensors")
    ap.add_argument("--width", type=int, default=1216)
    ap.add_argument("--height", type=int, default=832)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=6.5)
    ap.add_argument("--seed", type=int, default=random.randint(1, 2**31))
    args = ap.parse_args()

    wf = {
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": args.ckpt}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": POS, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": NEG, "clip": ["4", 1]}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": args.width, "height": args.height,
                         "batch_size": 1}},
        "3": {"class_type": "KSampler",
              "inputs": {"seed": args.seed, "steps": args.steps,
                         "cfg": args.cfg, "sampler_name": "dpmpp_2m",
                         "scheduler": "karras", "denoise": 1.0,
                         "model": ["4", 0], "positive": ["6", 0],
                         "negative": ["7", 0], "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "worldbuilder_demo",
                         "images": ["8", 0]}},
    }

    client_id = str(uuid.uuid4())
    print(f"ckpt={args.ckpt}  {args.width}x{args.height}  steps={args.steps}  "
          f"cfg={args.cfg}  seed={args.seed}")
    pid = post_prompt(wf, client_id)
    print(f"prompt_id={pid}  waiting...")
    info = wait(pid)
    name = fetch_image(info, args.out)
    print(f"saved {name} -> {args.out}")


if __name__ == "__main__":
    main()
