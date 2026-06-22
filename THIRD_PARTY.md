# Third-party models & licenses

AIScene2Godot is **glue and tooling**. It bundles none of the models below — you
install them yourself from their sources. Their licenses are theirs, and several
are **non-commercial**. Check each before using any output commercially.

| Component | Used for | License (verify upstream) |
| --- | --- | --- |
| [MIDI-3D](https://github.com/VAST-AI-Research/MIDI-3D) | image → compositional object meshes | Apache-2.0 (code); check model weights |
| [MV-Adapter](https://github.com/huanngzh/MV-Adapter) | multi-view texturing | check upstream |
| [nvdiffrast](https://github.com/NVlabs/nvdiffrast) | differentiable rasterization (texture bake) | **NVIDIA Source Code License — non-commercial** |
| [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | panorama generation (experimental) | **non-commercial license** |
| [DiT360 / ComfyUI-DiT360Plus](https://github.com/thomashollier/ComfyUI-DiT360Plus) | seamless 360 panoramas (experimental) | Apache-2.0 (nodes); check LoRA weights |
| [Depth Anywhere](https://github.com/albert100121/Depth-Anywhere) / UniFuse | 360 depth (experimental) | Apache-2.0 / check weights |
| [MoGe-2](https://github.com/microsoft/MoGe) | per-face metric depth (experimental) | check upstream |
| [LaMa (big-lama)](https://github.com/advimman/lama) | texture-seam inpainting | check upstream |
| [Godot Engine](https://godotengine.org/) | the viewer runtime | MIT |

## The practical bottom line

- **nvdiffrast and FLUX.1-dev are non-commercial.** If your pipeline touches the
  MIDI *texturing* path (nvdiffrast) or the experimental panorama path (FLUX),
  your output is encumbered for commercial use. The MIDI *geometry* path avoids
  nvdiffrast.
- This repo's own code is MIT (see `LICENSE`) — but that only covers the viewer,
  `to_godot.py`, and the `experimental/` scripts, not anything they orchestrate.
- When in doubt, read the upstream license. This table is a pointer, not legal
  advice, and licenses change.
