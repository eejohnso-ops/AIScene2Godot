# Running MIDI (+ texturing) on Windows / Blackwell (RTX 50-series)

MIDI and its texturing stack (MV-Adapter, nvdiffrast) are research code that
assumes Linux and an older CUDA. On Windows with a Blackwell GPU (RTX 5080/5090,
compute `sm_120`) almost every step needs a workaround. This is the full list,
in order, that actually got it working — written down so you don't lose a day to
each one.

> Environment used: Windows, RTX 5090 (sm_120), `uv`-managed venv, Python 3.10,
> PyTorch cu128. Adjust versions to taste, but the *shapes* of these problems are
> stable.

## 0. The recurring theme: torch version pinning

Blackwell needs **CUDA 12.8 (`cu128`) PyTorch ≥ 2.7**, but several extensions
(`torch-cluster`, `nvdiffrast`) compile against a **specific** torch version. The
cu128 index will happily install the *latest* torch (e.g. 2.11), which then
mismatches those extensions. The fix throughout is to **pin torch to the version
your extensions were built for** (we used `torch==2.8.0+cu128`) and re-pin it any
time another package bumps it.

```bash
uv pip install "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cu128
```

Verify Blackwell actually works (not just `is_available()`):
```bash
python -c "import torch; x=torch.randn(8,device='cuda'); print((x*2).sum())"
```
A number with no "no kernel image is available" error = good.

## 1. torch-cluster: ABI mismatch ("Entry Point Not Found")

`import torch_cluster` popping a Windows "Entry Point Not Found" dialog means the
prebuilt wheel doesn't match your torch. Install the PyG wheel matching your
*exact* torch version:
```bash
uv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```
If torch is newer than the available wheel, downgrade torch (see §0).

## 2. MV-Adapter: don't let its dependencies install

`pip install git+.../MV-Adapter` fails ("unsatisfiable" + build error) because its
`install_requires` is huge, unpinned, and includes `cvcuda_cu12` (no Windows
wheel) and an unpinned `gradio` that fights MIDI's pin. Install **just the
package**, then add only the genuinely-missing extras:
```bash
# clone, then copy the package folder into the MIDI repo root (bypasses the build):
git clone https://github.com/huanngzh/MV-Adapter
# copy MV-Adapter/mvadapter  ->  MIDI-3D/mvadapter
uv pip install controlnet_aux timm kornia sentencepiece spandrel
```

## 3. nvdiffrast: the hardest part

`nvdiffrast` compiles a CUDA extension (`_nvdiffrast_c`). You need:

- **Visual Studio Build Tools** with the *Desktop development with C++* workload
  (provides `cl.exe`).
- **CUDA Toolkit 12.8** (provides `nvcc`). **Custom install — UNCHECK the Display
  Driver**, or it may downgrade your Blackwell driver and break the GPU.

Then build from the **x64 Native Tools Command Prompt for VS** (so `cl.exe` is on
PATH), with the critical env var:
```bat
set DISTUTILS_USE_SDK=1
uv pip install ninja
uv pip install --no-build-isolation C:\path\to\nvdiffrast
```
`DISTUTILS_USE_SDK=1` is the fix for the build dying with *"the VC environment is
activated but DISTUTILS_USE_SDK is not set."* Without it, torch refuses to use the
already-active MSVC env.

> nvdiffrast's `__init__` does `version('nvdiffrast')`, so a folder-copy alone
> won't import — it needs real package metadata, i.e. a proper install. The build
> above provides it.

## 4. triton

Diffusers/MV-Adapter want `triton`, which was Linux-only for years. Use the
Windows port:
```bash
uv pip install triton-windows
```
Re-check torch afterward (§0) in case it got bumped.

## 5. pymeshlab API rename

`ImportError: cannot import name 'Percentage' from 'pymeshlab'` — newer pymeshlab
renamed `Percentage` to `PercentageValue`. Patch the import sites
(`mvadapter/utils/mesh_utils/mesh_process.py`, `midi/utils/mesh_process.py`):
```python
try:
    from pymeshlab import Percentage
except ImportError:
    from pymeshlab import PercentageValue as Percentage
```

## 6. cvcuda → OpenCV

`cvcuda` (NVIDIA CV-CUDA) has no Windows wheel. It's only used in
`mvadapter/utils/mesh_utils/cv_ops.py` for inpaint + erode/dilate on texture
maps — all of which have OpenCV equivalents. A drop-in `cv2` replacement is in
this repo at [`patches/mvadapter_cv_ops_opencv.py`](../patches/mvadapter_cv_ops_opencv.py)
— copy it over `mvadapter/utils/mesh_utils/cv_ops.py`.

## 7. Runtime assets

- **`big-lama.pt`** (LaMa inpainting, for texture seams) — MIDI's script prints
  the URL; grab it to `checkpoints/big-lama.pt`:
  ```bash
  curl -L "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt" -o checkpoints/big-lama.pt
  ```
- **`gltflib`** (GLB export): `uv pip install gltflib`.

## 8. gradio demo won't launch (localhost / frpc)

`gradio_demo.py` may fail its localhost check, and `share=True` downloads `frpc`,
which antivirus flags as riskware. Skip the web UI entirely — use MIDI's CLI:
```bash
python -m scripts.inference_midi --rgb img.png --seg seg.png --output-dir ./out
python -m scripts.image_to_textured_scene --rgb_image img.png --seg_image seg.png --output out_tex
```
(The example data ships with segmentation maps, so you can smoke-test without
running Grounded-SAM.)

## VRAM

Textured generation peaks around **30 GB** — fits a 32 GB 5090 only if you close
the Godot editor, browsers (hardware accel), and anything else on the GPU first.

---

### Order of operations that worked

torch (cu128, pinned) → torch-cluster (matching wheel) → mvadapter (package only) →
extras (controlnet_aux, timm, kornia, sentencepiece, spandrel) → triton-windows →
VS Build Tools + CUDA Toolkit → nvdiffrast (`DISTUTILS_USE_SDK=1`, native-tools
shell) → pymeshlab patch → cvcuda→cv2 patch → big-lama → gltflib → run the CLI.
