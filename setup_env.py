#!/usr/bin/env python3
"""
setup_env.py -- automated environment setup for AIScene2Godot.

Handles the dependency gauntlet documented in docs/midi-windows-blackwell-setup.md:
    - Core deps (room shell pipeline)
    - SegFormer (surface segmentation)
    - DepthAnything V2 checkpoint
    - MIDI + MV-Adapter + nvdiffrast (3D object generation)
    - All patches (pymeshlab, cvcuda->cv2)

Run levels:
    python setup_env.py              # core deps only (room shell)
    python setup_env.py --full       # everything including MIDI

Each step checks whether it's already done before running, so it's safe to
re-run after a partial failure.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()

# Where to clone external repos if not already present
EXTERNAL_DIR = SCRIPT_DIR.parent / "external"
MIDI_DIR = EXTERNAL_DIR / "MIDI-3D"
MV_ADAPTER_DIR = EXTERNAL_DIR / "MV-Adapter"
DEPTH_ANYTHING_DIR = EXTERNAL_DIR / "Depth-Anything-V2"

CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"


class Step:
    _count = 0
    _total = 0

    @classmethod
    def set_total(cls, n: int):
        cls._total = n
        cls._count = 0

    def __init__(self, name: str):
        Step._count += 1
        self.name = name
        self.n = Step._count
        self.t0 = time.time()
        print(f"\n{'─' * 60}")
        print(f"[{self.n}/{Step._total}] {name}")
        print(f"{'─' * 60}")

    def skip(self, reason: str):
        print(f"  SKIP: {reason}")

    def info(self, msg: str):
        print(f"  {msg}")

    def done(self, msg: str = ""):
        elapsed = time.time() - self.t0
        suffix = f"  {msg}" if msg else ""
        print(f"  OK.{suffix}  ({elapsed:.1f}s)")

    def warn(self, msg: str):
        print(f"  WARN: {msg}")

    def fail(self, msg: str):
        print(f"  FAIL: {msg}")


def _pip(*args, **kwargs):
    cmd = [sys.executable, "-m", "pip", "install", *args]
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _pip_quiet(*args, **kwargs):
    r = _pip(*args, **kwargs)
    if r.returncode != 0:
        print(f"    pip error: {r.stderr.strip()[:200]}")
    return r.returncode == 0


def _can_import(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def _git_clone(url: str, dest: Path):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", url, str(dest)], check=True)


def _detect_cuda_version() -> str | None:
    try:
        r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if "release" in line.lower():
                parts = line.split("release")[-1].strip().split(",")[0].strip()
                return parts
    except FileNotFoundError:
        pass
    return None


# ---------------------------------------------------------------------------
# SETUP STEPS
# ---------------------------------------------------------------------------

def setup_core(s: Step):
    """Core deps for the room shell pipeline."""
    needed = []
    for pkg in ["numpy", "scipy", "open3d", "trimesh"]:
        if not _can_import(pkg):
            needed.append(pkg)
    if not _can_import("PIL"):
        needed.append("Pillow")

    if not needed:
        s.skip("All core packages already installed")
        return

    s.info(f"Installing: {', '.join(needed)}")
    _pip_quiet(*needed)
    s.done(f"Installed {len(needed)} packages")


def setup_torch(s: Step):
    """Install PyTorch with CUDA support."""
    if _can_import("torch"):
        import torch
        cuda = torch.cuda.is_available()
        s.info(f"PyTorch {torch.__version__} already installed "
               f"(CUDA: {'yes' if cuda else 'no'})")
        if cuda:
            try:
                x = torch.randn(4, device="cuda")
                _ = (x * 2).sum()
                s.skip("PyTorch + CUDA working")
                return
            except Exception as e:
                s.warn(f"CUDA available but failed: {e}")

    cuda_ver = _detect_cuda_version()
    s.info(f"Detected CUDA toolkit: {cuda_ver or 'not found'}")

    if cuda_ver and cuda_ver.startswith("12.8"):
        index = "https://download.pytorch.org/whl/cu128"
        s.info("Installing torch for CUDA 12.8...")
        _pip_quiet("torch==2.8.0", "torchvision==0.23.0",
                   f"--index-url={index}")
    elif cuda_ver and cuda_ver.startswith("12."):
        index = "https://download.pytorch.org/whl/cu124"
        s.info(f"Installing torch for CUDA {cuda_ver}...")
        _pip_quiet("torch", "torchvision", f"--index-url={index}")
    else:
        s.info("Installing torch (CPU only — no CUDA toolkit found)...")
        _pip_quiet("torch", "torchvision")

    s.done("Verify: python -c \"import torch; print(torch.cuda.is_available())\"")


def setup_segformer(s: Step):
    """Install HuggingFace transformers for SegFormer."""
    if _can_import("transformers"):
        s.skip("transformers already installed")
        return
    s.info("Installing transformers (SegFormer)...")
    _pip_quiet("transformers")
    s.done()


def setup_depth_anything(s: Step):
    """Clone Depth-Anything-V2 and download the metric indoor checkpoint."""
    ckpt = CHECKPOINTS_DIR / "depth_anything_v2_metric_hypersim_vitl.pth"
    if ckpt.is_file():
        s.skip(f"Checkpoint exists: {ckpt}")
        return

    if not DEPTH_ANYTHING_DIR.exists():
        s.info("Cloning Depth-Anything-V2...")
        _git_clone("https://github.com/DepthAnything/Depth-Anything-V2",
                    DEPTH_ANYTHING_DIR)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    s.info("Downloading metric indoor checkpoint (~330 MB)...")
    s.info("This is a large download. If it fails, download manually from:")
    s.info("  https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large")
    s.info(f"  Save to: {ckpt}")

    url = ("https://huggingface.co/depth-anything/"
           "Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/"
           "depth_anything_v2_metric_hypersim_vitl.pth")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(ckpt))
        s.done(f"Saved to {ckpt}")
    except Exception as e:
        s.fail(f"Download failed: {e}")
        s.info(f"Download manually and save to: {ckpt}")


def setup_midi_clone(s: Step):
    """Clone MIDI-3D repo."""
    if MIDI_DIR.exists():
        s.skip(f"MIDI already cloned at {MIDI_DIR}")
        return
    s.info("Cloning MIDI-3D...")
    _git_clone("https://github.com/VAST-AI-Research/MIDI-3D", MIDI_DIR)
    s.done()


def setup_torch_cluster(s: Step):
    """Install torch-cluster matching the torch version."""
    if _can_import("torch_cluster"):
        s.skip("torch-cluster already installed")
        return

    if not _can_import("torch"):
        s.fail("PyTorch not installed — run setup_torch first")
        return

    import torch
    tv = torch.__version__.split("+")[0]
    cuda_tag = ""
    if torch.cuda.is_available():
        cv = torch.version.cuda
        if cv:
            cuda_tag = f"+cu{cv.replace('.', '')}"
    wheel_url = f"https://data.pyg.org/whl/torch-{tv}{cuda_tag}.html"

    s.info(f"Installing torch-cluster from: {wheel_url}")
    ok = _pip_quiet("torch-cluster", f"-f={wheel_url}")
    if ok:
        s.done()
    else:
        s.warn("torch-cluster install failed. Try manually:\n"
               f"  pip install torch-cluster -f {wheel_url}")


def setup_mv_adapter(s: Step):
    """Install MV-Adapter (package only, no full deps)."""
    if _can_import("mvadapter"):
        s.skip("mvadapter already importable")
        return

    if not MIDI_DIR.exists():
        s.fail("MIDI not cloned — run setup_midi_clone first")
        return

    if not MV_ADAPTER_DIR.exists():
        s.info("Cloning MV-Adapter...")
        _git_clone("https://github.com/huanngzh/MV-Adapter", MV_ADAPTER_DIR)

    mv_src = MV_ADAPTER_DIR / "mvadapter"
    mv_dst = MIDI_DIR / "mvadapter"
    if not mv_dst.exists():
        s.info(f"Copying mvadapter package to {mv_dst}")
        shutil.copytree(str(mv_src), str(mv_dst))

    s.info("Installing MV-Adapter extras...")
    _pip_quiet("controlnet_aux", "timm", "kornia", "sentencepiece", "spandrel")
    s.done()


def setup_triton(s: Step):
    """Install triton-windows."""
    if platform.system() != "Windows":
        s.skip("Not Windows, standard triton should work")
        return
    if _can_import("triton"):
        s.skip("triton already installed")
        return
    s.info("Installing triton-windows...")
    _pip_quiet("triton-windows")
    s.done()


def setup_nvdiffrast(s: Step):
    """Check/guide nvdiffrast build (requires VS Build Tools + CUDA Toolkit)."""
    if _can_import("nvdiffrast"):
        s.skip("nvdiffrast already installed")
        return

    has_cl = shutil.which("cl") or shutil.which("cl.exe")
    cuda_ver = _detect_cuda_version()

    if not has_cl:
        s.warn("cl.exe not found on PATH.")
        s.info("nvdiffrast requires Visual Studio Build Tools with the")
        s.info("'Desktop development with C++' workload.")
        s.info("Install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        s.info("")
        s.info("After installing, open 'x64 Native Tools Command Prompt for VS'")
        s.info("and re-run this script from there.")
        return

    if not cuda_ver:
        s.warn("nvcc not found. Install CUDA Toolkit 12.8:")
        s.info("  https://developer.nvidia.com/cuda-downloads")
        s.info("  IMPORTANT: Custom install, UNCHECK the Display Driver!")
        return

    s.info(f"cl.exe: found  |  CUDA: {cuda_ver}")
    s.info("Building nvdiffrast (this may take a few minutes)...")
    s.info("If this fails, open 'x64 Native Tools Command Prompt for VS' and run:")
    s.info("  set DISTUTILS_USE_SDK=1")
    s.info("  pip install ninja")
    s.info("  pip install --no-build-isolation nvdiffrast")

    env = os.environ.copy()
    env["DISTUTILS_USE_SDK"] = "1"
    _pip_quiet("ninja")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation",
         "nvdiffrast"],
        env=env, capture_output=True, text=True)
    if r.returncode == 0:
        s.done()
    else:
        s.fail("Build failed. See instructions above for manual build.")
        s.info(f"Error: {r.stderr.strip()[:300]}")


def setup_patches(s: Step):
    """Apply pymeshlab + cvcuda patches."""
    if not MIDI_DIR.exists():
        s.skip("MIDI not cloned — patches not needed yet")
        return

    applied = 0

    # pymeshlab Percentage -> PercentageValue
    for rel in ["midi/utils/mesh_process.py",
                "mvadapter/utils/mesh_utils/mesh_process.py"]:
        target = MIDI_DIR / rel
        if not target.is_file():
            continue
        text = target.read_text(encoding="utf-8")
        if "PercentageValue" in text:
            continue
        old = "from pymeshlab import Percentage"
        new = ("try:\n"
               "    from pymeshlab import Percentage\n"
               "except ImportError:\n"
               "    from pymeshlab import PercentageValue as Percentage")
        if old in text:
            target.write_text(text.replace(old, new), encoding="utf-8")
            s.info(f"Patched pymeshlab import in {rel}")
            applied += 1

    # cvcuda -> cv2
    cv_ops = MIDI_DIR / "mvadapter" / "utils" / "mesh_utils" / "cv_ops.py"
    patch_src = SCRIPT_DIR / "patches" / "mvadapter_cv_ops_opencv.py"
    if cv_ops.is_file() and patch_src.is_file():
        existing = cv_ops.read_text(encoding="utf-8")
        if "cvcuda" in existing.lower() or "import cvcuda" in existing:
            shutil.copy2(str(patch_src), str(cv_ops))
            s.info("Replaced cvcuda cv_ops.py with OpenCV version")
            applied += 1
        else:
            s.info("cv_ops.py already patched")

    if applied:
        s.done(f"{applied} patch(es) applied")
    else:
        s.skip("All patches already applied")


def setup_runtime_assets(s: Step):
    """Download big-lama.pt and install gltflib."""
    if not MIDI_DIR.exists():
        s.skip("MIDI not cloned")
        return

    _pip_quiet("gltflib", "pymeshlab")

    lama = MIDI_DIR / "checkpoints" / "big-lama.pt"
    if lama.is_file():
        s.info("big-lama.pt already present")
    else:
        lama.parent.mkdir(parents=True, exist_ok=True)
        url = ("https://github.com/Sanster/models/releases/download/"
               "add_big_lama/big-lama.pt")
        s.info("Downloading big-lama.pt (~200 MB)...")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, str(lama))
            s.info(f"Saved to {lama}")
        except Exception as e:
            s.warn(f"Download failed: {e}")
            s.info(f"Download manually: {url}")
            s.info(f"Save to: {lama}")

    s.done()


def setup_fast_simplification(s: Step):
    """Install fast-simplification for mesh decimation."""
    if _can_import("fast_simplification"):
        s.skip("fast-simplification already installed")
        return
    s.info("Installing fast-simplification...")
    _pip_quiet("fast-simplification")
    s.done()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="AIScene2Godot environment setup")
    ap.add_argument("--full", action="store_true",
                    help="install everything including MIDI (default: core only)")
    ap.add_argument("--midi-only", action="store_true",
                    help="install only the MIDI stack (assumes core is done)")
    args = ap.parse_args()

    print("=" * 60)
    print("AIScene2Godot — Environment Setup")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python:   {sys.version.split()[0]}  ({sys.executable})")
    print("=" * 60)

    if args.midi_only:
        steps = [
            ("Install/verify PyTorch + CUDA", setup_torch),
            ("Install torch-cluster", setup_torch_cluster),
            ("Clone MIDI-3D", setup_midi_clone),
            ("Install MV-Adapter", setup_mv_adapter),
            ("Install triton (Windows)", setup_triton),
            ("Build nvdiffrast", setup_nvdiffrast),
            ("Apply patches", setup_patches),
            ("Download runtime assets", setup_runtime_assets),
        ]
    elif args.full:
        steps = [
            ("Install core dependencies", setup_core),
            ("Install/verify PyTorch + CUDA", setup_torch),
            ("Install SegFormer (transformers)", setup_segformer),
            ("Install fast-simplification", setup_fast_simplification),
            ("Download DepthAnything V2 checkpoint", setup_depth_anything),
            ("Clone MIDI-3D", setup_midi_clone),
            ("Install torch-cluster", setup_torch_cluster),
            ("Install MV-Adapter", setup_mv_adapter),
            ("Install triton (Windows)", setup_triton),
            ("Build nvdiffrast", setup_nvdiffrast),
            ("Apply patches", setup_patches),
            ("Download runtime assets", setup_runtime_assets),
        ]
    else:
        steps = [
            ("Install core dependencies", setup_core),
            ("Install/verify PyTorch + CUDA", setup_torch),
            ("Install SegFormer (transformers)", setup_segformer),
            ("Install fast-simplification", setup_fast_simplification),
            ("Download DepthAnything V2 checkpoint", setup_depth_anything),
        ]

    Step.set_total(len(steps))
    t0 = time.time()

    for name, fn in steps:
        s = Step(name)
        try:
            fn(s)
        except Exception as e:
            s.fail(str(e))

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Setup complete!  ({elapsed:.0f}s)")
    if not args.full and not args.midi_only:
        print("\nThis installed the core (room shell) pipeline.")
        print("For MIDI 3D object generation, re-run with --full or --midi-only.")
    print(f"{'=' * 60}")

    print("\nNext: python build_scene.py photo.jpg")


if __name__ == "__main__":
    main()
